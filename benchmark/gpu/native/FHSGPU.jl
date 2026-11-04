"""
Matrix assembly on a KernelAbstractions backend with the generic per-state code of FermionicHilbertSpaces.

One kernel thread per column runs `FHS.chunked_operator_indices_and_amplitudes_generic!` for each term, the same loop
as on the CPU. Spaces and terms reach the device through `Adapt.adapt_structure` methods that keep only what
`basisstate`, `state_index` and `apply_local_operators` read.
"""
module FHSGPU

using Adapt: Adapt, adapt
using KernelAbstractions: KernelAbstractions as KA, @kernel, @index
using FermionicHilbertSpaces
const FHS = FermionicHilbertSpaces
using FermionicHilbertSpaces: NCMul, AbstractSubspace
using SparseArrays: SparseMatrixCSC

## Device copies of operators

Adapt.adapt_structure(to, x::NCMul) = NCMul(adapt(to, x.coeff), map(f -> adapt(to, f), Tuple(x.factors)))
Adapt.adapt_structure(to, x::FHS.ProductOperator{C}) where {C} =
    FHS.ProductOperator{C}(map(op -> adapt(to, op), x.ops), map(H -> adapt(to, H), x.spaces), Tuple(x.inds))

# A factor keeps the fields its `apply_local_operator(s)` method reads. Labels and bases only identify the mode,
# which the precomputation has already resolved.
Adapt.adapt_structure(to, f::FHS.FermionSym) = FHS.FermionSym(f.creation, nothing, nothing)
Adapt.adapt_structure(to, b::FHS.BosonSym) = FHS.BosonSym{Nothing,Nothing,Nothing}(nothing, nothing, nothing, b.exp)
Adapt.adapt_structure(to, s::FHS.SpinSym) = FHS.SpinSym{Nothing}(FHS._spin_op_char(s.op), nothing, s.exponent)

## Device copies of spaces

function Adapt.adapt_structure(to, H::FHS.FermionicSpace{F}) where {F}
    modes = Base.OneTo(FHS.nbr_of_modes(H))
    FHS.FermionicSpace{F,Int,FHS.FermionicGroup{Nothing},typeof(modes),Nothing}(modes, nothing, FHS.FermionicGroup(nothing))
end
Adapt.adapt_structure(to, H::FHS.TruncatedBosonicHilbertSpace) = FHS.TruncatedBosonicHilbertSpace(adapt(to, H.sym), H.dimension)
Adapt.adapt_structure(to, H::FHS.SpinSpace{T}) where {T} = FHS.SpinSpace{T,Nothing}(H.spin, nothing)
function Adapt.adapt_structure(to, H::FHS.ProductSpace{B}) where {B}
    factors = map(f -> adapt(to, f), H.factors)
    FHS.ProductSpace{B,typeof(factors),Nothing,typeof(H.fast_path),typeof(H.lininds),typeof(H.cartinds),Nothing}(factors, nothing, nothing, H.fast_path, H.lininds, H.cartinds)
end
# Every subspace becomes a ConstrainedSpace: the subspace interface is all that `basisstate` and `state_index` use.
Adapt.adapt_structure(to, H::AbstractSubspace) =
    FHS.ConstrainedSpace(adapt(to, parent(H)), adapt(to, FHS.subspace_states(H)), adapt(to, device_index(FHS.subspace_index(H))))

## Subspace indices

Adapt.adapt_structure(to, ix::FHS.IndexTable) = FHS.IndexTable(adapt(to, ix.positions))

"""
    SortedIndex(keys, positions, key)

Subspace index for the device: `key(state, parent)` is looked up by binary search in the sorted `keys`.
Replaces the `Dict`-based indices, which have no device version.
"""
struct SortedIndex{K,P,F}
    keys::K
    positions::P
    key::F
end
struct RootKey end
struct StateKey end
_key(::RootKey, state, parent) = FHS.state_index(state, parent)
_key(::StateKey, state, parent) = state
function SortedIndex(d::AbstractDict, key)
    ks = collect(keys(d))
    perm = sortperm(ks)
    SortedIndex(ks[perm], Int32.(collect(values(d))[perm]), key)
end
Adapt.adapt_structure(to, ix::SortedIndex) = SortedIndex(adapt(to, ix.keys), adapt(to, ix.positions), ix.key)
function FHS.state_position(ix::SortedIndex, state, parent)
    i = _sorted_position(ix.keys, _key(ix.key, state, parent))
    return iszero(i) ? 0 : Int(ix.positions[i])
end
function _sorted_position(keys, k)
    lo, hi = 1, length(keys)
    while lo <= hi
        mid = (lo + hi) >>> 1
        if isless(keys[mid], k)
            lo = mid + 1
        elseif isless(k, keys[mid])
            hi = mid - 1
        else
            return mid
        end
    end
    return 0
end

device_index(ix) = ix
device_index(ix::FHS.RootIndexDict) = SortedIndex(ix.positions, RootKey())
device_index(ix::AbstractDict) = SortedIndex(ix, StateKey())

## Device compatibility check

# Stands in for arrays when checking that everything else is isbits, and for the arrays of the group spaces a
# `ProductOperator` carries: those are used for dispatch and scalar data only, never indexed.
struct NoStorage{T,N} <: AbstractArray{T,N}
    dims::NTuple{N,Int}
end
Base.size(a::NoStorage) = a.dims
Base.getindex(::NoStorage, i::Int...) = throw(ArgumentError("The arrays of a term's group spaces are not available on the device."))
struct DropArrays end
Adapt.adapt_storage(::DropArrays, a::DenseArray) = NoStorage{eltype(a),ndims(a)}(size(a))

function check_isbits(x, what...)
    y = adapt(DropArrays(), x)
    isbits(y) && return y
    throw(ArgumentError(string(what..., " has no device version: $(_nonbits_path(y, "")) is not isbits. Its type needs an `Adapt.adapt_structure` method that keeps only isbits data and arrays.")))
end
function _nonbits_path(x, path)
    for name in fieldnames(typeof(x))
        isdefined(x, name) || continue
        field = getfield(x, name)
        isbits(field) || return _nonbits_path(field, "$path.$name")
    end
    return "`$(isempty(path) ? "the value" : path[2:end])::$(typeof(x))`"
end

## Kernels

struct DenseSink{M}
    M::M
end
struct SlotSink{R,V}
    rows::R
    vals::V
    slot::Int
end
Adapt.adapt_structure(to, s::DenseSink) = DenseSink(adapt(to, s.M))
Adapt.adapt_structure(to, s::SlotSink) = SlotSink(adapt(to, s.rows), adapt(to, s.vals), s.slot)
at_slot(s::DenseSink, slot) = s
at_slot(s::SlotSink, slot) = SlotSink(s.rows, s.vals, slot)
FHS.push_inds_amps!(s::DenseSink, outind, inind, amp) = (s.M[outind, inind] += amp; nothing)
FHS.push_inds_amps!(s::SlotSink, outind, inind, amp) = (s.rows[s.slot, inind] = outind; s.vals[s.slot, inind] = amp; nothing)

@kernel function column_kernel!(sink, space, termgroups, coeff, projection)
    j = @index(Global)
    if !iszero(coeff)
        FHS.push_inds_amps!(at_slot(sink, 1), j, j, coeff)
    end
    _apply_terms!(sink, space, termgroups, j, Int(!iszero(coeff)), projection)
end
_apply_terms!(sink, space, ::Tuple{}, j, slot, projection) = nothing
function _apply_terms!(sink, space, termgroups::Tuple, j, slot, projection)
    terms = first(termgroups)
    for t in eachindex(terms)
        op, precomp = terms[t]
        FHS.chunked_operator_indices_and_amplitudes_generic!(at_slot(sink, slot + t), op, space, precomp, j:j, 0; projection)
    end
    _apply_terms!(sink, space, Base.tail(termgroups), j, slot + length(terms), projection)
end

# Sorts the entries of column `j` by row and sums duplicates, in place at the top of the column.
@kernel function compact_kernel!(counts, rows, vals)
    j = @index(Global)
    c = 0
    for s in axes(rows, 1)
        r = rows[s, j]
        iszero(r) && continue
        v = vals[s, j]
        p = c
        while p > 0 && rows[p, j] > r
            p -= 1
        end
        if p > 0 && rows[p, j] == r
            vals[p, j] += v
        else
            for q in c:-1:(p + 1)
                rows[q + 1, j] = rows[q, j]
                vals[q + 1, j] = vals[q, j]
            end
            rows[p + 1, j] = r
            vals[p + 1, j] = v
            c += 1
        end
    end
    counts[j] = c
end

@kernel function gather_kernel!(rowval, nzval, colptr, rows, vals)
    j = @index(Global)
    start = colptr[j] - 1
    for s in 1:(colptr[j + 1] - colptr[j])
        rowval[start + s] = rows[s, j]
        nzval[start + s] = vals[s, j]
    end
end

## Formats for `representation`

"""
    DeviceDenseRepr(backend), DeviceDenseRepr{T}(backend)
    DeviceCSCRepr(backend), DeviceCSCRepr{T}(backend)

Formats for `representation` that assemble the matrix on the KernelAbstractions `backend`: a dense device matrix,
or the CSC arrays `(; colptr, rowval, nzval, m, n)` with sorted rows. `T` fixes the element type, as in `Matrix{T}`.
"""
struct DeviceDenseRepr{T,B}
    backend::B
end
struct DeviceCSCRepr{T,B}
    backend::B
end
DeviceDenseRepr{T}(backend) where {T} = DeviceDenseRepr{T,typeof(backend)}(backend)
DeviceCSCRepr{T}(backend) where {T} = DeviceCSCRepr{T,typeof(backend)}(backend)
DeviceDenseRepr(backend) = DeviceDenseRepr{Missing}(backend)
DeviceCSCRepr(backend) = DeviceCSCRepr{Missing}(backend)
const DeviceRepr{T} = Union{DeviceDenseRepr{T},DeviceCSCRepr{T}}

FHS._canonicalize_repr(r::DeviceRepr) = r
FHS._repr_eltype(default, ::DeviceRepr{T}) where {T} = T === Missing ? default : T
function FHS._terms_matrix(::Type{T}, terms, coeff, H, r::DeviceRepr, ::FHS.NoChunking; projection=false, _...) where {T}
    T <: Real && !(mapreduce(FHS.mat_eltype, promote_type, terms; init=Bool) <: Real) &&
        throw(ArgumentError("The element type $T cannot hold the matrix elements of the operator."))
    return assemble(r, prepare(terms, convert(T, coeff), H, r.backend), T; projection)
end
# FHS builds single terms and multiples of the identity without `_terms_matrix`.
FHS._term_matrix_representation(op, H, r::DeviceRepr, chunking; kwargs...) =
    FHS._terms_matrix(FHS._repr_eltype(FHS.mat_eltype(op), r), (op,), false, H, r, chunking; kwargs...)
FHS._convert_repr_eltype(M, r::DeviceRepr) = to_device(SparseMatrixCSC{FHS._repr_eltype(eltype(M), r),Int32}(M), r)
to_device(S, r::DeviceDenseRepr) = adapt(r.backend, Matrix(S))
to_device(S, r::DeviceCSCRepr) =
    (; colptr=adapt(r.backend, S.colptr), rowval=adapt(r.backend, S.rowval), nzval=adapt(r.backend, S.nzval), m=S.m, n=S.n)

## Host side

# The terms as FHS passes them, with tuple factors and their precomputation. Terms of equal type share one device
# vector.
function prepare(terms, coeff, H, backend)
    Hdev = adapt(backend, H)
    check_isbits(Hdev, "The space")
    prepared = map(collect(terms)) do term
        t = FHS.__concretize(term, FHS.TupleConcretizer())
        check_isbits((t, FHS.precomputation_before_operator_application(t, H)), "The term ", term)
    end
    bytype = Dict{DataType,Vector{Any}}()
    foreach(t -> push!(get!(() -> [], bytype, typeof(t)), t), prepared)
    groups = Tuple(adapt(backend, collect(T, ts)) for (T, ts) in bytype)
    return (; Hdev, groups, coeff, nslots=length(prepared) + !iszero(coeff), n=Int(FHS.dim(H)))
end
function run_columns!(sink, p, backend; projection)
    column_kernel!(backend)(sink, p.Hdev, p.groups, p.coeff, projection; ndrange=p.n)
    return sink
end

assemble(r::DeviceDenseRepr, p, T; projection) =
    run_columns!(DenseSink(KA.zeros(r.backend, T, p.n, p.n)), p, r.backend; projection).M
function assemble(r::DeviceCSCRepr, p, T; projection)
    backend, n = r.backend, p.n
    rows = KA.zeros(backend, Int32, p.nslots, n)
    vals = KA.zeros(backend, T, p.nslots, n)
    run_columns!(SlotSink(rows, vals, 0), p, backend; projection)
    counts = KA.zeros(backend, Int32, n)
    compact_kernel!(backend)(counts, rows, vals; ndrange=n)
    # The prefix sum moves `n` integers through the host, little next to the `nslots × n` slots.
    host_colptr = vcat(Int32(1), Int32(1) .+ cumsum(Array(counts)))
    nnz = Int(host_colptr[end]) - 1
    colptr = adapt(backend, host_colptr)
    rowval = KA.allocate(backend, Int32, nnz)
    nzval = KA.allocate(backend, T, nnz)
    gather_kernel!(backend)(rowval, nzval, colptr, rows, vals; ndrange=n)
    return (; colptr, rowval, nzval, m=n, n)
end

end
