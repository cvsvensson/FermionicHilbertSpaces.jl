"""
Matrix assembly on a KernelAbstractions backend with the generic per-state code of FermionicHilbertSpaces.

One kernel thread per column runs `chunked_operator_indices_and_amplitudes_generic!` for each term, the same loop as on
the CPU. Spaces and terms reach the device through `Adapt.adapt_structure` methods that keep only what `basisstate`,
`state_index` and `apply_local_operators` read, so a new space or algebra needs an adapt rule and no device copy of its
algebra.

Each term gives at most one entry per column, so term `k` writes slot `k` of column `j` and threads never share memory.
A second kernel sorts each column and sums duplicates in place, and a third copies the entries into CSC arrays.

Measured slower on an NVIDIA T1000: sorting while inserting, compaction by repeated selection of the smallest row,
COO followed by cuSPARSE `sparse`, one launch per term group, and a prefix sum on the host.
"""
module KernelAbstractionsExt

using AcceleratedKernels: AcceleratedKernels as AK
using Adapt: Adapt, adapt
using KernelAbstractions: KernelAbstractions as KA, @index, @kernel
using NonCommutativeProducts: NCMul
using SparseArrays: SparseMatrixCSC
using FermionicHilbertSpaces: AbstractSubspace, BosonSym, ConstrainedSpace, DeviceCSCRepr, DeviceDenseRepr, DeviceRepr,
    FermionSym, FermionicGroup, FermionicSpace, IndexTable, NoChunking, ProductOperator, ProductSpace, RootIndexDict,
    SpinSpace, SpinSym, TruncatedBosonicHilbertSpace, TupleConcretizer, __concretize, _device_csc_matrix, _repr_eltype,
    _spin_op_char, chunked_operator_indices_and_amplitudes_generic!, dim, mat_eltype, nbr_of_modes,
    precomputation_before_operator_application, state_index, subspace_index, subspace_states
import FermionicHilbertSpaces: _convert_repr_eltype, _term_matrix_representation, _terms_matrix, push_inds_amps!,
    state_position

## Device copies of operators

Adapt.adapt_structure(to, x::NCMul) = NCMul(adapt(to, x.coeff), map(f -> adapt(to, f), Tuple(x.factors)))
Adapt.adapt_structure(to, x::ProductOperator{C}) where {C} =
    ProductOperator{C}(map(op -> adapt(to, op), x.ops), map(H -> adapt(to, H), x.spaces), Tuple(x.inds))

# A factor keeps the fields its `apply_local_operator(s)` method reads. Labels and bases only identify the mode,
# which the precomputation has already resolved.
Adapt.adapt_structure(to, f::FermionSym) = FermionSym(f.creation, nothing, nothing)
Adapt.adapt_structure(to, b::BosonSym) = BosonSym{Nothing, Nothing, Nothing}(nothing, nothing, nothing, b.exp)
Adapt.adapt_structure(to, s::SpinSym) = SpinSym{Nothing}(_spin_op_char(s.op), nothing, s.exponent)

## Device copies of spaces

function Adapt.adapt_structure(to, H::FermionicSpace{F}) where {F}
    modes = Base.OneTo(nbr_of_modes(H))
    return FermionicSpace{F, Int, FermionicGroup{Nothing}, typeof(modes), Nothing}(modes, nothing, FermionicGroup(nothing))
end
Adapt.adapt_structure(to, H::TruncatedBosonicHilbertSpace) = TruncatedBosonicHilbertSpace(adapt(to, H.sym), H.dimension)
Adapt.adapt_structure(to, H::SpinSpace{T}) where {T} = SpinSpace{T, Nothing}(H.spin, nothing)
function Adapt.adapt_structure(to, H::ProductSpace{B}) where {B}
    factors = map(f -> adapt(to, f), H.factors)
    return ProductSpace{B, typeof(factors), Nothing, typeof(H.fast_path), typeof(H.lininds), typeof(H.cartinds), Nothing}(factors, nothing, nothing, H.fast_path, H.lininds, H.cartinds)
end
# Every subspace becomes a ConstrainedSpace: the subspace interface is all that `basisstate` and `state_index` use.
Adapt.adapt_structure(to, H::AbstractSubspace) =
    ConstrainedSpace(adapt(to, parent(H)), adapt(to, subspace_states(H)), adapt(to, device_index(subspace_index(H))))

## Subspace indices

Adapt.adapt_structure(to, ix::IndexTable) = IndexTable(adapt(to, ix.positions))

"""
    SortedIndex(keys, positions, key)

Subspace index for the device: `key(state, parent)` is looked up by binary search in the sorted `keys`.
Replaces the `Dict`-based indices, which have no device version.
"""
struct SortedIndex{K, P, F}
    keys::K
    positions::P
    key::F
end
struct RootKey end
struct StateKey end
_key(::RootKey, state, parent) = state_index(state, parent)
_key(::StateKey, state, parent) = state
function SortedIndex(d::AbstractDict, key)
    ks = collect(keys(d))
    perm = sortperm(ks)
    return SortedIndex(ks[perm], Int32.(collect(values(d))[perm]), key)
end
Adapt.adapt_structure(to, ix::SortedIndex) = SortedIndex(adapt(to, ix.keys), adapt(to, ix.positions), ix.key)
function state_position(ix::SortedIndex, state, parent)
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
device_index(ix::RootIndexDict) = SortedIndex(ix.positions, RootKey())
device_index(ix::AbstractDict) = SortedIndex(ix, StateKey())

## Device compatibility check

# Stands in for arrays when checking that everything else is isbits, and for the arrays of the group spaces a
# `ProductOperator` carries: those are used for dispatch and scalar data only, never indexed.
struct NoStorage{T, N} <: AbstractArray{T, N}
    dims::NTuple{N, Int}
end
Base.size(a::NoStorage) = a.dims
Base.getindex(::NoStorage, i::Int...) = throw(ArgumentError("The arrays of a term's group spaces are not available on the device."))
struct DropArrays end
Adapt.adapt_storage(::DropArrays, a::DenseArray) = NoStorage{eltype(a), ndims(a)}(size(a))

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
struct SlotSink{R, V}
    rows::R
    vals::V
    slot::Int
end
Adapt.adapt_structure(to, s::DenseSink) = DenseSink(adapt(to, s.M))
Adapt.adapt_structure(to, s::SlotSink) = SlotSink(adapt(to, s.rows), adapt(to, s.vals), s.slot)
at_slot(s::DenseSink, slot) = s
at_slot(s::SlotSink, slot) = SlotSink(s.rows, s.vals, slot)
push_inds_amps!(s::DenseSink, outind, inind, amp) = (s.M[outind, inind] += amp; nothing)
# Slot `s` of column `j` is at `[j, s]`, so neighbouring threads write neighbouring addresses.
push_inds_amps!(s::SlotSink, outind, inind, amp) = (s.rows[inind, s.slot] = outind; s.vals[inind, s.slot] = amp; nothing)

@kernel function column_kernel!(sink, space, termgroups, coeff, projection)
    j = @index(Global)
    if !iszero(coeff)
        push_inds_amps!(at_slot(sink, 1), j, j, coeff)
    end
    _apply_terms!(sink, space, termgroups, j, Int(!iszero(coeff)), projection)
end
# Inlined, like the FHS loop it calls: a call that is not inlined passes the terms through local memory and
# needs many more registers.
@inline _apply_terms!(sink, space, ::Tuple{}, j, slot, projection) = nothing
@inline function _apply_terms!(sink, space, termgroups::Tuple, j, slot, projection)
    terms = first(termgroups)
    for t in eachindex(terms)
        op, precomp = terms[t]
        chunked_operator_indices_and_amplitudes_generic!(at_slot(sink, slot + t), op, space, precomp, j:j, 0; projection)
    end
    return _apply_terms!(sink, space, Base.tail(termgroups), j, slot + length(terms), projection)
end

# Sorts the entries of column `j` by row and sums duplicates, in place at the start of the column. The count goes
# to `colptr[j + 1]`.
@kernel function compact_kernel!(colptr, rows, vals)
    j = @index(Global)
    c = 0
    for s in axes(rows, 2)
        r = rows[j, s]
        iszero(r) && continue
        v = vals[j, s]
        p = c
        while p > 0 && rows[j, p] > r
            p -= 1
        end
        if p > 0 && rows[j, p] == r
            vals[j, p] += v
        else
            for q in c:-1:(p + 1)
                rows[j, q + 1] = rows[j, q]
                vals[j, q + 1] = vals[j, q]
            end
            rows[j, p + 1] = r
            vals[j, p + 1] = v
            c += 1
        end
    end
    colptr[j + 1] = c
end

# One thread per stored entry, so that the writes are coalesced.
@kernel function gather_kernel!(rowval, nzval, colptr, rows, vals)
    k = @index(Global)
    lo, hi = 1, length(colptr) - 1
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        if colptr[mid] <= k
            lo = mid
        else
            hi = mid - 1
        end
    end
    rowval[k] = rows[lo, k - colptr[lo] + 1]
    nzval[k] = vals[lo, k - colptr[lo] + 1]
end

## Formats for `representation`

function _terms_matrix(::Type{T}, terms, coeff, H, r::DeviceRepr, ::NoChunking; projection = false, _...) where {T}
    T <: Real && !(mapreduce(mat_eltype, promote_type, terms; init = Bool) <: Real) &&
        throw(ArgumentError("The element type $T cannot hold the matrix elements of the operator."))
    return assemble(r, prepare(terms, convert(T, coeff), H, r.backend), T; projection)
end
# FHS builds single terms and multiples of the identity without `_terms_matrix`.
_term_matrix_representation(op, H, r::DeviceRepr, chunking; kwargs...) =
    _terms_matrix(_repr_eltype(mat_eltype(op), r), (op,), false, H, r, chunking; kwargs...)
_convert_repr_eltype(M, r::DeviceRepr) = to_device(SparseMatrixCSC{_repr_eltype(eltype(M), r), Int32}(M), r)
to_device(S, r::DeviceDenseRepr) = adapt(r.backend, Matrix(S))
to_device(S, r::DeviceCSCRepr) =
    _device_csc_matrix(r.backend, adapt(r.backend, S.colptr), adapt(r.backend, S.rowval), adapt(r.backend, S.nzval), size(S))

## Host side

device_zeros(backend, T, dims...) = fill!(KA.allocate(backend, T, dims...), zero(T))

# The terms as FHS passes them, with tuple factors and their precomputation. Terms of equal type share one device
# vector.
function prepare(terms, coeff, H, backend)
    Hdev = adapt(backend, H)
    check_isbits(Hdev, "The space")
    prepared = map(collect(terms)) do term
        t = __concretize(term, TupleConcretizer())
        check_isbits((t, precomputation_before_operator_application(t, H)), "The term ", term)
    end
    bytype = Dict{DataType, Vector{Any}}()
    foreach(t -> push!(get!(() -> [], bytype, typeof(t)), t), prepared)
    groups = Tuple(adapt(backend, collect(T, ts)) for (T, ts) in bytype)
    return (; Hdev, groups, coeff, nslots = length(prepared) + !iszero(coeff), n = Int(dim(H)))
end
function run_columns!(sink, p, backend; projection)
    column_kernel!(backend)(sink, p.Hdev, p.groups, p.coeff, projection; ndrange = p.n)
    return sink
end

assemble(r::DeviceDenseRepr, p, T; projection) =
    run_columns!(DenseSink(device_zeros(r.backend, T, p.n, p.n)), p, r.backend; projection).M
function assemble(r::DeviceCSCRepr, p, T; projection)
    backend, n = r.backend, p.n
    # A zero row marks an empty slot; values are read only where the row is set.
    rows = device_zeros(backend, Int32, n, p.nslots)
    vals = KA.allocate(backend, T, n, p.nslots)
    run_columns!(SlotSink(rows, vals, 0), p, backend; projection)
    colptr = device_zeros(backend, Int32, n + 1)
    compact_kernel!(backend)(colptr, rows, vals; ndrange = n)
    fill!(view(colptr, 1:1), 1)
    AK.accumulate!(+, colptr; init = zero(Int32))
    nnz = Int(only(Array(view(colptr, (n + 1):(n + 1))))) - 1
    rowval = KA.allocate(backend, Int32, nnz)
    nzval = KA.allocate(backend, T, nnz)
    nnz > 0 && gather_kernel!(backend)(rowval, nzval, colptr, rows, vals; ndrange = nnz)
    return _device_csc_matrix(backend, colptr, rowval, nzval, (n, n))
end

end
