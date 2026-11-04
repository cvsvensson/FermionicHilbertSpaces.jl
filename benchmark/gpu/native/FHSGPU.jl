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
using FermionicHilbertSpaces: NCMul, NCAdd, NCterms, AbstractSubspace

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

function check_isbits(x, what)
    y = adapt(DropArrays(), x)
    isbits(y) && return y
    throw(ArgumentError("$what has no device version: $(_nonbits_path(y, "")) is not isbits. Its type needs an `Adapt.adapt_structure` method that keeps only isbits data and arrays."))
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

## Host side

# The terms of `op` as on the CPU: products split by group on spaces with several groups, with tuple factors and
# their precomputation. Terms of equal type share one device vector.
function device_terms(op, H, backend)
    terms, coeff = op isa Number ? ((), op) : op isa NCAdd ? (collect(NCterms(op)), op.coeff) :
                   op isa NCMul ? ((op,), 0) : ((NCMul(1, [op]),), 0)
    bases = isempty(terms) ? () : FHS._checked_group_ids(op, H)
    prepared = map(terms) do term
        t = length(bases) > 1 ? FHS.partition_product(term, bases, FHS.groups(H)) : term
        t = FHS.__concretize(t, FHS.TupleConcretizer())
        check_isbits((t, FHS.precomputation_before_operator_application(t, H)), "The term $term")
    end
    bytype = Dict{DataType,Vector{Any}}()
    foreach(t -> push!(get!(() -> [], bytype, typeof(t)), t), prepared)
    groups = Tuple(adapt(backend, collect(T, ts)) for (T, ts) in bytype)
    return groups, coeff, length(prepared)
end

function _check_eltype(op, ::Type{T}) where {T}
    T <: Real && !(FHS.mat_eltype(op) <: Real) &&
        throw(ArgumentError("The element type $T cannot hold the matrix elements of the operator, of type $(FHS.mat_eltype(op))."))
end

# Everything a column kernel needs, on `backend`, and the number of slots per column.
function prepare(op, H, backend, ::Type{T}) where {T}
    _check_eltype(op, T)
    Hdev = adapt(backend, H)
    check_isbits(Hdev, "The space")
    groups, coeff, nterms = device_terms(op, H, backend)
    return (; Hdev, groups, coeff=convert(T, coeff), nslots=nterms + !iszero(coeff), n=Int(FHS.dim(H)))
end
function run_columns!(sink, p, backend; projection)
    column_kernel!(backend)(sink, p.Hdev, p.groups, p.coeff, projection; ndrange=p.n)
    KA.synchronize(backend)
    return sink
end

"""
    device_matrix(op, H, backend, T=Float64; projection=false)

The dense matrix of `op` on `H` with element type `T`, assembled on the KernelAbstractions `backend`.
"""
function device_matrix(op, H, backend, ::Type{T}=Float64; projection=false) where {T}
    p = prepare(op, H, backend, T)
    return run_columns!(DenseSink(KA.zeros(backend, T, p.n, p.n)), p, backend; projection).M
end

"""
    device_csc(op, H, backend, T=Float64; projection=false) -> (; colptr, rowval, nzval, m, n)

The CSC arrays of the sparse matrix of `op` on `H` with element type `T`, assembled on `backend`, with sorted rows
and summed duplicates.
"""
function device_csc(op, H, backend, ::Type{T}=Float64; projection=false) where {T}
    p = prepare(op, H, backend, T)
    n = p.n
    rows = KA.zeros(backend, Int32, p.nslots, n)
    vals = KA.zeros(backend, T, p.nslots, n)
    run_columns!(SlotSink(rows, vals, 0), p, backend; projection)
    counts = KA.zeros(backend, Int32, n)
    compact_kernel!(backend)(counts, rows, vals; ndrange=n)
    KA.synchronize(backend)
    # The prefix sum moves `n` integers through the host, little next to the `nslots × n` slots.
    host_colptr = vcat(Int32(1), Int32(1) .+ cumsum(Array(counts)))
    nnz = Int(host_colptr[end]) - 1
    colptr = adapt(backend, host_colptr)
    rowval = KA.allocate(backend, Int32, nnz)
    nzval = KA.allocate(backend, T, nnz)
    gather_kernel!(backend)(rowval, nzval, colptr, rows, vals; ndrange=n)
    KA.synchronize(backend)
    return (; colptr, rowval, nzval, m=n, n)
end

end
