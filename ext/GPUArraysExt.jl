module GPUArraysExt

using Adapt: Adapt, adapt
using GPUArrays: GPUArrays, @allowscalar, AbstractGPUArray, AbstractGPUSparseMatrixCSC, AbstractGPUSparseMatrixCSR
using KernelAbstractions: @index, @kernel, allocate, get_backend, synchronize
using NonCommutativeProducts: NCMul
using SparseArrays: SparseArrays, SparseMatrixCSC
import FermionicHilbertSpaces: _canonicalize_repr, _convert_repr_eltype, _repr_eltype, _skip_outside, _terms_matrix,
    push_inds_amps!, state_position
using FermionicHilbertSpaces: AbstractHilbertSpace, AbstractSubspace, BosonSym, ConstrainedSpace, DeviceFermionicSpace,
    DeviceProductSpace, DeviceSpinSym, FermionSym, FermionicSpace, IndexTable, ProductOperator, ProductSpace, RootIndexDict,
    SpinSpace, SpinSym, TruncatedBosonicHilbertSpace, TupleConcretizer, __concretize, _spin_shift,
    chunked_operator_indices_and_amplitudes_generic!, dim, mat_eltype, nbr_of_modes, precomputation_before_operator_application,
    state_index, subspace_index, subspace_states

const GPUSparseMatrix = Union{AbstractGPUSparseMatrixCSC, AbstractGPUSparseMatrixCSR}

# The format selected by the GPU array type `F`, with element type `T` (`Missing` to take it from the operator),
# assembled on the KernelAbstractions `backend`.
struct GPURepr{T, F, B}
    backend::B
end
GPURepr{T, F}(backend) where {T, F} = GPURepr{T, F, typeof(backend)}(backend)

_format_eltype(::Type{<:AbstractArray{T}}) where {T} = T
_format_eltype(::Type) = Missing
_index_type(::Type{<:SparseArrays.AbstractSparseArray{<:Any, Ti}}) where {Ti} = Ti
_index_type(::Type) = Int32
_backend(A::Type{<:AbstractGPUArray}) = get_backend(adapt(A, Matrix{Bool}(undef, 0, 0)))

_canonicalize_repr(F::Type{<:AbstractGPUArray}) = GPURepr{_format_eltype(F), F}(_backend(F))
_canonicalize_repr(F::Type{<:GPUSparseMatrix}) = GPURepr{_format_eltype(F), F}(_backend(GPUArrays.dense_array_type(F)))
_repr_eltype(default, ::GPURepr{T}) where {T} = T === Missing ? default : T

_csc_type(F::Type{<:AbstractGPUSparseMatrixCSC}) = GPUArrays.sparse_array_type(F)
_csc_type(F::Type{<:AbstractGPUSparseMatrixCSR}) = GPUArrays.csc_type(F)
_from_csc(::Type{<:AbstractGPUSparseMatrixCSC}, A) = A
_from_csc(::Type{<:AbstractGPUSparseMatrixCSR}, A) = GPUArrays.csr_type(A)(A)
_sparse_matrix(F, colptr, rowval, nzval, n) = _from_csc(F, _csc_type(F)(colptr, rowval, nzval, (n, n)))

# Multiples of the identity, which FHS builds without `_terms_matrix`
function _convert_repr_eltype(M, r::GPURepr{<:Any, F}) where {F}
    S = SparseMatrixCSC{_repr_eltype(eltype(M), r), _index_type(F)}(M)
    F <: AbstractGPUArray && return adapt(r.backend, Matrix(S))
    return _sparse_matrix(F, adapt(r.backend, S.colptr), adapt(r.backend, S.rowval), adapt(r.backend, S.nzval), size(S, 1))
end
# Takes precedence over the method for `AbstractSparseArray` types, which converts the result with `sparse`
_convert_repr_eltype(M, F::Type{<:GPUSparseMatrix}) = _convert_repr_eltype(M, _canonicalize_repr(F))

## Device copies of spaces and operators

# The rules that map FHS types to their device copies apply only under this adaptor, so that a user's `adapt` of an
# object holding FHS spaces or operators is unaffected. It adapts the arrays inside to `arrays`, a backend.
struct DeviceAdaptor{A}
    arrays::A
end
Adapt.adapt_storage(to::DeviceAdaptor, a::AbstractArray) = adapt(to.arrays, a)

# Each keeps the fields that `basisstate`, `state_index` and the operator application read. Labels, bases, symbols and
# mode lists only identify modes, which the precomputation has already resolved.
Adapt.adapt_structure(::DeviceAdaptor, f::FermionSym) = FermionSym(f.creation, nothing, nothing)
Adapt.adapt_structure(::DeviceAdaptor, b::BosonSym) = BosonSym{Nothing, Nothing, Nothing}(nothing, nothing, nothing, b.exp)
Adapt.adapt_structure(::DeviceAdaptor, s::SpinSym) = DeviceSpinSym(_spin_shift(s), s.exponent)

Adapt.adapt_structure(::DeviceAdaptor, H::FermionicSpace{F}) where {F} = DeviceFermionicSpace{F}(nbr_of_modes(H))
Adapt.adapt_structure(to::DeviceAdaptor, H::TruncatedBosonicHilbertSpace) = TruncatedBosonicHilbertSpace(adapt(to, H.sym), H.dimension)
Adapt.adapt_structure(::DeviceAdaptor, H::SpinSpace{T}) where {T} = SpinSpace{T, Nothing}(H.spin, nothing)
function Adapt.adapt_structure(to::DeviceAdaptor, H::ProductSpace{B}) where {B}
    ismissing(H.fast_path) && throw(ArgumentError("Product spaces whose groups have no internal representation as integers are not supported on GPUs."))
    factors = adapt(to, H.factors)
    return DeviceProductSpace{B, typeof(factors), typeof(H.lininds), typeof(H.cartinds)}(factors, H.lininds, H.cartinds)
end
Adapt.adapt_structure(to::DeviceAdaptor, H::AbstractSubspace) =
    ConstrainedSpace(adapt(to, parent(H)), adapt(to, subspace_states(H)), adapt(to, _sorted_index(subspace_index(H))))
# Limited to device copies, whose arrays the backend's own adaptor must reach
Adapt.adapt_structure(to, H::ConstrainedSpace{<:Any, <:Any, <:AbstractGPUArray}) =
    ConstrainedSpace(H.parent, adapt(to, H.states), adapt(to, H.index))
Adapt.@adapt_structure IndexTable

# A subspace index that finds `key(state, parent)` by binary search in the sorted `keys`, for the `Dict` indices,
# which have no device version
struct SortedIndex{K, P, F}
    keys::K
    positions::P
    key::F
end
Adapt.@adapt_structure SortedIndex
function SortedIndex(d::AbstractDict, key)
    ks = collect(keys(d))
    perm = sortperm(ks)
    return SortedIndex(ks[perm], collect(values(d))[perm], key)
end
_root_key(state, parent) = state_index(state, parent)
_state_key(state, parent) = state
_sorted_index(index) = index
_sorted_index(index::RootIndexDict) = SortedIndex(index.positions, _root_key)
_sorted_index(index::AbstractDict) = SortedIndex(index, _state_key)
function state_position(index::SortedIndex, state, parent)
    k = index.key(state, parent)
    i = searchsortedfirst(index.keys, k)
    return i <= length(index.keys) && isequal(index.keys[i], k) ? index.positions[i] : 0
end

## Terms

# Terms reach the kernels as elements of a device array, so a term holding an array fails the isbits check.
Adapt.adapt_structure(to::DeviceAdaptor, x::NCMul) = NCMul(x.coeff, adapt(to, x.factors))
Adapt.adapt_structure(to::DeviceAdaptor, x::ProductOperator{C}) where {C} =
    ProductOperator{C}(adapt(to, x.ops), adapt(to, x.spaces), Tuple(x.inds))

# The terms with tuple factors and their precomputation. Terms of equal type share one device vector.
function _device_terms(terms, H, backend)
    to = DeviceAdaptor(backend)
    prepared = map(collect(terms)) do term
        t = __concretize(term, TupleConcretizer())
        _check_isbits(adapt(to, (t, precomputation_before_operator_application(t, H))), term)
    end
    return Tuple(adapt(backend, collect(T, filter(x -> x isa T, prepared))) for T in unique(map(typeof, prepared)))
end

function _check_isbits(x, original)
    isbits(x) || throw(ArgumentError("`$original` is not supported on GPUs: its device copy of type `$(typeof(x))` is not isbits. Its type needs an `Adapt.adapt_structure` method that keeps only isbits data and arrays."))
    return x
end
# The states and index of a subspace are device arrays, which reach the kernels through the backend's own adaptor
_bits_part(space) = space
_bits_part(space::AbstractSubspace) = parent(space)

## Kernels

struct SlotSink{R, V}
    rows::R
    vals::V
    slot::Int
end
Adapt.@adapt_structure SlotSink
_at_slot(M::AbstractMatrix, slot) = M
_at_slot(s::SlotSink, slot) = SlotSink(s.rows, s.vals, slot)
# Slot `s` of column `j` is at `[j, s]`, so that neighbouring threads write neighbouring addresses
push_inds_amps!(s::SlotSink, outind, inind, amp) = (s.rows[inind, s.slot] = outind; s.vals[inind, s.slot] = amp; nothing)

# Passed as `projection`: the kernels drop the entries outside the space and set `flag[1]`. With `projection = false`,
# the host then throws the error of the CPU path.
struct OutsideFlag{F}
    flag::F
end
Adapt.@adapt_structure OutsideFlag
function _skip_outside(outind, p::OutsideFlag, newstate, state, space)
    iszero(outind) || return false
    p.flag[1] = true
    return true
end

@kernel function _column_kernel!(sink, space, termgroups, coeff, projection)
    j = @index(Global)
    iszero(coeff) || push_inds_amps!(_at_slot(sink, 1), j, j, coeff)
    _apply_terms!(sink, space, termgroups, j, Int(!iszero(coeff)), projection)
end
# Inlined like the loop it calls: a call that is not inlined passes the terms through slow local memory
@inline _apply_terms!(sink, space, ::Tuple{}, j, slot, projection) = nothing
@inline function _apply_terms!(sink, space, termgroups::Tuple, j, slot, projection)
    terms = first(termgroups)
    for t in eachindex(terms)
        op, precomp = terms[t]
        chunked_operator_indices_and_amplitudes_generic!(_at_slot(sink, slot + t), op, space, precomp, j:j, 0; projection)
    end
    return _apply_terms!(sink, space, Base.tail(termgroups), j, slot + length(terms), projection)
end

# Sorts the entries of column `j` by row and sums duplicates, in place at the start of the column. A zero row marks
# an empty slot. The count goes to `counts[j + 1]`.
@kernel function _compact_kernel!(counts, rows, vals)
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
    counts[j + 1] = c
end

# One thread per stored entry, so that the writes are coalesced
@kernel function _gather_kernel!(rowval, nzval, colptr, rows, vals)
    k = @index(Global)
    j = searchsortedlast(colptr, k)
    rowval[k] = rows[j, k - colptr[j] + 1]
    nzval[k] = vals[j, k - colptr[j] + 1]
end

## Assembly

# JLArrays runs kernels on launch and defines no `synchronize` for its backend
_synchronize(backend) = applicable(synchronize, backend) ? synchronize(backend) : nothing

# Measured slower on an NVIDIA T1000 than slots, compaction and gather: sorting while inserting, compaction by repeated
# selection of the smallest row, COO followed by cuSPARSE `sparse`, one launch per term group, and a prefix sum on the host.
function _terms_matrix(::Type{T}, terms, coeff, H::AbstractHilbertSpace, r::GPURepr, chunking; projection = false, _...) where {T}
    T <: Real && !(mapreduce(mat_eltype, promote_type, terms; init = Bool) <: Real) &&
        throw(ArgumentError("The element type $T cannot hold the complex matrix elements of the operator."))
    backend = r.backend
    space = adapt(DeviceAdaptor(backend), H)
    _check_isbits(_bits_part(space), H)
    termgroups = _device_terms(terms, H, backend)
    flag = fill!(allocate(backend, Bool, 1), false)
    run! = sink -> _column_kernel!(backend)(sink, space, termgroups, convert(T, coeff), OutsideFlag(flag); ndrange = dim(H))
    M = _assemble(r, T, run!, dim(H), length(terms) + !iszero(coeff))
    !projection && @allowscalar(flag[1]) && _skip_outside(0, false)
    return M
end

function _assemble(r::GPURepr{<:Any, <:AbstractGPUArray}, ::Type{T}, run!, n, nslots) where {T}
    M = fill!(allocate(r.backend, T, n, n), zero(T))
    run!(M)
    _synchronize(r.backend)
    return M
end
function _assemble(r::GPURepr{<:Any, F}, ::Type{T}, run!, n, nslots) where {T, F}
    backend, Ti = r.backend, _index_type(F)
    max(nslots, 1) * n < typemax(Ti) ||
        throw(ArgumentError("The $n × $n matrix may need up to $(nslots * n) stored entries, more than its index type $Ti can count. Use a sparse format with `Int64` indices."))
    rows = fill!(allocate(backend, Ti, n, nslots), zero(Ti))
    vals = allocate(backend, T, n, nslots)
    run!(SlotSink(rows, vals, 0))
    _synchronize(backend)
    counts = fill!(allocate(backend, Ti, n + 1), one(Ti))
    _compact_kernel!(backend)(counts, rows, vals; ndrange = n)
    _synchronize(backend)
    colptr = accumulate(+, counts)
    nnz = Int(@allowscalar colptr[end]) - 1
    rowval = allocate(backend, Ti, nnz)
    nzval = allocate(backend, T, nnz)
    nnz > 0 && _gather_kernel!(backend)(rowval, nzval, colptr, rows, vals; ndrange = nnz)
    _synchronize(backend)
    return _sparse_matrix(F, colptr, rowval, nzval, n)
end

end
