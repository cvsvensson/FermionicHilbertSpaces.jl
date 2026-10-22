

# Combines `spaces` into one space without applying any new constraint.
# Constraints already present on the input spaces are carried over.
function tensor_product_no_constraints(spaces)
    atoms = Iterators.flatten(Iterators.map(atomic_factors, spaces))
    groups = groupby(group_id, atoms; sortkeys=false, sortvals=false)

    factors = map((id, atoms) -> combine_into_group(id, atoms), keys(groups), values(groups))
    space = if length(factors) == 1
        only(factors)
    else
        ProductSpace(Tuple(factors))
    end
    if any(isconstrained, spaces)
        mapper = state_mapper(space, spaces)
        states = _find_combined_states(space, spaces, mapper)
        return ConstrainedSpace(space, states)
    end
    return space
end
"""
    tensor_product(spaces...; constraint=NoSymmetry())
    tensor_product(spaces; constraint=NoSymmetry())

Construct the composite Hilbert space from the spaces in `spaces`, with optional keyword argument `constraint`.
"""
tensor_product(spaces...; kwargs...) = tensor_product(spaces; kwargs...)
function tensor_product(spaces; constraint=NoSymmetry())
    if length(spaces) == 1
        return constrain_space(only(spaces), constraint)
    end
    full_space = tensor_product_no_constraints(spaces)
    if constraint == NoSymmetry()
        combined = _combined_sector_constraint(spaces)
        !ismissing(combined) && return sector_space(full_space, basisstates(full_space), sector_function(combined, full_space), combined)
        return full_space
    end

    constraint = supply_missing_constraint_info(constraint, full_space, spaces)

    states = supports_branch_pruning(constraint) ? generate_states(spaces, constraint, full_space) : basisstates(full_space)
    if supports_sector_grouping(constraint) && supports_filtering(constraint)
        return sector_space(full_space, states, sector_function(constraint, full_space), constraint)
    elseif supports_filtering(constraint)
        filtered_states = collect(Iterators.filter(filter_function(constraint, full_space), states))
        return ConstrainedSpace(full_space, filtered_states)
    else
        return ConstrainedSpace(full_space, states)
    end
end

size_compatible(m, H) = size(m) == ntuple(_ -> dim(H), ndims(m))
size_compatible(m::UniformScaling, H) = true
kron_sizes_compatible(ms, Hs) = all(size_compatible(m, H) for (m, H) in zip(ms, Hs))

@testitem "Kron size compatibility and error handling" begin
    using LinearAlgebra, Random
    import FermionicHilbertSpaces: kron_sizes_compatible
    Random.seed!(1234)
    @fermions a
    H1 = hilbert_space(a, 1:2)
    H2 = hilbert_space(a, 3:3)
    H3 = hilbert_space(a, 4:6)
    Hs = [H1, H2, H3]
    ms = m1, m2, m3 = [rand(dim(H), dim(H)) for H in Hs]
    vs = v1, v2, v3 = [rand(dim(H)) for H in Hs]
    @test kron_sizes_compatible(ms, Hs)
    @test kron_sizes_compatible([I, I, m3], Hs)
    @test !kron_sizes_compatible(ms, reverse(Hs))
    @test !kron_sizes_compatible([I, I, m2], Hs)
    @test kron_sizes_compatible(vs, Hs)
    @test !kron_sizes_compatible(vs, reverse(Hs))
    @test_throws ArgumentError generalized_kron(ms, reverse(Hs))
    H12 = tensor_product(H1, H2)
    m_too_large = rand(dim(H12) + 1, dim(H12) + 1)
    @test_throws ArgumentError partial_trace(m_too_large, H12 => H1)
end

"""
    generalized_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs))

Compute the tensor product of matrices or vectors in `ms` with respect to the spaces `Hs`, respectively. Return a matrix in the space `H`, which defaults to the `tensor_product` of `Hs`.

The relative order of fermionic embeddings, and so the sign of the result, follows the fermion mode ordering of `H`,
not the order of `ms`. For the ordered product in which the order of `ms` and `Hs` given by the caller
decides the signs, use `tensor_product(ms, Hs => H)`, which is the usual entry point.
`generalized_kron(ms, Hs => H)` is also accepted, and `generalized_kron(Hs => H)` returns a function of `ms`.
`Hs => H` reads input => output; see [Pair arguments](@ref pair_convention).
"""
function generalized_kron(ms, Hs, H::AbstractHilbertSpace=tensor_product(Hs); kwargs...)
    kron_sizes_compatible(ms, Hs) || throw(ArgumentError("The sizes of `ms` must match the dimensions of `Hs`, got sizes $(map(size, ms)) and dimensions $(map(dim, Hs))."))
    N = ndims(first(ms))
    mout = allocate_tensor_product_result(ms, Hs, H)
    extend_state = state_mapper(H, Hs)
    if N == 1
        return generalized_kron_vec!(mout, Tuple(ms), Tuple(Hs), H, extend_state; kwargs...)
    elseif N == 2
        return generalized_kron_mat!(mout, Tuple(ms), Tuple(Hs), H, extend_state; kwargs...)
    end
    throw(ArgumentError("Only 1D or 2D arrays are supported, got $(N)D."))
end

generalized_kron(Hs::Pair; kwargs...) = (ms...) -> generalized_kron(ms, Hs; kwargs...)
generalized_kron(ms, Hs::Pair; kwargs...) = generalized_kron(ms, first(Hs), last(Hs); kwargs...)

uniform_to_sparse_type(::Type{UniformScaling{T}}) where {T} = SparseMatrixCSC{T,Int}
uniform_to_sparse_type(::Type{T}) where {T} = T
function allocate_tensor_product_result(ms, Hs, H)
    T = Base.promote_eltype(ms...)
    N = ndims(first(ms))
    types = map(uniform_to_sparse_type ∘ typeof, ms)
    MT = Base.promote_op(kron, types...)
    Nout = dim(H)
    _mout = Zeros(T, ntuple(j -> Nout, N))
    try
        convert(MT, _mout)
    catch
        Array(_mout)
    end
end

tensor_product_iterator(m::SparseArrays.AbstractSparseVecOrMat, ::AbstractHilbertSpace) = zip(findnz(m)[1:2]...)
tensor_product_iterator(m::AbstractArray, ::AbstractHilbertSpace) = CartesianIndices(m)
tensor_product_iterator(::UniformScaling, H::AbstractHilbertSpace) = diagind(I(length(basisstates(H))), IndexCartesian())

"""
    _foreach_kron_term(op, inds, ms, Hs, H, extend_state; phase_factors, skipmissing)

Kernel shared by the dense and sparse generalized matrix kron. Calls `op(K1, K2, v)` for each
output matrix element in `H`, where `v` is the product of the selected input elements, the
recombination weights and the phase factor. `inds` supplies one row/column index pair per
local matrix. When `skipmissing=false`, a missing state throws `ArgumentError`.
"""
function _foreach_kron_term(op, inds, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state;
    phase_factors=true, skipmissing=false)
    pfh = phase_factors ? kron_phase_factor(extend_state) : (f1, f2) -> 1
    for I in inds
        I1 = map(i -> i[1], I)
        I2 = map(i -> i[2], I)
        state1 = map(basisstate, I1, Hs)
        state2 = map(basisstate, I2, Hs)
        fullstates1, amps1 = combine_states(state1, extend_state)
        fullstates2, amps2 = combine_states(state2, extend_state)
        v = prod(ntuple(i -> ms[i][I1[i], I2[i]], length(ms)))
        for (fullstate1, w1) in zip(fullstates1, amps1)
            K1 = state_index(fullstate1, H)
            if iszero(K1)
                skipmissing && continue
                throw(ArgumentError("The state `$fullstate1` is not in the full Hilbert space."))
            end
            for (fullstate2, w2) in zip(fullstates2, amps2)
                K2 = state_index(fullstate2, H)
                if iszero(K2)
                    skipmissing && continue
                    throw(ArgumentError("The state `$fullstate2` is not in the full Hilbert space."))
                end
                op(K1, K2, w1 * w2 * v * pfh(fullstate1, fullstate2))
            end
        end
    end
    return nothing
end

function generalized_kron_mat!(mout::AbstractMatrix{T}, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors::Bool=true, skipmissing=false) where T
    fill!(mout, zero(T))
    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)
    _foreach_kron_term(inds, ms, Hs, H, extend_state; phase_factors, skipmissing) do K1, K2, v
        mout[K1, K2] += v
    end
    return mout
end

function generalized_kron_mat!(mout::SparseMatrixCSC{T}, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors::Bool=true, skipmissing=false) where T
    inds = Base.product(map(tensor_product_iterator, ms, Hs)...)
    Is, Js, Vs = Int[], Int[], T[]
    sizehint!(Is, length(inds))
    sizehint!(Js, length(inds))
    sizehint!(Vs, length(inds))
    _foreach_kron_term(inds, ms, Hs, H, extend_state; phase_factors, skipmissing) do K1, K2, v
        push!(Is, K1)
        push!(Js, K2)
        push!(Vs, v)
    end
    return mout .= SparseArrays.sparse!(Is, Js, Vs, size(mout, 1), size(mout, 2))
end

"""
    _foreach_kron_vec_term(op, inds, ms, Hs, H, extend_state; phase_factors, skipmissing)

Kernel for the generalized vector kron. Calls `op(K, v)` for each output vector entry in `H`,
where `v` is the product of the selected input entries, the recombination weight and the
phase factor. When `skipmissing=false`, a missing state throws `ArgumentError`.
"""
function _foreach_kron_vec_term(op, inds, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state;
    phase_factors=true, skipmissing=false)
    mapper = state_mapper(H, Hs)
    pfu = phase_factors ? phase_factor_u(mapper) : state -> 1
    for I in inds
        TI = Tuple(I)
        fock = map(basisstate, TI, Hs)
        states, amps = combine_states(fock, extend_state)
        v = mapreduce((i1, m) -> m[i1], *, TI, ms)
        for (fullfock, w) in zip(states, amps)
            K = state_index(fullfock, H)
            if iszero(K)
                skipmissing && continue
                throw(ArgumentError("The state `$fullfock` is not in the full Hilbert space."))
            end
            op(K, w * v * pfu(fullfock))
        end
    end
    return nothing
end

function generalized_kron_vec!(mout, ms::Tuple, Hs::Tuple, H::AbstractHilbertSpace, extend_state; phase_factors=true, skipmissing=false)
    fill!(mout, zero(eltype(mout)))
    inds = CartesianIndices(Tuple(map(length ∘ basisstates, Hs)))
    _foreach_kron_vec_term(inds, ms, Hs, H, extend_state; phase_factors, skipmissing) do K, v
        mout[K] += v
    end
    return mout
end


@testitem "generalized_kron_vec! direct call" begin
    using Random
    Random.seed!(1234)
    @fermions a
    H1 = hilbert_space(a, 1:1)
    H2 = hilbert_space(a, 2:3)
    Hs = (H1, H2)
    H = tensor_product(Hs)

    v1 = rand(ComplexF64, dim(H1))
    # v1 = FermionicHilbertSpaces.project_on_parity(v1, H1, 1)
    v2 = rand(ComplexF64, dim(H2))
    v2 = FermionicHilbertSpaces.project_on_parity(v2, H2, 1)
    mout = zeros(ComplexF64, dim(H))
    mapper = FermionicHilbertSpaces.state_mapper(H, Hs)
    rho1 = v1 * v1'
    rho2 = v2 * v2'

    v12 = FermionicHilbertSpaces.generalized_kron_vec!(mout, (v1, v2), Hs, H, mapper)
    @test v12 ≈ tensor_product((v1, v2), Hs, H)
    @test v12 ≈ generalized_kron((v1, v2), Hs, H)

    rho12 = tensor_product((rho1, rho2), Hs, H)
    rho_v12 = v12 * v12'
    @test rho12 ≈ rho_v12 # only true because we projected v2 to a definite parity sector

    ## test mixed spaces
    @boson b
    Hb = hilbert_space(b, 2)
    Hmix = tensor_product(H1, Hb)
    H = tensor_product(H1, H2, Hb)
    vb = rand(ComplexF64, dim(Hb))
    vmix = tensor_product((v1, vb), (H1, Hb) => Hmix)
    v12b = tensor_product((v1, v2, vb), (H1, H2, Hb) => H)
    vmix2 = tensor_product((vmix, v2), (Hmix, H2) => H)
    @test v12b ≈ vmix2

    # test error handling for missing states
    @fermions a
    H1 = hilbert_space(a, 1:1)
    H2 = hilbert_space(a, 2:2)
    Hfull = tensor_product(H1, H2)                    # |00>, |01>, |10>, |11>
    H = hilbert_space(a, 1:2, NumberConservation(1))  # only |01>, |10>
    v1 = rand(ComplexF64, dim(H1))
    v2 = rand(ComplexF64, dim(H2))
    # the combinations |00> and |11> do not exist in H
    @test_throws ArgumentError generalized_kron((v1, v2), (H1, H2) => H)
end

"""
    tensor_product(ms, Hs, H::AbstractHilbertSpace; kwargs...)
    tensor_product(ms, Hs => H; kwargs...)
    tensor_product(Hs => H; kwargs...)

Compute the ordered product of the fermionic embeddings of the matrices `ms` in the spaces `Hs` into the space `H`.
`kwargs` can be passed a bool `phase_factors`. The last form returns a function of `ms`.
`Hs => H` reads input => output; see [Pair arguments](@ref pair_convention).
"""
function tensor_product(ms::Union{<:AbstractVector,<:Tuple}, Hs, H::AbstractHilbertSpace; kwargs...)
    # See eq. 26 in J. Phys. A: Math. Theor. 54 (2021) 393001
    if all(_issquare, ms)
        return mapreduce(((m, fine_basis),) -> embed(m, fine_basis, H; kwargs...), *, zip(ms, Hs))
    elseif all(_isket, ms)
        return generalized_kron(map(vec, ms), Hs, H; kwargs...)
    elseif all(_isbra, ms)
        # All matrices are row vectors, so we can treat them as vectors and then reshape the result
        return transpose(generalized_kron(map(vec, ms), Hs, H; kwargs...))
    end
    throw(ArgumentError("The elements of `ms` must be either all square matrices, all column vectors or all row vectors, got sizes $(map(size, ms))."))
end
_issquare(m) = size(m, 1) == size(m, 2)
_issquare(m::UniformScaling) = true
_isket(m::AbstractVecOrMat) = ndims(m) == 1 || (ndims(m) == 2 && size(m, 2) == 1)
_isket(m::UniformScaling) = false
_isbra(m::AbstractVecOrMat) = ndims(m) == 2 && size(m, 1) == 1
_isbra(m::UniformScaling) = false

tensor_product(ms::Union{<:AbstractVector,<:Tuple}, HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = tensor_product(ms, first(HsH), last(HsH); kwargs...)
tensor_product(HsH::Pair{<:Any,<:AbstractHilbertSpace}; kwargs...) = (ms...) -> tensor_product(ms, first(HsH), last(HsH); kwargs...)

struct PhaseMap{F}
    phases::Matrix{Int}
    fockstates::Vector{F}
end
struct LazyPhaseMap{M,F} <: AbstractMatrix{Int}
    fockstates::Vector{F}
end
Base.size(p::LazyPhaseMap) = (length(p.fockstates), length(p.fockstates))
function Base.show(io::IO, p::LazyPhaseMap{M,F}) where {M,F}
    print(io, "LazyPhaseMap{$M,$F}(")
    show(io, p.fockstates)
    print(")")
end
Base.show(io::IO, ::MIME"text/plain", p::LazyPhaseMap) = show(io, p)
Base.getindex(p::LazyPhaseMap{M}, n1::Int, n2::Int) where {M} = phase_factor_f(p.fockstates[n1], p.fockstates[n2], M)
function phase_map(fockstates, M::Int)
    phases = zeros(Int, length(fockstates), length(fockstates))
    for (n1, f1) in enumerate(fockstates)
        for (n2, f2) in enumerate(fockstates)
            phases[n1, n2] = phase_factor_f(f1, f2, M)
        end
    end
    PhaseMap(phases, fockstates)
end
phase_map(N::Int) = phase_map(map(FockNumber, UnitRange{UInt64}(0, 2^N - 1)), N)
phase_map(H::AbstractHilbertSpace) = phase_map(collect(basisstates(H)), nbr_of_modes(H))
function LazyPhaseMap(N::Int)
    states = map(FockNumber, UnitRange{UInt64}(0, 2^N - 1))
    LazyPhaseMap{N,eltype(states)}(states)
end
SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(::LazyPhaseMap, rest...) = SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(rest...)
(p::PhaseMap)(op::AbstractMatrix) = p.phases .* op
(p::LazyPhaseMap)(op::AbstractMatrix) = p .* op
@testitem "Phase map: sign pattern structure" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: fermions
    # see App 2 in https://arxiv.org/pdf/2006.03087
    ns = 1:4
    phis = Dict(zip(ns, FermionicHilbertSpaces.phase_map.(ns)))
    lazyphis = Dict(zip(ns, FermionicHilbertSpaces.LazyPhaseMap.(ns)))
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @test all(sum(phis[n].phases .== -1) == (2^n - 2) * 2^n / 2 for n in ns)
    @fermions f
    for N in ns
        H = hilbert_space(f, 1:N)
        c = fermions(H)
        # Now let's make commuting fermions (hardcore bosons)
        q = Dict(k => embed(only(fermions(hilbert_space(f[k])))[2], hilbert_space(f[k]) => H; phase_factors=false) for k in 1:N)
        @test all(map(n -> q[n] == phis[N](c[n]), 1:N))
        c2 = map(n -> phis[N](c[n]), 1:N)
        @test phis[N](phis[N](c[1])) == c[1]
        # c is fermionic
        @test all([c[n] * c[n2] == -c[n2] * c[n] for n in 1:N, n2 in 1:N])
        @test all([c[n]' * c[n2] == -c[n2] * c[n]' + I * (n == n2) for n in 1:N, n2 in 1:N])
        # c2 is hardcore bosons
        @test all([c2[n] * c2[n2] == c2[n2] * c2[n] for n in 1:N, n2 in 1:N])
        @test all([c2[n]' * c2[n2] == (-c2[n2] * c2[n]' + I) * (n == n2) + (n !== n2) * (c2[n2] * c2[n]') for n in 1:N, n2 in 1:N])
    end

    H1 = hilbert_space(f, 1:1)
    c1 = fermions(H1)
    H2 = hilbert_space(f, 2:2)
    c2 = fermions(H2)
    H12 = hilbert_space(f, 1:2)
    c12 = fermions(H12)
    p1 = FermionicHilbertSpaces.LazyPhaseMap(1)
    p2 = FermionicHilbertSpaces.phase_map(2)
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps((c1[1], I(2)), (p1, p1), p2) == c12[1]
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps((I(2), c2[2]), (p1, p1), p2) == c12[2]

    ms = (rand(2, 2), rand(2, 2))
    @test FermionicHilbertSpaces.fermionic_tensor_product_with_kron_and_maps(ms, (p1, p1), p2) == generalized_kron(ms, (H1, H2), H12)
end

function fermionic_tensor_product_with_kron_and_maps(ops, phis, phi)
    phi(kron(reverse(map((phi, op) -> phi(op), phis, ops))...))
end

partial_trace(v::AbstractVector, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; kwargs...) = partial_trace(vec_to_density_matrix(v), H, Hsub; kwargs...)
function partial_trace(m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(m, Hsub, H, complement), kwargs...)
    size_compatible(m, H) || throw(ArgumentError("The size of `m` must match `dim(H)` = $(dim(H)), got $(size(m))."))
    isnothing(complement) && return change_basis(m, H, Hsub; kwargs...)
    mout = zeros(promote_type(eltype(m), partial_trace_phase_factor_eltype(H)), dim(Hsub), dim(Hsub))
    partial_trace!(mout, m, H, Hsub, complement, alg; kwargs...)
end

"""
    basis_transformation(Hfrom, Hto)

Return `(U, dropped)`, where `U` is the sparse `dim(Hto) × dim(Hfrom)` matrix with
`U[i, j] = 1` if basis state `j` of `Hfrom` is basis state `i` of `Hto`, and `dropped` are the
indices of the states of `Hfrom` that do not exist in `Hto`. States of `Hto` that do not exist
in `Hfrom` give zero rows. `Hfrom` and `Hto` must have the same atoms (see `match_same_atoms`),
but their groups may come in different orders.
"""
function basis_transformation(Hfrom, Hto)
    match_same_atoms(Hfrom, Hto)
    # the mapper puts each state of Hfrom in the group order of Hto
    return _basis_transformation(Hfrom, Hto, state_mapper(Hto, (Hfrom,)))
end
function _basis_transformation(Hfrom, Hto, mapper)
    Is = Int[]
    Js = Int[]
    dropped = Int[]
    _to_state(f)  = only(first(combine_states((f,), mapper)))
    for (j, f) in enumerate(basisstates(Hfrom))
        i = state_index(_to_state(f), Hto)
        if iszero(i)
            push!(dropped, j)
        else
            push!(Is, i)
            push!(Js, j)
        end
    end
    U = SparseArrays.sparse!(Is, Js, ones(Int, length(Is)), dim(Hto), dim(Hfrom))
    return U, dropped
end

"""
    change_basis(m, Hfrom => Hto; skipmissing=false)
    change_basis(m, Hfrom, Hto; skipmissing=false)

Express `m` in the basis of `Hto`, where `Hfrom` and `Hto` have the same atoms but possibly
different (or differently ordered) basis states. Unless `skipmissing=true`, throws an
`ArgumentError` if `m` has nonzero elements involving states of `Hfrom` that are not in `Hto`.
"""
change_basis(m, Hs::PairWithHilbertSpaces; kwargs...) = change_basis(m, first(Hs), last(Hs); kwargs...)
function change_basis(m, Hfrom::AbstractHilbertSpace, Hto::AbstractHilbertSpace; skipmissing=false, kwargs...)
    U, dropped = basis_transformation(Hfrom, Hto)
    if !skipmissing
        for j in dropped
            _has_support_on(m, j) && throw(ArgumentError("The state $(basisstate(j, Hfrom)) has nonzero weight but is not in the target Hilbert space. Pass `skipmissing=true` to drop it."))
        end
    end
    return _change_basis(m, U)
end
_change_basis(m::AbstractMatrix, U) = U * m * U'
_change_basis(v::AbstractVector, U) = U * v
_change_basis(m::UniformScaling, U) = m.λ * (U * U')

_has_support_on(m::AbstractMatrix, j) = !iszero(view(m, j, :)) || !iszero(view(m, :, j))
_has_support_on(v::AbstractVector, j) = !iszero(v[j])
_has_support_on(m::UniformScaling, j) = !iszero(m.λ)

function _checked_basis_transformation(Hfrom, Hto; skipmissing=false)
    U, dropped = basis_transformation(Hfrom, Hto)
    if !skipmissing && !isempty(dropped)
        throw(ArgumentError("The state $(basisstate(first(dropped), Hfrom)) is not in the target Hilbert space. Pass `skipmissing=true` to drop such states."))
    end
    return U
end
function basis_transformation_map(Hfrom, Hto; skipmissing=false, kwargs...)
    U = _checked_basis_transformation(Hfrom, Hto; skipmissing)
    kron(conj(U), U)
end
# transpose(basis_transformation_map(Hfrom, Hto)), whose column pointer runs over dim(Hto)^2 instead of dim(Hfrom)^2 columns
function transposed_basis_transformation_map(Hfrom, Hto; skipmissing=false, kwargs...)
    U = _checked_basis_transformation(Hfrom, Hto; skipmissing)
    kron(sparse(U'), sparse(transpose(U)))
end

"""
    partial_trace(m, H => Hsub; complement=complementary_subsystem(H, Hsub))
    partial_trace(H => Hsub; kwargs...)  # -> PartialTraceMap

Compute the partial trace of `m` from `H` to `Hsub`. Fermionic phase factors are included if both `H` and `Hsub` are Fermionic, unless specified otherwise in `kwargs`.
`H => Hsub` reads input => output; see [Pair arguments](@ref pair_convention).
"""
partial_trace(m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = partial_trace(m, Hs...; kwargs...)

abstract type AbstractPartialTraceAlg end
struct SubsystemPartialTraceAlg <: AbstractPartialTraceAlg end
struct FullPartialTraceAlg <: AbstractPartialTraceAlg end

_effective_dim(m::AbstractMatrix, H) = length(m)
_effective_dim(m::AbstractSparseMatrix, H) = nnz(m)
_effective_dim(::UniformScaling, H) = dim(H)

default_partial_trace_alg(m::Union{AbstractMatrix,UniformScaling}, Hsub, H, Hcomp) = dim(Hsub)^2 * dim(Hcomp) < _effective_dim(m, H) ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()
default_partial_trace_alg(m::Union{AbstractMatrix,UniformScaling}, Hsub, H, ::Nothing) = dim(Hsub)^2 < _effective_dim(m, H) ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()
default_partial_trace_alg(Hsub, H, Hcomp) = dim(Hsub)^2 * dim(Hcomp) < dim(H)^2 ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()
default_partial_trace_alg(Hsub, H, ::Nothing) = dim(Hsub)^2 < dim(H)^2 ? SubsystemPartialTraceAlg() : FullPartialTraceAlg()

partial_trace_phase_factor_eltype(::AbstractAtomicHilbertSpace) = Int

"""
    _foreach_partial_trace_term(op, inds, H, Hsub, complement, mapper, alg; phase_factors, skipmissing)

Kernel shared by the direct and sparse-map partial trace. Calls `op(J1, J2, K1, K2, v)` for
each contribution, where `(J1, J2)` indexes the input matrix in `H`, `(K1, K2)` the output
matrix in `Hsub`, and `v` excludes the input matrix element. `inds` is ignored by
`SubsystemPartialTraceAlg` (pass `nothing`); `FullPartialTraceAlg` visits the index pairs in
`inds`. When `skipmissing=false`, a missing state throws `ArgumentError`.
"""
function _foreach_partial_trace_term(op, inds, H, Hsub, complement, mapper, ::SubsystemPartialTraceAlg;
    phase_factors=true, skipmissing=true)
    # `inds` is unused: this algorithm enumerates subsystem and complement states directly.
    substates = basisstates(Hsub)
    barstates = basisstates(complement)
    for f1 in substates
        K1 = state_index(f1, Hsub)
        for f2 in substates
            K2 = state_index(f2, Hsub)
            s2 = phase_factors ? partial_trace_phase_factor(f1, f2, Hsub) : 1
            for fbar in barstates
                fullstates1, amps1 = combine_states((f1, fbar), mapper)
                fullstates2, amps2 = combine_states((f2, fbar), mapper)
                for (fullf1, w1) in zip(fullstates1, amps1)
                    J1 = state_index(fullf1, H)
                    if iszero(J1)
                        skipmissing && continue
                        throw(ArgumentError("The state `$fullf1` is not in the full Hilbert space."))
                    end
                    for (fullf2, w2) in zip(fullstates2, amps2)
                        J2 = state_index(fullf2, H)
                        if iszero(J2)
                            skipmissing && continue
                            throw(ArgumentError("The state `$fullf2` is not in the full Hilbert space."))
                        end
                        s1 = phase_factors ? partial_trace_phase_factor(fullf1, fullf2, H) : 1
                        op(J1, J2, K1, K2, w1 * w2 * s2 * conj(s1))
                    end
                end
            end
        end
    end
    return nothing
end

function _foreach_partial_trace_term(op, inds, H, Hsub, complement, mapper, ::FullPartialTraceAlg;
    phase_factors=true, skipmissing=false)
    splitsamps = map(Base.Fix2(split_state, mapper), basisstates(H))
    for I in inds
        J1, J2 = I[1], I[2]
        # `splitsamps` has one entry per basis state of `H`, so these lookups bounds-check J1 and J2.
        splits1, amps1 = splitsamps[J1]
        splits2, amps2 = splitsamps[J2]
        f1 = @inbounds basisstate(J1, H)
        f2 = @inbounds basisstate(J2, H)
        for ((f1sub, f1bar), w1) in zip(splits1, amps1)
            K1 = state_index(f1sub, Hsub)
            if iszero(K1)
                skipmissing && continue
                throw(ArgumentError("The state `$f1sub` is not in the subsystem Hilbert space."))
            end
            for ((f2sub, f2bar), w2) in zip(splits2, amps2)
                f1bar != f2bar && continue
                K2 = state_index(f2sub, Hsub)
                if iszero(K2)
                    skipmissing && continue
                    throw(ArgumentError("The state `$f2sub` is not in the subsystem Hilbert space."))
                end
                s1 = phase_factors ? partial_trace_phase_factor(f1, f2, H) : 1
                s2 = phase_factors ? partial_trace_phase_factor(f1sub, f2sub, Hsub) : 1
                op(J1, J2, K1, K2, w1 * w2 * s2 * conj(s1))
            end
        end
    end
    return nothing
end

"""
    partial_trace!(mout, m, H, Hsub; complement=complementary_subsystem(H, Hsub), alg, phase_factors=true, skipmissing)
    partial_trace!(mout, m, H => Hsub; kwargs...)

In-place partial trace of `m` (an operator on `H` or a pure state) from `H` to `Hsub`, written
to `mout`, which must have size `(dim(Hsub), dim(Hsub))`. See [`partial_trace`](@ref).

`alg` is `SubsystemPartialTraceAlg()` or `FullPartialTraceAlg()`, chosen from the dimensions
and sparsity of `m` by default. `skipmissing` defaults to `true` for `SubsystemPartialTraceAlg()`
and `false` for `FullPartialTraceAlg()`.
"""
partial_trace!(mout, v::AbstractVector, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; kwargs...) = partial_trace!(mout, vec_to_density_matrix(v), H, Hsub; kwargs...)
function partial_trace!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(m, Hsub, H, complement), kwargs...)
    size_compatible(m, H) || throw(ArgumentError("The size of `m` must match `dim(H)` = $(dim(H)), got $(size(m))."))
    size(mout) == (dim(Hsub), dim(Hsub)) || throw(DimensionMismatch("The output matrix must have size ($(dim(Hsub)), $(dim(Hsub))), got $(size(mout))"))
    if isnothing(complement)
        U = basis_transformation(Hsub, H)
        mout .= U * m * U'
        return mout
    end
    partial_trace!(mout, m, H, Hsub, complement, alg; kwargs...)
end
partial_trace!(mout, m, Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = partial_trace!(mout, m, Hs...; kwargs...)

function partial_trace!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, ::SubsystemPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=true, phase_factors=true)
    fill!(mout, zero(eltype(mout)))
    _foreach_partial_trace_term(nothing, H, Hsub, complement, mapper, SubsystemPartialTraceAlg(); skipmissing, phase_factors) do J1, J2, K1, K2, v
        mout[K1, K2] += v * m[J1, J2]
    end
    return mout
end

function partial_trace!(mout, m, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement, ::FullPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); phase_factors=true, skipmissing=false)
    fill!(mout, zero(eltype(mout)))
    _foreach_partial_trace_term(tensor_product_iterator(m, H), H, Hsub, complement, mapper, FullPartialTraceAlg(); phase_factors, skipmissing) do J1, J2, K1, K2, v
        mout[K1, K2] += v * m[J1, J2]
    end
    return mout
end

"""
    _transposed_superoperator(Is, Js, Vs, nout, nin)

Return the `nout × nin` superoperator `S` with `S[Is[p], Js[p]] = Vs[p]` (repeated entries are
summed) as `transpose(St)`, where `St` is a `SparseMatrixCSC`. For partial-trace maps `nin = dim(H)^2`
is much larger than `nout = dim(Hsub)^2`, and storing the transpose keeps the column pointer at length
`nout + 1`. Building `St` with `sparse(Js, Is, Vs, nin, nout)` would need `O(nin)` scratch space, so the
entries are sorted by output index with a counting sort instead.
"""
function _transposed_superoperator(Is::AbstractVector{<:Integer}, Js::AbstractVector{<:Integer}, Vs::AbstractVector{T}, nout::Integer, nin::Integer) where T
    N = length(Vs)
    length(Is) == length(Js) == N || throw(DimensionMismatch("Is, Js and Vs must have the same length"))
    colptr = zeros(Int, nout + 1)
    colptr[1] = 1
    for (I, J) in zip(Is, Js)
        1 <= I <= nout || throw(ArgumentError("row index $I out of range 1:$nout"))
        1 <= J <= nin || throw(ArgumentError("column index $J out of range 1:$nin"))
        colptr[I+1] += 1
    end
    cumsum!(colptr, colptr)
    next = colptr[1:nout]
    perm = Vector{Int}(undef, N)
    for p in 1:N
        perm[next[Is[p]]] = p
        next[Is[p]] += 1
    end
    # sort each column of St by row and sum repeated entries
    rows = sizehint!(Int[], N)
    vals = sizehint!(T[], N)
    newptr = similar(colptr)
    newptr[1] = 1
    for K in 1:nout
        r = colptr[K]:(colptr[K+1]-1)
        length(r) > 1 && sort!(view(perm, r); by=p -> Js[p])
        for q in r
            p = perm[q]
            if length(vals) >= newptr[K] && rows[end] == Js[p]
                vals[end] += Vs[p]
            else
                push!(rows, Js[p])
                push!(vals, Vs[p])
            end
        end
        newptr[K+1] = length(vals) + 1
    end
    return transpose(SparseMatrixCSC(nin, nout, newptr, rows, vals))
end

# The adjoint of a stored superoperator, materialized so that it is again a SparseMatrixCSC or its transpose
_adjoint_superop(S::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}) = conj(parent(S))
_adjoint_superop(S::SparseMatrixCSC) = transpose(conj(S))
_adjoint_superop(S) = S'

# w = S * vec(ψψ†) for S = transpose(St), without forming ψψ†
function _apply_superop_on_pure_state!(w::AbstractVector, S::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}, ψ::AbstractVector)
    D = length(ψ)
    size(S, 2) == D^2 || throw(DimensionMismatch("The map must have `length(ψ)^2` = $(D^2) columns, got $(size(S, 2))"))
    length(w) == size(S, 1) || throw(DimensionMismatch("The output vector must have length $(size(S, 1)), got $(length(w))"))
    St = parent(S)
    rows = rowvals(St)
    vals = nonzeros(St)
    @inbounds for K in axes(St, 2)
        acc = zero(eltype(w))
        for p in nzrange(St, K)
            k, j = divrem(rows[p] - 1, D) .+ 1 # rows[p] is the column-major linear index of (j, k) in ψψ†
            acc += vals[p] * (ψ[j] * conj(ψ[k]))
        end
        w[K] = acc
    end
    return w
end

_superop_mul_matrix!(y::AbstractVector, S::AbstractMatrix, m::SparseMatrixCSC) = mul!(y, S, m[:])
# y = S * vec(m) for S = transpose(St) and a sparse D × D matrix m: each stored entry of S looks up
# its matrix element of m by binary search in the column of m.
function _superop_mul_matrix!(y::AbstractVector, S::LinearAlgebra.Transpose{<:Any,<:SparseMatrixCSC}, m::SparseMatrixCSC)
    D = size(m, 1)
    size(m, 2) == D && size(S, 2) == D^2 || throw(DimensionMismatch("The input matrix must be square with $(size(S, 2)) entries, got size $(size(m))"))
    length(y) == size(S, 1) || throw(DimensionMismatch("The output vector must have length $(size(S, 1)), got $(length(y))"))
    St = parent(S)
    rows = rowvals(St)
    vals = nonzeros(St)
    mrows = rowvals(m)
    mvals = nonzeros(m)
    @inbounds for K in axes(St, 2)
        acc = zero(eltype(y))
        for p in nzrange(St, K)
            k, j = divrem(rows[p] - 1, D) .+ 1
            rng = nzrange(m, k)
            i = searchsortedfirst(view(mrows, rng), j)
            if i <= length(rng) && mrows[rng[i]] == j
                acc += vals[p] * mvals[rng[i]]
            end
        end
        y[K] = acc
    end
    return y
end

"""
    partial_trace(H => Hsub; kwargs...)

Create a callable partial-trace operation from `H` to `Hsub`.

The operation can be applied directly as `op(m)` and can be converted to a sparse
matrix map with `sparse(op)`.
"""
struct PartialTraceMap{H,HS,C,K,M}
    H::H
    Hsub::HS
    complement::C
    kwargs::K
    map::M
end

function PartialTraceMap(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), kwargs...)
    map = partial_trace_map(H, Hsub, complement, alg; kwargs...)
    PartialTraceMap(H, Hsub, complement, kwargs, map)
end

vec_to_density_matrix(v::AbstractVector) = LowRankMatrix(v, conj(v))
vec_to_density_matrix(v::SparseVector) = v*v'

# Vectorize into a type the sparse map can multiply without the generic getindex fallback
_vectorize_pt_density_matrix(rho::DenseMatrix) = vec(rho)
_vectorize_pt_density_matrix(rho::SparseMatrixCSC) = rho[:]
_vectorize_pt_density_matrix(rho::AbstractMatrix) = vec(Matrix(rho)) # Adjoint, Hermitian, Diagonal, LowRankMatrix, ...

_devectorize_pt_output(w::AbstractVector, n) = reshape(w, n, n)
function _devectorize_pt_output(w::SparseVector, n)
    idx = SparseArrays.nonzeroinds(w)
    sparse(rem.(idx .- 1, n) .+ 1, div.(idx .- 1, n) .+ 1, nonzeros(w), n, n)
end

function _canonicalize_pt_input(v::AbstractVector, H::AbstractHilbertSpace)
    if length(v) == dim(H)
        return _vectorize_pt_density_matrix(vec_to_density_matrix(v))
    elseif length(v) == dim(H)^2
        return v
    else
        throw(DimensionMismatch("The input vector must have length $(dim(H)^2) (interpreted as a vectorized density matrix) or $(dim(H)) (interpreted as a pure state, which is then converted to a density matrix), got $(length(v))"))
    end
end
_check_pt_input_size(m::AbstractMatrix, H::AbstractHilbertSpace) =
    size(m) == (dim(H), dim(H)) || throw(DimensionMismatch("The input matrix must have size ($(dim(H)), $(dim(H))), got $(size(m))"))
function _canonicalize_pt_input(m::AbstractMatrix, H::AbstractHilbertSpace)
    _check_pt_input_size(m, H)
    return _vectorize_pt_density_matrix(m)
end
function _canonicalize_pt_input(m::UniformScaling, H::AbstractHilbertSpace)
    return _vectorize_pt_density_matrix(m.λ * I(dim(H)))
end
function _canonicalize_pt_output(v::AbstractVector, Hsub::AbstractHilbertSpace)
    length(v) == dim(Hsub)^2 || throw(DimensionMismatch("The output vector must have length $(dim(Hsub)^2), got $(length(v))"))
    return v
end
function _canonicalize_pt_output(m::AbstractMatrix, Hsub::AbstractHilbertSpace)
    size(m) == (dim(Hsub), dim(Hsub)) || throw(DimensionMismatch("The output matrix must have size ($(dim(Hsub)), $(dim(Hsub))), got $(size(m))"))
    return vec(m)
end

function (op::PartialTraceMap)(in::AbstractMatrix)
    v = _canonicalize_pt_input(in, op.H)
    _devectorize_pt_output(op.map * v, dim(op.Hsub))
end
# Sparse input: apply the map to the stored entries, without vectorizing the matrix
function (op::PartialTraceMap)(in::SparseMatrixCSC)
    _check_pt_input_size(in, op.H)
    n = dim(op.Hsub)
    w = _superop_mul_matrix!(zeros(promote_type(eltype(op.map), eltype(in)), n^2), op.map, in)
    return sparse(reshape(w, n, n))
end
function (op::PartialTraceMap)(out, in::SparseMatrixCSC)
    vout = _canonicalize_pt_output(out, op.Hsub)
    _check_pt_input_size(in, op.H)
    _superop_mul_matrix!(vout, op.map, in)
    out
end
function (op::PartialTraceMap)(in::AbstractVector)
    if length(in) == dim(op.H)
        # pure state: apply the map directly, without forming the density matrix
        w = _apply_ptmap_on_vec(op, in)
        return reshape(w, dim(op.Hsub), dim(op.Hsub))
    end
    v = _canonicalize_pt_input(in, op.H)
    _devectorize_pt_output(op.map * v, dim(op.Hsub))
end
function (op::PartialTraceMap)(in::UniformScaling)
    v = _canonicalize_pt_input(in, op.H)
    _devectorize_pt_output(op.map * v, dim(op.Hsub))
end

function _apply_ptmap_on_vec(ptmap::PartialTraceMap, ψ::AbstractVector)
    w = zeros(eltype(ψ), dim(ptmap.Hsub)^2)
    _apply_ptmap_on_vec!(w, ptmap, ψ)
    return w
end
_apply_ptmap_on_vec!(w::AbstractVector, ptmap::PartialTraceMap, ψ::AbstractVector) = _apply_superop_on_pure_state!(w, ptmap.map, ψ)

function (op::PartialTraceMap)(out, in::AbstractMatrix)
    vout = _canonicalize_pt_output(out, op.Hsub)
    vin = _canonicalize_pt_input(in, op.H)
    mul!(vout, op.map, vin)
    out
end
function (op::PartialTraceMap)(out, in::AbstractVector)
    if length(in) == dim(op.H)
        # pure state: apply the map directly, without forming the density matrix
        vout = _canonicalize_pt_output(out, op.Hsub)
        _apply_ptmap_on_vec!(vout, op, in)
        return out
    end
    vout = _canonicalize_pt_output(out, op.Hsub)
    vin = _canonicalize_pt_input(in, op.H)
    mul!(vout, op.map, vin)
    out
end
SparseArrays.sparse(op::PartialTraceMap) = op.map

partial_trace(Hs::Pair{<:AbstractHilbertSpace,<:AbstractHilbertSpace}; kwargs...) = PartialTraceMap(first(Hs), last(Hs); kwargs...)
function partial_trace_map(H, Hsub; complement=complementary_subsystem(H, Hsub), alg=default_partial_trace_alg(Hsub, H, complement), kwargs...)
    partial_trace_map(H, Hsub, complement, alg; kwargs...)
end

partial_trace_map(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, ::Nothing, ::AbstractPartialTraceAlg; kwargs...) = transpose(transposed_basis_transformation_map(H, Hsub; kwargs...))
function partial_trace_map(H, Hsub, complement::AbstractHilbertSpace, ::SubsystemPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=true, phase_factors=true)
    indK = LinearIndices((1:dim(Hsub), 1:dim(Hsub)))
    indJ = LinearIndices((1:dim(H), 1:dim(H)))
    T = partial_trace_phase_factor_eltype(H)
    n_est = dim(Hsub)^2 * dim(complement)
    Is = sizehint!(Int[], n_est)
    Js = sizehint!(Int[], n_est)
    Vs = sizehint!(T[], n_est)
    _foreach_partial_trace_term(nothing, H, Hsub, complement, mapper, SubsystemPartialTraceAlg(); skipmissing, phase_factors) do J1, J2, K1, K2, v
        push!(Is, indK[K1, K2])
        push!(Js, indJ[J1, J2])
        push!(Vs, v)
    end
    return _transposed_superoperator(Is, Js, Vs, dim(Hsub)^2, dim(H)^2)
end

function partial_trace_map(H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, complement::AbstractHilbertSpace, ::FullPartialTraceAlg, mapper=state_mapper(H, (Hsub, complement)); skipmissing=false, phase_factors=true)
    indK = LinearIndices((1:dim(Hsub), 1:dim(Hsub)))
    indJ = LinearIndices((1:dim(H), 1:dim(H)))
    T = partial_trace_phase_factor_eltype(H)
    n_est = dim(Hsub)^2 * dim(complement)
    Is = sizehint!(Int[], n_est)
    Js = sizehint!(Int[], n_est)
    Vs = sizehint!(T[], n_est)
    _foreach_partial_trace_term(CartesianIndices((dim(H), dim(H))), H, Hsub, complement, mapper, FullPartialTraceAlg(); phase_factors, skipmissing) do J1, J2, K1, K2, v
        push!(Is, indK[K1, K2])
        push!(Js, indJ[J1, J2])
        push!(Vs, v)
    end
    return _transposed_superoperator(Is, Js, Vs, dim(Hsub)^2, dim(H)^2)
end


function project_on_parities(op::AbstractArray, H, Hs, parities)
    length(Hs) == length(parities) || throw(ArgumentError("The number of `parities` must match the number of subsystems, got $(length(parities)) parities and $(length(Hs)) subsystems."))
    for (bsub, parity) in zip(Hs, parities)
        op = project_on_subparity(op, H, bsub, parity)
    end
    return op
end

function project_on_subparity(op::AbstractArray, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, parity)
    P = embed(parityoperator(Hsub), Hsub, H)
    return project_on_parity(op, P, parity)
end

project_on_parity(op::AbstractArray, H::AbstractHilbertSpace, parity) = project_on_parity(op, parityoperator(H), parity)

function project_on_parity(op::AbstractMatrix, P::AbstractMatrix, parity)
    parity in (-1, 1) ||
        throw(ArgumentError("`parity` must be either `1` or `-1`, got $parity."))
    return (op + parity * (P * op * P)) / 2
end
function project_on_parity(op::AbstractVector, P::AbstractMatrix, parity)
    if parity == 1
        return (op .+ P * op) ./ 2
    elseif parity == -1
        return (op .- P * op) ./ 2
    else
        throw(ArgumentError("`parity` must be either `1` or `-1`, got $parity."))
    end
end

@testitem "Parity projection" begin
    import FermionicHilbertSpaces: project_on_parity, project_on_parities
    @fermions f
    Hs = [hilbert_space(f, (2k-1):2k) for k in 1:3]
    H = tensor_product(Hs)
    op = rand(ComplexF64, dim(H), dim(H))
    local_parity_iter = (1, -1)
    all_parities = Base.product([local_parity_iter for _ in 1:length(Hs)]...)
    @test sum(project_on_parities(op, H, Hs, parities) for parities in all_parities) ≈ op

    ops = [rand(ComplexF64, dim(H), dim(H)) for H in Hs]
    for parities in all_parities
        projected_ops = [project_on_parity(op, Hsub, parity) for (op, Hsub, parity) in zip(ops, Hs, parities)]
        local op = tensor_product(projected_ops, Hs, H)
        @test op ≈ project_on_parities(op, H, Hs, parities)
    end
end

@testitem "Partial trace map" begin
    using LinearAlgebra
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsub = hilbert_space(f, [2, 4])
    m = rand(ComplexF64, dim(H), dim(H))

    msub = partial_trace(m, H => Hsub)
    pt = partial_trace(H => Hsub)
    msub_map = pt(vec(m))
    @test msub ≈ reshape(msub_map, (dim(Hsub), dim(Hsub)))

    H = hilbert_space(f, 1:4, NumberConservation(2))
    Hsub = subregion(hilbert_space(f, [1, 3, 4]), H)
    m = rand(ComplexF64, dim(H), dim(H))
    msub = partial_trace(m, H => Hsub)
    pt = partial_trace(H => Hsub)
    msub_map = pt(vec(m))
    @test msub ≈ reshape(msub_map, (dim(Hsub), dim(Hsub)))

    Hvac = hilbert_space(f, 1:1, NumberConservation(0))
    Hmode = hilbert_space(f, 2:2)
    Hproduct = tensor_product(Hvac, Hmode)
    ρ = [1 2 + im; 3 - im 4]
    @test dim(Hvac) == 1
    @test partial_trace(ρ, Hproduct => Hvac) ≈ reshape([tr(ρ)], 1, 1)
    @test partial_trace(ρ, Hproduct => Hmode) ≈ ρ
end

@testitem "Partial trace with missing states" begin
    using LinearAlgebra
    @fermions a
    H1 = hilbert_space(a, 1:2, NumberConservation(1))
    H2 = hilbert_space(a, 3:4, NumberConservation(1))
    H = hilbert_space(a, 1:4, NumberConservation(2))
    m1 = rand(ComplexF64, dim(H1), dim(H1))
    m2 = rand(ComplexF64, dim(H2), dim(H2))
    p = FermionicHilbertSpaces.state_mapper(H, (H1, H2))

    m12 = tensor_product((m1, m2), (H1, H2) => H)
    m1ext = embed(m1, H1 => H)
    m2ext = embed(m2, H2 => H)
    @test m1ext * m2ext ≈ m12
    m21 = tensor_product((m2, m1), (H2, H1) => H)
    @test m2ext * m1ext ≈ m21
    alg1 = FermionicHilbertSpaces.FullPartialTraceAlg()
    alg2 = FermionicHilbertSpaces.SubsystemPartialTraceAlg()
    m12_traced1_alg1 = partial_trace(m12, H => H1; alg=alg1)
    m12_traced2_alg1 = partial_trace(m12, H => H2; alg=alg1)
    m12_traced1_alg2 = partial_trace(m12, H => H1; alg=alg2)
    m12_traced2_alg2 = partial_trace(m12, H => H2; alg=alg2)
    @test m12_traced1_alg1 ≈ m1 * tr(m2)
    @test m12_traced2_alg1 ≈ m2 * tr(m1)
    @test m12_traced1_alg2 ≈ m1 * tr(m2)
    @test m12_traced2_alg2 ≈ m2 * tr(m1)

    m12full = rand(ComplexF64, dim(H), dim(H))
    @test_throws ArgumentError partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.FullPartialTraceAlg(), skipmissing=false)
    @test partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.FullPartialTraceAlg(), skipmissing=true) ≈
        partial_trace(m12full, H => H1; alg=FermionicHilbertSpaces.SubsystemPartialTraceAlg(), skipmissing=true)

    H = hilbert_space(a, 1:4, NumberConservation(1))
    @test_throws ArgumentError generalized_kron((m1, m2), (H1, H2) => H)
    @test generalized_kron((m1, m2), (H1, H2) => H; skipmissing=true) ≈ tensor_product((m1, m2), (H1, H2) => H) # Since tensor_product first embeds both operators in the full Hilbert space (which can be done) and then multiplies them, it won't complain about missing states.
end

@testitem "Partial trace and embed of sparse matrices" begin
    using SparseArrays
    # the map builder sums repeated entries like `sparse`
    Is, Js, Vs = [2, 1, 2, 1, 3, 2], [9, 1, 4, 1, 2, 9], [1, 2, 3, -2, 5, 6]
    @test FermionicHilbertSpaces._transposed_superoperator(Is, Js, Vs, 4, 9) == sparse(Is, Js, Vs, 4, 9)

    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    Hfull = hilbert_space(f, 1:2)
    # a subsystem, and a space with the same atoms (a basis transformation)
    for (H, Hsub) in ((Hcons, subregion(Hfull, Hcons)), (Hfull, hilbert_space(f, 1:2, ParityConservation())))
        pt = partial_trace(H => Hsub)
        emb = embed(Hsub => H; skipmissing=true)
        m = sprand(ComplexF64, dim(H), dim(H), 0.3)
        msub = sprand(ComplexF64, dim(Hsub), dim(Hsub), 0.5)
        @test pt(m) ≈ partial_trace(Matrix(m), H => Hsub)
        @test pt(zeros(ComplexF64, dim(Hsub), dim(Hsub)), m) ≈ partial_trace(Matrix(m), H => Hsub)
        @test emb(msub) ≈ embed(Matrix(msub), Hsub => H; skipmissing=true)
        @test pt'(msub) ≈ emb(msub)
        @test emb'(m) ≈ pt(m)
        @test_throws DimensionMismatch pt(sprand(dim(H) + 1, dim(H) + 1, 0.5))
    end
end

@testitem "Partial trace and embed across truncations" begin
    import FermionicHilbertSpaces: FullPartialTraceAlg, SubsystemPartialTraceAlg
    @fermions f
    @bosons b
    @spin s
    Hb1, Hb2 = (hilbert_space(b[1], n) for n in 2:3)
    Hf2 = hilbert_space(f, 1:2)
    Hf1 = hilbert_space(f, 1:2, NumberConservation(1))
    Hs1 = hilbert_space(s, 1//2)
    Hs2 = hilbert_space(s, 3//2)
    tp(h) = h
    tp((h1, h2)::Tuple) = tensor_product((h1, h2))
    for (hsmall, hbig) in [(Hb1, Hb2), (Hb1, (Hb2, Hf1)),
        (Hf1, Hf2), (Hf1, (Hb1, Hf2)),
        ((Hb1, Hf1), (Hb2, Hf1)), ((Hb1, Hf1), (Hb1, Hf2)), ((Hb1, Hf1), (Hb2, Hf2)),
        ((Hb2, Hf1), (Hb2, Hf2)),
        ((Hb1, Hf2), (Hb2, Hf2)),
        (Hs1, Hs2), ((Hs1, (Hs2, Hf1)))]
        Hsmall = tp(hsmall)
        Hbig = tp(hbig)
        dsmall = dim(Hsmall)
        dbig = dim(Hbig)
        Msmall = rand(dsmall, dsmall)
        Mbig = rand(dbig, dbig)

        @test_throws ArgumentError partial_trace(Mbig, Hbig => Hsmall, alg=FullPartialTraceAlg())
        @test size(partial_trace(Mbig, Hbig => Hsmall, alg=FullPartialTraceAlg(), skipmissing=true)) == (dsmall, dsmall)
        @test size(partial_trace(Mbig, Hbig => Hsmall, alg=SubsystemPartialTraceAlg(), skipmissing=true)) == (dsmall, dsmall)
        @test_throws ArgumentError partial_trace(Hbig => Hsmall, alg=FullPartialTraceAlg()) # drops states, so errors

        @test size(embed(Msmall, Hsmall => Hbig)) == (dim(Hbig), dim(Hbig))
        emb = embed(Hsmall => Hbig)
        @test emb(Msmall) ≈ embed(Msmall, Hsmall => Hbig)


        same_space = try
            FermionicHilbertSpaces.match_atoms(Hbig, Hsmall)
            true
        catch e
            false
        end
        if same_space # if Hsmall has the same labels, but smaller basis set:
            @test size(partial_trace(Msmall, Hsmall => Hbig)) == (dim(Hbig), dim(Hbig))

            pt = partial_trace(Hsmall => Hbig)
            @test_throws ArgumentError embed(Hbig => Hsmall) # drops states, so errors
            emb = embed(Hsmall => Hbig)

            inds = indices(Hsmall, Hbig)[:]
            Mbigrestricted = zeros(eltype(Mbig), dim(Hbig), dim(Hbig))
            Mbigrestricted[inds, inds] = Mbig[inds, inds]
            ptM = partial_trace(Mbigrestricted, Hbig => Hsmall, alg=FullPartialTraceAlg(), skipmissing=false)
            @test ptM == Mbigrestricted[inds, inds]
            @test partial_trace(Mbigrestricted, Hbig => Hsmall, alg=SubsystemPartialTraceAlg(), skipmissing=false) == ptM

            @test embed(Mbigrestricted, Hbig => Hsmall) ≈ Mbigrestricted[inds, inds]
            @test_throws ArgumentError embed(Mbig, Hbig => Hsmall) # drops states, so errors
            @test emb(Msmall) ≈ pt(Msmall) ≈ partial_trace(Msmall, Hsmall => Hbig) ≈ embed(Msmall, Hsmall => Hbig)
        end
    end
end

@testitem "change_basis requires the same atoms" begin
    import FermionicHilbertSpaces: change_basis, basis_transformation
    @fermions f
    @bosons b
    Hb1 = hilbert_space(b[1], 3)
    Hb2 = hilbert_space(b[2], 3)
    Hf = hilbert_space(f, 1:2)

    # Different atoms with identical basis states
    @test_throws ArgumentError basis_transformation(Hb1, Hb2)
    @test_throws ArgumentError change_basis(rand(3, 3), Hb1, Hb2)
    @test_throws ArgumentError indices(Hb1, Hb2)
    # Missing or extra atoms
    @test_throws ArgumentError change_basis(rand(3, 3), Hb1, tensor_product(Hb1, Hf))
    @test_throws ArgumentError change_basis(rand(12, 12), tensor_product(Hb1, Hf), Hb1)

    # Pair syntax
    m = rand(3, 3)
    @test change_basis(m, Hb1 => hilbert_space(b[1], 4)) == change_basis(m, Hb1, hilbert_space(b[1], 4))

    # Groups in different orders
    Hfb = tensor_product(Hf, Hb1)
    Hbf = tensor_product(Hb1, Hf)
    op = f[1]' * f[2] * b[1]' + hc
    mfb = representation(op, Hfb)
    mbf = representation(op, Hbf)
    @test change_basis(mfb, Hfb => Hbf) ≈ mbf
    @test change_basis(mbf, Hbf => Hfb) ≈ mfb
    v = rand(dim(Hfb))
    @test change_basis(change_basis(v, Hfb => Hbf), Hbf => Hfb) == v
end
