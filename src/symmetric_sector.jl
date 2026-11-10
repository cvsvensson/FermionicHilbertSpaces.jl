"""
    Translation(sites)

The cyclic translations of `sites`, given as positions in `factors(H)` or as factor spaces of `H`. The constraint
`Translation(sites) => k`, with an integer `k`, keeps the states of momentum `k`: a state with a single excitation
has the amplitude `cispi(2k * j / N)` on `sites[j]`, up to normalization, where `N = length(sites)`. It gives a
[`SymmetricSector`](@ref) and keeps the sectors of a [`SectorHilbertSpace`](@ref).

```julia
@spins S 1 // 2
Sz = sum(S[j][:z] for j in 1:N)
Hk = hilbert_space(S, 1:N, (Sz, Translation(1:N) => k))
representation(ham, sector(0, Hk), :lazy)
```
"""
struct Translation{S}
    sites::S
end
as_constraint(::Translation) = throw(ArgumentError("A translation needs a momentum, as in `Translation(sites) => k`."))

struct TranslationConstraint{S} <: AbstractConstraint
    translation::Translation{S}
    momentum::Int
end
as_constraint((t, k)::Pair{<:Translation}) = TranslationConstraint(t, k)

# The permutations of `factors(H)[sites]` and their weights, which select momentum `k` in `_orbit_block_basis`
function _permutations_and_weights(c::TranslationConstraint)
    N = length(c.translation.sites)
    return [[mod1(i + s, N) for i in 1:N] for s in 0:(N - 1)], [cispi(-2 * c.momentum * s / N) for s in 0:(N - 1)]
end
_site_spaces(H, sites) = eltype(sites) <: Integer ? collect(factors(H))[sites] : collect(sites)

# A constraint with a translation is applied as the rest of the constraint followed by the translation.
_split_translation(c) = (c, nothing)
function _split_translation(c::ProductConstraint)
    i = findfirst(x -> x isa TranslationConstraint, c.constraints)
    isnothing(i) && return (c, nothing)
    return as_constraint(collect(c.constraints[[1:(i - 1); (i + 1):end]])), c.constraints[i]
end

"""
    OrbitRep(state, index)

Basis state `index` of a [`SymmetricSector`](@ref): the normalized symmetric combination of the orbit of the parent
basis state `state`, its representative.
"""
struct OrbitRep{B} <: AbstractBasisState
    state::B
    index::Int
end

"""
    SymmetricSector

The states of a parent space with a fixed momentum under [`Translation`](@ref)s, with one basis vector per orbit
of parent basis states, built by `constrain_space(H, Translation(sites) => k)`. Its basis states are
[`OrbitRep`](@ref)s, so functions that need product states, such as `partial_trace` and `tensor_product`, throw.

Matrix representations, including `:lazy`, apply the operator to one representative per basis vector. They
throw an `ArgumentError` unless the operator commutes with the translation by one site on every representative,
a check that costs about as much as building the sparse matrix. Use `embed(v, Hk => H)` and
`change_basis(v, H => Hk; projection = true)` to move states between `Hk` and its parent `H`.
"""
struct SymmetricSector{B, P, T, M <: SparseMatrixCSC{T}} <: AbstractHilbertSpace{OrbitRep{B}}
    parent::P
    symmetry::TranslationConstraint
    # `amplitudes[c, t]` is the amplitude of parent basis state `t` in basis vector `c`, with one entry per column.
    amplitudes::M
    reps::Vector{B}
    norms::Vector{Float64}
    transposed::Bool
end

function constrain_space(space::AbstractHilbertSpace, c::TranslationConstraint)
    perms, weights = _permutations_and_weights(c)
    T = all(isreal, weights) ? Float64 : ComplexF64
    Q = _orbit_block_basis(space, _site_spaces(space, c.translation.sites), perms, T.(weights), _QR(), T)
    reps = [basisstate(rowvals(Q)[first(nzrange(Q, n))], space) for n in axes(Q, 2)]
    norms = [real(nonzeros(Q)[first(nzrange(Q, n))]) for n in axes(Q, 2)]
    H = SymmetricSector(space, c, sparse(transpose(Q)), reps, norms, false)
    space isa SectorHilbertSpace || return H
    # Translations commute with the quantities of the sectors, so each orbit lies in one sector.
    qn = Dict(state => q for q in quantumnumbers(space) for state in _sector_states(q, space))
    return sector_space(H, collect(basisstates(H)), s -> qn[s.state], space.constraint)
end

Base.parent(H::SymmetricSector) = H.parent
dim(H::SymmetricSector) = length(H.reps)
basisstate(n::Integer, H::SymmetricSector) = OrbitRep(H.reps[n], Int(n))
basisstates(H::SymmetricSector) = BasisStates(H, Iterators.map(n -> basisstate(n, H), 1:dim(H)))
_state_index(state::OrbitRep, ::SymmetricSector) = state.index
_state_index(state, H::SymmetricSector) = _state_index(interpret_state(state, H), H)
interpret_state(state::OrbitRep, ::SymmetricSector) = state
interpret_state(state, ::SymmetricSector) = throw(ArgumentError("`$state` is not a basis state of a `SymmetricSector`. Represent it on the parent space `H` and use `change_basis(v, H => Hk; projection = true)`."))
has_arithmetic_state_index(::SymmetricSector) = true
isconstrained(::SymmetricSector) = true
groups(H::SymmetricSector) = groups(H.parent)
Base.:(==)(H1::SymmetricSector, H2::SymmetricSector) = H1 === H2 || (H1.parent == H2.parent && H1.symmetry == H2.symmetry && H1.transposed == H2.transposed)
Base.hash(H::SymmetricSector, h::UInt) = hash((H.parent, H.symmetry, H.transposed), h)
Base.show(io::IO, H::SymmetricSector) = print(io, "$(dim(H))-dimensional SymmetricSector(", H.symmetry.translation, " => ", H.symmetry.momentum, ")")
# The basis of the transposed space is the complex conjugate one.
TransposedSpace(H::SymmetricSector) = SymmetricSector(TransposedSpace(H.parent), H.symmetry, H.amplitudes, H.reps, H.norms, !H.transposed)

_amplitude_eltype(H::SymmetricSector) = eltype(H.amplitudes)
_check_operator(op, H::SymmetricSector) = H.transposed ? nothing : _check_invariant(op, H)

# For an operator that commutes with the translations, ⟨c′|op|c⟩ = Σ_t conj(q_c′(t)) ⟨t|op|r⟩ / q_c(r), where `r`
# is the representative of `c` and `q_c` the amplitudes of basis vector `c`.
function _apply_local_operators(op, state::OrbitRep, H::SymmetricSector, precomp)
    newstate, amp = _apply_local_operators(op, state.state, H.parent, precomp)
    T = promote_type(typeof(amp), eltype(H.amplitudes))
    iszero(amp) && return state, zero(T)
    t = state_index(newstate, H.parent)
    iszero(t) && return OrbitRep(newstate, 0), T(amp) # outside the parent, which `projection` decides on
    p = SparseArrays.getcolptr(H.amplitudes)[t]
    # an orbit that has no state at this momentum
    p == SparseArrays.getcolptr(H.amplitudes)[t + 1] && return state, zero(T)
    c, q = rowvals(H.amplitudes)[p], nonzeros(H.amplitudes)[p]
    return OrbitRep(H.reps[c], c), T(amp * (H.transposed ? q : conj(q)) / H.norms[state.index])
end

# Throws unless `op(g r) == g op(r)` for the translation `g` by one site and every representative `r`.
function _check_invariant(op, H::SymmetricSector)
    space = H.parent
    op isa AbstractSym && (op = NCMul(1, [op]))
    terms = _precomputed_terms([partition_product(term, _checked_group_ids(op, space), groups(space)) for term in NCterms(op)], space)
    mapper = state_mapper(space, _site_spaces(space, H.symmetry.translation.sites))
    N = length(H.symmetry.translation.sites)
    g = _match_perm([mod1(i + 1, N) for i in 1:N], unique_split_state(first(H.reps), mapper))
    shifted = [permute_state(r, mapper, g) for r in H.reps]
    T = mat_eltype(op)
    gop, opg = (Int[], Int[], T[]), (Int[], Int[], T[])
    for (term, precomp) in terms
        _add_invariance_entries!(gop, opg, term, precomp, H.reps, shifted, space, mapper, g)
    end
    sparse(gop..., dim(space), dim(H)) ≈ sparse(opg..., dim(space), dim(H)) ||
        throw(ArgumentError("The operator does not commute with the translation, so it has no representation on the momentum sector."))
    return nothing
end
function _add_invariance_entries!(gop, opg, term, precomp, reps, shifted, space, mapper, g)
    for (n, (r, gr)) in enumerate(zip(reps, shifted))
        s, amp = _apply_local_operators(term, r, space, precomp)
        # images outside the parent are left to `projection`
        iszero(amp) || iszero(state_index(s, space)) || push_inds_amps!(gop, state_index(permute_state(s, mapper, g), space), n, amp)
        s, amp = _apply_local_operators(term, gr, space, precomp)
        i = iszero(amp) ? 0 : state_index(s, space)
        iszero(i) || push_inds_amps!(opg, i, n, amp)
    end
    return nothing
end

# A symmetric sector, or a subspace of one, with the orbit indices of its basis states
const _OrbitSpace = Union{SymmetricSector, AbstractSubspace{<:Any, <:SymmetricSector}}
_orbits(H::SymmetricSector) = (H, 1:dim(H))
_orbits(H::AbstractSubspace{<:Any, <:SymmetricSector}) = (parent(H), [state.index for state in basisstates(H)])

complementary_subsystem(H::AbstractHilbertSpace, ::_OrbitSpace) = nothing
function basis_transformation(Hfrom::_OrbitSpace, Hto::_OrbitSpace)
    (H1, from), (H2, to) = _orbits(Hfrom), _orbits(Hto)
    H1 == H2 || throw(ArgumentError("Both spaces must be built from the same `SymmetricSector`."))
    position = Dict(c => i for (i, c) in enumerate(to))
    js = findall(c -> haskey(position, c), from)
    U = sparse([position[from[j]] for j in js], js, ones(Int, length(js)), length(to), length(from))
    return U, setdiff(eachindex(from), js)
end
# The basis vectors of `Hfrom` in `Hto`
function basis_transformation(Hfrom::_OrbitSpace, Hto::AbstractHilbertSpace)
    Hk, cols = _orbits(Hfrom)
    U, dropped = basis_transformation(Hk.parent, Hto)
    Q = sparse(transpose(Hk.amplitudes[cols, :]))
    return U * Q, findall(!iszero, vec(sum(abs, Q[dropped, :]; dims = 1)))
end
# The projection onto `Hto`, so that a state outside it is not detected
function basis_transformation(Hfrom::AbstractHilbertSpace, Hto::_OrbitSpace)
    U, _ = basis_transformation(Hto, Hfrom)
    return sparse(U'), Int[]
end
function change_basis(m, Hfrom::AbstractHilbertSpace, Hto::_OrbitSpace; projection = false, kwargs...)
    Hfrom isa _OrbitSpace || projection || throw(ArgumentError("A state of `$Hfrom` is in general not in the momentum sector. Pass `projection = true` to project it."))
    return invoke(change_basis, Tuple{Any, Union{AbstractHilbertSpace, Axes}, Union{AbstractHilbertSpace, Axes}}, m, Hfrom, Hto; projection, kwargs...)
end

@testitem "Translation sectors" begin
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: Translation, Axes, change_basis, highest_weight_states
    N = 6
    @spins S 1 // 2
    Sz = sum(S[j][:z] for j in 1:N)
    Splus = sum(S[j][:+] for j in 1:N)
    ham = sum(S[j][:]'S[mod1(j + 1, N)][:] for j in 1:N)
    space = hilbert_space(S, 1:N, Sz)
    perms = [[mod1(i + s, N) for i in 1:N] for s in 0:(N - 1)]
    Hks = [hilbert_space(S, 1:N, (Sz, Translation(1:N) => k)) for k in 0:(N - 1)]

    @test sort(reduce(vcat, [eigvals(Hermitian(representation(ham, sector(0, Hk), Matrix))) for Hk in Hks])) ≈
        eigvals(Hermitian(representation(ham, sector(0, space), Matrix)))

    k = 1
    Hk = Hks[k + 1]
    H0 = sector(0, Hk)
    L, M = representation(ham, H0, :lazy), representation(ham, H0)
    v = randn(ComplexF64, dim(H0))
    @test L * v ≈ M * v
    @test L' * v ≈ M' * v
    @test transpose(L) * v ≈ transpose(M) * v

    Q(m) = symmetric_sector(sector(m, space), factors(space), (perms, [cispi(-2k * s / N) for s in 0:(N - 1)]))
    @test representation(Splus, Axes(sector(1, Hk), H0)) ≈ Q(1)' * representation(Splus, Axes(sector(1, space), sector(0, space))) * Q(0)

    # a single excitation has amplitude cispi(2k j / N) on site j
    magnon = embed([1], sector(N // 2 - 1, Hk) => space)
    amplitudes = [magnon[state_index(join(i == j ? "↓" : "↑" for i in 1:N), space)] for j in 1:N]
    @test amplitudes ≈ amplitudes[1] * [cispi(2k * j / N) for j in 0:(N - 1)]

    @test change_basis(embed(v, H0 => space), space => H0; projection = true) ≈ v
    @test_throws ArgumentError change_basis(embed(v, H0 => space), space => H0)
    @test_throws ArgumentError representation(S[1][:z], H0)
    @test_throws ArgumentError representation("↑↑↑↓↓↓", H0)
    @test_throws MethodError partial_trace(Matrix(I, dim(H0), dim(H0)), H0 => hilbert_space(S[1]))
    @test_throws MethodError tensor_product(parent(H0), hilbert_space(S[1]))

    count_multiplets(H) = Dict(s => size(V, 2) for (s, V) in highest_weight_states(H, Splus))
    Hfull = hilbert_space(S, 1:N, Sz => 0:(N // 2))
    @test mergewith(+, [count_multiplets(constrain_space(Hfull, Translation(1:N) => k)) for k in 0:(N - 1)]...) == count_multiplets(Hfull)
end
