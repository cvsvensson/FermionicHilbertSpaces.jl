"""
    Permutation(g)

A symmetry that permutes the modes of a space: `g` maps each mode label to the label of its image, so the
permutation acts on operators as `f[l] -> f[g(l)]`. The constraint `Permutation(g) => k`, with an integer `k`, keeps
the states on which the permutation acts as `cispi(2k / n)`, where `n` is its order. Several such pairs in one
constraint keep the states of the group the permutations generate. The result is a [`SymmetricSector`](@ref), which
keeps the sectors of a [`SectorHilbertSpace`](@ref).

```julia
T(j) = mod1(j + 1, N)
Hk = hilbert_space(S, 1:N, (Sz, Permutation(T) => k))   # momentum k: amplitude cispi(2k j / N) on site j
representation(ham, sector(0, Hk), :lazy)
```
"""
struct Permutation{G}
    map::G
end
as_constraint(::Permutation) = throw(ArgumentError("A permutation needs a charge, as in `Permutation(g) => k`."))

struct PermutationConstraint{M <: Tuple, C <: Tuple} <: AbstractConstraint
    maps::M
    charges::C
end

"""
    LocalMap(f)

A symmetry that acts with the single-site operator `f(l)` on every site `l`, which must permute the basis states of the
site in the same way on every site, as `l -> 2S[l][:x]` does for spin 1/2. `LocalMap(f) => k` works like
[`Permutation`](@ref) and combines with it.
"""
struct LocalMap{F}
    op::F
end
as_constraint(::LocalMap) = throw(ArgumentError("A local map needs a charge, as in `LocalMap(f) => k`."))

function as_constraint((p, k)::Pair{<:Union{Permutation, LocalMap}})
    k isa Integer || throw(ArgumentError("The charge of a permutation is an integer, got `$k`."))
    return PermutationConstraint((p isa Permutation ? p.map : p,), (k,))
end

# A group element is a factor permutation and a permutation `loc` of the local basis states, applied on every site. The
# two commute, since `loc` is the same on every site.
_generator(g, space, atoms, d) = (_atom_permutation(g, atoms), collect(1:d))
function _generator(l::LocalMap, space, atoms, d)
    phase_factor_u(state_mapper(space, atoms)) == Returns(1) || throw(ArgumentError("A `LocalMap` does not support fermions."))
    locs = map(atoms) do atom
        M = Matrix(representation(l.op(atomic_id(atom).label), atom))
        loc = [findfirst(!iszero, c) for c in eachcol(M)]
        all(isone, sum(!iszero, M; dims = 1)) && all(isone, M[CartesianIndex.(loc, axes(M, 2))]) && isperm(loc) ||
            throw(ArgumentError("`$(l.op)` does not permute the basis states of `$atom`."))
        loc
    end
    allequal(locs) || throw(ArgumentError("`$(l.op)` does not act the same way on every site."))
    return (collect(eachindex(atoms)), first(locs))
end
_compose(g, h) = (h[1][g[1]], g[2][h[2]])

struct SiteMap{P, A}
    perm::P
    loc::Vector{Int}
    atom::A
end
_match_perm(g::SiteMap, substates::NTuple{N, Any}) where {N} = SiteMap(_match_perm(g.perm, substates), g.loc, g.atom)
permute_state(state, mapper, g::SiteMap) = permute_state(state, mapper, g.perm, s -> basisstate(g.loc[state_index(s, g.atom)], g.atom))

# The permutation of `atoms` that `g` induces through the labels of their modes
function _atom_permutation(g, atoms)
    modes = map(atomic_id, atoms)
    position = Dict(m => n for (n, m) in enumerate(modes))
    perm = map(modes) do m
        m isa Mode || throw(ArgumentError("A permutation symmetry acts on mode labels, but `$m` is not a mode."))
        image = Mode(m.field, g(m.label))
        get(() -> throw(ArgumentError("`$g` maps `$m` to `$image`, which is not in the space.")), position, image)
    end
    isperm(perm) || throw(ArgumentError("`$g` does not permute the modes of the space."))
    return perm
end

# The least common multiple of the cycle lengths
function _order(perm)
    n, seen = 1, falses(length(perm))
    for i in eachindex(perm)
        len, j = 0, i
        while !seen[j]
            seen[j] = true
            j, len = perm[j], len + 1
        end
        n = lcm(n, max(len, 1))
    end
    return n
end

# All elements of the group and their characters. A word in the generators has the product of their characters, so
# two words for one element must agree.
function _group_and_characters(generators, λs)
    e = map(x -> collect(eachindex(x)), first(generators))
    characters = Dict(e => one(eltype(λs)))
    queue = [e]
    while !isempty(queue)
        h = popfirst!(queue)
        for (g, λ) in zip(generators, λs)
            gh, λgh = _compose(g, h), characters[h] * λ
            if haskey(characters, gh)
                characters[gh] ≈ λgh || throw(ArgumentError("The charges are not consistent with the group the permutations generate."))
            else
                characters[gh] = λgh
                push!(queue, gh)
            end
        end
    end
    return collect(keys(characters)), collect(values(characters))
end

# A constraint with a permutation symmetry is applied as the rest of the constraint followed by the symmetry.
_split_translation(c) = (c, nothing)
_split_translation(c::PermutationConstraint) = (NoSymmetry(), c)
function _split_translation(c::ProductConstraint)
    cs = collect(c.constraints)
    perms, rest = filter(x -> x isa PermutationConstraint, cs), filter(x -> !(x isa PermutationConstraint), cs)
    isempty(perms) && return (c, nothing)
    joined = PermutationConstraint(Tuple(g for p in perms for g in p.maps), Tuple(k for p in perms for k in p.charges))
    return (isempty(rest) ? NoSymmetry() : as_constraint(rest)), joined
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

The sector `constrain_space(H, Permutation(g) => k)` of a [`Permutation`](@ref), with one basis vector per orbit of the
basis states of `H`. Representing an operator throws an `ArgumentError` unless relabeling its modes by each
permutation leaves it unchanged. Move states with
`embed(v, Hk => H)` and `change_basis(v, H => Hk; projection = true)`.
"""
struct SymmetricSector{B, P, T, M <: SparseMatrixCSC{T}, C <: PermutationConstraint} <: AbstractHilbertSpace{OrbitRep{B}}
    parent::P
    symmetry::C
    # `amplitudes[c, t]` is the amplitude of parent basis state `t` in basis vector `c`, with one entry per column.
    amplitudes::M
    reps::Vector{B}
    norms::Vector{Float64}
    transposed::Bool
end

function constrain_space(space::AbstractHilbertSpace, c::PermutationConstraint)
    atoms = collect(atomic_factors(space))
    d = any(g -> g isa LocalMap, c.maps) ? dim(first(atoms)) : 0
    generators = [_generator(g, space, atoms, d) for g in c.maps]
    elements, λs = _group_and_characters(generators, [cispi(2k / lcm(_order(g[1]), _order(g[2]))) for (g, k) in zip(generators, c.charges)])
    perms = iszero(d) ? first.(elements) : [SiteMap(p, loc, first(atoms)) for (p, loc) in elements]
    T = all(isreal, λs) ? Float64 : ComplexF64
    Q = _orbit_block_basis(space, atoms, perms, T.(conj.(λs)), _QR(), T; check = false)
    reps = [basisstate(rowvals(Q)[first(nzrange(Q, n))], space) for n in axes(Q, 2)]
    norms = [real(nonzeros(Q)[first(nzrange(Q, n))]) for n in axes(Q, 2)]
    H = SymmetricSector(space, c, sparse(transpose(Q)), reps, norms, false)
    space isa SectorHilbertSpace || return H
    qn = Dict(state => q for q in quantumnumbers(space) for state in _sector_states(q, space))
    for n in axes(Q, 2)
        allequal(qn[basisstate(t, space)] for t in rowvals(Q)[nzrange(Q, n)]) ||
            throw(ArgumentError("The symmetry maps states between sectors of the space. Constrain to a sector it keeps first."))
    end
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
# The functions that split basis states into factors, such as `partial_trace`, `tensor_product` and `subregion`,
# get the factors of a space from these.
atomic_factors(H::SymmetricSector) = throw(_no_factors_error(H))
factors(H::SymmetricSector) = throw(_no_factors_error(H))
_no_factors_error(H) = ArgumentError("The basis states of `$H` are translation orbits, not product states, so it has no factors. Use its parent space.")
Base.:(==)(H1::SymmetricSector, H2::SymmetricSector) = H1 === H2 || (H1.parent == H2.parent && H1.symmetry == H2.symmetry && H1.transposed == H2.transposed)
Base.hash(H::SymmetricSector, h::UInt) = hash((H.parent, H.symmetry, H.transposed), h)
Base.show(io::IO, H::SymmetricSector) = print(io, "$(dim(H))-dimensional SymmetricSector(", join(("$(g isa LocalMap ? g : "Permutation($g)") => $k" for (g, k) in zip(H.symmetry.maps, H.symmetry.charges)), ", "), ")")
# The basis of the transposed space is the complex conjugate one. Its operators are transposes of operators already
# checked on `H`, so `_check_operator` skips it.
TransposedSpace(H::SymmetricSector) = SymmetricSector(TransposedSpace(H.parent), H.symmetry, H.amplitudes, H.reps, H.norms, !H.transposed)

# A symmetric sector, or a subspace of one, with the orbit indices of its basis states
const _OrbitSpace = Union{SymmetricSector, AbstractSubspace{<:OrbitRep, <:SymmetricSector}}
_orbits(H::SymmetricSector) = (H, 1:dim(H))
_orbits(H::AbstractSubspace{<:OrbitRep, <:SymmetricSector}) = (parent(H), [state.index for state in basisstates(H)])

_amplitude_eltype(H::SymmetricSector) = eltype(H.amplitudes)
_precomputation_before_operator_application(op, H::_OrbitSpace) = _precomputation_before_operator_application(op, parent(first(_orbits(H))))
_check_operator(op, H::SymmetricSector) = H.transposed ? nothing : _check_invariant(op, H)
# The output index of a basis state is its orbit in the input space, so both spaces must share the orbits.
function _check_operator(op, ax::Axes{<:Tuple{_OrbitSpace, _OrbitSpace}})
    Hout, Hin = ax.spaces
    first(_orbits(Hout)) == first(_orbits(Hin)) || throw(ArgumentError("`Axes(Hout, Hin)` needs two sectors of the same `SymmetricSector`."))
    return _check_operator(op, Hin)
end

# For an operator that commutes with the translations, ⟨c′|op|c⟩ = Σ_t conj(q_c′(t)) ⟨t|op|r⟩ / q_c(r), where `r`
# is the representative of `c` and `q_c` the amplitudes of basis vector `c`.
function _apply_local_operators(op, state::OrbitRep, H::SymmetricSector, precomp)
    newstate, amp = _apply_local_operators(op, state.state, H.parent, precomp)
    T = promote_type(typeof(amp), eltype(H.amplitudes))
    iszero(amp) && return state, zero(T)
    t = state_index(newstate, H.parent)
    iszero(t) && return OrbitRep(newstate, 0), T(amp) # outside the parent, which `projection` decides on
    p = SparseArrays.getcolptr(H.amplitudes)[t]
    # an orbit that has no state in this sector
    p == SparseArrays.getcolptr(H.amplitudes)[t + 1] && return state, zero(T)
    c, q = rowvals(H.amplitudes)[p], nonzeros(H.amplitudes)[p]
    return OrbitRep(H.reps[c], c), T(amp * (H.transposed ? q : conj(q)) / H.norms[state.index])
end

_check_invariant(op, H::SymmetricSector) = foreach(g -> _check_invariant(op, H, g), H.symmetry.maps)
_check_invariant(op, H, g) = iszero(_relabel(op, g) - op) || throw(ArgumentError("The operator changes under the permutation `$g`, so it has no representation on the symmetric sector."))
# Numeric, since conjugating symbolic factors by a local map is not implemented. The permutations commute with the
# operator (checked symbolically) and with the local map, so it suffices to compare on the orbit representatives.
function _check_invariant(op, H, l::LocalMap)
    atoms = collect(atomic_factors(H.parent))
    mapper = state_mapper(H.parent, atoms)
    perm, loc = _generator(l, H.parent, atoms, 0)
    g = _match_perm(SiteMap(perm, loc, first(atoms)), unique_split_state(first(H.reps), mapper))
    gstate(s) = permute_state(s, mapper, g)
    A, B = (Dict{Tuple{Int, eltype(H.reps)}, mat_eltype(op)}() for _ in 1:2)
    bases = _checked_group_ids(op, H.parent)
    for term in NCterms(op)
        term = length(bases) > 1 ? partition_product(term, bases, groups(H.parent)) : term
        _add_elements!(A, term, H.reps, H.parent)
        _add_elements!(B, term, map(gstate, H.reps), H.parent)
    end
    gA = Dict((k, gstate(s)) => a for ((k, s), a) in A)
    tol = sqrt(eps()) * maximum(abs, values(A); init = 1.0)
    all(x -> abs(get(gA, x, 0) - get(B, x, 0)) <= tol, union(keys(gA), keys(B))) ||
        throw(ArgumentError("The operator changes under `$(l.op)`, so it has no representation on the symmetric sector."))
    return nothing
end
_relabel(op::Op, g) = Op(op.kind, Mode(op.mode.field, g(op.mode.label)))
_relabel(m::NCMul, g) = m.coeff * prod(_relabel(x, g) for x in m.factors)
_relabel(a::NCAdd, g) = a.coeff + sum((c * _relabel(m, g) for (m, c) in a.dict); init = 0)
_relabel(op, g) = throw(ArgumentError("Only operators on fields can be checked for symmetry, got `$op`."))

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
    Hfrom isa _OrbitSpace || projection || throw(ArgumentError("A state of `$Hfrom` is in general not in the symmetric sector. Pass `projection = true` to project it."))
    return invoke(change_basis, Tuple{Any, Union{AbstractHilbertSpace, Axes}, Union{AbstractHilbertSpace, Axes}}, m, Hfrom, Hto; projection, kwargs...)
end

@testitem "Translation sectors" begin
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: Permutation, Axes, change_basis, highest_weight_states, permutation_operator
    N = 6
    @spins S 1 // 2
    Sz = sum(S[j][:z] for j in 1:N)
    Splus = sum(S[j][:+] for j in 1:N)
    ham = sum(S[j][:]'S[mod1(j + 1, N)][:] for j in 1:N)
    space = hilbert_space(S, 1:N, Sz)
    perms = [[mod1(i + s, N) for i in 1:N] for s in 0:(N - 1)]
    T(j) = mod1(j + 1, N)
    R(j) = N + 1 - j
    Hks = [hilbert_space(S, 1:N, (Sz, Permutation(T) => k)) for k in 0:(N - 1)]

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
    @test_throws ArgumentError partial_trace(Matrix(I, dim(H0), dim(H0)), H0 => hilbert_space(S[1]))
    @test_throws ArgumentError tensor_product(parent(H0), hilbert_space(S[1]))
    @test_throws ArgumentError representation(Sz, Axes(sector(0, Hks[3]), H0))
    @test_throws ArgumentError hilbert_space(S, 1:N, Permutation(j -> j + 1) => k)
    @test_throws ArgumentError hilbert_space(S, 1:N, Permutation(T) => 1 // 2)
    @test sum(k -> dim(hilbert_space(S, 1:N, Permutation(T) => k)), 0:(N - 1)) == 2^N
    @test size(representation(Sz, hilbert_space(S, 1:N, (Sz => N // 2, Permutation(T) => 1)))) == (0, 0)

    E0(cs...) = eigvals(Hermitian(representation(ham, sector(0, hilbert_space(S, 1:N, (Sz, cs...))), Matrix)))
    @test sort([E0(Permutation(T) => 0, Permutation(R) => 0); E0(Permutation(T) => 0, Permutation(R) => 1)]) ≈ E0(Permutation(T) => 0)
    U = permutation_operator(space, factors(space), circshift(1:N, -1))
    @test U * embed(v, H0 => space) ≈ cispi(2k / N) * embed(v, H0 => space)
    @test_throws ArgumentError hilbert_space(S, 1:N, (Sz, Permutation(T) => 1, Permutation(R) => 0))

    count_multiplets(H) = Dict(s => size(V, 2) for (s, V) in highest_weight_states(H, Splus))
    Hfull = hilbert_space(S, 1:N, Sz => 0:(N // 2))
    @test mergewith(+, [count_multiplets(constrain_space(Hfull, Permutation(T) => k)) for k in 0:(N - 1)]...) == count_multiplets(Hfull)
end

@testitem "Translation sectors of a Hubbard ring" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: Permutation
    N = 4
    T((i, σ)) = (mod1(i + 1, N), σ)
    flip((i, σ)) = (i, σ == :↑ ? :↓ : :↑)
    @fermions f
    ham = sum(-f[i, σ]' * f[mod1(i + 1, N), σ] + hc for σ in (:↑, :↓) for i in 1:N) + 4 * sum(f[i, :↑]'f[i, :↑] * f[i, :↓]'f[i, :↓] for i in 1:N)
    Nσ(σ) = sum(f[i, σ]'f[i, σ] for i in 1:N)
    H = hilbert_space(f, [(i, σ) for σ in (:↑, :↓) for i in 1:N], (Nσ(:↑), Nσ(:↓)))
    Hks = [constrain_space(H, Permutation(T) => k) for k in 0:(N - 1)]
    for q in ((2, 1), (2, 2))
        @test sort(reduce(vcat, [eigvals(Hermitian(representation(ham, sector(q, Hk), Matrix))) for Hk in Hks])) ≈
            eigvals(Hermitian(representation(ham, sector(q, H), Matrix)))
    end
    @test_throws ArgumentError representation(f[1, :↑]'f[1, :↑], sector((2, 1), Hks[1]))

    Hn = hilbert_space(f, [(i, σ) for σ in (:↑, :↓) for i in 1:N], NumberConservation(3))
    Ek(cs...) = eigvals(Hermitian(representation(ham, constrain_space(Hn, (Permutation(T) => 0, cs...)), Matrix)))
    @test sort([Ek(Permutation(flip) => 0); Ek(Permutation(flip) => 1)]) ≈ Ek()
    @test_throws ArgumentError constrain_space(H, Permutation(flip) => 0)
end

@testitem "Spin inversion sectors" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: Permutation, LocalMap
    N = 6
    @spins S 1 // 2
    T(j) = mod1(j + 1, N)
    flip = LocalMap(j -> 2S[j][:x])
    Sz = sum(S[j][:z] for j in 1:N)
    ham = sum(S[j][:]'S[mod1(j + 1, N)][:] for j in 1:N)
    E(cs...) = eigvals(Hermitian(representation(ham, hilbert_space(S, 1:N, (Sz => 0, Permutation(T) => 0, cs...)), Matrix)))
    @test sort([E(flip => 0); E(flip => 1)]) ≈ E()
    @test_throws ArgumentError representation(Sz, hilbert_space(S, 1:N, (Permutation(T) => 0, flip => 0)))
    @test_throws ArgumentError hilbert_space(S, 1:N, (Sz, flip => 0))
    @test_throws ArgumentError hilbert_space(S, 1:N, LocalMap(j -> S[j][:z]) => 0)
end
