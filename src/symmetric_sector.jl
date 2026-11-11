"""
    Relabel(g)

The symmetry that moves the mode with label `l` to the mode with label `g(l)`. The constraint `Relabel(g) => λ` keeps
the states whose amplitudes obey `ψ(g(s)) == λ * ψ(s)`, where `g(s)` is the product state `s` with every mode moved,
with the sign of reordering fermions; `λ` is the eigenvalue of `U'` for the `U` that maps each `s` to `g(s)`. It needs
`λ^n ≈ 1` for the order `n` of the permutation of the modes, several pairs keep their common eigenstates, and the
result is a [`SymmetricSector`](@ref).

```julia
T(j) = mod1(j + 1, N)
Hk = hilbert_space(S, 1:N, (Sz, Relabel(T) => cispi(2k / N)))   # momentum k: amplitude cispi(2k j / N) on site j
```
"""
struct Relabel
    map
end
Base.show(io::IO, r::Relabel) = print(io, "Relabel(", r.map, ")")

"""
    LocalMap(f)

The symmetry that acts on site `l` of each product state with `f(l)`, a single-site operator or its matrix, which must
permute the basis states of the site, as `l -> 2S[l][:x]` does for spin 1/2. `LocalMap(f) => λ` keeps the states with
`ψ(f(s)) == λ * ψ(s)`, like [`Relabel`](@ref), and composes with it: `Relabel(g) ∘ LocalMap(f) => λ` is the map that
applies the local map to `s` and then moves the modes.
"""
struct LocalMap
    op
end
Base.show(io::IO, l::LocalMap) = print(io, "LocalMap(", l.op, ")")

# The fields are not type parameters, so that spaces with different maps share their compiled methods.
struct SymmetryConstraint <: AbstractConstraint
    maps::Tuple
    eigenvalues::Tuple
end
_leaves(g) = (g,)
_leaves(c::ComposedFunction) = (_leaves(c.outer)..., _leaves(c.inner)...)
# Whether `g` is a composition of `Relabel`s and `LocalMap`s. Other compositions of functions are quantities.
function _is_symmetry(g)
    ours = map(l -> l isa Union{Relabel, LocalMap}, _leaves(g))
    any(ours) && !all(ours) && throw(ArgumentError("A symmetry is a composition of `Relabel`s and `LocalMap`s, got `$g`."))
    return all(ours)
end
const _MaybeSymmetry = Union{Relabel, LocalMap, ComposedFunction}
hilbert_space(::Field, ::AbstractVector, g::Union{Relabel, LocalMap}) = as_constraint(g)
constrain_space(space, g::Union{Relabel, LocalMap}) = as_constraint(g)
as_constraint(g::_MaybeSymmetry) = _is_symmetry(g) ? throw(ArgumentError("A symmetry needs an eigenvalue, as in `Relabel(g) => 1`.")) : invoke(as_constraint, Tuple{Function}, g)
as_constraint(p::Pair{<:_MaybeSymmetry}) = _is_symmetry(first(p)) ? SymmetryConstraint((first(p),), (last(p),)) : invoke(as_constraint, Tuple{Pair}, p)

# A group element `(perm, locs)` maps a product state to the one whose factor `i` is the image of the old factor
# `perm[i]` under `locs[i]`, a permutation of the basis states of site `i`. The element of a symmetry is the inverse of
# the `U` that maps each product state to its image, so a composition has the elements of its parts in reverse order.
_element(r::Relabel, space, atoms) = (_atom_permutation(r.map, atoms), [collect(1:dim(atom)) for atom in atoms])
function _element(l::LocalMap, space, atoms)
    phase_factor_u(state_mapper(space, atoms)) == Returns(1) || throw(ArgumentError("A `LocalMap` does not support fermions."))
    locs = map(atoms) do atom
        x = l.op(atomic_id(atom).label)
        M = x isa AbstractMatrix ? x : representation(x, atom, Matrix)
        d = dim(atom)
        size(M) == (d, d) || throw(ArgumentError("`$l` gives a matrix of size $(size(M)) for `$atom`, which needs size ($d, $d)."))
        loc = [argmax(abs.(c)) for c in eachcol(M)]
        isperm(loc) && M ≈ (axes(M, 1) .== permutedims(loc)) || throw(ArgumentError("`$l` does not permute the basis states of `$atom`."))
        invperm(loc)
    end
    return (collect(eachindex(atoms)), locs)
end
_element(c::ComposedFunction, space, atoms) = _compose(_element(c.inner, space, atoms), _element(c.outer, space, atoms))
_compose((p, ls), (q, ms)) = (q[p], [l[ms[i]] for (l, i) in zip(ls, p)])
function _order(g, e)
    n, h = 1, g
    while h != e
        n, h = n + 1, _compose(g, h)
    end
    return n
end

struct SiteMap{P, A}
    perm::P
    locs::Vector{Vector{Int}}
    atoms::A
end
_match_perm(g::SiteMap, substates::NTuple{N, Any}) where {N} = SiteMap(_match_perm(g.perm, substates), g.locs, g.atoms)
permute_state(state, mapper, g::SiteMap) = permute_state(state, mapper, g.perm, (i, s) -> basisstate(g.locs[i][state_index(s, g.atoms[i])], g.atoms[i]))

_image(@nospecialize(g), m::Mode) = Mode(m.field, g(m.label))

# The permutation of `atoms` that `g` induces through the labels of their modes
function _atom_permutation(@nospecialize(g), atoms)
    modes = map(atomic_id, atoms)
    position = Dict(m => n for (n, m) in enumerate(modes))
    perm = map(modes) do m
        m isa Mode || throw(ArgumentError("A relabeling moves single modes, but the factor `$m` is not one. The pairs of a Majorana space cannot be relabeled."))
        image = _image(g, m)
        get(() -> throw(ArgumentError("`$g` maps `$m` to `$image`, which is not in the space.")), position, image)
    end
    isperm(perm) || throw(ArgumentError("`$g` does not permute the modes of the space."))
    return perm
end

# `λ` as an exact `cispi`, so that real eigenvalues give real characters
function _root_of_unity(λ, n)
    λ isa Number && ComplexF64(λ)^n ≈ 1 || throw(ArgumentError("The eigenvalue of a symmetry of order $n is a root of unity with `λ^$n ≈ 1`, got `$λ`."))
    return cispi(2 * round(Int, n * angle(ComplexF64(λ)) / 2π) / n)
end

_canonical(λ) = all(isinteger, reim(λ)) ? (isreal(λ) ? Int(real(λ)) : Complex{Int}(λ)) : λ
_rounded(λ) = λ isa Complex{<:AbstractFloat} ? round(λ; digits = 4) : λ

# All elements of the group with identity `e` and their characters. A word in the generators has the product of their
# characters, so two words for one element must agree.
function _group_and_characters(e, generators, λs)
    characters = Dict(e => one(eltype(λs)))
    queue = [e]
    while !isempty(queue)
        h = popfirst!(queue)
        for (g, λ) in zip(generators, λs)
            gh, λgh = _compose(g, h), characters[h] * λ
            if haskey(characters, gh)
                characters[gh] ≈ λgh || throw(ArgumentError("The eigenvalues are not consistent with the group the symmetries generate."))
            else
                characters[gh] = λgh
                push!(queue, gh)
            end
        end
    end
    return collect(keys(characters)), collect(values(characters))
end

# A constraint with a symmetry is applied as the rest of the constraint followed by the symmetry.
_split_symmetry(c) = (c, nothing)
_split_symmetry(c::SymmetryConstraint) = (NoSymmetry(), c)
function _split_symmetry(c::ProductConstraint)
    cs = collect(c.constraints)
    syms, rest = filter(x -> x isa SymmetryConstraint, cs), filter(x -> !(x isa SymmetryConstraint), cs)
    isempty(syms) && return (c, nothing)
    joined = SymmetryConstraint(Tuple(g for s in syms for g in s.maps), Tuple(λ for s in syms for λ in s.eigenvalues))
    return (isempty(rest) ? NoSymmetry() : as_constraint(rest)), joined
end

struct OrbitRep{B} <: AbstractBasisState
    state::B
    index::Int
end

"""
    SymmetricSector

The sector `constrain_space(H, Relabel(g) => λ)` of a [`Relabel`](@ref) or [`LocalMap`](@ref), with one basis vector
per orbit of the basis states of `H`. Representing an operator throws an `ArgumentError` unless each symmetry leaves it
unchanged. Move states with `embed(v, Hk => H)` and `change_basis(v, H => Hk; projection = true)`.
"""
struct SymmetricSector{B, P, T, M <: SparseMatrixCSC{T}} <: AbstractHilbertSpace{OrbitRep{B}}
    parent::P
    symmetry::SymmetryConstraint
    # `amplitudes[c, t]` is the amplitude of parent basis state `t` in basis vector `c`, with one entry per column.
    amplitudes::M
    reps::Vector{B}
    norms::Vector{Float64}
    transposed::Bool
end

function constrain_space(space::AbstractHilbertSpace, c::SymmetryConstraint)
    atoms = collect(atomic_factors(space))
    e = _element(Relabel(identity), space, atoms)
    generators = [_element(g, space, atoms) for g in c.maps]
    roots = [_root_of_unity(λ, _order(g, e)) for (g, λ) in zip(generators, c.eigenvalues)]
    # The sector does not depend on how its eigenvalues were written.
    c = SymmetryConstraint(c.maps, Tuple(map(_canonical, roots)))
    elements, λs = _group_and_characters(e, generators, roots)
    # Without local maps the elements are plain permutations, which are faster to apply and carry the fermionic signs.
    perms = all(g -> last(g) == last(e), elements) ? first.(elements) : [SiteMap(perm, locs, atoms) for (perm, locs) in elements]
    T = all(isreal, λs) ? Float64 : ComplexF64
    # The elements are the inverses `U'` of the `U` that map each product state `s` to its image `g(s)`, so the projector
    # with the weights `conj(λ)` keeps the states with `U' * ψ == λ * ψ`, which is `ψ(g(s)) == λ * ψ(s)`.
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
_no_factors_error(H) = ArgumentError("The basis states of `$H` are orbits of a symmetry, not product states, so it has no factors. Use its parent space.")
Base.:(==)(H1::SymmetricSector, H2::SymmetricSector) = H1 === H2 || (H1.parent == H2.parent && H1.symmetry == H2.symmetry && H1.transposed == H2.transposed)
Base.hash(H::SymmetricSector, h::UInt) = hash((H.parent, H.symmetry, H.transposed), h)
Base.show(io::IO, H::SymmetricSector) = print(io, "$(dim(H))-dimensional SymmetricSector(", join(("$g => $(_rounded(λ))" for (g, λ) in zip(H.symmetry.maps, H.symmetry.eigenvalues)), ", "), ")")
# The basis of the transposed space is the complex conjugate one.
TransposedSpace(H::SymmetricSector) = SymmetricSector(TransposedSpace(H.parent), H.symmetry, H.amplitudes, H.reps, H.norms, !H.transposed)

# A symmetric sector, or a subspace of one, with the orbit indices of its basis states
const _OrbitSpace = Union{SymmetricSector, AbstractSubspace{<:OrbitRep, <:SymmetricSector}}
_orbits(H::SymmetricSector) = (H, 1:dim(H))
_orbits(H::AbstractSubspace{<:OrbitRep, <:SymmetricSector}) = (parent(H), [state.index for state in basisstates(H)])

_amplitude_eltype(H::SymmetricSector) = eltype(H.amplitudes)
_precomputation_before_operator_application(op, H::_OrbitSpace) = _precomputation_before_operator_application(op, parent(first(_orbits(H))))
_check_operator(op, H::SymmetricSector) = _check_invariant(op, H)
# The output index of a basis state is its orbit in the input space, so both spaces must share the orbits.
function _check_operator(op, ax::Axes{<:Tuple{_OrbitSpace, _OrbitSpace}})
    Hout, Hin = ax.spaces
    first(_orbits(Hout)) == first(_orbits(Hin)) || throw(_axes_error())
    return _check_operator(op, Hin)
end
_check_operator(op, ::Axes{<:Tuple{_OrbitSpace, AbstractHilbertSpace}}) = throw(_axes_error())
_check_operator(op, ::Axes{<:Tuple{AbstractHilbertSpace, _OrbitSpace}}) = throw(_axes_error())
_axes_error() = ArgumentError("`Axes(Hout, Hin)` with a symmetric sector needs two sectors of the same `SymmetricSector`. Use `embed` and `change_basis` to move between it and its parent space.")

# For an operator that commutes with the symmetries, ⟨c′|op|c⟩ = Σ_t conj(q_c′(t)) ⟨t|op|r⟩ / q_c(r), where `r`
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

function _check_invariant(op, H::SymmetricSector)
    trivial_operator(op) && return nothing
    # An operator is invariant exactly when its transpose is, so a transposed sector is checked on the space itself.
    space = H.transposed ? TransposedSpace(H.parent) : H.parent
    return foreach(g -> _check_invariant(op, space, g), H.symmetry.maps)
end
# With `U_perm * U_loc` the element of `g`, where `U_perm` permutes the sites and `U_loc` acts within them, `op` is
# invariant when `U_perm' * op * U_perm == U_loc * op * U_loc'`. The left side is `op` with relabeled modes, which
# keeps the signs of reordering fermions. With local maps, the two sides are compared through their expectation values
# in random product states `u`, which vanish in all of them only for the zero operator. The right side has that of
# `op` in `U_loc' * u`.
function _check_invariant(op, space, g)
    err = ArgumentError("The operator changes under `$g`, so it has no representation on the symmetric sector.")
    relabeled = foldr(_permute, _leaves(g); init = op)
    any(l -> l isa LocalMap, _leaves(g)) || return iszero(relabeled - op) || throw(err)
    atoms = collect(atomic_factors(space))
    perm, locs = _element(g, space, atoms)
    left, right = _site_products(relabeled, atoms), _site_products(op, atoms)
    rng = Xoshiro(1)
    for _ in 1:3
        u = [normalize(randn(rng, ComplexF64, dim(atom))) for atom in atoms]
        v = copy(u)
        foreach((n, loc) -> v[n] = u[n][loc], perm, locs)
        x, y = _expectation_values(left, u), _expectation_values(right, v)
        abs(sum(x) - sum(y)) <= 1.0e-12 * sum(abs, y) || throw(err)
    end
    return nothing
end
_permute(r::Relabel, op) = _relabel(op, r.map)
_permute(::LocalMap, op) = op
_relabel(op::Op, @nospecialize(g)) = Op(op.kind, _image(g, op.mode))
_relabel(m::NCMul, @nospecialize(g)) = m.coeff * prod(_relabel(x, g) for x in m.factors)
_relabel(a::NCAdd, @nospecialize(g)) = a.coeff + sum((c * _relabel(m, g) for (m, c) in a.dict); init = 0)
_relabel(op, g) = throw(ArgumentError("Only operators on fields can be checked for symmetry, got `$op`."))

# The terms of `op` without its constant, each as its coefficient and the matrices of its factors multiplied site by site
function _site_products(op, atoms)
    position = Dict(atomic_id(atom) => n for (n, atom) in enumerate(atoms))
    matrices = Dict{Any, Matrix{ComplexF64}}()
    return map(first(_operator_terms(op))) do term
        term.coeff isa Union{_NumericWeight, Complex{<:_NumericWeight}} || throw(ArgumentError("Checking an operator against a `LocalMap` needs numeric coefficients, got `$(term.coeff)`."))
        sites = Pair{Int, Matrix{ComplexF64}}[]
        for o in term.factors
            n = get(() -> throw(ArgumentError("Only operators on the modes of the space can be checked for symmetry, got `$o`.")), position, atomic_id(o))
            A = get!(() -> representation(o, atoms[n], Matrix), matrices, o)
            k = findfirst(==(n) ∘ first, sites)
            isnothing(k) ? push!(sites, n => A) : (sites[k] = n => last(sites[k]) * A)
        end
        ComplexF64(term.coeff) => sites
    end
end
_expectation_values(terms, u) = ComplexF64[c * prod(dot(u[n], A, u[n]) for (n, A) in sites) for (c, sites) in terms]

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
_check_projection(Hfrom::AbstractHilbertSpace, ::_OrbitSpace) = Hfrom isa _OrbitSpace || throw(ArgumentError("A state of `$Hfrom` is in general not in the symmetric sector. Pass `projection = true` to project it."))


@testitem "Symmetric sectors of a spin ring" begin
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: Relabel, Axes, change_basis, TransposedSpace
    N = 6
    @spins S 1 // 2
    Sz = sum(S[j][:z] for j in 1:N)
    Splus = sum(S[j][:+] for j in 1:N)
    ham = sum(S[j][:]'S[mod1(j + 1, N)][:] for j in 1:N)
    space = hilbert_space(S, 1:N, Sz)
    perms = [[mod1(i + s, N) for i in 1:N] for s in 0:(N - 1)]
    T(j) = mod1(j + 1, N)
    R(j) = N + 1 - j
    Hks = [hilbert_space(S, 1:N, (Sz, Relabel(T) => cispi(2k / N))) for k in 0:(N - 1)]

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

    # ψ(T(s)) == λ ψ(s): a single excitation has amplitude cispi(2k j / N) on site j
    magnon = embed([1], sector(N // 2 - 1, Hk) => space)
    amplitudes = [magnon[state_index(join(i == j ? "↓" : "↑" for i in 1:N), space)] for j in 1:N]
    @test amplitudes ≈ amplitudes[1] * [cispi(2k * j / N) for j in 0:(N - 1)]

    @test change_basis(embed(v, H0 => space), space => H0; projection = true) ≈ v
    @test_throws ArgumentError change_basis(embed(v, H0 => space), space => H0)
    @test_throws ArgumentError representation(S[1][:z], H0)
    @test_throws ArgumentError representation(S[1][:z], TransposedSpace(H0))
    @test_throws ArgumentError representation("↑↑↑↓↓↓", H0)
    @test_throws ArgumentError partial_trace(Matrix(I, dim(H0), dim(H0)), H0 => hilbert_space(S[1]))
    @test_throws ArgumentError tensor_product(parent(H0), hilbert_space(S[1]))
    @test_throws ArgumentError representation(Sz, Axes(sector(0, Hks[3]), H0))
    @test_throws ArgumentError representation(Sz, Axes(sector(0, space), H0))
    @test_throws ArgumentError hilbert_space(S, 1:N, Relabel(j -> j + 1) => 1)
    @test_throws ArgumentError hilbert_space(S, 1:N, Relabel(T) => im)
    @test_throws ArgumentError hilbert_space(S, 1:N, Relabel(R) => typemin(Int) + 1)
    @test_throws "needs an eigenvalue" hilbert_space(S, 1:N, Relabel(T))
    @test hilbert_space(S, 1:N, Relabel(T) => 1.0) == hilbert_space(S, 1:N, Relabel(T) => exp(0im))
    @test_throws ArgumentError constrain_space(space, Relabel(T) => 1, basisstates(space))
    @test sum(k -> dim(hilbert_space(S, 1:N, Relabel(T) => cispi(2k / N))), 0:(N - 1)) == 2^N
    @test size(representation(Sz, hilbert_space(S, 1:N, (Sz => N // 2, Relabel(T) => -1)))) == (0, 0)

    # two sectors with the same representatives are orthogonal
    Ha, Hb = hilbert_space(S, 1:N, Relabel(T) => 1), hilbert_space(S, 1:N, Relabel(T) => -1)
    @test_throws ArgumentError representation(2 * one(ham), Axes(Ha, Hb))
    @test representation(2 * one(ham), Axes(Ha, Ha)) == 2I

    E0(cs...) = eigvals(Hermitian(representation(ham, sector(0, hilbert_space(S, 1:N, (Sz, cs...))), Matrix)))
    @test sort([E0(Relabel(T) => 1, Relabel(R) => 1); E0(Relabel(T) => 1, Relabel(R) => -1)]) ≈ E0(Relabel(T) => 1)
    @test_throws ArgumentError hilbert_space(S, 1:N, (Sz, Relabel(T) => cispi(1 / 3), Relabel(R) => 1))
end

@testitem "Symmetric sectors of a Hubbard ring" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: Relabel
    N = 4
    T((i, σ)) = (mod1(i + 1, N), σ)
    flip((i, σ)) = (i, σ == :↑ ? :↓ : :↑)
    labels = [(i, σ) for σ in (:↑, :↓) for i in 1:N]
    @fermions f
    ham = sum(-f[i, σ]' * f[mod1(i + 1, N), σ] + hc for σ in (:↑, :↓) for i in 1:N) + 4 * sum(f[i, :↑]'f[i, :↑] * f[i, :↓]'f[i, :↓] for i in 1:N)
    Nσ(σ) = sum(f[i, σ]'f[i, σ] for i in 1:N)
    H = hilbert_space(f, labels, (Nσ(:↑), Nσ(:↓)))
    Hks = [constrain_space(H, Relabel(T) => im^k) for k in 0:(N - 1)]
    for q in ((2, 1), (2, 2))
        @test sort(reduce(vcat, [eigvals(Hermitian(representation(ham, sector(q, Hk), Matrix))) for Hk in Hks])) ≈
            eigvals(Hermitian(representation(ham, sector(q, H), Matrix)))
    end
    @test_throws ArgumentError representation(f[1, :↑]'f[1, :↑], sector((2, 1), Hks[1]))
    @test eltype(representation(Float32(1) * ham, Hks[3])) == Float32
    @test eltype(representation(Float32(1) * ham, Hks[2])) == ComplexF32

    Hn = hilbert_space(f, labels, NumberConservation(3))
    Ek(cs...) = eigvals(Hermitian(representation(ham, constrain_space(Hn, (Relabel(T) => 1, cs...)), Matrix)))
    @test sort([Ek(Relabel(flip) => 1); Ek(Relabel(flip) => -1)]) ≈ Ek()
    @test_throws ArgumentError constrain_space(H, Relabel(flip) => 1)

    # ψ(T(s)) == λ ψ(s) for three fermions, where moving the modes of `s` gives `T(s)` with its sign
    Hf = hilbert_space(f, labels)
    c(l) = representation(f[l...]', Hf)
    vacuum = representation(prod(f[l...] * f[l...]' for l in labels), Hf) * ones(dim(Hf))
    s = [(3, :↑), (4, :↑), (1, :↓)]
    Hi = constrain_space(Hf, Relabel(T) => im)
    ψ = embed(randn(dim(Hi)), Hi => Hf)
    @test dot(prod(c ∘ T, s) * vacuum, ψ) ≈ im * dot(prod(c, s) * vacuum, ψ)

    # A composition of relabelings keeps the fermionic signs
    H2(λ) = hilbert_space(f, 1:2, Relabel(j -> 3 - j) ∘ Relabel(identity) => λ)
    hop2 = f[1]'f[2] + hc
    @test sort(reduce(vcat, [eigvals(Hermitian(representation(hop2, H2(λ), Matrix))) for λ in (1, -1)])) ≈ [-1, 0, 0, 1]
    @test_throws ArgumentError representation(f[1] * f[2] + hc, H2(1))
end

@testitem "Spin inversion sectors" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: Relabel, LocalMap, permutation_operator
    N = 6
    @spins S 1 // 2
    T(j) = mod1(j + 1, N)
    flip = LocalMap(j -> 2S[j][:x])
    Sz = sum(S[j][:z] for j in 1:N)
    ham = sum(S[j][:]'S[mod1(j + 1, N)][:] for j in 1:N)
    E(op, H) = eigvals(Hermitian(representation(op, H, Matrix)))
    H(cs...) = hilbert_space(S, 1:N, (Sz => 0, cs...))
    @test sort([E(ham, H(Relabel(T) => 1, flip => 1)); E(ham, H(Relabel(T) => 1, flip => -1))]) ≈ E(ham, H(Relabel(T) => 1))
    @test_throws ArgumentError representation(Sz, hilbert_space(S, 1:N, (Relabel(T) => 1, flip => 1)))
    @test_throws ArgumentError hilbert_space(S, 1:N, (Sz, flip => 1))
    @test_throws ArgumentError hilbert_space(S, 1:N, LocalMap(j -> S[j][:z]) => 1)
    @test_throws ArgumentError hilbert_space(S, 1:N, LocalMap(j -> [0 1 0; 1 0 0; 0 0 1]) => 1)
    @test_throws ArgumentError representation(1.0e9 + S[1][:z], hilbert_space(S, 1:N, flip => 1))
    @test E(ham, H(flip ∘ (Relabel(T) ∘ flip) => -1)) ≈ E(ham, H(Relabel(T) => -1))
    @test_throws ArgumentError H(sin ∘ Relabel(T) => 1)
    @test_throws "needs an eigenvalue" hilbert_space(S, 1:N, flip)

    # A staggered field is kept by neither the translation nor the inversion, but by their product
    staggered = ham + sum((-1)^j * S[j][:z] for j in 1:N)
    @test sort(reduce(vcat, [E(staggered, H(Relabel(T) ∘ flip => cispi(2k / N))) for k in 0:(N - 1)])) ≈ E(staggered, H())
    @test_throws ArgumentError E(staggered, H(Relabel(T) => 1))

    # The two orders of a translation and the inversion of the even sites are different maps `U` of the product states
    T4(j) = mod1(j + 1, 4)
    fliphalf = LocalMap(j -> iseven(j) ? 2S[j][:x] : 1)
    ring(c) = sum(S[j][:x] * S[T4(j)][:x] + c[j] * S[j][:z] for j in 1:4)
    a, b = (1, 1, -1, -1), (1, -1, -1, 1)
    H4 = hilbert_space(S, 1:4)
    UT = permutation_operator(H4, factors(H4), invperm(T4.(1:4)))
    Uf = representation(prod(2S[j][:x] for j in 2:2:4), H4)
    for (g, U, kept, changed) in ((fliphalf ∘ Relabel(T4), Uf * UT, b, a), (Relabel(T4) ∘ fliphalf, UT * Uf, a, b))
        @test sort(reduce(vcat, [E(ring(kept), hilbert_space(S, 1:4, g => im^k)) for k in 0:3])) ≈ E(ring(kept), H4)
        @test_throws ArgumentError E(ring(changed), hilbert_space(S, 1:4, g => 1))
        Hi = hilbert_space(S, 1:4, g => im)
        ψ = embed(randn(dim(Hi)), Hi => H4)
        @test U' * ψ ≈ im * ψ
    end

    # A product over many sites
    H16 = hilbert_space(S, 1:16, (sum(S[j][:z] for j in 1:16) => 0, flip => 1))
    @test representation(prod(2S[j][:z] for j in 1:16), H16) ≈ I

    # Spin 1 with m -> -m given as a matrix
    @spins L 1
    flip1 = LocalMap(j -> [0 0 1; 0 1 0; 1 0 0])
    ham1 = sum(L[j][:]'L[mod1(j + 1, 4)][:] + L[j][:z]^2 for j in 1:4)
    @test sort([E(ham1, hilbert_space(L, 1:4, flip1 => 1)); E(ham1, hilbert_space(L, 1:4, flip1 => -1))]) ≈ E(ham1, hilbert_space(L, 1:4))

    # A local map of order three
    C = [0 0 1; 1 0 0; 0 1 0]
    L2 = hilbert_space(L, 1:2)
    Hc = hilbert_space(L, 1:2, LocalMap(j -> C) => cispi(2 / 3))
    ψ = embed(randn(dim(Hc)), Hc => L2)
    @test kron(C, C)' * ψ ≈ cispi(2 / 3) * ψ
end
