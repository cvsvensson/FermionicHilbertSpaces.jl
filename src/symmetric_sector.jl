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
as_constraint(::Relabel) = throw(ArgumentError("A symmetry needs an eigenvalue, as in `Relabel(g) => 1`."))
hilbert_space(::Field, ::AbstractVector, r::Relabel) = as_constraint(r)
constrain_space(space, r::Relabel) = as_constraint(r)

# The fields are not type parameters, so that spaces with different maps share their compiled methods.
struct SymmetryConstraint <: AbstractConstraint
    maps::Tuple
    eigenvalues::Tuple
end
as_constraint((r, λ)::Pair{Relabel}) = SymmetryConstraint((r.map,), (λ,))

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
    N = length(first(generators))
    characters = Dict(collect(1:N) => one(eltype(λs)))
    queue = [collect(1:N)]
    while !isempty(queue)
        h = popfirst!(queue)
        for (g, λ) in zip(generators, λs)
            gh, λgh = h[g], characters[h] * λ
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

The sector `constrain_space(H, Relabel(g) => λ)` of a [`Relabel`](@ref), with one basis vector per orbit of the basis
states of `H`. Representing an operator throws an `ArgumentError` unless relabeling its modes by each map leaves it
unchanged. `embed(v, Hk => H)` gives a state of the sector on the parent space.
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
    generators = [_atom_permutation(g, atoms) for g in c.maps]
    roots = [_root_of_unity(λ, _order(g)) for (g, λ) in zip(generators, c.eigenvalues)]
    # The sector does not depend on how its eigenvalues were written.
    c = SymmetryConstraint(c.maps, Tuple(map(_canonical, roots)))
    perms, λs = _group_and_characters(generators, roots)
    T = all(isreal, λs) ? Float64 : ComplexF64
    # The permutation operator of `perm` is `U'` for the `U` that maps each product state `s` to `g(s)`, so the projector
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
interpret_state(state, ::SymmetricSector) = throw(ArgumentError("`$state` is not a basis state of a `SymmetricSector`. Represent it on the parent space instead."))
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
Base.show(io::IO, H::SymmetricSector) = print(io, "$(dim(H))-dimensional SymmetricSector(", join(("Relabel($g) => $(_rounded(λ))" for (g, λ) in zip(H.symmetry.maps, H.symmetry.eigenvalues)), ", "), ")")
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
_axes_error() = ArgumentError("`Axes(Hout, Hin)` with a symmetric sector needs two sectors of the same `SymmetricSector`. Use `embed` to move to its parent space.")

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
    for g in H.symmetry.maps
        iszero(_relabel(op, g) - op) || throw(ArgumentError("The operator changes under `Relabel($g)`, so it has no representation on the symmetric sector."))
    end
    return nothing
end
_relabel(op::Op, @nospecialize(g)) = Op(op.kind, _image(g, op.mode))
_relabel(m::NCMul, @nospecialize(g)) = m.coeff * prod(_relabel(x, g) for x in m.factors)
_relabel(a::NCAdd, @nospecialize(g)) = a.coeff + sum((c * _relabel(m, g) for (m, c) in a.dict); init = 0)
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


@testitem "Symmetric sectors of a spin ring" begin
    using LinearAlgebra, SparseArrays
    using FermionicHilbertSpaces: Relabel, Axes, change_basis, TransposedSpace, symmetric_sector
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
end
