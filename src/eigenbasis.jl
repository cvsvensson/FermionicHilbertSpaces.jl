"""
    EigenState

The basis state of an [`eigenbasis`](@ref) space: eigenstate number `index` of the space, in the sector `qn`.
"""
struct EigenState{Q} <: AbstractBasisState
    qn::Q
    index::Int
end
_eigen_qn(state::EigenState) = state.qn

# One atom whose basis states are the columns of `vectors`, given in the basis of `parent`. Operators act on it as on
# `parent`, and its support is that of `parent`.
struct EigenbasisSpace{Q, P, M, E, G} <: AbstractAtomicHilbertSpace{EigenState{Q}}
    parent::P
    vectors::M
    energies::Vector{E}
    qns::Vector{Q}
    key::UInt
    support::G
end
EigenbasisSpace(parent, vectors, energies, qns, key = hash(vectors)) = EigenbasisSpace(parent, vectors, energies, qns, key, _support(group_ids(parent)))

"""
    eigenbasis(ham, H)

The space spanned by the eigenstates of the Hermitian `ham`, a symbolic operator or its matrix on `H`, found sector by
sector of `H` and ordered by increasing energy within each sector. Give `H` the symmetries of `ham` as constraints:
degenerate eigenstates of different sectors are otherwise mixed. `change_basis(ρ, Heig => H)` returns to `H`.

```julia
Heig = eigenbasis(ham, hilbert_space(f, 1:2, NumberConservation()))
Hlr, Hl, Hr, left, right = open_system(Heig)
L = representation(-1im * (left(ham) - right(ham)) + dissipator(f[1]), Hlr)
```
"""
function eigenbasis(ham, H::AbstractHilbertSpace)
    M = ham isa AbstractMatrix ? ham : representation(ham, H)
    size(M) == (dim(H), dim(H)) || throw(DimensionMismatch("The matrix of `ham` must have the size $((dim(H), dim(H))), got $(size(M))."))
    qns = collect(quantumnumbers(H))
    blocks = [indices(qn, H) for qn in qns]
    _check_block_diagonal(M, blocks)
    decompositions = map(inds -> _hermitian_eigen(M[inds, inds]), blocks)
    S = SparseArrays.blockdiag(map(F -> sparse(F.vectors), decompositions)...)[invperm(reduce(vcat, blocks)), :]
    # Exact zeros between states that a symmetry of `ham` separates keep matrices on constrained subspaces exact
    SparseArrays.droptol!(S, _roundoff(S))
    Heig = EigenbasisSpace(H, S, reduce(vcat, [F.values for F in decompositions]), reduce(vcat, [fill(qn, length(inds)) for (qn, inds) in zip(qns, blocks)]))
    H isa SectorHilbertSpace || return Heig
    return sector_space(Heig, collect(basisstates(Heig)), _eigen_qn, SectorConstraint(_eigen_qn))
end
function _check_block_diagonal(M, blocks)
    block = zeros(Int, size(M, 1))
    foreach(((n, inds),) -> block[inds] .= n, enumerate(blocks))
    tol = _roundoff(M)
    for (i, j, v) in zip(findnz(sparse(M))...)
        block[i] == block[j] || abs(v) <= tol || throw(ArgumentError("The Hamiltonian couples the sectors of the space, so it has no eigenbasis sector by sector."))
    end
    return
end
function _hermitian_eigen(A)
    A = Matrix(A)
    norm(A - A') <= _roundoff(A) || throw(ArgumentError("The Hamiltonian must be Hermitian."))
    return LinearAlgebra.eigen(Hermitian(A))
end

"""
    energies(Heig)

The energies of the basis states of an [`eigenbasis`](@ref) space or of a sector of one. With `E = energies(Heig)`,
`A .* g.(E .- transpose(E))` weights each matrix element of `A` on `Heig` by its Bohr frequency. The populations alone
follow a rate equation only when no energies are degenerate.
"""
energies(H::EigenbasisSpace) = H.energies
energies(H::AbstractSubspace{<:EigenState, <:EigenbasisSpace}) = [parent(H).energies[state.index] for state in basisstates(H)]

Base.parent(H::EigenbasisSpace) = H.parent
dim(H::EigenbasisSpace) = length(H.energies)
basisstate(n::Integer, H::EigenbasisSpace) = EigenState(H.qns[n], Int(n))
basisstates(H::EigenbasisSpace) = BasisStates(H, Iterators.map(n -> basisstate(n, H), 1:dim(H)))
_state_index(state::EigenState, H::EigenbasisSpace) = checkbounds(Bool, H.qns, state.index) && isequal(H.qns[state.index], state.qn) ? state.index : 0
interpret_state(state::EigenState, ::EigenbasisSpace) = state
interpret_state(state, H::EigenbasisSpace) = throw(ArgumentError("`$state` is not a basis state of `$H`. Represent it on the space `H` the eigenbasis was built on and use `change_basis(v, H => Heig)`."))
has_arithmetic_state_index(::EigenbasisSpace) = true
atomic_id(H::EigenbasisSpace) = (:eigenbasis, H.key, atom_ids(H.parent)...)
group_id(H::EigenbasisSpace) = H.support
add_tag(H::EigenbasisSpace, tag) = EigenbasisSpace(add_tag(H.parent, tag), H.vectors, H.energies, H.qns, H.key)
# The transposed space has the complex conjugate basis, so that its matrices are the transposes.
TransposedSpace(H::EigenbasisSpace) = EigenbasisSpace(TransposedSpace(H.parent), conj(H.vectors), H.energies, H.qns, H.key)
Base.:(==)(H1::EigenbasisSpace, H2::EigenbasisSpace) = H1.parent == H2.parent && H1.key == H2.key
Base.hash(H::EigenbasisSpace, h::UInt) = hash((H.parent, H.key), h)
Base.show(io::IO, H::EigenbasisSpace) = print(io, "$(dim(H))-dimensional eigenbasis of ", H.parent)
_amplitude_eltype(H::EigenbasisSpace) = eltype(H.vectors)

_state_label_atomic(s::EigenState, H) = string(s.index)
particle_number(::EigenState) = throw(ArgumentError("Eigenstates have no particle number or parity, so `NumberConservation` and `ParityConservation` do not apply to an eigenbasis. Constrain an operator instead, such as `sum(f[k]'f[k] for k in 1:N) => n`."))

# Operators are rotated from the parent, and ketbras of the eigenbasis are matrix units.
_rotated(op, root; projection) = _basis_vectors(root)' * representation(op, parent(root); projection) * _basis_vectors(root)
_has_state(m::NCMul) = any(x -> x isa SymbolicState, m.factors)
function _eigen_matrix(op::NCMul, root::EigenbasisSpace; projection = false)
    matrices = []
    run = []
    for x in op.factors
        x isa SymbolicState || (push!(run, x); continue)
        isempty(run) || push!(matrices, _rotated(NCMul(1, run), root; projection))
        run = []
        push!(matrices, _ketbra_matrix(x, root))
    end
    isempty(run) || push!(matrices, _rotated(NCMul(1, run), root; projection))
    return op.coeff * prod(matrices)
end
_istransposed(root) = all(g -> g isa TransposedSpace, groups(parent(root)))
function _ketbra_matrix(s::SymbolicState, root)
    isketbra(s) && _same_space_id(s.space, root) || throw(ArgumentError("Only ketbras of the eigenbasis act on it, got `$s`."))
    return sparse([state_index(s.ket, root)], [state_index(s.bra, root)], [1], dim(root), dim(root))
end
function _term_matrix_representation(op::NCMul, H::EigenbasisSpace, repr::Union{EagerSparseRepr, EagerDenseRepr}, chunking; projection = false, kwargs...)
    M = _istransposed(H) ? transpose(_eigen_matrix(op, TransposedSpace(H); projection)) : _eigen_matrix(op, H; projection)
    return repr isa EagerDenseRepr ? Matrix(M) : M
end

_acts_by_matrix(::EigenbasisSpace) = true
# An operator maps an eigenstate to a superposition: a column of its matrix acting on a ket, a row acting on a bra.
# The product is left unevaluated, for the matrix representation, when a single operator can leave an eigenbasis that
# does not span all states of its groups, and on a bra of several groups, which operators of a group ordered between
# them cannot reach.
function _superposition(op, s::SymbolicState, onket::Bool)
    root = _product_parent(s.space)
    dim(root) == prod(dim, atomic_factors(parent(root))) || return nothing
    onket || length(_members(group_id(root))) == 1 || return nothing
    S, M = _basis_vectors(root), representation(op, parent(root))
    i = state_index(onket ? s.ket : s.bra, root)
    inds, amps = findnz(onket ? S' * (M * S[:, i]) : conj(S' * (M' * S[:, i])))
    terms = [_scaled_state(a, onket ? SymbolicState(s.space, basisstate(i, root), s.bra) : SymbolicState(s.space, s.ket, basisstate(i, root))) for (i, a) in zip(inds, amps) if !iszero(a)]
    return isempty(terms) ? 0 : NonCommutativeProducts.AddTerms(terms)
end
_apply_local_operators(op, ::EigenState, ::EigenbasisSpace, precomp) = throw(ArgumentError("An operator maps an eigenstate to a superposition, so it is applied through its matrix."))
# The fermionic partial transpose has phases that depend on the Fock states, so it is not a transpose in another basis.
partial_transpose_phase_factor(f1, f2, H::EigenbasisSpace) = partial_transpose_phase_factor_eltype(parent(H)) <: Real ? 1 :
    throw(ArgumentError("The fermionic partial transpose is defined in the Fock basis. Change the basis of `$H` to its parent space and take the partial transpose there."))

# Orthonormal columns in `parent(H)`, with exact zeros between states that a symmetry of `parent(H)` separates, so that
# operators that keep a sector have exact zeros outside it.
_basis_vectors(H::EigenbasisSpace) = H.vectors

@testitem "Eigenbasis" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: eigenbasis, energies, change_basis, embed, Axes, Kets, vector_representation, partial_transpose, constrain_space
    @fermions f
    @bosons b
    ham = sum(0.3k * f[k]'f[k] for k in 1:3) + sum(0.5 * f[k]'f[k + 1] + hc for k in 1:2) + 0.7 * f[1]'f[1] * f[2]'f[2]
    H = hilbert_space(f, 1:3, NumberConservation())
    Heig = eigenbasis(ham, H)
    M = representation(ham, H)
    Fs = [eigen(Hermitian(Matrix(M[indices(n, H), indices(n, H)]))) for n in quantumnumbers(H)]
    S = cat([F.vectors for F in Fs]...; dims = (1, 2))
    @test quantumnumbers(Heig) == quantumnumbers(H)
    @test energies(Heig) ≈ reduce(vcat, [F.values for F in Fs])
    for op in (ham, f[1], f[1]' * f[2]' * f[3] + 0.2im * f[2]'f[1])
        @test representation(op, Heig) ≈ S' * representation(op, H) * S
    end
    @test representation(f[1], Axes(sector(1, Heig), sector(2, Heig))) ≈ (S' * representation(f[1], H) * S)[indices(1, Heig), indices(2, Heig)]
    ψ = randn(ComplexF64, dim(H))
    @test change_basis(ψ, Heig => H) ≈ S * ψ

    # A factor of a tensor product with other groups
    H0 = hilbert_space(f, 1:2)
    ham0 = 0.3 * f[1]'f[1] + (0.5im * f[1]'f[2] + 0.4 * f[1]'f[2]' + hc)
    S0 = eigen(Hermitian(Matrix(representation(ham0, H0)))).vectors
    Hb = hilbert_space(b[1], 3)
    op = f[1]' * b[1] + hc + b[1]'b[1] * f[2]'f[2]
    U = kron(I(3), S0)
    @test representation(op, tensor_product(eigenbasis(ham0, H0), Hb)) ≈ U' * representation(op, tensor_product(H0, Hb)) * U

    @test_throws ArgumentError eigenbasis(ham0, H)
    @test_throws ArgumentError tensor_product(Heig, hilbert_space(f, 4:4))
    @test representation(f[1]' * f[2] + 0.5, Heig, :lazy) * ψ ≈ representation(f[1]' * f[2] + 0.5, Heig) * ψ
    ψ1 = randn(ComplexF64, dim(sector(1, Heig)))
    @test transpose(representation(f[1]' * f[2] + 0.5im * f[2]' * f[3], sector(1, Heig), :lazy)) * ψ1 ≈ transpose(representation(f[1]' * f[2] + 0.5im * f[2]' * f[3], sector(1, Heig))) * ψ1
    # A matrix on one eigenbasis is not a matrix on another eigenbasis of the same space
    other = eigenbasis(ham0 + 0.2 * f[2]'f[2], H0)
    @test_throws ArgumentError embed(representation(f[1], other), other => tensor_product(eigenbasis(ham0, H0), Hb))

    # An operator maps an eigenstate to the column of its matrix
    E0 = eigenbasis(ham0, H0)
    @test vector_representation(f[1]' * Kets(E0)[2], E0) ≈ representation(f[1]', E0)[:, 2]
    # Changing basis of a product with an eigenbasis factor, and of eigen sectors
    ρ = randn(ComplexF64, 12, 12)
    @test change_basis(ρ, tensor_product(E0, Hb) => tensor_product(H0, Hb)) ≈ U * ρ * U'
    @test change_basis(U * ρ * U', tensor_product(H0, Hb) => tensor_product(E0, Hb)) ≈ ρ
    ψ1 = randn(dim(sector(1, H)))
    @test change_basis(change_basis(ψ1, sector(1, H) => sector(1, Heig)), sector(1, Heig) => sector(1, H)) ≈ ψ1
    @test_throws ArgumentError change_basis(ψ1, sector(1, H) => sector(2, Heig))
    # Operator constraints on larger eigenbases, where the rotated diagonal carries more roundoff
    H6 = hilbert_space(f, 1:6, NumberConservation())
    E6 = eigenbasis(sum(0.5 * f[k]'f[k + 1] + hc for k in 1:5) + sum(0.1k * f[k]'f[k] for k in 1:6), H6)
    @test dim(constrain_space(E6, sum(f[k]'f[k] for k in 1:6) => 2)) == binomial(6, 2)
    # Constraints on an eigenbasis without sectors, also on products with another group
    @fermions g
    N3, Ng = sum(f[k]'f[k] for k in 1:3), g[1]'g[1]
    E3 = eigenbasis(ham, hilbert_space(f, 1:3))
    C = constrain_space(E3, N3 => 1)
    @test representation(ham, C) ≈ Diagonal(energies(Heig)[indices(1, Heig)])
    Hg = hilbert_space(g, 1:1)
    @test dim(tensor_product(E3, Hg; constraint = N3 * Ng => 0)) == dim(tensor_product(hilbert_space(f, 1:3), Hg; constraint = N3 * Ng => 0))
    Pc = tensor_product(E3, Hg; constraint = N3 + Ng => 2)
    op = f[1]' * g[1] + hc
    @test svdvals(Matrix(representation(op, Pc))) ≈ svdvals(Matrix(representation(op, tensor_product(hilbert_space(f, 1:3), Hg; constraint = N3 + Ng => 2))))
    # The fermionic partial transpose is defined in the Fock basis only
    @test_throws ArgumentError partial_transpose(Matrix(I, 3 * dim(E0), 3 * dim(E0)), tensor_product(E0, Hb), E0)

    # Ketbras of eigenstates are matrix units, also next to operators and in products with other groups
    k = Kets(Heig)
    E12 = zeros(dim(Heig), dim(Heig)); E12[1, 2] = 1
    @test representation(k[1] * k[2]', Heig) ≈ E12
    @test representation(k[1] * k[2]' * f[1]' * f[2] + 2 * f[3]'f[3], Heig) ≈ E12 * S' * representation(f[1]' * f[2], H) * S + 2 * S' * representation(f[3]'f[3], H) * S
    P = tensor_product(Heig, Hb)
    @test representation(k[1] * k[2]' * b[1], P) ≈ tensor_product((E12, representation(b[1], Hb)), (Heig, Hb) => P)
end

@testitem "Eigenbasis of several groups" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: eigenbasis, Kets
    @fermions f
    @fermions g
    @bosons b
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(b[1], 3))
    ham = 0.3 * f[1]'f[1] + 0.7 * f[2]'f[2] + (0.5 * f[1]'f[2] + hc) + 0.4 * b[1]'b[1] + 0.2 * (f[1]'f[1] - f[2]'f[2]) * (b[1] + b[1]')
    Heig = eigenbasis(ham, H)
    S = eigen(Hermitian(Matrix(representation(ham, H)))).vectors
    op = f[1]' * b[1] + 0.3im * f[2]' * f[1] * b[1]'b[1]
    @test representation(op, Heig) ≈ S' * representation(op, H) * S
    Hb = hilbert_space(b[2], 2)
    U = kron(I(2), S)
    op2 = op * b[2] + f[1]' * f[2] * b[2]'
    @test representation(op2, tensor_product(Heig, Hb)) ≈ U' * representation(op2, tensor_product(H, Hb)) * U
    # Spaces and states of some of the same groups cannot be combined with it
    @test_throws ArgumentError tensor_product(Heig, hilbert_space(f, 3:3))
    Hbg = eigenbasis(b[1]'b[1] + 0.3 * (g[1]' * b[1] + hc), tensor_product(hilbert_space(b[1], 3), hilbert_space(g, 1:1)))
    @test_throws ArgumentError Kets(Heig)[1] * Kets(Hbg)[1]
    @test_throws ArgumentError tensor_product(Heig, Heig)
    # Operators of any of its groups act on a ketbra of it
    E12 = zeros(dim(Heig), dim(Heig)); E12[1, 2] = 1
    @test representation(b[1] * Kets(Heig)[1] * Kets(Heig)[2]' * f[1], Heig) ≈ representation(b[1], Heig) * E12 * representation(f[1], Heig)
    s = Kets(Heig)[1] * Kets(Heig)[2]'
    @test iszero(s * g[1] * b[1] - s * b[1] * g[1])
    # An eigenbasis of a space that contains an eigenbasis
    H2 = tensor_product(Heig, hilbert_space(g, 1:1))
    E2 = eigenbasis(representation(ham + g[1]'g[1] + 0.2 * (g[1]' * f[1] + hc), H2, Matrix), H2)
    S2 = eigen(Hermitian(representation(ham + g[1]'g[1] + 0.2 * (g[1]' * f[1] + hc), H2, Matrix))).vectors
    @test representation(f[1], E2) ≈ S2' * representation(f[1], H2) * S2
end

@testitem "Lindbladian in an eigenbasis" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: eigenbasis, change_basis, open_system, dissipator, Kets
    @fermions f
    ham = sum(0.3k * f[k]'f[k] for k in 1:3) + sum(0.5 * f[k]'f[k + 1] + hc for k in 1:2) + 0.7 * f[1]'f[1] * f[2]'f[2]
    Ntot = sum(f[k]'f[k] for k in 1:3)
    H = hilbert_space(f, 1:3, NumberConservation())
    Heig = eigenbasis(ham, H)
    jumps = [0.4 * f[1], 0.3 * f[3]']
    lindbladian(left, right) = -1im * (left(ham) - right(ham)) + sum(dissipator, jumps)
    spectrum(L) = sort(eigvals(Matrix(L)); by = x -> round.(reim(x); digits = 8))
    function steady_state(L, Hlr, Hl, Hr)
        ρ = reshape(nullspace(Matrix(L))[:, 1], Hlr => (Hl, Hr))
        return ρ / tr(ρ)
    end
    Hlr, Hl, Hr, left, right = open_system(H)
    Elr, El, Er = open_system(Heig)
    for (Hd, Ed) in ((Hlr, Elr), (tensor_product((Hl, Hr); constraint = left(Ntot) - right(Ntot) => 0), tensor_product((El, Er); constraint = left(Ntot) - right(Ntot) => 0)))
        L, Le = representation(lindbladian(left, right), Hd), representation(lindbladian(left, right), Ed)
        @test spectrum(Le) ≈ spectrum(L)
        @test svdvals(Matrix(representation(left(f[1]) * right(f[1]'), Ed))) ≈ svdvals(Matrix(representation(left(f[1]) * right(f[1]'), Hd)))
        @test change_basis(steady_state(Le, Ed, El, Er), Heig => H) ≈ steady_state(L, Hd, Hl, Hr)
    end

    # From matrices on the eigenbasis
    h, c = representation(ham, Heig), representation(jumps[1], Heig)
    sandwich(A, B) = tensor_product((A, transpose(B)), (El, Er) => Elr)
    L = -1im * (embed(h, El => Elr) - embed(transpose(h), Er => Elr)) + sandwich(c, c') - (embed(c'c, El => Elr) + embed(transpose(c'c), Er => Elr)) / 2
    @test L ≈ representation(-1im * (left(ham) - right(ham)) + dissipator(jumps[1]), Elr)
    # A jump operator written with ketbras of eigenstates
    k = Kets(Heig)
    J = k[2] * k[1]' + 0.5im * k[3] * k[2]'
    c = representation(J, Heig)
    @test representation(dissipator(J), Elr) ≈ sandwich(c, c') - (embed(c'c, El => Elr) + embed(transpose(c'c), Er => Elr)) / 2

    # The populations: the Pauli rate matrix with rates |⟨a|J|b⟩|² from b to a
    populations = tensor_product((El, Er); constraint = s -> s.states[1].index == s.states[2].index)
    R = sum(J -> abs2.(representation(J, Heig, Matrix)), jumps)
    @test representation(lindbladian(left, right), populations; projection = true) ≈ R - Diagonal(vec(sum(R; dims = 1)))
end
