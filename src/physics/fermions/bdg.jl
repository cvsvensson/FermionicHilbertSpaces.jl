struct NambuState <: AbstractFockState
    state::SingleParticleState
    hole::Bool
end
NambuState(i::Integer, hole::Bool) = NambuState(SingleParticleState(i), hole)
Base.:(==)(n1::NambuState, n2::NambuState) = n1.state == n2.state && n1.hole == n2.hole
Base.hash(n::NambuState, h::UInt) = hash(n.hole, hash(n.state, h))
_state_label_atomic(s::NambuState, H::_FermionModeSpace) = (s.hole ? "h" : "p") * _state_label_atomic(s.state, H)

# Ψ' m Ψ == Ψ' m_ph Ψ + tr(m) / 2 by the anticommutation relations, since the vector of adjoints Ψ' equals τx Ψ, and
# m_ph = -τx transpose(m_ph) τx is the unique particle-hole symmetric representative.
function normal_order_to_bdg(m::AbstractMatrix)
    n = size(m, 1) ÷ 2
    swap = [(n + 1):2n; 1:n]
    return (m - permutedims(m)[swap, swap]) / 2
end

struct BdGHilbertSpace{H}
    parent::H
    function BdGHilbertSpace(f::SymbolicFermionBasis, labels)
        states = vec([NambuState(i, hole) for (i, label) in enumerate(labels), hole in (true, false)])
        H = hilbert_space(f, labels, states)
        return new{typeof(H)}(H)
    end
end
function Base.show(io::IO, H::BdGHilbertSpace)
    return if get(io, :compact, false)
        print(io, "BdGHilbertSpace($(dim(H))-dim, $(div(dim(H), 2)) modes)")
    else
        print(io, "$(dim(H))-dimensional BdGHilbertSpace\n")
        print(io, "Physical modes: ", div(dim(H), 2), "\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
    end
end
"""
    BdG()

Marker passed as the last argument of `hilbert_space(f, labels, BdG())` to select the Bogoliubov-de Gennes (Nambu) space of non-interacting fermions with superconducting pairing.
"""
struct BdG end

"""
    hilbert_space(f::SymbolicFermionBasis, labels, ::BdG)

The space of Nambu states `[f[l] for l in labels; f[l]' for l in labels]`. `representation` gives the particle-hole symmetric BdG matrix, `[h Δ; -Δ* -h*] / 2` for `op = Σᵢⱼ h[i, j] f[i]' f[j] + (Δ[i, j] f[i]' f[j]' + h.c.) / 2`, and only quadratic operators can be represented.

```julia
H = hilbert_space(f, 1:2, BdG())
representation(f[1]' * f[1], H, :dense) # diagm([1, 0, -1, 0]) / 2
```
"""
hilbert_space(f::SymbolicFermionBasis, labels, ::BdG) = BdGHilbertSpace(f, labels)
dim(h::BdGHilbertSpace) = dim(h.parent)
mode_ordering(h::BdGHilbertSpace) = mode_ordering(h.parent)
Base.:(==)(a::BdGHilbertSpace, b::BdGHilbertSpace) = a === b || a.parent == b.parent
Base.hash(x::BdGHilbertSpace, h::UInt) = hash(x.parent, h)
modes(H::BdGHilbertSpace) = modes(H.parent)
basisstates(h::BdGHilbertSpace) = basisstates(h.parent)
Base.parent(h::BdGHilbertSpace) = h.parent
Base.keys(h::BdGHilbertSpace) = keys(h.parent)
Base.@propagate_inbounds basisstate(ind::Integer, H::BdGHilbertSpace) = basisstate(ind, parent(H))
_state_index(state, H::BdGHilbertSpace) = _state_index(state, parent(H))


representation(op, H::BdGHilbertSpace, repr=EagerSparseRepr(); chunking=NoChunking(), kwargs...) = matrix_representation(op, H, repr; chunking=chunking, kwargs...)
function matrix_representation(op, H::BdGHilbertSpace, repr=EagerSparseRepr(); chunking=NoChunking(), kwargs...)
    repr isa Symbol && return Base.inferencebarrier(matrix_representation)(op, H, _canonicalize_repr(repr); chunking, kwargs...)
    _isscalar_operator(op) && return _identity_free_zero(H, repr)
    return normal_order_to_bdg(_nambu_matrix(op, H, repr))
end
# The matrix `m` with `op == c + Ψ' m Ψ` for a scalar `c`, of which the BdG matrix is the particle-hole symmetric part.
function _nambu_matrix(op, H, repr)
    _is_quadratic_fermion_operator(op, false) || throw(ArgumentError("Only quadratic fermionic operators can be represented on a BdG Hilbert space. Use `hilbert_space(f, labels)` and `representation(op, H)` for other operators."))
    return _quadratic_matrix(_bdg_entry, remove_identity(op), H, repr)
end
function _bdg_entry(term, H)
    instate, outstate = _nambu_in_out(term, map(f -> mode_position(f, H), term.factors))
    return state_index(outstate, H), state_index(instate, H), term.coeff
end

# The term c_a^(d1) c_b^(d2), with `a` and `b` at `positions`, maps the Nambu state (b, !d2) to (a, d1)
function _nambu_in_out(term, positions)
    a, b = term.factors
    return NambuState(positions[2], !b.creation), NambuState(positions[1], a.creation)
end
function _precomputation_before_operator_application(op::FermionSym, space::AbstractHilbertSpace{NambuState})
    position = mode_position(op, space)
    position == 0 && throw(ArgumentError("The operator `$op` contains a factor that is not part of the fermionic space `$space`."))
    return position
end
function apply_local_operators(op::NCMul, state::NambuState, H::AbstractHilbertSpace, positions; transpose)
    length(op.factors) == 2 || throw(ArgumentError("Exactly two fermions must act on a `NambuState`, got $(length(op.factors))."))
    instate, outstate = _nambu_in_out(op, positions)
    transpose && ((instate, outstate) = (outstate, instate))
    return state == instate ? (outstate, op.coeff) : (state, zero(op.coeff))
end

@testitem "BdG" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    h = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
    H = hilbert_space(f, 1:2, BdG())
    @test representation(h + 1, H) == representation(h, H)

    h = rand(ComplexF64, 2, 2) + hc
    Δ = rand(ComplexF64, 2, 2)
    Δ = Δ - transpose(Δ)
    bdg_ham = [h/2 Δ/2
        -conj(Δ)/2 -conj(h / 2)]
    op = sum(f[n]' * h[n, m] * f[m] for n in 1:2, m in 1:2) +
        sum(f[n]' * Δ[n, m] * f[m]' / 2 + hc for n in 1:2, m in 1:2)
    bdg_ham2 = representation(op, H)
    @test bdg_ham ≈ bdg_ham2
end

@testitem "BdG matrix of non-hermitian operators" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, BdG())
    Ψ = [[f[i] for i in 1:N]; [f[i]' for i in 1:N]]
    τx = [0I I(N); I(N) 0I]
    c() = rand(-5:5) // 1 + rand(-5:5) // 1 * im
    op = sum(c() * f[i]' * f[j] + c() * f[i]' * f[j]' + c() * f[i] * f[j] for i in 1:N, j in 1:N)
    M = representation(op, H, :dense)
    @test M == -τx * transpose(M) * τx
    rest = op - sum(M[i, j] * Ψ[i]' * Ψ[j] for i in 1:2N, j in 1:2N)
    @test FermionicHilbertSpaces.NonCommutativeProducts.isscalar(rest)
end

@testitem "BdG equation of motion" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, BdG())
    Hfock = hilbert_space(f, 1:N)
    Ψ = [[f[i] for i in 1:N]; [f[i]' for i in 1:N]]
    op = sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + rand(ComplexF64) * f[i] * f[j] for i in 1:N, j in 1:N)
    M = representation(op, H, :dense)
    rep(x) = representation(x, Hfock, :dense)
    for k in 1:2N
        @test rep(op * Ψ[k] - Ψ[k] * op) ≈ -2 * sum(M[k, j] * rep(Ψ[j]) for j in 1:2N)
    end
end

@testitem "BdG expectation values" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, BdG())
    Hfock = hilbert_space(f, 1:N)
    ham = sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + hc for i in 1:N, j in 1:N)
    ρ = exp(-representation(ham, Hfock, :dense))
    ρ /= tr(ρ)
    Ψ = [[f[i] for i in 1:N]; [f[i]' for i in 1:N]]
    G = [tr(ρ * representation(Ψ[i]' * Ψ[j], Hfock, :dense)) for i in 1:2N, j in 1:2N]
    for op in (f[1]' * f[2], f[1] * f[2], f[1]' * f[2]', f[1] * f[2]', ham + 2)
        @test expectation(op, G, H) ≈ tr(ρ * representation(op, Hfock, :dense))
    end
end

@testitem "BdG representation formats" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    H = hilbert_space(f, 1:3, BdG())
    op = sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + hc for i in 1:3, j in 1:3)
    M = representation(op, H)
    D = representation(op, H, :dense)
    @test D isa Matrix && D == M
    @test representation(op, H, ComplexF64) == M
    @test_throws ArgumentError representation(op, H, :lazy)
end

@testitem "Space of Nambu states" begin
    using FermionicHilbertSpaces.FreeFermions
    using LinearAlgebra, OhMyThreads, PartitionedArrays
    import FermionicHilbertSpaces: StateChunking, EagerDenseRepr, PartitionedSparseRepr
    using FermionicHilbertSpaces.FreeFermions: normal_order_to_bdg
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, vec([NambuState(n, hole) for (n, hole) in Base.product(1:N, (true, false))]))
    op = sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + hc for i in 1:N, j in 1:N) + 2I
    M = representation(op, H)
    @test normal_order_to_bdg(M - 2I) ≈ representation(op, hilbert_space(f, 1:N, BdG()))
    @test representation(op, H, EagerDenseRepr()) == M
    @test representation(op, H; chunking=StateChunking(StaticScheduler(; nchunks=2))) ≈ M
    v = rand(ComplexF64, dim(H))
    @test representation(op, H, :lazy) * v ≈ M * v
    M_dist = representation(op, H, PartitionedSparseRepr(; backend=DebugArray))
    w = pvector(inds -> rand(ComplexF64, length(inds)), M_dist.col_partition)
    @test collect(M_dist * w) ≈ M * collect(w)
    @test_throws ArgumentError representation(f[1]' * f[2]' * f[3] * f[1], H)
end

@testitem "BdG space does not combine with generic space operations" begin
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    H = hilbert_space(f, 1:2, BdG())
    @test !(H isa FermionicHilbertSpaces.AbstractHilbertSpace)
    m = rand(dim(H), dim(H))
    @test_throws MethodError partial_trace(m, H, H)
    @test_throws MethodError embed(m, H, H)
    @test_throws MethodError subregion(H, H)
end

function isbdgmatrix(A::AbstractMatrix)
    # Check if A is a BdG matrix, i.e., it has the form:
    # [H, Δ; -conj(Δ), -conj(H)]
    # where H is Hermitian and Δ is antisymmetric.
    size(A, 1) == size(A, 2) || return false
    N = div(size(A, 1), 2)
    inds1, inds2 = axes(A)
    H = @views A[inds1[1:N], inds2[1:N]]
    Δ = @views A[inds1[1:N], inds2[(N+1):2N]]
    Hd = @views A[inds1[(N+1):2N], inds2[(N+1):2N]]
    Δd = @views A[inds1[(N+1):2N], inds2[1:N]]
    _isbdgmatrix(H, Δ, Hd, Δd)
end
function _isbdgmatrix(H, Δ, Hd, Δd)
    indsm, indsn = axes(H)
    if indsm != indsn
        return false
    end
    for i = first(indsn):last(indsn), j = (i):last(indsn)
        if H[i, j] != conj(H[j, i])
            return false
        end
        if H[i, j] != -conj(Hd[i, j])
            return false
        end
        if Δ[i, j] != -conj(Δd[i, j])
            return false
        end
        if Δ[i, j] != -Δ[j, i]
            return false
        end
    end
    return true
end

function quasiparticle_adjoint(v::AbstractVector)
    Base.require_one_based_indexing(v)
    N = div(length(v), 2)
    out = similar(v)
    for i in 1:N
        out[i] = conj(v[i+N])
        out[i+N] = conj(v[i])
    end
    return out
end
quasiparticle_adjoint_index(n, N) = 2N + 1 - n


abstract type AbstractBdGEigenAlg end

"""
    BdGEigen(cutoff = nothing)

Algorithm for `eigen(A, BdGEigen())` on a Hermitian BdG matrix `A` with `N` modes, giving sorted eigenvalues and orthonormal eigenvectors. Columns `k` and `2N + 1 - k` are particle-hole partners with eigenvalues `e` and `-e`, and a column `[u; v]` is the quasiparticle `Σᵢ u[i] f[i]' + v[i] f[i]`. Eigenvalues below `cutoff` (relative, default `2N * eps`) are set to zero.

```julia
H = hilbert_space(f, 1:2, BdG())
vals, vecs = eigen(representation(f[1]' * f[2] + f[1]' * f[2]' + hc, H, :dense), BdGEigen()) # vals ≈ [-1, 0, 0, 1]
```
"""
struct BdGEigen{T <: Union{Nothing, Real}} <: AbstractBdGEigenAlg
    cutoff::T
end
BdGEigen() = BdGEigen(nothing)

function LinearAlgebra.eigen(A::AbstractMatrix, alg::BdGEigen)
    rtol = sqrt(eps(real(float(eltype(A)))))
    isapprox(A, A'; rtol) || throw(ArgumentError("BdGEigen requires a Hermitian BdG matrix, the representation of a hermitian operator."))
    F = eigen(Hermitian(Matrix(A)))
    cutoff = something(alg.cutoff, size(A, 1) * eps(eltype(F.values)))
    tol = cutoff * maximum(abs, F.values; init = zero(eltype(F.values)))
    return Eigen(enforce_ph_symmetry(F.values, complex(F.vectors), tol)...)
end

# Within a degenerate eigenspace `eigen` returns an arbitrary basis, so the partner of a
# negative-energy column is built as its particle-hole conjugate. That map is antiunitary and
# takes the e-eigenspace onto the -e one. For a small splitting e the computed C v_k is only
# orthogonal to v_k up to eps * norm(A) / e, so the columns are Löwdin orthonormalized, i.e.
# replaced by the polar factor of their SVD. If C V = V P with P the reversal permutation, then
# conj(V'V) = P V'V P, so the result stays paired.
function enforce_ph_symmetry(es, ops, tol)
    N = div(length(es), 2)
    for k in 1:N
        k2 = quasiparticle_adjoint_index(k, N)
        isapprox(es[k], -es[k2]; atol = tol, rtol = sqrt(eps(eltype(es)))) || @warn "The spectrum is not particle-hole symmetric: es[$k] = $(es[k]) and es[$k2] = $(es[k2])."
        if es[k] < -tol
            ops[:, k2] = quasiparticle_adjoint(ops[:, k])
            es[k2] = -es[k]
        end
    end
    m = count(>=(-tol), view(es, 1:N))
    zero_modes = (N - m + 1):(N + m)
    if !isempty(zero_modes)
        es[zero_modes] .= 0
        ops[:, zero_modes] = canonicalize_zero_modes(ops[:, zero_modes])
    end
    U, _, V = svd(ops)
    return es, U * V'
end

# The zero-energy eigenspace is particle-hole invariant, so the Takagi factorization of
# transpose(Z) * σx * Z gives a self-conjugate (Majorana) basis Y of it. Columns j and 2m+1-j
# are then built from Y as a particle-hole pair.
function canonicalize_zero_modes(Z)
    n = div(size(Z, 1), 2)
    m = div(size(Z, 2), 2)
    U, _ = symmetric_SVD(Symmetric(transpose(Z) * [Z[(n + 1):end, :]; Z[1:n, :]]))
    Y = Z * conj(U)
    Ya = Y[:, 1:m]
    Yb = reverse(Y[:, (m + 1):end]; dims = 2)
    return [Ya + im * Yb reverse(Ya - im * Yb; dims = 2)] / sqrt(2)
end

#Symmetric form of SVD for symmetric matrices, Takagi/Autonne decomposition
function symmetric_SVD(M)
    issymmetric(M) || throw(ArgumentError("`M` must be symmetric."))
    n = size(M, 1)
    block_matrix = [
        -real(M) imag(M);
        imag(M) real(M)
    ]
    vals, Z = eigen(Symmetric(block_matrix))
    pos = vals .> 0
    U = Z[(n + 1):end, pos] + 1im * Z[1:n, pos]
    return U, Diagonal(vals[pos])
end

@testitem "Takagi" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions: symmetric_SVD
    import Random: seed!
    seed!(1)
    m = rand(4, 4)
    m = m + m'  # Make it symmetric
    U, S = symmetric_SVD(m)
    @test U'U ≈ U * U' ≈ I
    @test m ≈ U * S * transpose(U)
    @test (S .> 0) == I
end

@testitem "BdG on Kitaev Sweet spot" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    symham = f[1]' * f[2] + f[1]' * f[2]' + hc
    H = hilbert_space(f, 1:2, BdG())
    vals, vecs = eigen(representation(symham, H, :dense), BdGEigen())
    @test vals ≈ [-1, 0, 0, 1]
    @test 1 ≈ abs(vecs[:, 2][1]^2 + vecs[:, 3][1]^2) / (abs2(vecs[:, 2][1]) + abs2(vecs[:, 3][1]))

    # three site
    symham = f[1]' * f[2] + f[2]' * f[3] + f[1]' * f[2]' + f[2]' * f[3]' + hc
    H = hilbert_space(f, 1:3, BdG())
    Hmb = hilbert_space(f, 1:3)
    matmb = representation(symham, Hmb, :dense)
    vals, vecs = eigen(representation(symham, H, :dense), BdGEigen())
    @test 1 ≈ abs(vecs[:, 3][1]^2 + vecs[:, 4][1]^2) / (abs2(vecs[:, 3][1]) + abs2(vecs[:, 4][1]))

    # The BdG matrix is half the Nambu Hamiltonian, so a positive eigenvalue e_k gives a
    # quasiparticle energy 2e_k, and the many-body energies are tr(h)/2 - Σe_k + Σ n_k 2e_k.
    function mb_energies(e, trh)
        occupations = Iterators.product(ntuple(_ -> (0, 1), length(e))...)
        sort(vec([trh / 2 + sum((2 .* n .- 1) .* e) for n in occupations]))
    end
    @test eigvals(Hermitian(matmb)) ≈ mb_energies(vals[4:6], 0)

    h = rand(ComplexF64, 3, 3) + hc
    Δ = rand(ComplexF64, 3, 3)
    Δ = Δ - transpose(Δ)
    symham = sum(f[n]' * h[n, m] * f[m] for n in 1:3, m in 1:3) +
             sum(f[n]' * Δ[n, m] * f[m]' / 2 + hc for n in 1:3, m in 1:3)
    vals, _ = eigen(representation(symham, H, :dense), BdGEigen())
    @test eigvals(Hermitian(representation(symham, Hmb, :dense))) ≈ mb_energies(vals[4:6], real(tr(h)))
end

@testitem "BdGEigen quasiparticles" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    function test_quasiparticles(op, N)
        F = eigen(representation(op, hilbert_space(f, 1:N, BdG())), BdGEigen())
        @test issorted(F.values)
        @test F.vectors' * F.vectors ≈ I
        Hmb = hilbert_space(f, 1:N)
        Hm = representation(op, Hmb, :dense)
        γ(k) = representation(sum(F.vectors[i, k] * f[i]' + F.vectors[N + i, k] * f[i] for i in 1:N), Hmb, :dense)
        for k in 1:2N
            @test Hm * γ(k) - γ(k) * Hm ≈ 2F.values[k] * γ(k) atol = 1.0e-12
            @test γ(2N + 1 - k) ≈ γ(k)'
        end
    end
    test_quasiparticles(sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + hc for i in 1:3, j in 1:3), 3)
    test_quasiparticles(f[1]' * f[1] + f[2]' * f[2], 2)
    # Two spinful sites (modes 1, 2 and 3, 4) with singlet pairing: every level is doubly degenerate
    test_quasiparticles(0.3 * sum(f[i]' * f[i] for i in 1:4) + f[1]' * f[3] + f[2]' * f[4] + 0.7 * (f[1]' * f[2]' + f[3]' * f[4]') + hc, 4)
    # Two decoupled Kitaev chains at the sweet spot: four zero modes
    test_quasiparticles(f[1]' * f[2] + f[1]' * f[2]' + f[3]' * f[4] + f[3]' * f[4]' + hc, 4)

    function test_bdg_eigen(A)
        F = eigen(A, BdGEigen())
        N = div(size(A, 1), 2)
        @test issorted(F.values)
        @test F.vectors' * F.vectors ≈ I
        @test A * F.vectors ≈ F.vectors * Diagonal(F.values)
        for k in 1:2N
            @test F.values[2N + 1 - k] == -F.values[k]
            @test F.vectors[:, 2N + 1 - k] ≈ FreeFermions.quasiparticle_adjoint(F.vectors[:, k])
        end
        return F
    end
    # Kitaev-type chain with a tiny but nonzero Majorana splitting
    op = sum(f[i]' * f[i + 1] + 0.9 * f[i]' * f[i + 1]' + hc for i in 1:11) + 0.3 * sum(f[i]' * f[i] for i in 1:12)
    vals = test_bdg_eigen(representation(op, hilbert_space(f, 1:12, BdG()), :dense)).values
    @test 0 < vals[13] < 1.0e-6
    # Zero modes at a large energy scale
    test_bdg_eigen(1.0e8 * representation(f[1]' * f[2] + f[1]' * f[2]' + f[3]' * f[4] + f[3]' * f[4]' + hc, hilbert_space(f, 1:4, BdG()), :dense))
    @test_throws ArgumentError eigen([0 1; 0 0], BdGEigen())
end

# The operators Ψ_j^† that create the basis states of `H`, such that `representation(Σ_ab M_ab Ψ_a^† Ψ_b, H) == M`
_nambu_creators(H::SingleParticleHilbertSpace) = map(adjoint, modes(H))
_nambu_creators(H::BdGHilbertSpace) = vcat(map(adjoint, modes(H)), modes(H))

"""
    quasiparticle(v, H)

The quasiparticle creation operator `γ' = Σ_n (v[n] * f[n]' + v[n+N] * f[n])` for a column `v` of BdG eigenvectors on a `BdG` space `H` with `N` modes, or `γ' = Σ_n v[n] * f[n]'` on a `SingleParticle` space.

```julia
@fermions f
H = hilbert_space(f, 1:2, BdG())
quasiparticle([1, 0, 0, 1], H) # f[2] + f†[1]
```
"""
function quasiparticle(v, H::Union{BdGHilbertSpace, SingleParticleHilbertSpace})
    creators = _nambu_creators(H)
    length(v) == length(creators) || throw(DimensionMismatch("The vector has length $(length(v)), but the space has dimension $(length(creators))."))
    return sum(c * x for (c, x) in zip(creators, v))
end

"""
    symbolic_operator(M, H)

The symbolic operator `op` with `representation(op, H) == M`, for a BdG matrix `M` on a `BdG` space or any matrix on a `SingleParticle` space. On a `BdG` space, `op` is traceless.

```julia
@fermions f
H = hilbert_space(f, 1:2, BdG())
symbolic_operator(representation(f[1]' * f[1], H, :dense), H) # -0.5I + f†[1]*f[1]
```
"""
function symbolic_operator(M, H::Union{BdGHilbertSpace, SingleParticleHilbertSpace})
    creators = _nambu_creators(H)
    size(M) == (length(creators), length(creators)) || throw(DimensionMismatch("The matrix has size $(size(M)), but the space has dimension $(length(creators))."))
    terms = (M[a, b] * ca * cb' for (a, ca) in zip(axes(M, 1), creators), (b, cb) in zip(axes(M, 2), creators) if !iszero(M[a, b]))
    return sum(terms; init = 0)
end

@testitem "BdG quadratic operators and quasiparticles" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    N = 3
    H = hilbert_space(f, 1:N, BdG())
    Hmb = hilbert_space(f, 1:N)
    h = rand(ComplexF64, N, N) + hc
    Δ = rand(ComplexF64, N, N)
    Δ = Δ - transpose(Δ)
    op = sum(f[n]' * h[n, m] * f[m] for n in 1:N, m in 1:N) +
        sum(f[n]' * Δ[n, m] * f[m]' / 2 + hc for n in 1:N, m in 1:N)
    M = representation(op, H, :dense)
    @test representation(symbolic_operator(M, H), H) ≈ M
    @test representation(symbolic_operator(M, H) - op, Hmb, :dense) ≈ -tr(h) / 2 * I

    vals, vecs = eigen(M, BdGEigen())
    Hmat = representation(op, Hmb, :dense)
    for k in (N + 1):2N
        q = representation(quasiparticle(vecs[:, k], H), Hmb, :dense)
        @test Hmat * q - q * Hmat ≈ 2vals[k] * q
    end
end

@testitem "Single particle quadratic operators and quasiparticles" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    @fermions f
    H = hilbert_space(f, 1:3, SingleParticle())
    h = rand(ComplexF64, 3, 3) + hc
    op = symbolic_operator(h, H)
    @test representation(op, H) ≈ h
    Hmat = representation(op, hilbert_space(f, 1:3), :dense)
    vals, vecs = eigen(Hermitian(h))
    q = representation(quasiparticle(vecs[:, 1], H), hilbert_space(f, 1:3), :dense)
    @test Hmat * q - q * Hmat ≈ vals[1] * q
end
