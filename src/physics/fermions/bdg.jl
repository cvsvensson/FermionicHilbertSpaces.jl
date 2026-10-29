struct NambuState <: AbstractFockState
    state::SingleParticleState
    hole::Bool
end
NambuState(i::Integer, hole::Bool) = NambuState(SingleParticleState(i), hole)
Base.:(==)(n1::NambuState, n2::NambuState) = n1.state == n2.state && n1.hole == n2.hole
Base.hash(n::NambuState, h::UInt) = hash(n.hole, hash(n.state, h))

function normal_order_to_bdg(m::AbstractMatrix)
    n = size(m, 1)
    h = m[1:(n÷2), 1:(n÷2)] / 2
    δ = m[1:(n÷2), (n÷2+1):end]
    δd = m[(n÷2+1):end, 1:(n÷2)]
    Δ = (δ - transpose(δ) + (δd - transpose(δd))') / 4
    [h Δ
        -conj(Δ) -conj(h)]
end

struct BdGHilbertSpace{H}
    parent::H
    function BdGHilbertSpace(f::SymbolicFermionBasis, labels)
        states = vec([NambuState(i, hole) for (i, label) in enumerate(labels), hole in (true, false)])
        H = hilbert_space(f, labels, states)
        return new{typeof(H)}(H)
    end
end
"""
    BdG()

Marker passed as the last argument of `hilbert_space(f, labels, BdG())` to select the Bogoliubov-de Gennes (Nambu) space of non-interacting fermions with superconducting pairing.
"""
struct BdG end

"""
    hilbert_space(f::SymbolicFermionBasis, labels, ::BdG)

The Nambu space with one particle state and one hole state for each of the modes `f[l]`, `l in labels`. The matrix representation of a quadratic operator is the Bogoliubov-de Gennes matrix

    [H   Δ
    -Δ* -H*]

where `H` is hermitian and `Δ` is antisymmetric, without any contribution from the identity operator. Operators that are not quadratic in the fermions cannot be represented. The space wraps a constrained fermionic space, available through `parent`, and does not combine with tensor products or partial traces.
"""
hilbert_space(f::SymbolicFermionBasis, labels, ::BdG) = BdGHilbertSpace(f, labels)
@deprecate bdg_hilbert_space(f, labels) hilbert_space(f, labels, BdG())
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
    _isscalar_operator(op) && return _identity_free_zero(H, repr)
    _is_quadratic_fermion_operator(op, false) || throw(ArgumentError("Only quadratic fermionic operators can be represented on a BdG Hilbert space. Use `hilbert_space(f, labels)` and `representation(op, H)` for other operators."))
    normal_order_to_bdg(_quadratic_matrix(_bdg_entry, remove_identity(op), H, repr))
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

@testitem "BdG representation formats" begin
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
    using LinearAlgebra, OhMyThreads, PartitionedArrays
    import FermionicHilbertSpaces: NambuState, StateChunking, EagerDenseRepr, PartitionedSparseRepr, normal_order_to_bdg
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
    BdGEigen(cutoff=1e-12)

Algorithm for `eigen(A, BdGEigen())`, which diagonalizes a BdG matrix `A = representation(op, hilbert_space(f, labels, BdG()), :dense)` with `N` modes and returns an `Eigen` with eigenvalues sorted in ascending order and orthonormal eigenvectors. A column `[u; v]` with eigenvalue `e` gives the quasiparticle `γ' = Σᵢ u[i] * f[labels[i]]' + v[i] * f[labels[i]]`, which satisfies `op * γ' - γ' * op = 2e * γ'`. Columns `k` and `2N + 1 - k` are particle-hole partners: their eigenvalues are `e` and `-e` and their quasiparticles are adjoints of each other. `cutoff` is the tolerance used when enforcing this pairing.

```julia
@fermions f
H = hilbert_space(f, 1:2, BdG())
op = f[1]' * f[2] + f[1]' * f[2]' + hc
vals, vecs = eigen(representation(op, H, :dense), BdGEigen())
```
"""
struct BdGEigen{T<:Number} <: AbstractBdGEigenAlg
    cutoff::T
end
BdGEigen() = BdGEigen(DEFAULT_PH_CUTOFF)

function LinearAlgebra.eigen(A::AbstractMatrix, alg::BdGEigen; canon_alg=SVDCanon())
    return Eigen(enforce_ph_symmetry(eigen(A), canon_alg; cutoff = alg.cutoff)...)
end


struct ProjectionCanon end
struct SVDCanon end

function enforce_ph_symmetry(_es, _ops, canon_alg=SVDCanon(); cutoff=DEFAULT_PH_CUTOFF)
    p = sortperm(_es)
    es = _es[p]
    ops = complex(_ops[:, p])
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff && isapprox(es[k], -es[k2], atol=cutoff)
            @warn "es[k] = $(es[k]) != $(-es[k2]) = -es[k_adj]"
        end
        v1 = ops[:, k]
        v2 = ops[:, k2]
        ph1 = ph(v1)
        ph_overlap = dot(ph1, v2)
        if isapprox(abs(ph_overlap), 1; atol=cutoff)
            ops[:, k] .= v1
            ops[:, k2] .= v2 ./ ph_overlap
        else
            o1, o2 = eachcol(canonicalize_particle_pair(v1, v2, canon_alg))
            if abs(dot(o1, v1)) > abs(dot(o2, v1))
                ops[:, k] = o1
                ops[:, k2] = o2
            else
                ops[:, k] = o2
                ops[:, k2] = o1
            end
        end
    end
    es, ops
end

function enforce_ph_symmetry(F::Eigen, canon_alg=SVDCanon(); cutoff=DEFAULT_PH_CUTOFF)
    if isreal(F.values)
        enforce_ph_symmetry(real(F.values), F.vectors, canon_alg; cutoff)
    else
        throw(ArgumentError("The eigenvalues must be real, got $(F.values)."))
    end
end

function canonicalize_particle_pair(v1, v2, ::ProjectionCanon)
    ph = quasiparticle_adjoint
    ph1, ph2 = ph(v1), ph(v2)

    # Construct all Majorana candidates
    all_majs = if isreal(v1) && isreal(v2)
        [ph1 + v1, ph2 + v2]
    else
        [ph1 + v1, 1im * (ph1 - v1), ph2 + v2, 1im * (ph2 - v2)]
    end

    # Use QR with pivoting to find two most linearly independent Majoranas
    # Stack as columns and find best basis
    M = hcat(all_majs...)
    Q, R, p = qr(M, ColumnNorm())  # pivoted QR
    # Take first 2 columns (most independent, via pivot order)
    majs = M[:, p[1:2]]
    # Orthonormalize via Cholesky of Gram matrix
    HM = Hermitian(majs' * majs)
    X = try
        cholesky(HM)
    catch
        vals = eigvals(HM)
        reg = max(1e-12, 1e6 * eps(eltype(vals)))
        @warn "Cholesky failed, eigenvals = $vals. Adding regularization."
        cholesky(HM + reg * I)
    end
    T = [1 1; 1im -1im] / sqrt(2)
    newmajs = majs * (X.U \ T)
    return newmajs
end

#Symmetric form of SVD for symmetric matrices, Takagi/Autonne decomposition
function symmetric_SVD(M)
    issymmetric(M) || throw(ArgumentError("`M` must be symmetric."))
    n = size(M, 1)
    block_matrix = [
        -real(M) imag(M);
        imag(M) real(M)
    ]
    S = schur(Symmetric(block_matrix))
    # Extract positive eigenvalues
    pos_eigenval_positions = diag(S.T) .> 0
    D = Diagonal(diag(S.T)[pos_eigenval_positions])
    # Reconstruct complex unitary matrix
    U = S.Z[(n+1):end, pos_eigenval_positions] + 1im * S.Z[1:n, pos_eigenval_positions]
    return U, D
end

@testitem "Takagi" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: symmetric_SVD
    import Random: seed!
    seed!(1)
    m = rand(4, 4)
    m = m + m'  # Make it symmetric
    U, S = symmetric_SVD(m)
    @test U'U ≈ U * U' ≈ I
    @test m ≈ U * S * transpose(U)
    @test (S .> 0) == I
end

canonicalize_particle_pair(v1, v2, ::SVDCanon) = canonicalize_particle_pair([v1 v2], SVDCanon())


function canonicalize_particle_pair(vs::AbstractMatrix, ::SVDCanon)
    size(vs, 2) == 2 || throw(ArgumentError("`vs` must have exactly two columns for SVD canonicalization, got $(size(vs, 2))."))
    vs'vs ≈ I || throw(ArgumentError("The quasiparticle vectors `vs` must be orthonormal."))
    n = div(size(vs, 1), 2)
    mat = Symmetric(transpose(vs) * [0I I(n); I(n) 0I] * vs)
    U, D = symmetric_SVD(mat)
    # check D ≈ I ?
    vs * conj(U) * [1 1; 1im -1im] / sqrt(2)
end


@testitem "BdG Canonicalization" begin
    using FermionicHilbertSpaces: canonicalize_particle_pair, SVDCanon, ProjectionCanon, quasiparticle_adjoint, BdGEigen, isbdgmatrix
    using LinearAlgebra
    @fermions f
    N = 4
    H = hilbert_space(f, 1:N, BdG())
    ham = sum(rand(ComplexF64) * f[n]'f[k] + rand(ComplexF64) * f[n]'f[k]' + hc for (n, k) in Iterators.product(1:N, 1:N))
    h = representation(ham, H)
    @test ishermitian(h)
    @test isbdgmatrix(h)
    F = eigen(collect(h))
    pairs = [F.vectors[:, [n, 2N + 1 - n]] for n in 1:N]
    for vs in pairs
        os = canonicalize_particle_pair(vs, SVDCanon())
        @test os'os ≈ I
        @test quasiparticle_adjoint(os[:, 1]) ≈ os[:, 2]
        @test quasiparticle_adjoint(os[:, 2]) ≈ os[:, 1]
        os = canonicalize_particle_pair(eachcol(vs)..., ProjectionCanon())
        @test os'os ≈ I
        @test quasiparticle_adjoint(os[:, 1]) ≈ os[:, 2]
        @test quasiparticle_adjoint(os[:, 2]) ≈ os[:, 1]
    end
end

const DEFAULT_PH_CUTOFF = 1e-12

@testitem "BdG on Kitaev Sweet spot" begin
    using FermionicHilbertSpaces: BdGEigen
    using LinearAlgebra
    @fermions f
    symham = f[1]' * f[2] + f[1]' * f[2]' + hc
    H = hilbert_space(f, 1:2, BdG())
    mat = representation(symham, H)
    vals, vecs = eigen(Matrix(mat), BdGEigen(); canon_alg=FermionicHilbertSpaces.SVDCanon())
    @test vals ≈ [-1, 0, 0, 1]
    @test 1 ≈ abs(vecs[:, 2][1]^2 + vecs[:, 3][1]^2) / (abs2(vecs[:, 2][1]) + abs2(vecs[:, 3][1]))
    vals2, vecs2 = eigen(Matrix(mat), BdGEigen(); canon_alg=FermionicHilbertSpaces.ProjectionCanon())
    @test vals ≈ vals2
    @test 1 ≈ abs(vecs2[:, 2][1]^2 + vecs2[:, 3][1]^2) / (abs2(vecs2[:, 2][1]) + abs2(vecs2[:, 3][1]))

    # three site
    symham = f[1]' * f[2] + f[2]' * f[3] + f[1]' * f[2]' + f[2]' * f[3]' + hc
    H = hilbert_space(f, 1:3, BdG())
    Hmb = hilbert_space(f, 1:3)
    matmb = representation(symham, Hmb, :dense)
    mat = representation(symham, H)
    vals, vecs = eigen(Matrix(mat), BdGEigen(); canon_alg=FermionicHilbertSpaces.SVDCanon())
    @test 1 ≈ abs(vecs[:, 3][1]^2 + vecs[:, 4][1]^2) / (abs2(vecs[:, 3][1]) + abs2(vecs[:, 4][1]))
    vals2, vecs2 = eigen(Matrix(mat), BdGEigen(); canon_alg=FermionicHilbertSpaces.ProjectionCanon())
    @test vals ≈ vals2
    @test 1 ≈ abs(vecs2[:, 3][1]^2 + vecs2[:, 4][1]^2) / (abs2(vecs2[:, 3][1]) + abs2(vecs2[:, 4][1]))

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
    @fermions f
    N = 3
    op = sum(rand(ComplexF64) * f[i]' * f[j] + rand(ComplexF64) * f[i]' * f[j]' + hc for i in 1:N, j in 1:N)
    F = eigen(representation(op, hilbert_space(f, 1:N, BdG()), :dense), BdGEigen())
    @test issorted(F.values)
    @test F.vectors' * F.vectors ≈ I
    Hmb = hilbert_space(f, 1:N)
    Hm = representation(op, Hmb, :dense)
    γ(k) = representation(sum(F.vectors[i, k] * f[i]' + F.vectors[N + i, k] * f[i] for i in 1:N), Hmb, :dense)
    for k in 1:2N
        @test Hm * γ(k) - γ(k) * Hm ≈ 2F.values[k] * γ(k)
        @test γ(2N + 1 - k) ≈ γ(k)'
    end
end