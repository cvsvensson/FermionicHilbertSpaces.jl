struct NambuState
    state::SingleParticleState
    hole::Bool
end
NambuState(i::Integer, hole::Bool) = NambuState(SingleParticleState(i), hole)

function togglefermions(sites, daggers, f::NambuState)
    (length(sites) == 2 == length(daggers)) || throw(ArgumentError("Must act with exactly two fermions on a NambuState"))
    allowed = sites[2] == only(f.state.sites) && daggers[2] == f.hole
    state = NambuState(SingleParticleState(sites[1]), !daggers[1])
    return state, allowed
end

function normal_order_to_bdg(m::AbstractMatrix)
    n = size(m, 1)
    h = m[1:n÷2, 1:n÷2] / 2
    δ = m[1:n÷2, n÷2+1:end]
    # hd = m[n÷2+1:end, n÷2+1:end]
    δd = m[n÷2+1:end, 1:n÷2]
    # h = (h - conj(hd)) / 2
    Δ = (δ + δd') / 2
    [h Δ
        -conj(Δ) -conj(h)]
end

struct BdGHilbertSpace{H} <: AbstractFockHilbertSpace
    parent::H
    function BdGHilbertSpace(labels)
        states = vec([NambuState(i, hole) for (i, label) in enumerate(labels), hole in (true, false)])
        H = hilbert_space(labels, states)
        return new{typeof(H)}(H)
    end
end
Base.size(h::BdGHilbertSpace) = size(h.parent)
Base.size(h::BdGHilbertSpace, dim) = size(h.parent, dim)
mode_ordering(h::BdGHilbertSpace) = mode_ordering(h.parent)
Base.keys(h::BdGHilbertSpace) = keys(h.parent)

function matrix_representation(op, H::BdGHilbertSpace)
    isquadratic(op) || throw(ArgumentError("Operator must be quadratic in fermions to be represented on a BdG Hilbert space."))
    normal_order_to_bdg(matrix_representation(remove_identity(op), H.parent))
end

@testitem "BdG" begin
    import FermionicHilbertSpaces: BdGHilbertSpace
    @fermions f
    h = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
    H = BdGHilbertSpace(1:2)
    @test matrix_representation(h + 1, H) == matrix_representation(h, H)
end


function isantisymmetric(A::AbstractMatrix)
    indsm, indsn = axes(A)
    if indsm != indsn
        return false
    end
    for i = first(indsn):last(indsn), j = (i):last(indsn)
        if A[i, j] != -A[j, i]
            return false
        end
    end
    return true
end
function bdg_view(A::AbstractMatrix)
    isbdgmatrix(A) || throw(ArgumentError("Matrix must be a BdG matrix."))
    H = @views A[inds1[1:N], inds2[1:N]]
    Δ = @views A[inds1[1:N], inds2[N+1:2N]]
    return H, Δ
end
function isbdgmatrix(A::AbstractMatrix)
    # Check if A is a BdG matrix, i.e., it has the form:
    # [H, Δ; -conj(Δ), -conj(H)]
    # where H is Hermitian and Δ is antisymmetric.
    size(A, 1) == size(A, 2) || return false
    N = div(size(A, 1), 2)
    inds1, inds2 = axes(A)
    H = @views A[inds1[1:N], inds2[1:N]]
    Δ = @views A[inds1[1:N], inds2[N+1:2N]]
    Hd = @views A[inds1[N+1:2N], inds2[N+1:2N]]
    Δd = @views A[inds1[N+1:2N], inds2[1:N]]
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

struct SkewEigenAlg{T<:Number} <: AbstractBdGEigenAlg
    cutoff::T # tolerance for particle-hole symmetry
end
struct BdGEigen{T<:Number} <: AbstractBdGEigenAlg
    cutoff::T # tolerance for particle-hole symmetry
end
BdGEigen() = BdGEigen(DEFAULT_PH_CUTOFF)
SkewEigenAlg() = SkewEigenAlg(DEFAULT_PH_CUTOFF)

function LinearAlgebra.eigen(A::AbstractMatrix, alg::BdGEigen)
    enforce_ph_symmetry(eigen(A), cutoff=alg.cutoff)
end


struct ProjectionCanon end
struct SVDCanon end

function enforce_ph_symmetry(_es, _ops, canon_alg=ProjectionCanon(); cutoff=DEFAULT_PH_CUTOFF)
    p = sortperm(_es)
    es = _es[p]
    ops = (_ops[:, p])
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
            o1, o2 = canonicalize_particle_pair(v1, v2, canon_alg)
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

function enforce_ph_symmetry(F::Eigen, canon_alg=ProjectionCanon(); cutoff=DEFAULT_PH_CUTOFF)
    if isreal(F.values)
        enforce_ph_symmetry(real(F.values), F.vectors, canon_alg; cutoff)
    else
        throw(ArgumentError("Eigenvalues must be real"))
    end
end

function canonicalize_particle_pair(v1, v2, ::ProjectionCanon)
    ph = quasiparticle_adjoint
    ph1 = ph(v1)
    ph2 = ph(v2)
    all_majs = [ph1 + v1, 1im * (ph1 - v1), ph2 + v2, 1im * (ph2 - v2)]
    sort!(all_majs, by=norm)
    majplus = all_majs[1]
    majminus = argmin(y -> abs(dot(y, majplus)) / (norm(majplus) * norm(y)), all_majs)
    # majminus = all_majs[findfirst(y -> abs(dot(y, majplus)) < 0.5 * norm(majplus) * norm(y), all_majs)]
    majs = [majplus majminus]
    HM = Hermitian(majs' * majs)
    X = try
        cholesky(HM)
    catch
        vals = eigvals(HM)
        reg = 100 * eps(eltype(vals))
        @warn "Cholesky failed, matrix is not positive definite? eigenvals = $vals. Adding $(reg) * I"
        @debug "Cholesky failed. Input:" _es _ops
        cholesky(HM + reg * I)
    end
    newmajs = majs * inv(X.U) * [1 1; 1im -1im] / sqrt(2)
    # o1 = (newmajs[:, 1] + newmajs[:, 2])
    # o2 = (newmajs[:, 1] - newmajs[:, 2])
    # normalize!(o1)
    # normalize!(o2)
    return newmajs
end

#Symmetric form of SVD for symmetric matrices, Takagi/Autonne decomposition
function symmetric_SVD(M)
    issymmetric(M) || throw(ArgumentError("Matrix must be symmetric"))
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
    size(vs, 2) == 2 || throw(ArgumentError("Matrix must have exactly two columns for SVD canonicalization"))
    vs'vs ≈ I || throw(ArgumentError("Quasiparticles must be orthonormal"))
    n = div(size(vs, 1), 2)
    mat = Symmetric(transpose(vs) * [0I I(n); I(n) 0I] * vs)
    U, D = symmetric_SVD(mat)
    # check D ≈ I ?
    particles = vs * conj(U) * [1 1; 1im -1im] / sqrt(2)
    (particles)
end


@testitem "BdG Canonicalization" begin
    using FermionicHilbertSpaces: canonicalize_particle_pair, SVDCanon, ProjectionCanon, quasiparticle_adjoint, BdGHilbertSpace, BdGEigen, isbdgmatrix
    using LinearAlgebra
    @fermions f
    N = 4
    H = BdGHilbertSpace(1:N)
    ham = sum(rand(ComplexF64) * f[n]'f[k] + rand(ComplexF64) * f[n]'f[k]' + hc for (n, k) in Iterators.product(1:N, 1:N))
    h = matrix_representation(ham, H)
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

## Many body 
function many_body_density_matrix_exp(_G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())
    G = _G - tr(_G)I / size(_G, 1)
    vals, vecs = diagonalize(BdGMatrix(G; check=false), alg)
    clamp_val(e) = clamp(e, -1 / 2 + eps(e), 1 / 2 - eps(e))
    f(e) = log((e + 1 / 2) / (1 / 2 - e))
    vals2 = map(f ∘ clamp_val, vals[1:div(length(vals), 2)])
    H = vecs * Diagonal(vcat(vals2, -reverse(vals2))) * vecs'
    N = length(vals2)
    _H = Hermitian(H[1:N, 1:N])
    Δ = H[1:N, N+1:2N]
    Δ = (Δ - transpose(Δ)) / 2
    @assert _H ≈ -transpose(H[N+1:2N, N+1:2N])
    @assert Δ ≈ -transpose(Δ)
    @assert Δ ≈ -conj(H[N+1:2N, 1:N])
    Hmb = Matrix(many_body_hamiltonian(_H, Δ, c))
    rho = exp(Hmb)
    return rho / tr(rho)
end
const DEFAULT_PH_CUTOFF = 1e-12

# remove_trace(A) = A - tr(A)I / size(A, 1)
"""
    many_body_density_matrix(G, labels, alg=SkewEigenAlg())

Compute the many-body density matrix for a given correlator G. 
"""
function many_body_density_matrix(_G, labels=1:div(size(_G, 2), 2), alg=BdGEigen())
    G = _G - tr(_G)I / size(_G, 1)
    isbdgmatrix(G) || throw(ArgumentError("G must be a BdG matrix."))
    vals, vecs = eigen(G, alg)
    @fermions f
    N = length(labels)
    qps = [sum((n > N ? f[labels[n-N]]' : f[labels[n]]) * vecs[n, i] for n in 1:2N) for i in 1:size(vecs, 2)]
    prod((1 * (1 / 2 - e) + 2e * (qp' * qp)) for (e, qp) in zip(vals[1:N], qps))
end
many_body_density_matrix(_G, H::BdGHilbertSpace, alg=BdGEigen()) = many_body_density_matrix(_G, keys(H), alg)
many_body_density_matrix(_G, H::AbstractHilbertSpace, alg=BdGEigen()) = matrix_representation(many_body_density_matrix(_G, keys(H), alg), H)


"""
    many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))

Construct the many-body Hamiltonian for a given BdG Hamiltonian consisting of hoppings `H` and pairings `Δ`.
"""
# function many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))
#     sum((H[i, j] * c[i]' * c[j] - conj(H[i, j]) * c[i] * c[j]') / 2 - (Δ[i, j] * c[i] * c[j] - conj(Δ[i, j]) * c[i]' * c[j]') / 2 for (i, j) in Base.product(keys(c), keys(c)))
# end

