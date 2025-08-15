struct NambuState <: AbstractBasisState
    state::SingleParticleState
    hole::Bool
end
NambuState(i::Integer, hole::Bool) = NambuState(SingleParticleState(i), hole)
Base.:(==)(n1::NambuState, n2::NambuState) = n1.state == n2.state && n1.hole == n2.hole
Base.hash(n::NambuState, h::UInt) = hash(n.hole, hash(n.state, h))

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
    δd = m[n÷2+1:end, 1:n÷2]
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
bdg_hilbert_space(labels) = BdGHilbertSpace(labels)
Base.size(h::BdGHilbertSpace) = size(h.parent)
Base.size(h::BdGHilbertSpace, dim) = size(h.parent, dim)
mode_ordering(h::BdGHilbertSpace) = mode_ordering(h.parent)
modes(H::BdGHilbertSpace) = modes(H.parent)
Base.keys(h::BdGHilbertSpace) = keys(h.parent)
basisstates(h::BdGHilbertSpace) = basisstates(h.parent)

function matrix_representation(op, H::BdGHilbertSpace)
    isquadratic(op) || throw(ArgumentError("Operator must be quadratic in fermions to be represented on a BdG Hilbert space."))
    normal_order_to_bdg(matrix_representation(remove_identity(op), H.parent))
end

function operator_inds_amps!((outinds, ininds, amps), op, ordering, states::AbstractVector{NambuState}, fock_to_ind)
    isquadratic(op) && return operator_inds_amps_bdg!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
    return operator_inds_amps_generic!((outinds, ininds, amps), op, ordering, states, fock_to_ind)
end

function operator_inds_amps_bdg!((outinds, ininds, amps), op::FermionMul, ordering, states, fock_to_ind)
    if length(op.factors) != 2
        throw(ArgumentError("Only two-fermion operators supported for free fermions"))
    end
    nambustates = (NambuState(getindex(ordering, op.factors[1].label), op.factors[1].creation),
        NambuState(getindex(ordering, op.factors[2].label), !op.factors[2].creation))
    inind = fock_to_ind[nambustates[2]]
    outind = fock_to_ind[nambustates[1]]
    push!(outinds, outind)
    push!(ininds, inind)
    push!(amps, op.coeff)
    return (outinds, ininds, amps)
end

@testitem "BdG" begin
    @fermions f
    h = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
    H = bdg_hilbert_space(1:2)
    @test matrix_representation(h + 1, H) == matrix_representation(h, H)

    h = rand(ComplexF64, 2, 2) + hc
    Δ = rand(ComplexF64, 2, 2)
    Δ = Δ - transpose(Δ)
    bdg_ham = [h/2 Δ/2
        -conj(Δ)/2 -conj(h / 2)]
    op = sum(f[n]' * h[n, m] * f[m] for n in 1:2, m in 1:2) +
         sum(f[n]' * Δ[n, m] * f[m]' / 2 + hc for n in 1:2, m in 1:2)
    bdg_ham2 = matrix_representation(op, H)
    @test bdg_ham ≈ bdg_ham2
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
    vs * conj(U) * [1 1; 1im -1im] / sqrt(2)
end


@testitem "BdG Canonicalization" begin
    using FermionicHilbertSpaces: canonicalize_particle_pair, SVDCanon, ProjectionCanon, quasiparticle_adjoint, BdGEigen, isbdgmatrix
    using LinearAlgebra
    @fermions f
    N = 4
    H = bdg_hilbert_space(1:N)
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

const DEFAULT_PH_CUTOFF = 1e-12

