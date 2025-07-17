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
    h = m[1:n÷2, 1:n÷2]
    δ = m[1:n÷2, n÷2+1:end]
    hd = m[n÷2+1:end, n÷2+1:end]
    δd = m[n÷2+1:end, 1:n÷2]
    h = (h - conj(hd)) / 2
    Δ = (δ - transpose(δd)) / 2
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


function many_body_density_matrix_exp(G, c=FermionBasis(1:div(size(G, 1), 2), qn=parity); alg=SkewEigenAlg())
    G = remove_trace(G)
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
function enforce_ph_symmetry(_es, _ops; cutoff=DEFAULT_PH_CUTOFF, energysort=identity)
    p = sortperm(_es, by=energysort)
    es = _es[p]
    ops = complex(_ops[:, p])
    N = div(length(es), 2)
    ph = quasiparticle_adjoint
    for k in Iterators.take(eachindex(es), N)
        k2 = quasiparticle_adjoint_index(k, N)
        if es[k] > cutoff && isapprox(es[k], -es[k2], atol=cutoff)
            @warn "es[k] = $(es[k]) != $(-es[k2]) = -es[k_adj]"
        end
        op = ops[:, k]
        op_ph = ph(op)
        if abs(dot(op_ph, op)) < cutoff #op is not a majorana
            ops[:, k2] = op_ph
        else #it is at least a little bit of majorana
            op2 = ops[:, k2]
            majplus = begin
                v = ph(op) + op
                if norm(v) > cutoff
                    v
                else
                    1im * (ph(op) - op)
                end
            end
            majminus = begin
                v = ph(op2) - op2
                if norm(v) > cutoff && abs(dot(v, majplus)) < norm(majplus)^2
                    v
                else
                    1im * (ph(op2) + op2)
                end
            end
            majs = [majplus majminus]
            if !all(norm.(eachcol(majs)) .> cutoff)
                @warn "Norm of majoranas = $(norm.(eachcol(majs)))"
                @debug "Norm of majoranas is small. Majoranas:" majs
            end
            HM = Hermitian(majs' * majs)
            X = try
                cholesky(HM)
            catch
                vals = eigvals(HM)
                @warn "Cholesky failed, matrix is not positive definite? eigenvals = $vals. Adding $cutoff * I"
                @debug "Cholesky failed. Input:" _es _ops
                cholesky(HM + cutoff * I)
            end
            newmajs = majs * inv(X.U)
            if !(newmajs' * newmajs ≈ I)
                @warn "New majoranas are not orthogonal? $(norm(newmajs' * newmajs - I))"
                @debug "New majoranas are not orthogonal? New majoranas:" newmajs
            end
            o1 = (newmajs[:, 1] + newmajs[:, 2])
            o2 = (newmajs[:, 1] - newmajs[:, 2])
            normalize!(o1)
            normalize!(o2)
            if abs(dot(o1, op)) > abs(dot(o2, op))
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
function enforce_ph_symmetry(F::Eigen; cutoff=DEFAULT_PH_CUTOFF)
    if isreal(F.values)
        enforce_ph_symmetry(real(F.values), F.vectors; cutoff)
    else
        throw(ArgumentError("Eigenvalues must be real"))
    end
end

"""
    many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))

Construct the many-body Hamiltonian for a given BdG Hamiltonian consisting of hoppings `H` and pairings `Δ`.
"""
# function many_body_hamiltonian(H::AbstractMatrix, Δ::AbstractMatrix, c::FermionBasis=FermionBasis(1:size(H, 1), qn=parity))
#     sum((H[i, j] * c[i]' * c[j] - conj(H[i, j]) * c[i] * c[j]') / 2 - (Δ[i, j] * c[i] * c[j] - conj(Δ[i, j]) * c[i]' * c[j]') / 2 for (i, j) in Base.product(keys(c), keys(c)))
# end
