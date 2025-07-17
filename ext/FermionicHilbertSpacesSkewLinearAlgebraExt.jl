module FermionicHilbertSpacesSkewLinearAlgebraExt

using FermionicHilbertSpaces, FermionicHilbertSpacesSkewLinearAlgebraExt

function bdg_to_skew(B::BdGMatrix; check=true)
    bdg_to_skew(B.H, B.Δ; check)
end
function bdg_to_skew(H::AbstractMatrix, Δ::AbstractMatrix; check=true)
    N = size(H, 1)
    T = real(promote_type(eltype(H), eltype(Δ)))
    A = zeros(T, 2N, 2N)
    for i in 1:N, j in 1:N
        A[i, j] = imag(H[i, j] + Δ[i, j])
        A[i+N, j] = real(H[i, j] + Δ[i, j])
        A[j, i+N] = -A[i+N, j]
        A[i+N, j+N] = imag(H[i, j] - Δ[i, j])
    end
    if check
        return SkewHermitian(A)
    else
        return skewhermitian!(A)
    end
end
bdg_to_skew(bdgham::AbstractMatrix; check=true) = bdg_to_skew(BdGMatrix(bdgham; check); check)

function skew_to_bdg(A::AbstractMatrix)
    BdGMatrix(_skew_to_bdg(A)...)
end
function _skew_to_bdg(A::AbstractMatrix)
    N = div(size(A, 1), 2)
    T = complex(eltype(A))
    H = zeros(T, N, N)
    Δ = zeros(T, N, N)
    for i in 1:N, j in i:N
        H[i, j] = (A[i+N, j] - A[i, j+N] + 1im * (A[i, j] + A[i+N, j+N])) / 2
        H[j, i] = conj(H[i, j])
        Δ[i, j] = (A[i+N, j] + A[i, j+N] + 1im * (A[i, j] - A[i+N, j+N])) / 2
        Δ[j, i] = -Δ[i, j]
        if i == j
            Δ[j, j] = 0
        end
    end
    return Hermitian(H), Δ
end

function skew_to_bdg(v::AbstractVector)
    N = div(length(v), 2)
    T = complex(eltype(v))
    uv = zeros(T, 2N)
    for i in 1:N
        uv[i] = (v[i] - 1im * v[i+N]) / sqrt(2)
        uv[i+N] = (v[i] + 1im * v[i+N]) / sqrt(2)
    end
    return uv
end

function skew_eigen_to_bdg(_es, ops)
    es = imag(-_es)
    pair_itr = collect(Iterators.partition(es, 2)) #take each eigenpair
    p = sortperm(pair_itr, by=Base.Fix1(-, 0) ∘ abs ∘ first) #permutation to sort the pairs
    internal_p = map(pair -> sortperm(pair), pair_itr[p]) #permutation within each pair
    pinds = vcat([2p - 1 + pp[1] - 1 for (p, pp) in zip(p, internal_p)],
        reverse([2p - 1 + pp[2] - 1 for (p, pp) in zip(p, internal_p)])) #puts all negative energies first and positive energies at the adjoint indices
    H = stack(skew_to_bdg, eachcol(ops))
    return es[pinds], H[:, pinds]
end

function diagonalize(A::AbstractMatrix, alg::SkewEigenAlg)
    es, ops = eigen(bdg_to_skew(A))
    enforce_ph_symmetry(skew_eigen_to_bdg(es, ops)...; cutoff=alg.cutoff)
end



end
