"""
    correlation_matrix(state, H, Hbdg::BdGHilbertSpace)

The Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩` of a normalized state vector or density matrix `state` on the many-body space `H`, where `Ψ = (f; f')` lists the annihilation and then the creation operators of the modes of `Hbdg` in its order. A hermitian quadratic operator `op` without identity part, with `A = representation(op, Hbdg)` and `N` modes, has `⟨op⟩ = sum(A .* G) + tr(A[1:N, 1:N])`. A BdG space over a subset of the modes gives the reduced correlation matrix.

```jldoctest
julia> @fermions f;

julia> H = hilbert_space(f, 1:2);

julia> ψ = [0, 1, 0, 0]; # f[1]' acting on the vacuum

julia> correlation_matrix(ψ, H, hilbert_space(f, 1:2, BdG()))
4×4 Matrix{Int64}:
 1  0  0  0
 0  0  0  0
 0  0  0  0
 0  0  0  1
```
"""
function correlation_matrix(state, H, Hbdg::BdGHilbertSpace)
    c = modes(Hbdg)
    Ψ = [c; adjoint.(c)]
    return [_expectation(state, representation(Ψi' * Ψj, H)) for Ψi in Ψ, Ψj in Ψ]
end
_expectation(ψ::AbstractVector, op) = dot(ψ, op, ψ)
_expectation(ρ::AbstractMatrix, op) = dot(ρ', op)

"""
    correlation_matrix(F::LinearAlgebra.Eigen)

The Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩` of the BdG ground state, where `F` is the result of `eigen(A, BdGEigen())`, assuming no zero-energy modes. It equals `correlation_matrix(ψ, H, Hbdg)` for the many-body ground state `ψ`.

```jldoctest
julia> using LinearAlgebra; @fermions f;

julia> A = representation(-f[1]' * f[1], hilbert_space(f, 1:1, BdG()), :dense);

julia> correlation_matrix(Eigen(eigen(A, FermionicHilbertSpaces.BdGEigen())...)) ≈ [1 0; 0 0]
true
```
"""
function correlation_matrix(F::Eigen)
    N = div(size(F.vectors, 2), 2)
    U = F.vectors[:, 1:N]
    return conj(U) * transpose(U)
end

"""
    ground_state_parity(F::LinearAlgebra.Eigen)

The fermionic parity, `1` or `-1`, of the BdG ground state, where `F` is the result of `eigen(A, BdGEigen())`, assuming no zero-energy modes. It equals the parity of the many-body ground state.

```jldoctest
julia> using LinearAlgebra; @fermions f;

julia> A = representation(-f[1]' * f[1], hilbert_space(f, 1:1, BdG()), :dense);

julia> ground_state_parity(Eigen(eigen(A, FermionicHilbertSpaces.BdGEigen())...))
-1
```
"""
function ground_state_parity(F::Eigen)
    N = div(size(F.vectors, 2), 2)
    d = det(F.vectors[:, [2N:-1:(N + 1); 1:N]])
    p = sign(real(d))
    isapprox(d, p; atol = sqrt(eps(real(float(one(d)))))) ||
        throw(ArgumentError("The eigenvectors do not pair particles and holes exactly; their determinant is $d."))
    return Int(p)
end

@testitem "BdG correlation matrix and parity" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: BdGEigen
    @fermions f
    N = 3
    Hmb = hilbert_space(f, 1:N)
    Hbdg = hilbert_space(f, 1:N, BdG())
    h = rand(ComplexF64, N, N) + hc
    Δ = rand(ComplexF64, N, N)
    Δ = Δ - transpose(Δ)
    quadratic = sum(f[n]' * h[n, m] * f[m] + Δ[n, m] * f[n]' * f[m]' + hc for n in 1:N, m in 1:N)
    parities = map([-20, 0, 20]) do μ
        ham = quadratic - μ * sum(f[n]' * f[n] for n in 1:N)
        ψ = eigvecs(Hermitian(representation(ham, Hmb, :dense)))[:, 1]
        vals, vecs = eigen(representation(ham, Hbdg, :dense), BdGEigen())
        F = Eigen(vals, vecs)
        G = correlation_matrix(F)
        @test G ≈ correlation_matrix(ψ, Hmb, Hbdg) ≈ correlation_matrix(ψ * ψ', Hmb, Hbdg)
        A = representation(ham, Hbdg, :dense)
        @test sum(A .* G) + tr(A[1:N, 1:N]) ≈ dot(ψ, representation(ham, Hmb, :dense), ψ)
        @test ground_state_parity(F) == round(real(dot(ψ, parityoperator(Hmb), ψ)))
        ground_state_parity(F)
    end
    @test parities[1] == 1 && parities[3] == -1
    vals, vecs = eigen(representation(quadratic, Hbdg, :dense), BdGEigen())
    @test_throws ArgumentError ground_state_parity(Eigen(vals, 2 * vecs))
end
