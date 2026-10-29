"""
    correlation_matrix(state, H, Hbdg::BdGHilbertSpace)

The Nambu correlation matrix `Γ[i, j] = ⟨Ψ[j]' * Ψ[i]⟩` of a normalized state vector or density matrix `state` on the many-body space `H`, where `Ψ = [f; f']` lists the annihilation and then the creation operators of the modes of `Hbdg` in its mode order. For a quadratic operator with BdG matrix `A = representation(op, Hbdg)`, `⟨op⟩ = tr(A * Γ)` up to a constant. A BdG space over a subset of the modes gives the reduced correlation matrix.

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
    return [_expectation(state, representation(Ψj' * Ψi, H)) for Ψi in Ψ, Ψj in Ψ]
end
_expectation(ψ::AbstractVector, op) = dot(ψ, op, ψ)
_expectation(ρ::AbstractMatrix, op) = tr(op * ρ)

"""
    correlation_matrix(vecs)

The Nambu correlation matrix of the BdG ground state, the quasiparticle vacuum, from the eigenvectors `vecs` returned by `eigen(A, BdGEigen())`: the projector onto the negative-energy eigenvectors. It equals `correlation_matrix(ψ, H, Hbdg)` for the many-body ground state `ψ`.
"""
function correlation_matrix(vecs)
    N = div(size(vecs, 2), 2)
    U = vecs[:, 1:N]
    return U * U'
end

"""
    ground_state_parity(vecs)

The fermionic parity, `1` or `-1`, of the BdG ground state from the eigenvectors `vecs` returned by `eigen(A, BdGEigen())`. It equals the parity of the many-body ground state.
"""
function ground_state_parity(vecs)
    N = div(size(vecs, 2), 2)
    return Int(sign(real(det(vecs[:, [2N:-1:N+1; 1:N]]))))
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
    parities = map([-6, 0, 6]) do μ
        ham = quadratic - μ * sum(f[n]' * f[n] for n in 1:N)
        ψ = eigvecs(Hermitian(representation(ham, Hmb, :dense)))[:, 1]
        _, vecs = eigen(representation(ham, Hbdg, :dense), BdGEigen())
        @test correlation_matrix(vecs) ≈ correlation_matrix(ψ, Hmb, Hbdg) ≈ correlation_matrix(ψ * ψ', Hmb, Hbdg)
        @test ground_state_parity(vecs) ≈ dot(ψ, parityoperator(Hmb), ψ)
        ground_state_parity(vecs)
    end
    @test parities[1] == 1 && parities[3] == -1
end
