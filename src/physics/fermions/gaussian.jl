"""
    correlation_matrix(state, H, Hbdg::BdGHilbertSpace)

The Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩` of a normalized state vector or density matrix `state` on the many-body space `H`, where `Ψ = (f; f')` lists the annihilation and then the creation operators of the modes of `Hbdg` in its order. `expectation(op, G, Hbdg)` then gives `⟨op⟩` for a quadratic operator `op`. A BdG space over a subset of the modes gives the reduced correlation matrix.

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
    return [expectation(Ψi' * Ψj, state, H) for Ψi in Ψ, Ψj in Ψ]
end

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

"""
    gaussian_state(G, H, Hbdg::BdGHilbertSpace)

The density matrix on the many-body space `H` of the Gaussian state with Nambu correlation matrix `G`, so that `correlation_matrix(gaussian_state(G, H, Hbdg), H, Hbdg) ≈ G`. `H` must have exactly the modes of `Hbdg`.

```jldoctest
julia> @fermions f;

julia> G = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]; # f[1] occupied, f[2] empty

julia> gaussian_state(G, hilbert_space(f, 1:2), hilbert_space(f, 1:2, BdG())) ≈ [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
true
```
"""
function gaussian_state(G, H, Hbdg::BdGHilbertSpace)
    N = length(modes(Hbdg))
    dim(H) == 2^N || throw(ArgumentError("The many-body space must have exactly the $N modes of the BdG space."))
    # transpose(G)[i, j] = ⟨Ψ_j' Ψ_i⟩, so ⟨γ' γ⟩ = v' * transpose(G) * v for γ' = quasiparticle(v, Hbdg)
    occupations, vecs = eigen(transpose(G) - I / 2, BdGEigen())
    factors = map(1:N) do k
        creator = representation(quasiparticle(vecs[:, k], Hbdg), H, :dense)
        n = occupations[k] + 1 / 2
        n * creator * creator' + (1 - n) * creator' * creator
    end
    return prod(factors)
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
        @test expectation(ham, G, Hbdg) ≈ expectation(ham, ψ, Hmb)
        @test ground_state_parity(F) == round(real(dot(ψ, parityoperator(Hmb), ψ)))
        ground_state_parity(F)
    end
    @test parities[1] == 1 && parities[3] == -1
    vals, vecs = eigen(representation(quadratic, Hbdg, :dense), BdGEigen())
    @test_throws ArgumentError ground_state_parity(Eigen(vals, 2 * vecs))
end

@testitem "Gaussian state from a correlation matrix" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: BdGEigen
    @fermions f
    N = 4
    Hmb = hilbert_space(f, 1:N)
    Hbdg = hilbert_space(f, 1:N, BdG())
    h = rand(ComplexF64, N, N) + hc
    Δ = rand(ComplexF64, N, N)
    Δ = Δ - transpose(Δ)
    ham = sum(f[n]' * h[n, m] * f[m] + Δ[n, m] * f[n]' * f[m]' + hc for n in 1:N, m in 1:N)

    G = correlation_matrix(eigen(representation(ham, Hbdg, :dense), BdGEigen()))
    ψ = eigvecs(Hermitian(representation(ham, Hmb, :dense)))[:, 1]
    for sub in ([1, 3], 1:N)
        Hsub = hilbert_space(f, sub)
        Gsub = G[[sub; sub .+ N], [sub; sub .+ N]]
        @test gaussian_state(Gsub, Hsub, hilbert_space(f, sub, BdG())) ≈ partial_trace(ψ * ψ', Hmb => Hsub)
    end

    ρ = exp(-Hermitian(representation(ham, Hmb, :dense)))
    ρ /= tr(ρ)
    Gthermal = correlation_matrix(ρ, Hmb, Hbdg)
    @test gaussian_state(Gthermal, Hmb, Hbdg) ≈ ρ
end
