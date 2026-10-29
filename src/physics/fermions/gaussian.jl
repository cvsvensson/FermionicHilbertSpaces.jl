"""
    correlation_matrix(state, H, Hbdg::BdGHilbertSpace)

The Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩` of a normalized state vector or density matrix `state` on the many-body space `H`, where `Ψ = (f; f')` lists the annihilation and then the creation operators of the modes of `Hbdg` in its order. `expectation(op, G, Hbdg)` then gives `⟨op⟩`. A BdG space over a subset of the modes gives the reduced correlation matrix. On a Majorana space `hilbert_space(γ, labels, SingleParticle())`, `Ψ = γ / √2`, so `G[i, j] = ⟨γ_i γ_j⟩ / 2`.

```jldoctest
julia> using FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions; @fermions f;

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
function correlation_matrix(state, H, Hfree::Union{BdGHilbertSpace, MajoranaSingleParticleHilbertSpace})
    Ψ = adjoint.(_nambu_creators(Hfree))
    return [expectation(Ψi' * Ψj, state, H) for Ψi in Ψ, Ψj in Ψ]
end

"""
    correlation_matrix(F::LinearAlgebra.Eigen)

The Nambu correlation matrix `G[i, j] = ⟨Ψ_i' Ψ_j⟩` of the BdG ground state, where `F` is the result of `eigen(A, BdGEigen())`, assuming no zero-energy modes. It equals `correlation_matrix(ψ, H, Hbdg)` for the many-body ground state `ψ`. For the matrix `R` of a Majorana operator, `F = eigen(Hermitian(R))` gives `G[i, j] = ⟨γ_i γ_j⟩ / 2`.

```jldoctest
julia> using FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions, LinearAlgebra; @fermions f;

julia> A = representation(-f[1]' * f[1], hilbert_space(f, 1:1, BdG()), :dense);

julia> correlation_matrix(eigen(A, BdGEigen())) ≈ [1 0; 0 0]
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
julia> using FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions, LinearAlgebra; @fermions f;

julia> A = representation(-f[1]' * f[1], hilbert_space(f, 1:1, BdG()), :dense);

julia> ground_state_parity(eigen(A, BdGEigen()))
-1
```
"""
function ground_state_parity(F::Eigen)
    N = div(size(F.vectors, 2), 2)
    return _parity(det(F.vectors[:, [2N:-1:(N + 1); 1:N]]))
end
function _parity(d)
    p = sign(real(d))
    isapprox(d, p; atol = sqrt(eps(real(float(one(d)))))) ||
        throw(ArgumentError("The eigenvectors do not pair particles and holes exactly; their determinant is $d."))
    return Int(p)
end

"""
    ground_state_parity(F::LinearAlgebra.Eigen, H)

The fermionic parity of the ground state of a quadratic Majorana operator with matrix `R` on the Majorana space `H`, where `F = eigen(Hermitian(R))`, assuming no zero-energy modes. The parity refers to the fermions formed by consecutive pairs of `labels`, as in `hilbert_space(γ, labels)`; a different label order can flip the sign.
"""
function ground_state_parity(F::Eigen, ::MajoranaSingleParticleHilbertSpace)
    N = div(size(F.vectors, 2), 2)
    # The quasiparticle of a negative-energy column u = (x + iy) / √2 annihilates the ground state, so the Majoranas
    # γ_x = x ⋅ γ and γ_y = y ⋅ γ have i γ_x γ_y = -1 there. The columns x, y form an orthogonal W, and the product of
    # these pair parities is det(W) times the parity.
    W = sqrt(2) * stack(Iterators.flatten((real(u), imag(u)) for u in eachcol(F.vectors[:, 1:N])))
    return _parity((-1)^N * det(W))
end

"""
    gaussian_state(G, H, Hbdg::BdGHilbertSpace)

The density matrix on the many-body space `H` of the Gaussian state with Nambu correlation matrix `G`, so that `correlation_matrix(gaussian_state(G, H, Hbdg), H, Hbdg) ≈ G`. `H` must have exactly the modes of `Hbdg`.

```jldoctest
julia> using FermionicHilbertSpaces, FermionicHilbertSpaces.FreeFermions; @fermions f;

julia> G = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]; # f[1] occupied, f[2] empty

julia> gaussian_state(G, hilbert_space(f, 1:2), hilbert_space(f, 1:2, BdG())) ≈ [0 0 0 0; 0 1 0 0; 0 0 0 0; 0 0 0 0]
true
```
"""
function gaussian_state(G, H, Hbdg::BdGHilbertSpace)
    N = length(modes(Hbdg))
    dim(H) == 2^N || throw(ArgumentError("The many-body space must have exactly the $N modes of the BdG space."))
    # transpose(G)[i, j] = ⟨Ψ_j' Ψ_i⟩, so ⟨γ' γ⟩ = v' * transpose(G) * v for γ' = quasiparticle(v, Hbdg)
    vals, vecs = eigen(transpose(G) - I / 2, BdGEigen())
    tol = sqrt(eps(real(eltype(vals))))
    all(λ -> abs(λ) <= 1 / 2 + tol, vals) ||
        throw(ArgumentError("The occupations of G must lie in [0, 1]."))
    creators = [representation(quasiparticle(vecs[:, k], Hbdg), H) for k in 1:N]
    # Annihilating every quasiparticle in a random vector leaves the vacuum, almost surely nonzero.
    vacuum = normalize(foldl((ψ, c) -> c' * ψ, creators; init = randn(eltype(vecs), dim(H))))
    # ρ = W * W', where the columns of W are the quasiparticle occupation states with nonzero
    # probability, scaled by its square root. The cost scales with the rank of ρ.
    W = reshape(vacuum, :, 1)
    for (c, λ) in zip(creators, vals)
        n = λ + 1 / 2
        n < tol && continue
        W = n > 1 - tol ? c * W : [sqrt(1 - n) * W sqrt(n) * (c * W)]
    end
    return W * W'
end

@testitem "BdG correlation matrix and parity" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
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
        F = eigen(representation(ham, Hbdg, :dense), BdGEigen())
        G = correlation_matrix(F)
        @test G ≈ correlation_matrix(ψ, Hmb, Hbdg) ≈ correlation_matrix(ψ * ψ', Hmb, Hbdg)
        @test expectation(ham, G, Hbdg) ≈ expectation(ham, ψ, Hmb)
        @test ground_state_parity(F) == round(real(dot(ψ, parityoperator(Hmb), ψ)))
        ground_state_parity(F)
    end
    @test parities[1] == 1 && parities[3] == -1
    F = eigen(representation(quadratic, Hbdg, :dense), BdGEigen())
    @test_throws ArgumentError ground_state_parity(Eigen(F.values, 2 * F.vectors))
end

@testitem "Majorana ground state" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
    @majoranas γ
    N = 3
    Hmb = hilbert_space(γ, 1:2N)
    H = hilbert_space(γ, 1:2N, SingleParticle())
    A = randn(2N, 2N)
    quadratic = 1im / 4 * sum((A[i, j] - A[j, i]) * γ[i] * γ[j] for i in 1:2N, j in 1:2N)
    quartic = γ[1] * γ[2] * γ[3] * γ[6] + 2 * γ[2] * γ[4] * γ[5] * γ[1] * γ[3] * γ[6]
    parities = map([-20, 20]) do μ
        ham = quadratic + 1im * μ / 2 * sum(γ[2k - 1] * γ[2k] for k in 1:N)
        vals, vecs = eigen(Hermitian(representation(ham, Hmb, :dense)))
        ψ = vecs[:, 1]
        F = eigen(Hermitian(representation(ham, H, :dense)))
        @test vals[1] ≈ sum(filter(<(0), F.values))
        G = correlation_matrix(F)
        @test G ≈ correlation_matrix(ψ, Hmb, H) ≈ [expectation(γ[i] * γ[j], ψ, Hmb) / 2 for i in 1:2N, j in 1:2N]
        @test expectation(quartic, G, H) ≈ expectation(quartic, ψ, Hmb)
        @test expectation(mean_field(quartic, G, H), G, H) ≈ expectation(quartic, G, H)
        @test ground_state_parity(F, H) == round(real(dot(ψ, parityoperator(Hmb), ψ)))
        ground_state_parity(F, H)
    end
    @test parities == [1, -1]
    F = eigen(Hermitian(representation(quadratic, H, :dense)))
    @test_throws ArgumentError ground_state_parity(Eigen(F.values, 2 * F.vectors), H)
end

@testitem "Gaussian state from a correlation matrix" begin
    using LinearAlgebra
    using FermionicHilbertSpaces.FreeFermions
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
    @test_throws ArgumentError gaussian_state(Diagonal([2; zeros(N - 1); -1; ones(N - 1)]), Hmb, Hbdg)
end
