# # Majorana nanowire
# A spinful wire with Rashba spin-orbit coupling, a Zeeman field ``V_z`` and s-wave pairing ``Δ`` becomes a
# topological superconductor between two critical fields, with Majorana zero modes at its ends. The model is
# quadratic, so `BdG()` turns it into a BdG matrix, and `BdGEigen` gives its particle-hole symmetric spectrum.
using FermionicHilbertSpaces
using FermionicHilbertSpaces.FreeFermions
using Plots
using LinearAlgebra

function ham(c, N; μ, t, Δ, Vz, α)
    h = 0
    for j in 1:N
        h += -μ * (c[j, :↑]' * c[j, :↑] + c[j, :↓]' * c[j, :↓])
        h += Vz / 2 * (c[j, :↑]' * c[j, :↑] - c[j, :↓]' * c[j, :↓])
        h += Δ * c[j, :↑]' * c[j, :↓]' + hc
    end
    for j in 1:(N - 1)
        h += -t * (c[j + 1, :↑]' * c[j, :↑] + c[j + 1, :↓]' * c[j, :↓]) + hc
        ## Spin-orbit coupling i α c†_{j+1} σ_y c_j
        h += α * (c[j + 1, :↑]' * c[j, :↓] - c[j + 1, :↓]' * c[j, :↑]) + hc
    end
    return h
end

@fermions c
N = 40
t = 1.0
μ = -2.0
Δ = 0.5
α = 0.5
Hbdg = hilbert_space(c, [(n, s) for n in 1:N for s in (:↑, :↓)], BdG())

# The bulk gap closes at ``k = 0`` and ``k = π`` when ``V_z/2 = \sqrt{(μ ± 2t)^2 + Δ^2}``.
Vz_c = 2sqrt((μ + 2t)^2 + Δ^2)
Vz_c2 = 2sqrt((μ - 2t)^2 + Δ^2)

# ## Spectrum versus Zeeman field
# The BdG matrix is half the Nambu Hamiltonian, so the quasiparticle energies are twice its eigenvalues.
# Between the critical fields (dashed) two levels stay close to zero energy.
Vzs = range(0.0, Vz_c2 * 1.2, length = 40)
energies = stack(Vzs) do Vz
    vals, _ = eigen(representation(ham(c, N; μ, t, Δ, Vz, α), Hbdg, Matrix), BdGEigen())
    2vals
end
p1 = plot(Vzs ./ t, energies' ./ t; legend = false, color = :blue, xlabel = "Vz/t", ylabel = "E/t", ylims = (-0.4, 0.4), frame = :box)
vline!(p1, [Vz_c, Vz_c2] ./ t; color = :red, linestyle = :dash)

# ## Spatial profile of the lowest mode
# With `nmodes = 2N` modes (two spins per site), column `nmodes` of the eigenvectors is the non-positive energy
# closest to zero. Its weight on a site sums the particle and hole components of both spins. In the topological
# phase it sits at the ends of the wire, while in the trivial phase it spreads through the bulk.
nmodes = 2N
densities = map((0.5Vz_c, 1.2Vz_c)) do Vz
    _, vecs = eigen(representation(ham(c, N; μ, t, Δ, Vz, α), Hbdg, Matrix), BdGEigen())
    spin_weights = map(x -> sum(abs2, x), Iterators.partition(vecs[:, nmodes], 2))
    spin_weights[1:N] .+ spin_weights[(N + 1):end]
end
p2 = plot(1:N, densities[1]; label = "Trivial", marker = :circle, xlabel = "Site", ylabel = "Weight", frame = :box)
plot!(p2, 1:N, densities[2]; label = "Topological", marker = :square)
