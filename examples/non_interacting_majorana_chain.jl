using FermionicHilbertSpaces
using Plots
using LinearAlgebra

function ham(c, N; μ, t, Δ, Vz, α)
    ham = 0
    # On-site terms
    for j in 1:N
        ham += -μ * (c[j, :↑]' * c[j, :↑] + c[j, :↓]' * c[j, :↓])
        ham += Vz / 2 * (c[j, :↑]' * c[j, :↑] - c[j, :↓]' * c[j, :↓])
        ham += Δ * c[j, :↑]' * c[j, :↓]' + hc
    end
    # Hopping terms
    for j in 1:(N-1)
        # Kinetic energy
        ham += -t * (c[j+1, :↑]' * c[j, :↑] + c[j+1, :↓]' * c[j, :↓]) + hc
        # Spin-orbit coupling: i*α * c†_{j+1} σ_y c_j
        ham += α * (c[j+1, :↑]' * c[j, :↓] - c[j+1, :↓]' * c[j, :↑]) + hc
    end
    return ham
end
##
@fermions c
N = 40
t = 1.0
μ = -2.0
Δ = 0.5
α = 0.5
Vz_c = 2sqrt((μ + 2t)^2 + Δ^2)
Vz_c2 = 2sqrt((μ - 2t)^2 + Δ^2)
Hbdg = bdg_hilbert_space(c, [(n, s) for n in 1:N for s in (:↑, :↓)])

## Scan Zeeman field
Vzs = range(0.0, Vz_c2 * 1.2, length=40)
energies = stack(Vzs) do Vz
    symham = ham(c, N; μ, t, Δ, Vz, α)
    matham = matrix_representation(symham, Hbdg)
    vals, _ = eigen(Matrix(matham), FermionicHilbertSpaces.BdGEigen())
    vals
end

# Plot spectrum vs Vz
p1 = plot(legend=false, xlabel="Vz", ylabel="Energy", title="Topological Phase Transition", ylims=0.2 .* (-1, 1))
plot!(p1, Vzs, energies', color=:blue, alpha=0.9, lw=1)
vline!(p1, [Vz_c], color=:red, linestyle=:dash, label="Vz_c")
vline!(p1, [Vz_c2], color=:red, linestyle=:dash, label="Vz_c2")
display(p1)
##
Vzs = [Vz_c * 0.5, Vz_c * 1.2]  # Trivial and topological
densities = map(Vzs) do Vz
    symham = ham(c, N; μ, t, Δ, Vz, α)
    matham = matrix_representation(symham, Hbdg)
    vals, vecs = eigen(Matrix(matham), FermionicHilbertSpaces.BdGEigen())
    # Each colum of vecs is a quasiparticle wavefunction, and particle-hole symmetry ensures has been enforced by using FermionicHilbertSpaces.BdGEigen.
    low_energy_mode = vecs[:, 2N]
    spin_avg = map(norm, Iterators.partition(low_energy_mode, 2))
    density = abs2.(spin_avg[1:N]) .+ abs2.(spin_avg[N+1:end])
end

# Plot spatial profiles
p2 = plot(1:N, densities[1], label="Trivial", marker=:circle)
plot!(p2, 1:N, densities[2], label="Topological", marker=:square)
xlabel!("Site")
ylabel!("Probability density")
title!("Spatial profile of lowest energy mode")
display(p2)