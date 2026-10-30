# # Anderson impurity

# Creation and annihilation operators change the particle number, so they are rectangular maps between sectors, and
# `representation(op, Axes(Hout, Hin))` returns exactly that map. We use it to build the Lehmann sum for the Green's
# function of a single-impurity Anderson model[^1] and watch the impurity level split into Hubbard peaks as the interaction grows.

using FermionicHilbertSpaces, LinearAlgebra, Plots
import FermionicHilbertSpaces: Axes

# The impurity is site `0`, coupled with strength `V` to `L` bath sites with energies `ε`. Spin-up and spin-down particle
# numbers are conserved separately, so the space splits into sectors labelled by `[N↑, N↓]`. At half filling the ground state lies in the sector `q0`, with `(L + 1) ÷ 2` particles of each spin.

@fermions c
L = 5
ε = range(-1, 1; length = L)
V = 0.25
q0 = [L + 1, L + 1] .÷ 2
n(i, σ) = c[i, σ]' * c[i, σ]
spins = (:↑, :↓)
siam(U) = U * (n(0, :↑) - 1 / 2) * (n(0, :↓) - 1 / 2) +
    sum(ε[k] * n(k, σ) for k in 1:L, σ in spins) +
    sum(V * c[0, σ]' * c[k, σ] + hc for k in 1:L, σ in spins)

Hup = hilbert_space(c, [(i, :↑) for i in 0:L])
Hdn = hilbert_space(c, [(i, :↓) for i in 0:L])
H = tensor_product(Hup, Hdn; constraint = NumberConservation(Hup) * NumberConservation(Hdn))


# Creating or annihilating a spin-up impurity particle on the ground state `|0⟩` leads
# to the neighbouring sectors, and the Green's function is a sum of poles over their eigenstates `|m⟩`,
# ```math
# G(z) = \sum_m \frac{|⟨m|d^†_↑|0⟩|^2}{z - (E_m - E_0)} + \sum_m \frac{|⟨m|d_↑|0⟩|^2}{z + (E_m - E_0)}.
# ```
function lehmann(ham, H, q0)
    eig(q) = eigen(Hermitian(representation(ham, sector(q, H), :dense)))
    H0, Hp, Hm = sector(q0, H), sector(q0 + [1, 0], H), sector(q0 - [1, 0], H)
    E0s, V0s = eig(q0)
    E0, ψ0 = E0s[1], V0s[:, 1]
    Ep, Vp = eig(q0 + [1, 0])
    Em, Vm = eig(q0 - [1, 0])
    wp = abs2.(Vp' * (representation(c[0, :↑]', Axes(Hp, H0)) * ψ0))
    wm = abs2.(Vm' * (representation(c[0, :↑], Axes(Hm, H0)) * ψ0))
    return (; poles = [Ep .- E0; E0 .- Em], weights = [wp; wm])
end
G(z, (; poles, weights)) = sum(weights ./ (z .- poles))

# Without interaction the impurity Green's function is known in closed form, and the Lehmann sum reproduces it.
G0(z) = 1 / (z - sum(V^2 ./ (z .- ε)))
zs = range(-3, 3; length = 101) .+ 0.1im
sol0 = lehmann(siam(0), H, q0)
@assert [G(z, sol0) for z in zs] ≈ G0.(zs)

# The spectral function is `A(ω) = -Im G(ω + iη)/π`. With interaction, the weight moves from the level near zero
# into two Hubbard peaks that approach `±U/2` (dashed lines) as `U` grows. Each curve is shifted up by one.
ωs = range(-6, 6; length = 1201)
η = 0.15
plt = plot(;
    xlabel = "ω", ylabel = "A(ω), shifted", yticks = false, legend = false, frame = :box, size = (500, 400),
    title = "Impurity spectral function"
)
for (i, U) in enumerate((0, 3, 6, 9))
    sol = lehmann(siam(U), H, q0)
    plot!(plt, ωs, [i - 1 - imag(G(ω + im * η, sol)) / π for ω in ωs]; lw = 2, color = i)
    U > 0 && plot!(plt, [-U, -U, NaN, U, U] ./ 2, i .- [1, 0, NaN, 1, 0]; color = :gray, ls = :dash)
    annotate!(plt, -6, i - 0.75, text(" U = $U", 9, :left))
end
plt

# [^1]: P. W. Anderson, "Localized Magnetic States in Metals", Phys. Rev. 124, 41 (1961).
