# Fig. 1 of Shapourian & Ryu, arXiv:1804.08637 (PRB 98, 075102):
# logarithmic negativity of the two-mode Werner state with the
# fermionic vs. the bosonic partial transpose.

using FermionicHilbertSpaces
using LinearAlgebra, Plots

@fermions f
H = hilbert_space(f, 1:2)     # mode 1 ∈ A, mode 2 ∈ B
HA = hilbert_space(f, [1])

# Singlet |Ψs⟩ = (f₁† − f₂†)|0⟩/√2, Eq. (80)
ψs = representation(Ket("01", H) - Ket("10", H)) / √2

werner(p) = (1 - p)/4 * I + p * (ψs * ψs')           # Eq. (82)

RA = partial_transpose(H, HA)                        # fermionic R_A
TA = partial_transpose(H, HA; phase_factors=false) # ordinary T_A
logneg(ρ) = log(sum(svdvals(Matrix(ρ))))             # Eq. (31)

E_fermion(p) = logneg(RA(werner(p)))
E_boson(p) = logneg(TA(werner(p)))

E_fermion_exact(p) = log((1 + p)/2 + sqrt(5p^2 - 2p + 1)/2)  # Eq. (84)
E_boson_exact(p) = log(3(1 + p)/4 + abs(1 - 3p)/4)         # Eq. (83)

ps = range(0, 1, length=201)
fig = plot(ps, E_fermion.(ps) ./ log(2); label="Fermionic", lw=4,
    xlabel="p", ylabel="E / log 2", title="Two-mode Werner state negativity",
    xlims=(0, 1), ylims=(0, 1.02), frame=:box, legend=:topleft, size=(500, 250))
plot!(fig, ps, E_boson.(ps) ./ log(2); label="Bosonic", lw=4)
plot!(fig, ps, E_fermion_exact.(ps) ./ log(2); label="Eq. (84)", ls=:dot, color=:black, lw=3)
plot!(fig, ps, E_boson_exact.(ps) ./ log(2); label="Eq. (83)", ls=:dash, color=:black, lw=2)
display(fig)
