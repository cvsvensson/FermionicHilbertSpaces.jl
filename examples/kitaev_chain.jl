# # Interacting Kitaev chain

# This tutorial builds an interacting Kitaev chain, finds its parity-sector ground states and
# measures how localized its many-body Majoranas are. It assumes [Getting started](index.md).
# Locality of many-body Majoranas is discussed in Ref. [^1].
#
# [^1]: W. Samuelson *et al.*, [PRX Quantum 7, 020341 (2026)](https://doi.org/10.1103/pw51-tnsz), [arXiv:2510.20538](https://arxiv.org/abs/2510.20538).

using FermionicHilbertSpaces
using Arpack, LinearAlgebra, Plots

# ## Hamiltonian
# `hilbert_space(f, 1:N, ParityConservation())` restricts the space to states of definite total parity.
# The Hamiltonian is a symbolic expression in `f`; `hc` adds the Hermitian conjugate.
@fermions f
N = 12
H = hilbert_space(f, 1:N, ParityConservation())

kitaev_chain(f, N, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:(N-1)) +
    sum(Δ * f[i] * f[i+1] + hc for i in 1:(N-1)) +
    sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:(N-1)) +
    sum(μ[i] * f[i]' * f[i] for i in 1:N)

# The parameters are close to the sweet spot with perfectly localized Majoranas.
U = 4.0
t = 1.0
δΔ = 0.4
Δ = t + U / 2 - δΔ # detuned from the sweet spot
μ = fill(-U, N)
μ[[1, N]] .= -U / 2 # edge chemical potentials
hsym = kitaev_chain(f, N, μ, t, Δ, U)

# ## Ground states in each parity sector
# `representation` turns the symbolic Hamiltonian into a matrix. Since parity is conserved,
# we diagonalize each parity sector separately and `embed` its ground state into the full space `H`.
(Eo, o), (Ee, e) = map(quantumnumbers(H)) do parity
    Hsec = sector(parity, H)
    vals, vecs = eigs(representation(hsym, Hsec); nev=1, which=:SR)
    (; energy=first(vals), ground_state=embed(vecs[:, 1], Hsec => H))
end;
# The two energies are nearly degenerate.
Eo - Ee

# ## Majorana locality
# The ground-state pair defines two many-body Majorana operators, $\gamma = |o\rangle\langle e| + \mathrm{h.c.}$
# and $\tilde\gamma = i|o\rangle\langle e| + \mathrm{h.c.}$, and the parity-difference operator $|o\rangle\langle o| - |e\rangle\langle e|$.
# The norm of the reduction of each to a single site, `partial_trace(x, H => Hsite)`, shows where it is supported.
oe = o * e'
γ = oe + oe'
γ̃ = 1im * oe + (1im * oe)'
δρ = o * o' - e * e'

Hsites = [hilbert_space(f, i:i) for i in 1:N]
locality(x) = [norm(svdvals(partial_trace(x, H => Hsite)), 1) for Hsite in Hsites]

kwargs = (; lw=4, legendfontsize=15, marker=true, markerstrokewidth=2)
plot(xlabel="Site", title="Majorana locality measures"; frame=:box, size=(500, 300),
    xticks=1:3:N, ylims=(-0.1, 2.1), legendposition=:top, labelfontsize=15)
plot!(1:N, locality(γ); label="‖γₙ‖", kwargs...)
plot!(1:N, locality(γ̃); label="‖γ̃ₙ‖", kwargs...)
plot!(1:N, locality(δρ); label="‖(iγγ̃)ₙ‖", kwargs...)
