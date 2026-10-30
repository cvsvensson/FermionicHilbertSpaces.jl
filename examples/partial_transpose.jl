# # Entanglement and negativity
#
# This tutorial computes reduced density matrices, entanglement entropy and the logarithmic
# negativity of fermionic states. It assumes the partial trace and partial transpose conventions
# of [Fermionic tensor products and partial traces](fermions.md). States must have definite
# parity, or be mixtures of parity eigenstates (superselection).
#
# We follow Fig. 1 of Shapourian and Ryu, [arXiv:1804.08637](https://arxiv.org/abs/1804.08637):
# two modes, with mode 1 in subsystem `A` and mode 2 in `B`.
using FermionicHilbertSpaces
import FermionicHilbertSpaces: logarithmic_negativity
using LinearAlgebra

@fermions f
H = hilbert_space(f, 1:2)
HA = hilbert_space(f, [1])

# ## Reduced state and entropy
# The singlet ``(f_1^\dagger - f_2^\dagger)|0\rangle/\sqrt2`` is a pure state with odd parity.
# Its reduced density matrix on `A` is maximally mixed, so the entropy is ``\log 2``.
ψ = representation(Ket("01", H) - Ket("10", H)) / √2
ρ = ψ * ψ'
ρA = partial_trace(ρ, H => HA)
entropy(ρ) = -sum(λ * log(λ) for λ in eigvals(Hermitian(ρ)) if λ > 1.0e-12)
entropy(ρA) ≈ log(2)

# ## Logarithmic negativity
# `logarithmic_negativity(ρ, H, HA)` uses the fermionic partial transpose. For a pure state
# it equals the Rényi-1/2 entropy, here again ``\log 2``.
logarithmic_negativity(ρ, H, HA) ≈ log(2)

# ## Fermionic versus bosonic partial transpose
# The two-mode Werner state mixes the singlet with the identity.
werner(p) = (1 - p) / 4 * I + p * ρ

# With `phase_factors=false` the partial transpose is the ordinary matrix transpose on `A`,
# which gives a different negativity.
E_fermion(p) = logarithmic_negativity(werner(p), H, HA)
E_boson(p) = logarithmic_negativity(werner(p), H, HA; phase_factors = false)

# Both agree with the closed forms, Eqs. (84) and (83) of the reference.
E_fermion_exact(p) = log((1 + p) / 2 + sqrt(5p^2 - 2p + 1) / 2)
E_boson_exact(p) = log(3(1 + p) / 4 + abs(1 - 3p) / 4)
ps = range(0, 1, length = 21)
maximum(abs, E_fermion.(ps) .- E_fermion_exact.(ps)), maximum(abs, E_boson.(ps) .- E_boson_exact.(ps))

# The fermionic negativity is nonzero for every `p > 0`, while the bosonic one vanishes
# for `p ≤ 1/3`.
E_fermion(0.2), E_boson(0.2)
