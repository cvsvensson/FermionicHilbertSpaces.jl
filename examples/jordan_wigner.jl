# # Jordan-Wigner transformation

# This tutorial builds the same chain as spins and as fermions, checks that their many-body spectra agree,
# and then reads the ground-state energy off the single-particle spectrum.
# It assumes you have read the Home page.

using FermionicHilbertSpaces
using LinearAlgebra
using FermionicHilbertSpaces.FreeFermions

# ## The XX chain as spins
# Spin operators are the usual ones, normalised so that `S[k][:x]^2 == 1/4` for spin 1/2.
# The open XX chain is ``H = \sum_k (S^x_k S^x_{k+1} + S^y_k S^y_{k+1})``.
N = 8
@spins S 1 // 2
Hspin = hilbert_space(S, 1:N)
xx = sum(S[k][:x] * S[k+1][:x] + S[k][:y] * S[k+1][:y] for k in 1:(N-1))
Espin = eigvals(Hermitian(representation(xx, Hspin, :dense)))

# ## The same model as fermions
# Since ``S^x_k S^x_{k+1} + S^y_k S^y_{k+1} = (S^+_k S^-_{k+1} + S^-_k S^+_{k+1})/2``, the chain only flips neighbouring spins. With ``S^+_k = c_k^\dagger \prod_{j<k} (1-2c_j^\dagger c_j)``, each flip ``S^+_k S^-_{k+1}`` becomes the hopping ``c_k^\dagger c_{k+1}``.
# The strings of neighbouring sites cancel up to ``1-2c_k^\dagger c_k``, which ``c_k^\dagger`` absorbs. Only a bond from site N back to site 1 would keep a string (the parity of sites 1 to N-1). So the open chain maps exactly onto free fermions, while a ring does not.
@fermions c
Hferm = hilbert_space(c, 1:N)
hopping = sum((c[k]' * c[k+1] + hc) / 2 for k in 1:(N-1))
Eferm = eigvals(Hermitian(representation(hopping, Hferm, :dense)))
@assert Espin ≈ Eferm

# ## Fixed magnetisation and fixed particle number
# Spin up is an occupied site, so total magnetisation ``M = \sum_k S^z_k`` corresponds to ``M + N/2`` fermions.
# Restricting both spaces to the half-filled sector gives the same spectrum again.
Hspin_half = hilbert_space(S, 1:N, AdditiveConstraint(0; maps=s -> s.m))
Hferm_half = hilbert_space(c, 1:N, NumberConservation(N ÷ 2))
E_half_spin = eigvals(Hermitian(representation(xx, Hspin_half, :dense)))
E_half_ferm = eigvals(Hermitian(representation(hopping, Hferm_half, :dense)))
@assert E_half_spin ≈ E_half_ferm

# ## The single-particle picture
# The hopping Hamiltonian is quadratic, so its single-particle matrix `h` determines everything.
# `SingleParticle()` gives a space with one basis state per site, on which `representation` returns `h` itself.
Hsp = hilbert_space(c, 1:N, SingleParticle())
h = representation(hopping, Hsp, :dense)
ε = eigvals(Hermitian(h))
# The open chain has standing-wave modes with ``ε_q = \cos(qπ/(N+1))``.
@assert ε ≈ sort([cos(q * π / (N + 1)) for q in 1:N])

# The many-body ground state fills every negative single-particle level.
@assert minimum(Eferm) ≈ sum(filter(<(0), ε))
