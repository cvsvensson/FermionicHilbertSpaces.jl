# # Entanglement and negativity
#
# In its topological phase the Kitaev chain hosts a Majorana zero mode at each end, and together
# they form one fermion shared by the two ends. This page shows that this long-range entanglement
# is detected by the fermionic logarithmic negativity between the first and last site, and missed
# by the ordinary partial transpose. On the package side it shows how to reduce a many-body ground
# state to a non-contiguous subsystem with `partial_trace` and compute its negativity with
# `logarithmic_negativity`, which applies the fermionic partial transpose with the correct signs.
# The result is from H. Shapourian, K. Shiozaki and S. Ryu,
# [Phys. Rev. Lett. 118, 216402 (2017)](https://doi.org/10.1103/PhysRevLett.118.216402).
#
# We take the even-parity ground state (a parity eigenstate, so its reduced state on the non-contiguous ends is well defined) of a chain of `N` sites, reduce it to the two end sites,
# and compute the negativity between them. With `phase_factors = false` the same function uses
# the ordinary matrix transpose on site 1 instead, as if the modes were qubits.
using FermionicHilbertSpaces
import FermionicHilbertSpaces: logarithmic_negativity
using LinearAlgebra, Arpack, Plots

@fermions f
kitaev(N, μ; t = 1.0, Δ = 1.0) = sum(-μ * f[i]' * f[i] for i in 1:N) +
    sum(-t * f[i]' * f[i + 1] + Δ * f[i] * f[i + 1] + hc for i in 1:(N - 1))

function end_negativity(N, μ; phase_factors = true)
    H = hilbert_space(f, 1:N, ParityConservation())
    Heven = sector(1, H)
    _, vecs = eigs(representation(kitaev(N, μ), Heven); nev = 1, which = :SR)
    ψ = embed(vecs[:, 1], Heven => H)
    Hends = hilbert_space(f, [1, N])
    ρ = partial_trace(ψ * ψ', H => Hends)
    return logarithmic_negativity(ρ, Hends, hilbert_space(f, [1]); phase_factors)
end;

# At the sweet spot ``\mu = 0``, ``t = \Delta`` the end modes are exactly decoupled from the bulk,
# and the fermionic negativity is ``\log\sqrt2`` for every chain length. Away from it the end modes
# spread into the bulk and less of them sits on the end sites, so the negativity decreases. For the
# longer chains, deep in either phase, it does not depend on how far apart the ends are and vanishes beyond the transition
# at ``\mu = 2t``; in the short chain the ends also couple through the bulk, which leaves a tail.
# The bosonic negativity is zero throughout: viewed as qubits, the two ends are not entangled.
@assert end_negativity(8, 0.0) ≈ log(2) / 2
μs = range(0, 4, length = 41)
plot(xlabel = "μ / t", ylabel = "negativity between the end sites", frame = :box)
for N in (4, 8, 12)
    plot!(μs, [end_negativity(N, μ) for μ in μs]; label = "fermionic, N = $N", lw = 2)
end
plot!(μs, [end_negativity(8, μ; phase_factors = false) for μ in μs]; label = "bosonic, N = 8", lw = 2, ls = :dash)
hline!([log(2) / 2]; label = "log √2", color = :gray, ls = :dot)
