# Non-interacting hamiltonians
A quadratic fermionic hamiltonian on ``N`` modes is described by an ``N \times N`` (or ``2N \times 2N``) matrix instead of a ``2^N``-dimensional one. This page shows how to get these matrices from symbolic operators, turn them into quasiparticles, and use them for expectation values, entanglement and Majorana problems. The functionality lives in the submodule `FermionicHilbertSpaces.FreeFermions`, loaded with `using FermionicHilbertSpaces.FreeFermions`; see [Free fermions on a 2D grid](@ref) for a large-scale example.

## Single-particle and BdG matrices
A number conserving quadratic hamiltonian ``H = E\mathbf{1} + \sum_{ij} c_i^\dagger\, h_{ij}  c_j`` acts on the single particle states ``c_n^\dagger \ket{0}`` as the matrix ``h`` (plus ``E``). On the space `hilbert_space(c, 1:N, SingleParticle())`, `representation` returns ``h`` and drops the identity contribution.

Without number conservation, a hermitian operator respecting super-selection is
```math
H = E\mathbf{1} + \sum_{ij} \left(c_i^\dagger\, h_{ij}  c_j + \frac{1}{2}\left(\Delta_{ij} c_i^\dagger c_j^\dagger + \Delta^*_{ij} c_j c_i\right)\right),
```
with ``h`` hermitian, ``E`` real and ``\Delta`` antisymmetric. Writing it with the Nambu operators ``\Psi_n = c_n`` and ``\Psi_{n+N} = c_n^\dagger`` as ``H = E'\mathbf{1} + \sum_{ij} \Psi_i^\dagger\, M_{ij} \Psi_j`` is ambiguous because the doubling is redundant. Requiring the particle-hole symmetry ``M = -\tau_x M^T \tau_x`` fixes
```math
M = \frac{1}{2}\begin{pmatrix}
h & \Delta \
-\Delta^* & -h^*
\end{pmatrix}, \quad E' = E + \frac{1}{2}\mathrm{tr}\, h,
```
and the same symmetrization gives a unique ``M`` for any quadratic operator, with ``[H, \Psi_i] = -2 (M\Psi)_i``. On `hilbert_space(c, 1:N, BdG())`, `representation` returns this ``M``. The BdG space does not combine with tensor products or partial traces.

`eigen(M, BdGEigen())` diagonalizes a Hermitian BdG matrix while keeping the particle-hole pairing of its eigenvectors. The eigenvalues are sorted, so for ``N`` modes columns ``k`` and ``2N + 1 - k`` are partners with eigenvalues ``e`` and ``-e``. A column `[u; v]` gives the quasiparticle ``\gamma^\dagger = \sum_i u_i c_i^\dagger + v_i c_i``, which raises the energy by ``2e``. Eigenvalues with `abs(e) <= cutoff * maximum(abs, vals)` are set to zero, with the default `cutoff = 2N * eps` as for `rank`.
```@example free
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@fermions c
N = 2
h = Hermitian(rand(ComplexF64, N, N))
Δ = rand(N, N) |> m -> m - transpose(m)
E = rand()
number_conserving = E * I + sum(c[i]' * h[i, j] * c[j] for i in 1:N, j in 1:N)
Hsp = hilbert_space(c, 1:N, SingleParticle())
@assert representation(number_conserving, Hsp) ≈ h

op = number_conserving + sum((Δ[i, j] * c[i]' * c[j]' + conj(Δ[i, j]) * c[j] * c[i]) / 2 for i in 1:N, j in 1:N)
Hbdg = hilbert_space(c, 1:N, BdG())
M = representation(op, Hbdg)
@assert M ≈ [h Δ; -conj(Δ) -conj(h)] / 2

vals, vecs = eigen(Hermitian(representation(op, Hbdg, :dense)), BdGEigen())
vals
```

## Gaussian states
On the single-particle and BdG spaces, `expectation(op, G, H)` takes the correlation matrix of the state: ``G_{ij} = \langle c_i^\dagger c_j \rangle`` on the single-particle space and ``G_{ij} = \langle \Psi_i^\dagger \Psi_j \rangle`` on the BdG space. The state is the Gaussian state with this correlation matrix, such as a thermal state of a quadratic hamiltonian, so operators of any degree are evaluated with Wick's theorem. `mean_field(op, G, H)` replaces every product in `op` by its terms of order at most two in the fluctuations around the state, the Hartree, Fock and pairing terms, which gives an operator with the same expectation value that, for an even `op`, can be represented on `H`. For a thermal state of the single-particle hamiltonian `h`, ``G = (1 + e^{\beta h})^{-T}``:
```@example free
Hs = hilbert_space(c, 1:4, SingleParticle())
ham = sum(c[n]' * c[n+1] + hc for n in 1:3) + 1
hs = representation(ham, Hs, :dense)
G = transpose(inv(I + exp(10 * hs)))
U = 2 * c[1]' * c[1] * c[2]' * c[2]
mf = mean_field(U, G, Hs)
@assert expectation(mf, G, Hs) ≈ expectation(U, G, Hs)
expectation(ham, G, Hs)
```

For the BdG ground state, the correlation matrix is `correlation_matrix(F)`, where `F` is the result of `eigen(M, BdGEigen())`, and `ground_state_parity(F)` is its fermionic parity. For a many-body state `ψ` on `H`, `correlation_matrix(ψ, H, Hbdg)` gives the same matrix for the modes of `Hbdg`. The correlation matrix of a subset of the modes is `partial_trace(G, Hbdg => Hsub)` with `Hsub` a BdG space over those modes, and `von_neumann_entropy(Gsub, Hsub)` is the entanglement entropy of that subsystem. `gaussian_state(G, H, Hbdg)` goes the other way and builds the many-body density matrix with correlation matrix `G`. The example checks all of this against the many-body ground state:
```@example free
import FermionicHilbertSpaces: von_neumann_entropy
N = 4
h = Hermitian(rand(ComplexF64, N, N))
Δ = rand(ComplexF64, N, N) |> m -> m - transpose(m)
ham = sum(c[i]' * h[i, j] * c[j] + Δ[i, j] * c[i]' * c[j]' + hc for i in 1:N, j in 1:N)
Hbdg = hilbert_space(c, 1:N, BdG())
F = eigen(Hermitian(representation(ham, Hbdg, :dense)), BdGEigen())
G = correlation_matrix(F)
sub = 1:2
Hsub = hilbert_space(c, sub, BdG())
Gsub = partial_trace(G, Hbdg => Hsub)
S = von_neumann_entropy(Gsub, Hsub)

H = hilbert_space(c, 1:N)
ψ = eigvecs(Hermitian(representation(ham, H, :dense)))[:, 1]
ρsub = partial_trace(ψ * ψ', H => hilbert_space(c, sub))
@assert S ≈ von_neumann_entropy(ρsub)
@assert Gsub ≈ correlation_matrix(ψ, H, Hsub)
@assert expectation(ham, G, Hbdg) ≈ expectation(ham, ψ, H)
@assert gaussian_state(Gsub, hilbert_space(c, sub), Hsub) ≈ ρsub
S
```

## Back to symbolic operators, and Majoranas
`symbolic_operator(M, H)` turns a matrix on a `BdG` or `SingleParticle` space back into a symbolic operator, and `quasiparticle(v, H)` turns an eigenvector into the operator that creates the corresponding quasiparticle. On the `BdG` space, `symbolic_operator` returns the traceless operator, which differs from the original one by a constant.

Quadratic Majorana operators get the same treatment with `hilbert_space(γ, labels, SingleParticle())`, where `representation` gives the antisymmetric matrix of the operator. Below, `correlation_matrix(F)` is the ground-state matrix ``G_{ij} = \langle \gamma_i \gamma_j \rangle / 2`` that `expectation` takes, and `ground_state_parity(F, H)` is the ground-state parity:
```@example free
Hq = hilbert_space(c, 1:2, BdG())
Mq = representation(c[1]' * c[2] + c[1]' * c[2]' + hc, Hq, :dense)
_, vq = eigen(Mq, BdGEigen())
symbolic_operator(Mq, Hq), quasiparticle(vq[:, end], Hq)

@majoranas γ
Hm = hilbert_space(γ, 1:6, SingleParticle())
hamm = 1im / 2 * sum(γ[2k] * γ[2k + 1] for k in 1:2) + 0.1im * sum(γ[2k - 1] * γ[2k] for k in 1:3)
Fm = eigen(Hermitian(representation(hamm, Hm, :dense)))
Gm = correlation_matrix(Fm)
ground_state_parity(Fm, Hm), expectation(hamm, Gm, Hm)
```
