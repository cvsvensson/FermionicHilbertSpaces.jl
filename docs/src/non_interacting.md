# Non-interacting hamiltonians
The functionality on this page lives in the submodule `FermionicHilbertSpaces.FreeFermions`, loaded with `using FermionicHilbertSpaces.FreeFermions`.

## Number conserving non-interacting hamiltonians
A quadratic fermionic hamiltonian with number conservation can be written as 
```math
H = E\mathbf{1} + \sum_{ij} c_i^\dagger\, h_{ij}  c_j.
```
The matrix element between two single particle states with modes ``n`` and ``m``, i.e. ``\ket{n} = c_n^\dagger \ket{0}``, is
```math
 \bra{m}H\ket{n} = E\delta_{nm} + h_{nm}.
```
If ``H`` is hermitian so is ``h`` and ``E`` is real, but we don't have to restrict to this case here.

One can manually define the hilbert space using only the single particle states as
```@example single_particle_hilbert_space
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@fermions c
N = 2
H = hilbert_space(c, 1:N, SingleParticleState.(1:N))
h = rand(ComplexF64, N, N)
E = rand(ComplexF64)
op = E * I + sum(c[i]' * h[i, j] * c[j] for i in 1:N, j in 1:N)
representation(op, H) == h + E * I
```
Often, $h_{nm}$ is of interest because diagonalizing it gives information on the quasiparticles in the system.

!!! tip "Use `SingleParticle` instead"
    For convenience, `hilbert_space(c, 1:N, SingleParticle())` can be used to define the hilbert space which will give only the single particle states, and will remove the contribution of the identity operator when calling `representation`:
    ```julia
    H = hilbert_space(c, 1:N, SingleParticle())
    representation(op,H) == h # true
    ```

See [Free fermions on a 2D grid](@ref) for an example of how to use this.

## Non-interacting hamiltonians without number conservation
A general non-interacting hermitian operator respecting super-selection can be written as
```math
H = E\mathbf{1} + \sum_{ij} \left(c_i^\dagger\, h_{ij}  c_j + \frac{1}{2}\left(\Delta_{ij} c_i^\dagger c_j^\dagger + \Delta^*_{ij} c_j c_i\right)\right),
```
where ``h`` is hermitian, ``E`` is real and ``\Delta`` is antisymmetric. We can do something similar as above by doubling the single particle states with the Nambu states ``\Psi_n = c_n`` for ``n \leq N`` and ``\Psi_{n+N} = c_n^\dagger``, where ``N`` is the number of fermions. Then the hamiltonian can be written as
```math
H = E'\mathbf{1} + \sum_{ij} \Psi_i^\dagger\, M_{ij} \Psi_j.
```
Since we introduced a redundancy by doubling the single particle states, ``M`` is not unique. Requiring the particle-hole symmetry ``M = -\tau_x M^T \tau_x`` fixes it to
```math
M = \frac{1}{2}\begin{pmatrix}
h & \Delta \
-\Delta^* & -h^*
\end{pmatrix}, \quad E' = E + \frac{1}{2}\mathrm{tr}\, h.
```
The same symmetrization gives a unique ``M`` for any quadratic operator, also non-hermitian ones, and then ``[H, \Psi_i] = -2 (M\Psi)_i``.
We can get this representation manually as follows:
```@example bdg_particle_hilbert_space
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
using FermionicHilbertSpaces.FreeFermions: normal_order_to_bdg
N = 2
states = vec([NambuState(n, hole) for (n, hole) in Base.product(1:N, (true, false))])
@fermions c
H = hilbert_space(c, 1:N, states)
h = Hermitian(rand(N, N))
Δ = rand(N, N) |> m -> m - transpose(m)
E = rand()
op = E * I + sum(c[i]' * h[i, j] * c[j] + (Δ[i, j] * c[i]' * c[j]' + conj(Δ[i, j]) * c[j] * c[i]) / 2 for i in 1:N, j in 1:N)
M = representation(op, H) |> normal_order_to_bdg
M ≈ [h Δ; -conj(Δ) -conj(h)] / 2
```
The matrix returned by `representation` on this space depends on the ordering of operators in the symbolic operator `op`, and `FreeFermions.normal_order_to_bdg` converts it to the particle-hole symmetric choice above.
!!! tip "Use `BdG` instead"
    By defining the hilbert space with `hilbert_space(c, 1:N, BdG())`, one automatically gets the Nambu states and `representation` will return ``M`` without the need to manually convert it:
    ```julia
    Hbdg = hilbert_space(c, 1:N, BdG())
    representation(op, Hbdg) ≈ M # true
    ```
    The BdG space wraps the constrained space above, available through `parent`, and does not combine with tensor products or partial traces.

### Quasiparticles
`eigen(A, BdGEigen())` diagonalizes a BdG matrix while keeping the particle-hole pairing of its eigenvectors. The eigenvalues are sorted, so for `N` modes columns `k` and `2N + 1 - k` are partners with eigenvalues `e` and `-e` whose quasiparticles are adjoints of each other. A column `[u; v]` gives the quasiparticle ``\gamma^\dagger = \sum_i u_i c_i^\dagger + v_i c_i``, which raises the energy by ``2e``. The matrix must be Hermitian, and eigenvalues with `abs(e) <= cutoff * maximum(abs, vals)` are set to zero, with the default `cutoff = 2N * eps` as for `rank`.
```@example bdg_particle_hilbert_space
Hbdg = hilbert_space(c, 1:N, BdG())
vals, vecs = eigen(representation(op, Hbdg, :dense), BdGEigen())
vals
```

## Expectation values
On the single-particle and BdG spaces, `expectation(op, G, H)` takes the correlation matrix of the state: ``G_{ij} = \langle c_i^\dagger c_j \rangle`` on the single-particle space and ``G_{ij} = \langle \Psi_i^\dagger \Psi_j \rangle`` on the BdG space. The state is the Gaussian state with this correlation matrix, such as a thermal state of a quadratic hamiltonian, so operators of any degree are evaluated with Wick's theorem. For a thermal state of the single-particle hamiltonian `h`, ``G = (1 + e^{\beta h})^{-T}``:
```@example expectation
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@fermions c
H = hilbert_space(c, 1:4, SingleParticle())
ham = sum(c[n]' * c[n+1] + hc for n in 1:3) + 1
h = representation(ham, H, :dense)
G = transpose(inv(I + exp(10 * h)))
expectation(ham, G, H)
```
`mean_field(op, G, H)` replaces every product in `op` by its terms of order at most two in the fluctuations around the state, the Hartree, Fock and pairing terms. The result is of degree at most two, so for an even `op` it can be represented on `H`, and it has the same expectation value as `op`:
```@example expectation
U = 2 * c[1]' * c[1] * c[2]' * c[2]
mf = mean_field(U, G, H)
expectation(mf, G, H) ≈ expectation(U, G, H)
```

## Correlation matrices and entanglement
For the BdG ground state, the correlation matrix ``G`` of the previous section is `correlation_matrix(F)`, where `F` is the result of `eigen(A, BdGEigen())`, and `ground_state_parity(F)` is its fermionic parity. For a many-body state `ψ` on `H`, `correlation_matrix(ψ, H, Hbdg)` gives the same matrix for the modes of `Hbdg`.

The rows and columns of a subsystem of modes form the reduced correlation matrix. Its eigenvalues ``\lambda`` give the entanglement entropy ``S = -\sum \lambda \log \lambda`` of the subsystem, which equals the entropy of the reduced many-body density matrix:
```@example entanglement
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@fermions f
N = 4
h = Hermitian(rand(ComplexF64, N, N))
Δ = rand(ComplexF64, N, N) |> m -> m - transpose(m)
ham = sum(f[i]' * h[i, j] * f[j] + Δ[i, j] * f[i]' * f[j]' + hc for i in 1:N, j in 1:N)
Hbdg = hilbert_space(f, 1:N, BdG())
A = representation(ham, Hbdg, :dense)
G = correlation_matrix(eigen(A, BdGEigen()))
sub = 1:2
Gsub = G[[sub; sub .+ N], [sub; sub .+ N]]
entropy(λ) = -sum(x -> x > 0 ? x * log(x) : zero(x), λ)
S = entropy(eigvals(Hermitian(Gsub)))

H = hilbert_space(f, 1:N)
ψ = eigvecs(Hermitian(representation(ham, H, :dense)))[:, 1]
ρsub = partial_trace(ψ * ψ', H => hilbert_space(f, sub))
@assert S ≈ entropy(eigvals(Hermitian(ρsub)))
@assert Gsub ≈ correlation_matrix(ψ, H, hilbert_space(f, sub, BdG()))
@assert expectation(ham, G, Hbdg) ≈ expectation(ham, ψ, H)
S
```

Conversely, `gaussian_state(G, H, Hbdg)` builds the many-body density matrix on `H` of the Gaussian state with correlation matrix `G`. The reduced correlation matrix gives the reduced density matrix:
```@example entanglement
gaussian_state(Gsub, hilbert_space(f, sub), hilbert_space(f, sub, BdG())) ≈ ρsub
```

## From matrices and eigenvectors back to symbolic operators
`symbolic_operator(M, H)` turns a matrix on a `BdG` or `SingleParticle` space back into a symbolic operator, and `quasiparticle(v, H)` turns an eigenvector into the operator that creates the corresponding quasiparticle. On the `BdG` space, `symbolic_operator` returns the traceless operator, which differs from the original one by a constant.
```@example bdg_quasiparticles
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@fermions c
H = hilbert_space(c, 1:2, BdG())
M = representation(c[1]' * c[2] + c[1]' * c[2]' + hc, H, :dense)
vals, vecs = eigen(M, BdGEigen())
symbolic_operator(M, H), quasiparticle(vecs[:, end], H)
```

## Majoranas
Quadratic Majorana operators get the same treatment with `hilbert_space(γ, labels, SingleParticle())`, where `representation` gives the antisymmetric matrix of the operator. Below, `correlation_matrix(F)` is the ground-state matrix ``G_{ij} = \langle \gamma_i \gamma_j angle / 2`` that `expectation` takes, and `ground_state_parity(F, H)` is the ground-state parity:
```@example majoranas
using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.FreeFermions
@majoranas γ
H = hilbert_space(γ, 1:6, SingleParticle())
ham = 1im / 2 * sum(γ[2k] * γ[2k + 1] for k in 1:2) + 0.1im * sum(γ[2k - 1] * γ[2k] for k in 1:3)
F = eigen(Hermitian(representation(ham, H, :dense)))
G = correlation_matrix(F)
ground_state_parity(F, H), expectation(ham, G, H)
```
