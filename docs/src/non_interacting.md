# Non-interacting hamiltonians

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
@fermions c
N = 2
H = hilbert_space(c, 1:N, FermionicHilbertSpaces.SingleParticleState.(1:N))
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
H = E\mathbf{1} + \sum_{ij} c_i^\dagger\, h_{ij}  c_j + \frac{1}{2}\sum_{ij} \left(\Delta_{ij} c_i^\dagger c_j^\dagger + \Delta^\dagger_{ij} c_i c_j\right).
```
where ``h`` is hermitian, ``E`` is real and ``\Delta`` is antisymmetric. We can do something similar as above by doubling the single particle states with the Nambu states ``\Psi_n = c_n`` for ``n \leq N`` and ``\Psi_{n+N} = c_n^\dagger``, where ``N`` is the number of fermions. Then the hamiltonian can be written as
```math
H = \left(E + \tfrac{1}{2}\operatorname{tr} h\right)\mathbf{1} + \sum_{ij} \Psi_i^\dagger\, \mathcal{H}_{ij} \Psi_j.
```
Since we introduced a redundancy by doubling the single particle states, ``\mathcal{H}`` is not unique. One common choice is to use the representation 
```math
\mathcal{H} = \frac{1}{2}\begin{pmatrix}
h & \Delta \\
\Delta^\dagger & -h^*
\end{pmatrix}
```
We can get this representation manually as follows:
```@example bdg_particle_hilbert_space
using FermionicHilbertSpaces, LinearAlgebra
N = 2
states = vec([FermionicHilbertSpaces.NambuState(n, hole) for (n, hole) in Base.product(1:N, (true, false))])
@fermions c
H = hilbert_space(c, 1:N, states)
h = Hermitian(rand(N, N))
Δ = rand(N, N) |> m -> m - transpose(m)
E = rand()
op = E * I + sum(c[i]' * h[i, j] * c[j] + (Δ[i, j] * c[i]' * c[j]' + Δ'[i, j] * c[i] * c[j]) / 2 for i in 1:N, j in 1:N)
ℋ = representation(op - E * I, H) |> FermionicHilbertSpaces.normal_order_to_bdg
ℋ ≈ [h Δ; Δ' -conj(h)] / 2
```
The matrix returned by `representation` will depend on the ordering of operators in the symbolic operator `op`, and it includes the identity part of `op`. If `op` is normal ordered (which is the default) and has no identity part, one can use `FermionicHilbertSpaces.normal_order_to_bdg` to convert it to the choice above.
!!! tip "Use `BdG` instead"
    By defining the hilbert space with `hilbert_space(c, 1:N, BdG())`, one automatically gets the Nambu states and `representation` will return ``\mathcal{H}`` without the need to manually convert it or remove the identity part:
    ```julia
    Hbdg = hilbert_space(c, 1:N, BdG())
    representation(op, Hbdg) ≈ ℋ # true
    ```

### Quasiparticles
`eigen(A, BdGEigen())` diagonalizes a BdG matrix while keeping the particle-hole pairing of its eigenvectors. The eigenvalues are sorted, so for `N` modes columns `k` and `2N + 1 - k` are partners with eigenvalues `e` and `-e`. A column `[u; v]` gives the quasiparticle ``\gamma^\dagger = \sum_i u_i c_i^\dagger + v_i c_i``, which raises the energy by ``2e``. See [`BdGEigen`](@ref) for details.
```@example bdg_particle_hilbert_space
Hbdg = hilbert_space(c, 1:N, BdG())
vals, vecs = eigen(representation(op, Hbdg, :dense), BdGEigen())
vals
```

## From matrices and eigenvectors back to symbolic operators
`quadratic_operator(M, H)` inverts `representation` on the `BdG` and `SingleParticle` spaces. On the `BdG` space it returns ``\sum_{ij} \Psi_i^\dagger M_{ij} \Psi_j`` with ``\Psi_n = c_n`` for ``n \leq N`` and ``\Psi_{n+N} = c_n^\dagger``, which is traceless and so differs from the original operator by the constant ``E + \tfrac{1}{2}\operatorname{tr} h``. `quasiparticle(v, H)` turns an eigenvector into the symbolic operator that creates the corresponding quasiparticle. Both results are ordinary symbolic operators, so they can be multiplied, adjointed and represented on the many-body space:
```@example bdg_quasiparticles
using FermionicHilbertSpaces, LinearAlgebra
@fermions c
N = 2
Hbdg = hilbert_space(c, 1:N, BdG())
op = c[1]' * c[2] + c[1]' * c[2]' + hc
M = representation(op, Hbdg, :dense)
vals, vecs = eigen(M, FermionicHilbertSpaces.BdGEigen())
q = quasiparticle(vecs[:, end], Hbdg)
Hmb = hilbert_space(c, 1:N)
Hmat = representation(op, Hmb, :dense)
Qmat = representation(q, Hmb, :dense)
Hmat * Qmat - Qmat * Hmat ≈ 2 * vals[end] * Qmat
```
For an eigenvector with eigenvalue `e`, `q` raises the many-body energy by `2e`, because each quasiparticle appears in ``\Psi^\dagger M \Psi`` both through itself and through its particle-hole partner with eigenvalue `-e`.
