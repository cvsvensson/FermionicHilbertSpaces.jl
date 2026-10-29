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
H = E\mathbf{1} + \sum_{ij} c_i^\dagger\, h_{ij}  c_j + \Delta_{ij} c_i^\dagger c_j^\dagger + \Delta^\dagger_{ij} c_i c_j.
```
where ``h`` is hermitian, ``E`` is real and ``\Delta`` is antisymmetric. We can do something similar as above by doubling the single particle states with the Nambu states ``\Psi_n = c_n \text{for n \leq N}, \Psi_n = c_n^\dagger \text{for n > N}``, where ``N`` is the number of fermions. Then the hamiltonian can be written as
```math
H = E\mathbf{1} + \sum_{ij} \Psi_i^\dagger\, \mathcal{H}_{ij} \Psi_j.
```
Since we introduced a redundancy by doubling the single particle states, ``\mathcal{H}`` is not unique. One common choice is to use the representation 
```math
\mathcal{H} = \begin{pmatrix}
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
op = E * I + sum(c[i]' * h[i, j] * c[j] + (Δ[i, j] * c[i]' * c[j]' + Δ'[i, j] * c[i] * c[j]) for i in 1:N, j in 1:N)
ℋ = representation(op, H) |> FermionicHilbertSpaces.normal_order_to_bdg
FermionicHilbertSpaces.isbdgmatrix(ℋ)
```
The matrix returned by `representation` will depend on the ordering of operators in the symbolic operator `op`. If it is normal ordered (which is the default), one can use `FermionicHilbertSpaces.normal_order_to_bdg` to convert it to the choice above.
!!! tip "Use `BdG` instead"
    By defining the hilbert space with `hilbert_space(c, 1:N, BdG())`, one automatically gets the Nambu states and `representation` will return a matrix of the form above without the need to manually convert it:
    ```julia
    Hbdg = hilbert_space(c, 1:N, BdG())
    representation(op, Hbdg)
    #= example output
    4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 12 stored entries:
    0.449635   0.297613    ⋅         0.18844
    0.297613   0.169731  -0.18844     ⋅
    ⋅        -0.18844   -0.449635  -0.297613
    0.18844     ⋅        -0.297613  -0.169731 =#
    ```

## Correlation matrices and entanglement
The Nambu correlation matrix ``\Gamma_{ij} = \langle \Psi_j^\dagger \Psi_i \rangle`` of the BdG ground state is `correlation_matrix(vecs)`, where `vecs` are the eigenvectors from `eigen(A, BdGEigen())`, and `ground_state_parity(vecs)` is its fermionic parity. For a many-body state `ψ` on `H`, `correlation_matrix(ψ, H, Hbdg)` gives the same matrix for the modes of `Hbdg`.

The rows and columns of a subsystem of modes form the reduced correlation matrix. Its eigenvalues ``\lambda`` give the entanglement entropy ``S = -\sum \lambda \log \lambda`` of the subsystem, which equals the entropy of the reduced many-body density matrix:
```@example entanglement
using FermionicHilbertSpaces, LinearAlgebra
@fermions f
N = 4
h = Hermitian(rand(ComplexF64, N, N))
Δ = rand(ComplexF64, N, N) |> m -> m - transpose(m)
ham = sum(f[i]' * h[i, j] * f[j] + Δ[i, j] * f[i]' * f[j]' + hc for i in 1:N, j in 1:N)
_, vecs = eigen(representation(ham, hilbert_space(f, 1:N, BdG()), :dense), FermionicHilbertSpaces.BdGEigen())
Γ = correlation_matrix(vecs)
A = 1:2
ΓA = Γ[[A; A .+ N], [A; A .+ N]]
entropy(λ) = -sum(x -> x > 0 ? x * log(x) : zero(x), λ)
S = entropy(eigvals(Hermitian(ΓA)))

H = hilbert_space(f, 1:N)
ψ = eigvecs(Hermitian(representation(ham, H, :dense)))[:, 1]
ρA = partial_trace(ψ * ψ', H => hilbert_space(f, A))
@assert S ≈ entropy(eigvals(Hermitian(ρA)))
@assert ΓA ≈ correlation_matrix(ψ, H, hilbert_space(f, A, BdG()))
S
```
