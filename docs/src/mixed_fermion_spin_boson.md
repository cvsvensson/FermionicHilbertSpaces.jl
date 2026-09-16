# Mixed fermion-spin-boson example

This example combines several features at once: fermionic, spin, and bosonic Hilbert spaces; custom constraints; and reduced density matrices.

Consider a spinful fermionic mode $f$, a localized spin-half $S$, and a bosonic mode $a$ with the Hamiltonian
$$
H = \omega a^\dagger a + \Delta_f s_z + \Delta_S S_z
    + J\vec{s} \cdot \vec{S}
    + g\left(a^\dagger f_\downarrow^\dagger f_\uparrow
    + a f_\uparrow^\dagger f_\downarrow\right).
$$
where $s_\alpha = \tfrac{1}{2} f^\dagger \sigma_\alpha f$ is the spin operator of the fermion. This model conserves $Q = a^\dagger a + s_z + S_z$ and the fermion number $N_f$. We will calculate the ground state in the $N_f = 1$, $Q = 0$ sector, and then the mutual information between different subsystems.

```@example mixed_fermion_spin_boson
using FermionicHilbertSpaces
using FermionicHilbertSpaces: FilterConstraint, particle_number, SectorConstraint
using LinearAlgebra

@fermions f
@spin S 1//2
@boson a
H_up = hilbert_space(f, [:↑])
H_down = hilbert_space(f, [:↓])
H_spin = hilbert_space(S)
H_boson = hilbert_space(a, 4)  # Truncated boson with 4 levels
spaces = [H_up, H_down, H_spin, H_boson]

Q = FilterConstraint(iszero ∘ sum,
    [state -> 1//2 * particle_number(state),
    state -> -1//2 * particle_number(state),
    state -> state.m,
    state -> particle_number(state)],
    spaces)
Nf = NumberConservation(1, [H_up, H_down])
H = tensor_product(spaces; constraint = Q * Nf)
basisstates(H)
```

The sector of interest has three states. Let's define the symbolic operators, build the Hamiltonian, and compute the ground state.

```@example mixed_fermion_spin_boson
sz = (f[:↑]'*f[:↑] - f[:↓]'*f[:↓]) / 2
sx = (f[:↑]'*f[:↓] + f[:↓]'*f[:↑]) / 2
sy = (f[:↑]'*f[:↓] - f[:↓]'*f[:↑]) / (2im)
s = [sx, sy, sz]
ω, Δf, ΔS, J, g = 1.0, 0.8, 1.1, 0.25, 0.15
ham = ω*a'*a + Δf*sz + ΔS*S[:z] + J*s'*S[:] +
    g * (a'*f[:↓]'*f[:↑] + a*f[:↑]'*f[:↓])

ψ = eigvecs(Hermitian(representation(ham, H, :dense)))[:, 1]
```

Now compute the reduced density matrices and the mutual information between different subsystem pairs.

```@example mixed_fermion_spin_boson
rho = ψ*ψ'
entropy(rho) = sum(-p * log(p) for p in eigvals(rho) if p > 1e-12)
function mutual_info(rho, HA, HB, H)
    HAB = tensor_product(HA, HB)
    rhoA = partial_trace(rho, H => HA)
    rhoB = partial_trace(rho, H => HB)
    rhoAB = partial_trace(rho, H => HAB)
    entropy(rhoA) + entropy(rhoB) - entropy(rhoAB)
end

H_fermions = tensor_product(H_up, H_down; constraint = NumberConservation(1))
subsystems = [(H_fermions, H_spin), (H_fermions, H_boson), (H_spin, H_boson)]
[mutual_info(rho, HA, HB, H) for (HA, HB) in subsystems]
```

This yields the mutual information for the three subsystem pairs.
