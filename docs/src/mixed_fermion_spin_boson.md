# Mixed fermion-spin-boson example

This example combines several features at once: fermionic, spin, and bosonic Hilbert spaces; constraints on operators acting on all of them; and reduced density matrices.

Consider a spinful fermionic mode $f$, a localized spin-half $S$, and a bosonic mode $a$ with the Hamiltonian
```math
H = \omega a^\dagger a + \Delta_f s_z + \Delta_S S_z
    + J\vec{s} \cdot \vec{S}
    + g\left(a^\dagger f_\downarrow^\dagger f_\uparrow
    + a f_\uparrow^\dagger f_\downarrow\right),
```
where ``s_\alpha = \tfrac{1}{2} f^\dagger \sigma_\alpha f`` is the spin operator of the fermion. This model conserves ``Q = a^\dagger a + s_z + S_z`` and the fermion number ``N_f``. We will calculate the ground state in the ``N_f = 1``, ``Q = 0`` sector, and then the mutual information between different subsystems.

```@example mixed_fermion_spin_boson
using FermionicHilbertSpaces, LinearAlgebra
import FermionicHilbertSpaces: von_neumann_entropy

@fermions f
@spin S 1//2
@boson a
H_up = hilbert_space(f, [:↑])
H_down = hilbert_space(f, [:↓])
H_spin = hilbert_space(S)
H_boson = hilbert_space(a, 4)  # Truncated boson with 4 levels
spaces = [H_up, H_down, H_spin, H_boson]

n_up = f[:↑]'f[:↑]
n_down = f[:↓]'f[:↓]
Q = a'a + 1//2 * (n_up - n_down) + S[:z]
H = tensor_product(spaces; constraint = (Q => 0, n_up + n_down => 1))
basisstates(H)
```

The sector of interest has three states. Let's define the symbolic operators, build the Hamiltonian, and compute the ground state.

```@example mixed_fermion_spin_boson
sz = (n_up - n_down) / 2
sx = (f[:↑]'*f[:↓] + f[:↓]'*f[:↑]) / 2
sy = (f[:↑]'*f[:↓] - f[:↓]'*f[:↑]) / (2im)
s = [sx, sy, sz]
ω, Δf, ΔS, J, g = 1.0, 0.8, 1.1, 0.25, 0.15
ham = ω*a'*a + Δf*sz + ΔS*S[:z] + J*s'*S[:] +
    g * (a'*f[:↓]'*f[:↑] + a*f[:↑]'*f[:↓])

ψ = eigvecs(Hermitian(representation(ham, H, Matrix)))[:, 1]
```

Now compute the reduced density matrices and the mutual information between different subsystem pairs.

```@example mixed_fermion_spin_boson
function mutual_info(ψ, HA, HB, H)
    HAB = tensor_product(HA, HB)
    S = von_neumann_entropy
    S(partial_trace(ψ, H => HA)) + S(partial_trace(ψ, H => HB)) - S(partial_trace(ψ, H => HAB))
end

H_fermions = tensor_product(H_up, H_down; constraint = NumberConservation(1))
subsystems = [(H_fermions, H_spin), (H_fermions, H_boson), (H_spin, H_boson)]
[mutual_info(ψ, HA, HB, H) for (HA, HB) in subsystems]
```

The three entries are the mutual information of the fermion with the spin, of the fermion with the boson, and of the spin with the boson.
