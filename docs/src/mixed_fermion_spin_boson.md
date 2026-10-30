# Mixed fermion-spin-boson example

This example combines fermionic, spin and bosonic Hilbert spaces with custom constraints in one model, and uses partial traces to ask which of the three subsystems the ground state entangles.

Consider a spinful fermionic mode $f$, a localized spin-half $S$, and a bosonic mode $a$ with the Hamiltonian
```math
H = \omega a^\dagger a + \Delta_f s_z + \Delta_S S_z
    + J\vec{s} \cdot \vec{S}
    + g\left(a^\dagger f_\downarrow^\dagger f_\uparrow
    + a f_\uparrow^\dagger f_\downarrow\right),
```
where ``s_\alpha = \tfrac{1}{2} f^\dagger \sigma_\alpha f`` is the spin operator of the fermion. The model conserves ``Q = a^\dagger a + s_z + S_z`` and the fermion number ``N_f``. We build the ``N_f = 1``, ``Q = 0`` sector from a `FilterConstraint` and a `NumberConservation`, find its ground state, and compute the mutual information between the fermion, the spin and the boson:

```@example mixed_fermion_spin_boson
using FermionicHilbertSpaces, LinearAlgebra
import FermionicHilbertSpaces: particle_number, von_neumann_entropy

@fermions f
@spin S 1//2
@boson a
H_up = hilbert_space(f, [:↑])
H_down = hilbert_space(f, [:↓])
H_spin = hilbert_space(S)
H_boson = hilbert_space(a, 4)  # truncated to 4 levels
spaces = [H_up, H_down, H_spin, H_boson]

Q = FilterConstraint(iszero ∘ sum;
    maps = [state -> 1//2 * particle_number(state),
    state -> -1//2 * particle_number(state),
    state -> state.m,
    state -> particle_number(state)],
    spaces)
Nf = NumberConservation(1; spaces = [H_up, H_down])
H = tensor_product(spaces; constraint = Q * Nf)

sz = (f[:↑]'*f[:↑] - f[:↓]'*f[:↓]) / 2
sx = (f[:↑]'*f[:↓] + f[:↓]'*f[:↑]) / 2
sy = (f[:↑]'*f[:↓] - f[:↓]'*f[:↑]) / (2im)
s = [sx, sy, sz]
ω, Δf, ΔS, J, g = 1.0, 0.8, 1.1, 0.25, 0.15
ham = ω*a'*a + Δf*sz + ΔS*S[:z] + J*s'*S[:] +
    g * (a'*f[:↓]'*f[:↑] + a*f[:↑]'*f[:↓])
ψ = eigvecs(Hermitian(representation(ham, H, :dense)))[:, 1]

function mutual_info(ρ, HA, HB, H)
    HAB = tensor_product(HA, HB)
    S = von_neumann_entropy
    S(partial_trace(ρ, H => HA)) + S(partial_trace(ρ, H => HB)) - S(partial_trace(ρ, H => HAB))
end

H_fermions = tensor_product(H_up, H_down; constraint = NumberConservation(1))
subsystems = [(H_fermions, H_spin), (H_fermions, H_boson), (H_spin, H_boson)]
[mutual_info(ψ, HA, HB, H) for (HA, HB) in subsystems]
```
The three entries are the mutual information of the fermion with the spin, of the fermion with the boson, and of the spin with the boson.
