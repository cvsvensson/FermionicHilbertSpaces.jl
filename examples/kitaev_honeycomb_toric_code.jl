# # Toric code from the Kitaev honeycomb model

# Kitaev's honeycomb model has a spin-1/2 on each site and an Ising coupling on each link,
# ```math
# H = -J_x \sum_{x\text{-links}} σ^x_j σ^x_k - J_y \sum_{y\text{-links}} σ^y_j σ^y_k - J_z \sum_{z\text{-links}} σ^z_j σ^z_k.
# ```
# For $J_z \gg J_x, J_y$, the two spins on each z-link align, and fourth-order perturbation theory in
# the x- and y-links gives the toric code $H_\mathrm{eff} = -J_\mathrm{eff} \sum_p W_p$, with
# $J_\mathrm{eff} = J_x^2 J_y^2 / (16 J_z^3)$ and $W_p$ the product of the six link operators around hexagon $p$.
# Symbolic kets let us compute $J_\mathrm{eff}$ by applying spin operators on a torus too large to enumerate,
# and check it against this formula.
using FermionicHilbertSpaces
import FermionicHilbertSpaces.NonCommutativeProducts.VectorInterface: inner

# We draw the honeycomb as a brick wall on an `Lx × Ly` torus. Horizontal links alternate between
# x and y, and a vertical z-link joins `(i, j)` and `(i, j + 1)` when `i + j` is even.
Lx, Ly = 8, 8
site(i, j) = (mod1(i, Lx), mod1(j, Ly))
@spins S 1 // 2
H = hilbert_space(S, [site(i, j) for i in 1:Lx for j in 1:Ly])
K(α, a, b) = 4 * S[a][α] * S[b][α] # σᵅσᵅ, with Pauli matrices σ = 2S
hlink(i, j) = K(iseven(i + j) ? :x : :y, site(i, j), site(i + 1, j))
zlink(i, j) = K(:z, site(i, j), site(i, j + 1))

# The plaquette operator of the hexagon with lower left corner `(i, j)`, for `i + j` even, multiplies the links around it.
hexagon(i, j) = hlink(i, j) * hlink(i + 1, j) * zlink(i + 2, j) * hlink(i + 1, j + 1) * hlink(i, j + 1) * zlink(i, j)

# The all-up state has every z-link aligned. Applying a plaquette operator gives a second aligned state.
ψ = Ket(fill("↑", Lx * Ly), H)
ψp = hexagon(3, 3) * ψ

# The matrix element of $H_\mathrm{eff}$ between the two states is $⟨ψ_p| V G V G V G V |ψ⟩$, where
# $V$ holds the x- and y-links and $G = Q/(E_0 - H_0)$ is the resolvent of the z-links $H_0$ outside the aligned subspace.
# Lower orders do not connect the two states.
# $H_0$ is diagonal in the $σ^z$ basis, so $G$ divides each basis state in a ket by its energy denominator.
function plaquette_amplitude(Jx, Jy, Jz)
    H0 = -Jz * sum(zlink(i, j) for i in 1:Lx, j in 1:Ly if iseven(i + j))
    V = -sum((iseven(i + j) ? Jx : Jy) * hlink(i, j) for i in 1:Lx, j in 1:Ly)
    E0 = -Jz * Lx * Ly / 2
    energy(state) = inner(state, H0 * state)
    G(φ) = sum(c / (E0 - energy(state)) * state for (state, c) in φ.dict if !(energy(state) ≈ E0))
    ## Split the product as ⟨φp|G|φ⟩ with φ = VGV|ψ⟩, since V and G are Hermitian.
    φ, φp = V * G(V * ψ), V * G(V * ψp)
    ## The middle G is only needed on the basis states that φ and φp share.
    return sum(conj(c) * φ.dict[state] / (E0 - energy(state)) for (state, c) in φp.dict if haskey(φ.dict, state))
end

# The amplitude equals $-J_\mathrm{eff}$ for any couplings, since it is exactly the fourth-order term.
for (Jx, Jy, Jz) in ((1.0, 1.0, 10.0), (0.3, 0.7, 2.0))
    @assert real(plaquette_amplitude(Jx, Jy, Jz)) ≈ -Jx^2 * Jy^2 / (16Jz^3)
end

# See A. Kitaev, Ann. Phys. **321**, 2 (2006), arXiv:cond-mat/0506438, for the derivation.
