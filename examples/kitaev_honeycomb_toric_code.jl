# # Toric code from the Kitaev honeycomb model

# Symbolic kets let us apply spin operators on a lattice far too large to enumerate. We use this to derive the toric
# code from Kitaev's honeycomb model[^1] on a 64-spin torus: we compute its coupling by fourth-order perturbation
# theory and show that its ground space is four-fold degenerate.

# ## The toric-code limit

# Kitaev's honeycomb model has a spin-1/2 on each site and an Ising coupling on each link,
# ```math
# H = -J_x \sum_{x\text{-links}} σ^x_j σ^x_k - J_y \sum_{y\text{-links}} σ^y_j σ^y_k - J_z \sum_{z\text{-links}} σ^z_j σ^z_k.
# ```
# For ``J_z \gg J_x, J_y`` the two spins on each z-link align, which leaves one effective spin per z-link. Fourth-order
# perturbation theory in the x- and y-links gives the toric code ``H_\mathrm{eff} = -J_\mathrm{eff} \sum_p W_p`` on
# these effective spins, with ``J_\mathrm{eff} = J_x^2 J_y^2 / (16 J_z^3)`` and ``W_p`` the product of the six link
# operators around hexagon ``p``. The ``W_p`` commute and square to one, so the ground space of ``H_\mathrm{eff}`` is
# where every ``W_p = +1``. There are as many hexagons as z-links, so each relation among the ``W_p``, a set of them
# whose product is the identity on the aligned states, doubles the dimension of the ground space.

# ## An 8 × 8 torus

# We draw the honeycomb as a brick wall on an `Lx × Ly` torus. Horizontal links alternate between x and y, and a
# vertical z-link joins `(i, j)` and `(i, j + 1)` when `i + j` is even. The hexagon with lower left corner `(i, j)`,
# for `i + j` even, is bounded by four horizontal links and two z-links.
using FermionicHilbertSpaces
using FermionicHilbertSpaces.NonCommutativeProducts: terms, coefficient, factors

Lx, Ly = 8, 8
site(i, j) = (mod1(i, Lx), mod1(j, Ly))
@spins S 1 // 2
H = hilbert_space(S, [site(i, j) for i in 1:Lx for j in 1:Ly])
K(α, a, b) = 4 * S[a][α] * S[b][α] # σᵅσᵅ, with Pauli matrices σ = 2S
hlink(i, j) = K(iseven(i + j) ? :x : :y, site(i, j), site(i + 1, j))
zlink(i, j) = K(:z, site(i, j), site(i, j + 1))
hexagon(i, j) = hlink(i, j) * hlink(i + 1, j) * zlink(i + 2, j) * hlink(i + 1, j + 1) * hlink(i, j + 1) * zlink(i, j)
corners = [(i, j) for i in 1:Lx, j in 1:Ly if iseven(i + j)]
(log2_full_dimension = Lx * Ly, log2_effective_dimension = length(corners))

# The all-up state has every z-link aligned, and a plaquette operator maps it to a second aligned state.
# The matrix element of ``H_\mathrm{eff}`` between the two is ``⟨ψ_p| V G V G V G V |ψ⟩``, where ``V`` holds the x-
# and y-links and ``G = Q/(E_0 - H_0)`` is the resolvent of the z-links ``H_0`` outside the aligned subspace; lower
# orders do not connect the two states. ``H_0`` is diagonal in the ``σ^z`` basis, so ``G`` divides each term of a ket
# by its energy denominator. The amplitude equals ``-J_\mathrm{eff}`` for any couplings, since it is exactly the
# fourth-order term.
ψ = Ket(fill("↑", Lx * Ly), H)
ψp = hexagon(3, 3) * ψ
function plaquette_amplitude(Jx, Jy, Jz)
    H0 = -Jz * sum(zlink(i, j) for (i, j) in corners)
    V = -sum((iseven(i + j) ? Jx : Jy) * hlink(i, j) for i in 1:Lx, j in 1:Ly)
    E0 = -Jz * length(corners)
    energy(term) = real(coefficient(H0 * term) / coefficient(term))
    G(φ) = sum(term / (E0 - energy(term)) for term in terms(φ) if !(energy(term) ≈ E0))
    ## Split the product as ⟨φp|G|φ⟩ with φ = VGV|ψ⟩, since V and G are Hermitian.
    φ, φp = V * G(V * ψ), V * G(V * ψp)
    ## The middle G is only needed on the basis states that φ and φp share.
    amplitudes = Dict(factors(term) => coefficient(term) for term in terms(φ))
    return sum(
        conj(coefficient(term)) * amplitudes[factors(term)] / (E0 - energy(term))
            for term in terms(φp) if haskey(amplitudes, factors(term))
    )
end
for (Jx, Jy, Jz) in ((1.0, 1.0, 10.0), (0.3, 0.7, 2.0))
    @assert real(plaquette_amplitude(Jx, Jy, Jz)) ≈ -Jx^2 * Jy^2 / (16Jz^3)
end

# On the aligned states, the hexagons in the even rows multiply to the identity, and so do those in the odd rows. A
# product of Pauli operators that maps the all-up state and each state with one z-link flipped to itself acts as the
# identity there. These are the only relations[^1], so the ground space is four-dimensional: the topological
# degeneracy of the toric code on a torus.
apply_hexagons(cs, φ) = foldl((φ, (i, j)) -> hexagon(i, j) * φ, cs; init = φ)
test_states = [ψ; [K(:x, site(i, j), site(i, j + 1)) * ψ for (i, j) in corners]]
for row_parity in (iseven, isodd)
    row = filter(c -> row_parity(c[2]), corners)
    @assert all(φ -> apply_hexagons(row, φ) == φ, test_states)
end

# [^1]: A. Kitaev, Ann. Phys. **321**, 2 (2006), arXiv:cond-mat/0506438.
