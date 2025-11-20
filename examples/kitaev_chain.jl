# # Interacting Kitaev Chain Tutorial

# In this example, we study the interacting Kitaev chain.
# We show how the Hamiltonian can be constructed symbolically or by using matrix representations of the fermionic operators,
# and how to restrict the Hilbert space to a subspace of a given parity.
# We solve for the ground states and characterize the locality of the many-body Majoranas.

# We start by importing the necessary packages.
using FermionicHilbertSpaces, LinearAlgebra, Plots
using Arpack
import FermionicHilbertSpaces: dim, indices
# Then we define the Hilbert space with `N` sites and parity conservation.
N = 12
H = hilbert_space(1:N, ParityConservation())

# Symbolic fermions can be defined using the `@fermions` macro,
@fermions f

# Let's define the interacting Kitaev chain Hamiltonian.
# It is a function of the fermions `f` and parameters `N`, `μ`, `t`, `Δ`, and `U`,
# representing the number of sites, chemical potential, hopping amplitude, pairing amplitude, and interaction strength, respectively.
# Note the use of the Hermitian conjugate `hc`, which simplifies the expression for the Hamiltonian.
kitaev_chain(f, N, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:N-1) +
                                 sum(Δ * f[i] * f[i+1] + hc for i in 1:N-1) +
                                 sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:N-1) +
                                 sum(μ[i] * f[i]' * f[i] for i in 1:N)

# Define parameters close to the sweet spot with perfectly localized Majoranas.
U = 4.0
t = 1.0
δΔ = 0.4
Δ = t + U / 2 - δΔ # slightly detuned from the sweet spot
μ = fill(-U / 2, N) # edge chemical potential
μ[2:N-1] .= -U # bulk chemical potential

# We can now construct the Hamiltonian using symbolic fermions for a symbolic representation
hsym = kitaev_chain(f, N, μ, t, Δ, U)
# To convert the symbolic Hamiltonian to a matrix representation, we can use the `matrix_representation` function.
matrix_representation(hsym, H)


# Now, let's diagonalize the system.
# Since parity is conserved, we can work in the even and odd parity sectors separately.
# To do this, we can create two new Hilbert spaces for the even and odd sectors
Heven = hilbert_space(1:N, ParityConservation(1))
Hodd = hilbert_space(1:N, ParityConservation(-1))
heven = matrix_representation(hsym, Heven)
hodd = matrix_representation(hsym, Hodd)
# and then diagonalize each sector separately.
oddeigs = eigs(hodd; nev=1, which=:SR)
eveneigs = eigs(heven; nev=1, which=:SR)
oddvec = oddeigs[2][:, 1] # odd ground state
evenvec = eveneigs[2][:, 1] # even ground state
# The ground states are almost degenerate, as expected.
first(oddeigs[1]) - first(eveneigs[1])

# Now, we construct the ground state Majoranas.
# First, we need to pad the lowest energy odd and even states into the full Hilbert space.
o = zeros(eltype(oddvec), dim(H))
e = zeros(eltype(evenvec), dim(H))
o[indices(Hodd, H)] = oddvec
e[indices(Heven, H)] = evenvec

# Then, we can construct the ground state Majorana operators as
# γ = o * e' + hc
# γ̃ = 1im * o * e' + hc
# but that takes a lot of memory for large systems. We can use Kronecker.jl to avoid this
using Kronecker
oe = o ⊗ e'
ee = e ⊗ e'
oo = o ⊗ o'

# Now we can compute the reduction of the Majorana operators to each mode.
Hmodes = [hilbert_space(i:i) for i in 1:N]
eoR = [partial_trace(oe, H => Hmode) for Hmode in Hmodes];
eeR = [partial_trace(ee, H => Hmode) for Hmode in Hmodes];
ooR = [partial_trace(oo, H => Hmode) for Hmode in Hmodes];
γ_reductions = [norm(svdvals(eoR + hc), 1) for eoR in eoR]
γ̃_reductions = [norm(svdvals(1im * eoR + hc), 1) for eoR in eoR]
LD = [norm(svdvals(ooR - eeR), 1) for (ooR, eeR) in zip(ooR, eeR)]
# We can plot the reductions to visualize the localization of the Majorana modes.
##
lw = 4
legendfontsize = 15
marker = true
markerstrokewidth = 2
plot(xlabel="Site", title="Majorana quality measures"; frame=:box, size=(500, 300), xticks=1:3:N, yscale=:identity, legendfontsize, ylims=(-1e-1, 2), legendposition=:top, labelfontsize=15)
plot!(1:N, γ_reductions; label="‖γₙ‖", lw, legendfontsize, marker, markerstrokewidth)
plot!(1:N, γ̃_reductions; label="‖γ̃ₙ‖", lw, legendfontsize, marker, markerstrokewidth)
plot!(1:N, LD; label="‖(iγγ̃)ₙ‖", lw, legendfontsize, marker, markerstrokewidth)
