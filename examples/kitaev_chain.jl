# # Interacting Kitaev Chain Tutorial

# In this example, we study the interacting Kitaev chain.
# We show how the Hamiltonian can be constructed symbolically or by using matrix representations of the fermionic operators,
# and how to restrict the Hilbert space to a subspace of a given parity.
# We solve for the ground states and characterize the locality of the many-body Majoranas.

# We start by importing the necessary packages.
using FermionicHilbertSpaces, LinearAlgebra, Plots
using Arpack
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
import FermionicHilbertSpaces: indices, sector, quantumnumbers
(Eo, o), (Ee, e) = map(quantumnumbers(H)) do parity
    Hsec = sector(parity, H)
    ham = matrix_representation(hsym, Hsec)
    vals, vecs = eigs(ham; nev=1, which=:SR)
    inds = indices(Hsec, H)
    ground_state = zeros(eltype(vecs), dim(H))
    ground_state[inds] = vecs[:, 1]
    (; energy=first(vals), ground_state)
end
# The ground states are almost degenerate, as expected.
# Then, we can construct the ground state Majorana operators as
# `γ = o * e' + hc` and 
# `γ̃ = 1im * o * e' + hc`
# but that takes a lot of memory for large systems. We can use LowRankMatrices.jl to avoid this
using LowRankMatrices
γ = LowRankMatrix(o, e) + hc
γ̃ = 1im * LowRankMatrix(o, e) + hc
δρ = LowRankMatrix(o, o) - LowRankMatrix(e, e)

# Now we can compute the reduction of the Majorana operators to each mode.
Hmodes = [hilbert_space(i:i) for i in 1:N]
γR = [partial_trace(γ, H => Hmode) for Hmode in Hmodes];
γ̃R = [partial_trace(γ̃, H => Hmode) for Hmode in Hmodes];
δρR = [partial_trace(δρ, H => Hmode) for Hmode in Hmodes];
γ_reductions = [norm(svdvals(γ), 1) for γ in γR]
γ̃_reductions = [norm(svdvals(γ̃), 1) for γ̃ in γ̃R]
LD = [norm(svdvals(δρ), 1) for δρ in δρR]
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
