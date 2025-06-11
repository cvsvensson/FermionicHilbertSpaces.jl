# # Interacting Kitaev Chain Tutorial

# In this example, we study the interacting Kitaev chain.
# We show how the Hamiltonian can be constructed symbolically or by using matrix representations of the fermionic operators,
# and how to restrict the Hilbert space to a subspace of a given parity.
# We solve for the ground states and characterize the locality of the many-body Majoranas.

# We start by importing the necessary packages.
using FermionicHilbertSpaces, LinearAlgebra, Plots

# Then we define the Hilbert space with `N` sites and parity conservation.
N = 10
H = hilbert_space(1:N, ParityConservation())

# Fermions are defined either as symbolic fermions `f` using the `@fermions` macro,
@fermions fsym
# or as sparse matrix representations `fmat` using the `fermions` function, taking H as argument.
fmat = fermions(H)

# Let's define the interacting Kitaev chain Hamiltonian.
# It is a function of the fermions `f` and parameters `N`, `μ`, `t`, `Δ`, and `U`,
# representing the number of sites, chemical potential, hopping amplitude, pairing amplitude, and interaction strength, respectively.
# Here, f can be either symbolic (`fsym`) or a matrix representation (`fmat`).
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
params = (μ, t, Δ, U)

# We can now construct the Hamiltonian using symbolic fermions for a symbolic representation
hsym = kitaev_chain(fsym, N, μ, t, Δ, U)
# or using matrix representations for the matrix representation.
hmat = kitaev_chain(fmat, N, μ, t, Δ, U)
# To convert the symbolic Hamiltonian to a matrix representation, we can use the `matrix_representation` function.
matrix_representation(hsym, H) ≈ hmat # true


# Now, let's diagonalize the system.
# Since parity is conserved, we can work in the even and odd parity sectors separately.
# To do this, we can create two new Hilbert spaces for the even and odd sectors
Heven = hilbert_space(1:N, ParityConservation(1))
Hodd = hilbert_space(1:N, ParityConservation(-1))
heven = matrix_representation(hsym, Heven)
hodd = matrix_representation(hsym, Hodd)
# and then diagonalize each sector separately.
oddvec = eigvecs(Matrix(hodd))[:, 1] # odd ground state
evenvec = eigvecs(Matrix(heven))[:, 1] # even ground state
# The ground states are almost degenerate, as expected.
first(eigvals(Matrix(hodd))) - first(eigvals(Matrix(heven))) # ≈ 10^-4

# Now, we construct the ground state Majoranas.
# First, we need to embed the lowest energy odd and even states into the full Hilbert space.
# We can do this by defining a `DirectSum` type that holds the spaces of the odd and even sectors.
struct DirectSum{HS}
    spaces::HS
end
# Then, we can extend the odd and even vectors to the full Hilbert space.
function extend(v, p=Pair{<:AbstractHilbertSpace,<:DirectSum})
    mapreduce(H -> H == first(p) ? v : zeros(size(H, 1)), vcat, last(p).spaces)
end

Hsum = DirectSum((Hodd, Heven))
o = extend(oddvec, Hodd => Hsum) # odd ground state in the full Hilbert space
e = extend(evenvec, Heven => Hsum) # even ground state in the full Hilbert space
# Then, we can construct the ground state Majorana operators as
γ = o * e' + hc
γ̃ = 1im * o * e' + hc

# Finally, we can check the locality of the Majorana operators.
# This is done by tracing down the Majorana operators to each mode.
# So let's first define the Hilbert space for each mode.
Hmodes = [hilbert_space(i:i) for i in 1:N]
# Now we can compute the reduction of the Majorana operators to each mode.
γ_reductions = [norm(partial_trace(γ, H => Hmode)) for Hmode in Hmodes]
γ̃_reductions = [norm(partial_trace(γ̃, H => Hmode)) for Hmode in Hmodes]
# We can plot the reductions to visualize the localization of the Majorana modes.
import DisplayAs
plot(xlabel="Site", ylabel="||γᵢ|| / √2", title="Majorana Locality", frame=:box, size=(500, 300))
plot!(1:N, γ_reductions / sqrt(2), label="γ", lw=2)
DisplayAs.PNG(plot!(1:N, γ̃_reductions / sqrt(2), label="γ̃", lw=2))

