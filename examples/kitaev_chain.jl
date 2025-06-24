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

function extend(v, p)
    mapreduce(H -> H == first(p) ? v : zeros(size(H, 1)), vcat, last(p))
end
struct Rank1Matrix{T} <: AbstractMatrix{T}
    vec1::Vector{T}
    vec2::Vector{T}
end
Base.getindex(m::Rank1Matrix, i::Int, j::Int) = m.vec1[i] * conj(m.vec2[j])
Base.size(m::Rank1Matrix) = (length(m.vec1), length(m.vec2))

Hsum = (Hodd, Heven)
o = extend(oddvec, Hodd => Hsum) # odd ground state in the full Hilbert space
e = extend(evenvec, Heven => Hsum) # even ground state in the full Hilbert space
# Then, we can construct the ground state Majorana operators as
# γ = o * e' + hc
# γ̃ = 1im * o * e' + hc
yv = Rank1Matrix(o, e)
ee = Rank1Matrix(e, e)
oo = Rank1Matrix(o, o)

Hmodes = [hilbert_space(i:i) for i in 1:N]
@time eoR = [partial_trace(yv, H => Hmode) for Hmode in Hmodes];
@time eeR = [partial_trace(ee, H => Hmode) for Hmode in Hmodes];
@time ooR = [partial_trace(oo, H => Hmode) for Hmode in Hmodes];
γ_reductions = [norm(svdvals(eoR + hc), 1) for eoR in eoR]
γ̃_reductions = [norm(svdvals(1im * eoR + hc), 1) for eoR in eoR]
LD = [norm(svdvals(ooR - eeR), 1) for (ooR, eeR) in zip(ooR, eeR)]
# Now we can compute the reduction of the Majorana operators to each mode.
# γ_reductions = [norm(svdvals(partial_trace(γ, H => Hmode)), 1) for Hmode in Hmodes]
# γ̃_reductions = [norm(svdvals(partial_trace(γ̃, H => Hmode)), 1) for Hmode in Hmodes]
# LD = [norm(svdvals(partial_trace(γ * γ̃, H => Hmode)), 1) for Hmode in Hmodes]
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
