using FermionicHilbertSpaces
using LinearAlgebra, Plots

@boson a
@spin s

Ha = hilbert_space(a, 16)
Hs = hilbert_space(s, 1//2)
H  = tensor_product(Hs, Ha)
n_photon = matrix_representation(a'a, H)

# The Hamiltonian of a cavity coupled to a double quantum dot (DQD)
# containing a single electron can be written as
cavity_dqd_ham(ωr, ϵ, t, g) =
    ωr * a' * a +
    ϵ * s[:z] + t * s[:x] +
    g * (a' + a) * s[:z]
# where `ωr` is the cavity frequency, `ϵ` is the detuning of the DQD,
# `t` is the inter-dot tunneling amplitude, and `g` is the coupling strength.
# The DQD is modeled as a two-level system with Pauli operators `s[:z]` and `s[:x]` in the |L>, |R> basis,
# while the cavity mode is described by the bosonic operator `a`.
# The coupling term `g * (a' + a) * s[:z]` represents the interaction between the cavity field
# and the dipole moment of the DQD.

function cavity_dqd_eigensystem(ωr, ϵ, t, g)
    Hsym = cavity_dqd_ham(ωr, ϵ, t, g)
    M = matrix_representation(Hsym, H)
    vals, vecs = eigen(Hermitian(Matrix(M)))
    return vals, vecs
end

function branch_data(ωr, ϵ, t, g; branches=2:3)
    vals, vecs = cavity_dqd_eigensystem(ωr, ϵ, t, g)
    splittings = vals[branches] .- vals[1]
    photon_numbers = [dot(ψ, n_photon, ψ) for ψ in eachcol(vecs[:, branches])]
    return (; splittings, photon_numbers)
end

ωr = 1.0
t  = 0.5
g  = 0.1

# For H = ϵ s_z + t s_x the qubit splitting is sqrt(ϵ^2 + t^2)
ϵ_res = sqrt(ωr^2 - t^2)
δϵs = range(-3g, 3g, length=50)
data = [branch_data(ωr, ϵ_res + δϵ, t, g) for δϵ in δϵs]
splittings = reduce(hcat, getproperty.(data, :splittings))'
photon_numbers = reduce(hcat, getproperty.(data, :photon_numbers))'

plot(
    δϵs,
    splittings,
    label = "",
    line_z = photon_numbers,
    color = :viridis,
    xlabel = "ϵ - ϵ_res",
    ylabel = "Energy splitting",
    linewidth = 3,
    colorbar_title = "Photon number",
)
