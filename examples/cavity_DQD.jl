using FermionicHilbertSpaces
using LinearAlgebra, Plots

@fermions f
@boson a

Ha = hilbert_space(a, 8)
Hf = hilbert_space(f, [(j, σ) for j in 1:2 for σ in [:↑, :↓]], NumberConservation(1))
H = tensor_product(Hf, Ha)
n_photon = matrix_representation(a'a, H)
n_f(j, σ) = f[j, σ]'f[j, σ]

h_dqd(; δϵ, t, α, U, B) = sum(δϵ * (n_f(1, σ) - n_f(2, σ)) for σ in [:↑, :↓]) +
                        B * sum(n_f(j, :↑) - n_f(j, :↓) for j in 1:2) +
                        t * sum(f[1, σ]'f[2, σ] + hc for σ in [:↑, :↓]) +
                        t * α * (f[1, :↓]'f[2, :↑] - f[1, :↑]'f[2, :↓] + hc) +
                        U * sum(n_f(j, :↑) * n_f(j, :↓) for j in 1:2)

h_cavity_dqd(; ωr, δϵ, t, α, U, B, g) = ωr * a' * a + h_dqd(; δϵ, t, α, U, B) +
                                            g * (a' + a) * sum(n_f(1, σ) - n_f(2, σ) for σ in [:↑, :↓])

ωr = 1.0
t  = 0.2
α = 0.2
U  = 0.0
g  = 0.1
B = 0.2
fix_params = (; ωr, t, α, U, B, g)

function cavity_dqd_eigensystem(; ωr, δϵ, t, α, U, B, g)
    Hsym = h_cavity_dqd(; ωr, δϵ, t, α, U, B, g)
    M = matrix_representation(Hsym, H)
    vals, vecs = eigen(Hermitian(Matrix(M)))
    return vals, vecs
end

function branch_data(; ωr, δϵ, t, α, U, B, g, branches=3:6)
    vals, vecs = cavity_dqd_eigensystem(; ωr, δϵ, t, α, U, B, g)
    splittings = vals[branches] .- vals[1]
    photon_numbers = [dot(ψ, n_photon, ψ) for ψ in eachcol(vecs[:, branches])]
    return (; splittings, photon_numbers)
end

δϵ_res = sqrt(ωr^2/4 - t^2)
δϵs = range(-3g, 3g, length=50)
data = [branch_data(;ωr, δϵ=δϵ_res + δϵ, t, α, U, B, g) for δϵ in δϵs]
splittings = reduce(hcat, getproperty.(data, :splittings))'
photon_numbers = reduce(hcat, getproperty.(data, :photon_numbers))'

plot(
    δϵ_res .+ δϵs,
    splittings,
    label = "",
    line_z = photon_numbers,
    color = :viridis,
    xlabel = "δϵ",
    ylabel = "Energy splitting",
    linewidth = 3,
    colorbar_title = "Photon number",
)
