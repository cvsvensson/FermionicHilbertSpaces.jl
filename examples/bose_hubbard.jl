using FermionicHilbertSpaces
using Arpack, LinearAlgebra

N = 6
max_occupancy = 3
total_particles = N
@bosons b 1:N

Hs = hilbert_space.(values(b), max_occupancy)
H = tensor_product(Hs, NumberConservation(total_particles))

number_ops = [matrix_representation(b[i]'b[i], H) for i in 1:N]
hopping_ops = [matrix_representation(b[i]'b[i+1], H) for i in 1:(N-1)]

function bose_hubbard_observables(U, t)
    ham = -t * sum(b[i]'b[i+1] + hc for i in 1:(N-1)) +
          U * sum(b[i]'b[i] * (b[i]'b[i] - 1) for i in 1:N)
    M = matrix_representation(ham, H)
    _, vecs = eigs(M; nev=1, which=:SR)
    ψ = vecs[:, 1]
    occupation = sum(dot(ψ, ni, ψ) for ni in number_ops) / N
    fluctuation = sum(dot(ψ, ni^2, ψ) - dot(ψ, ni, ψ)^2 for ni in number_ops) / N
    coherence = sum(dot(ψ, hij, ψ) for hij in hopping_ops) / (N - 1)
    return (; occupation, fluctuation, coherence)
end

##
t = 1.0
Us = range(0, 20, length=10)
data = [bose_hubbard_observables(U, t) for U in Us]
##
occs = getproperty.(data, :occupation)
flucts = getproperty.(data, :fluctuation)
cohs = getproperty.(data, :coherence)
using Plots
plot(Us, flucts, xlabel="U", label="Average Fluctuations")
plot!(Us, occs, label="Average Occupation")
plot!(Us, cohs, label="Average Coherence")

