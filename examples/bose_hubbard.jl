using FermionicHilbertSpaces, Arpack

N = 6
max_occupancy = 3
total_particles = N
@bosons b 1:N

Hs = hilbert_space.(values(b), max_occupancy)
Hfull = tensor_product(Hs)
H = constrain_space(Hfull, NumberConservation(total_particles))

function bose_hubbard_observables(U, t, N, H, nis, bijs)
    ham = -t * sum(b[i]'b[i+1] + hc for i in 1:(N-1)) +
          U * sum(b[i]'b[i] * (b[i]'b[i] - 1) for i in 1:N)
    M = matrix_representation(ham, H)
    _, vecs = eigs(M; nev=1, which=:SR)
    ψ = vecs[:, 1]
    occ = sum(ψ' * ni * ψ for ni in nis) / N
    fluct = sum(ψ' * ni^2 * ψ - (ψ' * ni * ψ)^2 for ni in nis) / N
    coh = sum(ψ' * bij * ψ for bij in bijs) / (N - 1)
    return (; occ, fluct, coh)
end
nis = [matrix_representation(b[i]'b[i], H) for i in 1:N]
bijs = [matrix_representation(b[i]'b[i+1], H) for i in 1:(N-1)]

t = 1.0
Us = range(0, 20, length=10)
data = [bose_hubbard_observables(U, t, N, H, nis, bijs) for U in Us]

occs   = getproperty.(data, :occ)
flucts = getproperty.(data, :fluct)
cohs   = getproperty.(data, :coh)
using Plots
plot(Us, flucts, xlabel="U", label="Average Fluctuations")
plot!(Us, occs, label="Average Occupation")
plot!(Us, cohs, label="Average Coherence")

