using FermionicHilbertSpaces
using Symbolics
using LinearAlgebra

@variables Δ t U μ
@fermions f
N = 14
H = hilbert_space(1:N)
fmat = fermions(H)
kitaev_chain(f, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:N-1) +
                                 sum(Δ * f[i] * f[i+1] + hc for i in 1:N-1) +
                                 sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:N-1) +
                                 sum(μ* f[i]' * f[i] for i in 1:N)

@time hsym = kitaev_chain(f, μ, t, Δ, U)
# @time hmat = kitaev_chain(fmat, μ, t, Δ, U) #very slow
@time hmat_sym = matrix_representation(hsym, H);
@profview hmat_sym = matrix_representation(hsym, H)
@time hnum = kitaev_chain(fmat, 1.0, 1.0, 1.0, 1.0)
@time hsymnum = kitaev_chain(f, 1.0, 1.0, 1.0, 1.0)
@btime hmatsymnum = matrix_representation(hsymnum, H);

build_function(hsym, (μ,t,Δ,U))