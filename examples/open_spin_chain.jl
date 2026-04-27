using FermionicHilbertSpaces
using LinearAlgebra
using OrdinaryDiffEqTsit5
using Plots

# Reproduction of QuTiP tutorial 005 (spin chain)
# https://nbviewer.org/urls/qutip.org/qutip-tutorials/tutorials-v5/time-evolution/005_spin-chain.ipynb

# Model parameters
N = 5
h = fill(2pi, N)
Jx = fill(0.2pi, N - 1)
Jy = fill(0.2pi, N - 1)
Jz = fill(0.2pi, N - 1)
γ_deph = fill(0.02, N)

tspan = (0.0, 100.0)
ts = range(tspan[1], tspan[2], length=200)

@spins S 1 // 2
H = hilbert_space(S, 1:N)

ham_sym = -sum(h[k] * S[k][:z] for k in 1:N) -
          2 * sum(Jx[k] * S[k][:x] * S[k+1][:x] +
                  Jy[k] * S[k][:y] * S[k+1][:y] +
                  Jz[k] * S[k][:z] * S[k+1][:z] for k in 1:(N-1))

ham_mat = matrix_representation(ham_sym, H)
Sz_mats = [matrix_representation(-2 * S[k][:z], H) for k in 1:N]

using FermionicHilbertSpaces: SpinState, ProductState, state_index
state_ind = state_index(ProductState(SpinState((-1)^(n > 1) * 1 // 2) for n in 1:N), H)
ψ0 = zeros(ComplexF64, dim(H))
ψ0[state_ind] = 1.0

prob_closed = ODEProblem(MatrixOperator(-1im * ham_mat), ψ0, tspan)
tols = (reltol=1e-6, abstol=1e-6)
sol_closed = solve(prob_closed, Tsit5(); saveat=ts, tols...)

expvals_closed = [real.(map(ψ -> ψ' * mz * ψ, sol_closed.u)) for mz in Sz_mats]
max_norm_error = maximum(abs.(map(ψ -> norm(ψ) - 1.0, sol_closed.u)))
println("Closed-system max norm error: ", max_norm_error)

# Open-system dephasing with the package's left/right interface.
using FermionicHilbertSpaces: open_system
Hopen, Hleft, Hright, left, right = open_system(S, 1:N)

lindbladian_sym = 1im * (left(ham_sym) - right(ham_sym))
dissipator(op) = left(op) * right(op) - 0.5 * (left(op)' * left(op) + right(op)' * right(op))
for k in 1:N
    jump_op = 2 * sqrt(γ_deph[k]) * S[k][:z]
    lindbladian_sym += dissipator(jump_op)
end

lindbladian_mat = matrix_representation(lindbladian_sym, Hopen)

ρ0 = ψ0 * ψ0'
ρ0_vec = reshape(ρ0, (Hleft, Hright) => Hopen)

prob_open = ODEProblem(MatrixOperator(lindbladian_mat), ρ0_vec, tspan)
sol_open = solve(prob_open, Tsit5(); saveat=ts, tols...)

to_mat(v) = reshape(v, Hopen => (Hleft, Hright))
expvals_open = [real.(map(v -> tr(Sz_mats[k] * to_mat(v)), sol_open.u)) for k in 1:N]
trace_errors = map(v -> abs(real(tr(to_mat(v))) - 1.0), sol_open.u)
println("Open-system max trace error: ", maximum(trace_errors))

# Plot edge spin observables
p1 = plot(ts, expvals_closed[[1, end]],
    label=["<Sz_1> (closed)" "<Sz_N> (closed)"],
    xlabel="time", lw=2, frame=:box)

p2 = plot(ts, expvals_open[[1, end]],
    label=["<Sz_1> (dephasing)" "<Sz_N> (dephasing)"],
    xlabel="time", lw=2, frame=:box)

plot(p1, p2; layout=(1, 2), size=0.7 .* (1100, 420))

