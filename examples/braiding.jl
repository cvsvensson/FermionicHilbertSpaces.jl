# This example demonstrates how to combine symbolic Majoranas with symbolic parameters to create a symbolic matrix representation
# We compile that matrix to a fast function for repeated use.
# We use it to calculate the time evolution of a Majorana braiding protocol
using FermionicHilbertSpaces
using Symbolics, LinearAlgebra, Plots, OrdinaryDiffEqTsit5

@majoranas γ
H = hilbert_space(γ, [0, 1, 2, 3, 22, 33], ParityConservation())
@variables Δ[1:3]::Real
symbolic_ham = sum(1im * Δ[i] * γ[0] * γ[i] for i in 1:3)
ham = representation(symbolic_ham, H, :dense)
construct_ham, construct_ham! = build_function(ham, Δ, expression=Val{false})
exchange_gate = representation(sqrt(1 / 2) * (I + γ[3] * γ[2]), H)
##
smooth_step(x, k) = 1 / 2 + tanh(k * x) / 2
# Give the value of the three deltas at time t in the three point majorana braiding protocol
function braiding_deltas(t, T, Δmax, Δmin, k)
    Δ1 = Δtrajectory(t, T, Δmax, Δmin, k)
    Δ2 = Δtrajectory(t - T / 3, T, Δmax * 0.85, Δmin, k)
    Δ3 = Δtrajectory(t - 2T / 3, T, Δmax * 0.7, Δmin, k)
    return Δ1, Δ2, Δ3
end
function Δtrajectory(t, T, Δmax, Δmin, k)
    dΔ = Δmax - Δmin
    Δmin + dΔ * smooth_step(cos(2pi * t / T), k)
end

##
function hamiltonian((T, Δmax, Δmin, k), t)
    Δs = braiding_deltas(t, T, Δmax, Δmin, k)
    ham = zeros(ComplexF64, 8, 8)
    construct_ham!(ham, Δs)
end
function drho!(du, u, p, t)
    ham = hamiltonian(p, t)
    mul!(du, ham, u, 1im, 0)
    return du
end
##
u0 = eigvecs(representation(1.0im * γ[2] * γ[22] + 1.0im * γ[0] * γ[1], H, :dense) + parityoperator(H))[:, 1]

T = 1000
k = 20
Δmax = 1
Δmin = 0
p = (T, Δmax, Δmin, k)
tspan = (0.0, 2T)

prob = ODEProblem(drho!, u0, tspan, p)
ts = range(0, tspan[2], 500)
deltas = stack([braiding_deltas(t, p...) for t in ts])'
plot(ts, deltas, label=["Δ1" "Δ2" "Δ3"], xlabel="t")
## Plot spectrum
spectrum = stack([eigvals(hamiltonian(p, t)) for t in ts])'
plot(ts, spectrum, title="Energies")
## Solve the ODE and check the norm
@time sol = solve(prob, Tsit5(), reltol=1e-8)
@assert all(isapprox(1; atol=1e-3) ∘ norm ∘ sol, ts)
@assert isapprox(abs(dot(sol(2T), exchange_gate^2, u0)), 1, atol=1e-2)
@assert isapprox(abs(dot(sol(T), exchange_gate, u0)), 1, atol=1e-2)
## Measure the parities
measurements = [representation(1.0im * γ[i] * γ[j], H) for (i, j) in [(2, 22), (3, 33)]]
plot(ts, [real(sol(t)'m * sol(t)) for m in measurements, t in ts]', xlabel="t", label=["P2" "P3"], frame=:box, size=(400, 250), lw=2)

## Calculate the Non-abelian berry pase with the Kato method
function ground_state_projector(t, p)
    ham = hamiltonian(p, t)
    vecs = eigvecs(ham)
    ground_states = vecs[:, 1:4]
    return ground_states * ground_states'
end

function kato_ode!(du, u, p, t)
    P1 = ground_state_projector(t, p)
    T = p[1]
    dt = T / 1e4
    P2 = ground_state_projector(t + dt, p)
    A = ((P2 - P1) * P1 - P1 * (P2 - P1)) / dt
    mul!(du, A, u)
end
##
U0 = Matrix{ComplexF64}(I, 8, 8)
kato_prob = ODEProblem(kato_ode!, U0, tspan, p)
ts = range(0, tspan[2], 100)
## Solve the ODE and check the norm
@time kato_sol = solve(kato_prob, Tsit5(); saveat=[0, T, 2T], reltol=1e-10, abstol=1e-10);
kato_sol(2T)' * kato_sol(2T) ≈ I
N = dot(exchange_gate, exchange_gate) 
1 ≈ dot(kato_sol(2T), exchange_gate^2) / N
1 ≈ dot(kato_sol(T), exchange_gate) / N

