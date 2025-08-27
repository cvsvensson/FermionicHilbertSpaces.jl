using FermionicHilbertSpaces, LinearAlgebra, Plots, OrdinaryDiffEqTsit5
using Symbolics

H = majorana_hilbert_space([0, 1, 2, 3, 22, 33], ParityConservation())
@majoranas γ
@variables Δ[1:3]::Real
symbolic_ham = sum(1im * Δ[i] * γ[0] * γ[i] for i in 1:3)
ham = collect(matrix_representation(symbolic_ham, H))
construct_ham, construct_ham! = build_function(ham, Δ, expression=Val{false})
exchange_gate = matrix_representation(sqrt(1im / 2) * (I + γ[3] * γ[2]), H)
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
u0 = eigvecs(Matrix(matrix_representation(1.0im * γ[2] * γ[22] + 1.0im * γ[0] * γ[1], H) + parityoperator(H)))[:, 1]
# u0' * P2 * u0

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
plot(ts, map(norm ∘ sol, ts), label="norm", xlabel="t")
isapprox(abs2(sol[end]' * exchange_gate^2 * u0), 1, atol=1e-2)
isapprox(abs2(sol(T)' * exchange_gate * u0), 1, atol=1e-2)
## lets measure the parities
measurements = [matrix_representation(1.0im * γ[i] * γ[j], H) for (i, j) in [(2, 22), (3, 33)]]
plot(ts, [real(sol(t)'m * sol(t)) for m in measurements2, t in ts]', xlabel="t", label=["P2" "P3"], frame=:box, size=(400, 250), lw=2)

## Let's calculate the Non-abelian berry pase with the Kato method
const totalparity = parityoperator(H)
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
    # dP = (P2 - P1) / dt
    A = ((P2 - P1) * P1 - P1 * (P2 - P1)) / dt
    mul!(du, A, u)
end
##
U0 = Matrix{ComplexF64}(I, 8, 8)
kato_prob = ODEProblem(kato_ode!, U0, tspan, p)
ts = range(0, tspan[2], 100)
## Solve the ODE and check the norm
@time kato_sol = solve(kato_prob, Tsit5(); saveat=[0, T, 2T], reltol=1e-10, abstol=1e-10); #saveat=ts, reltol=1e-4)
kato_sol[end]' * kato_sol[end] ≈ I
1 ≈ dot(kato_sol[end], exchange_gate^2) / (dot(exchange_gate, exchange_gate)) |> abs
1 ≈ dot(kato_sol(T), exchange_gate) / (dot(exchange_gate, exchange_gate)) |> abs

