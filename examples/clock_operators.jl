# # Clock operators and symmetry-resolved entanglement
# 
# This example defines a custom ``Z_D`` clock algebra and connects it to the
# generic Hilbert-space and matrix-representation machinery. It then uses the
# global clock charge to diagonalise a Hamiltonian sector by sector and to
# resolve the entanglement entropy of a subsystem.

using FermionicHilbertSpaces, LinearAlgebra
import FermionicHilbertSpaces:
    AbstractBasisState,
    AbstractSym,
    GenericHilbertSpace,
    apply_local_operator,
    mat_eltype,
    symbolic_group,
    SectorConstraint
import FermionicHilbertSpaces.NonCommutativeProducts: @nc, mul_effect

# ## Defining the clock algebra
# 
# ``X`` shifts the local clock value and ``Z`` measures it. Their local
# algebra is ``ZX = exp(2πim/D) XZ``. The `adjoint` flag lets the same type
# represent both directions of the shift and both powers of the phase.
struct ClockOp{T} <: AbstractSym # A clock operator of type T (:X or :Z).
    site::Int
    adjoint::Bool
end
Base.adjoint(op::ClockOp{T}) where T = ClockOp{T}(op.site, !op.adjoint)
X(i) = ClockOp{:X}(i, false)
Z(i) = ClockOp{:Z}(i, false)
@nc ClockOp
## No symbolic simplification when multiplying clock operators.
mul_effect(::ClockOp, ::ClockOp) = nothing
struct ClockState{D} <: AbstractBasisState # A clock state with dimension D.
    n::Int
    ClockState{D}(n) where D = new{D}(mod(n, D))
end
## Action on local basis states.
apply_local_operator(op::ClockOp{:X}, state::ClockState{D}, _, _) where D =
    ClockState{D}(op.adjoint ? (state.n - 1) : state.n + 1), 1
apply_local_operator(op::ClockOp{:Z}, state::ClockState{D}, _, _) where D =
    state, exp((op.adjoint ? -1 : 1) * 2π * im * state.n / D)
## Clock operators have complex matrix elements.
mat_eltype(::Type{<:ClockOp}) = ComplexF64
## Associate operators with Hilbert-space factors.
symbolic_group(op::ClockOp) = op.site

# This is enough to get most of the functionality of this package.
# ## Hamiltonian and conservation laws
# Let's take the case D = 3 for 6 sites. We define a hamiltonian which conserves the particle number modulo D, and calculate the symmetry resolved entanglement entropy in the ground state.
D = 4
L = 6
Hs = [GenericHilbertSpace(k, ClockState{D}.(0:(D-1))) for k in 1:L]
Hfull = tensor_product(Hs)
clock_hamiltonian(J, h) = -J * sum(X(i)' * X(i + 1) + hc for i in 1:(L-1)) - h * sum(Z(i) + hc for i in 1:L)

# Define a constraint and a function which divides the hilbert space up into sectors, and some functions to calculate the entropy and the ground state
function clock_sectors(H)
    constraint = SectorConstraint(values -> mod(sum(values), D),
        state -> state.n, factors(H))
    return constrain_space(H, constraint)
end
Hzd = clock_sectors(Hfull)
Hhalf = clock_sectors(tensor_product(Hs[1:div(L, 2)]))
using Arpack
function sector_ground_state(q, J, h)
    Hq = sector(q, Hzd)
    # hmat = representation(clock_hamiltonian(J, h), Hq, :dense)
    hmat = (representation(clock_hamiltonian(J, h), Hq))
    vals, vecs = eigs(Hermitian(hmat), nev=1, which=:SR)
    return Hq, vals[1], vecs[:, 1]
end

##
# Replace the final calculation/printing block with this.
# S(A) = H(pₛ) + Σₛ pₛ Sₛ: charge-sharing entropy plus
# entanglement remaining within a fixed subsystem charge sector.

shannon(p; tol=1e-12) = -sum(x * log(x) for x in p if x > tol)
von_neumann(ρ) = shannon(eigvals(Hermitian(ρ)))

function symmetry_resolved_entropy(ρ, Hsub, D; tol=1e-12)
    p, Sq = zeros(D), zeros(D)
    for q in 0:(D-1)
        inds = indices(q, Hsub)
        block = ρ[inds, inds]
        p[q+1] = real(tr(block))
        p[q+1] > tol && (Sq[q+1] = von_neumann(block / p[q+1]))
    end
    Hcharge = shannon(p)
    Swithin = dot(p, Sq)
    Stotal = von_neumann(ρ)
    @assert isapprox(Stotal, Hcharge + Swithin; atol=1e-6)
    (; p, Sq, Hcharge, Swithin, Stotal)
end

ptmap = partial_trace(sector(0, Hzd) => Hhalf)
hs = range(0.05, 2.5; length=30)
@profview @time r = map(hs) do h
    _, _, ψ0 = sector_ground_state(0, 1.0, h)
    ρhalf = ptmap(ψ0)
    symmetry_resolved_entropy(ρhalf, Hhalf, D)
end

S = getproperty.(r, :Stotal)
Hq = getproperty.(r, :Hcharge)
Sw = getproperty.(r, :Swithin)
p = getproperty.(r, :p)
Sq = getproperty.(r, :Sq)

# Top: what makes up the entanglement?  Bottom: the individual charge sectors.
p1 = plot(hs, S, lw=3, c=:black, label="total  S(A)",
    xlabel="h / J", ylabel="entropy",
    title="Half-chain entanglement")
plot!(p1, hs, Hq, lw=2, c=:dodgerblue, label="charge entropy  H(pₛ)")
plot!(p1, hs, Sw/maximum(Sw), lw=2, c=:orangered, label="within-sector  ΣₛpₛSₛ")
vline!(p1, [1], ls=:dash, c=:gray, label=false)

p2 = plot(hs, getindex.(p, 1), lw=3, c=:seagreen, label="p₀",
    xlabel="h / J", ylabel="probability / entropy",
    title="Resolved charge sectors")
plot!(p2, hs, getindex.(p, 2), lw=3, c=:mediumpurple, label="p₁ = p₂")
plot!(p2, hs, getindex.(Sq, 1), lw=2, ls=:dash, c=:seagreen, label="S₀")
plot!(p2, hs, getindex.(Sq, 2), lw=2, ls=:dash, c=:mediumpurple, label="S₁ = S₂")
vline!(p2, [1], ls=:dash, c=:gray, label=false)

plot(p1, p2; layout=(1, 2), size=(800, 300))

