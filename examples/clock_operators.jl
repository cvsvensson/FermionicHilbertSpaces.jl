# # Defining Clock operators 
# 
# This example defines the ``Z_D`` clock operators and connects it to the
# generic Hilbert-space and matrix-representation machinery. It then uses the
# global clock charge to diagonalise a Hamiltonian sector by sector and to
# resolve the entanglement entropy of a subsystem.
#
# The pieces a new algebra needs are listed in [Extending and performance hooks](../extending.md).

using FermionicHilbertSpaces, LinearAlgebra, Plots, Arpack
import FermionicHilbertSpaces: @field, AbstractBasisState, Op, apply_local_operator, mat_eltype

# ## Defining the clock operators and states
# The clock operators are defined on a local Hilbert space of dimension ``D``, with basis states ``|n\rangle`` for ``n = 0, 1, ..., D-1``. The operator ``X`` shifts the local clock value, ``X |n\rangle = |n + 1\rangle``, and ``Z`` measures the local clock value: ``Z |n\rangle = \exp(2π i n / D) |n\rangle``.
struct Clock{T} # The kind of a clock operator: T is :X or :Z.
    adjoint::Bool
end
Clock(T::Symbol) = Clock{T}(false)
Base.adjoint(k::Clock{T}) where {T} = Clock{T}(!k.adjoint)
## The sites are the modes `c[i]` of a field of clock operators, and `c[i][:X]` is the operator of kind `Clock(:X)` on `c[i]`.
@field c::Clock
## Operators print as they are written: `c[1][:X]` and `c[1][:X]'`.
Base.show(io::IO, op::Op{Clock{T}}) where {T} = print(io, op.mode, "[", repr(T), "]", op.kind.adjoint ? "'" : "")
## Products on the same site are kept as they are. Operators on different sites commute and are sorted by site.
struct ClockState{D} <: AbstractBasisState # A clock state with dimension D.
    n::Int
    ClockState{D}(n) where D = new{D}(mod(n, D))
end
## Action on local basis states.
function apply_local_operator(op::Clock{:X}, state::ClockState{D}, space, precomp) where D
    newstate = ClockState{D}(state.n + (op.adjoint ? -1 : 1))
    amp = 1
    newstate, amp
end
function apply_local_operator(op::Clock{:Z}, state::ClockState{D}, space, precomp) where D
    amp = exp((op.adjoint ? -1 : 1) * 2π * im * state.n / D)
    state, amp
end
## Clock operators have complex matrix elements.
mat_eltype(::Type{<:Clock}) = ComplexF64

# ## Using it in practice: Hamiltonian and conservation laws
# Let's take the case D = 3 for 6 sites. We define a hamiltonian which conserves the clock charge, the sum of the local clock values modulo D, and calculate the symmetry resolved entanglement entropy in the ground state.
D = 3
L = 6
states = [ClockState{D}(n) for n in 0:(D-1)]
Hfull = hilbert_space(c, 1:L, states)
clock_hamiltonian(J, h) = -J * sum(c[i][:X]' * c[i + 1][:X] + hc for i in 1:(L - 1)) - h * sum(c[i][:Z] + hc for i in 1:L)

# The clock charge, the sum of the local clock values mod D, divides the Hilbert space into sectors.
clock_charge(state) = mod(sum(s.n for s in state.states), D)
Hzd = constrain_space(Hfull, clock_charge)
Hhalf = hilbert_space(c, 1:div(L, 2), states, clock_charge)

# Functions to calculate entropies.
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
    (; Hcharge, Swithin, Stotal)
end
# We now calculate and plot the symmetry resolved entanglement entropy in the ground state of the hamiltonian for a range of values of ``h``
Hgs = sector(0, Hzd)
J = 1.0
hs = range(0.05, 2.5; length=30)
ptmap = partial_trace(Hgs => Hhalf)
r = map(hs) do h
    hmat = representation(clock_hamiltonian(J, h), Hgs)
    vals, vecs = eigs(Hermitian(hmat), nev=1, which=:SR)
    ρhalf = ptmap(vecs[:, 1])
    symmetry_resolved_entropy(ρhalf, Hhalf, D)
end

p1 = plot(
    hs ./ J, getproperty.(r, :Stotal), lw = 3, c = :black, label = "total  S(A)",
    xlabel = "h/J", ylabel = "entropy", size = (400, 250),
    title="Half-chain entanglement", yscale=:log10, legend=:bottom)
plot!(p1, hs ./ J, getproperty.(r, :Hcharge), lw = 2, label = "charge entropy  H(pₛ)")
plot!(p1, hs ./ J, getproperty.(r, :Swithin), lw = 2, label = "within-sector  ΣₛpₛSₛ")

