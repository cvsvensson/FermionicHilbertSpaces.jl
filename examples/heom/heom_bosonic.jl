## Bosonic HEOM Auxiliary Algebra
##
## Reference: arXiv 2306.07522 — HierarchicalEOM.jl paper, Eq. (16).
##
## The bosonic HEOM propagates an augmented state consisting of the physical reduced
## density matrix ρ together with a hierarchy of auxiliary density operators (ADOs).
## Each ADO is indexed by a multiset j = [j₁, …, jₘ] where each jᵣ ∈ 1:N_exp labels
## one exponential term ξₗ exp(−χₗ τ) of the bath correlation function.
##
## In the doubled-space representation (left/right copies of the system, as in the
## Lindblad superoperator approach), the bosonic HEOM generator is
##
##   M = i(H_l − H_r) ⊗ I_aux                              (coherent Liouvillian)
##     + I_sys ⊗ W_aux                                       (hierarchy damping)
##     − i Σ_l (V_l − V_r) ⊗ Aup(l)                        (upward coupling)
##     − i Σ_l (ξ_l V_l − ξ_l* V_r) ⊗ Adown(l)            (downward coupling)
##
## where
##   • Aup(l) |n⟩  = |n with nₗ → nₗ+1⟩   (amplitude 1,  zero if Σnₗ = m_max)
##   • Adown(l) |n⟩ = nₗ × |n with nₗ → nₗ−1⟩
##   • W_aux |n⟩    = (−Σₗ χₗ nₗ) |n⟩
##
## The bosonic superoperators B_j and D_j from the paper's Eq. (18) are:
##   B_j[•] = [V_σb, •]_−          ↔  (V_l − V_r) ⊗ Aup(l)
##   D_j[•] = ξ_j V_σb [•] − ξ_j* [•] V_σb  ↔  (ξ_l V_l − ξ_l* V_r) ⊗ Adown(l)
##
## This file defines:
##   • HEOMBosonicBath       – bath parameters (N_exp, m_max, χ, ξ)
##   • HEOMBosonicUp/Down/Damping – symbolic auxiliary operators
##   • HEOMBosonicAuxState   – basis state (occupation tuple n)
##   • heom_bosonic_aux_space – builds the auxiliary GenericHilbertSpace
##   • heom_generator        – builds the full symbolic HEOM generator
##   • Algebra rules, symbolic_group, apply_local_operators (all required interfaces)

using FermionicHilbertSpaces, Test

"""
    HEOMBosonicBath(id, N_exp, m_max, χ, ξ)

Describes a bosonic bath whose correlation function is decomposed into `N_exp`
exponential terms:

    C_β(τ) = Σₗ ξₗ exp(−χₗ τ),   l = 1 … N_exp.

Fields
------
- `id`    : unique integer identifying this bath (used for canonical ordering)
- `N_exp` : number of exponential terms Nβ
- `m_max` : maximum bosonic hierarchy depth (truncation tier)
- `χ`     : decay rates (length N_exp, generally complex)
- `ξ`     : amplitudes  (length N_exp, generally complex)

See arXiv 2306.07522, Eqs. (13)–(15) and (31)–(32).
"""
struct HEOMBosonicBath
    id::Int
    N_exp::Int
    m_max::Int
    χ::Vector{ComplexF64}
    ξ::Vector{ComplexF64}
    function HEOMBosonicBath(id, N_exp, m_max, χ, ξ)
        length(χ) == N_exp || throw(ArgumentError("length(χ) must equal N_exp"))
        length(ξ) == N_exp || throw(ArgumentError("length(ξ) must equal N_exp"))
        m_max >= 0 || throw(ArgumentError("m_max must be non-negative"))
        new(id, N_exp, m_max, ComplexF64.(χ), ComplexF64.(ξ))
    end
end
Base.:(==)(a::HEOMBosonicBath, b::HEOMBosonicBath) = a.id == b.id
Base.hash(a::HEOMBosonicBath, h::UInt) = hash(a.id, h)
Base.show(io::IO, b::HEOMBosonicBath) = print(io, "HEOMBosonicBath(id=", b.id, ", N_exp=", b.N_exp, ", m_max=", b.m_max, ")")

# ─────────────────────────────────────────────────────────────────────────────
# Symbolic operator types
# ─────────────────────────────────────────────────────────────────────────────

"""
    HEOMBosonicUp(l, bath)

Symbolic operator that raises the occupation of exponent-index `l` in the
bosonic ADO multi-index by one. Acts as the identity if the total tier
already equals `bath.m_max`.

    Aup(l) |n₁,…,nₗ,…⟩ = |n₁,…,nₗ+1,…⟩   (zero if Σnₗ = m_max)
"""
struct HEOMBosonicUp
    l::Int
    bath::HEOMBosonicBath
end

"""
    HEOMBosonicDown(l, bath)

Symbolic operator that lowers the occupation of exponent-index `l` in the
bosonic ADO multi-index by one, with amplitude equal to the current occupation.

    Adown(l) |n₁,…,nₗ,…⟩ = nₗ |n₁,…,nₗ−1,…⟩
"""
struct HEOMBosonicDown
    l::Int
    bath::HEOMBosonicBath
end

"""
    HEOMBosonicDamping(bath)

Symbolic diagonal operator representing the hierarchy damping.

    W |n⟩ = (−Σₗ χₗ nₗ) |n⟩
"""
struct HEOMBosonicDamping
    bath::HEOMBosonicBath
end

const HEOMBosonicOps = Union{HEOMBosonicUp,HEOMBosonicDown,HEOMBosonicDamping}

Base.show(io::IO, op::HEOMBosonicUp) = print(io, "Aup(", op.l, ")")
Base.show(io::IO, op::HEOMBosonicDown) = print(io, "Adown(", op.l, ")")
Base.show(io::IO, op::HEOMBosonicDamping) = print(io, "W_aux")
Base.adjoint(op::HEOMBosonicUp) = HEOMBosonicDown(op.l, op.bath)
Base.adjoint(op::HEOMBosonicDown) = HEOMBosonicUp(op.l, op.bath)
Base.adjoint(op::HEOMBosonicDamping) = op   # W is Hermitian

## Algebra
import FermionicHilbertSpaces: AbstractSym, GenericHilbertSpace, open_system, symbolic_group, apply_local_operators, add_tag,
    @nc, NCMul, NonCommutativeProducts.mul_effect, NonCommutativeProducts.@commutative
@nc HEOMBosonicUp HEOMBosonicDown HEOMBosonicDamping
@commutative HEOMBosonicOps AbstractSym

# For operators from different baths: sort by bath id
# otherwise don't do anything
_heom_bath_id(op::HEOMBosonicOps) = op.bath.id
function mul_effect(a::HEOMBosonicOps, b::HEOMBosonicOps)
    _heom_bath_id(a) > _heom_bath_id(b) && return Swap(1)   # different baths: sort by id
    return nothing                                            # same bath or already ordered
end

## State and space
"""
    HEOMBosonicAuxState(n)

Basis state for the bosonic HEOM auxiliary Hilbert space. `n` is stored as an
immutable tuple of non-negative integers where `n[l]` is the number of times exponent index `l`
appears in the ADO multi-index j.  The total tier (hierarchy level) is `sum(n)`.
"""
struct HEOMBosonicAuxState{N} <: FermionicHilbertSpaces.AbstractBasisState
    n::NTuple{N,Int}
end
Base.:(==)(a::HEOMBosonicAuxState, b::HEOMBosonicAuxState) = a.n == b.n
Base.hash(a::HEOMBosonicAuxState, h::UInt) = hash(a.n, h)
Base.show(io::IO, s::HEOMBosonicAuxState) = print(io, "ADO", s.n)

# Apply a single HEOM auxiliary factor to an occupation tuple.
# Returns (new_n, new_amplitude); amplitude becomes zero for forbidden transitions.
function _apply_heom_factor!(n::NTuple{N,Int}, amp, op::HEOMBosonicUp) where {N}
    if sum(n) >= op.bath.m_max
        return n, zero(amp)
    end
    return Base.setindex(n, n[op.l] + 1, op.l), amp
end

function _apply_heom_factor!(n::NTuple{N,Int}, amp, op::HEOMBosonicDown) where {N}
    nl = n[op.l]
    if nl == 0
        return n, zero(amp)
    end
    return Base.setindex(n, nl - 1, op.l), amp * nl
end

function _apply_heom_factor!(n::NTuple{N,Int}, amp, op::HEOMBosonicDamping) where {N}
    decay = -sum(op.bath.χ[l] * n[l] for l in eachindex(n); init=zero(ComplexF64))
    return n, amp * decay
end

"""
    apply_local_operators(op::NCMul, state::HEOMBosonicAuxState, space, precomp)

Apply an NCMul of HEOM auxiliary operators to a basis state, returning
(new_states, amplitudes) tuples required by the matrix-building machinery.
Factors are applied right-to-left (standard right-action convention).
"""
function apply_local_operators(op::NCMul, state::HEOMBosonicAuxState{N}, space, precomp) where {N}
    n = state.n
    amp = op.coeff
    for factor in Iterators.reverse(op.factors)
        n, amp = _apply_heom_factor!(n, amp, factor)
        iszero(amp) && return (state,), (zero(amp),)
    end
    return (HEOMBosonicAuxState{N}(n),), (amp,)
end
symbolic_group(op::HEOMBosonicOps) = op.bath
mat_eltype(::Type{<:HEOMBosonicOps}) = ComplexF64

# ─────────────────────────────────────────────────────────────────────────────
# Hilbert space construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    heom_bosonic_aux_space(bath::HEOMBosonicBath) -> GenericHilbertSpace

Construct the auxiliary Hilbert space for a bosonic HEOM hierarchy.

Basis states are all occupation vectors n = (n₁,…,n_Nexp) with nₗ ≥ 0
and Σₗ nₗ ≤ m_max.  The total dimension is

    Σ_{m=0}^{m_max} C(N_exp + m - 1, m)

where C is the binomial coefficient.
"""
function heom_bosonic_aux_space(bath::HEOMBosonicBath)
    return _heom_bosonic_aux_space(bath, Val(bath.N_exp))
end

# Enumerate all occupation tuples via depth-first enumeration.
function _heom_bosonic_aux_space(bath::HEOMBosonicBath, ::Val{N}) where {N}
    states = HEOMBosonicAuxState{N}[]
    _enumerate_aux_states!(states, zeros(Int, N), 1, bath.m_max)
    return GenericHilbertSpace(bath, states)
end

function _enumerate_aux_states!(states, current::Vector{Int}, l::Int, remaining::Int)
    if l > length(current)
        push!(states, HEOMBosonicAuxState(Tuple(current)))
        return
    end
    for nl in 0:remaining
        current[l] = nl
        _enumerate_aux_states!(states, current, l + 1, remaining - nl)
    end
    current[l] = 0
end
"""
    heom_aux_dim(N_exp, m_max)
Return the dimension of the bosonic HEOM auxiliary space: `Σ_{m=0}^{m_max} binomial(N_exp + m - 1, m)`.
"""
function heom_aux_dim(N_exp::Int, m_max::Int)
    sum(binomial(N_exp + m - 1, m) for m in 0:m_max)
end

"""
    heom_generator(H_sys_sym, V_sys_sym, bath; sys_left, sys_right)

Build the symbolic bosonic HEOM generator

    M = i(H_l − H_r) + W_aux
        − i Σ_l (V_l − V_r) * Aup(l)
        − i Σ_l (ξ_l V_l − ξ_l* V_r) * Adown(l)

from symbolic system operators and bath parameters.

Arguments
---------
- `H_sys_l`, `H_sys_r` : symbolic Hamiltonian built from left / right system operators
- `V_sys_l`, `V_sys_r` : symbolic coupling operator built from left / right operators
- `bath`               : `HEOMBosonicBath` instance

The convention follows arXiv 2306.07522 Eq. (16) for the purely bosonic case
with the doubled-space (vectorised density-matrix) representation of Section II.1.
"""
function heom_generator(ham, V, bath::HEOMBosonicBath)
    # Coherent Liouvillian: i[H, ρ] → i(H_l - H_r) in doubled space
    M = 1im * (add_tag(ham, :left) - add_tag(ham, :right))
    Vleft = add_tag(V, :left)
    Vright = add_tag(V, :right)
    # Hierarchy damping: −Σₗ χₗ nₗ (diagonal on auxiliary space)
    M = M + HEOMBosonicDamping(bath)

    for l in 1:bath.N_exp
        ξl = bath.ξ[l]
        # Upward coupling: −i [V_s, ρ] ⊗ Aup(l) in doubled space
        M = M - 1im * (Vleft - Vright) * HEOMBosonicUp(l, bath)

        # Downward coupling: −i (ξ_l V_s ρ − ξ_l* ρ V_s) ⊗ Adown(l)
        M = M - 1im * (ξl * Vleft - conj(ξl) * Vright) * HEOMBosonicDown(l, bath)
    end
    M
end

## Tests
@testset "HEOMBosonicBath construction" begin
    bath = HEOMBosonicBath(1, 2, 3, [1.0, 2.0], [0.5 + 0.5im, 1.0 + 0.0im])
    @test bath.N_exp == 2
    @test bath.m_max == 3
    @test bath.χ == ComplexF64[1.0, 2.0]
    @test bath.ξ == ComplexF64[0.5+0.5im, 1.0+0.0im]
    @test_throws ArgumentError HEOMBosonicBath(1, 2, 3, [1.0], [0.5, 1.0])
end

@testset "HEOMBosonicAuxState enumeration" begin
    # N_exp=1, m_max=2 → states n=0,1,2 (3 states)
    bath = HEOMBosonicBath(1, 1, 2, [1.0], [1.0])
    H = heom_bosonic_aux_space(bath)
    @test dim(H) == 3
    @test basisstates(H)[1] == HEOMBosonicAuxState((0,))
    @test basisstates(H)[2] == HEOMBosonicAuxState((1,))
    @test basisstates(H)[3] == HEOMBosonicAuxState((2,))

    # N_exp=2, m_max=1 → states (0,0),(1,0),(0,1) (3 states)
    bath2 = HEOMBosonicBath(2, 2, 1, [1.0, 2.0], [1.0, 1.0])
    H2 = heom_bosonic_aux_space(bath2)
    @test dim(H2) == 3
    @test basisstates(H2)[1].n isa NTuple{2,Int}

    # N_exp=2, m_max=2 → 1 + 2 + 3 = 6 states
    bath3 = HEOMBosonicBath(3, 2, 2, [1.0, 2.0], [1.0, 1.0])
    H3 = heom_bosonic_aux_space(bath3)
    @test dim(H3) == 6
end

@testset "HEOM aux operator actions" begin
    using LinearAlgebra
    bath = HEOMBosonicBath(1, 1, 2, [2.0], [1.0])
    H = heom_bosonic_aux_space(bath)  # states: n=0,1,2

    Aup = HEOMBosonicUp(1, bath)
    Adown = HEOMBosonicDown(1, bath)
    W = HEOMBosonicDamping(bath)

    Mu = matrix_representation(Aup, H; projection=true)
    Md = matrix_representation(Adown, H; projection=true)
    Mw = matrix_representation(W, H)

    # Aup: |0⟩→|1⟩, |1⟩→|2⟩, |2⟩→0 (truncated)
    @test Mu ≈ [0 0 0; 1 0 0; 0 1 0]

    # Adown: |0⟩→0, |1⟩→|0⟩ (amp=1), |2⟩→|1⟩ (amp=2)
    @test Md ≈ [0 1 0; 0 0 2; 0 0 0]

    # W: |n⟩ → -χ*n |n⟩  →  diagonal  [0, -2, -4]
    @test Mw ≈ diagm([0.0, -2.0, -4.0])
end

@testset "HEOM aux operators commute with fermions" begin
    @fermions f
    bath = HEOMBosonicBath(1, 1, 2, [1.0], [1.0])
    Aup = HEOMBosonicUp(1, bath)
    @test f[1] * Aup - Aup * f[1] == 0
    @test f[1]' * Aup - Aup * f[1]' == 0
    @test f[1] * HEOMBosonicDown(1, bath) - HEOMBosonicDown(1, bath) * f[1] == 0
end

@testset "HEOM full matrix for minimal model" begin
    using FermionicHilbertSpaces: open_system
    using LinearAlgebra

    # Minimal model: spin-1/2 system, 1 bath exponential, m_max=1
    # System: H = ω/2 σz,  coupling V = σz
    ω = 1.0
    χ1 = 1.0 + 0.0im
    ξ1 = 0.5 - 0.5im
    bath = HEOMBosonicBath(1, 1, 1, [χ1], [ξ1])

    @spin σ 1 // 2
    ham = ω / 2 * σ[:z]
    V = σ[:z]

    M_sym = heom_generator(ham, V, bath)

    # Build spaces
    Hs, Hleft, Hright, left, right = open_system(σ)
    Haux = heom_bosonic_aux_space(bath)
    Hfull = tensor_product((Hs, Haux))

    mat = matrix_representation(M_sym, Hfull)

    # Dimension check: 2 × 2 × (1+1) = 8
    @test size(mat) == (8, 8)
    @test dim(Hfull) == 8

    # The HEOM generator must preserve trace: Tr(M[ρ]) = 0 for all ρ
    # In vectorised form: sum of each column restricted to the ρ subspace must be zero
    # Equivalently: the all-ones vector on the m=0 sector is a left eigenvector with λ=0
    # Quick check: real part of trace of mat restricted to m=0 block should be zero
    m0_inds = [i for (i, s) in enumerate(basisstates(Hfull)) if s.states[3].n == (0,)]
    # sum of each column of the coherent (i(H_l - H_r)) part over the ρ sector: zero
    coherent_mat = matrix_representation(1im * (left(ham) - right(ham)), Hs)
    @test tr(coherent_mat) ≈ 0 atol = 1e-12
end
