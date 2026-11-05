# # Bosonic HEOM as a custom operator algebra
# The hierarchical equations of motion (HEOM) evolve the system density matrix together with a
# hierarchy of auxiliary density operators (ADOs). Here the ADO index is a custom Hilbert space
# with its own operators, so the HEOM generator is a symbolic sum that `representation` turns into
# a matrix on the doubled system space times the ADO space. The generator is Eq. (16) of
# [arXiv:2306.07522](https://arxiv.org/abs/2306.07522); `comparison.jl` checks it against HierarchicalEOM.jl.

using FermionicHilbertSpaces
import FermionicHilbertSpaces: AbstractBasisState, AbstractSym,
    apply_local_operator, mat_eltype, symbolic_group, add_tag
import FermionicHilbertSpaces.NonCommutativeProducts: @nc

# The bath correlation function is ``C(τ) = \sum_l ξ_l e^{-χ_l τ}``, and the hierarchy is
# truncated at total tier `m_max`.
struct HEOMBosonicBath{L,V1,V2}
    id::L
    m_max::Int
    χ::V1
    ξ::V2
end
Base.:(==)(a::HEOMBosonicBath, b::HEOMBosonicBath) = (a.id, a.m_max, a.χ, a.ξ) == (b.id, b.m_max, b.χ, b.ξ)
Base.hash(b::HEOMBosonicBath, h::UInt) = hash((b.id, b.m_max), h)

# An ADO is labelled by occupations ``n``, where ``n_l`` counts how often exponent ``l`` appears in its multi-index.
struct ADO{N} <: AbstractBasisState
    n::NTuple{N,Int}
end
function heom_bosonic_aux_space(bath)
    m = bath.m_max
    ns = (reverse(n) for n in Iterators.product(ntuple(_ -> 0:m, length(bath.χ))...) if sum(n) <= m)
    return GenericHilbertSpace(bath.id, [ADO(n) for n in ns])
end

# `HEOMOp{:up}(l)` raises ``n_l`` with amplitude ``n_l+1`` and vanishes at the truncation tier,
# `HEOMOp{:down}(l)` lowers ``n_l``, and `HEOMOp{:damping}` multiplies by ``-\sum_l χ_l n_l``.
struct HEOMOp{T, B} <: AbstractSym
    l::Int
    bath::B
end
HEOMOp{T}(l, bath) where {T} = HEOMOp{T, typeof(bath)}(l, bath)
Base.:(==)(a::HEOMOp{T}, b::HEOMOp{T}) where {T} = a.l == b.l && a.bath == b.bath
Base.hash(op::HEOMOp{T}, h::UInt) where {T} = hash((T, op.l, op.bath), h)
@nc HEOMOp
symbolic_group(op::HEOMOp) = op.bath.id
mat_eltype(::Type{<:HEOMOp}) = Float64

function apply_local_operator(op::HEOMOp{:up}, s::ADO, space, precomp)
    sum(s.n) < op.bath.m_max || return s, 0.0
    return ADO(Base.setindex(s.n, s.n[op.l] + 1, op.l)), float(s.n[op.l] + 1)
end
function apply_local_operator(op::HEOMOp{:down}, s::ADO, space, precomp)
    s.n[op.l] > 0 || return s, 0.0
    return ADO(Base.setindex(s.n, s.n[op.l] - 1, op.l)), 1.0
end
apply_local_operator(op::HEOMOp{:damping}, s::ADO, space, precomp) = s, -sum(op.bath.χ .* s.n)

# The generator of ``d\rho/dt`` for the system Hamiltonian `H` and the coupling `V` to the bath.
function heom_generator(H, V, bath)
    Vl, Vr = add_tag(V, :left), add_tag(V, :right)
    M = -1im * (add_tag(H, :left) - add_tag(H, :right)) + HEOMOp{:damping}(0, bath)
    for (l, ξ) in enumerate(bath.ξ)
        M += -1im * (Vl - Vr) * HEOMOp{:down}(l, bath) - 1im * (ξ * Vl - conj(ξ) * Vr) * HEOMOp{:up}(l, bath)
    end
    return M
end
