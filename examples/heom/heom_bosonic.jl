# # Bosonic HEOM as a custom operator algebra
# The hierarchical equations of motion (HEOM) evolve the system density matrix together with a
# hierarchy of auxiliary density operators (ADOs). Here the ADO index is a custom Hilbert space
# with its own operators, so the HEOM generator is a symbolic sum that `representation` turns into
# a matrix on the doubled system space times the ADO space. The generator is Eq. (16) of
# [arXiv:2306.07522](https://arxiv.org/abs/2306.07522); `comparison.jl` checks it against HierarchicalEOM.jl.

using FermionicHilbertSpaces
import FermionicHilbertSpaces: AbstractBasisState, Field, Op, apply_local_operator, add_tag

# The bath correlation function is ``C(τ) = \sum_l ξ_l e^{-χ_l τ}``, and the hierarchy is
# truncated at total tier `m_max`.
struct HEOMBosonicBath{V1, V2}
    id::Symbol
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
# The ADO index is the single mode of a field named after the bath.
heom_mode(bath) = Field{HEOMKind}(bath.id)[]
function heom_bosonic_aux_space(bath)
    m = bath.m_max
    ns = (reverse(n) for n in Iterators.product(ntuple(_ -> 0:m, length(bath.χ))...) if sum(n) <= m)
    return hilbert_space(heom_mode(bath), [ADO(n) for n in ns])
end

# The kind `HEOMKind(:up, l, bath)` raises ``n_l`` with amplitude ``n_l+1`` and vanishes at the truncation tier,
# `:down` lowers ``n_l``, and `:damping` multiplies by ``-\sum_l χ_l n_l``.
struct HEOMKind{B}
    type::Symbol
    l::Int
    bath::B
end
HEOMOp(type, l, bath) = Op(HEOMKind(type, l, bath), heom_mode(bath))

function apply_local_operator(k::HEOMKind, s::ADO, space, precomp)
    if k.type === :up
        sum(s.n) < k.bath.m_max || return s, 0.0
        return ADO(Base.setindex(s.n, s.n[k.l] + 1, k.l)), float(s.n[k.l] + 1)
    elseif k.type === :down
        s.n[k.l] > 0 || return s, 0.0
        return ADO(Base.setindex(s.n, s.n[k.l] - 1, k.l)), 1.0
    else
        return s, -sum(k.bath.χ .* s.n)
    end
end

# The generator of ``d\rho/dt`` for the system Hamiltonian `H` and the coupling `V` to the bath.
function heom_generator(H, V, bath)
    Vl, Vr = add_tag(V, :left), add_tag(V, :right)
    M = -1im * (add_tag(H, :left) - add_tag(H, :right)) + HEOMOp(:damping, 0, bath)
    for (l, ξ) in enumerate(bath.ξ)
        M += -1im * (Vl - Vr) * HEOMOp(:down, l, bath) - 1im * (ξ * Vl - conj(ξ) * Vr) * HEOMOp(:up, l, bath)
    end
    return M
end
