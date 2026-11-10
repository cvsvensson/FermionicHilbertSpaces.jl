module RimuExt

using Rimu: Rimu, AbstractHamiltonian
import Rimu: default_starting_vector
using Rimu.Interfaces: AbstractOperatorColumn
import Rimu.Interfaces: starting_address, operator_column, parent_operator, diagonal_element, offdiagonals,
    num_offdiagonals, random_offdiagonal
import Rimu.Hamiltonians: dimension
import FermionicHilbertSpaces: rimu_hamiltonian
using FermionicHilbertSpaces: AbstractBasisState, dim, isbasisstate, matrix_column, _states_of, _column_terms, _apply_term,
    _apply_term_groups!

# Mutable so that Rimu's `==` checks compare identity: on Julia 1.13, `===` can return false
# for a large immutable compared with itself.
mutable struct FHSHamiltonian{T, A, O, S, C, G} <: AbstractHamiltonian{T}
    const op::O
    const space::S
    const start::A
    const column::C
    const termgroups::G
    const identity::T
    const nterms::Int
end

function rimu_hamiltonian(op, H, start)
    column = matrix_column(op, H)
    state = only(_states_of(H, (start,)))
    termgroups, identity = _column_terms(op, H)
    # Rimu's solvers and walker populations need floating point matrix elements.
    T = float(typeof(identity))
    nterms = sum(length, termgroups; init = 0)
    return FHSHamiltonian{T, typeof(state), typeof(op), typeof(H), typeof(column), typeof(termgroups)}(
        op, H, state, column,
        termgroups, identity, nterms
    )
end

Base.show(io::IO, h::FHSHamiltonian) = print(io, "rimu_hamiltonian(", h.op, ", ", h.space, ", ", h.start, ")")
starting_address(h::FHSHamiltonian) = h.start
dimension(h::FHSHamiltonian, _) = dim(h.space)
default_starting_vector(state::AbstractBasisState; population = 10, kwargs...) =
    default_starting_vector(state => population; kwargs...)

# The column is lazy: each function applies the terms of the operator to the address when called.
struct FHSColumn{A, T, O} <: AbstractOperatorColumn{A, T, O}
    operator::O
    address::A
end

operator_column(h::FHSHamiltonian{T, A}, address) where {T, A} = FHSColumn{A, T, typeof(h)}(h, address)

parent_operator(c::FHSColumn) = c.operator
starting_address(c::FHSColumn) = c.address
diagonal_element(c::FHSColumn) = _apply_term_groups!(nothing, c.operator.identity, c.operator.termgroups, c.address, c.operator.space)
function offdiagonals(c::FHSColumn)
    elements = c.operator.column(c.address)
    !isempty(elements) && first(first(elements)) == c.address && popfirst!(elements)
    return elements
end
num_offdiagonals(c::FHSColumn) = c.operator.nterms
# Applies one uniformly random term. Terms that map the address to itself, to zero or outside the space spawn nothing,
# which keeps the sum over terms of amplitude / probability unbiased.
function random_offdiagonal(c::FHSColumn{A, T}) where {A, T}
    h = c.operator
    iszero(h.nterms) && return c.address, 1.0, zero(T)
    prob = 1 / h.nterms
    newstate, amp = _apply_term(h.termgroups, rand(1:h.nterms), c.address, h.space)
    (iszero(amp) || newstate == c.address || !isbasisstate(newstate, h.space)) && return c.address, prob, zero(T)
    return newstate, prob, convert(T, amp)
end

end
