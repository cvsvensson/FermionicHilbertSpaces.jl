module RimuExt

using Rimu: Rimu, AbstractHamiltonian
import Rimu: default_starting_vector
using Rimu.Interfaces: AbstractOperatorColumn
import Rimu.Interfaces: starting_address, operator_column, parent_operator, diagonal_element, offdiagonals,
    num_offdiagonals, random_offdiagonal
import Rimu.Hamiltonians: dimension
import FermionicHilbertSpaces: rimu_hamiltonian
using FermionicHilbertSpaces: AbstractBasisState, dim, isbasisstate, _states_of, _column_terms, _apply_term,
    _apply_term_groups!, _diagonal_and_count

# Mutable so that Rimu's `==` checks compare identity: on Julia 1.13, `===` can return false
# for a large immutable compared with itself.
mutable struct FHSHamiltonian{T, A, O, S, G} <: AbstractHamiltonian{T}
    const op::O
    const space::S
    const start::A
    const termgroups::G
    const identity::T
    const nterms::Int
end

function rimu_hamiltonian(op, H, start)
    state = only(_states_of(H, (start,)))
    termgroups, identity = _column_terms(op, H)
    # Rimu's solvers and walker populations need floating point matrix elements.
    T = float(typeof(identity))
    return FHSHamiltonian{T, typeof(state), typeof(op), typeof(H), typeof(termgroups)}(
        op, H, state, termgroups, identity, sum(length, termgroups; init = 0)
    )
end

Base.show(io::IO, h::FHSHamiltonian) = print(io, "rimu_hamiltonian(", h.op, ", ", h.space, ", ", h.start, ")")
starting_address(h::FHSHamiltonian) = h.start
dimension(h::FHSHamiltonian, _) = dim(h.space)
default_starting_vector(state::AbstractBasisState; population = 10, kwargs...) =
    default_starting_vector(state => population; kwargs...)

# The off-diagonals of a column are the terms of the operator that map the address to another basis state of the space,
# one per term: terms with the same new state are not summed, so that each is drawn with probability
# 1 / `num_offdiagonals`.
struct FHSColumn{A, T, O} <: AbstractOperatorColumn{A, T, O}
    operator::O
    address::A
    diagonal::T
    noffdiagonals::Int
end

function operator_column(h::FHSHamiltonian{T, A}, address) where {T, A}
    diagonal, count = _diagonal_and_count(h.termgroups, address, h.space, convert(T, h.identity))
    return FHSColumn{A, T, typeof(h)}(h, address, diagonal, count)
end

parent_operator(c::FHSColumn) = c.operator
starting_address(c::FHSColumn) = c.address
diagonal_element(c::FHSColumn) = c.diagonal
num_offdiagonals(c::FHSColumn) = c.noffdiagonals
function offdiagonals(c::FHSColumn{A, T}) where {A, T}
    h = c.operator
    elements = Pair{A, T}[]
    _apply_term_groups!(elements, c.diagonal, h.termgroups, c.address, h.space)
    return filter!(((s, _),) -> isbasisstate(s, h.space), elements)
end
# Draws uniform terms until one maps the address to another basis state, which picks each of these terms with
# probability 1 / `num_offdiagonals`.
function random_offdiagonal(c::FHSColumn{A, T}) where {A, T}
    iszero(c.noffdiagonals) && return c.address, 1.0, zero(T)
    h = c.operator
    while true
        newstate, amp = _apply_term(h.termgroups, rand(1:h.nterms), c.address, h.space)
        (iszero(amp) || newstate == c.address || !isbasisstate(newstate, h.space)) && continue
        return newstate, 1 / c.noffdiagonals, convert(T, amp)
    end
end

end
