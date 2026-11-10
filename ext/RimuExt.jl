module RimuExt

using Rimu: Rimu, AbstractHamiltonian
import Rimu: default_starting_vector
using Rimu.Interfaces: AbstractOperatorColumn
import Rimu.Interfaces: starting_address, operator_column, parent_operator, diagonal_element, offdiagonals,
    num_offdiagonals, random_offdiagonal
import Rimu.Hamiltonians: dimension
import FermionicHilbertSpaces: rimu_hamiltonian
using FermionicHilbertSpaces: AbstractBasisState, dim, matrix_column, _states_of

struct FHSHamiltonian{T,A,O,S,C} <: AbstractHamiltonian{T}
    op::O
    space::S
    start::A
    column::C
end

function rimu_hamiltonian(op, H, start)
    column = matrix_column(op, H)
    state = only(_states_of(H, (start,)))
    A, T = _pair_types(eltype(column(state)))
    # Rimu's solvers and walker populations need floating point matrix elements.
    return FHSHamiltonian{float(T),A,typeof(op),typeof(H),typeof(column)}(op, H, state, column)
end
_pair_types(::Type{Pair{A,T}}) where {A,T} = (A, T)

Base.show(io::IO, h::FHSHamiltonian) = print(io, "rimu_hamiltonian(", h.op, ", ", h.space, ", ", h.start, ")")
starting_address(h::FHSHamiltonian) = h.start
dimension(h::FHSHamiltonian, _) = dim(h.space)
default_starting_vector(state::AbstractBasisState; population = 10, kwargs...) =
    default_starting_vector(state => population; kwargs...)

struct FHSColumn{A,T,O} <: AbstractOperatorColumn{A,T,O}
    operator::O
    address::A
    diagonal::T
    offdiagonals::Vector{Pair{A,T}}
end

function operator_column(h::FHSHamiltonian{T,A}, address) where {T,A}
    elements = h.column(address)
    i = findfirst(p -> p.first == address, elements)
    diagonal = isnothing(i) ? zero(T) : popat!(elements, i).second
    return FHSColumn{A,T,typeof(h)}(h, address, diagonal, elements)
end

parent_operator(c::FHSColumn) = c.operator
starting_address(c::FHSColumn) = c.address
diagonal_element(c::FHSColumn) = c.diagonal
offdiagonals(c::FHSColumn) = c.offdiagonals
num_offdiagonals(c::FHSColumn) = length(c.offdiagonals)
# A column without off-diagonal elements spawns nothing.
function random_offdiagonal(c::FHSColumn{A,T}) where {A,T}
    isempty(c.offdiagonals) && return c.address, 1.0, zero(T)
    newaddress, value = rand(c.offdiagonals)
    return newaddress, 1 / length(c.offdiagonals), value
end

end
