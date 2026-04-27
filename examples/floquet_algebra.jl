import FermionicHilbertSpaces.NonCommutativeProducts: @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, @commutative, mul_effect
import FermionicHilbertSpaces: apply_local_operator, symbolic_group, FermionSym, SpinSym
## Floquet algebra, see https://arxiv.org/pdf/2503.20186v1
struct FloquetBasis
    id::Int
end
struct FloquetLadder
    shift::Int
    basis::FloquetBasis
end
struct FloquetNumber
    power::Int
    basis::FloquetBasis
end
struct FloquetState
    mode::Int
end
const Floquets = Union{FloquetLadder,FloquetNumber}
FloquetLadder(b::FloquetBasis) = FloquetLadder(1, b)
Base.adjoint(F::FloquetLadder) = FloquetLadder(-F.shift, F.basis)
Base.adjoint(N::FloquetNumber) = N
@nc FloquetLadder FloquetNumber
@commutative Floquets FermionSym
@commutative Floquets SpinSym
function mul_effect(a::Floquets, b::Floquets)
    a.basis.id > b.basis.id && return Swap(1)
    a.basis.id < b.basis.id && return nothing
    return floquet_mul(a, b)
end
function floquet_mul(a::FloquetLadder, b::FloquetLadder)
    total_shift = a.shift + b.shift
    total_shift == 0 && return 1
    return FloquetLadder(total_shift, a.basis)
end
function floquet_mul(a::FloquetNumber, b::FloquetNumber)
    a.power == b.power == 0 && return 1
    return FloquetNumber(a.power + b.power, a.basis)
end
floquet_mul(a::FloquetLadder, ::FloquetNumber) = return AddTerms((Swap(1), -a.shift * a))
floquet_mul(::FloquetNumber, ::FloquetLadder) = return nothing


apply_local_operator(op::FloquetLadder, state::FloquetState, amp) = (FloquetState(state.mode + op.shift), amp)
apply_local_operator(op::FloquetNumber, state::FloquetState, amp) = (state, amp * state.mode^op.power)

symbolic_group(f::Floquets) = f.basis
symbolic_group(f::FloquetBasis) = f
