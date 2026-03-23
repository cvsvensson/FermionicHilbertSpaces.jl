using FermionicHilbertSpaces
import FermionicHilbertSpaces.NonCommutativeProducts: @nc, Swap, NCAdd, NCMul, NCterms, AddTerms, @commutative, mul_effect
import FermionicHilbertSpaces: apply_local_operators, symbolic_group
##
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
FloquetLadder(b::FloquetBasis) = FloquetLadder(1, b)
Base.adjoint(F::FloquetLadder) = FloquetLadder(-F.shift, F.basis)
Base.adjoint(N::FloquetNumber) = N
@nc FloquetLadder FloquetNumber
@commutative FloquetLadder FermionicHilbertSpaces.SpinSym
@commutative FloquetNumber FermionicHilbertSpaces.SpinSym
function mul_effect(a::FloquetLadder, b::FloquetLadder)
    if a.basis.id > b.basis.id
        return Swap(1)
    elseif a.basis.id < b.basis.id
        return nothing
    end
    total_shift = a.shift + b.shift
    if total_shift == 0
        return 1
    else
        return FloquetLadder(total_shift, a.basis)
    end
end
function mul_effect(a::FloquetNumber, b::FloquetNumber)
    if a.basis.id > b.basis.id
        return Swap(1)
    elseif a.basis.id < b.basis.id
        return nothing
    end
    if a.power == b.power == 0
        return 1
    end
    return FloquetNumber(a.power + b.power, a.basis)
end
function mul_effect(a::FloquetLadder, b::FloquetNumber)
    if a.basis.id > b.basis.id
        return Swap(1)
    elseif a.basis.id < b.basis.id
        return nothing
    end
    return AddTerms((Swap(1), NCMul(a.shift, [b, a])))
end
function mul_effect(::FloquetNumber, ::FloquetLadder)
    if a.basis.id > b.basis.id
        return Swap(1)
    end
    return nothing
end
function apply_local_operators(op, state::FloquetState, space, precomp)
    state, amp = foldr((op, (state, amp)) -> apply_local_operator(op, state, amp), op.factors, init=(state, 1))
    return ((state, amp),)
end
apply_local_operator(op::FloquetLadder, state::FloquetState, amp) = (FloquetState(state.mode + op.shift), amp)
function apply_local_operator(op::FloquetNumber, state::FloquetState, amp)
    (state, amp * state.mode^op.power)
end
symbolic_group(f::FloquetLadder) = f.basis
symbolic_group(f::FloquetNumber) = f.basis
symbolic_group(f::FloquetBasis) = f

##
@spin σ
floquet_basis = FloquetBasis(0)
F = FloquetLadder(floquet_basis)
N = FloquetNumber(1, floquet_basis)
##
function Heff(ħω, V)
    h0 = σ[:z]
    hp1 = V * σ[:x] * F
    hm1 = V * σ[:x] * F'
    h0 + ħω * N + 1 / ħω * (hp1 * hm1 - hm1 * hp1) + 1 / ħω * (h0 * hp1 - hp1 * h0) - 1 / ħω * (h0 * hm1 - hm1 * h0)
end
##
Hfloc = FermionicHilbertSpaces.GenericHilbertSpace(floquet_basis, FloquetState.(-1:1))
Hspin = FermionicHilbertSpaces.SpinSpace{1 // 2}(σ)
H = tensor_product((Hspin, Hfloc))
ham = Heff(10, 1)
mat = matrix_representation(Heff(10, 1), H; projection=true)