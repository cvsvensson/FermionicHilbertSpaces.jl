struct Kets{B,H}
    space::H
end
Kets(space::H) where H = Kets{statetype(space),H}(space)

struct SymbolicState{K,B,H} <: AbstractSym
    space::H
    ket::K
    bra::B
end
SymbolicState(state, space) = SymbolicState(space, state, nothing)
has_ket(s::SymbolicState) = !isnothing(s.ket)
has_bra(s::SymbolicState) = !isnothing(s.bra)
isket(s::SymbolicState) = has_ket(s) && !has_bra(s)
isbra(s::SymbolicState) = has_bra(s) && !has_ket(s)
isketbra(s::SymbolicState) = has_ket(s) && has_bra(s)

function Base.show(io::IO, s::SymbolicState)
    if isket(s)
        print(io, "Ket(")
        show(IOContext(io, :compact => true), s.ket)
    elseif isbra(s)
        print(io, "Bra(")
        show(IOContext(io, :compact => true), s.bra)
    else
        print(io, "KetBra(")
        show(IOContext(io, :compact => true), s.ket)
        print(io, ", ")
        show(IOContext(io, :compact => true), s.bra)
    end
    print(io, ")")
end

Base.:(==)(a::SymbolicState, b::SymbolicState) = a.ket == b.ket && a.bra == b.bra && atomic_id(a.space) == atomic_id(b.space)
Base.hash(a::SymbolicState, h::UInt) = hash(a.bra, hash(a.ket, hash(atomic_id(a.space), h)))
Base.isless(a::SymbolicState, b::SymbolicState) = hash(a) < hash(b)
Base.adjoint(s::SymbolicState) = SymbolicState(s.space, s.bra, s.ket)

symbolic_group(s::SymbolicState) = group_id(s.space)
group_id(s::SymbolicState) = group_id(s.space)
atomic_id(s::SymbolicState) = (atomic_id(s.space), s.ket, s.bra)

@nc SymbolicState AbstractSym

interpret_state(state::B, ::Type{B}) where B = state

function interpret_state(input::AbstractString, ::Type{B}) where {I,B<:FockNumber{I}}
    all(c -> c == '0' || c == '1', input) || throw(ArgumentError("Fock strings must contain only '0' and '1', got \"$input\""))
    bits = map(c -> c == '1', collect(input))
    return B(focknbr_from_bits(bits, I))
end

function interpret_state(input::AbstractString, ::Type{BosonicState})
    n = try
        parse(Int, input)
    catch
        throw(ArgumentError("Invalid bosonic occupation string \"$input\""))
    end
    BosonicState(n)
end

function interpret_state(input::Tuple, ::Type{ProductState{T}}) where {T<:Tuple}
    types = fieldtypes(T)
    length(input) == length(types) || throw(ArgumentError("Product-state tuple has length $(length(input)), expected $(length(types))"))
    parsed = ntuple(i -> interpret_state(input[i], types[i]), length(types))
    return ProductState(parsed)
end

function (k::Kets{B})(inputs...) where B
    raw = if length(inputs) == 1
        only(inputs)
    elseif B <: ProductState
        inputs
    else
        throw(ArgumentError("Expected a single state input for non-product spaces, got $(length(inputs))"))
    end
    parsed = interpret_state(raw, B)
    return SymbolicState(k.space, parsed, nothing)
end

function _same_space_id(a, b)
    atomic_id(a) == atomic_id(b)
end

function _apply_symbolic_operator_to_state(op::AbstractSym, state::AbstractBasisState, space::AbstractHilbertSpace; transpose=false)
    term = NCMul(1, [op])
    precomp = _precomputation_before_operator_application(term, space)
    if transpose
        return apply_local_operators(term, state, space, precomp; transpose=true)
    end
    return _apply_local_operators(term, state, space, precomp)
end

function _term_matrix_representation(op::NCMul{<:Any,<:SymbolicState}, H::AbstractHilbertSpace, t::EagerSparseRepr; kwargs...)
    matrix_representation(only(op.factors), H, t; kwargs...)
end

_order_hash(x) = hash(symbolic_group(x))

function NonCommutativeProducts.mul_effect(a::SymbolicState, b::SymbolicState)
    if !_same_space_id(a.space, b.space)
        return _order_hash(a) > _order_hash(b) ? Swap(1) : nothing
    end
    if has_bra(a) && has_ket(b)
        coeff = a.bra == b.ket ? 1 : 0
        iszero(coeff) && return 0
        if isnothing(a.ket) && isnothing(b.bra)
            return coeff
        end
        return coeff * SymbolicState(a.space, a.ket, b.bra)
    end
    if isket(a) && isbra(b)
        return SymbolicState(a.space, a.ket, b.bra)
    end
    if isket(a) && isket(b)
        throw(ArgumentError("Products ket*ket are undefined for basis states in the same space"))
    end
    if isbra(a) && isbra(b)
        throw(ArgumentError("Products bra*bra are undefined for basis states in the same space"))
    end
    return nothing
end

function apply_local_operator(op::SymbolicState, state::AbstractBasisState, space::AbstractHilbertSpace, precomp)
    _same_space_id(space, op.space) || throw(ArgumentError("Symbolic state space does not match the provided space"))
    isketbra(op) || throw(ArgumentError("apply_local_operator is only defined for ketbra SymbolicState operators"))
    return state == op.bra ? (op.ket, 1) : (state, 0)
end
function apply_local_operators(_op::NCMul{C,<:SymbolicState}, state::AbstractBasisState, space::AbstractHilbertSpace, precomp) where C
    op = only(_op.factors)
    _same_space_id(space, op.space) || throw(ArgumentError("Symbolic state space does not match the provided space"))
    isketbra(op) || throw(ArgumentError("apply_local_operator is only defined for ketbra SymbolicState operators"))
    return state == op.bra ? (op.ket, op.coeff) : (state, zero(eltype(op.coeff)))
end
_precomputation_before_operator_application(op::NCMul{C,<:SymbolicState}, space::AbstractHilbertSpace) where C = nothing
_precomputation_before_operator_application(op::NCMul{C,<:SymbolicState}, space::FermionicSpace{<:FockNumber}) where C = nothing
_precomputation_before_operator_application(op::NCMul{C,<:SymbolicState}, space::TransposedSpace) where C = nothing

function NonCommutativeProducts.mul_effect(op::AbstractSym, s::SymbolicState)
    if !(symbolic_group(op) == group_id(s.space))
        return _order_hash(op) > _order_hash(s) ? Swap(1) : nothing
    end
    if !has_ket(s)
        return nothing
    end
    newket, amp = _apply_symbolic_operator_to_state(op, s.ket, s.space)
    return iszero(amp) ? 0 : amp * SymbolicState(s.space, newket, s.bra)
end

function NonCommutativeProducts.mul_effect(s::SymbolicState, op::AbstractSym)
    if !(symbolic_group(op) == group_id(s.space))
        return _order_hash(s) > _order_hash(op) ? Swap(1) : nothing
    end
    if !has_bra(s)
        return nothing
    end
    newbra, amp = _apply_symbolic_operator_to_state(op, s.bra, s.space; transpose=true)
    return iszero(amp) ? 0 : amp * SymbolicState(s.space, s.ket, newbra)
end


representation(s::SymbolicState; kwargs...) = representation(s, s.space; kwargs...)

function _operator_type(op::SymbolicState)
    if isket(op)
        return :kets
    elseif isbra(op)
        return :bras
    else
        return :ketbras
    end
end
function _operator_type(op)
    hasket = false
    hasbra = false
    hasketbra = false
    hasop = false
    function f(nc::SymbolicState)
        if isket(nc)
            hasket = true
        elseif isbra(nc)
            hasbra = true
        else
            hasketbra = true
        end
        return 0
    end
    f(nc) = hasop = true
    NonCommutativeProducts.ncmap(f, op)
    if hasket && !hasbra && !hasketbra && !hasop
        return :kets
    elseif hasbra && !hasket && !hasketbra && !hasop
        return :bras
    elseif hasketbra && !hasket && !hasbra && !hasop
        return :ketbras
    elseif !hasket && !hasbra && hasop
        return :operator
    else
        return :mixed
    end
end

"""
    vector_representation(op::NCMul, space) -> vector or adjoint-vector

For a product of ket SymbolicStates, apply each ket's operator action in sequence
and return a sparse column vector. For bras, return the adjoint row vector.
"""
function vector_representation(state::AbstractBasisState, space::AbstractHilbertSpace, repr::EagerSparseRepr=EagerSparseRepr(); T=Int, kwargs...)
    ind = state_index(state, space)
    v = SparseArrays.spzeros(T, dim(space))
    v[ind] = one(T)
    return v
end
function vector_representation(state::AbstractBasisState, space::AbstractHilbertSpace, repr::EagerDenseRepr; T=Int, kwargs...)
    ind = state_index(state, space)
    v = zeros(T, dim(space))
    v[ind] = one(T)
    return v
end
function vector_representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    vector_representation(1 * s, space, repr; kwargs...)
end

function vector_representation(op::NCMul, space::AbstractHilbertSpace, repr=EagerSparseRepr(); type=_operator_type(op), kwargs...)
    symstates = op.factors
    group_ids = map(state -> group_id(state.space), symstates)
    perm = map(id -> findfirst(==(id) ∘ group_id, factors(space)), group_ids)
    all(!isnothing, perm) || throw(ArgumentError("Spaces of symbolic states in NCMul do not match the factor spaces of the provided ProductSpace"))
    basis_state(symstate::SymbolicState) = type == :kets ? symstate.ket : symstate.bra
    vec = if length(group_ids) == 1
        op.coeff * vector_representation(basis_state(only(symstates)), space, repr)
    else
        prodstate = ProductState(ntuple(n -> basis_state(symstates[perm[n]]), length(perm)))
        op.coeff * vector_representation(prodstate, space, repr)
    end
    if type == :kets
        return vec
    elseif type == :bras
        return transpose(vec)
    else
        throw(ArgumentError("vector_representation called on an NCMul that is not purely kets or bras"))
    end
end
function vector_representation(op::NCAdd, space::AbstractHilbertSpace, repr=EagerSparseRepr(); type=_operator_type(op), kwargs...)
    iszero(op.coeff) || throw(ArgumentError("NCAdd with nonzero coeff is not supported for vector_representation"))
    return sum(coeff * vector_representation(term, space, repr; type, kwargs...) for (term, coeff) in pairs(op.dict))
end

function representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    _same_space_id(space, s.space) || throw(ArgumentError("Symbolic state space does not match the provided space"))
    if isket(s) || isbra(s)
        return vector_representation(s, space, repr; kwargs...)
    end
    matrix_representation(s, space, repr; kwargs...)
end
function matrix_representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    _same_space_id(space, s.space) || throw(ArgumentError("Symbolic state space does not match the provided space"))
    isketbra(s) || throw(ArgumentError("matrix_representation is only defined for ketbra SymbolicState operators"))
    ket_vec = vector_representation(SymbolicState(space, s.ket, nothing), space, repr)
    bra_vec = vector_representation(SymbolicState(space, nothing, s.bra), space, repr)
    return ket_vec * bra_vec
end


function representation(op, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    type = _operator_type(op)
    if type == :kets || type == :bras
        return vector_representation(op, space, repr; type, kwargs...)
    elseif type == :ketbras
        return matrix_representation(op, space, repr; kwargs...)
    else
        return matrix_representation(op, space, repr; kwargs...)
    end
end


@testitem "Unified representation API" begin
    using SparseArrays, LinearAlgebra
    @fermions f
    @boson b

    Hf = hilbert_space(f, 1:2)
    Hb = hilbert_space(b, 3)
    Hprod = tensor_product(Hf, Hb)

    op = f[1]' * f[2] + hc + 2
    @test representation(op, Hf) == matrix_representation(op, Hf)
    @test representation(op, Hf, :lazy) == matrix_representation(op, Hf, :lazy)
    @test representation(op, Hf, :dense) == representation(op, Hf, :sparse)

    sf = basisstate(3, Hf)
    vf = representation(sf, Hf)
    @test vf[3] == 1
    @test count(!iszero, vf) == 1

    sp = basisstate(4, Hprod)
    vp = representation(sp, Hprod)
    @test vp[4] == 1
    @test count(!iszero, vp) == 1

    Hcons = constrain_space(Hf, NumberConservation(1))
    scons = basisstate(1, Hcons)
    vcons = representation(scons, Hcons)
    @test vcons[1] == 1
    @test count(!iszero, vcons) == 1

    Hsec = hilbert_space(f, 1:3, NumberConservation())
    Hn1 = sector(1, Hsec)
    ssec = basisstate(1, Hn1)
    vsec = representation(ssec, Hn1)
    @test vsec[1] == 1
    @test count(!iszero, vsec) == 1
end

@testitem "Symbolic ket and bra algebra" begin
    using LinearAlgebra

    @fermions f
    Hf = hilbert_space(f, 1:2)
    vf = Kets(Hf)

    @test vf("10")' * vf("10") == 1
    @test vf("10")' * vf("11") == 0

    Mouter = representation(vf("10") * vf("10")', Hf)
    @test Mouter[state_index(vf("10").ket, Hf), state_index(vf("10").ket, Hf)] == 1
    @test count(!iszero, Mouter) == 1

    ketbra = vf("10") * vf("01")'
    Mkb = representation(ketbra, Hf)
    @test Mkb[state_index(vf("10").ket, Hf), state_index(vf("01").ket, Hf)] == 1

    @test representation(vf("10"), Hf)[state_index(vf("10").ket, Hf)] == 1
    @test representation(vf("10")', Hf)[state_index(vf("10").ket, Hf)] == 1

    @test f[2]' * vf("10") == -vf("11")
    @test vf("10")' * f[2] == -vf("11")'
    @test representation(f[2]' * vf("10"), Hf) ≈ matrix_representation(f[2]', Hf) * representation(vf("10"), Hf)
    @test representation(vf("10")' * f[2], Hf) ≈ representation(vf("10")', Hf) * matrix_representation(f[2], Hf)

    @test vf("11")' * f[2]' * vf("10") == -1

    @boson b
    Hb = hilbert_space(b, 5)
    vb = Kets(Hb)
    @test vb("3")' * vb("3") == 1
    @test vb("3")' * vb("4") == 0

    H = tensor_product(Hf, Hb)
    sb = vb("1")
    sf = vf("0")
    rep = representation(sb * sf, H)
    @test rep ≈ tensor_product((representation(sf), representation(sb)), (Hf, Hb) => H)
    @test size(rep) == (dim(H),)

    sb = 0.5 * vb("1")' + 1im * vb("4")' + 2 * vb("2")'
    sf = vf("0")' + 10 * vf("1")'
    rep = representation(sb * sf, H)
    @test rep ≈ tensor_product((representation(sf, Hf), representation(sb, Hb)), (Hf, Hb) => H)
    @test size(rep) == (1, dim(H))
    @test iszero(sb * sf - sf * sb)

    Hcons = tensor_product(Hf, Hb; constraint=ParityConservation())
    sb = vb("1")
    sf = vf("0")
    rep = representation(sb * sf, Hcons)
    @test rep ≈ tensor_product((representation(sf), representation(sb)), (Hf, Hb) => Hcons)
    @test size(rep) == (dim(Hcons),)

    sb = 0.5 * vb("1")' + 1im * vb("4")' + 2 * vb("2")'
    sf = vf("0")' + 10 * vf("1")'
    rep = representation(sb * sf, Hcons)
    @test rep ≈ tensor_product((representation(sf, Hf), representation(sb, Hb)), (Hf, Hb) => Hcons)
    @test size(rep) == (1, dim(Hcons))
    @test iszero(sb * sf - sf * sb)

    using FermionicHilbertSpaces: TransposedSpace
    op = vb("1")vb("0")' + vb("1")vb("3")'
    @test representation(op, TransposedSpace(Hb)) == transpose(representation(op, Hb)) == representation(op', Hb)
    op2 = vb("0")vb("2")' + vb("3")vb("0")'
    @test representation(op * op2, Hb) == representation(op, Hb) * representation(op2, Hb)
    @test representation(op * op2, TransposedSpace(Hb)) ==
          representation(op2, TransposedSpace(Hb)) * representation(op, TransposedSpace(Hb)) ==
          transpose(representation(op * op2, Hb)) == transpose(representation(op2, Hb)) * transpose(representation(op, Hb))


    H = tensor_product(Hf, TransposedSpace(Hb))
    opb = vb("1")vb("0")' + vb("1")vb("4")'
    opf = vf("0")vf("1")' + 2 * vf("1")vf("0")'
    op = opb * opf
    @test representation(op, H) == representation(opb, H) * representation(opf, H) == representation(opb, H) * representation(opf, H)

    @test_nowarn representation(op, constrain_space(H, ParityConservation(1))) # Just to check that no error is thrown
    Hcons = constrain_space(H, NumberConservation())
    representation(opb, Hcons) * representation(opf, Hcons) == representation(opb, Hcons) * representation(opf, Hcons)
    @test b' * op == (sqrt(2) * vb("2")vb("0")' + sqrt(2) * vb("2")vb("4")') * opf
end
