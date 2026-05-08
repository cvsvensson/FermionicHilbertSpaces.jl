struct Kets{B,H}
    space::H
end
Kets(space::H) where H = Kets{statetype(space),H}(space)

struct SymbolicState{B,H} <: AbstractSym
    state::B
    space::H
    ket::Bool
end
SymbolicState(state, space) = SymbolicState(state, space, true)

isket(s::SymbolicState) = s.ket
isbra(s::SymbolicState) = !s.ket

function Base.show(io::IO, s::SymbolicState)
    if s.ket
        print(io, "Ket(")
    else
        print(io, "Bra(")
    end
    show(IOContext(io, :compact => true), s.state)
    print(io, ")")
end

Base.:(==)(a::SymbolicState, b::SymbolicState) = a.ket == b.ket && a.state == b.state && atomic_id(a.space) == atomic_id(b.space)
Base.hash(a::SymbolicState, h::UInt) = hash(a.ket, hash(a.state, hash(atomic_id(a.space), h)))
Base.isless(a::SymbolicState, b::SymbolicState) = hash(a) < hash(b)
Base.adjoint(s::SymbolicState) = SymbolicState(s.state, s.space, !s.ket)

symbolic_group(s::SymbolicState) = group_id(s.space)
group_id(s::SymbolicState) = group_id(s.space)
atomic_id(s::SymbolicState) = (atomic_id(s.space), s.state, s.ket)

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
    return SymbolicState(parsed, k.space, true)
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

_order_hash(x) = hash(group_id(x), hash(typeof(x)))

function NonCommutativeProducts.mul_effect(a::SymbolicState, b::SymbolicState)
    if !_same_space_id(a.space, b.space)
        return _order_hash(a) > _order_hash(b) ? Swap(1) : nothing
    end
    if a.ket == b.ket
        throw(ArgumentError("Products ket*ket and bra*bra are undefined for basis states in the same space"))
    end
    if isbra(a) && isket(b)
        return a.state == b.state ? 1 : 0
    end
    return nothing
end

function NonCommutativeProducts.mul_effect(op::AbstractSym, s::SymbolicState)
    if !_same_space_id(group_id(op), group_id(s.space))
        return _order_hash(op) > _order_hash(s) ? Swap(1) : nothing
    end
    if isbra(s)
        return nothing
    end
    newstate, amp = _apply_symbolic_operator_to_state(op, s.state, s.space)
    return iszero(amp) ? 0 : amp * SymbolicState(newstate, s.space, true)
end

function NonCommutativeProducts.mul_effect(s::SymbolicState, op::AbstractSym)
    if !_same_space_id(group_id(op), group_id(s.space))
        return _order_hash(s) > _order_hash(op) ? Swap(1) : nothing
    end
    if isket(s)
        return nothing
    end
    newstate, amp = _apply_symbolic_operator_to_state(op, s.state, s.space; transpose=true)
    return iszero(amp) ? 0 : amp * SymbolicState(newstate, s.space, false)
end


representation(s::SymbolicState; kwargs...) = representation(s, s.space; kwargs...)

_operator_type(op::SymbolicState) = op.ket ? :kets : :bras
function _operator_type(op)
    hasket = false
    hasbra = false
    hasop = false
    function f(nc::SymbolicState)
        isket(nc) ? (hasket = true) : (hasbra = true)
        return 0
    end
    f(nc) = hasop = true
    NonCommutativeProducts.ncmap(f, op)
    if hasket && !hasbra && !hasop
        return :kets
    elseif hasbra && !hasket && !hasop
        return :bras
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
function vector_representation(state::AbstractBasisState, space::AbstractHilbertSpace)
    ind = state_index(state, space)
    v = SparseArrays.spzeros(Int, dim(space))
    v[ind] = 1
    return v
end
function vector_representation(s::SymbolicState, space::AbstractHilbertSpace; kwargs...)
    _same_space_id(space, s.space) || throw(ArgumentError("Symbolic state space does not match the provided space"))
    vec = vector_representation(s.state, space)
    return isket(s) ? vec : adjoint(vec)
end

function vector_representation(op::NCMul, space::AbstractHilbertSpace; type=_operator_type(op), kwargs...)
    symstates = op.factors
    group_ids = map(state -> group_id(state.space), symstates)
    perm = map(id -> findfirst(==(id) ∘ group_id, factors(space)), group_ids)
    all(!isnothing, perm) || throw(ArgumentError("Spaces of symbolic states in NCMul do not match the factor spaces of the provided ProductSpace"))
    vec = if length(group_ids) == 1
        op.coeff * vector_representation(only(symstates).state, space)
    else
        prodstate = ProductState(ntuple(n -> symstates[perm[n]].state, length(perm)))
        op.coeff * vector_representation(prodstate, space)
    end
    if type == :kets
        return vec
    elseif type == :bras
        return transpose(vec)
    else
        throw(ArgumentError("vector_representation called on an NCMul that is not purely kets or bras"))
    end
end
function vector_representation(op::NCAdd, space::AbstractHilbertSpace; type=_operator_type(op), kwargs...)
    iszero(op.coeff) || throw(ArgumentError("NCAdd with nonzero coeff is not supported for vector_representation"))
    return sum(coeff * vector_representation(term, space; type, kwargs...) for (term, coeff) in pairs(op.dict))
end


function representation(op, space::AbstractHilbertSpace; kwargs...)
    type = _operator_type(op)
    if type == :kets || type == :bras
        return vector_representation(op, space; type)
    else
        return matrix_representation(op, space; kwargs...)
    end
end


@testitem "Unified representation API" begin
    using SparseArrays, LinearAlgebra
    using FermionicHilbertSpaces: basisstate
    @fermions f
    @boson b

    Hf = hilbert_space(f, 1:2)
    Hb = hilbert_space(b, 3)
    Hprod = tensor_product(Hf, Hb)

    op = f[1]' * f[2] + hc + 2
    @test representation(op, Hf) == matrix_representation(op, Hf)
    @test representation(op, Hf; lazy=true) == matrix_representation(op, Hf; lazy=true)

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
    @test Mouter[state_index(vf("10").state, Hf), state_index(vf("10").state, Hf)] == 1
    @test count(!iszero, Mouter) == 1

    @test representation(vf("10"), Hf)[state_index(vf("10").state, Hf)] == 1
    @test representation(vf("10")', Hf)[state_index(vf("10").state, Hf)] == 1

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
    @test representation(vb("1")vb("0")', TransposedSpace(Hb)) == transpose(representation(vb("1")vb("0")', Hb))
end