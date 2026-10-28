struct Kets{B,H}
    space::H
    Kets(space::H) where H = new{statetype(space),H}(space)
end
Ket(label, H) = Kets(H)(label)
# The basis kets of the space, in the order of `basisstates`
Base.length(k::Kets) = dim(k.space)
Base.firstindex(k::Kets) = 1
Base.lastindex(k::Kets) = length(k)
Base.keys(k::Kets) = Base.OneTo(length(k))
Base.getindex(k::Kets, i::Integer) = SymbolicState(k.space, basisstate(i, k.space), nothing)
Base.iterate(k::Kets, i=1) = i > length(k) ? nothing : (k[i], i + 1)
Base.eltype(::Type{Kets{B,H}}) where {B,H} = _ket_type(B, H)
_ket_type(B, H) = SymbolicState{B,Nothing,H}
# A ket of a product space is a sum with a single term, the product of kets of its groups, whose type is inferred
_ket_type(::Type{B}, ::Type{H}) where {B<:ProductState,H} = Base.promote_op(getindex, Kets{B,H}, Int)

function Base.show(io::IO, k::Kets)
    print(io, "Kets(")
    show(IOContext(io, :compact => true), k.space)
    print(io, ")")
end

@testitem "Kets show" begin
    @fermions f
    kets = Kets(hilbert_space(f, 1:2))
    @test occursin("Kets", repr(kets))
end

@testitem "Symbolic state show" begin
    @fermions f
    @boson b
    @spin s 1 // 2
    vf = Kets(hilbert_space(f, 1:2))
    @test repr(vf("00")) == "|00⟩"
    @test repr(vf("10") * vf("01")') == "|10⟩⟨01|"
    @test repr(Ket("2", hilbert_space(b, 4))') == "⟨2|"
    @test repr(Ket("↑", hilbert_space(s))) == "|↑⟩"
end

@testitem "Kets indexing" begin
    @fermions f
    @boson b
    Hf = hilbert_space(f, 1:2, NumberConservation())
    H = tensor_product(hilbert_space(f, 1:2), hilbert_space(b, 3))
    for space in (Hf, H)
        kets = Kets(space)
        @test length(kets) == length(collect(kets)) == dim(space)
        for i in 1:dim(space)
            v = representation(kets[i], space)
            @test v[i] == 1 && count(!iszero, v) == 1
        end
    end
    @test Kets(Hf)[2] == Kets(Hf)("10")
    @test Kets(H)[6] == Kets(H)("10", "1")
    @test Kets(H)[end] == Kets(H)("11", "2")
    @test eachindex(Kets(H)) == 1:dim(H)
end

@testitem "Kets eltype" begin
    @fermions f
    @boson b
    @boson a
    @spin s 1 // 2
    Hf, Hb, Hs = hilbert_space(f, 1:2), hilbert_space(b, 3), hilbert_space(s)
    spaces = (Hf, hilbert_space(f, 1:2, NumberConservation()), Hb, Hs,
        tensor_product(Hf, Hb), tensor_product(Hf, Hs, Hb), tensor_product(hilbert_space(a, 1), Hb),
        tensor_product(Hf, hilbert_space(f, 3:4)))
    for space in spaces
        kets = Kets(space)
        k = @inferred kets[1]
        @test isconcretetype(eltype(kets))
        @test eltype(kets) == typeof(k)
        @test eltype(collect(kets)) == eltype(kets)
        @test kets[end] == kets[dim(space)]
    end
end

@testitem "Product space states" begin
    import FermionicHilbertSpaces: SymbolicState, interpret_state
    @fermions f
    @boson b
    @spin s 1 // 2
    # The groups of H are not in sorted order, so the factors of its states are reordered
    H = tensor_product(hilbert_space(b, 3), hilbert_space(s), hilbert_space(f, 1:2))
    kf, ks, kb = Ket("01", hilbert_space(f, 1:2)), Ket("↓", hilbert_space(s)), Ket("2", hilbert_space(b, 3))
    x = interpret_state(("2", "↓", "01"), H)
    @test (@inferred SymbolicState(H, x, nothing)) == kf * ks * kb
    @test (@inferred SymbolicState(H, nothing, x)) == kf' * ks' * kb'
    @test (@inferred SymbolicState(H, x, x)) == (kf * kf') * (ks * ks') * (kb * kb')
end

struct SymbolicState{K,B,H} <: AbstractSym
    space::H
    ket::K
    bra::B
end

# A state of a product space is the product of states of its groups
function SymbolicState(space, ket::ProductState, bra::Nothing)
    _group_product(map(SymbolicState, groups(space), ket.states))
end
function SymbolicState(space, ket::Nothing, bra::ProductState)
    _group_product(map((g, b) -> SymbolicState(g, nothing, b), groups(space), bra.states))
end
function SymbolicState(space, ket::ProductState, bra::ProductState)
    _group_product(map(SymbolicState, groups(space), ket.states, bra.states))
end
# The sorted product of states of distinct groups, which commute. It is built directly rather than with `*`
# so that its type is inferrable, and equals the autosorted product: a sum with the single term `1 * factors`.
function _group_product(states::Tuple)
    factors = sort!([states...]; lt=_group_isless)
    NCAdd(NCMul(1, factors))
end
SymbolicState(space, state) = SymbolicState(space, state, nothing)
has_ket(s::SymbolicState) = !isnothing(s.ket)
has_bra(s::SymbolicState) = !isnothing(s.bra)
isket(s::SymbolicState) = has_ket(s) && !has_bra(s)
isbra(s::SymbolicState) = has_bra(s) && !has_ket(s)
isketbra(s::SymbolicState) = has_ket(s) && has_bra(s)

function Base.show(io::IO, s::SymbolicState)
    has_ket(s) && print(io, "|", _state_label(s.ket, s.space), "⟩")
    has_bra(s) && print(io, "⟨", _state_label(s.bra, s.space), "|")
end

Base.:(==)(a::SymbolicState, b::SymbolicState) = a.ket == b.ket && a.bra == b.bra && _same_space_id(a.space, b.space)
Base.hash(a::SymbolicState, h::UInt) = hash(a.bra, hash(a.ket, hash(group_id(a.space), h)))
# States are ordered by group, then by space, then by ket and bra, where a missing ket or bra comes first.
function Base.isless(a::SymbolicState, b::SymbolicState)
    _group_isless(a, b) && return true
    _group_isless(b, a) && return false
    if a.space !== b.space
        ia, ib = atom_ids(a.space), atom_ids(b.space)
        isless(ia, ib) && return true
        isless(ib, ia) && return false
    end
    _isless_missing_first(a.ket, b.ket) && return true
    _isless_missing_first(b.ket, a.ket) && return false
    _isless_missing_first(a.bra, b.bra)
end
_isless_missing_first(::Nothing, ::Nothing) = false
_isless_missing_first(::Nothing, _) = true
_isless_missing_first(_, ::Nothing) = false
_isless_missing_first(a, b) = isless(a, b)
Base.adjoint(s::SymbolicState) = SymbolicState(s.space, s.bra, s.ket)

symbolic_group(s::SymbolicState) = group_id(s.space)
group_id(s::SymbolicState) = group_id(s.space)

@nc SymbolicState AbstractSym

## Grading
# Each space carries a grade k^m b^n in the bicyclic monoid, where a ket has (m, n) = (1, 0) and a bra (0, 1).
# States in different spaces commute, so the grade of an expression is the product over spaces, stored as a map
# from the group of the space to its bicyclic grade. Spaces with the identity grade (0, 0) are left out,
# and the identity overall is `nothing`. Operators have the identity grade.
# NonCommutativeProducts stores grades in sets, so the dict of a StateGrade must never be mutated after construction.
struct Bicyclic
    m::Int
    n::Int
end
function _bicyclic_mul(g::Bicyclic, h::Bicyclic)
    t = max(g.n, h.m)
    Bicyclic(g.m - g.n + t, h.n - h.m + t)
end
_isidentity(g::Bicyclic) = g == Bicyclic(0, 0)

struct StateGrade
    grades::Dict{Any,Bicyclic}
end
Base.:(==)(a::StateGrade, b::StateGrade) = a.grades == b.grades
Base.isequal(a::StateGrade, b::StateGrade) = isequal(a.grades, b.grades)
Base.hash(a::StateGrade, h::UInt) = hash(a.grades, hash(StateGrade, h))
function Base.show(io::IO, g::StateGrade)
    print(io, "StateGrade(")
    join(io, ("$(k) => ($(v.m), $(v.n))" for (k, v) in g.grades), ", ")
    print(io, ")")
end
_stategrade(d) = isempty(d) ? nothing : StateGrade(d)

function NonCommutativeProducts.grade(s::SymbolicState)
    g = Bicyclic(Int(has_ket(s)), Int(has_bra(s)))
    _isidentity(g) ? nothing : StateGrade(Dict{Any,Bicyclic}(group_id(s.space) => g))
end
# One grading for all spaces, so that it combines with other declared gradings into a ProductGrade
NonCommutativeProducts.grading(::Type{StateGrade}) = StateGrade
function NonCommutativeProducts.mul_grade(a::StateGrade, b::StateGrade)
    d = copy(a.grades)
    for (k, g) in b.grades
        h = haskey(d, k) ? _bicyclic_mul(d[k], g) : g
        _isidentity(h) ? delete!(d, k) : (d[k] = h)
    end
    _stategrade(d)
end
NonCommutativeProducts.adjoint_grade(a::StateGrade) = StateGrade(Dict{Any,Bicyclic}(k => Bicyclic(g.n, g.m) for (k, g) in a.grades))

interpret_state(state::B, ::Type{B}) where B = state
interpret_state(state, space::AbstractHilbertSpace) = interpret_state(state, statetype(space))
interpret_state(state, space::Union{TransposedSpace,AbstractSubspace}) = interpret_state(state, parent(space))
# A string or character spells a state of a space whose states are `AbstractBasisState`s; on other spaces it is looked up as it is
state_index(state::Union{<:AbstractString,<:AbstractChar}, space::AbstractHilbertSpace{<:AbstractBasisState}) =state_index(interpret_state(state, space), space)
basisstate(state::Union{<:AbstractString,<:AbstractChar}, space::AbstractHilbertSpace) = basisstate(_checked_state_index(state, space), space)
function _checked_state_index(state, space)
    i = state_index(state, space)
    iszero(i) && throw(ArgumentError("The state `$state` is not in the Hilbert space."))
    i
end

function interpret_state(input::Union{AbstractString,AbstractChar}, ::Type{B}) where {I,B<:FockNumber{I}}
    all(c -> c == '0' || c == '1', input) || throw(ArgumentError("Fock strings must contain only `'0'` and `'1'`, got `\"$input\"`."))
    bits = map(c -> c == '1', collect(input))
    return B(focknbr_from_bits(bits, I))
end

function interpret_state(input::Union{AbstractString,AbstractChar}, ::Type{BosonicState})
    n = try
        parse(Int, input)
    catch
        throw(ArgumentError("A bosonic occupation string must be an integer, got `\"$input\"`."))
    end
    BosonicState(n)
end

function interpret_state(input, space::ProductSpace{B}) where B
    length(input) == length(groups(space)) || throw(ArgumentError("Product-state input must have length $(length(groups(space))), got $(length(input))"))
    return B(map(interpret_state, input, groups(space)))
end

function (k::Kets{B})(inputs...) where B
    raw = if length(inputs) == 1
        only(inputs)
    elseif B <: ProductState
        inputs
    else
        throw(ArgumentError("Expected a single state input for non-product spaces, got $(length(inputs))"))
    end
    parsed = interpret_state(raw, k.space)
    return SymbolicState(k.space, parsed, nothing)
end

# Symbolic states are tied to the degrees of freedom of their space, not to its basis
_same_space_id(a, b) = a === b || atom_ids(a) == atom_ids(b)

function _apply_symbolic_operator_to_state(op::AbstractSym, state::AbstractBasisState, space::AbstractHilbertSpace; transpose=false)
    # Tuple-backed, so that the term and its precomputation (a map over the factors) need no vectors
    term = NCMul(1, (op,))
    precomp = precomputation_before_operator_application(term, space)
    if transpose
        return apply_local_operators(term, state, space, precomp; transpose=true)
    end
    return _apply_local_operators(term, state, space, precomp)
end

function _term_matrix_representation(op::NCMul{<:Any,<:SymbolicState}, H::AbstractHilbertSpace, t::EagerSparseRepr; kwargs...)
    matrix_representation(only(op.factors), H, t; kwargs...)
end

function NonCommutativeProducts.mul_effect(a::SymbolicState, b::SymbolicState)
    if !_same_space_id(a.space, b.space)
        group_id(a) == group_id(b) && throw(ArgumentError("Cannot multiply symbolic states on different spaces of the same group, got `$(a.space)` and `$(b.space)`."))
        return _group_isless(b, a) ? Swap(1) : nothing
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

function apply_local_operator(op::SymbolicState, state, space::AbstractHilbertSpace, precomp)
    _same_space_id(space, op.space) || throw(ArgumentError("The space of the symbolic state must match the provided space."))
    isketbra(op) || throw(ArgumentError("`apply_local_operator` is only defined for ketbra `SymbolicState` operators."))
    return state == op.bra ? (op.ket, 1) : (state, 0)
end

_scaled_state(amp, state) = iszero(amp) ? 0 : NCMul(amp * 1, (state,))

function NonCommutativeProducts.mul_effect(op::AbstractSym, s::SymbolicState)
    if !(symbolic_group(op) == group_id(s.space))
        return _group_isless(s, op) ? Swap(1) : nothing
    end
    if !has_ket(s)
        throw(ArgumentError("Cannot act with operator $op on the left of $s"))
    end
    newket, amp = _apply_symbolic_operator_to_state(op, s.ket, s.space)
    return _scaled_state(amp, SymbolicState(s.space, newket, s.bra))
end

function NonCommutativeProducts.mul_effect(s::SymbolicState, op::AbstractSym)
    if !(symbolic_group(op) == group_id(s.space))
        return _group_isless(op, s) ? Swap(1) : nothing
    end
    if !has_bra(s)
        throw(ArgumentError("Cannot act with operator $op on the right of $s"))
    end
    newbra, amp = _apply_symbolic_operator_to_state(op, s.bra, s.space; transpose=true)
    return _scaled_state(amp, SymbolicState(s.space, s.ket, newbra))
end

# Kets are basis vectors: ⟨a|b⟩ is zero unless a and b are the same basis state of the same group, so `inner` only
# forms the products of terms with equal keys. Bras and ketbras have no key and are paired with every term.
NonCommutativeProducts.contraction_key(s::SymbolicState) = isket(s) ? (group_id(s.space), s.ket) : nothing

# An operator string acting on a sum of kets in one space is applied to each basis state with one precomputation, as
# in `matrix_representation`. Other products are sorted factor by factor.
function NonCommutativeProducts.mul_terms!!(c, x::NCMul{<:Any,<:AbstractSym}, b::NCAdd{<:Any,<:NCMul{<:Any,<:SymbolicState}})
    space = _ket_sum_space(b)
    if isnothing(space) || !all(op -> !(op isa SymbolicState) && symbolic_group(op) == group_id(space), x.factors)
        return invoke(NonCommutativeProducts.mul_terms!!, Tuple{Any,NCMul,NCAdd}, c, x, b)
    end
    precomp = precomputation_before_operator_application(x, space)
    for (factors, coeff) in NonCommutativeProducts.termpairs(b)
        newket, amp = _apply_local_operators(x, only(factors).ket, space, precomp)
        iszero(amp) && continue
        c = add!!(c, NCMul(amp * coeff, [SymbolicState(space, newket, nothing)]))
    end
    return c
end
# The space of the sum `b` if each of its terms is a single ket in that space, and `nothing` otherwise
function _ket_sum_space(b::NCAdd)
    space = nothing
    for (factors, _) in NonCommutativeProducts.termpairs(b)
        length(factors) == 1 || return nothing
        s = only(factors)
        (s isa SymbolicState && isket(s)) || return nothing
        if isnothing(space)
            space = s.space
        elseif !_same_space_id(space, s.space)
            return nothing
        end
    end
    return space
end


representation(s::SymbolicState; kwargs...) = representation(s, s.space; kwargs...)
function representation(op; kwargs...)
    reducer(::Missing, s2) = s2
    reducer(::Missing, ::Missing) = missing
    reducer(s1, ::Missing) = s1
    reducer(s1, s2) = s1==s2 ? s1 : throw(ArgumentError("Cannot represent a product of symbolic states in different spaces"))
    type = _operator_type(op)
    if type in (:kets, :bras, :ketbras)
        _iszero_ncadd(op) && throw(ArgumentError("Cannot infer the space of the zero state `$op`. Pass the Hilbert space, as in representation(op, H)."))
        space =NonCommutativeProducts.ncmapreduce(op -> op.space, (reducer, reducer), op; scalarmap=x->missing)
        return representation(op, space; kwargs...)
    elseif type == :mixed
        throw(_mixed_error(op))
    else
        throw(ArgumentError("Don't know the space of the operator $op. Pass the Hilbert space, as in representation(op, H)."))
    end
end
function _mixed_error(op)
    ArgumentError("Cannot represent $op, which combines kets or bras with operators or with states of other spaces. Apply the operators to the states first, or represent the operators and the states separately with representation.")
end

function _operator_type(op::SymbolicState)
    if isket(op)
        return :kets
    elseif isbra(op)
        return :bras
    else
        return :ketbras
    end
end
_operator_type(::Union{UniformScaling,Number}) = :operator
_operator_type(op) = throw(ArgumentError("Cannot represent an object of type $(typeof(op)). Expected a symbolic operator or state, or a number. Basis states and state strings can be represented with representation(state, H)."))
# Returns one of
#   :kets, :bras, :ketbras  the expression is built from symbolic states of that kind (and scalars) only,
#   :operator               there are no kets and bras: operators, scalars, and ketbras combined with operators,
#   :mixed                  kets or bras are combined with other states or operators, which has no representation.
function _operator_type(op::Union{NCMul,NCAdd,AbstractSym})
    _iszero_ncadd(op) && return _zero_operator_type(op)
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
    elseif !hasket && !hasbra
        return :operator
    else
        return :mixed
    end
end
_iszero_ncadd(op) = op isa NCAdd && iszero(op.coeff) && isempty(op.dict)
# A zero has no factors left to inspect, so classify it by its grades instead
function _zero_operator_type(op)
    types = unique(_grade_type(g) for g in NonCommutativeProducts.grades(op))
    isempty(types) && return :operator
    length(types) == 1 ? only(types) : :mixed
end
_grade_type(::Nothing) = :operator
function _grade_type(g::StateGrade)
    allequal(values(g.grades)) || return :mixed
    b = first(values(g.grades))
    b == Bicyclic(1, 0) && return :kets
    b == Bicyclic(0, 1) && return :bras
    b == Bicyclic(1, 1) && return :ketbras
    return :mixed
end
_grade_type(g) = :mixed

"""
    vector_representation(state, space, [repr]; kwargs...) -> vector or transposed vector

Return the vector of `state` in `space`. `state` is a basis state, a symbolic ket or bra, or a product or sum of
symbolic kets or of symbolic bras. Kets give a column vector, and bras give the transpose of one.
The optional `repr` selects the format: `:sparse` (the default), `:dense`, or a format object
(`EagerSparseRepr()`, `EagerDenseRepr()`).
"""
function vector_representation(state::B, space::AbstractHilbertSpace{B}, repr::EagerSparseRepr{T}=EagerSparseRepr(); kwargs...) where {B,T}
    ind = _checked_state_index(state, space)
    v = _zero_vector(space, repr)
    v[ind] = one(eltype(v))
    return v
end
function vector_representation(state::B, space::AbstractHilbertSpace{B}, repr::EagerDenseRepr{T}; kwargs...) where {B,T}
    ind = _checked_state_index(state, space)
    v = _zero_vector(space, repr)
    v[ind] = one(eltype(v))
    return v
end
_vector_eltype(::Union{EagerSparseRepr{T},EagerDenseRepr{T}}) where T = T === Missing ? Int : T
_zero_vector(space, repr::EagerSparseRepr, T=_vector_eltype(repr)) = SparseArrays.spzeros(T, dim(space))
_zero_vector(space, repr::EagerDenseRepr, T=_vector_eltype(repr)) = zeros(T, dim(space))
_zero_matrix(space, ::EagerSparseRepr, T) = SparseArrays.spzeros(T, dim(space), dim(space))
_zero_matrix(space, ::EagerDenseRepr, T) = zeros(T, dim(space), dim(space))
function vector_representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    vector_representation(1 * s, space, repr; kwargs...)
end
vector_representation(op, space, repr; kwargs...) = vector_representation(op, space, _canonicalize_repr(repr); kwargs...)

function vector_representation(op::Union{NCMul,NCAdd}, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    _vector_representation(op, space, repr, _operator_type(op); kwargs...)
end
# `kind` is `:kets` or `:bras`, as computed by `_operator_type`
function _vector_representation(op::NCMul, space::AbstractHilbertSpace, repr, kind; kwargs...)
    symstates = op.factors
    group_ids = map(state -> group_id(state.space), symstates)
    perm = map(id -> findfirst(==(id) ∘ group_id, groups(space)), group_ids)
    all(!isnothing, perm) || throw(ArgumentError("Spaces of symbolic states in NCMul do not match the factor spaces of the provided ProductSpace"))
    basis_state(symstate::SymbolicState) = kind == :kets ? symstate.ket : symstate.bra
    vec = if length(group_ids) == 1
        op.coeff * vector_representation(basis_state(only(symstates)), space, repr)
    else
        prodstate = ProductState(ntuple(n -> basis_state(symstates[perm[n]]), length(perm)))
        op.coeff * vector_representation(prodstate, space, repr)
    end
    if kind == :kets
        return vec
    elseif kind == :bras
        return transpose(vec)
    else
        throw(ArgumentError("`vector_representation` requires an `NCMul` that is purely kets or purely bras."))
    end
end
function _vector_representation(op::NCAdd, space::AbstractHilbertSpace, repr, kind; kwargs...)
    iszero(op.coeff) || throw(ArgumentError("`vector_representation` does not support an `NCAdd` with nonzero `coeff`, got $(op.coeff)."))
    if isempty(op.dict) # a zero ket or bra
        kind in (:kets, :bras) || throw(ArgumentError("`vector_representation` requires an `NCAdd` that is purely kets or purely bras."))
        # same eltype as the nonzero path, where each basis vector is scaled by a coefficient
        _repr = _canonicalize_repr(repr)
        vec = _zero_vector(space, _repr, promote_type(_vector_eltype(_repr), valtype(op.dict)))
        return kind == :kets ? vec : transpose(vec)
    end
    terms = collect(pairs(op.dict))
    return _pairwise_sum(terms, 1, length(terms)) do (term, coeff)
        coeff * _vector_representation(term, space, repr, kind; kwargs...)
    end
end
# Summing in a balanced tree keeps the cost of adding many sparse vectors quasilinear
function _pairwise_sum(f, xs, lo, hi)
    lo == hi && return f(xs[lo])
    mid = (lo + hi) >> 1
    return _pairwise_sum(f, xs, lo, mid) + _pairwise_sum(f, xs, mid + 1, hi)
end

function representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    if isket(s) || isbra(s)
        _same_space_id(space, s.space) || throw(ArgumentError("The space of the symbolic ket or bra must match the provided space. A vector needs a state for every factor of a product space."))
        return vector_representation(s, space, repr; kwargs...)
    end
    matrix_representation(s, space, repr; kwargs...)
end
function matrix_representation(s::SymbolicState, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    isketbra(s) || throw(ArgumentError("`matrix_representation` is only defined for ketbra `SymbolicState` operators. Use `vector_representation(state, H)` for a ket or bra."))
    # a ketbra on one factor of a product space acts as the identity on the other factors
    _same_space_id(space, s.space) || return matrix_representation(1 * s, space, repr; kwargs...)
    ket_vec = vector_representation(SymbolicState(space, s.ket, nothing), space, repr)
    bra_vec = vector_representation(SymbolicState(space, nothing, s.bra), space, repr)
    return ket_vec * bra_vec
end


function representation(op, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    type = _operator_type(op)
    if type == :kets || type == :bras
        return _vector_representation(op, space, repr, type; kwargs...)
    elseif type == :ketbras
        _repr = _canonicalize_repr(repr)
        if _iszero_ncadd(op) && _repr isa Union{EagerSparseRepr,EagerDenseRepr}
            return _zero_matrix(space, _repr, _repr_eltype(mat_eltype(op), _repr))
        end
        return matrix_representation(op, space, repr; kwargs...)
    elseif type == :mixed
        throw(_mixed_error(op))
    else
        return matrix_representation(op, space, repr; kwargs...)
    end
end
function representation(op::Union{AbstractString,AbstractChar}, space::AbstractHilbertSpace, repr=EagerSparseRepr(); kwargs...)
    state = interpret_state(op, space)
    return vector_representation(state, space, repr; kwargs...)
end
function representation(state::B, space::AbstractHilbertSpace{B}, repr=EagerSparseRepr(); kwargs...) where B
    return vector_representation(state, space, repr; kwargs...)
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
    vfd = representation(sf, Hf, :dense)
    @test vf[3] == 1
    @test count(!iszero, vf) == 1
    @test vfd isa Vector
    @test vfd == vf

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

@testitem "vector_representation of symbolic kets and bras" begin
    @fermions f
    Hf = hilbert_space(f, 1:2)
    vf = Kets(Hf)
    ket = 2 * vf("10") + vf("01")
    v = vector_representation(ket, Hf)
    @test v == representation(ket, Hf)
    @test vector_representation(ket', Hf) == transpose(v)
    @test vector_representation(1 * vf("10"), Hf, :dense) isa Vector
end

@testitem "representation of different kinds of input" begin
    using LinearAlgebra, SparseArrays
    @fermions f
    @majoranas γ
    @boson b
    @spin s 1 // 2
    Hf = hilbert_space(f, 1:2)
    df = dim(Hf)
    idf = Matrix(I, df, df)

    @testset "scalars" begin
        @test Matrix(representation(2, Hf)) == 2 * idf
        @test Matrix(representation(I, Hf)) == idf
        @test iszero(representation(0, Hf))
        @test representation(2, Hf, :dense) == 2 * idf
        @test_throws ArgumentError representation(2)
        @test_throws ArgumentError representation(I)
    end

    @testset "operators that reduce to a scalar" begin
        op = f[1] - f[1] + 2
        @test Matrix(representation(op, Hf)) == 2 * idf
        @test Matrix(representation((f[1] + 2) - f[1], Hf)) == 2 * idf
        @test Matrix(representation(f[1]' * f[1] - f[1]' * f[1] + 2, Hf)) == 2 * idf
        @test_throws ArgumentError representation(op)
    end

    @testset "zero operators" begin
        @test iszero(representation(0 * f[1], Hf))
        @test size(representation(0 * f[1], Hf)) == (df, df)
        @test iszero(representation(f[1] - f[1], Hf))
        @test size(representation(f[1] - f[1], Hf)) == (df, df)
        @test_throws ArgumentError representation(0 * f[1])
        @test_throws ArgumentError representation(f[1] - f[1])
    end

    @testset "spin" begin
        Hs = hilbert_space(s)
        id2 = Matrix(I, 2, 2)
        # s[:I] and products that reduce to a number are Rationals or expressions without operators
        @test Matrix(representation(s[:I], Hs)) == id2
        @test Matrix(representation(s[:z] * s[:z], Hs)) ≈ id2 / 4
        @test iszero(representation(s[:+] * s[:+], Hs))
        @test representation(s[:x], Hs) == matrix_representation(s[:x], Hs)
        @test Matrix(representation(s[:x], Hs)) ≈ [0 0.5; 0.5 0]
        @test_throws ArgumentError representation(s[:I])
    end

    @testset "majoranas and bosons" begin
        Hm = hilbert_space(γ, 1:4)
        @test representation(γ[1], Hm) == matrix_representation(γ[1], Hm)
        @test Matrix(representation(γ[1] * γ[1], Hm)) == Matrix(I, dim(Hm), dim(Hm))
        m12 = Matrix(representation(γ[1] * γ[2], Hm))
        @test Matrix(representation(γ[1] * γ[2] + 1, Hm)) == m12 + I
        @test_throws ArgumentError representation(γ[1])

        Hb = hilbert_space(b, 4)
        @test representation(b, Hb) == matrix_representation(b, Hb)
        nb = Matrix(representation(b' * b, Hb))
        @test sort(diag(nb)) ≈ collect(0:3)
        @test Matrix(representation(b' * b + 2, Hb)) ≈ nb + 2I
        @test_throws ArgumentError representation(b)
    end

    @testset "ketbras combined with operators" begin
        Hb = hilbert_space(b, 3)
        H = tensor_product(Hf, Hb)
        vf, vb = Kets(Hf), Kets(Hb)
        kb = vb("0") * vb("1")'
        @test representation(kb, H) ≈ embed(representation(kb, Hb), Hb => H)
        @test representation(kb, H) == matrix_representation(kb, H)
        op = f[1]' * kb
        @test representation(op, H) == matrix_representation(op, H)
        @test representation(op, H) ≈ representation(f[1]', H) * representation(kb, H)
        @test representation(op, H, :dense) ≈ representation(op, H)
        @test_throws ArgumentError representation(op)

        proj = vf("10") * vf("10")'
        @test Matrix(representation(proj + 1, Hf)) == Matrix(representation(proj, Hf)) + I
    end

    @testset "kets and bras combined with other objects" begin
        Hb = hilbert_space(b, 3)
        H = tensor_product(Hf, Hb)
        vf, vb = Kets(Hf), Kets(Hb)
        @test_throws ArgumentError representation(f[1]' * vb("0"), H)
        @test_throws ArgumentError representation(f[1]' * vb("0"))
        @test_throws ArgumentError representation(vf("10") * vb("1")', H)
        @test_throws ArgumentError representation(vb("1"), H)
        @test_throws ArgumentError representation(vf("10") + 1, Hf)
    end

    @testset "basis states, strings and unsupported input" begin
        @test representation("10", Hf) == representation(basisstate("10", Hf), Hf)
        @test_throws ArgumentError representation("10")
        @test_throws ArgumentError representation([1, 2], Hf)
    end

    @testset "single-particle and BdG spaces" begin
        h = f[1]' * f[2] + hc
        Hsp = hilbert_space(f, 1:2, SingleParticle())
        @test representation(h, Hsp) == matrix_representation(h, Hsp)
        @test size(representation(2, Hsp)) == (2, 2)
        @test iszero(representation(2, Hsp))
        @test iszero(representation(I, Hsp))
        @test iszero(representation(f[1]' * f[1] - f[1]' * f[1] + 2, Hsp))
        @test_throws ArgumentError representation(f[1], Hsp)
        @test_throws ArgumentError representation(b' * b, Hsp)
        @test_throws ArgumentError representation(Ket("10", Hf), Hsp)

        hbdg = f[1]' * f[2] + 1im * f[1]' * f[2]' + hc
        Hbdg = hilbert_space(f, 1:2, BdG())
        @test representation(hbdg, Hbdg) == matrix_representation(hbdg, Hbdg)
        @test size(representation(2, Hbdg)) == (4, 4)
        @test iszero(representation(2, Hbdg))
        @test iszero(representation(I, Hbdg))
        @test_throws ArgumentError representation(f[1], Hbdg)
        @test_throws ArgumentError representation(b' * b, Hbdg)
        @test_throws ArgumentError representation(Ket("10", Hf), Hbdg)
    end
end

@testitem "basisstate from string" begin
    @fermions f
    H = hilbert_space(f, 1:2)
    for s in ("00", "10", "01", "11")
        @test state_index(basisstate(s, H), H) == state_index(s, H)
    end
    @test basisstate('1', hilbert_space(f, 1:1)) == basisstate(2, hilbert_space(f, 1:1))
    @test_throws ArgumentError basisstate("111", H)
    @test_throws ArgumentError basisstate("12", H)
end

@testitem "Symbolic ket and bra algebra" begin
    using LinearAlgebra

    @fermions f
    Hf = hilbert_space(f, 1:2)
    vf = Kets(Hf)

    @test Ket("10", Hf) == vf("10")
    @test vf("10")' * vf("10") == 1
    @test vf("10")' * vf("11") == 0
    @test iszero(f[1] * vf("00"))
    @test_throws ArgumentError f[1] * vf("0")'
    @test_throws ArgumentError vf("0") * f[1]

    Mouter = representation(vf("10") * vf("10")', Hf)
    @test Mouter[state_index(vf("10").ket, Hf), state_index(vf("10").ket, Hf)] == 1
    @test count(!iszero, Mouter) == 1

    ketbra = vf("10") * vf("01")'
    Mkb = representation(ketbra, Hf)
    @test Mkb[state_index(vf("10").ket, Hf), state_index(vf("01").ket, Hf)] == 1

    @test representation(vf("10"), Hf)[state_index(vf("10").ket, Hf)] == 1
    @test representation(vf("10")', Hf)[state_index(vf("10").ket, Hf)] == 1

    ket_sum = vf("10") + 2 * vf("01")
    bra_sum = vf("10")' + 2 * vf("01")'
    ketbra_sum = vf("10") * vf("10")' + 2 * vf("01") * vf("01")'
    @test representation(ket_sum) == representation(ket_sum, Hf)
    @test representation(bra_sum) == representation(bra_sum, Hf)
    @test representation(ketbra_sum) == representation(ketbra_sum, Hf)

    @test f[2]' * vf("10") == -vf("11")
    @test vf("10")' * f[2] == -vf("11")'
    @test representation(f[2]' * vf("10"), Hf) ≈ representation(f[2]', Hf) * representation(vf("10"), Hf)
    @test representation(vf("10")' * f[2], Hf) ≈ representation(vf("10")', Hf) * representation(f[2], Hf)

    @test vf("11")' * f[2]' * vf("10") == -1

    @testset "operator action matches matrix representation" begin
        cases = (
            (f[1], vf("00")),
            (f[1]', vf("00")),
            (f[1] * f[2], vf("11")),
            (f[1]' * f[2], vf("01")),
            (f[1] + f[2], vf("10")),
            (f[1], vf("00") + 2 * vf("10")),
            (f[1], vf("00") + vf("01")),
            (f[1] + f[2], vf("00")),
        )

        for (op, state) in cases
            @test representation(op, Hf) * representation(state, Hf) ≈
                representation(op * state, Hf)
        end
    end

    @boson b
    Hb = hilbert_space(b, 5)
    vb = Kets(Hb)
    @test vb("3")' * vb("3") == 1
    @test vb("3")' * vb("4") == 0
    @test_throws ArgumentError representation(vf("10") + vb("3"))

    H = tensor_product(Hf, Hb)
    vfb = Kets(H)
    sb = vb("2")
    sf = vf("0")
    rep = representation(sb * sf, H)
    @test rep ≈ tensor_product((representation(sf), representation(sb)), (Hf, Hb) => H)
    @test size(rep) == (dim(H),)
    @test sb * sf == vfb("0", "2")
    @test sb * sf * (sb * sf)' == vfb("0", "2") * vfb("0", "2")'

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
    @test representation(op, Hcons) ≈ representation(opb, Hcons) * representation(opf, Hcons)
    @test b' * op == (sqrt(2) * vb("2")vb("0")' + sqrt(2) * vb("2")vb("4")') * opf
end

@testitem "Operators on symbolic states give the coefficient amp * state" begin
    using FermionicHilbertSpaces.NonCommutativeProducts: NCAdd
    @fermions f
    vf = Kets(hilbert_space(f, 1:2))
    @spin s 1 // 2
    vs = Kets(hilbert_space(s))
    # the amplitude 1.0 of a spin raising operator keeps the coefficient type Float64
    for (x, ref) in ((f[2]' * vf("10"), -1 * vf("11")), (vf("10")' * f[2], -1 * vf("11")'),
        (s[:+] * vs("↓"), 1.0 * vs("↑")), (vs("↑")' * s[:+], 1.0 * vs("↓")'), (s[:z] * vs("↑"), 0.5 * vs("↑")))
        @test x == ref
        @test valtype(x.dict) == valtype(NCAdd(ref).dict)
    end
    # a vanishing amplitude removes the term, also inside sums
    @test f[1] * vf("00") == 0
    @test s[:+] * vs("↑") == 0
    @test (f[1] + f[2]') * vf("00") == vf("01")
end

@testitem "Operators on symbolic states agree with matrix representations" begin
    # the operator acts as a product with a Tuple of factors, on fermionic, constrained, spin, boson and product spaces
    @fermions f
    @spin s 1
    @boson b
    for (H, ops) in ((hilbert_space(f, 1:3), (f[2]', f[1], f[3]' * f[1])),
        (hilbert_space(f, 1:3, NumberConservation(1)), (f[2]' * f[1], f[3]' * f[2])),
        (hilbert_space(s), (s[:+], s[:-], s[:z])), (hilbert_space(b, 4), (b, b')),
        (tensor_product(hilbert_space(f, 1:2), hilbert_space(s)), (f[1]', s[:+], f[2]' * f[1] * s[:z])))
        for op in ops, st in basisstates(H)
            ket = FermionicHilbertSpaces.SymbolicState(H, st, nothing)
            v = Vector(representation(ket, H))
            M = representation(op, H, :dense)
            Mv, vM = M * v, vec(transpose(v) * M)
            @test iszero(Mv) ? iszero(op * ket) : Vector(representation(op * ket, H)) ≈ Mv
            @test iszero(vM) ? iszero(ket' * op) : vec(collect(representation(ket' * op, H))) ≈ vM
        end
    end
end

@testitem "Operators on sums of kets and inner products of sums agree with matrix representations" begin
    using LinearAlgebra, Random
    using FermionicHilbertSpaces.NonCommutativeProducts.VectorInterface: inner
    Random.seed!(1)
    @fermions f
    @spin s 1
    @boson b
    for (H, op) in ((hilbert_space(f, 1:4), 0.5 + f[2]' * f[1] + hc + 2.0 * f[3]' * f[3] * f[4]' * f[4] - f[4]),
        (hilbert_space(f, 1:4, NumberConservation(2)), f[2]' * f[1] + 1im * f[4]' * f[3] + hc),
        (hilbert_space(s), s[:+] + 0.3 * s[:z] * s[:-]), (hilbert_space(b, 4), b' * b + b + 1),
        (tensor_product(hilbert_space(f, 1:2), hilbert_space(s)), f[1]' * s[:+] + f[2] + hc))
        states = collect(basisstates(H))
        ψ = sum(randn() * FermionicHilbertSpaces.SymbolicState(H, st, nothing) for st in states)
        φ = sum(randn(ComplexF64) * FermionicHilbertSpaces.SymbolicState(H, st, nothing) for st in states[1:2:end])
        v, w = Vector(representation(ψ, H)), Vector(representation(φ, H))
        M = representation(op, H, :dense)
        @test Vector(representation(op * ψ, H)) ≈ M * v
        @test inner(φ, ψ) ≈ dot(w, v)
        @test inner(φ, op * ψ) ≈ dot(w, M * v)
        @test norm(ψ) ≈ norm(v)
    end
    # Products that are not an operator string on a sum of kets in one space are sorted factor by factor
    Hf, Hs = hilbert_space(f, 1:2), hilbert_space(s)
    kf, ks = Kets(Hf), Kets(Hs)
    @test (f[1]' + s[:+]) * (kf("00") + kf("01")) == kf("10") + kf("11") + s[:+] * (kf("00") + kf("01"))
    @test f[1]' * (kf("00") + kf("01") * kf("00")') == kf("10") + f[1]' * kf("01") * kf("00")'
    @test f[1]' * (kf("00") + ks("0")) == kf("10") + f[1]' * ks("0")
    # Only kets have contraction keys, so the ketbra term is paired with |00⟩ and leaves the ket |01⟩
    @test_throws ArgumentError inner(kf("01") + kf("00") * kf("01")', kf("00"))
end

@testitem "Interleaved operators and states in product spaces" begin
    using LinearAlgebra
    @fermions f
    @boson b
    Hf = hilbert_space(f, 1:2)
    fspaces = [Hf, constrain_space(Hf, NumberConservation()), constrain_space(Hf, collect(basisstates(Hf)))]
    for Hf in fspaces
        Hb = hilbert_space(b, 4)
        H = tensor_product(Hf, Hb)
        vf = Kets(Hf)
        vb = Kets(Hb)
        vfb = Kets(H)

        # Test: op_f * vf_ket * op_b * vb_ket should reorder to op_f * op_b * vf_ket * vb_ket
        # Mixed order: operator_f, state_f, operator_b, state_b
        mixed = f[1]' * vf("0") * b' * vb("2")
        @test mixed == b' * f[1]' * vf("0") * vb("2")
        @test mixed == b' * f[1]' * vb("2") * vf("0")
        @test mixed == f[1]' * b' * vf("0") * vb("2")
        @test mixed == f[1]' * b' * vfb("0", "2")

        # Test with adjoint (bra)
        mixed = vb("2")'vf("0")' * f[1] * b'
        @test mixed == vf("0")' * vb("2")' * f[1] * b'
        @test mixed == vf("0")' * f[1] * vb("2")' * b'
        @test mixed == vb("2")' * vf("0")' * f[1] * b'
        @test mixed == vfb("0", "2")' * f[1] * b'

        num = vb("2")'vf("0")' * f[1] * b' * vb("1")vf("1")
        @test iszero(num - sqrt(2))
        @test num == vb("2")'vf("0")' * f[1] * b' * vf("1")vb("1")
        @test num == vb("2")'vf("0")' * b' * f[1] * vf("1")vb("1")
        @test num == vf("0")'vb("2")' * b' * f[1] * vf("1")vb("1")
        @test num == vf("0")' * f[1] * vb("2")' * b' * vb("1")vf("1")
        @test num == vf("0")' * f[1] * vb("2")' * vf("1") * b' * vb("1")
        @test num == vfb("0", "2")' * f[1] * b' * vfb("1", "1")

        opf = vf("1")vf("0")'
        opb = vb("0")vb("1")'
        mixed = opf * opb * f[1] * b'
        @test opf * opb == vfb("1", "0") * vfb("0", "1")'
        @test mixed == opf * f[1] * opb * b'
        @test mixed == opb * opf * b' * f[1]
        @test mixed * vf("1") == vb("0")vb("0")' * vf("1")
        @test mixed * vb("0") == vf("1")vf("1")' * vb("0")
        @test FermionicHilbertSpaces._operator_type(mixed) == :ketbras

        @spin s 1 // 2
        Hs = hilbert_space(s)
        vs = Kets(Hs)
        vfbs = Kets(tensor_product(Hf, Hs, Hb))
        @test vb("0") * vs("↑") * vf("1") == vfbs("1", "↑", "0")

        @test iszero((vb("0") * vs("↑") * vf("0"))' * vb("0") * vs("↑") * vf("0") - 1)
        long_op = s[:x] * f[1] * b * (1 + s[:y]) * (1 + f[2]') * (1 + s[:z]) * b' + hc
        ket = long_op * vb("0") * vs("↑") * vf("1")
        @test FermionicHilbertSpaces._operator_type(ket) == :kets
        bra = (vb("0") * vs("↑") * vf("0"))' * long_op
        @test FermionicHilbertSpaces._operator_type(bra) == :bras

        num = (vb("0") * vs("↑") * vf("0"))' * long_op * vb("0") * vs("↑") * vf("1")
        @test FermionicHilbertSpaces.NonCommutativeProducts.isscalar(num)
        @test num == vfbs("0", "↑", "0")' * long_op * vfbs("1", "↑", "0")

        ketbra = vb("0") * vs("↑") * vf("1") * (vb("0") * vs("↑") * vf("0"))'
        @test FermionicHilbertSpaces._operator_type(ketbra) == :ketbras
    end
end

@testitem "Grading of symbolic states" begin
    import FermionicHilbertSpaces.NonCommutativeProducts as NC
    using FermionicHilbertSpaces: _operator_type
    @fermions f
    @boson b
    Hf = hilbert_space(f, 1:2)
    Hb = hilbert_space(b, 4)
    vf = Kets(Hf)
    vb = Kets(Hb)
    ket, bra, ketbra = NC.grade(vf("10")), NC.grade(vf("10")'), NC.grade(vf("10") * vf("01")')

    # the grade is the kind of an expression, independent of the labels
    @test NC.grade(vf("01")) == NC.grade(vf("10") * vf("01")' * vf("01")) == ket
    @test NC.grade(f[1] * vf("10") * vf("01")') == ketbra
    @test NC.grade(vf("10")' * vf("10")) === NC.grade(f[1]' * f[2]) === nothing

    # each space is graded separately, independently of order and bracketing
    @test NC.grade(vf("10") * vb("1")) != ket
    @test NC.grade(vf("10") * vb("1")) == NC.grade(vb("1") * vf("10"))
    @test NC.grade((vf("10") * vb("1")) * vf("01")') == NC.grade(vf("10") * (vb("1") * vf("01")'))
    @test NC.grade((vf("10") * vb("1"))') == NC.grade(vf("10")' * vb("1")')

    # zeros keep their kind
    z = f[1] * vf("00")
    zkb = z * vf("10")'
    @test iszero(z) && iszero(zkb)
    @test NC.grades(z) == NC.grades(vf("10") * vf("00")' * vf("11")) == Set([ket])
    @test NC.grades(z') == NC.grades(vf("00")' * f[1]') == Set([bra])
    @test NC.grades(vf("10")' * vf("11")) == Set([nothing])
    @test NC.grades(b * vb("0") * vf("10")) == NC.grades(vb("0") * vf("10"))
    @test _operator_type.((z, z', zkb, vf("10") - vf("10"), b * vb("0") * vf("10"))) == (:kets, :bras, :ketbras, :kets, :kets)

    # zeros represent like nonzero expressions of the same kind, but have no space to infer
    for c in (1, 1.0, 1.0im), x in (c * vf("10"), c * vf("10")', c * vf("10") * vf("01")')
        zx = x - x
        @test iszero(representation(zx, Hf))
        @test typeof(representation(zx, Hf)) == typeof(representation(x, Hf))
        @test typeof(representation(zx, Hf, :dense)) == typeof(representation(x, Hf, :dense))
        @test size(representation(zx, Hf)) == size(representation(x, Hf))
    end
    @test_throws ArgumentError representation(z)
    @test_throws ArgumentError representation(z')

    # states combined with operators or states of other spaces are mixed
    @test _operator_type(b * vf("10")) == _operator_type(vf("10")' * b) == _operator_type(vb("0") * vf("10")') == :mixed
    @test_throws ArgumentError representation(b * vf("10"))

    # products respect the grading
    kb = FermionicHilbertSpaces.SymbolicState(Hf, vf("10").ket, vf("01").ket)
    pairs = [(vf("10")', vf("10")), (vf("10")', vf("01")), (vf("10"), vf("01")'), (kb, vf("01")), (kb, vf("10")),
        (vf("01")', kb), (kb, kb), (kb, kb'), (f[1], kb), (f[1], vf("10")), (f[1], vf("00")), (vf("00")', f[1]'),
        (vf("10"), vb("1")), (b, vb("0"))]
    @test all(((x, y),) -> NC.check_grading(x, y), pairs)
end

