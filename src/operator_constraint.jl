# Constraints written as `quantity => spec`, where `quantity` is a diagonal symbolic operator or a function of the basis state.
const ConstraintVector = AbstractVector{<:Union{Pair, AbstractConstraint}}
const QuantitySpec = Union{Pair, Tuple, ConstraintVector, Base.Generator, NCAdd, NCMul, AbstractSym, Function}
const ConstraintLike = Union{AbstractConstraint, QuantitySpec}

as_constraint(c::AbstractConstraint) = c
# Collected into a vector so that a long tuple of constraints does not compile code for its length.
function as_constraint(@nospecialize(cs::Union{Tuple, ConstraintVector, Base.Generator}))
    constraints = AbstractConstraint[]
    for c in cs
        c = as_constraint(c)
        c isa ProductConstraint ? append!(constraints, c.constraints) : push!(constraints, c)
    end
    isempty(constraints) && return NoSymmetry()
    return length(constraints) == 1 ? only(constraints) : ProductConstraint(map(identity, constraints))
end
as_constraint((quantity, spec)::Pair) = quantity_constraint(quantity, quantity isa Function ? _top_rule(spec) : _operator_rule(spec))
as_constraint(quantity) = quantity_constraint(quantity, AllValues())

# A rule says how `quantity => spec` treats a value `v` of the quantity: `_keep_label(rule, v, c)` returns whether `v`
# is kept and its sector label, which is `nothing` when the rule does not label.
struct AllValues end
struct ValueIs{T}
    value::T
end
struct ValueIn{C}
    values::C
end
# Reads `f(v)` by `rest`, or, without `rest`, keeps the values where `f` returns `true` if it returns a `Bool` and
# labels by `f(v)` otherwise. Which of the two is known once `f` is applied, so `predicate` is set by the first value.
mutable struct ValueMap{F, R}
    const f::F
    const rest::R
    predicate::Union{Nothing, Bool}
    const cache::Dict{Any, Any}
end
ValueMap(f, rest) = ValueMap(f, rest, nothing, Dict{Any, Any}())

_rule(x::Union{Number, AbstractString, Symbol, AbstractChar}) = ValueIs(x)
_rule(xs::Union{AbstractArray, AbstractSet, Tuple}) = ValueIn(xs)
function _rule((f, rest)::Pair)
    f isa _NotCallable && throw(ArgumentError("In a constraint `quantity => f => rest`, `f` must be a function applied to the value, got `$(repr(f))`."))
    return ValueMap(f, _rule(rest))
end
_rule(x::Union{Missing, Nothing}) = throw(ArgumentError("`=> $x` is not a constraint; leave it out to keep all values and label sectors by them."))
_rule(::AbstractDict) = throw(ArgumentError("A dictionary is not a constraint; use a function such as `v -> d[v]`."))
_rule(f) = ValueMap(f, nothing)
const _NotCallable = Union{Number, AbstractString, Symbol, AbstractChar, AbstractArray, AbstractSet, Tuple, Pair, Missing, Nothing, AbstractDict}
_top_rule(x::Bool) = throw(ArgumentError("`=> $x` is not a constraint; pass a function such as `isodd` to keep the values where it returns `true`."))
_top_rule(x) = _rule(x)
# The values of an operator are exact numbers, so a float could only match them by accident.
_operator_rule(x) = _top_rule(x)
_operator_rule(x::AbstractFloat) = ValueIs(_exact(x))
_operator_rule(xs::Union{AbstractArray, AbstractSet, Tuple}) = ValueIn(any(x -> x isa AbstractFloat, xs) ? map(_exact, collect(xs)) : xs)
_operator_rule(x::Union{AbstractString, Symbol, AbstractChar}) = throw(ArgumentError("The values of a constraint operator are numbers, got `$(repr(x))`. Label sectors by other values with a function, such as `N => (v -> iseven(v) ? \"even\" : \"odd\")`."))
_exact(x::AbstractFloat) = isinteger(x) ? Int(x) : throw(ArgumentError("The allowed values of a constraint operator must be exact, such as `1//10` instead of `0.1`, got $x."))
_exact(x) = x

_keep_label(::AllValues, v, c) = (true, v)
_keep_label(r::ValueIs, v, c) = (isequal(v, r.value), nothing)
_keep_label(r::ValueIn, v, c) = (v in r.values, v)
function _keep_label(r::ValueMap, v, c)
    w = get(r.cache, v, r)
    w === r && (w = r.cache[v] = _apply(r, v, c))
    return isnothing(r.rest) ? (w isa Bool ? (w, nothing) : (true, w)) : _keep_label(r.rest, w, c)
end
function _apply(r::ValueMap, v, c)
    w = try
        r.f(v)
    catch e
        throw(ArgumentError("The constraint `$c` failed to apply `$(r.f)` to the value $(repr(v)): $(sprint(showerror, e))"))
    end
    isnothing(r.rest) || return w
    isnothing(r.predicate) && (r.predicate = w isa Bool)
    r.predicate == (w isa Bool) || throw(ArgumentError("In the constraint `$c`, `$(r.f)` returns a `Bool` for some values and $(repr(w)) for $(repr(v)); it must either select values by returning a `Bool` or label them."))
    return w
end
# Whether the rule labels sectors, or `nothing` when that is not known yet.
_labels(::AllValues) = true
_labels(::ValueIs) = false
_labels(::ValueIn) = true
_labels(r::ValueMap) = isnothing(r.rest) ? (isnothing(r.predicate) ? nothing : !r.predicate) : _labels(r.rest)
_reset!(r) = nothing
_reset!(r::ValueMap) = (r.predicate = nothing; _reset!(r.rest))
_show_rule(io, ::AllValues) = nothing
_show_rule(io, r::ValueIs) = print(io, " => ", repr(r.value))
_show_rule(io, r::ValueIn) = print(io, " => ", r.values)
_show_rule(io, r::ValueMap) = (print(io, " => ", r.f); isnothing(r.rest) || _show_rule(io, r.rest))
_kept_label(c, value) = ((keep, label) = _keep_label(c.rule, value, c); keep ? label : missing)

constrain_space(space, @nospecialize(c::QuantitySpec), args...) = constrain_space(space, as_constraint(c), args...)
constrain_space(space::AbstractHilbertSpace, @nospecialize(cs::ConstraintVector)) = constrain_space(space, as_constraint(cs))
Base.:*(c::AbstractConstraint, p::Pair) = c * as_constraint(p)
Base.:*(p::Pair, c::AbstractConstraint) = as_constraint(p) * c

# The constraint `f => spec` on a function `f` of the basis state, which is evaluated on each state.
struct StateConstraint{F, R} <: AbstractConstraint
    f::F
    rule::R
end
quantity_constraint(f::Function, rule) = StateConstraint(f, rule)
supports_branch_pruning(::StateConstraint) = false
supports_filtering(::StateConstraint) = true
supports_sector_grouping(c::StateConstraint) = _labels(c.rule) === true
supply_missing_constraint_info(c::StateConstraint, space, spaces) = c
sector_function(c::StateConstraint, ::AbstractHilbertSpace) = state -> _kept_label(c, c.f(state))
filter_function(c::StateConstraint, space::AbstractHilbertSpace) = !ismissing ∘ sector_function(c, space)
localize_constraint(c::StateConstraint, H::SectorHilbertSpace) = SectorConstraint(only, (sector_function(c, H),), (H,))
Base.show(io::IO, c::StateConstraint) = (print(io, c.f); _show_rule(io, c.rule))

# The constraint `op => spec` on a diagonal operator `op`. `terms` holds its values on `spaces`, and `allowed` the sorted
# values that `spec` keeps, for pruning, or `missing` when it keeps all values or they are not listed.
struct OperatorConstraint{O, R, H, D, A} <: AbstractAdditiveConstraint
    op::O
    rule::R
    spaces::H
    terms::D
    allowed::A
end
quantity_constraint(op, rule) = OperatorConstraint(op, rule, missing, nothing, missing)
allowed_values(c::OperatorConstraint) = c.allowed
function with_spaces(c::OperatorConstraint, spaces)
    units = collect(spaces)
    number = _number_constraint(c.op, units, _number_total(c.rule))
    terms = isnothing(number) ? diagonal_terms(c.op, units) : number
    return OperatorConstraint(c.op, c.rule, units, terms, _pruning_targets(c.rule, c, _value_range(terms)))
end
# The terms index the states of each space by position, which tagging and transposing keep.
map_spaces(f, c::OperatorConstraint) = OperatorConstraint(c.op, c.rule, _map_spaces(f, c.spaces), map_spaces(f, c.terms), c.allowed)

# Whether a function `spec` selects or labels is decided by applying it to the value of one of the candidate states.
_resolve_labels!(c, space, states) = nothing
_resolve_labels!(c::ProductConstraint, space, states) = foreach(ci -> _resolve_labels!(ci, space, states), c.constraints)
function _resolve_labels!(c::Union{OperatorConstraint, StateConstraint}, space, states)
    isnothing(_labels(c.rule)) && !isempty(states) && filter_function(c, space)(first(states))
    return nothing
end

_pruning_targets(::AllValues, c, values) = missing
function _pruning_targets(r::Union{ValueIs, ValueIn}, c, values)
    targets = sort!(filter(x -> x isa Real, collect(r isa ValueIs ? (r.value,) : r.values)))
    _check_reachable(c, targets, values)
    return targets
end
# `spec` is applied to the values in the range of the operator, which may include values that no state has. If that
# fails, the values are found at the states instead, so that errors do not depend on the range.
function _pruning_targets(r::ValueMap, c, values)
    isnothing(values) && return missing
    targets = try
        filter(v -> first(_keep_label(r, v, c)), values)
    catch
        _reset!(r)
        return missing
    end
    _check_reachable(c, targets, values)
    return length(targets) == length(values) ? missing : targets
end
_check_reachable(c, targets, values) = isnothing(values) || any(in(values), targets) || throw(ArgumentError("No value of `$(c.op)` is allowed by `$c`; its values are in $(isone(step(values)) ? "$(first(values)):$(last(values))" : values)."))
# The values of the operator lie in `lo:step:hi`, or `nothing` when that range is too long to list.
const MAX_LISTED_VALUES = 2^16
_value_range(terms) = _value_range(terms.constant, terms.tables)
_value_range(c::NumberConservation) = _value_range(0, [unique(contribution(c, k, s) for s in basisstates(H)) for (k, H) in enumerate(c.spaces)])
function _value_range(constant, value_sets)
    lo = constant + sum(minimum, value_sets; init = zero(constant))
    hi = constant + sum(maximum, value_sets; init = zero(constant))
    step = mapreduce(vs -> mapreduce(v -> v - minimum(vs), gcd, vs), gcd, value_sets; init = zero(lo))
    iszero(step) && return lo:one(lo):lo
    (hi - lo) / step < MAX_LISTED_VALUES || return nothing
    return lo:step:hi
end

# A weighted count `Σₖ wₖ nₖ` of single fermion or boson modes is the quantity of `NumberConservation`, which is faster.
_number_total(::AllValues) = missing
_number_total(r::ValueIs{<:Integer}) = [r.value]
_number_total(r::ValueIn{<:Union{AbstractArray{<:Integer}, Tuple{Vararg{Integer}}}}) = r.values
_number_total(r) = nothing
function _number_constraint(op, units, allowed)
    isnothing(allowed) && return nothing
    terms, constant = _operator_terms(op)
    iszero(constant) || return nothing
    unit_of = Dict(id => k for (k, H) in enumerate(units) for id in atom_ids(H))
    weights = zeros(Int, length(units))
    for t in terms
        w, x = t.coeff, last(t.factors)
        (_isexact(w) && isreal(w) && isinteger(real(w))) || return nothing
        (length(t.factors) == 2 && _is_annihilator(x) && first(t.factors) == x') || return nothing
        k = get(unit_of, atomic_id(x), 0)
        (iszero(k) || length(atom_ids(units[k])) != 1) && return nothing
        weights[k] = Int(real(w))
    end
    return NumberConservation(allowed, units, weights)
end
_is_annihilator(x) = false
# The total number `n` when `c` restricts the sum of the numbers of all `modes` to `n`, whose states are built directly.
_fixed_total_number(c, modes) = nothing
function _fixed_total_number(c::OperatorConstraint{<:Any, <:ValueIs{<:Integer}, Missing}, modes)
    n = c.rule.value
    number = _number_constraint(c.op, modes, [n])
    (isnothing(number) || !all(isone, number.weights) || !(0 <= n <= length(modes))) && return nothing
    return n
end
branch_constraint(c::OperatorConstraint{<:Any, <:Any, <:Any, <:NumberConservation}, allspaces) = branch_constraint(c.terms, allspaces)
_additive_sum(c::OperatorConstraint{<:Any, <:Any, <:Any, <:NumberConservation}, substates, S::Type) = _additive_sum(c.terms, substates, S)
supports_sector_grouping(c::OperatorConstraint) = _labels(c.rule) === true
# Without spaces, the operator is split over the atoms of the constrained space, so any split fits.
branch_spaces(::OperatorConstraint{<:Any, <:Any, Missing}) = ()
supports_branch_pruning(c::OperatorConstraint{<:Any, <:Any, Missing}) = !(c.rule isa AllValues)
sector_function(c::OperatorConstraint{<:Any, <:Any, Missing}, space::AbstractHilbertSpace) = sector_function(with_spaces(c, atomic_factors(space)), space)
function sector_function(c::OperatorConstraint, space::AbstractHilbertSpace)
    mapper = state_mapper(space, c.spaces)
    S = _sum_type(c.allowed)
    return state -> _kept_label(c, _additive_sum(c, unique_split_state(state, mapper), S))
end
Base.show(io::IO, c::OperatorConstraint) = (print(io, c.op); _show_rule(io, c.rule))

function _additive_sum(c::OperatorConstraint, substates, ::Type)
    positions = map(state_index, substates, c.spaces)
    terms = c.terms
    return terms.constant + sum(g -> terms.tables[g][_table_index(terms.strides[g], positions, terms.units[g])], eachindex(terms.tables); init = zero(terms.constant))
end
_table_index(strides, positions, units) = 1 + sum(j -> strides[j] * (positions[units[j]] - 1), eachindex(units); init = 0)

# The value of an operator on the product of some spaces is `constant` plus, for each group `g`, the entry of `tables[g]`
# at the positions of the states of the spaces `units[g]`, with the first varying fastest.
struct DiagonalTerms{T}
    constant::T
    units::Vector{Vector{Int}}
    strides::Vector{Vector{Int}}
    tables::Vector{Vector{T}}
end

# Each term goes to the group of its support, the spaces it acts on. A term whose support lies inside larger supports is
# shared equally among them, so that a sum of local products such as `Σᵢ nᵢnᵢ₊₁` regains the expanded single-site pieces
# of each product and its groups take few values, which keeps the bounds for pruning tight.
function diagonal_terms(op, spaces)
    terms, constant = _operator_terms(op)
    unit_of = Dict(id => k for (k, H) in enumerate(spaces) for id in atom_ids(H))
    supports = map(terms) do term
        sort!(unique(map(x -> _unit_of(x, unit_of, spaces), term.factors)))
    end
    distinct = unique(supports)
    groups = filter(s -> !any(t -> length(t) > length(s) && issubset(s, t), distinct), distinct)
    foreach(g -> _check_table_size(spaces[g], terms[findfirst(==(g), supports)]), groups)
    group_terms = [Tuple{Any, NCMul}[] for _ in groups]
    for (term, support) in zip(terms, supports)
        owners = findall(g -> issubset(support, g), groups)
        foreach(g -> push!(group_terms[g], (term.coeff * (1 // length(owners)), term)), owners)
    end
    tables = Vector{Rational{Int}}[_diagonal_table(ts, spaces[g]) for (g, ts) in zip(groups, group_terms)]
    _check_exact(constant, constant)
    isreal(constant) || throw(ArgumentError("A constraint operator must be Hermitian, but it has the constant term $(_number_string(constant))."))
    strides = [cumprod([1; map(dim, spaces[g][1:(end - 1)])]) for g in groups]
    return _diagonal_terms(Rational{Int}(real(constant)), groups, strides, tables)
end
# Values are summed as integers when the groups with fractional values become integral by sharing the constant equally,
# as for `Σᵢ (Sᶻᵢ + 1//2)`. This depends on the terms of the operator rather than on how many there are.
function _diagonal_terms(constant, units, strides, tables)
    fractional = findall(t -> !all(isinteger, t), tables)
    share = isempty(fractional) ? zero(constant) : constant / length(fractional)
    rest = constant - share * length(fractional)
    integral = isinteger(rest) && all(g -> all(x -> isinteger(x + share), tables[g]), fractional)
    integral || return DiagonalTerms{Rational{Int}}(constant, units, strides, tables)
    return DiagonalTerms{Int}(Int(rest), units, strides, [Int.(g in fractional ? t .+ share : t) for (g, t) in enumerate(tables)])
end
# A term acting on several spaces is tabulated on their product space, whose dimension grows exponentially.
const MAX_TABLE_SIZE = 2^12
function _check_table_size(spaces, term)
    entries = prod(dim, spaces)
    (length(spaces) == 1 || entries <= MAX_TABLE_SIZE) && return nothing
    throw(ArgumentError("The constraint operator has the term `$(_term_string(term))` acting on $(length(spaces)) spaces together, which needs a table of $entries values, more than the limit of $MAX_TABLE_SIZE. Use `ParityConservation` for a parity, and otherwise write the quantity as a sum of terms that each act on few spaces."))
end
_operator_terms(op::NCAdd) = collect(NCterms(op)), op.coeff
_operator_terms(op::NCMul) = [op], 0
_operator_terms(op::AbstractSym) = _operator_terms(1 * op)
_operator_terms(op) = throw(ArgumentError("A constraint quantity must be a symbolic operator or a function of the basis state, got `$op`."))

function _unit_of(x, unit_of, spaces)
    k = get(unit_of, atomic_id(x), 0)
    iszero(k) || return k
    k = something(findfirst(H -> _acts_in(x, H), spaces), 0)
    iszero(k) && throw(ArgumentError("The constraint operator acts on `$x`, which is not in the space."))
    return k
end
# A symbol that names no atom, such as a single Majorana, acts in the space that contains it.
_acts_in(x, H) = false

_isexact(x) = x isa Union{Integer, Rational}
_isexact(x::Complex) = _isexact(real(x)) && _isexact(imag(x))
_check_exact(x, term) = _isexact(x) || throw(ArgumentError("A constraint operator needs integer or rational coefficients, such as `1//2` instead of `0.5`, so that its values are exact. Got the term `$(_term_string(term))`."))
_term_string(term::NCMul) = string(isone(term.coeff) ? "" : _number_string(term.coeff) * "*", NCMul(1, term.factors))
_term_string(x) = _number_string(x)
_number_string(x::Rational) = isinteger(x) ? string(numerator(x)) : "($x)"
_number_string(z::Complex) = iszero(imag(z)) ? _number_string(real(z)) : iszero(real(z)) ? (isone(imag(z)) ? "im" : "$(_number_string(imag(z)))im") : "($(_number_string(real(z))) + $(_number_string(imag(z)))im)"
_number_string(x) = string(x)

# The diagonal of a sum of weighted terms on the product of `spaces`, indexed by the positions of the states of `spaces`.
# Each term's diagonal is found without its coefficient, as the floating point values of a monomial are fractions with small
# denominators, and then multiplied by the exact coefficient.
function _diagonal_table(weighted_terms, spaces)
    H = length(spaces) == 1 ? only(spaces) : tensor_product(spaces)
    monomials = map(((_, t),) -> representation(NCMul(1, t.factors), H), weighted_terms)
    if !isdiag(sum(map(((w, _), m) -> complex(float(w)) * m, weighted_terms, monomials)))
        k = something(findfirst(!isdiag, monomials), 1)
        throw(ArgumentError("A constraint operator must be diagonal in the basis, but its term `$(_term_string(weighted_terms[k][2]))` is not."))
    end
    foreach(((w, t),) -> _check_exact(w, t), weighted_terms)
    contributions = map(((w, t), m) -> w .* map(v -> _exact_value(v, t), Vector(diag(m))), weighted_terms, monomials)
    k = findfirst(c -> !all(isreal, c), contributions)
    isnothing(k) || throw(ArgumentError("A constraint operator must be Hermitian, but its term `$(_term_string(weighted_terms[k][2]))` has complex values."))
    values = Rational{Int}.(real.(sum(contributions)))
    length(spaces) == 1 && return values
    mapper = state_mapper(H, spaces)
    return vec([values[state_index(only(first(combine_states(states, mapper))), H)] for states in Iterators.product(map(basisstates, spaces)...)])
end
function _exact_value(x::Real, term)
    _isexact(x) && return Rational{Int}(x)
    r = rationalize(Int, x; tol = 1.0e-10 * max(1, abs(x)))
    abs(r - x) <= 4 * eps(max(1.0, abs(x))) || throw(ArgumentError("The values of the constraint operator term `$(_term_string(term))` are not exact fractions, such as $x."))
    return r
end
_exact_value(z::Complex, term) = complex(_exact_value(real(z), term), _exact_value(imag(z), term))

function branch_constraint(c::OperatorConstraint, allspaces)
    ismissing(c.allowed) && return BranchConstraint((_, _, _) -> true)
    terms = c.terms
    unit_positions = _subspace_positions(c.spaces, allspaces)
    positions = map(units -> unit_positions[units], terms.units)
    n = length(allspaces)
    done = map(maximum, positions)
    order = sortperm(done)
    starts = searchsortedfirst.(Ref(done[order]), 1:(n + 1))
    T = typeof(terms.constant)
    lo, hi = zeros(T, n), zeros(T, n)
    for (d, table) in zip(done, terms.tables)
        lo[d] += minimum(table)
        hi[d] += maximum(table)
    end
    index = zeros(Int, n)
    increment = (partial, depth) -> begin
        index[depth] = state_index(partial[depth], allspaces[depth])
        value = zero(T)
        for i in starts[depth]:(starts[depth + 1] - 1)
            g = order[i]
            value += terms.tables[g][_table_index(terms.strides[g], index, positions[g])]
        end
        value
    end
    return _sum_branch_constraint(increment, lo, hi, terms.constant, c.allowed, nothing)
end

@testitem "Operator constraints keep the states with allowed values" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: atomic_factors
    # The reference evaluates each quantity on the whole space and filters its basis states.
    function brute_force(H, constraints)
        keep = trues(dim(H))
        for (op, allowed) in constraints
            keep .&= map(v -> any(≈(v), allowed), real(diag(Matrix(representation(op, H)))))
        end
        return Set(collect(basisstates(H))[keep])
    end
    samestates(H, states) = Set(basisstates(H)) == states
    @fermions f
    @spins S 1 // 2
    @boson a
    @bosons b
    @majoranas γ
    n(i) = f[i]'f[i]
    L = 4
    hubbard = [(i, σ) for i in 1:L for σ in (:↑, :↓)]
    nσ(i, σ) = f[i, σ]' * f[i, σ]
    Nup, Ndn = sum(nσ(i, :↑) for i in 1:L), sum(nσ(i, :↓) for i in 1:L)
    D = sum(nσ(i, :↑) * nσ(i, :↓) for i in 1:L)
    Hmixed = (hilbert_space(a, 4), hilbert_space(f, 1:2), hilbert_space(S[1]))
    cases = [
        (hilbert_space(f, hubbard), (Nup => 2, Ndn => 2, D => 0)),
        (hilbert_space(S, 1:6), (sum(S[i][:z] for i in 1:6) => 0,)),
        (hilbert_space(f, 1:6), (sum(n(i) for i in 1:6) => 3, sum(i * n(i) for i in 1:6) => 10)),
        (tensor_product(Hmixed), (a'a + n(1) + n(2) + S[1][:z] => [1 // 2, 3 // 2],)),
        (hilbert_space(b, 1:3, 4), (sum(b[i]'b[i] for i in 1:3) => 3,)),
        (hilbert_space(γ, 1:4), (1im * γ[1] * γ[2] => 1,)),
    ]
    for (H, constraint) in cases
        expected = brute_force(H, constraint)
        @test samestates(constrain_space(H, constraint), expected)
        H isa FermionicHilbertSpaces.MajoranaHilbertSpace && continue
        @test samestates(tensor_product(collect(atomic_factors(H)); constraint), expected)
    end
end

@testitem "Operator constraints match the existing constraints" begin
    using FermionicHilbertSpaces: AdditiveConstraint
    @fermions f
    @spins S 1 // 2
    N = sum(f[i]'f[i] for i in 1:5)
    modes = [hilbert_space(f[i]) for i in 1:5]
    @test basisstates(tensor_product(modes; constraint = N => 2)) == basisstates(tensor_product(modes; constraint = NumberConservation(2)))
    @test basisstates(hilbert_space(f, 1:5, N => 2)) == basisstates(hilbert_space(f, 1:5, NumberConservation(2)))
    Sz = sum(S[i][:z] for i in 1:5)
    Hsz, Hadd = hilbert_space(S, 1:5, Sz), hilbert_space(S, 1:5, AdditiveConstraint(missing; maps = s -> s.m))
    @test quantumnumbers(Hsz) == quantumnumbers(Hadd) && basisstates(Hsz) == basisstates(Hadd)
    up(i) = S[i][:z] + 1 // 2
    blockade = BranchConstraint((p, d, _) -> d == 1 || !(p[d].m > 0 && p[d - 1].m > 0))
    @test basisstates(hilbert_space(S, 1:8, sum(up(i) * up(i + 1) for i in 1:7) => 0)) == basisstates(hilbert_space(S, 1:8, blockade))
end

@testitem "Scalar, collection and bare quantities" begin
    @fermions f
    N = sum(f[i]'f[i] for i in 1:4)
    P = sum(i * f[i]'f[i] for i in 1:4)
    @test !(hilbert_space(f, 1:4, N => 2) isa SectorHilbertSpace)
    @test quantumnumbers(hilbert_space(f, 1:4, N => 1:2)) == [1, 2]
    @test quantumnumbers(hilbert_space(f, 1:4, N)) == 0:4
    @test quantumnumbers(hilbert_space(f, 1:4, (N => 2, P))) == 3:7
    @test quantumnumbers(hilbert_space(f, 1:4, (N, P => 3))) == [1, 2]
    @test quantumnumbers(hilbert_space(f, 1:2, (f[1]'f[1] + f[2]'f[2], f[1]'f[1] + 2f[2]'f[2]))) == [[0, 0], [1, 1], [1, 2], [2, 3]]
    @test basisstates(hilbert_space(f, 1:4, NumberConservation(2) * (P => 3))) == basisstates(hilbert_space(f, 1:4, (N => 2, P => 3)))
    @test dim(hilbert_space(f, 1:4, ())) == 16
    @test quantumnumbers(constrain_space(hilbert_space(f, 1:4, N), P => 3)) == [1, 2]
    @spins S 1 // 2
    @test quantumnumbers(hilbert_space(S, 1:3, sum(S[i][:z] for i in 1:3))) == (-3 // 2):(3 // 2)
    nup(s) = count(x -> x.m > 0, s.states)
    @test quantumnumbers(hilbert_space(S, 1:3, nup)) == 0:3
    @test dim(hilbert_space(S, 1:4, nup => 2)) == 6
    @test eltype(quantumnumbers(hilbert_space(S, 1:2, (S[1][:z] + S[2][:z], S[1][:z])))) == Vector{Rational{Int}}
    up(i) = S[i][:z] + 1 // 2
    @test dim(hilbert_space(S, 1:6, [up(i) * up(i + 1) => 0 for i in 1:5])) == 21
    @test dim(hilbert_space(S, 1:6, (up(i) * up(i + 1) => 0 for i in 1:5))) == 21
end

@testitem "Operator constraints on constrained and custom factors" begin
    @fermions f
    Hpair = hilbert_space(f, 1:2, NumberConservation(1))
    Hlevel = GenericHilbertSpace(:level, ["a", "b", "c"])
    Hs = (Hpair, Hlevel, hilbert_space(f[3]), hilbert_space(f[4]))
    N = sum(f[i]'f[i] for i in 1:4)
    full = collect(basisstates(tensor_product(Hs)))
    expected = filter(s -> count_ones(s.states[1].f) == 2, full)
    @test Set(basisstates(tensor_product(Hs; constraint = N => 2))) == Set(expected)
    @spins S 1 // 2
    @test all(L -> eltype(quantumnumbers(hilbert_space(S, 1:L, sum(S[i][:z] for i in 1:L)))) == Rational{Int}, 3:4)
end

@testitem "Invalid constraint operators throw" begin
    @fermions f
    @spins S 1 // 2
    @test_throws "diagonal" hilbert_space(f, 1:2, f[1]'f[2] + f[2]'f[1] => 1)
    @test_throws "diagonal" hilbert_space(S, 1:2, S[1][:x] + S[2][:x] => 0)
    @test_throws "rational" hilbert_space(f, 1:2, 0.5 * f[1]'f[1] => 1)
    @test_throws "Hermitian" hilbert_space(f, 1:2, 1im * f[1]'f[1] => 1)
    @test_throws ArgumentError hilbert_space(f, 1:2, f[3]'f[3] => 1)
    @test_throws "1//10" hilbert_space(f, 1:2, (1 // 10) * f[1]'f[1] => 0.1)
    @test dim(hilbert_space(f, 1:2, f[1]'f[1] => 1.0)) == 2
    @test_throws "ParityConservation" hilbert_space(f, 1:13, prod(1 - 2f[i]'f[i] for i in 1:13) => 1)
end

@testitem "Operator constraints with large denominators are exact" begin
    @spins S 1 // 2
    @test dim(hilbert_space(S, 1:2, (1 // 100003) * S[1][:z] => 1 // 200006)) == 2
end

@testitem "Specifications select and label the values of a quantity" begin
    using LinearAlgebra
    # The reference maps each kept state to its label, with `rule(v)` the label of the value `v`, `nothing` to keep
    # the state without a label and `missing` to drop it.
    reference(H, value, rule) = Dict(s => l for s in basisstates(H) for l in (rule(value(s)),) if !ismissing(l))
    labels(H::SectorHilbertSpace) = Dict(s => q for q in quantumnumbers(H) for s in basisstates(H)[indices(q, H)])
    labels(H) = Dict(s => nothing for s in basisstates(H))
    exact(x) = isinteger(x) ? Int(x) : rationalize(Int, x)
    function operator_values(op, H)
        values = exact.(real(diag(Matrix(representation(op, H)))))
        return s -> values[state_index(s, H)]
    end
    @fermions f
    H = hilbert_space(f, 1:4)
    N = sum(f[i]'f[i] for i in 1:4)
    P = sum(i * f[i]'f[i] for i in 1:4)
    @spins S 1 // 2
    Hspin = hilbert_space(S, 1:4)
    Sz = sum(S[i][:z] for i in 1:4)
    nup(s) = count(x -> x.m > 0, s.states)
    cases = [
        (H, N => isodd, N, v -> isodd(v) ? nothing : missing),
        (H, N => (v -> mod(v, 2)), N, v -> mod(v, 2)),
        (H, N => (v -> mod(v, 2)) => 0, N, v -> iseven(v) ? nothing : missing),
        (H, N => (v -> mod(v, 2)) => 0:1, N, v -> mod(v, 2)),
        (H, N => (v -> (-1)^v) => [-1, 1], N, v -> (-1)^v),
        (H, P => (v -> mod(v, 3)) => 0, P, v -> mod(v, 3) == 0 ? nothing : missing),
        (Hspin, Sz => abs => 0:1, Sz, v -> abs(v) <= 1 ? abs(v) : missing),
        (Hspin, nup => (v -> mod(v, 3)) => 0:1, nup, v -> mod(v, 3) <= 1 ? mod(v, 3) : missing),
    ]
    for (Hc, constraint, quantity, rule) in cases
        value = quantity isa Function ? quantity : operator_values(quantity, Hc)
        @test labels(constrain_space(Hc, constraint)) == reference(Hc, value, rule)
    end
    Hface = hilbert_space(f, 1:4, N => (v -> iseven(v) ? "😀" : "😢"))
    @test Set(basisstates(sector("😀", Hface))) == Set(s for s in basisstates(H) if iseven(count_ones(s.f)))
end

@testitem "Invalid specifications throw" begin
    @fermions f
    N = sum(f[i]'f[i] for i in 1:4)
    for spec in (missing, nothing, true, "a", :a, 'a', Dict(1 => 1), 1 => 2, 0.5, v -> v == 2 || v)
        @test_throws ArgumentError hilbert_space(f, 1:4, N => spec)
    end
    @test_throws "boom" hilbert_space(f, 1:4, N => (v -> v == 3 ? error("boom") : v))
    @test_throws "0:4" hilbert_space(f, 1:4, N => 7)
end

@testitem "Labels of several constraints combine" begin
    @fermions f
    N(r) = sum(f[i]'f[i] for i in r)
    H = constrain_space(hilbert_space(f, 1:4, N(1:4)), N(1:2))
    @test Set(quantumnumbers(H)) == Set([a + b, a] for a in 0:2, b in 0:2)
    @spins S 1 // 2
    nup(s) = count(x -> x.m > 0, s.states)
    pair(r) = hilbert_space(S, r, nup => (v -> mod(v, 2)))
    @test quantumnumbers(tensor_product(pair(1:2), pair(3:4))) == [[0, 0], [0, 1], [1, 0], [1, 1]]
end

@testitem "The diagonal of an operator is found numerically" begin
    import FermionicHilbertSpaces: AbstractBasisState, AbstractSym, apply_local_operator, symbolic_group
    import FermionicHilbertSpaces.NonCommutativeProducts: @nc
    # A cyclic shift, whose cube is the identity, which its products do not simplify to.
    struct Shift <: AbstractSym
        site::Int
    end
    @nc Shift
    struct Clock <: AbstractBasisState
        n::Int
    end
    apply_local_operator(::Shift, s::Clock, space, precomp) = Clock(mod(s.n + 1, 3)), 1
    symbolic_group(op::Shift) = op.site
    X(i) = Shift(i)
    Hs = [GenericHilbertSpace(k, Clock.(0:2)) for k in 1:2]
    @test Set(basisstates(tensor_product(Hs; constraint = X(1) * X(1) * X(1) + 2 * X(2) * X(2) * X(2) => 3))) == Set(basisstates(tensor_product(Hs)))
end
