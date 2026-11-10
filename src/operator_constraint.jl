# Constraints written as `quantity => spec`, where `quantity` is a diagonal symbolic operator or a function of the basis state.
const QuantitySpec = Union{Pair, Tuple, Base.Generator, NCAdd, NCMul, AbstractSym, Function}
const ConstraintLike = Union{AbstractConstraint, QuantitySpec, AbstractVector}

as_constraint(c::AbstractConstraint) = c
# Collected into a vector so that a long tuple of constraints does not compile code for its length.
function as_constraint(@nospecialize(cs::Union{Tuple, AbstractVector, Base.Generator}))
    constraints = AbstractConstraint[]
    for c in cs
        c = as_constraint(c)
        c isa ProductConstraint ? append!(constraints, c.constraints) : push!(constraints, c)
    end
    isempty(constraints) && return NoSymmetry()
    return length(constraints) == 1 ? only(constraints) : ProductConstraint(map(identity, constraints))
end
function as_constraint((quantity, spec)::Pair)
    spec isa Bool && throw(ArgumentError("`=> $spec` is not a constraint; pass a function such as `isodd` to keep the values where it returns `true`."))
    return quantity_constraint(quantity, _rule(spec))
end
as_constraint(quantity) = quantity_constraint(quantity, ValueRule{true}(identity, missing))
# Like a function at the end of a spec, a function of the state selects when it returns a `Bool`.
as_constraint(quantity::Function) = quantity_constraint(quantity, _rule(identity))
quantity_constraint(f::Function, rule) = StateConstraint(f, missing, missing, rule)
function quantity_constraint(op, rule)
    listed = rule.map === identity && rule.allowed isa Union{Tuple, AbstractArray, AbstractSet}
    for x in (listed ? rule.allowed : ())
        x isa AbstractFloat && !isinteger(x) && throw(ArgumentError("The values of a constraint operator are exact, so its allowed values must be integers or rationals, such as `$(rationalize(x))` instead of `$x`."))
    end
    return OperatorConstraint(op, rule, missing, nothing, missing)
end
# A vector holds constraints rather than basis states when its elements are constraints or quantities.
_holds_constraints(states) = false
_holds_constraints(xs::AbstractVector) = !isempty(xs) && all(x -> x isa Union{Pair, AbstractConstraint, NCAdd, NCMul, AbstractSym, Function}, xs)

constrain_space(space, @nospecialize(c::QuantitySpec), args...) = constrain_space(space, as_constraint(c), args...)
constrain_space(space, @nospecialize(cs::AbstractVector), states) = constrain_space(space, as_constraint(cs), states)
Base.:*(c::AbstractConstraint, q::Union{Pair, NCAdd, NCMul, AbstractSym}) = c * as_constraint(q)
Base.:*(q::Union{Pair, NCAdd, NCMul, AbstractSym}, c::AbstractConstraint) = as_constraint(q) * c

# How `quantity => spec` treats a value `v` of the quantity: `v` is kept when `map(v)` is in `allowed`, which `missing`
# allows entirely, and the sector is labelled by `map(v)` when `L` is true. A spec that ends in a function has
# `allowed = Bool`: the function selects values when it returns a `Bool` and labels them otherwise, and `L` is `nothing`
# until `resolve` decides which by applying it.
struct ValueRule{L, M, A}
    map::M
    allowed::A
end
ValueRule{L}(map::M, allowed::A) where {L, M, A} = ValueRule{L, M, A}(map, allowed)
_labels(::ValueRule{L}) where {L} = L
_rule(x::Union{Number, AbstractString, Symbol, AbstractChar}) = ValueRule{false}(identity, (x,))
_rule(xs::Union{AbstractArray, AbstractSet, Tuple}) = ValueRule{true}(identity, xs)
_rule((f, rest)::Pair) = (r = _rule(rest); ValueRule{_labels(r)}(_then(f, r.map), r.allowed))
_rule(f) = ValueRule{nothing}(f, Bool)
_then(f, ::typeof(identity)) = f
_then(f, g) = g ∘ f
resolve(r::ValueRule, value) = r
resolve(r::ValueRule{nothing}, value) = ValueRule{!(_apply(r.map, value) isa Bool)}(r.map, Bool)

# The sector label of the value `v`, `nothing` to keep it without a label, or `missing` to discard it.
function _label(r::ValueRule{L}, v) where {L}
    w = _apply(r.map, v)
    ismissing(w) && return missing
    if r.allowed === Bool
        (w isa Bool) == L && throw(ArgumentError("`$(r.map)` must return a `Bool` for all values of a constraint quantity, to select them, or for none, to label them, but it returns $(repr(w)) for $(repr(v)) and $(L ? "a label" : "a `Bool`") for others."))
        L || return w ? nothing : missing
    elseif !ismissing(r.allowed)
        w in r.allowed || return missing
    end
    L || return nothing
    isnothing(w) && throw(ArgumentError("`$(r.map)` returns `nothing` for the value $(repr(v)), which cannot label a sector; return `missing` to discard the states with that value."))
    return w
end
_apply(::typeof(identity), v) = v
function _apply(f, v)
    try
        return f(v)
    catch e
        throw(ArgumentError("Applying `$f` to the value $(repr(v)) of a constraint quantity failed: $(sprint(showerror, e))"))
    end
end
function _show_rule(io, r::ValueRule)
    r.map === identity || print(io, " => ", r.map)
    r.allowed isa Union{Missing, Type} && return
    return print(io, " => ", _labels(r) ? r.allowed : repr(only(r.allowed)))
end
Base.show(io::IO, c::StateConstraint) = (print(io, c.reducer); _show_rule(io, c.rule))

# The constraint `op => spec` on a diagonal operator `op`. `terms` holds its values on `spaces`, and `allowed` the sorted
# values that `spec` keeps, for pruning, or `missing` when it keeps all values or they are not listed.
struct OperatorConstraint{O, U, H, D, A} <: AbstractConstraint
    op::O
    rule::U
    spaces::H
    terms::D
    allowed::A
end
function with_spaces(c::OperatorConstraint, spaces)
    units = collect(spaces)
    terms = diagonal_terms(c.op, units)
    rule, allowed = _resolve_on_values(c, _value_range(terms))
    return OperatorConstraint(c.op, rule, units, terms, allowed)
end
supply_missing_constraint_info(c::OperatorConstraint, space, spaces) = ismissing(c.spaces) ? with_spaces(c, spaces) : c
# The terms index the states of each space by position, which tagging and transposing keep.
map_spaces(f, c::OperatorConstraint) = OperatorConstraint(c.op, c.rule, _map_spaces(f, c.spaces), c.terms, c.allowed)
# Without spaces, the operator is split over the atoms of the constrained space, so any split fits.
branch_spaces(c::OperatorConstraint) = coalesce(c.spaces, ())
supports_branch_pruning(c::OperatorConstraint) = !ismissing(ismissing(c.spaces) ? c.rule.allowed : c.allowed)
function quantity_function(c::OperatorConstraint, space::AbstractHilbertSpace)
    mapper, terms = state_mapper(space, c.spaces), c.terms
    return state -> _operator_value(terms, map(state_index, unique_split_state(state, mapper), c.spaces))
end
Base.show(io::IO, c::OperatorConstraint) = (print(io, c.op); _show_rule(io, c.rule))

const RuleConstraint = Union{StateConstraint, OperatorConstraint}
supports_filtering(::RuleConstraint) = true
supports_sector_grouping(c::RuleConstraint) = _labels(c.rule) === true
function sector_function(c::RuleConstraint, space::AbstractHilbertSpace)
    value, rule = quantity_function(c, space), c.rule
    return state -> _label(rule, value(state))
end
filter_function(c::RuleConstraint, space::AbstractHilbertSpace) = !ismissing ∘ sector_function(c, space)
# Whether a constraint follows the label rules of `quantity => spec`.
_spec_form(c) = false
_spec_form(::Union{OperatorConstraint, StateConstraint{<:Any, <:Any, <:Any, <:ValueRule}}) = true
_spec_form(c::ProductConstraint) = any(_spec_form, c.constraints)

# Decides whether a rule ending in a function selects or labels, by its value at the first of the candidate `states`.
# Without states it labels, so that the empty result has sectors.
resolve(c, space, states) = c
resolve(c::ProductConstraint, space, states) = ProductConstraint(map(ci -> resolve(ci, space, states), c.constraints))
function resolve(c::StateConstraint{<:Any, <:Any, <:Any, <:ValueRule{nothing}}, space, states)
    return StateConstraint(c.reducer, c.maps, c.spaces, _resolve_rule(c, space, states))
end
function resolve(c::OperatorConstraint, space, states)
    c = ismissing(c.spaces) ? with_spaces(c, atomic_factors(space)) : c
    c.rule isa ValueRule{nothing} || return c
    return OperatorConstraint(c.op, _resolve_rule(c, space, states), c.spaces, c.terms, c.allowed)
end
_resolve_rule(c, space, states) = isempty(states) ? ValueRule{true}(c.rule.map, Bool) : resolve(c.rule, quantity_function(c, space)(first(states)))

# Resolves the rule by the values of the operator and finds the values it keeps, for pruning. When there are too many
# values to list, or a function of the rule fails on one of them, which no state may have, this is left to the states.
const MAX_LISTED_VALUES = 2^16
function _resolve_on_values(c, values)
    rule = c.rule
    if length(values) > MAX_LISTED_VALUES || !_applies(rule.map, values)
        listed = rule.map === identity && rule.allowed isa Union{Tuple, AbstractArray, AbstractSet}
        return rule, listed ? sort!(filter(x -> x isa Real, collect(rule.allowed))) : missing
    end
    rule = resolve(rule, first(values))
    kept = filter(v -> !ismissing(_label(rule, v)), values)
    return rule, length(kept) == length(values) ? missing : kept
end
function _applies(f, values)
    try
        foreach(f, values)
    catch
        return false
    end
    return true
end
# The values of the operator lie in this range.
function _value_range(terms)
    lo = terms.constant + sum(minimum, terms.tables; init = zero(terms.constant))
    hi = terms.constant + sum(maximum, terms.tables; init = zero(terms.constant))
    step = mapreduce(t -> mapreduce(v -> v - minimum(t), gcd, t), gcd, terms.tables; init = zero(lo))
    return lo:(iszero(step) ? one(lo) : step):hi
end

# The value of an operator on the product of some spaces is `constant` plus, for each group `g`, the entry of `tables[g]`
# at the positions of the states of the spaces `units[g]`, with the first varying fastest.
struct DiagonalTerms{T}
    constant::T
    units::Vector{Vector{Int}}
    strides::Vector{Vector{Int}}
    tables::Vector{Vector{T}}
end
_operator_value(terms, positions) = terms.constant + sum(g -> terms.tables[g][_table_index(terms.strides[g], positions, terms.units[g])], eachindex(terms.tables); init = zero(terms.constant))
_table_index(strides, positions, units) = 1 + sum(j -> strides[j] * (positions[units[j]] - 1), eachindex(units); init = 0)

# Each term goes to the group of its support, the spaces it acts on. A term whose support lies inside larger supports is
# shared equally among them, so that a sum of local products such as `Σᵢ nᵢnᵢ₊₁` regains the expanded single-site pieces
# of each product and its groups take few values, which keeps the bounds for pruning tight. The values are integers when
# each group, plus an equal share of the constant, takes integer values, so that `Σᵢ (Sᶻᵢ + 1//2)` and a sum of local
# products of such terms are integers while the total `Sᶻ` is not.
function diagonal_terms(op, spaces)
    terms, constant = _operator_terms(op)
    # Sums with float coefficients, such as of spin `x` components, can leave a constant `0.0`.
    constant = iszero(constant) ? 0 : constant
    _check_exact(constant, constant)
    isreal(constant) || throw(ArgumentError("Constraint operators with complex values are not supported, but the constant term is $constant. $_ZN_HINT"))
    constant = Rational{Int}(real(constant))
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
    strides = Vector{Int}[cumprod([1; map(dim, spaces[g][1:(end - 1)])]) for g in groups]
    rational = DiagonalTerms(constant, groups, strides, tables)
    share = constant // max(1, length(tables))
    all(t -> all(v -> isinteger(v + share), t), tables) && (!isempty(tables) || isinteger(constant)) || return rational
    return DiagonalTerms(Int(first(_value_range(rational))), groups, strides, Vector{Int}[Int.(t .- minimum(t)) for t in tables])
end
# A term acting on several spaces is tabulated on their product space, whose dimension grows exponentially.
const MAX_TABLE_SIZE = 2^12
function _check_table_size(spaces, term)
    entries = prod(dim, spaces)
    (length(spaces) == 1 || entries <= MAX_TABLE_SIZE) && return nothing
    throw(ArgumentError("The constraint operator has the term `$term` acting on $(length(spaces)) spaces together, which needs a table of $entries values, more than the limit of $MAX_TABLE_SIZE. Use `ParityConservation` for a parity, and otherwise write the quantity as a sum of terms that each act on few spaces."))
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
const _ZN_HINT = "For a ℤ_N charge with values `ω^q`, constrain a sum `Q` of operators with the integer values `q` instead, as `Q => (v -> mod(v, N))`."
_check_exact(x, term) = _isexact(x) || throw(ArgumentError("A constraint operator needs integer or rational coefficients, such as `1//2` instead of `0.5`, so that its values are exact. Got the term `$term`."))

# The diagonal of a sum of weighted terms on the product of `spaces`, indexed by the positions of the states of `spaces`.
# Each term's diagonal is found without its coefficient, as the floating point values of a monomial are fractions with small
# denominators, and then multiplied by the exact coefficient.
function _diagonal_table(weighted_terms, spaces)
    H = length(spaces) == 1 ? only(spaces) : tensor_product(spaces)
    monomials = map(((_, t),) -> representation(NCMul(1, t.factors), H), weighted_terms)
    if !isdiag(sum(map(((w, _), m) -> complex(float(w)) * m, weighted_terms, monomials)))
        k = something(findfirst(!isdiag, monomials), 1)
        throw(ArgumentError("A constraint operator must be diagonal in the basis, but its term `$(weighted_terms[k][2])` is not."))
    end
    foreach(((w, t),) -> _check_exact(w, t), weighted_terms)
    contributions = map(((w, t), m) -> w .* map(v -> _exact_value(v, t), Vector(diag(m))), weighted_terms, monomials)
    k = findfirst(c -> !all(isreal, c), contributions)
    isnothing(k) || throw(ArgumentError("Constraint operators with complex values are not supported, but the term `$(weighted_terms[k][2])` has complex values. $_ZN_HINT"))
    values = Rational{Int}.(real.(sum(contributions)))
    length(spaces) == 1 && return values
    mapper = state_mapper(H, spaces)
    return vec([values[state_index(only(first(combine_states(states, mapper))), H)] for states in Iterators.product(map(basisstates, spaces)...)])
end
# The values of built-in operators have power-of-two denominators far below this bound, while an irrational value such as
# `√2` only rounds to a fraction with a larger denominator.
const MAX_DENOMINATOR = 2^16
function _exact_value(x::Real, term, hint = "")
    _isexact(x) && return Rational{Int}(x)
    tol = 4 * eps(max(1.0, abs(x)))
    r = rationalize(Int, x; tol)
    abs(r - x) <= tol && denominator(r) <= MAX_DENOMINATOR || throw(ArgumentError("The values of the constraint operator term `$term` are not exact fractions, such as $x.$hint"))
    return r
end
function _exact_value(z::Complex, term)
    hint = iszero(imag(z)) ? "" : " " * _ZN_HINT
    return complex(_exact_value(real(z), term, hint), _exact_value(imag(z), term, hint))
end

# The constraints on sums, which `generate_states` prunes by in one branch check.
const SumConstraint = Union{OperatorConstraint, AbstractAdditiveConstraint}
branch_constraint(c::SumConstraint, allspaces) = branch_constraint([c], allspaces)
# Sums with values that are not exact are pruned separately, so that the exact ones are not rounded. The spaces may
# be given as symbols, and a constraint without spaces is on all of them.
function branch_constraint(constraints::AbstractVector{<:SumConstraint}, spaces)
    allspaces = map(hilbert_space, spaces)
    constraints = map(c -> supply_missing_constraint_info(c, nothing, allspaces), constraints)
    sums = [_sum_terms(c, allspaces) for c in constraints if supports_branch_pruning(c)]
    isempty(sums) && return BranchConstraint((_, _, _) -> true)
    exact = map(s -> eltype(eltype(s.tables)) <: Union{Integer, Rational}, sums)
    allequal(exact) && return _sums_branch_constraint(sums, allspaces)
    return ProductConstraint([_sums_branch_constraint(sums[exact], allspaces), _sums_branch_constraint(sums[.!exact], allspaces)])
end
# Prunes on the integer `(value - first(r)) / step(r)`, since rational arithmetic is much slower. The allowed values
# off the lattice `r` are dropped, as no state has them.
function _sum_terms(c::OperatorConstraint, allspaces)
    units = _subspace_positions(c.spaces, allspaces)
    r = _value_range(c.terms)
    tables = Vector{Int}[Int[(v - minimum(t)) // step(r) for v in t] for t in c.terms.tables]
    allowed = Int[searchsortedfirst(r, a) - 1 for a in c.allowed if a in r]
    return (; constant = 0, positions = [units[u] for u in c.terms.units], c.terms.strides, tables, allowed, modulus = 0)
end
# Prunes by several sums at once. The groups of the terms are evaluated at the depth of their last space, and a step is
# the groups of one sum at one depth, so that each depth only evaluates the sums it changes.
function _sums_branch_constraint(sums, allspaces)
    T = mapreduce(s -> mapreduce(eltype, promote_type, s.tables; init = typeof(s.constant)), promote_type, sums; init = Int)
    owner = [k for (k, s) in enumerate(sums) for _ in s.tables]
    positions = Vector{Int}[p for s in sums for p in s.positions]
    depth = map(maximum, positions)
    tables = Vector{T}[t for s in sums for t in s.tables]
    strides = Vector{Int}[st for s in sums for st in s.strides]
    keys = collect(zip(depth, owner))
    order = sortperm(keys)
    keys = keys[order]
    steps = unique(keys)
    nsteps, step_depth, step_owner = length(steps), first.(steps), last.(steps)
    group_starts = push!(searchsortedfirst.(Ref(keys), steps), length(keys) + 1)
    step_groups = [order[group_starts[s]:(group_starts[s + 1] - 1)] for s in 1:nsteps]
    step_starts = [searchsortedfirst(step_depth, d) for d in 1:(length(allspaces) + 1)]
    lo = [sum(g -> minimum(tables[g]), gs) for gs in step_groups]
    hi = [sum(g -> maximum(tables[g]), gs) for gs in step_groups]
    # A step continues from the value after the previous step of its sum, or from its constant for the first, and the
    # later steps of its sum add between `remaining_min` and `remaining_max`.
    previous, remaining_min, remaining_max = zeros(Int, nsteps), zeros(T, nsteps), zeros(T, nsteps)
    latest, later_min, later_max = zeros(Int, length(sums)), zeros(T, length(sums)), zeros(T, length(sums))
    for s in 1:nsteps
        previous[s], latest[step_owner[s]] = latest[step_owner[s]], s
    end
    for s in nsteps:-1:1
        k = step_owner[s]
        remaining_min[s], remaining_max[s] = later_min[k], later_max[k]
        later_min[k] += lo[s]
        later_max[k] += hi[s]
    end
    constants = T[s.constant for s in sums]
    moduli = Int[s.modulus for s in sums]
    targets = [sort!(T[x for x in s.allowed if x isa Real && (!(T <: Integer) || isinteger(x))]) for s in sums]
    reachable(value, s) = _reachable(value + remaining_min[s], value + remaining_max[s], targets[step_owner[s]], moduli[step_owner[s]])
    sizes = map(dim, allspaces)
    cuts = _joint_cuts((; sizes, constants, moduli, tables, strides, positions, step_depth, step_owner, step_groups, reachable))
    values = zeros(T, nsteps)
    index = zeros(Int, length(allspaces))
    return BranchConstraint(
        (partial, d, _) -> begin
            index[d] = state_index(partial[d], allspaces[d])
            for s in step_starts[d]:(step_starts[d + 1] - 1)
                value = iszero(previous[s]) ? constants[step_owner[s]] : values[previous[s]]
                for g in step_groups[s]
                    value += tables[g][_table_index(strides[g], index, positions[g])]
                end
                values[s] = value
                reachable(value, s) || return false
            end
            all(cut -> insorted(_joint_key(cut, values, index), cut.allowed), cuts[d])
        end
    )
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
        (hilbert_space(S, 1:5), (sum(S[i][:z] for i in 1:5) => 1 // 2, S[1][:z] + S[2][:z] => 0)),
        (hilbert_space(S, 1:5), (sum(S[i][:z] for i in 1:5) => (1.0, 3 // 2),)),
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
    H = hilbert_space(f, 1:6)
    weighted = sum(i * n(i) for i in 1:6)
    expected = brute_force(H, (sum(n(i) for i in 1:6) => 2, weighted => 7))
    @test Set(FermionicHilbertSpaces.generate_states(H.modes, NumberConservation(2) * (weighted => 7), H)) == expected
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
    allup(s) = all(x -> x.m > 0, s.states)
    @test !(hilbert_space(S, 1:3, allup) isa SectorHilbertSpace) && dim(hilbert_space(S, 1:3, allup)) == 1
    @test quantumnumbers(hilbert_space(S, 1:3, allup => [false, true])) == [false, true]
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
end

@testitem "Labels are integers when every term of the operator takes integer values" begin
    @fermions f
    @spins S 1 // 2
    @bosons b
    labeltype(H, op) = eltype(quantumnumbers(constrain_space(H, op)))
    Sz(L) = sum(S[i][:z] for i in 1:L)
    @test labeltype(hilbert_space(S, 1:3), Sz(3)) == labeltype(hilbert_space(S, 1:4), Sz(4)) == Rational{Int}
    up(i) = S[i][:z] + 1 // 2
    @test labeltype(hilbert_space(S, 1:3), 2Sz(3)) == labeltype(hilbert_space(S, 1:3), sum(up(i) for i in 1:3)) == Int
    @test labeltype(hilbert_space(S, 1:4), sum(up(i) * up(i + 1) for i in 1:3)) == Int
    @test labeltype(hilbert_space(f, 1:3), sum(f[i]'f[i] for i in 1:3)) == Int
    @test labeltype(hilbert_space(b, 1:2, 5), sum(b[i]'b[i] for i in 1:2)) == Int
end

@testitem "Invalid constraint operators throw" begin
    @fermions f
    @spins S 1 // 2
    @test_throws "diagonal" hilbert_space(f, 1:2, f[1]'f[2] + f[2]'f[1] => 1)
    @test_throws "diagonal" hilbert_space(S, 1:2, S[1][:x] + S[2][:x] => 0)
    @test_throws "rational" hilbert_space(f, 1:2, 0.5 * f[1]'f[1] => 1)
    @test_throws "mod(v, N)" hilbert_space(f, 1:2, 1im * f[1]'f[1] => 1)
    @test_throws ArgumentError hilbert_space(f, 1:2, f[3]'f[3] => 1)
    @test_throws "1//10" hilbert_space(f, 1:2, (1 // 10) * f[1]'f[1] => 0.1)
    @test dim(hilbert_space(f, 1:2, f[1]'f[1] => 1.0)) == 2
    @test_throws "1//2" hilbert_space(f, 1:20, sum(i^4 * f[i]'f[i] for i in 1:20) => [0.5])
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
    for spec in (missing, nothing, true, Dict(1 => 1), 1 => 2, 0.5, v -> v == 2 || v)
        @test_throws ArgumentError hilbert_space(f, 1:4, N => spec)
    end
    @test_throws "boom" hilbert_space(f, 1:4, N => (v -> v == 3 ? error("boom") : v))
    @test dim(hilbert_space(f, 1:4, N => 7)) == 0
    @test isempty(quantumnumbers(hilbert_space(f, 1:4, N => [7, 8])))
    @test_throws "nothing" hilbert_space(f, 1:4, N => (v -> v > 2 ? nothing : v))
    cnt(s) = count_ones(s.f)
    @test_throws "Bool" hilbert_space(f, 1:4, cnt => (v -> v == 1 || v))
end

@testitem "Specifications on edge cases" begin
    @fermions f
    N = sum(f[i]'f[i] for i in 1:4)
    P = sum(i * f[i]'f[i] for i in 1:4)
    @test dim(hilbert_space(f, 1:4, N => 2 + 0im)) == dim(hilbert_space(f, 1:4, N => [2 + 0im])) == 6
    # A function that fails on a value no state has.
    @test quantumnumbers(hilbert_space(f, 1:4, (N => 2, P => (v -> v > 9 ? error("unreached") : mod(v, 2))))) == [0, 1]
    H = hilbert_space(f, 1:4)
    cnt(s) = count_ones(s.f)
    @test constrain_space(H, cnt => (v -> mod(v, 2)), empty(collect(basisstates(H)))) isa SectorHilbertSpace
    c = FilterConstraint(s -> true) * ((s -> Int(s.f)) => (v -> mod(v, 3)))
    spaces = fetch.([Threads.@spawn hilbert_space(f, 1:8, c) for _ in 1:4])
    @test allequal(map(Hc -> (quantumnumbers(Hc), collect(basisstates(Hc))), spaces))
end

@testitem "Vectors and products of constraints" begin
    @fermions f
    N = sum(f[i]'f[i] for i in 1:4)
    P = sum(i * f[i]'f[i] for i in 1:4)
    cnt(s) = count_ones(s.f)
    H = hilbert_space(f, 1:4)
    modes = [hilbert_space(f[i]) for i in 1:4]
    for c in ([N, P], [cnt, N => 2], [N => 2, ParityConservation()], FilterConstraint(s -> s.f != 3) * P)
        expected = quantumnumbers(constrain_space(H, Tuple(c isa AbstractVector ? c : c.constraints)))
        @test quantumnumbers(hilbert_space(f, 1:4, c)) == quantumnumbers(constrain_space(H, c)) == quantumnumbers(tensor_product(modes; constraint = c)) == expected
    end
end

@testitem "Constraint types keep their label rules" begin
    @fermions f
    Hnumber = hilbert_space(f, 1:4, NumberConservation())
    unlabelled = [
        NumberConservation(2) * FilterConstraint(s -> s.f != 3), ParityConservation() * FilterConstraint(s -> isodd(s.f)),
        FilterConstraint(s -> s.f > 2) * SectorConstraint(s -> isodd(s.f) ? 1.0 : 2),
    ]
    @test !any(c -> hilbert_space(f, 1:4, c) isa SectorHilbertSpace, unlabelled)
    @test !(constrain_space(hilbert_space(f, 1:4, NumberConservation(2)), FilterConstraint(s -> s.f != 3)) isa SectorHilbertSpace)
    H2 = constrain_space(Hnumber, NumberConservation(2))
    @test quantumnumbers(H2) == [2] && dim(sector(2, H2)) == 6
    @test quantumnumbers(constrain_space(Hnumber, ParityConservation())) == [-1, 1]
end

@testitem "Tensor products keep the labels of their factors" begin
    @fermions f
    N = sum(f[i]'f[i] for i in 1:4)
    factors = (hilbert_space(f, 1:2, f[1]'f[1] - f[2]'f[2]), hilbert_space(f[3]), hilbert_space(f[4]))
    for c in (N => isodd, N)
        @test quantumnumbers(tensor_product(factors; constraint = c)) == quantumnumbers(constrain_space(tensor_product(factors), c))
    end
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
    struct Root2 <: AbstractSym
        site::Int
    end
    @nc Root2
    apply_local_operator(::Root2, s::Clock, space, precomp) = s, sqrt(2)
    symbolic_group(op::Root2) = op.site
    @test_throws "not exact fractions" tensor_product(Hs; constraint = Root2(1) => 1)
end
