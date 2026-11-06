# Constraints written as `quantity => allowed`, where `quantity` is a diagonal symbolic operator or a function of the basis state.
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
as_constraint((quantity, allowed)::Pair) = quantity_constraint(quantity, allowed)
as_constraint(quantity) = quantity_constraint(quantity, missing)

# A scalar restricts without labelling sectors; a collection, or `missing` for all values, also labels them.
_allowed(value::Number) = [value], false
_allowed(values) = values, true
function quantity_constraint(f::Function, allowed)
    values, labelled = _allowed(allowed)
    labelled || return FilterConstraint(state -> f(state) in values, missing, missing)
    ismissing(values) && return SectorConstraint(FilterConstraint(f, missing, missing))
    return SectorConstraint(FilterConstraint(state -> (value = f(state); value in values ? value : missing), missing, missing))
end
quantity_constraint(op, allowed) = OperatorConstraint(op, _allowed(_exact_allowed(allowed))..., missing, nothing)
# The values of an operator are exact, so a float could only match them by accident.
_exact_allowed(x::AbstractFloat) = isinteger(x) ? Int(x) : throw(ArgumentError("The allowed values of a constraint operator must be exact, such as `1//10` instead of `0.1`, got $x."))
_exact_allowed(x::Union{Number, Missing}) = x
_exact_allowed(xs) = any(x -> x isa AbstractFloat, xs) ? [_exact_allowed(x) for x in xs] : xs

constrain_space(space, @nospecialize(c::QuantitySpec), args...) = constrain_space(space, as_constraint(c), args...)
constrain_space(space::AbstractHilbertSpace, @nospecialize(cs::ConstraintVector)) = constrain_space(space, as_constraint(cs))
Base.:*(c::AbstractConstraint, p::Pair) = c * as_constraint(p)
Base.:*(p::Pair, c::AbstractConstraint) = as_constraint(p) * c

# The constraint that the diagonal operator `op` has a value in `allowed`. `terms` holds its values on `spaces`.
struct OperatorConstraint{O, A, H, D} <: AbstractAdditiveConstraint
    op::O
    allowed::A
    labelled::Bool
    spaces::H
    terms::D
end
allowed_values(c::OperatorConstraint) = c.allowed
function with_spaces(c::OperatorConstraint, spaces)
    units = collect(spaces)
    return OperatorConstraint(c.op, c.allowed, c.labelled, units, diagonal_terms(c.op, units))
end
# The terms index the states of each space by position, which tagging and transposing keep.
map_spaces(f, c::OperatorConstraint) = OperatorConstraint(c.op, c.allowed, c.labelled, _map_spaces(f, c.spaces), c.terms)
supports_sector_grouping(c::OperatorConstraint) = c.labelled
# Without spaces, the operator is split over the atoms of the constrained space, so any split fits.
branch_spaces(::OperatorConstraint{<:Any, <:Any, Missing}) = ()
sector_function(c::OperatorConstraint{<:Any, <:Any, Missing}, space::AbstractHilbertSpace) = sector_function(with_spaces(c, atomic_factors(space)), space)
function Base.show(io::IO, c::OperatorConstraint)
    print(io, c.op)
    return ismissing(c.allowed) || print(io, " => ", c.labelled ? c.allowed : only(c.allowed))
end

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
# Values are summed as integers when each table's entries differ by integers, with the common fractions moved to the constant.
function _diagonal_terms(constant, units, strides, tables)
    shifts = map(t -> t[1] - floor(t[1]), tables)
    total = constant + sum(shifts; init = zero(constant))
    integral = isinteger(total) && all(((t, s),) -> all(x -> isinteger(x - s), t), zip(tables, shifts))
    integral || return DiagonalTerms{Rational{Int}}(constant, units, strides, tables)
    return DiagonalTerms{Int}(Int(total), units, strides, map((t, s) -> Int.(t .- s), tables, shifts))
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
    terms = c.terms
    unit_positions = _subspace_positions(c.spaces, allspaces)
    positions = map(units -> unit_positions[units], terms.units)
    n = length(allspaces)
    done = map(maximum, positions)
    order = sortperm(done)
    starts = searchsortedfirst.(Ref(done[order]), 1:(n + 1))
    T = typeof(terms.constant)
    remaining_min = zeros(T, n + 1)
    remaining_max = zeros(T, n + 1)
    for (d, table) in zip(done, terms.tables)
        remaining_min[d] += minimum(table)
        remaining_max[d] += maximum(table)
    end
    for i in n:-1:1
        remaining_min[i] += remaining_min[i + 1]
        remaining_max[i] += remaining_max[i + 1]
    end
    targets = c.allowed
    index = zeros(Int, n)
    # As for the additive constraints, `prefix[d]` is the value of the groups completed before depth `d` on the current branch.
    prefix = fill(terms.constant, n + 1)
    return BranchConstraint(
        (partial, depth, _) -> begin
            index[depth] = state_index(partial[depth], allspaces[depth])
            current = prefix[depth]
            for i in starts[depth]:(starts[depth + 1] - 1)
                g = order[i]
                current += terms.tables[g][_table_index(terms.strides[g], index, positions[g])]
            end
            prefix[depth + 1] = current
            _reachable(current + remaining_min[depth + 1], current + remaining_max[depth + 1], targets, nothing)
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
    up(i) = S[i][:z] + 1 // 2
    Hmixed = (hilbert_space(a, 4), hilbert_space(f, 1:2), hilbert_space(S[1]))
    cases = [
        (hilbert_space(f, 1:5), (sum(n(i) for i in 1:5) => 2,)),
        (hilbert_space(f, hubbard), (Nup => 2, Ndn => 2, D => 0)),
        (hilbert_space(S, 1:6), (sum(S[i][:z] for i in 1:6) => 0,)),
        (hilbert_space(f, 1:6), (sum(n(i) for i in 1:6) => 3, sum(i * n(i) for i in 1:6) => 10)),
        (hilbert_space(S, 1:6), (sum(up(i) * up(i + 1) for i in 1:5) => 0,)),
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
    @test eltype(quantumnumbers(hilbert_space(S, 1:4, sum(S[i][:z] for i in 1:4)))) == Int
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
