# Constraints written as `quantity => allowed`, where `quantity` is a diagonal symbolic operator or a function of the basis state.
const QuantitySpec = Union{Pair, Tuple, NCAdd, NCMul, AbstractSym, Function}
const ConstraintLike = Union{AbstractConstraint, QuantitySpec}

as_constraint(c::AbstractConstraint) = c
as_constraint(cs::Tuple) = prod(as_constraint, cs)
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
quantity_constraint(op, allowed) = OperatorConstraint(op, _allowed(allowed)..., missing, nothing)

constrain_space(space, c::QuantitySpec, args...) = constrain_space(space, as_constraint(c), args...)

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
    foreach(t -> _check_exact(t.coeff, op), terms)
    _check_exact(constant, op)
    unit_of = Dict(id => k for (k, H) in enumerate(spaces) for id in atom_ids(H))
    supports = map(terms) do term
        sort!(unique(map(x -> _unit_of(x, unit_of, spaces, op), term.factors)))
    end
    distinct = unique(supports)
    groups = filter(s -> !any(t -> length(t) > length(s) && issubset(s, t), distinct), distinct)
    ops = Any[0 for _ in groups]
    for (term, support) in zip(terms, supports)
        owners = findall(g -> issubset(support, g), groups)
        for g in owners
            ops[g] += term * (1 // length(owners))
        end
    end
    tables = Vector{Rational{Int}}[_diagonal_table(opg, spaces[g], op) for (g, opg) in zip(groups, ops)]
    strides = [cumprod([1; map(dim, spaces[g][1:(end - 1)])]) for g in groups]
    return _diagonal_terms(_exact_value(constant, op), groups, strides, tables)
end
# Values are summed as integers when each table's entries differ by integers, with the common fractions moved to the constant.
function _diagonal_terms(constant, units, strides, tables)
    shifts = map(t -> t[1] - floor(t[1]), tables)
    total = constant + sum(shifts; init = zero(constant))
    integral = isinteger(total) && all(((t, s),) -> all(x -> isinteger(x - s), t), zip(tables, shifts))
    integral || return DiagonalTerms{Rational{Int}}(constant, units, strides, tables)
    return DiagonalTerms{Int}(Int(total), units, strides, map((t, s) -> Int.(t .- s), tables, shifts))
end
_operator_terms(op::NCAdd) = collect(NCterms(op)), op.coeff
_operator_terms(op::NCMul) = [op], 0
_operator_terms(op::AbstractSym) = _operator_terms(1 * op)
_operator_terms(op) = throw(ArgumentError("A constraint quantity must be a symbolic operator or a function of the basis state, got `$op`."))

function _unit_of(x, unit_of, spaces, op)
    k = get(unit_of, atomic_id(x), 0)
    iszero(k) || return k
    k = something(findfirst(H -> _acts_in(x, H), spaces), 0)
    iszero(k) && throw(ArgumentError("The constraint operator `$op` acts on `$x`, which is not in the space."))
    return k
end
# A symbol that names no atom, such as a single Majorana, acts in the space that contains it.
_acts_in(x, H) = false

_isexact(x) = x isa Union{Integer, Rational}
_isexact(x::Complex) = _isexact(real(x)) && _isexact(imag(x))
_check_exact(x, op) = _isexact(x) || throw(ArgumentError("A constraint operator needs integer or rational coefficients, such as `1//2` instead of `0.5`, so that its values are exact. Got the coefficient $x in `$op`."))

# The diagonal of `op` on the product of `spaces`, indexed by the positions of the states of `spaces`.
function _diagonal_table(op, spaces, whole)
    H = length(spaces) == 1 ? only(spaces) : tensor_product(spaces)
    m = representation(op, H)
    isdiag(m) || throw(ArgumentError("A constraint operator must be diagonal in the basis, but `$whole` is not."))
    values = map(v -> _exact_value(v, whole), Vector(diag(m)))
    length(spaces) == 1 && return values
    mapper = state_mapper(H, spaces)
    return vec([values[state_index(only(first(combine_states(states, mapper))), H)] for states in Iterators.product(map(basisstates, spaces)...)])
end
# Spin and boson matrix elements are floating point, so the values of an operator with exact coefficients are rounded to the nearby fraction.
function _exact_value(v, op)
    tol = 1.0e-10 * max(1, abs(v))
    abs(imag(v)) <= tol || throw(ArgumentError("A constraint operator must be Hermitian, but `$op` has the value $v."))
    x = real(v)
    _isexact(x) && return Rational{Int}(x)
    return rationalize(Int, x; tol)
end

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
    @spins S 1 // 2
    @test quantumnumbers(hilbert_space(S, 1:3, sum(S[i][:z] for i in 1:3))) == (-3 // 2):(3 // 2)
    nup(s) = count(x -> x.m > 0, s.states)
    @test quantumnumbers(hilbert_space(S, 1:3, nup)) == 0:3
    @test dim(hilbert_space(S, 1:4, nup => 2)) == 6
end

@testitem "Invalid constraint operators throw" begin
    @fermions f
    @test_throws ArgumentError hilbert_space(f, 1:2, f[1]'f[2] + f[2]'f[1] => 1)
    @test_throws ArgumentError hilbert_space(f, 1:2, 0.5 * f[1]'f[1] => 1)
    @test_throws ArgumentError hilbert_space(f, 1:2, f[3]'f[3] => 1)
end
