# This is largely LLM generated.
export @parafermions

# =====================================================================
# Conventions
#
#   ω = exp(2πim / P)
#   degree(c†) = +1, degree(c) = -1
#
#   c_j |n> =
#       ω^(-σ * sum(n_i, i<j)) |n - e_j>,   n_j > 0
#
#   c_j† |n> =
#       ω^(+σ * sum(n_i, i<j)) |n + e_j>,   n_j < P-1
#
# For homogeneous operators on distinct modes i<j:
#
#   A_i B_j = ω^(-σ * degree(A_i)*degree(B_j)) B_j A_i.
#
# The integer encoding is little-endian, like FockNumber:
#
#   f = sum(n_i * P^(i-1), i=1:N).
#
# All phase exponents are accumulated modulo P.
# =====================================================================

function _pf_check_parameters(P, S)
    P isa Int && P >= 2 ||
        throw(ArgumentError("The parafermion order must be an Int ≥ 2"))
    S isa Int && (S == 1 || S == -1) ||
        throw(ArgumentError("sigma must be +1 or -1"))
    nothing
end

"""
Return ω^exponent, reducing the exponent first.

Special real roots are returned exactly, in ComplexF64 form.
General roots are numerical and should be compared with ≈.
"""
function _pf_root(::Val{P}, exponent::Integer) where P
    k = Int(mod(exponent, P))
    k == 0 && return ComplexF64(1)
    iseven(P) && k == P ÷ 2 && return ComplexF64(-1)
    P % 4 == 0 && k == P ÷ 4 && return ComplexF64(0, 1)
    P % 4 == 0 && k == 3 * (P ÷ 4) && return ComplexF64(0, -1)
    cispi(2k / P)
end

_pf_addmod(e, x, P) = Int(mod(e + x, P))

function default_parafock_representation(P::Int, N::Int)
    _pf_check_parameters(P, 1)
    N >= 0 || throw(ArgumentError("Negative number of modes"))
    big(P)^N - 1 <= typemax(UInt64) ? UInt64 : BigInt
end

# Convert, failing with a clear message instead of wrapping or InexactError.
function _pf_convert_checked(::Type{I}, x::Integer, what) where {I<:Integer}
    I === BigInt || x <= typemax(I) ||
        throw(ArgumentError("$what does not fit in the storage type $I"))
    convert(I, x)
end

# First n base-P digits of f (little-endian). Digits beyond the stored
# value are zero. No powers are formed, so nothing can overflow.
function _pf_decode(f::Integer, ::Val{P}, n::Integer) where P
    n >= 0 || throw(ArgumentError("Negative number of modes"))
    ds = zeros(Int, n)
    x = f
    i = 1
    while i <= n && !iszero(x)
        x, r = divrem(x, P)
        ds[i] = Int(r)
        i += 1
    end
    ds
end


# =====================================================================
# Fock states
# =====================================================================

struct ParaFockNumber{P,S,I<:Integer} <: AbstractFockState
    f::I

    function ParaFockNumber{P,S,I}(f::Integer) where {P,S,I<:Integer}
        _pf_check_parameters(P, S)
        f >= 0 || throw(ArgumentError("A Fock-state encoding must be nonnegative"))
        new{P,S,I}(I(f))
    end
end

# Storage is chosen from the argument's *type*, not its value, so the result
# type can be inferred. Every nonnegative integer of 64 bits or fewer fits
# in UInt64; wider types (Int128, UInt128, BigInt) get BigInt.
_pf_default_storage(::Type{<:Union{Bool,Int8,Int16,Int32,Int64,
    UInt8,UInt16,UInt32,UInt64}}) = UInt64
_pf_default_storage(::Type{<:Integer}) = BigInt

ParaFockNumber{P,S}(f::Integer) where {P,S} =
    ParaFockNumber{P,S,_pf_default_storage(typeof(f))}(f)

_pf_storagetype(::Type{<:ParaFockNumber{P,S,I}}) where {P,S,I} = I

ParaFockNumber{P}(f::Integer) where P = ParaFockNumber{P,1}(f)

ParaFockNumber{P,S,I}(f::ParaFockNumber{P,S}) where {P,S,I} =
    ParaFockNumber{P,S,I}(f.f)

Base.convert(::Type{ParaFockNumber{P,S,I}},
    f::ParaFockNumber{P,S}) where {P,S,I} =
    ParaFockNumber{P,S,I}(f.f)

parafermion_order(::ParaFockNumber{P}) where P = P
parafermion_sigma(::ParaFockNumber{P,S}) where {P,S} = S

Base.:(==)(a::ParaFockNumber{P,S},
    b::ParaFockNumber{Q,T}) where {P,S,Q,T} =
    P == Q && S == T && a.f == b.f

Base.hash(a::ParaFockNumber{P,S}, h::UInt) where {P,S} =
    hash((P, S, a.f), h)

Base.isless(a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S}) where {P,S} = isless(a.f, b.f)

Base.iszero(a::ParaFockNumber) = iszero(a.f)

Base.zero(::Type{ParaFockNumber{P,S,I}}) where {P,S,I} =
    ParaFockNumber{P,S,I}(0)

Base.zero(a::ParaFockNumber) = zero(typeof(a))

function Base.show(io::IO, a::ParaFockNumber{P,S}) where {P,S}
    print(io, "ParaFockNumber{", P, ",", S, "}(", a.f, ")")
end

function occupation(a::ParaFockNumber{P}, site::Integer) where P
    site >= 1 || throw(ArgumentError("Mode positions are one-based"))
    # Divide step by step instead of forming P^(site-1). That power can
    # overflow the storage type for sites beyond the stored digits.
    # Hot paths don't call this; they use cached place values.
    x = a.f
    for _ in 2:site
        iszero(x) && return 0
        x = div(x, P)
    end
    Int(rem(x, P))
end

function occupations(a::ParaFockNumber{P}, N::Integer) where P
    N >= 0 || throw(ArgumentError("Negative number of modes"))
    _pf_decode(a.f, Val(P), N)
end

# Shared code reads `bits` as 0/1 occupations and `parity` as the fermionic
# sign. Those meanings match the parafermionic ones only for P = 2, so both
# are defined only there. For P > 2 they throw instead of silently meaning
# something else.
bits(a::ParaFockNumber{2}, N) = occupations(a, N)
bits(::ParaFockNumber{P}, N) where P = throw(ArgumentError("bits is only defined for P = 2; use occupations(state, N) for base-$P digits"))

function parafock_from_digits(
    ns,
    ::Val{P},
    ::Val{S}=Val(1),
    ::Type{I}=default_parafock_representation(P, length(ns)),
) where {P,S,I<:Integer}
    _pf_check_parameters(P, S)
    value = big(0)
    place = big(1)
    for n in ns
        n isa Integer && 0 <= n < P ||
            throw(ArgumentError("Every occupation must lie in 0:$(P-1)"))
        value += n * place
        place *= P
    end
    ParaFockNumber{P,S,I}(value)
end
# Always Int: the digit sum is at most (P-1)·N.
function particle_number(a::ParaFockNumber{P}) where P
    x = a.f
    total = 0
    while !iszero(x)
        x, r = divrem(x, P)
        total += Int(r)
    end
    total
end

parafermion_charge(a::ParaFockNumber{P}) where P = mod(particle_number(a), P)

parity(a::ParaFockNumber{2}) = iseven(particle_number(a)) ? 1 : -1
parity(::ParaFockNumber{P}) where P = throw(ArgumentError(
    "parity is only defined for P = 2; use parafermion_charge(state) for the Z_$P charge"))

internal_rep(a::ParaFockNumber, ::AbstractHilbertSpace,
    ::Type{T}) where {T<:Integer} = T(a.f)

physical_rep(a::Integer, ::Type{ParaFockNumber{P,S,I}}) where {P,S,I} =
    ParaFockNumber{P,S,I}(a)

"""
Z_P charge of the modes strictly before the mode whose base-P place value
is `place`: Σ_{i<site} n_i mod P. `rem(f, place)` keeps exactly those
digits, so no powers are formed and no digit is decoded twice.
"""
function _pf_left_charge(f::Integer, place::Integer, ::Val{P}) where P
    x = rem(f, place)
    e = 0
    while !iszero(x)
        x, r = divrem(x, P)
        e += Int(r)
    end
    mod(e, P)
end

# P = 2: the place value is a power of two, so the left digits form a bit mask.
_pf_left_charge(f::Unsigned, place::Unsigned, ::Val{2}) =
    count_ones(f & (place - one(place))) & 1

# =====================================================================
# Groups and symbolic operators
# =====================================================================

struct ParafermionicGroup{P,S,T}
    id::T

    function ParafermionicGroup{P,S}(id::T) where {P,S,T}
        _pf_check_parameters(P, S)
        new{P,S,T}(id)
    end
end

parafermion_order(::ParafermionicGroup{P}) where P = P
parafermion_sigma(::ParafermionicGroup{P,S}) where {P,S} = S

Base.:(==)(a::ParafermionicGroup{P,S},
    b::ParafermionicGroup{Q,T}) where {P,S,Q,T} =
    P == Q && S == T && a.id == b.id

Base.hash(g::ParafermionicGroup{P,S}, h::UInt) where {P,S} =
    hash((P, S, g.id), h)

Base.isless(a::ParafermionicGroup{P,S},
    b::ParafermionicGroup{Q,T}) where {P,S,Q,T} =
    isless((P, S, a.id), (Q, T, b.id))

symbolic_group(g::ParafermionicGroup) = g
tags(g::ParafermionicGroup) = g.id

add_tag(g::ParafermionicGroup{P,S}, tag) where {P,S} =
    ParafermionicGroup{P,S}(add_tag(g.id, tag))

struct SymbolicParafermionBasis{G<:ParafermionicGroup}
    name::Symbol
    tags::G
end

Base.:(==)(a::SymbolicParafermionBasis, b::SymbolicParafermionBasis) =
    a.name == b.name && a.tags == b.tags

Base.hash(a::SymbolicParafermionBasis, h::UInt) =
    hash(a.name, hash(a.tags, h))

symbolic_group(a::SymbolicParafermionBasis) = a.tags
tags(a::SymbolicParafermionBasis) = a.tags

add_tag(a::SymbolicParafermionBasis, tag) =
    SymbolicParafermionBasis(a.name, add_tag(a.tags, tag))

parafermion_order(a::SymbolicParafermionBasis) =
    parafermion_order(symbolic_group(a))

parafermion_sigma(a::SymbolicParafermionBasis) =
    parafermion_sigma(symbolic_group(a))

"""
    parafermion_basis(name, p; sigma=1, group=nothing)

Create a Fock-parafermion species.

Pass the same `group` to several species to make them braid with one
another. Different groups commute.
"""
function parafermion_basis(name::Symbol, P::Int; sigma::Int=1, group=nothing)
    _pf_check_parameters(P, sigma)
    g = if group === nothing
        ParafermionicGroup{P,sigma}(Tags((gensym(:parafermions),)))
    else
        group isa ParafermionicGroup ||
            throw(ArgumentError("Expected a ParafermionicGroup"))
        parafermion_order(group) == P &&
            parafermion_sigma(group) == sigma ||
            throw(ArgumentError("Group order or sigma does not match"))
        group
    end
    SymbolicParafermionBasis(name, g)
end

"""
    @parafermions p a b ...

Create species sharing one braiding group, with sigma=+1.
Separate invocations create commuting groups.
"""
macro parafermions(p, xs...)
    isempty(xs) && error("Provide at least one species name")
    all(x -> x isa Symbol, xs) || error("Species names must be symbols")
    pp = gensym(:p)
    gg = gensym(:group)
    defs = [
        :($(esc(x)) = SymbolicParafermionBasis($(QuoteNode(x)), $gg))
        for x in xs
    ]
    quote
        local $pp = $(esc(p))
        _pf_check_parameters($pp, 1)
        local $gg =
            ParafermionicGroup{$pp,1}(Tags((gensym(:parafermions),)))
        $(defs...)
        tuple($(map(esc, xs)...))
    end
end

# Do not subtype AbstractFermionSym: its generic methods may assume CAR
# or integer-valued matrix elements.
struct ParafermionSym{L,B} <: AbstractSym
    creation::Bool
    label::L
    basis::B
end

Base.getindex(a::SymbolicParafermionBasis, i) =
    ParafermionSym(false, i, a)

Base.getindex(a::SymbolicParafermionBasis, is...) =
    ParafermionSym(false, is, a)

Base.adjoint(a::ParafermionSym) =
    ParafermionSym(!a.creation, a.label, a.basis)

Base.iszero(::ParafermionSym) = false

Base.:(==)(a::ParafermionSym, b::ParafermionSym) =
    a.creation == b.creation && a.label == b.label && a.basis == b.basis

Base.hash(a::ParafermionSym, h::UInt) =
    hash(a.creation, hash(a.label, hash(a.basis, h)))

symbolic_group(a::ParafermionSym) = symbolic_group(a.basis)
symbolic_basis(a::ParafermionSym) = a.basis
group_id(a::ParafermionSym) = symbolic_group(a)
atomic_id(a::ParafermionSym) = (a.basis, a.label)
label(a::ParafermionSym) = a.label
change_basis(a::ParafermionSym, basis) =
    ParafermionSym(a.creation, a.label, basis)

add_tag(a::ParafermionSym, tag) =
    change_basis(a, add_tag(a.basis, tag))

parafermion_order(a::ParafermionSym) = parafermion_order(a.basis)
parafermion_sigma(a::ParafermionSym) = parafermion_sigma(a.basis)

_normalize_sym(a::ParafermionSym) =
    ParafermionSym(false, a.label, a.basis)

_pf_degree(a::ParafermionSym) = a.creation ? 1 : -1
_pf_modekey(a::ParafermionSym) = (a.basis.name, a.label)

function Base.isless(a::ParafermionSym, b::ParafermionSym)
    ga, gb = group_id(a), group_id(b)
    ga == gb || return isless(ga, gb)
    ka, kb = _pf_modekey(a), _pf_modekey(b)
    ka == kb || return isless(ka, kb)
    a.creation && !b.creation
end

function Base.show(io::IO, a::ParafermionSym)
    print(io,
        _symbolic_name_with_tags(a.basis.name, a.basis; skip_first=1),
        a.creation ? "†" : "",
        "[", a.label, "]")
end

mat_eltype(::Type{<:ParafermionSym}) = ComplexF64

function NonCommutativeProducts.mul_effect(
    a::ParafermionSym, b::ParafermionSym
)
    ga, gb = group_id(a), group_id(b)

    # Different groups commute.
    if ga != gb
        return isless(b, a) ? Swap(1) : nothing
    end

    same_mode = atomic_id(a) == atomic_id(b)
    P = parafermion_order(a)

    if same_mode
        if P == 2
            a.creation == b.creation && return 0
            !a.creation && b.creation &&
                return AddTerms((Swap(-1), 1))
        end

        # For P>2, keep local words explicit. In particular, c*c is
        # NOT zero unless P=2. Numerical application implements all
        # finite-occupation identities without a symbolic rewrite.
        return nothing
    end

    isless(b, a) || return nothing

    # Here a is the later mode and b is the earlier mode.
    exponent = parafermion_sigma(a) * _pf_degree(a) * _pf_degree(b)
    Swap(_pf_root(Val(P), exponent))
end

@nc ParafermionSym

"""
The occupation operator, not merely c†c.

For these unit-amplitude ladders:
    N = sum((c†)^m * c^m, m=1:P-1).
"""
function parafermion_number(a::ParafermionSym)
    c = _normalize_sym(a)
    sum((c')^m * c^m for m in 1:(parafermion_order(c)-1))
end

# =====================================================================
# State mapper
# =====================================================================
# Ordered cross-block inversions: (i, j) with i ∈ X_s, j ∈ X_r, s < r, i > j.
# These are exactly the pairs that enter the l and u phases.
function _pf_inversions(blocks)
    pairs = Tuple{Int,Int}[]
    for s in eachindex(blocks), r in (s+1):lastindex(blocks)
        for i in blocks[s], j in blocks[r]
            i > j && push!(pairs, (Int(i), Int(j)))
        end
    end
    pairs
end

struct ParaFockMapper{P,S,N,F,T,W,I,PL,SM} <: AbstractStateMapper
    # Same field names as FockMapper: generic code reads `fermionpositions`
    # and `widths`.
    fermionpositions::T
    widths::W
    isfullpartition::Bool
    # Precomputed once, so split/combine and the phase hooks never form
    # powers or build BigInts per state.
    places::PL                          # NTuple{N,I}: places[i] = P^(i-1)
    maxstate::I                         # P^N - 1
    submax::SM                          # per block: P^width - 1
    sortedsites::Vector{Int}            # sorted union of all blocks
    inversions::Vector{Tuple{Int,Int}}  # see _pf_inversions
end

function ParaFockMapper(
    positions,
    ::Type{F},
    N::Int,
) where {P,S,I,F<:ParaFockNumber{P,S,I}}
    N >= 0 || throw(ArgumentError("Negative number of modes"))
    pos = Tuple(Tuple(Int(i) for i in X) for X in positions)

    for X in pos
        issorted(X) && allunique(X) ||
            throw(ArgumentError(
                "Each subregion must preserve the full-system mode order"))
        all(i -> 1 <= i <= N, X) ||
            throw(ArgumentError("A mode position lies outside the system"))
    end

    flat = collect(Int, Iterators.flatten(pos))
    full = length(flat) == N && sort(flat) == collect(1:N)
    widths = map(length, pos)

    # P^N - 1 must fit in I; then every place value P^(i-1) ≤ P^N - 1 fits too.
    maxstate = _pf_convert_checked(I, big(P)^N - 1, "The largest state of $N modes")
    places = ntuple(i -> I(big(P)^(i - 1)), N)
    submax = map(w -> I(big(P)^w - 1), widths)

    ParaFockMapper{P,S,N,F,typeof(pos),typeof(widths),I,typeof(places),typeof(submax)}(
        pos, widths, full, places, maxstate, submax,
        sort!(unique(flat)), _pf_inversions(pos))
end

function ParaFockMapper(
    positions, ::Val{P}, ::Val{S}=Val(1);
    N=maximum(Iterators.flatten(positions); init=0),
) where {P,S}
    I = default_parafock_representation(P, Int(N))
    ParaFockMapper(positions, ParaFockNumber{P,S,I}, Int(N))
end

unique_split(::ParaFockMapper) = true
unique_combine(::ParaFockMapper) = true

function Base.show(io::IO, fm::ParaFockMapper{P,S,N}) where {P,S,N}
    print(io, "ParaFockMapper{", P, ",", S, "}(",
        N, " modes → ", fm.fermionpositions, ")")
end

function split_state(
    state::ParaFockNumber{P,S},
    fm::ParaFockMapper{P,S,N,F},
) where {P,S,N,I,F<:ParaFockNumber{P,S,I}}
    state.f <= fm.maxstate ||
        throw(ArgumentError("State lies outside the mapper's full space"))

    f = convert(I, state.f)  # fits: checked above
    places = fm.places
    substates = map(fm.fermionpositions) do X
        value = zero(I)
        for (k, i) in enumerate(X)
            value += rem(div(f, places[i]), P) * places[k]
        end
        F(value)
    end

    (substates,), (1,)
end

function combine_states(
    states,
    fm::ParaFockMapper{P,S,N,F},
) where {P,S,N,I,F<:ParaFockNumber{P,S,I}}
    fm.isfullpartition ||
        throw(ArgumentError(
            "combine_states requires a full, non-overlapping partition"))
    length(states) == length(fm.fermionpositions) ||
        throw(DimensionMismatch("Wrong number of subsystem states"))

    value = zero(I)
    for (state, X, submax) in zip(states, fm.fermionpositions, fm.submax)
        state isa ParaFockNumber{P,S} ||
            throw(ArgumentError("Incompatible parafermionic state"))
        state.f <= submax ||
            throw(ArgumentError("Subsystem state has too many digits"))

        # Peel off local digits in order; local digit k goes to global mode X[k].
        x = state.f
        for i in X
            x, r = divrem(x, P)
            value += convert(I, r) * fm.places[i]
        end
    end

    (F(value),), (1,)
end


# =====================================================================
# Phase factors
#
# f(a,b; X) = ω^[σ Σ_{i<j, i,j∈X} b_i(a_j-b_j)]
#
# h(a,b; partition) = f_Y(a,b) / Π_X f_X(a,b)
#
# l(a,b; ordered partition)
#   = ω^[σ Σ_{s<r} Σ_{i∈X_s,j∈X_r,i>j} d_i d_j]
#
# u(n; ordered partition)
#   = ω^[σ Σ_{s<r} Σ_{i∈X_s,j∈X_r,i>j} n_i n_j]
#
# l converts the transported product into the ordered product
# of embeddings. u is the occupation-state reordering phase.
# =====================================================================

# ---------------------------------------------------------------------
# Every hook decodes each state ONCE into Int digits (a Vector on the
# generic paths, an NTuple{N,Int} in mapper closures) and then runs plain
# integer loops. Kernels return raw exponents; _pf_root reduces mod P
# once. Raw sums are bounded by ~N²P², so Int overflow would need
# N·P ≳ 3×10⁹.
# ---------------------------------------------------------------------

# All N digits of a state, from the mapper's cached place values. No allocation.
@inline function _pf_decode(a::ParaFockNumber{P,S}, fm::ParaFockMapper{P,S}) where {P,S}
    a.f <= fm.maxstate ||
        throw(ArgumentError("State lies outside the mapper's full space"))
    f = a.f
    map(place -> Int(rem(div(f, place), P)), fm.places)
end

# Σ_{i<j; i,j ∈ X} b_i (a_j - b_j), for X iterated in ascending order.
@inline function _pf_f_kernel(da, db, X)
    e = 0
    prefix = 0
    for j in X
        e += prefix * (da[j] - db[j])
        prefix += db[j]
    end
    e
end

# f_Y - Σ_X f_X, with Y = union of the blocks (`sites`, ascending) and each
# block ascending. Tuples of blocks are mapped (unrolled), so blocks of
# different lengths stay type-stable. Cost O(N), with no pair loop.
function _pf_h_kernel(da, db, sites, blocks)
    _pf_f_kernel(da, db, sites) -
        sum(map(X -> _pf_f_kernel(da, db, X), blocks); init=0)
end

# Σ_{(i,j) ∈ pairs} w_i w_j
@inline function _pf_pair_kernel(w, pairs)
    e = 0
    for (i, j) in pairs
        e += w[i] * w[j]
    end
    e
end

# Normalize an arbitrary partition (tuples, vectors, ranges) to Vector{Vector{Int}}.
function _pf_blocks(partition)
    blocks = [collect(Int, X) for X in partition]
    all(i -> i >= 1, Iterators.flatten(blocks)) ||
        throw(ArgumentError("Mode positions are one-based"))
    blocks
end

_pf_maxsite(blocks) = maximum(Iterators.flatten(blocks); init=0)

# Raw (unreduced) σ-weighted f exponent over an arbitrary index set.
function _pf_f_exponent(
    a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S},
    inds,
) where {P,S}
    X = sort!(unique!(collect(Int, inds)))
    isempty(X) && return 0
    first(X) >= 1 || throw(ArgumentError("Mode positions are one-based"))
    da = _pf_decode(a.f, Val(P), last(X))
    db = _pf_decode(b.f, Val(P), last(X))
    S * _pf_f_kernel(da, db, X)
end

# Use explicit tuple and Int signatures to avoid ambiguity with the
# existing untyped fermionic fallback methods.
phase_factor_f(a::ParaFockNumber{P,S}, b::ParaFockNumber{P,S},
    inds::NTuple) where {P,S} =
    _pf_root(Val(P), _pf_f_exponent(a, b, inds))

phase_factor_f(a::ParaFockNumber{P,S}, b::ParaFockNumber{P,S},
    inds::AbstractVector{<:Integer}) where {P,S} =
    _pf_root(Val(P), _pf_f_exponent(a, b, inds))

# Modes 1:N: a single pass over the digits of a and b together, with no
# allocation for fixed-width storage. Used by the partial-trace hook.
function phase_factor_f(
    a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S},
    N::Int,
) where {P,S}
    N >= 0 || throw(ArgumentError("Negative number of modes"))
    x, y = a.f, b.f
    e, prefix = 0, 0
    for _ in 1:N
        iszero(x) && iszero(y) && break  # remaining digits contribute nothing
        x, ra = divrem(x, P)
        y, rb = divrem(y, P)
        e += prefix * (Int(ra) - Int(rb))
        prefix += Int(rb)
    end
    _pf_root(Val(P), S * e)
end

inverse_phase_factor_f(a::ParaFockNumber, b::ParaFockNumber, inds) =
    conj(phase_factor_f(a, b, inds))

# Generic entry point (arbitrary partition). Mapper closures use _pf_h_phase.
function phase_factor_h(
    a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S},
    partition,
    masks=nothing,
) where {P,S}
    # `masks` is accepted only for interface compatibility.
    blocks = map(sort!, _pf_blocks(partition))
    sites = sort!(reduce(vcat, blocks; init=Int[]))
    allunique(sites) || throw(ArgumentError("h requires disjoint blocks"))
    n = isempty(sites) ? 0 : last(sites)
    da = _pf_decode(a.f, Val(P), n)
    db = _pf_decode(b.f, Val(P), n)
    _pf_root(Val(P), S * _pf_h_kernel(da, db, sites, blocks))
end

function _pf_h_phase(
    a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S},
    fm::ParaFockMapper{P,S},
) where {P,S}
    da = _pf_decode(a, fm)
    db = _pf_decode(b, fm)
    _pf_root(Val(P), S * _pf_h_kernel(da, db, fm.sortedsites, fm.fermionpositions))
end

function kron_phase_factor(fm::ParaFockMapper)
    fm.isfullpartition ||
        throw(ArgumentError("kron_phase_factor requires a full partition"))
    (a, b) -> _pf_h_phase(a, b, fm)
end

phase_factor_l(a::ParaFockNumber{P,S}, b::ParaFockNumber{P,S},
    X, Xbar) where {P,S} = phase_factor_l(a, b, (X, Xbar))

function phase_factor_l(
    a::ParaFockNumber{P,S},
    b::ParaFockNumber{P,S},
    partition,
) where {P,S}
    blocks = _pf_blocks(partition)
    n = _pf_maxsite(blocks)
    d = _pf_decode(a.f, Val(P), n) .- _pf_decode(b.f, Val(P), n)
    _pf_root(Val(P), S * _pf_pair_kernel(d, _pf_inversions(blocks)))
end

function phase_factor_u(
    partition,
    masks,
    state::ParaFockNumber{P,S},
) where {P,S}
    # `masks` is accepted only for interface compatibility.
    blocks = _pf_blocks(partition)
    n = _pf_decode(state.f, Val(P), _pf_maxsite(blocks))
    _pf_root(Val(P), S * _pf_pair_kernel(n, _pf_inversions(blocks)))
end

phase_factor_u(partition, state::ParaFockNumber) =
    phase_factor_u(partition, nothing, state)

function _pf_u_phase(state::ParaFockNumber{P,S}, fm::ParaFockMapper{P,S}) where {P,S}
    n = _pf_decode(state, fm)
    _pf_root(Val(P), S * _pf_pair_kernel(n, fm.inversions))
end

phase_factor_u(fm::ParaFockMapper) = state -> _pf_u_phase(state, fm)

# =====================================================================
# Hilbert spaces
# =====================================================================
struct ParafermionicSpace{
    F,L,PG<:ParafermionicGroup,A
} <: AbstractGroupedHilbertSpace{F}
    modes::Vector{L}
    mode_ordering::OrderedDict{L,Int}
    group::PG
    atomic_id::A
    maxstate::F   # P^N - 1, cached for allocation-free bounds checks

    function ParafermionicSpace(
        input_modes::AbstractVector{L},
        group::PG,
        ::Type{F},
    ) where {L<:ParafermionSym,PG<:ParafermionicGroup,F<:ParaFockNumber}
        isempty(input_modes) &&
            throw(ArgumentError("Cannot create a space with no modes"))

        ms = map(_normalize_sym, input_modes)
        all(m -> group_id(m) == group, ms) ||
            throw(ArgumentError("Not all modes belong to the same group"))

        P = parafermion_order(group)
        S = parafermion_sigma(group)
        F <: ParaFockNumber{P,S} ||
            throw(ArgumentError("State type and group are incompatible"))
        isconcretetype(F) ||
            throw(ArgumentError("State type must be concrete, e.g. ParaFockNumber{P,S,UInt64}"))

        order = OrderedDict{L,Int}(m => i for (i, m) in enumerate(ms))
        length(order) == length(ms) ||
            throw(ArgumentError("Duplicate parafermionic modes"))

        # Unlike a fermionic sign, a parafermionic exchange phase
        # distinguishes the two orientations. Symbolic ordering and
        # Jordan-Wigner ordering must therefore agree.
        issorted(ms; by=_pf_modekey) ||
            throw(ArgumentError(
                "Modes must follow the symbolic order (species name, label). " *
                    "Use ordered subregions; use state_mapper for block permutations."))

        # Check that the storage type can hold every state, and cache the
        # largest state for allocation-free bounds checks in state_index.
        maxstate = F(_pf_convert_checked(_pf_storagetype(F),
            big(P)^length(ms) - 1, "The largest state of $(length(ms)) modes"))

        id = length(ms) == 1 ? atomic_id(only(ms)) : map(atomic_id, ms)
        new{F,L,PG,typeof(id)}(ms, order, group, id, maxstate)
    end
end

function ParafermionicSpace(
    ms::AbstractVector{<:ParafermionSym},
    group::ParafermionicGroup{P,S},
) where {P,S}
    I = default_parafock_representation(P, length(ms))
    ParafermionicSpace(ms, group, ParaFockNumber{P,S,I})
end

function ParafermionicSpace(ms::AbstractVector{<:ParafermionSym})
    isempty(ms) && throw(ArgumentError("Cannot create a space with no modes"))
    ParafermionicSpace(ms, only(unique(map(group_id, ms))))
end

function ParafermionicSpace(
    factors::AbstractVector{<:ParafermionicSpace},
    group::ParafermionicGroup,
    ::Type{F},
) where {F<:ParaFockNumber}
    all(H -> group_id(H) == group, factors) ||
        throw(ArgumentError("Not all spaces belong to the same group"))
    ms = collect(Iterators.flatten(map(modes, factors)))
    # Group assembly uses the group's fixed total mode order.
    sort!(ms; by=_pf_modekey)
    ParafermionicSpace(ms, group, F)
end

function ParafermionicSpace(
    factors::AbstractVector{<:ParafermionicSpace},
    group::ParafermionicGroup{P,S},
) where {P,S}
    N = sum(nbr_of_modes, factors)
    I = default_parafock_representation(P, N)
    ParafermionicSpace(factors, group, ParaFockNumber{P,S,I})
end

Base.:(==)(a::ParafermionicSpace, b::ParafermionicSpace) =
    a.modes == b.modes && a.group == b.group

Base.hash(H::ParafermionicSpace, h::UInt) =
    hash(H.modes, hash(H.group, h))

statetype(::ParafermionicSpace{F}) where F = F
nbr_of_modes(H::ParafermionicSpace) = length(H.modes)
modes(H::ParafermionicSpace) = H.modes
mode_ordering(H::ParafermionicSpace) = H.mode_ordering
group_id(H::ParafermionicSpace) = H.group
atomic_id(H::ParafermionicSpace) = H.atomic_id
label(H::ParafermionicSpace) = label(only(H.modes))
isconstrained(::ParafermionicSpace) = false

parafermion_order(H::ParafermionicSpace) = parafermion_order(H.group)
parafermion_sigma(H::ParafermionicSpace) = parafermion_sigma(H.group)

maximum_particles(H::ParafermionicSpace) = (parafermion_order(H) - 1) * nbr_of_modes(H)
# P^N can exceed typemax(Int) for large (constrained) spaces, so dim stays
# Union{Int,BigInt} on purpose. It is now cheap (no BigInt for small spaces)
# and no longer on per-state paths, which use H.maxstate instead.
function dim(H::ParafermionicSpace)
    m = H.maxstate.f
    m < typemax(Int) ? Int(m) + 1 : big(m) + 1
end

function basisstates(H::ParafermionicSpace{F}) where F
    TypedIterator{F}(Iterators.map(F, 0:(dim(H)-1)))
end

function basisstate(ind::Integer, H::ParafermionicSpace{F}) where F
    1 <= ind <= dim(H) || throw(BoundsError(H, ind))
    F(ind - 1)
end

# Always returns Int: an index above typemax(Int) can't index a matrix anyway.
# No BigInt is created; comparing native integers with a BigInt goes straight
# to GMP.
function state_index(
    state::ParaFockNumber{P,S},
    H::ParafermionicSpace{<:ParaFockNumber{P,S}},
) where {P,S}
    state.f <= H.maxstate.f || throw(BoundsError(H, state))
    state.f < typemax(Int) || throw(ArgumentError(
        "State index exceeds typemax(Int); the space is too large to index"))
    Int(state.f) + 1
end

state_index(::ParaFockNumber, ::ParafermionicSpace) = throw(ArgumentError("Incompatible state"))

atomic_factors(H::ParafermionicSpace) =
    [ParafermionicSpace([m], H.group, statetype(H)) for m in H.modes]

add_tag(H::ParafermionicSpace, tag) =
    ParafermionicSpace(
        map(m -> add_tag(m, tag), H.modes),
        add_tag(H.group, tag),
        statetype(H),
    )

function Base.show(io::IO, H::ParafermionicSpace)
    print(io, "ParafermionicSpace{",
        parafermion_order(H), ",", parafermion_sigma(H), "}(")
    join(io, H.modes, ", ")
    print(io, ")")
end

function statetype(a::ParafermionSym)
    P, S = parafermion_order(a), parafermion_sigma(a)
    I = default_parafock_representation(P, 1)
    ParaFockNumber{P,S,I}
end

basisstates(a::ParafermionSym) =
    [statetype(a)(n) for n in 0:(parafermion_order(a)-1)]

nbr_of_modes(::ParafermionSym) = 1
maximum_particles(a::ParafermionSym) = parafermion_order(a) - 1
isconstrained(::ParafermionSym) = false

hilbert_space(a::ParafermionSym) =
    ParafermionicSpace([_normalize_sym(a)], symbolic_group(a))

hilbert_space(a::SymbolicParafermionBasis, labels::AbstractVector) =
    ParafermionicSpace([a[l] for l in labels])

hilbert_space(
    a::SymbolicParafermionBasis,
    labels::AbstractVector,
    states::AbstractVector{<:AbstractBasisState},
) = ConstrainedSpace(hilbert_space(a, labels), states)

issubsystem(Hsub::AbstractHilbertSpace, H::ParafermionicSpace) =
    isorderedsubsystem(Hsub, H)

function _find_position(a::ParafermionSym, H::ParafermionicSpace)
    get(H.mode_ordering, _normalize_sym(a), 0)
end

function _find_position(A::ParafermionicSpace, H::ParafermionicSpace)
    nbr_of_modes(A) == 1 ||
        throw(ArgumentError("Expected a single-mode subspace"))
    _find_position(only(modes(A)), H)
end

function combine_into_group(group::ParafermionicGroup, factors)
    xs = collect(factors)
    isempty(xs) && throw(ArgumentError("Cannot combine an empty group"))
    all(x -> group_id(x) == group, xs) ||
        throw(ArgumentError("Not all factors belong to the same group"))

    if all(x -> x isa ParafermionSym, xs)
        ms = ParafermionSym[x for x in xs]
        sort!(ms; by=_pf_modekey)
        return ParafermionicSpace(ms, group)
    elseif all(x -> x isa ParafermionicSpace, xs)
        hs = ParafermionicSpace[x for x in xs]
        return ParafermionicSpace(hs, group)
    end
    throw(ArgumentError("Expected all symbols or all grouped spaces"))
end

state_mapper(H::ParafermionicSpace, Hsub::AbstractHilbertSpace) =
    state_mapper(H, (Hsub,))

function state_mapper(H::ParafermionicSpace, Hs)
    positions = Tuple(
        Tuple(_find_position(atom, H) for atom in atomic_factors(A))
        for A in Hs
    )
    all(i -> i > 0, Iterators.flatten(positions)) ||
        throw(ArgumentError("All subspaces must belong to the full space"))
    ParaFockMapper(positions, statetype(H), nbr_of_modes(H))
end

function combine_states(states, H::ParafermionicSpace{ParaFockNumber{P,S,I}}) where {P,S,I}
    length(states) == nbr_of_modes(H) ||
        throw(DimensionMismatch("Expected one state per atomic mode"))

    value = zero(I)
    place = one(I)
    for state in states
        state isa ParaFockNumber{P,S} ||
            throw(ArgumentError("Incompatible atomic state"))
        state.f < P ||
            throw(ArgumentError("Expected a single-mode state"))
        value += state.f * place
        place *= P
    end
    (ParaFockNumber{P,S,I}(value),), (1,)
end

# Same hook semantics as the fermionic source: return f_H, NOT an
# already assembled trace ratio. The caller must use f_X / f_Y.
partial_trace_phase_factor(a, b, H::ParafermionicSpace) =
    phase_factor_f(a, b, nbr_of_modes(H))

inverse_partial_trace_phase_factor(a, b, H::ParafermionicSpace) =
    conj(partial_trace_phase_factor(a, b, H))

# Calls the existing representation machinery; builds no matrices here.
"""
    parafermions(H)
    parafermions(H, species)

Return an `OrderedDict` mapping each label to the matrix of the
annihilation operator for that mode of `H`. If several species in `H`
share labels, labels alone are ambiguous and the one-argument form throws;
use the second form to pick one species.
"""
function parafermions(H::ParafermionicSpace)
    allunique(label(m) for m in modes(H)) || throw(ArgumentError(
        "Labels are ambiguous: several species in $H share labels. " *
            "Use parafermions(H, species) to select one species."))
    OrderedDict(label(m) => representation(m, H) for m in modes(H))
end

function parafermions(H::ParafermionicSpace, species::SymbolicParafermionBasis)
    ms = filter(m -> symbolic_basis(m) == species, modes(H))
    isempty(ms) &&
        throw(ArgumentError("Species $(species.name) has no modes in $H"))
    OrderedDict(label(m) => representation(m, H) for m in ms)
end

operators(H::ParafermionicSpace) = parafermions(H)

# =====================================================================
# Numerical operator application
#
# Cache = (one-based mode position, base-P place value).
# =====================================================================
function _precomputation_before_operator_application(
    op::ParafermionSym,
    H::AbstractHilbertSpace{ParaFockNumber{P,S,I}},
) where {P,S,I}
    parafermion_order(op) == P && parafermion_sigma(op) == S ||
        throw(ArgumentError("Operator and state conventions do not match"))

    position = _find_position(op, H)
    position > 0 ||
        throw(ArgumentError("Operator ($op) is not part of space ($H)"))

    # Checked rather than assumed. Runs once per operator, not per state.
    place = _pf_convert_checked(I, big(P)^(position - 1),
        "The place value of mode $position")
    (position, place)
end

function _pf_apply(
    op::ParafermionSym,
    state::ParaFockNumber{P,S,I},
    cache;
    transpose::Bool=false,
) where {P,S,I}
    _, place = cache
    n = Int(rem(div(state.f, place), P))

    # transpose means ordinary matrix transpose, not adjoint.
    #
    # Transpose reverses the local shift, but DOES NOT conjugate
    # the diagonal Jordan-Wigner string.
    creation = xor(op.creation, transpose)

    if (creation && n == P - 1) || (!creation && n == 0)
        return state, ComplexF64(0)
    end

    exponent = S * _pf_degree(op) * _pf_left_charge(state.f, place, Val(P))
    amplitude = _pf_root(Val(P), exponent)

    # Add/subtract directly: `-1 * place` would wrap around for unsigned I.
    newstate = ParaFockNumber{P,S,I}(creation ? state.f + place : state.f - place)
    newstate, amplitude
end

function apply_local_operator(
    op::ParafermionSym,
    state::ParaFockNumber,
    H::ParafermionicSpace,
    cache;
    transpose::Bool=false,
)
    _pf_apply(op, state, cache; transpose=transpose)
end

function apply_local_operators(
    op::NCMul{<:Any,<:ParafermionSym},
    state::ParaFockNumber,
    H::ParafermionicSpace,
    caches;
    transpose::Bool=false,
)
    factors = op.factors
    length(factors) == length(caches) ||
        throw(DimensionMismatch("Wrong number of operator caches"))

    result = state
    phase = ComplexF64(1)
    N = length(factors)

    # AB acts on a ket as B then A.
    # (AB)^T acts as A^T then B^T.
    for m in eachindex(factors)
        n = transpose ? m : N - m + 1
        result, amplitude =
            _pf_apply(factors[n], result, caches[n]; transpose=transpose)
        iszero(amplitude) && return result, zero(op.coeff * phase)
        phase *= amplitude
    end

    result, op.coeff * phase
end


# =====================================================================
# API, robustness, transpose, BigInt storage, and rewritten phase hooks
# =====================================================================

@testitem "Parafermionic API and robustness" begin
    using LinearAlgebra
    import FermionicHilbertSpaces as FHS

    close(a, b) = isapprox(a, b; atol=2e-11, rtol=2e-11)
    root(p, e) = cispi(2 * mod(e, p) / p)

    # Independent reference exponents (no production helpers).
    f_exp(a, b, X) = sum((b[i] * (a[j] - b[j]) for i in X for j in X if i < j); init=0)
    l_exp(a, b, part) = sum(((a[i] - b[i]) * (a[j] - b[j])
                             for s in eachindex(part) for r in (s+1):length(part)
                             for i in part[s] for j in part[r] if i > j); init=0)
    u_exp(a, part) = l_exp(a, zero(a), part)

    @testset "occupation beyond the stored digits" begin
        x = FHS.ParaFockNumber{3}(5)                    # digits (2, 1)
        @test x.f isa UInt64
        @test FHS.occupation(x, 1) == 2
        @test FHS.occupation(x, 2) == 1
        @test FHS.occupation(x, 3) == 0
        @test FHS.occupation(x, 50) == 0                # 3^49 overflows UInt64
        @test FHS.occupation(x, 1000) == 0
        @test FHS.occupations(x, 45) == [2; 1; zeros(Int, 43)]
        @test_throws ArgumentError FHS.occupation(x, 0)
    end

    @testset "type stability and state_index" begin
        @test (@inferred FHS.ParaFockNumber{3,1}(5)).f isa UInt64
        @test FHS.ParaFockNumber{3}(big(5)).f isa BigInt
        @test FHS.ParaFockNumber{3}(big(5)) == FHS.ParaFockNumber{3}(5)
        c = FHS.parafermion_basis(:c, 3)
        H = hilbert_space(c, 1:3)
        s = FHS.basisstate(7, H)                        # digits (0, 2, 0)
        @test @inferred(FHS.state_index(s, H)) === 7
        @test @inferred(FHS.particle_number(s)) === 2
        @test FHS.parafermion_charge(s) === 2
        @test_throws BoundsError FHS.state_index(FHS.ParaFockNumber{3}(27), H)
        @test_throws ArgumentError FHS.state_index(FHS.ParaFockNumber{3,-1}(0), H)
    end

    @testset "fermion-only helpers throw for p > 2" begin
        x3 = FHS.parafock_from_digits([1, 2], Val(3))
        @test_throws ArgumentError FHS.bits(x3, 2)
        @test_throws ArgumentError FHS.parity(x3)
        x2 = FHS.parafock_from_digits([1, 1, 0], Val(2))
        @test FHS.bits(x2, 3) == [1, 1, 0]
        @test FHS.parity(x2) == 1
    end

    @testset "@parafermions and multi-species groups" begin
        ab = @parafermions 3 a b
        @test ab == (a, b)
        @test FHS.symbolic_group(a) == FHS.symbolic_group(b)
        @test FHS.parafermion_order(a) == 3
        @test FHS.parafermion_sigma(a) == 1
        @parafermions 3 d
        @test FHS.symbolic_group(d) != FHS.symbolic_group(a)

        H = FHS.ParafermionicSpace([a[1], a[2], b[1]])
        R(op) = representation(op, H)
        A1, B1 = R(a[1]), R(b[1])
        # Same group, a[1] precedes b[1]: a[1] b[1] = ω^(-1) b[1] a[1] (σ = +1).
        @test close(A1 * B1, root(3, -1) * B1 * A1)
        @test close(R(b[1] * a[1]), B1 * A1)

        # Label 1 occurs in both species, so labels alone are ambiguous.
        @test_throws ArgumentError FHS.parafermions(H)
        ops_a = FHS.parafermions(H, a)
        @test collect(keys(ops_a)) == [1, 2]
        @test close(ops_a[1], A1)
        @test collect(keys(FHS.parafermions(H, b))) == [1]
        @test_throws ArgumentError FHS.parafermions(H, d)
        @test collect(keys(FHS.parafermions(hilbert_space(a, 1:2)))) == [1, 2]
    end

    @testset "operator application with BigInt storage" begin
        p, N = 3, 41                                    # 3^41 - 1 > typemax(UInt64)
        c = FHS.parafermion_basis(:c, p)
        H = hilbert_space(c, 1:N)
        @test zero(FHS.statetype(H)).f isa BigInt

        ns = [mod(i, p) for i in 1:N]
        ns[N] = 0
        state = FHS.parafock_from_digits(ns, Val(p))
        cache = FHS._precomputation_before_operator_application(c[N]', H)

        up, amp_up = FHS.apply_local_operator(c[N]', state, H, cache)
        @test FHS.occupations(up, N) == [ns[1:(N-1)]; 1]
        @test close(amp_up, root(p, sum(ns[1:(N-1)])))

        down, amp_down = FHS.apply_local_operator(c[N], up, H, cache)
        @test down == state
        @test close(amp_up * amp_down, 1)

        vac = zero(FHS.statetype(H))
        @test iszero(last(FHS.apply_local_operator(c[N], vac, H, cache)))

        @test FHS.state_index(FHS.basisstate(5, H), H) === 5
        @test_throws ArgumentError FHS.state_index(up, H)   # index > typemax(Int)
    end

    @testset "transpose application: p=$p sigma=$sigma" for p in (2, 3), sigma in (-1, 1)
        c = FHS.parafermion_basis(:c, p; sigma=sigma)
        H = hilbert_space(c, 1:3)
        D = dim(H)
        states = collect(basisstates(H))
        precompute(op) = FHS._precomputation_before_operator_application(op, H)

        # Build the matrix of an application rule, column by column.
        function assemble(apply)
            M = zeros(ComplexF64, D, D)
            for s in states
                t, amp = apply(s)
                iszero(amp) || (M[FHS.state_index(t, H), FHS.state_index(s, H)] += amp)
            end
            M
        end

        for op in (c[1], c[2], c[3]')
            cache = precompute(op)
            M = representation(op, H)
            @test close(assemble(s -> FHS.apply_local_operator(op, s, H, cache)), M)
            @test close(assemble(s -> FHS.apply_local_operator(op, s, H, cache; transpose=true)),
                transpose(M))
        end

        for word in (c[1]' * c[2] * c[3], (1 + 2im) * c[2]' * c[3]')
            caches = map(precompute, word.factors)
            M = representation(word, H)
            for flag in (false, true)
                @test close(
                    assemble(s -> FHS.apply_local_operators(word, s, H, caches; transpose=flag)),
                    flag ? transpose(M) : M)
            end
        end
    end

    @testset "p=6 mixes exact and inexact roots" begin
        @test FHS._pf_root(Val(6), 3) === ComplexF64(-1)
        @test FHS._pf_root(Val(6), -3) === ComplexF64(-1)
        @test FHS._pf_root(Val(6), 12) === ComplexF64(1)
        @test FHS._pf_root(Val(4), 1) === ComplexF64(0, 1)
        @test close(FHS._pf_root(Val(6), 1), cispi(1 / 3))

        c = FHS.parafermion_basis(:c, 6)
        H = hilbert_space(c, 1:2)
        C1 = representation(c[1], H)
        C2 = representation(c[2], H)
        for (A, B, dA, dB) in ((C1, C2, -1, -1), (C1, C2', -1, 1),
            (C1', C2, 1, -1), (C1', C2', 1, 1))
            @test close(A * B, root(6, -dA * dB) * B * A)
        end
        @test close(C2^6, zeros(36, 36))
        @test norm(C2^5) > 0
    end

    @testset "phase hooks vs reference: p=$p sigma=$sigma" for p in (2, 3, 4), sigma in (-1, 1)
        c = FHS.parafermion_basis(:c, p; sigma=sigma)
        H = hilbert_space(c, 1:3)
        states = collect(basisstates(H))
        ns = [FHS.occupations(s, 3) for s in states]
        table(g) = [g(a, b) for a in states, b in states]

        f_ref(X) = [root(p, sigma * f_exp(na, nb, X)) for na in ns, nb in ns]
        @test close(table((a, b) -> FHS.phase_factor_f(a, b, 3)), f_ref(1:3))
        @test close(table((a, b) -> FHS.phase_factor_f(a, b, 2)), f_ref(1:2))
        @test close(table((a, b) -> FHS.phase_factor_f(a, b, (1, 3))), f_ref((1, 3)))
        @test close(table((a, b) -> FHS.phase_factor_f(a, b, [3, 1])), f_ref((1, 3)))

        for partition in (((1,), (2,), (3,)), ((3,), (1,), (2,)),
            ((1, 3), (2,)), ((2, 3), (1,)))
            mapper = FHS.state_mapper(H, [hilbert_space(c, collect(X)) for X in partition])
            h = FHS.kron_phase_factor(mapper)
            u = FHS.phase_factor_u(mapper)

            h_ref = [root(p, sigma * (f_exp(na, nb, 1:3) -
                sum(f_exp(na, nb, X) for X in partition)))
                     for na in ns, nb in ns]
            l_ref = [root(p, sigma * l_exp(na, nb, partition)) for na in ns, nb in ns]
            u_ref = [root(p, sigma * u_exp(na, partition)) for na in ns]

            @test close(table(h), h_ref)
            @test close(table((a, b) -> FHS.phase_factor_h(a, b, partition)), h_ref)
            @test close(table((a, b) -> FHS.phase_factor_l(a, b, partition)), l_ref)
            @test close(u.(states), u_ref)
            @test close([FHS.phase_factor_u(partition, s) for s in states], u_ref)
            @test_throws ArgumentError FHS.split_state(FHS.ParaFockNumber{p,sigma}(p^3), mapper)
        end
    end
end


# =====================================================================
# States, state mappers, and independent checks of every phase hook
# =====================================================================

@testitem "Parafermionic states, mappers, and phase factors" begin
    import FermionicHilbertSpaces as FHS

    close(a, b) = isapprox(a, b; atol=2e-12, rtol=2e-12)

    # Independent reference expressions: no production phase helpers.
    root(p, exponent) = cis(2π * mod(exponent, p) / p)

    function f_exponent(a, b, X)
        e = 0
        for i in X, j in X
            i < j && (e += b[i] * (a[j] - b[j]))
        end
        e
    end

    function l_exponent(a, b, partition)
        e = 0
        for s in eachindex(partition), r in (s+1):length(partition)
            for i in partition[s], j in partition[r]
                i > j && (e += (a[i] - b[i]) * (a[j] - b[j]))
            end
        end
        e
    end

    function u_exponent(a, partition)
        e = 0
        for s in eachindex(partition), r in (s+1):length(partition)
            for i in partition[s], j in partition[r]
                i > j && (e += a[i] * a[j])
            end
        end
        e
    end

    partitions = (
        ((1,), (2,), (3,)),
        ((3,), (1,), (2,)),
        ((1, 3), (2,)),
        ((2,), (1, 3)),
        ((2, 3), (1,)),
    )

    for p in (2, 3, 4), sigma in (-1, 1)
        @testset "p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:3)
            states = collect(basisstates(H))

            @test dim(H) == p^3
            @test length(states) == p^3
            @test allunique(states)

            for (index, state) in enumerate(states)
                ns = digits(index - 1; base=p, pad=3)

                @test state.f == index - 1
                @test FHS.occupations(state, 3) == ns
                @test FHS.particle_number(state) == sum(ns)
                @test FHS.parafermion_charge(state) == mod(sum(ns), p)
                @test FHS.state_index(state, H) == index
                @test FHS.basisstate(index, H) == state
                @test FHS.parafock_from_digits(
                    ns, Val(p), Val(sigma)
                ) == state
            end

            for partition in partitions
                Hs = [hilbert_space(c, collect(X)) for X in partition]
                mapper = FHS.state_mapper(H, Hs)
                h_hook = FHS.kron_phase_factor(mapper)
                u_hook = FHS.phase_factor_u(mapper)

                for state in states
                    candidates, weights = FHS.split_state(state, mapper)
                    parts = only(candidates)

                    @test only(weights) == 1

                    ns = FHS.occupations(state, 3)
                    for (part, X) in zip(parts, partition)
                        @test FHS.occupations(part, length(X)) ==
                            ns[collect(X)]
                    end

                    combined, weights2 =
                        FHS.combine_states(parts, mapper)

                    @test only(combined) == state
                    @test only(weights2) == 1
                    @test close(u_hook(state),
                        root(p, sigma * u_exponent(ns, partition)))
                end
            end

            # Partial and overlapping subregions may be split,
            # but cannot be uniquely combined into a full state.
            F = FHS.statetype(H)
            partial = FHS.ParaFockMapper(((1, 3),), F, 3)
            overlap = FHS.ParaFockMapper(((1, 2), (2, 3)), F, 3)

            for mapper in (partial, overlap)
                parts = only(first(FHS.split_state(last(states), mapper)))
                @test_throws ArgumentError FHS.combine_states(parts, mapper)
            end

            @test_throws ArgumentError FHS.ParaFockMapper(
                ((3, 1), (2,)), F, 3)
            @test_throws ArgumentError hilbert_space(c, Int[])
            @test_throws ArgumentError hilbert_space(c, [1, 1])
        end
    end

    # A direct regression against accidentally treating phases as signs.
    a = FHS.parafock_from_digits([1, 1], Val(3))
    b = FHS.parafock_from_digits([1, 0], Val(3))
    z = FHS.phase_factor_f(a, b, 2)

    @test close(z, cis(2π / 3))
    @test !close(z, conj(z))
    @test close(FHS.inverse_phase_factor_f(a, b, 2), conj(z))

    # Large encodings must not overflow UInt64.
    ns = fill(2, 50)
    state = FHS.parafock_from_digits(ns, Val(3))
    @test state.f isa BigInt
    @test state.f == big(3)^50 - 1
    @test FHS.occupations(state, 50) == ns
end


# =====================================================================
# Numerical and symbolic algebra
# =====================================================================

@testitem "Parafermionic ladder algebra and representation consistency" begin
    using LinearAlgebra
    import FermionicHilbertSpaces as FHS

    close(a, b) = isapprox(a, b; atol=2e-11, rtol=2e-11)

    for p in (2, 3, 4), sigma in (-1, 1)
        @testset "p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:3)

            D = Int(dim(H))
            Id = Matrix{ComplexF64}(I, D, D)
            Zeros = zeros(ComplexF64, D, D)
            R(op) = Matrix(representation(op, H))

            C = [R(c[j]) for j in 1:3]
            N = [R(FHS.parafermion_number(c[j])) for j in 1:3]

            for j in 1:3
                @test close(R(c[j]'), C[j]')

                # Independent occupation-basis check of the JW string.
                # Check every column, including the annihilation boundary.
                for n in 0:(D-1)
                    ns = digits(n; base=p, pad=3)
                    expected = zeros(ComplexF64, D)

                    if ns[j] > 0
                        row = n - p^(j - 1) + 1
                        exponent = -sigma * sum(ns[1:(j-1)])
                        expected[row] = cis(2π * mod(exponent, p) / p)
                    end

                    @test close(C[j][:, n+1], expected)
                end

                # Nilpotent Fock ladders, not cyclic clock shifts.
                @test close(C[j]^p, Zeros)
                @test close((C[j]')^p, Zeros)
                @test norm(C[j]^(p - 1)) > 0
                @test close(R(c[j]^p), Zeros)

                # Local finite-occupation identities.
                for m in 1:(p-1)
                    @test close((C[j]')^m * C[j]^m + C[j]^(p-m) * (C[j]')^(p-m), Id)
                end

                expected_number = Diagonal([
                    digits(n; base=p, pad=3)[j] for n in 0:(D-1)
                ])
                @test close(N[j], expected_number)
                @test close(N[j] * C[j] - C[j] * N[j], -C[j])
                @test close(N[j] * C[j]' - C[j]' * N[j], C[j]')

                # c†c is the occupied-state projector, not N for p>2.
                @test close(
                    C[j]' * C[j],
                    Diagonal([
                        Int(digits(n; base=p, pad=3)[j] > 0)
                        for n in 0:(D-1)
                    ]),
                )
                if p > 2
                    @test !close(C[j]' * C[j], N[j])
                end
            end

            # All four creation/annihilation exchange relations.
            for i in 1:3, j in (i+1):3
                for createA in (false, true), createB in (false, true)
                    A = createA ? c[i]' : c[i]
                    B = createB ? c[j]' : c[j]
                    degreeA = createA ? 1 : -1
                    degreeB = createB ? 1 : -1

                    q = cis(2π * mod(-sigma * degreeA * degreeB, p) / p)

                    MA, MB = R(A), R(B)
                    @test close(MA * MB, q * MB * MA)

                    # Exercise symbolic reordering in both directions.
                    @test close(R(A * B), MA * MB)
                    @test close(R(B * A), MB * MA)
                    @test close(R((A * B)'), (MA * MB)')
                end
            end

            # Longer words test application order, local powers,
            # complex coefficients, and symbolic adjoints.
            words = (
                c[3] * c[1]' * c[2],
                c[2]' * c[1] * c[2],
                c[1]' * c[1] * c[3]',
                (1 + 2im) * c[3]' * c[2] * c[1]',
            )
            expected_words = (
                C[3] * C[1]' * C[2],
                C[2]' * C[1] * C[2],
                C[1]' * C[1] * C[3]',
                (1 + 2im) * C[3]' * C[2] * C[1]',
            )

            for (word, expected) in zip(words, expected_words)
                @test close(R(word), expected)
                @test close(R(word'), expected')
            end

            @test_throws ArgumentError representation(c[4], H)
        end
    end

    # Separate groups commute, as in separate @fermions declarations.
    @testset "Independent groups commute" begin
        a = FHS.parafermion_basis(:a, 3)
        b = FHS.parafermion_basis(:b, 3)
        Ha = hilbert_space(a, [1])
        Hb = hilbert_space(b, [1])
        H = tensor_product((Ha, Hb))

        A = representation(a[1], H)
        B = representation(b[1], H)

        @test close(A * B, B * A)
        @test close(representation(b[1] * a[1], H), B * A)
    end
end


# =====================================================================
# Tensor products: absolute phase orientation and structural properties
# =====================================================================

@testitem "Parafermionic tensor product properties" begin
    using LinearAlgebra, Random
    import SparseArrays: issparse, spzeros
    import FermionicHilbertSpaces as FHS

    rng = MersenneTwister(1234)
    close(a, b) = isapprox(a, b; atol=3e-10, rtol=3e-10)

    # Occupation labels are zero-based; Julia matrix indices are not.
    for p in (2, 3), sigma in (-1, 1)
        function E!(M, a, b)
            fill!(M, 0)
            M[a+1, b+1] = 1
            M
        end
        A = spzeros(ComplexF64, p, p)
        B = spzeros(ComplexF64, p, p)
        global_unit = spzeros(ComplexF64, p^2, p^2)

        @testset "Matrix-unit phases: p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:2)
            H1 = hilbert_space(c, [1])
            H2 = hilbert_space(c, [2])

            # Exhaust all two-mode matrix units. These are independent
            # expected values, not products of production phase hooks.
            for a1 in 0:(p-1), b1 in 0:(p-1),
                a2 in 0:(p-1), b2 in 0:(p-1)

                A = E!(A, a1, b1)
                B = E!(B, a2, b2)
                global_unit = E!(global_unit, a1 + p*a2, b1 + p*b2)

                h = cis(2π * mod(sigma * b1 * (a2 - b2), p) / p)
                ell = cis(2π * mod(sigma * (a1 - b1) * (a2 - b2), p) / p)

                G = FHS.generalized_kron((A, B), (H1, H2), H)
                Greversed = FHS.generalized_kron((B, A), (H2, H1), H)

                @test close(G, h * global_unit)
                @test close(Greversed, G)

                @test close(tensor_product((A, B), (H1, H2), H),
                    h * global_unit)
                @test close(tensor_product((B, A), (H2, H1), H),
                    ell * h * global_unit)
            end
        end
    end

    # Noncontiguous blocks and nontrivial internal f_X phases.
    for p in (2, 3), sigma in (-1, 1)
        A = randn(rng, ComplexF64, p^2, p^2)
        B = randn(rng, ComplexF64, p, p)
        A2 = randn(rng, ComplexF64, p^2, p^2)
        B2 = randn(rng, ComplexF64, p, p)
        @testset "Interleaved blocks: p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:3)
            atoms = [hilbert_space(c, [j]) for j in 1:3]
            HX = hilbert_space(c, [1, 3])
            HZ = atoms[2]
            Hs = [HX, HZ]

            G = FHS.generalized_kron([A, B], Hs, H)
            T = tensor_product([A, B], Hs, H)

            # Ordered tensor product means product of embeddings.
            @test close(T,
                embed(A, HX, H) * embed(B, HZ, H))
            @test close(tensor_product([B, A], reverse(Hs), H),
                embed(B, HZ, H) * embed(A, HX, H))

            # Verify the l correction for interleaved blocks.
            states = basisstates(H)
            L = [
                FHS.phase_factor_l(a, b, ((1, 3), (2,)))
                for a in states, b in states
            ]
            @test close(T, L .* G)

            # Hilbert-Schmidt factorization: Szalay Eq. 18 analogue.
            G2 = FHS.generalized_kron([A2, B2], Hs, H)
            @test close(tr(G' * G2), tr(A' * A2) * tr(B' * B2))

            # Associativity under refinement of an interleaved block.
            localops = [
                randn(rng, ComplexF64, p, p) for _ in 1:3
            ]
            A13 = FHS.generalized_kron(
                localops[[1, 3]], atoms[[1, 3]], HX
            )
            nested = FHS.generalized_kron(
                [A13, localops[2]], Hs, H
            )
            flat = FHS.generalized_kron(localops, atoms, H)

            @test close(nested, flat)
            @test close(flat,
                prod(embed(localops[j], atoms[j], H) for j in 1:3))

            # Embedding through an intermediate subsystem.
            @test close(
                embed(embed(localops[3], atoms[3], HX), HX, H),
                embed(localops[3], atoms[3], H),
            )

            # Adjoint compatibility and multiplication within a
            # single embedded subsystem.
            EA = embed(A, HX, H)
            @test close(EA', embed(A', HX, H))
            @test close(
                EA * embed(A2, HX, H),
                embed(A * A2, HX, H),
            )

            # Identity factors give the canonical embedding.
            @test close(EA, FHS.generalized_kron((A, I), Hs, H))
            @test close(EA, tensor_product((A, I), Hs, H))
            @test close(EA, tensor_product((I, A), reverse(Hs), H))

            # Algebra representation and embedding must agree.
            for j in (1, 3)
                @test close(
                    embed(representation(c[j], HX), HX, H),
                    representation(c[j], H))
            end

            # Space construction and overlap rejection.
            H12 = hilbert_space(c, [1, 2])
            @test tensor_product(atoms[1], atoms[2]) ==
                tensor_product((atoms[1], atoms[2]))
            @test_throws ArgumentError tensor_product(H12, atoms[2])
        end
    end
end


# =====================================================================
# Partial trace: absolute phase orientation, adjointness, and nesting
# =====================================================================

@testitem "Parafermionic partial trace properties" begin
    using LinearAlgebra, Random
    import FermionicHilbertSpaces as FHS

    rng = MersenneTwister(5678)
    close(a, b) = isapprox(a, b; atol=3e-10, rtol=3e-10)

    function E(d, a, b)
        M = zeros(ComplexF64, d, d)
        M[a+1, b+1] = 1
        M
    end

    for p in (2, 3, 4), sigma in (-1, 1)
        @testset "Matrix-unit trace phases: p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:2)
            H1 = hilbert_space(c, [1])
            H2 = hilbert_space(c, [2])

            for a1 in 0:(p-1), b1 in 0:(p-1),
                a2 in 0:(p-1), b2 in 0:(p-1)

                M = E(p^2, a1 + p*a2, b1 + p*b2)

                # Retain the first mode: no traced mode precedes it.
                expected1 = a2 == b2 ? E(p, a1, b1) : zeros(ComplexF64, p, p)

                # Retain the second mode:
                # phase = ω^[-σ b1 (a2-b2)] when a1=b1.
                phase = cis(2π * mod(-sigma * b1 * (a2 - b2), p) / p)
                expected2 = a1 == b1 ? phase * E(p, a2, b2) : zeros(ComplexF64, p, p)

                @test close(partial_trace(M, H, H1), expected1)
                @test close(partial_trace(M, H, H2), expected2)
            end
        end
    end

    for p in (2, 3), sigma in (-1, 1)
        @testset "Interleaved trace: p=$p sigma=$sigma" begin
            c = FHS.parafermion_basis(:c, p; sigma=sigma)
            H = hilbert_space(c, 1:3)
            HX = hilbert_space(c, [1, 3])
            HZ = hilbert_space(c, [2])
            H3 = hilbert_space(c, [3])

            D = Int(dim(H))
            A = randn(rng, ComplexF64, p^2, p^2)
            B = randn(rng, ComplexF64, p, p)
            M = randn(rng, ComplexF64, D, D)
            M2 = randn(rng, ComplexF64, D, D)

            reduced = partial_trace(M, H, HX)

            # Pair syntax matches the fermionic tests.
            @test close(
                partial_trace(M, H => HX), reduced
            )

            # Partial trace is the Hilbert-Schmidt adjoint of embedding.
            # This is particularly sensitive to missing conjugations.
            @test close(
                tr(embed(A, HX => H)' * M),
                tr(A' * reduced),
            )

            # Linearity, trace preservation, and adjoint compatibility.
            alpha, beta = 0.7 + 0.2im, -0.3 + 0.9im
            @test close(
                partial_trace(alpha*M + beta*M2, H, HX),
                alpha*reduced + beta*partial_trace(M2, H, HX),
            )
            @test close(tr(reduced), tr(M))
            @test close(partial_trace(M', H, HX), reduced')

            # Identity normalization: trace is unnormalized.
            @test close(partial_trace(1.0*I(D), H, HX), p * I(p^2))

            # Sequential traces, including an interleaved intermediate
            # subsystem and a retained rightmost mode.
            @test close(
                partial_trace(
                    partial_trace(M, H, HX), HX, H3
                ),
                partial_trace(M, H, H3),
            )

            # Trace of a transported product and both ordered products.
            G = FHS.generalized_kron([A, B], [HX, HZ], H)
            T = tensor_product([A, B], [HX, HZ], H)
            Treversed = tensor_product([B, A], [HZ, HX], H)

            for product in (G, T, Treversed)
                @test close(partial_trace(product, H, HX), A * tr(B))
                @test close(partial_trace(product, H, HZ), B * tr(A))
            end

            # Explicit interleaved example:
            # retain modes (1,3), trace mode 2 at occupation 1.
            #
            # a=(0,1,1), b=(1,1,0), so the trace phase is ω^(-σ).
            a = 0 + p + p^2
            b = 1 + p
            input = E(p^3, a, b)
            expected = cis(2π * mod(-sigma, p) / p) * E(p^2, p, 1)

            @test close(partial_trace(input, H, HX), expected)
        end
    end
end


# =====================================================================
# Fermionic limit: compare against the actual existing implementation
# =====================================================================

@testitem "Parafermionic p=2 agrees with fermionic implementation" begin
    using LinearAlgebra, Random
    import FermionicHilbertSpaces as FHS
    using FermionicHilbertSpaces: @fermions

    rng = MersenneTwister(9012)
    close(a, b) = isapprox(a, b; atol=2e-11, rtol=2e-11)

    @fermions f
    HF = hilbert_space(f, 1:3)

    for sigma in (-1, 1)
        c = FHS.parafermion_basis(:c, 2; sigma=sigma)
        HP = hilbert_space(c, 1:3)

        @test dim(HP) == dim(HF)

        for j in 1:3
            @test close(
                representation(c[j], HP),
                representation(f[j], HF),
            )
            @test close(
                representation(c[j]', HP),
                representation(f[j]', HF),
            )
        end

        partitions = (
            ([1], [2], [3]),
            ([3], [1], [2]),
            ([1, 3], [2]),
            ([2], [1, 3]),
        )

        for partition in partitions
            Ps = [hilbert_space(c, X) for X in partition]
            Fs = [hilbert_space(f, X) for X in partition]
            ops = [
                randn(rng, ComplexF64, 2^length(X), 2^length(X))
                for X in partition
            ]

            @test close(
                FHS.generalized_kron(ops, Ps, HP),
                FHS.generalized_kron(ops, Fs, HF),
            )
            @test close(
                tensor_product(ops, Ps, HP),
                tensor_product(ops, Fs, HF),
            )
        end

        M = randn(rng, ComplexF64, 8, 8)
        for X in ([1], [2], [3], [1, 3], [2, 3])
            PX = hilbert_space(c, X)
            FX = hilbert_space(f, X)
            A = randn(rng, ComplexF64, 2^length(X), 2^length(X))

            @test close(
                embed(A, PX, HP),
                embed(A, FX, HF),
            )
            @test close(
                partial_trace(M, HP, PX),
                partial_trace(M, HF, FX),
            )
        end
    end
end
