"""
    @parafermions N a b ...

Create one or more species of ``\\mathbb{Z}_N`` Fock parafermions: `a[l]` is the annihilation operator of the mode
labelled `l`, with `N` states per mode. Modes of one block are ordered by species name and label, and for an
earlier mode `i` and a later mode `j`, `a[i] * a[j] == ω' * a[j] * a[i]` and `a[i]' * a[j] == ω * a[j] * a[i]'`
with `ω = cispi(2 / N)`. Parafermions of different blocks commute.
"""
macro parafermions(N, xs...)
    defs = map(x -> :($(esc(x)) = Field{Parafermion{$(esc(N))}}($(QuoteNode(x)), $xs)), xs)
    length(defs) == 1 && return only(defs)
    return Expr(:block, defs..., :(tuple($(map(esc, xs)...))))
end

struct Parafermion{N}
    creation::Bool
end
mode_operator(::Type{Parafermion{N}}) where {N} = Parafermion{N}(false)
Base.adjoint(k::Parafermion{N}) where {N} = Parafermion{N}(!k.creation)
mat_eltype(::Type{<:Parafermion}) = ComplexF64
Base.show(io::IO, x::Op{<:Parafermion}) = (show(io, x.mode.field); print(io, x.kind.creation ? "†" : ""); _show_label(io, x.mode.label))

_omega(N, k) = cispi(2 * rem(k, N, RoundNearest) / N)
_degree(k::Parafermion) = k.creation ? 1 : -1
# Operators on one mode stay as written: the relations of one mode are not pairwise rewriting rules.
function local_mul_effect(a::Op{Parafermion{N}}, b::Op{Parafermion{N}}) where {N}
    isless(b.mode, a.mode) || return nothing
    return Swap(_omega(N, _degree(a.kind) * _degree(b.kind)))
end

"""
    ParafermionState{N}(n)

Basis state of ``\\mathbb{Z}_N`` parafermion modes: the base-`N` digits of `n` are the occupations, the first mode
in the lowest digit.
"""
struct ParafermionState{N} <: AbstractBasisState
    n::Int
end
Base.isless(a::ParafermionState{N}, b::ParafermionState{N}) where {N} = a.n < b.n
interpret_state(s::AbstractString, ::Type{ParafermionState{N}}) where {N} = ParafermionState{N}(parse(Int, reverse(s); base = N))
_occupation(s::ParafermionState{N}, i) where {N} = (s.n ÷ N^(i - 1)) % N
particle_number(s::ParafermionState{N}) where {N} = sum(digits(s.n; base = N))

# The modes are sorted, since the exchange phase of two modes depends on which of them comes first.
struct ParafermionicSpace{N, M, G <: Group{Parafermion{N}}} <: AbstractGroupedHilbertSpace{ParafermionState{N}}
    modes::Vector{M}
    mode_ordering::OrderedDict{M, Int}
    group::G
    function ParafermionicSpace(modes::AbstractVector{M}, group::G) where {N, M <: Mode, G <: Group{Parafermion{N}}}
        all(m -> symbolic_group(m) == group, modes) || throw(ArgumentError("All modes must belong to the group `$group`."))
        modes = sort(modes)
        allunique(modes) || throw(ArgumentError("The modes of a parafermionic group must be distinct, got $modes."))
        Base.Checked.checked_pow(N, length(modes))
        return new{N, M, G}(modes, OrderedDict{M, Int}(m => i for (i, m) in enumerate(modes)), group)
    end
end
combine_into_group(group::Group{<:Parafermion}, atoms) = ParafermionicSpace(mapreduce(modes, vcat, atoms), group)
combine_into_group(group::Group{<:Parafermion}, atoms::AbstractVector{<:TransposedSpace}) = TransposedSpace(combine_into_group(group, map(parent, atoms)))
hilbert_space(m::Mode{<:Field{<:Parafermion}}) = ParafermionicSpace([m], symbolic_group(m))
hilbert_space(a::Field{<:Parafermion}, labels::AbstractVector) = ParafermionicSpace(map(l -> Mode(a, l), labels), Group(a))
hilbert_space(a::Field{<:Parafermion}, labels::AbstractVector, @nospecialize(constraint::ConstraintLike)) = constrain_space(hilbert_space(a, labels), constraint)

Base.:(==)(a::ParafermionicSpace, b::ParafermionicSpace) = a.modes == b.modes && a.group == b.group
Base.hash(H::ParafermionicSpace, h::UInt) = hash(H.modes, hash(H.group, h))
Base.show(io::IO, H::ParafermionicSpace{N}) where {N} = print(io, "ParafermionicSpace{", N, "}(", join(H.modes, ", "), ")")
modes(H::ParafermionicSpace) = H.modes
mode_ordering(H::ParafermionicSpace) = H.mode_ordering
nbr_of_modes(H::ParafermionicSpace) = length(H.modes)
group_id(H::ParafermionicSpace) = H.group
atomic_id(H::ParafermionicSpace) = only(H.modes)
atom_ids(H::ParafermionicSpace) = H.modes
atom_position(x, H::ParafermionicSpace) = get(H.mode_ordering, atomic_id(x), 0)
atomic_factors(H::ParafermionicSpace) = length(H.modes) == 1 ? [H] : map(m -> ParafermionicSpace([m], H.group), H.modes)
isconstrained(::ParafermionicSpace) = false
add_tag(H::ParafermionicSpace, tag) = ParafermionicSpace(map(m -> add_tag(m, tag), H.modes), add_tag(H.group, tag))
dim(H::ParafermionicSpace{N}) where {N} = N^length(H.modes)
basisstates(H::ParafermionicSpace{N}) where {N} = BasisStates(H, Iterators.map(ParafermionState{N}, 0:(dim(H) - 1)))
function basisstate(i::Integer, H::ParafermionicSpace{N}) where {N}
    1 <= i <= dim(H) || throw(BoundsError(H, i))
    return ParafermionState{N}(i - 1)
end
_state_index(s::ParafermionState{N}, H::ParafermionicSpace{N}) where {N} = 0 <= s.n < dim(H) ? s.n + 1 : 0
has_arithmetic_state_index(::ParafermionicSpace) = true

function _precomputation_before_operator_application(op::Op{Parafermion{N}}, space::AbstractHilbertSpace{ParafermionState{N}}) where {N}
    position = mode_position(op, space)
    iszero(position) && throw(ArgumentError("The operator `$op` is not part of the parafermionic space `$space`."))
    return N^(position - 1)
end
# The generalised Jordan–Wigner string: a creation operator gets the phase ω^(occupation of the earlier modes).
function apply_local_operator(k::Parafermion{N}, s::ParafermionState{N}, space, place) where {N}
    occupation = (s.n ÷ place) % N
    occupation == (k.creation ? N - 1 : 0) && return s, complex(0.0)
    string = particle_number(ParafermionState{N}(s.n % place))
    return ParafermionState{N}(s.n + _degree(k) * place), _omega(N, _degree(k) * string)
end

struct ParafermionMapper{N, P} <: AbstractStateMapper
    positions::P
    nbr_of_modes::Int
end
unique_split(::ParafermionMapper) = true
unique_combine(::ParafermionMapper) = true
state_mapper(H::ParafermionicSpace, Hs::AbstractHilbertSpace) = state_mapper(H, (Hs,))
function state_mapper(H::ParafermionicSpace{N}, Hs) where {N}
    positions = map(Hsub -> [atom_position(atom, H) for atom in atom_ids(Hsub)], Tuple(Hs))
    all(x -> x > 0, Iterators.flatten(positions)) || throw(ArgumentError("All subspaces in `Hs` must be part of `H`, got positions $positions."))
    return ParafermionMapper{N, typeof(positions)}(positions, nbr_of_modes(H))
end
function split_state(s::ParafermionState{N}, mapper::ParafermionMapper{N}) where {N}
    substates = map(mapper.positions) do positions
        ParafermionState{N}(sum(_occupation(s, i) * N^(k - 1) for (k, i) in enumerate(positions); init = 0))
    end
    return (substates,), (1,)
end
function combine_states(states, mapper::ParafermionMapper{N}) where {N}
    sum(length, mapper.positions) == mapper.nbr_of_modes && allunique(Iterators.flatten(mapper.positions)) ||
        throw(ArgumentError("`combine_states` requires a partition of the modes, got $(mapper.positions)."))
    n = 0
    for (s, positions) in zip(states, mapper.positions), (k, i) in enumerate(positions)
        n += _occupation(s, k) * N^(i - 1)
    end
    return (ParafermionState{N}(n),), (1,)
end
combine_states(states, ::ParafermionicSpace{N}) where {N} =
    (ParafermionState{N}(sum(s.n * N^(i - 1) for (i, s) in enumerate(states); init = 0)),), (1,)

# The phase factors of fermions (see `phase_factors.jl`) with ω in place of -1: the matrix unit |a⟩⟨b| is ω^e times
# an ordered product of mode operators, with e = Σ_{i<j} bᵢ (aⱼ - bⱼ).
function _matrix_unit_exponent(a::ParafermionState{N}, b::ParafermionState{N}) where {N}
    exponent = earlier = 0
    na, nb = a.n, b.n
    while na > 0 || nb > 0
        na, da = divrem(na, N)
        nb, db = divrem(nb, N)
        exponent += earlier * (da - db)
        earlier += db
    end
    return exponent
end
partial_trace_phase_factor(a, b, ::ParafermionicSpace{N}) where {N} = _omega(N, _matrix_unit_exponent(a, b))
partial_trace_phase_factor_eltype(::ParafermionicSpace) = ComplexF64
function kron_phase_factor(mapper::ParafermionMapper{N}) where {N}
    return function (a, b)
        parts = sum(map(_matrix_unit_exponent, only(first(split_state(a, mapper))), only(first(split_state(b, mapper)))))
        return _omega(N, _matrix_unit_exponent(a, b) - parts)
    end
end
# ω^(Σ nᵢ nⱼ) over the pairs of modes that the order of the subsystems puts in the opposite order.
function phase_factor_u(mapper::ParafermionMapper{N}) where {N}
    order = reduce(vcat, mapper.positions)
    return function (s)
        exponent = sum(_occupation(s, order[p]) * _occupation(s, order[q]) for q in eachindex(order) for p in 1:(q - 1) if order[p] > order[q]; init = 0)
        return _omega(N, exponent)
    end
end

@testitem "Parafermions" begin
    using LinearAlgebra
    import FermionicHilbertSpaces: particle_number, open_system
    N = 3
    ω = cispi(2 / N)
    @parafermions N c
    @test c[1] * c[2] == ω' * c[2] * c[1] && c[1]' * c[2] == ω * c[2] * c[1]'
    @test c[1] * c[1]' != c[1]' * c[1]

    # A generalised Jordan–Wigner construction from clock matrices, with the first mode in the last `kron` factor.
    Z, lower = Diagonal(ω .^ (0:(N - 1))), diagm(1 => ones(N - 1))
    H = hilbert_space(c, 1:3)
    cs = [representation(c[j], H) for j in 1:3]
    @test cs ≈ [kron(I(N^(3 - j)), lower, fill(Z', j - 1)...) for j in 1:3]
    @test iszero(cs[2]^N) && all((cs[2]')^m * cs[2]^m + cs[2]^(N - m) * (cs[2]')^(N - m) ≈ I for m in 1:(N - 1))
    @test cs[1] * cs[3] ≈ ω' * cs[3] * cs[1] && cs[1]' * cs[3] ≈ ω * cs[3] * cs[1]'
    @test representation(c[3]' * c[1] * c[2], H) ≈ cs[3]' * cs[1] * cs[2]

    # Tensor product, embedding and partial trace on interleaved subsystems agree with the representation on `H`.
    H13, H2 = hilbert_space(c, [1, 3]), hilbert_space(c, 2:2)
    @test tensor_product(H2, H13) == H == hilbert_space(c, [3, 1, 2])
    A, B = c[3]' * c[1] + c[1]', c[2]' + c[2] * c[2]
    MA, MB = representation(A, H13), representation(B, H2)
    @test tensor_product((MA, MB), (H13, H2) => H) ≈ representation(A * B, H)
    @test tensor_product((MB, MA), (H2, H13) => H) ≈ representation(B * A, H)
    @test embed(MA, H13 => H) ≈ representation(A, H)
    ρ = rand(dim(H), dim(H))
    @test tr(partial_trace(ρ, H => H13) * MA) ≈ tr(ρ * representation(A, H))
    # The state with amplitudes `v` is created from the vacuum by the operator `create(v, ops)`.
    create(v, ops) = sum(v[i] * prod(op'^n for (op, n) in zip(ops, digits(i - 1; base = N, pad = length(ops)))) for i in eachindex(v))
    v13, v2 = rand(dim(H13)), rand(dim(H2))
    @test tensor_product((v2, v13), (H2, H13) => H) ≈ representation(create(v2, [c[2]]) * create(v13, [c[1], c[3]]), H)[:, 1]
    Hlr, Hl, Hr, left, right = open_system(H13)
    @test reshape(representation(left(A) * right(A'), Hlr) * reshape(ρ[1:9, 1:9], (Hl, Hr) => Hlr), Hlr => (Hl, Hr)) ≈ MA * ρ[1:9, 1:9] * MA'

    # Hopping conserves the Z_N charge, and N = 2 is a fermion.
    Hq = constrain_space(H, s -> mod(particle_number(s), N))
    inds = [state_index(s, H) for s in basisstates(sector(1, Hq))]
    @test representation(c[1]' * c[2] + hc, sector(1, Hq)) ≈ (cs[1]' * cs[2] + cs[2]' * cs[1])[inds, inds]
    @parafermions 2 p
    @fermions f
    ham(a) = a[1]' * a[2] + a[2] * a[3] + hc + a[2]' * a[2]
    @test representation(ham(p), hilbert_space(p, 1:3)) ≈ representation(ham(f), hilbert_space(f, 1:3))
end
