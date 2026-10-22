# Both sides of a mapping are a Hilbert space or a collection of Hilbert spaces.
const _SpaceGroup = Union{AbstractHilbertSpace,Tuple{Vararg{AbstractHilbertSpace}},AbstractVector{<:AbstractHilbertSpace}}
const _SpaceMapping = Pair{<:_SpaceGroup,<:_SpaceGroup}

"""
    reshape(t, mappings...; repeat=false, skipmissing=false)

Reshape the array `t` by splitting or combining its indices according to
Hilbert-space mappings. The mappings are applied left to right, each one
consuming the next consecutive group of input indices. Each mapping reads
input => output; see [Pair arguments](@ref pair_convention).

# Mapping forms

    H => (H1, H2, ...)   split one index into several
    (H1, H2, ...) => H   combine several consecutive indices into one
    (H1, H2, ...) => (K1, K2, ...)  repartition several indices into several
    H => H               leave one index unchanged

# Specifying multiple mappings

Give one mapping per group of input indices:

    reshape(A, H1 => (H1a, H1b), (H2a, H2b) => H2, H3 => H3)

This splits the first index, combines the next two, and leaves the last
unchanged.

# Single-mapping shorthand

When only one mapping is given, it is replicated if needed:

- If it consumes all indices of `t`, it is applied once:

      reshape(v, H => (H1, H2))        # length-1 vector → 2-index tensor

- If `t` has twice as many indices as the mapping consumes, the mapping
  is applied to the first half and again to the second half. This covers
  the common case of a square operator matrix, where the first index is
  the output space and the second is the input space:

      reshape(m, H => (H1, H2))        # matrix → 4-index tensor
      # equivalent to reshape(m, H => (H1, H2), H => (H1, H2))

      reshape(T, (H1, H2) => H)        # 4-index tensor → matrix
      # equivalent to reshape(T, (H1, H2) => H, (H1, H2) => H)

For asymmetric operators, specify both mappings explicitly:

    reshape(m, Hout => (H1, H2), Hin => (K1, K2))

Many-to-many mappings are evaluated directly and are equivalent to an explicit
combine-then-split composition:

    reshape(A, (H1, H2) => (K1, K2))
    # equivalent to reshape(reshape(A, (H1, H2) => Hmid), Hmid => (K1, K2))

where `Hmid` is the canonical combined space of the input tuple.

When `repeat=true`, the single given mapping is applied to every
consecutive group of input indices. For example, with a 3-index array
where every index lives on `H`:

    reshape(A, H => (H1, H2); repeat=true)
    # equivalent to reshape(A, H => (H1, H2), H => (H1, H2), H => (H1, H2))

# Missing states

Some basis states of the input spaces may have no counterpart in the output
spaces, e.g. when splitting into a truncated space or combining into a
symmetry-constrained space. By default an `ArgumentError` is thrown if `t` has
a nonzero entry on such a state, since that amplitude would be lost. Zero
entries are fine, so e.g. combining unconstrained spaces into a
parity-conserving space works for parity-preserving operators. With
`skipmissing=true`, the components on missing states are silently dropped.

# Precomputed form

    reshape(mappings...; repeat=false, skipmissing=false)

returns a callable `ReshapeMap` that precomputes the transitions, and can be
applied as `op(t)` or in place as `op(out, t)`.
"""
function Base.reshape(t::AbstractArray, mapping::_SpaceMapping, rest::_SpaceMapping...; repeat=false, skipmissing=false)
    mappings = (mapping, rest...)
    mappings = if repeat
        _repeat_mapping(t, mappings)
    elseif length(mappings) == 1
        _expand_single_mapping(t, only(mappings))
    else
        mappings
    end

    Hsins, Hsouts = _validate(t, mappings)
    mappers = map(_mapper, Hsins, Hsouts)
    blocks = map(_transitions, Hsins, Hsouts, mappers)

    return _reshape(t, blocks, Hsouts; skipmissing)
end

struct ReshapeMap{M,P}
    mappings::M
    repeat::Bool
    skipmissing::Bool
    precomputed::P
end

function (op::ReshapeMap)(t::AbstractArray)
    mappings = _resolve_mappings(t, op.mappings; repeat=op.repeat)

    n = length(mappings)
    pre = if length(op.precomputed) == 1
        ntuple(_ -> only(op.precomputed), n)
    elseif length(op.precomputed) == n
        op.precomputed
    else
        throw(ArgumentError("Unexpected resolved mapping count $(length(mappings)) for reshape operation with $(length(op.precomputed)) precomputed mapping(s)."))
    end

    Hsins = map(p -> p.Hsin, pre)
    Hsouts = map(p -> p.Hsout, pre)
    blocks = map(p -> p.blocks, pre)
    _validate_from_spaces(t, Hsins, Hsouts)
    _reshape(t, blocks, Hsouts; op.skipmissing)
end

function (op::ReshapeMap)(out::AbstractArray, t::AbstractArray)
    mappings = _resolve_mappings(t, op.mappings; repeat=op.repeat)

    n = length(mappings)
    pre = if length(op.precomputed) == 1
        ntuple(_ -> only(op.precomputed), n)
    elseif length(op.precomputed) == n
        op.precomputed
    else
        throw(ArgumentError("Unexpected resolved mapping count $(length(mappings)) for reshape operation with $(length(op.precomputed)) precomputed mapping(s)."))
    end

    Hsins = map(p -> p.Hsin, pre)
    Hsouts = map(p -> p.Hsout, pre)
    blocks = map(p -> p.blocks, pre)
    _validate_from_spaces(t, Hsins, Hsouts)
    _reshape!(out, t, blocks, Hsouts; op.skipmissing)
end

function Base.reshape(mapping::_SpaceMapping, rest::_SpaceMapping...; repeat=false, skipmissing=false)
    mappings_t = (mapping, rest...)
    precomputed = Tuple(map(_precompute_mapping, mappings_t))
    ReshapeMap(mappings_t, repeat, skipmissing, precomputed)
end

function _resolve_mappings(t::AbstractArray, mappings; repeat=false)
    if repeat
        _repeat_mapping(t, mappings)
    elseif length(mappings) == 1
        _expand_single_mapping(t, only(mappings))
    else
        mappings
    end
end

function _precompute_mapping(mapping::Pair)
    Hsin, Hsout = _mapping_spaces(mapping)
    mapper = _mapper(Hsin, Hsout)
    blocks = _transitions(Hsin, Hsout, mapper)
    (; mapping, Hsin, Hsout, blocks)
end

_spaces(H::AbstractHilbertSpace) = (H,)
_spaces(Hs::Union{Tuple,AbstractVector}) = Tuple(Hs)
_spaces(x) = throw(ArgumentError("Expected a Hilbert space or a tuple of Hilbert \
                                  spaces, got $(typeof(x))"))

# Split one mapping into (input spaces, output spaces) and check that it is a
# split, a combine, or an identity.
function _mapping_spaces(mapping::Pair)
    Hsin, Hsout = _spaces(first(mapping)), _spaces(last(mapping))

    (isempty(Hsin) || isempty(Hsout)) &&
        throw(ArgumentError("Empty Hilbert space group in mapping $mapping"))

    length(Hsin) == 1 == length(Hsout) && only(Hsin) != only(Hsout) &&
        throw(ArgumentError("A one-to-one mapping must be the identity `H => H`, \
                             got $mapping"))

    return Hsin, Hsout
end

# Check that the mappings describe every index of t, and that the dimensions match.
function _validate(t::AbstractArray, mappings)
    Hsins = map(m -> _mapping_spaces(m)[1], mappings)
    Hsouts = map(m -> _mapping_spaces(m)[2], mappings)

    _validate_from_spaces(t, Hsins, Hsouts)
end

function _validate_from_spaces(t::AbstractArray, Hsins, Hsouts)

    Hsin_all = _flatten(Hsins)
    length(Hsin_all) == ndims(t) ||
        throw(DimensionMismatch("The mappings consume $(length(Hsin_all)) indices, \
                                 but the array has $(ndims(t))"))

    size(t) == map(dim, Hsin_all) ||
        throw(DimensionMismatch("The array has size $(size(t)), but the input \
                                 Hilbert spaces have dimensions $(map(dim, Hsin_all))"))

    return Hsins, Hsouts
end

# A single mapping either covers the whole array, or is applied to out and in indices.
function _expand_single_mapping(t::AbstractArray, mapping::Pair)
    n = length(_mapping_spaces(mapping)[1])
    ndims(t) == n && return (mapping,)
    ndims(t) == 2n && return (mapping, mapping)
    throw(ArgumentError("The mapping $mapping consumes $n indices, but the array has \
                         $(ndims(t)). Give one mapping per index group, or use `repeat=true`."))
end

function _repeat_mapping(t::AbstractArray, mappings)
    length(mappings) == 1 ||
        throw(ArgumentError("`repeat=true` requires exactly one mapping, got $(length(mappings))"))
    n = length(_mapping_spaces(only(mappings))[1])
    ndims(t) % n == 0 ||
        throw(DimensionMismatch("`ndims(t)` must be divisible by $n, got $(ndims(t))"))
    return ntuple(_ -> only(mappings), ndims(t) ÷ n)
end

# _flatten(groups::Tuple) = reduce((a, b) -> (a..., b...), groups; init=())
_flatten(groups::Tuple) = TupleTools.flatten(groups...)
struct _ComposedMapper{M1,M2}
    combine_mapper::M1
    split_mapper::M2
end

function _build_many_to_many_mapper(Hsin, Hsout)
    Hmid = try
        tensor_product(Hsin)
    catch err
        throw(ArgumentError("Cannot infer an intermediate space for many-to-many mapping $Hsin => $Hsout. \
                             Use an explicit two-step mapping `(H1, H2, ...) => H => (K1, K2, ...)`. \
                             Original error: $(sprint(showerror, err))"))
    end
    combine_mapper = state_mapper(Hmid, Hsin)
    split_mapper = state_mapper(Hmid, Hsout)
    return _ComposedMapper(combine_mapper, split_mapper)
end

function _mapper(Hsin, Hsout)
    if length(Hsin) == 1 && length(Hsout) > 1        # split
        state_mapper(only(Hsin), Hsout)
    elseif length(Hsin) > 1 && length(Hsout) == 1    # combine
        state_mapper(only(Hsout), Hsin)
    elseif length(Hsin) > 1 && length(Hsout) > 1     # many-to-many
        _build_many_to_many_mapper(Hsin, Hsout)
    else                                             # identity
        nothing
    end
end
# Each `_transitions` method returns `(; transitions, dropped)`. `transitions` is a vector
# of `(Iin, Iout, weight)` and `dropped` holds the input index tuples `Iin` for which at
# least one transition was discarded because the output state is missing from the output
# spaces (state index 0).
const _OneSpace = Tuple{AbstractHilbertSpace}
const _SeveralSpaces = Tuple{AbstractHilbertSpace,AbstractHilbertSpace,Vararg{AbstractHilbertSpace}}

# Split: H => (H1, H2, ...)
function _transitions(Hsin::_OneSpace, Hsout::_SeveralSpaces, mapper)
    H = only(Hsin)
    transitions = []
    dropped = NTuple{1,Int}[]
    for f in basisstates(H)
        Iin = (state_index(f, H),)
        substates, ws = split_state(f, mapper)
        isdropped = false
        for (states, w) in zip(substates, ws)
            Iout = map(state_index, states, Hsout)
            if any(iszero, Iout)
                isdropped = true
                continue
            end
            push!(transitions, (Iin, Iout, w))
        end
        isdropped && push!(dropped, Iin)
    end
    return (; transitions=map(identity, transitions), dropped)  # narrow the element type
end

# Combine: (H1, H2, ...) => H
function _transitions(Hsin::_SeveralSpaces, Hsout::_OneSpace, mapper)
    H = only(Hsout)
    transitions = []
    dropped = NTuple{length(Hsin),Int}[]
    for fs in Base.product(basisstates.(Hsin)...)
        Iin = map(state_index, fs, Hsin)
        states, ws = combine_states(fs, mapper)
        isdropped = false
        for (f, w) in zip(states, ws)
            Iout = (state_index(f, H),)
            if any(iszero, Iout)
                isdropped = true
                continue
            end
            push!(transitions, (Iin, Iout, w))
        end
        isdropped && push!(dropped, Iin)
    end
    return (; transitions=map(identity, transitions), dropped)
end

# Many-to-many: (H1, H2, ...) => (K1, K2, ...)
function _transitions(Hsin::_SeveralSpaces, Hsout::_SeveralSpaces, mapper::_ComposedMapper)
    transitions = []
    dropped = NTuple{length(Hsin),Int}[]
    for fs in Base.product(basisstates.(Hsin)...)
        Iin = map(state_index, fs, Hsin)
        mid_states, ws1 = combine_states(fs, mapper.combine_mapper)
        isdropped = false
        for (fmid, w1) in zip(mid_states, ws1)
            out_states, ws2 = split_state(fmid, mapper.split_mapper)
            for (states, w2) in zip(out_states, ws2)
                Iout = map(state_index, states, Hsout)
                if any(iszero, Iout)
                    isdropped = true
                    continue
                end
                push!(transitions, (Iin, Iout, w1 * w2))
            end
        end
        isdropped && push!(dropped, Iin)
    end
    return (; transitions=map(identity, transitions), dropped)
end

# Identity: H => H
function _transitions(Hsin::_OneSpace, Hsout::_OneSpace, ::Nothing)
    H = only(Hsin)
    return (; transitions=[((I,), (I,), 1) for I in 1:dim(H)], dropped=NTuple{1,Int}[])
end

_group_length(::AbstractVector{NTuple{N,Int}}) where {N} = N

# Throw if a nonzero entry of `t` sits on an input index whose transitions were partly
# dropped. Only the slices along dropped indices are visited.
function _check_dropped(t::AbstractArray, blocks)
    all(b -> isempty(b.dropped), blocks) && return nothing
    sz = size(t)
    offset = 0
    for b in blocks
        n = _group_length(b.dropped)
        if !isempty(b.dropped)
            # View t as (indices before, this group, indices after)
            gdims = sz[offset+1:offset+n]
            t3 = Base.reshape(t, prod(sz[1:offset]; init=1), prod(gdims), prod(sz[offset+n+1:end]; init=1))
            _check_dropped_group(t3, LinearIndices(gdims), b.dropped)
        end
        offset += n
    end
    return nothing
end
function _check_dropped_group(t3, lin, dropped)
    for Iin in dropped
        any(!iszero, view(t3, :, lin[Iin...], :)) && throw(ArgumentError(
            "Reshape would lose amplitude: the array has nonzero entries on basis states \
             that are missing in the output spaces. Pass `skipmissing=true` to drop them \
             silently."))
    end
end

function _reshape(t::AbstractArray, blocks, Hsouts; skipmissing=false)
    skipmissing || _check_dropped(t, blocks)
    Hsout_all = _flatten(Hsouts)
    tout = zeros(eltype(t), map(dim, Hsout_all)...)

    for transitions in Base.product(map(b -> b.transitions, blocks)...)
        Iin = _flatten(map(tr -> tr[1], transitions))
        tval = t[Iin...]
        iszero(tval) && continue

        Iout = _flatten(map(tr -> tr[2], transitions))
        w = prod(tr -> tr[3], transitions)
        tout[Iout...] += w * tval
    end

    return tout
end

function _reshape!(tout::AbstractArray, t::AbstractArray, blocks, Hsouts; skipmissing=false)
    Hsout_all = _flatten(Hsouts)
    size(tout) == map(dim, Hsout_all) || throw(DimensionMismatch("The output array has size $(size(tout)), but expected $(map(dim, Hsout_all))"))
    skipmissing || _check_dropped(t, blocks)

    fill!(tout, zero(eltype(tout)))
    for transitions in Base.product(map(b -> b.transitions, blocks)...)
        Iin = _flatten(map(tr -> tr[1], transitions))
        tval = t[Iin...]
        iszero(tval) && continue

        Iout = _flatten(map(tr -> tr[2], transitions))
        w = prod(tr -> tr[3], transitions)
        tout[Iout...] += w * tval
    end

    return tout
end

@testitem "Reshape Properties" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: permutation_operator, fermions

    @fermions f

    # ── 2-subsystem property tests across symmetry types ──────────────────────
    for (qn1, qn2, qn3) in [
        (NoSymmetry(), NoSymmetry(), NoSymmetry()),
        (ParityConservation(), ParityConservation(), ParityConservation()),
        (NumberConservation(), NumberConservation(), NumberConservation()),
        (NoSymmetry(), ParityConservation(), NumberConservation()),
    ]
        H1 = hilbert_space(f, [1, 2], qn1)
        H2 = hilbert_space(f, [3, 4], qn2)
        H = hilbert_space(f, 1:4, qn3)
        Hs = (H1, H2)
        d = dim(H)

        # Property 1: Round-trip identity for operators
        m = rand(ComplexF64, d, d)
        @test reshape(reshape(m, H => Hs), Hs => H) ≈ m

        # Property 2: Round-trip identity for state vectors
        v = rand(ComplexF64, d)
        m1 = reshape(v, H => Hs)
        @test reshape(m1, Hs => H) == v
        @test reshape(v, H => reverse(Hs)) == transpose(m1)

        # Property 2b: Callable reshape operation object matches direct call
        to_tensor = reshape(H => Hs)
        to_matrix = reshape(Hs => H)
        @test to_tensor(m) ≈ reshape(m, H => Hs)
        @test to_matrix(to_tensor(m)) ≈ m

        # Property 3: Norm invariance (reshape is an isometry)
        b = fermions(H)
        op = b[1]
        @test norm(op) ≈ norm(reshape(op, H => Hs))

        # Property 4: Multiplication consistency — tensor contraction equals matrix product
        m1, m2 = rand(ComplexF64, d, d), rand(ComplexF64, d, d)
        t1, t2 = reshape(m1, H => Hs), reshape(m2, H => Hs)
        d1, d2 = dim(H1), dim(H2)
        t3 = zeros(ComplexF64, d1, d2, d1, d2)
        for i in 1:d1, j in 1:d2, k in 1:d1, l in 1:d2
            for k1 in 1:d1, k2 in 1:d2
                t3[i, j, k, l] += t1[i, j, k1, k2] * t2[k1, k2, k, l]
            end
        end
        @test reshape(m1 * m2, H => Hs) ≈ t3

        # Property 5: Round-trip survives subsystem reorder (H2, H1)
        @test reshape(reshape(m, H => (H2, H1)), (H2, H1) => H) ≈ m

        # Property 6: Permutation covariance — permutation_operator is consistent with reshape
        P = permutation_operator(H, [H1, H2], [2, 1])
        @test P * P ≈ I
        @test reshape(reshape(P * m * P', H => Hs), Hs => H) ≈ P * m * P'
    end

    # ── Fixed-number sector: NumberConservation(1) ─────────────────────────────
    H_nc = hilbert_space(f, 1:4, NumberConservation())
    Hn1 = sector(1, H_nc)
    H1_ns = hilbert_space(f, [1, 2], NoSymmetry())
    H2_ns = hilbert_space(f, [3, 4], NoSymmetry())
    d_n1 = dim(Hn1)
    m_n1 = rand(ComplexF64, d_n1, d_n1)
    @test reshape(reshape(m_n1, Hn1 => (H1_ns, H2_ns)), (H1_ns, H2_ns) => Hn1) ≈ m_n1

    v_n1 = rand(ComplexF64, d_n1)
    @test reshape(reshape(v_n1, Hn1 => (H1_ns, H2_ns)), (H1_ns, H2_ns) => Hn1) ≈ v_n1

    # ── 3-subsystem: round-trip and subsystem reorder ─────────────────────────
    H1_3 = hilbert_space(f, [1], NoSymmetry())
    H2_3 = hilbert_space(f, [2], NoSymmetry())
    H3_3 = hilbert_space(f, [3], NoSymmetry())
    H_3 = tensor_product(H1_3, H2_3, H3_3)
    d_3 = dim(H_3)
    m3 = rand(ComplexF64, d_3, d_3)

    # ── 4-subsystem: rectangular operator, mixed reshape, identity ───────────
    H4_3 = hilbert_space(f, [4], NoSymmetry())
    H12_3 = tensor_product(H1_3, H2_3)
    H34_3 = tensor_product(H3_3, H4_3)

    # Rectangular operator: different out/in composite spaces
    R = rand(ComplexF64, dim(H12_3), dim(H34_3))
    RT = reshape(R, H12_3 => (H1_3, H2_3), H34_3 => (H3_3, H4_3))
    @test size(RT) == (dim(H1_3), dim(H2_3), dim(H3_3), dim(H4_3))
    @test reshape(RT, (H1_3, H2_3) => H12_3, (H3_3, H4_3) => H34_3) ≈ R

    # Mixed split and combine on a general 3-index tensor
    A = rand(ComplexF64, dim(H12_3), dim(H3_3), dim(H4_3))
    B = reshape(A, H12_3 => (H1_3, H2_3), (H3_3, H4_3) => H34_3)
    @test size(B) == (dim(H1_3), dim(H2_3), dim(H34_3))
    @test reshape(B, (H1_3, H2_3) => H12_3, H34_3 => (H3_3, H4_3)) ≈ A

    # Direct many-to-many mapping is equivalent to explicit two-step composition
    H13_3 = tensor_product(H1_3, H3_3)
    H24_3 = tensor_product(H2_3, H4_3)
    H1234 = tensor_product(H12_3, H34_3)
    M24 = rand(ComplexF64, dim(H12_3), dim(H34_3))
    M24_direct = reshape(M24, (H12_3, H34_3) => (H13_3, H24_3))
    M24_twostep = reshape(reshape(M24, (H12_3, H34_3) => H1234), H1234 => (H13_3, H24_3))
    @test M24_direct ≈ M24_twostep

    # Inverse many-to-many map restores the original matrix
    @test reshape(M24_direct, (H13_3, H24_3) => (H12_3, H34_3)) ≈ M24

    # Identity mappings leave the tensor unchanged
    @test reshape(A, H12_3 => H12_3, H3_3 => H3_3, H4_3 => H4_3) == A

    # repeat=true applies one mapping to every index group
    A3 = rand(ComplexF64, dim(H12_3), dim(H12_3), dim(H12_3))
    @test reshape(A3, H12_3 => (H1_3, H2_3); repeat=true) ≈
          reshape(A3, H12_3 => (H1_3, H2_3), H12_3 => (H1_3, H2_3),
        H12_3 => (H1_3, H2_3))
    to_split_repeat = reshape(H12_3 => (H1_3, H2_3); repeat=true)
    @test to_split_repeat(A3) ≈ reshape(A3, H12_3 => (H1_3, H2_3); repeat=true)

    # Standard ordering round-trip with ndims check
    t3 = reshape(m3, H_3 => (H1_3, H2_3, H3_3))
    @test ndims(t3) == 6
    @test m3 ≈ reshape(t3, (H1_3, H2_3, H3_3) => H_3)

    # Permuted subsystem orderings round-trip
    @test m3 ≈ reshape(reshape(m3, H_3 => (H1_3, H3_3, H2_3)), (H1_3, H3_3, H2_3) => H_3)
    @test m3 ≈ reshape(reshape(m3, H_3 => (H3_3, H2_3, H1_3)), (H3_3, H2_3, H1_3) => H_3)
end

@testitem "Mixed reshape mapping check" begin
    @fermions f
    # Build elementary spaces
    H1a = hilbert_space(f, [1], NoSymmetry())
    H1b = hilbert_space(f, [2], NumberConservation())
    H2a = hilbert_space(f, [3], NoSymmetry())
    H2b = hilbert_space(f, [4], ParityConservation())
    H3a = hilbert_space(f, [5], NoSymmetry())
    H3b = hilbert_space(f, [6], NumberConservation())
    H3c = hilbert_space(f, [7], NoSymmetry())
    H4 = hilbert_space(f, [8], NoSymmetry())

    # Composite spaces used in the mapping
    H1ab = tensor_product(H1a, H1b)
    H2ab = tensor_product(H2a, H2b)
    H3bc = tensor_product(H3b, H3c)
    H3ab = tensor_product(H3a, H3b)

    # Input tensor has indices: (H1ab, H2a, H2b, H3a, H3bc, H4)
    A = rand(ComplexF64,
        dim(H1ab), dim(H2a), dim(H2b), dim(H3a), dim(H3bc), dim(H4)
    )

    B = reshape(A,
        H1ab => (H1a, H1b),
        (H2a, H2b) => H2ab,
        (H3a, H3bc) => (H3ab, H3c),
        H4 => H4,
    )
    rmap = reshape(H1ab => (H1a, H1b), (H2a, H2b) => H2ab, (H3a, H3bc) => (H3ab, H3c), H4 => H4)
    @test B ≈ rmap(A)

    # Expected output indices: (H1a, H1b, H2ab, H3ab, H3c, H4)
    @test size(B) == (dim(H1a), dim(H1b), dim(H2ab), dim(H3ab), dim(H3c), dim(H4))

    # Inverse mapping should recover A
    A_rt = reshape(B,
        (H1a, H1b) => H1ab,
        H2ab => (H2a, H2b),
        (H3ab, H3c) => (H3a, H3bc),
        H4 => H4,
    )
    @test A_rt ≈ A

    # Optional: direct many-to-many equals explicit two-step composition
    H3abc = tensor_product(H3a, H3bc)
    B_twostep = reshape(
        reshape(A,
            H1ab => (H1a, H1b),
            (H2a, H2b) => H2ab,
            (H3a, H3bc) => H3abc,
            H4 => H4,
        ),
        H1a => H1a,
        H1b => H1b,
        H2ab => H2ab,
        H3abc => (H3ab, H3c),
        H4 => H4,
    )
    @test B_twostep ≈ B
end

@testitem "Reshape missing states" begin
    using LinearAlgebra, Random
    Random.seed!(1)
    @fermions f

    # Splitting into an even-parity subsystem drops the odd states of f[1:2]
    H = hilbert_space(f, 1:4)
    Ha = hilbert_space(f, 1:2)
    Hae = hilbert_space(f, 1:2, ParityConservation(1))
    Hb = hilbert_space(f, 3:4)
    even = [state_index(s, Ha) for s in basisstates(Hae)]
    v = rand(ComplexF64, dim(H))
    m = rand(ComplexF64, dim(H), dim(H))

    @test_throws ArgumentError reshape(v, H => (Hae, Hb))
    @test_throws ArgumentError reshape(m, H => (Hae, Hb))
    vs = reshape(v, H => (Hae, Hb); skipmissing=true)
    @test size(vs) == (2, 4)
    @test norm(vs) < norm(v)
    # Agrees with going through the full subsystem and projecting explicitly
    vfull = reshape(v, H => (Ha, Hb))
    @test vs ≈ vfull[even, :]
    ms = reshape(m, H => (Hae, Hb); skipmissing=true)
    @test size(ms) == (2, 4, 2, 4)

    # Entries that vanish on the missing states are fine
    veven = zero(vfull)
    veven[even, :] = vfull[even, :]
    @test reshape(reshape(veven, (Ha, Hb) => H), H => (Hae, Hb)) ≈ vfull[even, :]

    # Precomputed form, both call methods
    rmap = reshape(H => (Hae, Hb))
    @test_throws ArgumentError rmap(m)
    @test_throws ArgumentError rmap(zeros(ComplexF64, 2, 4, 2, 4), m)
    rmap_skip = reshape(H => (Hae, Hb); skipmissing=true)
    @test rmap_skip(m) == ms
    @test rmap_skip(zeros(ComplexF64, 2, 4, 2, 4), m) == ms

    # Many-to-many with missing states
    @test_throws ArgumentError reshape(vfull, (Ha, Hb) => (Hb, Hae))
    @test reshape(vfull, (Ha, Hb) => (Hb, Hae); skipmissing=true) ≈ transpose(vs)
    @test reshape(veven, (Ha, Hb) => (Hb, Hae)) ≈ transpose(vs)

    # Combining unconstrained spaces into an even-parity space
    H1 = hilbert_space(f, 1:2)
    H2 = hilbert_space(f, 3:4)
    He = hilbert_space(f, 1:4, ParityConservation(1))
    ψ = normalize(rand(ComplexF64, dim(He)))
    ρ = ψ * ψ'
    T = reshape(ρ, He => (H1, H2))
    @test reshape(T, (H1, H2) => He) ≈ ρ
    @test reshape((H1, H2) => He)(T) ≈ ρ
    @test reshape((H1, H2) => He)(similar(ρ), T) ≈ ρ
    # Generic operator on the full space has weight outside the even sector
    G = rand(ComplexF64, dim(H1), dim(H2), dim(H1), dim(H2))
    @test_throws ArgumentError reshape(G, (H1, H2) => He)
    @test_throws ArgumentError reshape((H1, H2) => He)(G)
    @test_throws ArgumentError reshape((H1, H2) => He)(similar(ρ), G)
    @test reshape(G, (H1, H2) => He; skipmissing=true) ≈
          reshape((H1, H2) => He; skipmissing=true)(G)
end

@testitem "Reshape missing states: truncated bosons" begin
    using LinearAlgebra, Random
    Random.seed!(1)
    @bosons b
    @fermions f

    # Truncated boson: 3 levels -> 2 levels drops the top level
    Hb3 = hilbert_space(b[1], 3)
    Hb2 = hilbert_space(b[1], 2)
    Hb4 = hilbert_space(b[1], 4)
    Hf = hilbert_space(f, 1:2)
    H = tensor_product(Hb3, Hf)
    v = rand(ComplexF64, dim(H))
    m = rand(ComplexF64, dim(H), dim(H))

    @test_throws ArgumentError reshape(v, H => (Hb2, Hf))
    @test_throws ArgumentError reshape(m, H => (Hb2, Hf))
    vs = reshape(v, H => (Hb2, Hf); skipmissing=true)
    @test size(vs) == (2, 4)
    @test norm(vs) < norm(v)
    # Agrees with going through the full space and truncating explicitly
    v3 = reshape(v, H => (Hb3, Hf))
    @test vs ≈ v3[1:2, :]
    ms = reshape(m, H => (Hb2, Hf); skipmissing=true)
    @test size(ms) == (2, 4, 2, 4)

    # Entries that vanish on the missing states are fine
    v_low = reshape(vcat(v3[1:2, :], zeros(1, 4)), (Hb3, Hf) => H)
    @test reshape(v_low, H => (Hb2, Hf)) ≈ v3[1:2, :]

    # Embedding 3 -> 4 drops nothing
    v4 = reshape(v, H => (Hb4, Hf))
    @test v4[1:3, :] ≈ v3
    @test iszero(v4[4, :])
    @test norm(v4) ≈ norm(v)

    # Precomputed form, both call methods
    rmap = reshape(H => (Hb2, Hf))
    @test_throws ArgumentError rmap(m)
    @test_throws ArgumentError rmap(zeros(ComplexF64, 2, 4, 2, 4), m)
    rmap_skip = reshape(H => (Hb2, Hf); skipmissing=true)
    @test rmap_skip(m) == ms
    @test rmap_skip(zeros(ComplexF64, 2, 4, 2, 4), m) == ms
    @test reshape(H => (Hb4, Hf))(v) == v4

    # Many-to-many with missing states
    @test_throws ArgumentError reshape(v3, (Hb3, Hf) => (Hf, Hb2))
    @test reshape(v3, (Hb3, Hf) => (Hf, Hb2); skipmissing=true) ≈ transpose(vs)
    @test reshape(v3, (Hb3, Hf) => (Hf, Hb4)) ≈ transpose(v4)
end
