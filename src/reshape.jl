"""
    reshape(t, mappings...; repeat=false)

Reshape the array `t` by splitting or combining its indices according to
Hilbert-space mappings. The mappings are applied left to right, each one
consuming the next consecutive group of input indices.

# Mapping forms

    H => (H1, H2, ...)   split one index into several
    (H1, H2, ...) => H   combine several consecutive indices into one
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

When `repeat=true`, the single given mapping is applied to every
consecutive group of input indices. For example, with a 3-index array
where every index lives on `H`:

    reshape(A, H => (H1, H2); repeat=true)
    # equivalent to reshape(A, H => (H1, H2), H => (H1, H2), H => (H1, H2))

"""
function Base.reshape(t::AbstractArray, mappings::PairWithHilbertSpace...; repeat=false)
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

    return _reshape(t, blocks, Hsouts)
end

Base.reshape(mappings::PairWithHilbertSpace...; kwargs...) = t -> reshape(t, mappings...; kwargs...)

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

    length(Hsin) > 1 && length(Hsout) > 1 &&
        throw(ArgumentError("Many-to-many mappings are not supported: $mapping. \
                             Use `(H1, H2) => H` followed by `H => (K1, K2)`."))

    length(Hsin) == 1 == length(Hsout) && only(Hsin) != only(Hsout) &&
        throw(ArgumentError("A one-to-one mapping must be the identity `H => H`, \
                             got $mapping"))

    return Hsin, Hsout
end

# Check that the mappings describe every index of t, and that the dimensions match.
function _validate(t::AbstractArray, mappings)
    Hsins = map(m -> _mapping_spaces(m)[1], mappings)
    Hsouts = map(m -> _mapping_spaces(m)[2], mappings)

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
                         $(ndims(t)). Give one mapping per index group, or use repeat=true."))
end

function _repeat_mapping(t::AbstractArray, mappings)
    length(mappings) == 1 ||
        throw(ArgumentError("repeat=true requires exactly one mapping"))
    n = length(_mapping_spaces(only(mappings))[1])
    ndims(t) % n == 0 ||
        throw(DimensionMismatch("ndims(t) = $(ndims(t)) is not divisible by $n"))
    return ntuple(_ -> only(mappings), ndims(t) ÷ n)
end

_flatten(groups::Tuple) = reduce((a, b) -> (a..., b...), groups; init=())
function _mapper(Hsin, Hsout)
    if length(Hsin) == 1 && length(Hsout) > 1        # split
        state_mapper(only(Hsin), Hsout)
    elseif length(Hsin) > 1 && length(Hsout) == 1    # combine
        state_mapper(only(Hsout), Hsin)
    else                                             # identity
        nothing
    end
end
# Split: H => (H1, H2, ...)
function _transitions(Hsin::Tuple{<:AbstractHilbertSpace}, Hsout, mapper)
    H = only(Hsin)
    transitions = []
    for f in basisstates(H)
        Iin = (state_index(f, H),)
        substates, ws = split_state(f, mapper)
        for (states, w) in zip(substates, ws)
            Iout = map(state_index, states, Hsout)
            any(iszero, Iout) && continue
            push!(transitions, (Iin, Iout, w))
        end
    end
    return map(identity, transitions)  # narrow the element type
end

# Combine: (H1, H2, ...) => H
function _transitions(Hsin, Hsout::Tuple{<:AbstractHilbertSpace}, mapper)
    H = only(Hsout)
    transitions = []
    for fs in Base.product(basisstates.(Hsin)...)
        Iin = map(state_index, fs, Hsin)
        states, ws = combine_states(fs, mapper)
        for (f, w) in zip(states, ws)
            Iout = (state_index(f, H),)
            any(iszero, Iout) && continue
            push!(transitions, (Iin, Iout, w))
        end
    end
    return map(identity, transitions)
end

# Identity: H => H
function _transitions(Hsin::Tuple{<:AbstractHilbertSpace}, Hsout::Tuple{<:AbstractHilbertSpace}, ::Nothing)
    H = only(Hsin)
    return [((I,), (I,), 1) for I in 1:dim(H)]
end
function _reshape(t::AbstractArray, blocks, Hsouts)
    Hsout_all = _flatten(Hsouts)
    tout = zeros(eltype(t), map(dim, Hsout_all)...)

    for transitions in Base.product(blocks...)
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

    # Identity mappings leave the tensor unchanged
    @test reshape(A, H12_3 => H12_3, H3_3 => H3_3, H4_3 => H4_3) == A

    # repeat=true applies one mapping to every index group
    A3 = rand(ComplexF64, dim(H12_3), dim(H12_3), dim(H12_3))
    @test reshape(A3, H12_3 => (H1_3, H2_3); repeat=true) ≈
          reshape(A3, H12_3 => (H1_3, H2_3), H12_3 => (H1_3, H2_3),
        H12_3 => (H1_3, H2_3))

    # Standard ordering round-trip with ndims check
    t3 = reshape(m3, H_3 => (H1_3, H2_3, H3_3))
    @test ndims(t3) == 6
    @test m3 ≈ reshape(t3, (H1_3, H2_3, H3_3) => H_3)

    # Permuted subsystem orderings round-trip
    @test m3 ≈ reshape(reshape(m3, H_3 => (H1_3, H3_3, H2_3)), (H1_3, H3_3, H2_3) => H_3)
    @test m3 ≈ reshape(reshape(m3, H_3 => (H3_3, H2_3, H1_3)), (H3_3, H2_3, H1_3) => H_3)
end

function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}) where {N,NL}
    rightindices::NTuple{N - NL,Int} = Tuple(setdiff(ntuple(identity, N), leftindices))
    reshape_to_matrix(t, leftindices, rightindices)
end
function reshape_to_matrix(t::AbstractArray{<:Any,N}, leftindices::NTuple{NL,Int}, rightindices::NTuple{NR,Int}) where {N,NL,NR}
    @assert NL + NR == N
    tperm = permutedims(t, (leftindices..., rightindices...))
    lsize = prod(i -> size(t, i), leftindices, init=1)
    rsize = prod(i -> size(t, i), rightindices, init=1)
    reshape(tperm, lsize, rsize)
end
