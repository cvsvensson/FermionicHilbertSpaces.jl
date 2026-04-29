
function Base.reshape(m::AbstractMatrix, H::AbstractHilbertSpace, Hs)
    mapper = state_mapper(H, Hs)
    _reshape_mat_to_tensor(m, H, Hs, Hs, mapper, mapper)
end
function Base.reshape(m::AbstractMatrix, H::AbstractHilbertSpace, Hsout, Hsin)
    mapperout = state_mapper(H, Hsout)
    mapperin = state_mapper(H, Hsin)
    _reshape_mat_to_tensor(m, H, Hsout, Hsin, mapperout, mapperin)
end
function Base.reshape(m::AbstractVector, H::AbstractHilbertSpace, Hs)
    _reshape_vec_to_tensor(m, H, Hs, state_mapper(H, Hs))
end
const PairWithHilbertSpace = Union{Pair{<:AbstractHilbertSpace,<:Any},Pair{<:Any,<:AbstractHilbertSpace}}
Base.reshape(Hs::PairWithHilbertSpace; kwargs...) = m -> reshape(m, first(Hs), last(Hs); kwargs...)
Base.reshape(m::AbstractArray, Hs::PairWithHilbertSpace; kwargs...) = reshape(m, first(Hs), last(Hs); kwargs...)

function Base.reshape(t::AbstractArray, Hs::Union{<:AbstractVector,Tuple}, H::AbstractHilbertSpace)
    mapper_in = state_mapper(H, Hs)
    if ndims(t) == 2 * length(Hs)
        return _reshape_tensor_to_mat(t, (Hs, mapper_in), (Hs, mapper_in), H, state_mapper)
    elseif ndims(t) == length(Hs)
        return _reshape_tensor_to_vec(t, Hs, H, mapper_in)
    else
        throw(ArgumentError("The number of dimensions in the tensor must match the number of subsystems"))
    end
end

function _reshape_vec_to_tensor(v::AbstractVector, H::AbstractHilbertSpace, Hs, mapper)
    dims = map(dim, Hs)
    fs = basisstates(H)
    Is = map(f -> state_index(f, H), fs)
    t = zeros(eltype(v), dims...)
    for (f, I) in zip(fs, Is)
        states, amps = split_state(f, mapper)
        for (substates, w) in zip(states, amps)
            Iout = state_index.(substates, Hs)
            t[Iout...] += w * v[I]
        end
    end
    return t
end

function _reshape_mat_to_tensor(m::AbstractMatrix, H::AbstractHilbertSpace, Hsout, Hsin, mapperout, mapperin)
    #reshape the matrix m in basis b into a tensor where each index pair has a basis in bs
    dimsin = map(dim, Hsin)
    dimsout = map(dim, Hsout)
    fs = basisstates(H)
    Js = map(f -> state_index(f, H), fs)
    t = zeros(eltype(m), dimsout..., dimsin...)
    for (J1, f1) in zip(Js, fs)
        substatesout, wsout = split_state(f1, mapperout)
        for (substatesout, wout) in zip(substatesout, wsout)
            Iout = map(state_index, substatesout, Hsout)
            for (J2, f2) in zip(Js, fs)
                substatesin, wsin = split_state(f2, mapperin)
                for (substatesin, win) in zip(substatesin, wsin)
                    Iin = map(state_index, substatesin, Hsin)
                    t[Iout..., Iin...] += wout * win * m[J1, J2]
                end
            end
        end
    end
    return t
end

function _reshape_tensor_to_mat(t, (Hsout, mapperout), (Hsin, mapperin), H::AbstractHilbertSpace, state_mapper)
    fsout = Base.product(basisstates.(Hsout)...)
    fsin = Base.product(basisstates.(Hsin)...)

    Jouts = map(f -> state_index.(f, Hsout), fsout)
    Jins = map(f -> state_index.(f, Hsin), fsin)

    m = zeros(eltype(t), dim(H), dim(H))
    for (fsin_tuple, Jin) in zip(fsin, Jins)
        states_in, amps_in = combine_states(fsin_tuple, mapperin)
        for (fullf_in, win) in zip(states_in, amps_in)
            Iin = state_index(fullf_in, H)
            for (fsout_tuple, Jout) in zip(fsout, Jouts)
                states_out, amps_out = combine_states(fsout_tuple, mapperout)
                for (fullf_out, wout) in zip(states_out, amps_out)
                    Iout = state_index(fullf_out, H)
                    tval = t[Jout..., Jin...]
                    iszero(tval) && continue
                    m[Iout, Iin] += wout * win * tval
                end
            end
        end
    end
    return m
end

function _reshape_tensor_to_vec(t, Hs, H::AbstractHilbertSpace, state_mapper)
    fs = Base.product(basisstates.(Hs)...)
    v = Vector{eltype(t)}(undef, dim(H))
    fill!(v, zero(eltype(v)))
    for fstuple in fs
        Is = state_index.(fstuple, Hs)
        substates, amps = combine_states(fstuple, state_mapper)
        for (substate, amp) in zip(substates, amps)
            Iout = state_index(substate, H)
            ismissing(Iout) && continue
            v[Iout] += amp * t[Is...]
        end
    end
    return v
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
