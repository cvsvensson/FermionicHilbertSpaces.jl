
##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
function phase_factor_f(focknbr1, focknbr2, subinds::NTuple)
    bitmask = focknbr_from_site_indices(subinds)
    pf = false
    masked_f1 = bitmask & focknbr1
    masked_f2 = bitmask & focknbr2
    for i in subinds
        pf ⊻= _phase_factor_f_bool(masked_f1, masked_f2, i)
    end
    return pf ? -1 : 1
end
function phase_factor_f(f1::FockNumber, f2::FockNumber, N::Int)
    _pair_parity(f1.f ⊻ f2.f, f2.f) ? -1 : 1
end

function phase_factor_f(focknbr1, focknbr2, N::Int)
    #Fall back which works for FixedNumberFockState
    pf = false
    for i in 1:N
        pf ⊻= _phase_factor_f_bool(focknbr1, focknbr2, i)
    end
    return pf ? -1 : 1
end
function _phase_factor_f_bool(focknbr1, focknbr2, i::Int)
    _bit(focknbr2, i) ? (jwstring_anti_bool(i, focknbr1) ⊻ jwstring_anti_bool(i, focknbr2)) : false
end

function kron_phase_factor(state_mapper::FockMapper{N}) where N
    inds = state_mapper.fermionpositions
    T = FockNumber{default_fock_representation(N)}
    masks = map(Xp -> T(focknbr_from_site_indices(Xp)), inds)
    function pfh(f1, f2)
        phase_factor_h(f1, f2, inds, masks)
    end
end

"""
Parity of #{(i,j) : bit j set in `a`, bit i set in `b`, j > i (0-indexed positions)}.
Returns true for odd count (contributes phase -1).
"""
function _pair_parity(a::T, b::T) where {T<:Integer}
    parity = false
    tmp = a
    while tmp != 0
        j = trailing_zeros(tmp)
        tmp &= tmp - 1
        parity ⊻= isodd(count_ones(b & ((one(T) << j) - one(T))))
    end
    return parity
end

function phase_factor_h(f1::FockNumber, f2::FockNumber,
    partition, masks=map(focknbr_from_site_indices, partition))::Int
    g_bits = f1.f ⊻ f2.f
    f2_bits = f2.f
    # All pairs across the full system
    parity = _pair_parity(g_bits, f2_bits)
    # Subtract same-partition pairs (XOR = add mod 2)
    for mask in masks
        m = mask.f
        parity ⊻= _pair_parity(g_bits & m, f2_bits & m)
    end
    return parity ? -1 : 1
end


@testitem "Phase factor f" begin
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, bits

    ## Appendix A.2
    N = 2
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test [phase_factor_f(f1, f2, N) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]

    N = 3
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test [phase_factor_f(f1, f2, N) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1 1 -1 -1 -1;
        1 1 -1 1 -1 1 -1 -1;
        1 1 1 -1 -1 1 1 1;
        1 1 -1 1 1 -1 1 1;
        1 1 1 -1 1 -1 -1 -1;
        1 1 -1 1 -1 1 -1 -1;
        1 1 1 -1 -1 1 1 1;
        1 1 -1 1 1 -1 1 1]
end

@testitem "Phase factor h" begin
    # Appendix B.1
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, bits
    @fermions a
    N = 2
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    h(p, fockstates) = [phase_factor_h(f1, f2, p) for f1 in fockstates, f2 in fockstates]

    import FermionicHilbertSpaces: state_mapper, kron_phase_factor
    h_mapper(p, fockstates, N) = begin
        H = hilbert_space(a, 1:N)
        Hs = map(p) do part
            hilbert_space(a, part)
        end
        mapper = state_mapper(H, Hs)
        _h = kron_phase_factor(mapper)
        [_h(f1, f2) for f1 in fockstates, f2 in fockstates]
    end

    result = [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]
    @test h([[1], [2]], fockstates) == result
    @test h_mapper([[1], [2]], fockstates, N) == result

    phf(f1, f2, subinds, N) = prod(s -> phase_factor_f(f1, f2, s), subinds) * phase_factor_f(f1, f2, N)
    phf(fockstates, subinds, N) = [phf(f1, f2, subinds, N) for f1 in fockstates, f2 in fockstates]
    let part = [[1], [2]]
        @test h(part, fockstates) == phf(fockstates, map(Tuple, part), N) == h_mapper(part, fockstates, N)
    end
    #
    N = 3
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    result = [1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1;
        1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1]
    @test h([[1, 3], [2]], fockstates) == result
    @test h_mapper([[1, 3], [2]], fockstates, N) == result

    partitions = [[[1], [2], [3]], [[2], [1], [3]], [[2], [3], [1]],
        [[1, 3], [2]], [[2, 3], [1]], [[3, 2], [1]], [[2, 1], [3]],
        [[1], [2, 3]], [[3], [2, 1]], [[2], [1, 3]],
        [[1, 2, 3]], [[2], [1], [3]]]
    for p in partitions
        if all(issorted, p)
            # h_mapper uses FockMapper which checks the ordering 
            @test h(p, fockstates) == phf(fockstates, map(Tuple, p), N) == h_mapper(p, fockstates, N)
        else
            @test h(p, fockstates) == phf(fockstates, map(Tuple, p), N)
        end
    end

    N = 7
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    partitions = [[[3, 2, 7, 5, 1], [4, 6]], [[7, 3, 2], [1, 5], [4, 6]]]
    for p in partitions
        if all(issorted, p)
            # h_mapper uses FockMapper which checks the ordering 
            @test h(p, fockstates) == phf(fockstates, map(Tuple, p), N) == h_mapper(p, fockstates, N)
        else
            @test h(p, fockstates) == phf(fockstates, map(Tuple, p), N)
        end
    end

end

function phase_factor_l(f1, f2, X, Xbar)::Int
    #X and Xbar are list of indices
    #(123b)
    phase = 1
    Xmask = focknbr_from_site_indices(X)
    masked_f1 = Xmask & f1
    masked_f2 = Xmask & f2
    for i in Xbar
        if xor(_bit(f1, i), _bit(f2, i))
            phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
        end
    end
    return phase
end
function phase_factor_l(f1, f2, partition)::Int
    #(126b)
    X = partition
    phase = 1
    for (s, Xs) in enumerate(X)
        for Xr in Iterators.drop(X, s)
            mask = focknbr_from_site_indices(Iterators.flatten((Xs, Xr)))
            phase *= phase_factor_l(mask & f1, mask & f2, Xs, Xr)
        end
    end
    return phase
end

@testitem "Phase factor l" begin
    # Appendix B.6
    import FermionicHilbertSpaces: phase_factor_l, bits
    N = 2
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    lX(p1, p2, fockstates) = [phase_factor_l(f1, f2, p1, p2) for f1 in fockstates, f2 in fockstates]
    lξ(p, fockstates) = [phase_factor_l(f1, f2, p) for f1 in fockstates, f2 in fockstates]

    p = [[1], [2]]
    @test lX(p..., fockstates) == lξ(p, fockstates) == [1 1 1 1;
              1 1 1 1;
              1 1 1 1;
              1 1 1 1]

    p = [[2], [1]]
    @test lX(p..., fockstates) == lξ(p, fockstates) == [1 1 1 -1;
              1 1 -1 1;
              1 -1 1 1;
              -1 1 1 1]
    #
    N = 3
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))

    p = [[1], [2, 3]]
    @test lX(p..., fockstates) == lξ(p, fockstates) == ones(Int, 2^N, 2^N)

    p = [[1, 2], [3]]
    @test lX(p..., fockstates) == lξ(p, fockstates) == ones(Int, 2^N, 2^N)

    p = [[2], [1, 3]]
    @test lX(p..., fockstates) == lξ(p, fockstates) ==
          [1 1 1 1 1 1 -1 -1;
              1 1 1 1 1 1 -1 -1;
              1 1 1 1 -1 -1 1 1;
              1 1 1 1 -1 -1 1 1;
              1 1 -1 -1 1 1 1 1;
              1 1 -1 -1 1 1 1 1;
              -1 -1 1 1 1 1 1 1;
              -1 -1 1 1 1 1 1 1]

    @test lξ([[1], [2], [3]], fockstates) == ones(Int, 2^N, 2^N)
    @test lξ([[2], [1], [3]], fockstates) == lξ([[2], [1, 3]], fockstates)
    @test lξ([[2], [3], [1]], fockstates) == lξ([[2, 3], [1]], fockstates) == lX([2, 3], [1], fockstates)
end

## Phase factor u
function phase_factor_u(partition, masks, state::FockNumber)
    phase = false
    for (s, (Xs, mask)) in enumerate(zip(partition, masks))
        for (r, Xr) in Iterators.drop(enumerate(partition), s)
            for i in Xr
                if _bit(state, i)
                    phase ⊻= jwstring_anti_bool(i, mask & state)
                end
            end
        end
    end
    return phase ? -1 : 1
end

phase_factor_u(::AtomicStateMapper) = state -> 1
function phase_factor_u(state_mapper::ProductSpaceMapper)
    mappers = state_mapper.factor_mappers
    phase_factor_maps = map(phase_factor_u, mappers)
    function phase_factor(state)
        pf = 1
        for (s, mapper, pfu) in zip(state.states, mappers, phase_factor_maps)
            isnothing(mapper) && continue
            pf *= pfu(s)
        end
        return pf
    end
end
function phase_factor_u(state_mapper::FockMapper{N}) where N
    inds = state_mapper.fermionpositions
    T = FockNumber{default_fock_representation(N)}
    masks = map(Xp -> T(focknbr_from_site_indices(Xp)), inds)
    function pfu(f1)
        phase_factor_u(inds, masks, f1)
    end
end


@testitem "Phase factor u" begin
    # Appendix C.4
    import FermionicHilbertSpaces: phase_factor_u, bits, focknbr_from_site_indices
    N = 2
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    # uX(p1, p2, fockstates) = [phase_factor_u(f1, p1, p2) for f1 in fockstates]
    uξ(p, fockstates) = [phase_factor_u(p, map(focknbr_from_site_indices, p), f1) for f1 in fockstates]

    p = [[1], [2]]
    @test uξ(p, fockstates) == [1, 1, 1, 1]

    p = [[2], [1]]
    @test uξ(p, fockstates) == [1, 1, 1, -1]

    #
    N = 3
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))

    p = [[1], [2, 3]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, 1, 1, 1]

    p = [[2, 3], [1]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, -1, -1, 1]

    p = [[2], [1, 3]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, 1, -1, -1]

    p = [[1, 3], [2]]
    @test uξ(p, fockstates) == [1, 1, 1, -1, 1, 1, 1, -1]

    p = [[3], [1, 2]]
    @test uξ(p, fockstates) == [1, 1, 1, -1, 1, -1, 1, 1]

    p = [[1, 2], [3]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, 1, 1, 1]

    p = [[1], [2], [3]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, 1, 1, 1]

    p = [[1], [3], [2]]
    @test uξ(p, fockstates) == [1, 1, 1, -1, 1, 1, 1, -1]

    p = [[2], [1], [3]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, 1, -1, -1]

    p = [[3], [1], [2]]
    @test uξ(p, fockstates) == [1, 1, 1, -1, 1, -1, 1, 1]

    p = [[2], [3], [1]]
    @test uξ(p, fockstates) == [1, 1, 1, 1, 1, -1, -1, 1]

    p = [[3], [2], [1]]
    @test uξ(p, fockstates) == [1, 1, 1, -1, 1, -1, -1, -1]

end
