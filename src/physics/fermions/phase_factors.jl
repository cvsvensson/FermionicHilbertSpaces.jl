
##https://iopscience.iop.org/article/10.1088/1751-8121/ac0646/pdf (10c)
function phase_factor_f(focknbr1, focknbr2, subinds::NTuple)::Int
    bitmask = focknbr_from_site_indices(subinds)
    pf = 1
    for i in subinds
        pf *= _phase_factor_f(bitmask & focknbr1, bitmask & focknbr2, i)
    end
    return pf
end
function phase_factor_f(focknbr1, focknbr2, N::Int)::Int
    pf = 1
    for i in 1:N
        pf *= _phase_factor_f(focknbr1, focknbr2, i)
    end
    return pf
end
function _phase_factor_f(focknbr1, focknbr2, i::Int)::Int
    _bit(focknbr2, i) ? (jwstring_anti(i, focknbr1) * jwstring_anti(i, focknbr2)) : 1
end

function kron_phase_factor(state_mapper::FockMapper)
    inds = state_mapper.fermionpositions
    N = sum(length, inds)
    T = FockNumber{default_fock_representation(N)}
    masks = map(Xp -> T(focknbr_from_site_indices(Xp)), inds)
    function pfh(f1, f2)
        phase_factor_h(f1, f2, inds, masks)
    end
end

function phase_factor_h(f1::AbstractFockState, f2::AbstractFockState, partition, masks=map(focknbr_from_site_indices, partition))::Int
    # this assumes that partition is a list of list of indices
    # masks is a list of bitmasks corresponding to the partition
    #(120b)
    phase = 1
    for (Xp, Xpmask) in zip(partition, masks)
        masked_f1 = Xpmask & f1
        masked_f2 = Xpmask & f2
        for X in partition
            if X === Xp
                continue
            end
            for i in X
                if _bit(f2, i)
                    phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
                end
            end
        end
    end
    return phase
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
