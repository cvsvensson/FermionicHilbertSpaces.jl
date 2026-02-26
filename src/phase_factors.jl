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

function consistent_ordering(subsystem, jw::JordanWignerOrdering)::Bool
    lastpos = 0
    for label in subsystem
        haskey(jw.ordering, label) || return false
        newpos = jw.ordering[label]
        newpos > lastpos || return false
        lastpos = newpos
    end
    return true
end

ispartition(Hs, H::AbstractHilbertSpace) = ispartition(map(keys, Hs), keys(H))

function _find_position(label, labels::JordanWignerOrdering)
    get(labels.ordering, label, 0)
end
function _find_position(label, labels)
    pos = findfirst(==(label), labels)
    isnothing(pos) && return 0
    return pos
end

function ispartition(partition, labels)
    n = length(labels)
    covered = falses(n)
    for subsystem in partition
        for label in subsystem
            pos = _find_position(label, labels)
            pos == 0 && return false
            covered[pos] && return false
            covered[pos] = true
        end
    end
    return all(covered)
end

function isorderedpartition(partition, jw::JordanWignerOrdering)
    n = length(jw)
    covered = falses(n)
    for subsystem in partition
        lastpos = 0
        for label in subsystem
            pos = get(jw.ordering, label, 0)
            pos == 0 && return false
            pos > lastpos || return false
            covered[pos] && return false
            covered[pos] = true
            lastpos = pos
        end
    end
    all(covered) || return false
    return true
end

isorderedpartition(Hs, H::AbstractHilbertSpace) = isorderedpartition(flat_fock_spaces(Hs), fock_part(H)) && ispartition(Hs, H)
ispartition(::Any, ::Nothing) = true
isorderedpartition(::Any, ::Nothing) = true
function isorderedsubsystem(subsystem, jw::JordanWignerOrdering)
    consistent_ordering(subsystem, jw) || return false
    issubsystem(subsystem, jw) || return false
    return true
end
isorderedsubsystem(H::AbstractHilbertSpace, H2::AbstractHilbertSpace) = isorderedsubsystem(fock_part(H), fock_part(H2)) && issubsystem(H, H2)
isorderedsubsystem(::Nothing, ::Nothing) = true
isorderedsubsystem(::Nothing, ::AbstractHilbertSpace) = true
function issubsystem(subsystem, jw::JordanWignerOrdering)
    all(in(s, jw) for s in subsystem) || return false
    return true
end
function issubsystem(sublabels, labels)
    all(in(s, labels) for s in sublabels) || return false
    return true
end
issubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace) = issubsystem(keys(Hsub), keys(H))

@testitem "partition" begin
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    jw = JordanWignerOrdering(1:3)
    ispart = Base.Fix2(ispartition, jw)
    @test ispart([[1], [2], [3]])
    @test !ispart([[1], [2]])
    @test !ispart([[1, 1, 1]])
    @test !ispart([[1], [1], [2]])
    @test ispart([[1], [2, 3]])
    @test !ispart([[1], [2, 3, 4]])
    @test ispart([[1, 2, 3]])
    @test !ispart([[1, 2]])
    @test ispart([[2], [1], [3]])
    @test ispart([[2], [3], [1]])
    @test ispart([[1, 3], [2]])
    @test ispart([[3, 1], [2]])
    @test !ispart([[3, 1], [2, 4]])
    @test ispart([[2], [1, 3]])
    @test !ispart([[2], [2, 3]])
    @test ispart([[], [1, 2, 3]])
    @test !ispart([[1], [1, 2, 3]])

    ## same for ispartvec
    ispartvec = Base.Fix2(ispartition, keys(jw))
    @test ispartvec([[1], [2], [3]])
    @test !ispartvec([[1], [2]])
    @test !ispartvec([[1, 1, 1]])
    @test !ispartvec([[1], [1], [2]])
    @test ispartvec([[1], [2, 3]])
    @test !ispartvec([[1], [2, 3, 4]])
    @test ispartvec([[1, 2, 3]])
    @test !ispartvec([[1, 2]])
    @test ispartvec([[2], [1], [3]])
    @test ispartvec([[2], [3], [1]])
    @test ispartvec([[1, 3], [2]])
    @test ispartvec([[3, 1], [2]])
    @test !ispartvec([[3, 1], [2, 4]])
    @test ispartvec([[2], [1, 3]])
    @test !ispartvec([[2], [2, 3]])
    @test ispartvec([[], [1, 2, 3]])
    @test !ispartvec([[1], [1, 2, 3]])

    ## Ordered partition
    isorderedpart = Base.Fix2(isorderedpartition, jw)

    @test isorderedpart([[1], [2], [3]])
    @test isorderedpart([[1], [2, 3]])
    @test isorderedpart([[1, 2, 3]])
    @test isorderedpart([[2], [1], [3]])
    @test isorderedpart([[2], [3], [1]])
    @test isorderedpart([[1, 3], [2]])
    @test !isorderedpart([[3, 1], [2]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[3, 1], [2, 4]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[2], [3, 1]])
    @test !isorderedpart([[1], [3, 2]])
    @test !isorderedpart([[1], [3, 1]])
    @test !isorderedpart([[3], [2, 1]])
    @test isorderedpart([[2], [1, 3]])
    @test !isorderedpart([[2], [2, 3]])
    @test isorderedpart([[], [1, 2, 3]])
    @test !isorderedpart([[1], [1, 2, 3]])
end

default_fock_type(jw::JordanWignerOrdering) = FockNumber{default_fock_representation(length(jw))}
function phase_factor_h(Hs, H::AbstractFockHilbertSpace)
    partition = map(modes, flat_fock_spaces(Hs))
    phase_factor_h(partition, H.jw)
end
function phase_factor_h(partition, jw::JordanWignerOrdering)
    isorderedpartition(partition, jw) || error("Partition is not ordered")
    T = default_fock_type(jw)
    masks = map(Xp -> focknbr_from_site_labels(Xp, jw)::T, partition)
    inds = map(partition) do X
        [getindex(jw, li) for li in X]
    end
    function pfh(f1, f2)
        phase_factor_h(f1, f2, partition, masks, inds)
    end
end
phase_factor_h(f1, f2, Hs, H::AbstractFockHilbertSpace) = phase_factor_h(f1, f2, map(modes, flat_fock_spaces(Hs)), H.jw)
function phase_factor_h(f1::AbstractFockState, f2::AbstractFockState, partition, jw::JordanWignerOrdering)::Int
    #(120b)
    phase = 1
    for Xp in partition
        Xpmask = focknbr_from_site_labels(Xp, jw)
        masked_f1 = Xpmask & f1
        masked_f2 = Xpmask & f2
        for X in partition
            if X === Xp
                continue
            end
            for li in X
                i = getindex(jw, li)
                if _bit(f2, i)
                    phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
                end
            end
        end
    end
    return phase
end
function phase_factor_h(f1::AbstractFockState, f2::AbstractFockState, partition, masks, inds)::Int
    #(120b)
    phase = 1
    for (Xp, Xpmask) in zip(partition, masks)
        masked_f1 = Xpmask & f1
        masked_f2 = Xpmask & f2
        for (X, Xinds) in zip(partition, inds)
            if X === Xp
                continue
            end
            for i in Xinds
                if _bit(f2, i)
                    phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
                end
            end
        end
    end
    return phase
end
@testitem "Phase factor f" begin
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, getindices, bits

    ## Appendix A.2
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))

    @test [phase_factor_f(f1, f2, N) for f1 in fockstates, f2 in fockstates] ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]

    N = 3
    jw = JordanWignerOrdering(1:N)
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
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, getindices, bits
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    h(p, fockstates, jw) = [phase_factor_h(f1, f2, p, jw) for f1 in fockstates, f2 in fockstates]
    h_fast(p, fockstates, jw) = begin
        _h = phase_factor_h(p, jw)
        [_h(f1, f2) for f1 in fockstates, f2 in fockstates]
    end
    @test h([[1], [2]], fockstates, jw) ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]
    @test h_fast([[1], [2]], fockstates, jw) ==
          [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]

    phf(f1, f2, subinds, N) = prod(s -> phase_factor_f(f1, f2, s), subinds) * phase_factor_f(f1, f2, N)
    phf(fockstates, subinds, N) = [phf(f1, f2, subinds, N) for f1 in fockstates, f2 in fockstates]
    let part = [[1], [2]]
        subinds = map(p -> Tuple(getindices(jw, p)), part)
        N = length(jw)
        @test h(part, fockstates, jw) == phf(fockstates, subinds, N)
    end
    #
    N = 3
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    @test h([[1, 3], [2]], fockstates, jw) == [1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1;
        1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1]
    @test h_fast([[1, 3], [2]], fockstates, jw) == [1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1;
        1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1]

    partitions = [[[1], [2], [3]], [[2], [1], [3]], [[2], [3], [1]],
        [[1, 3], [2]], [[2, 3], [1]], [[3, 2], [1]], [[2, 1], [3]],
        [[1], [2, 3]], [[3], [2, 1]], [[2], [1, 3]],
        [[1, 2, 3]], [[2], [1], [3]]]
    for p in partitions
        subinds = map(p -> Tuple(getindices(jw, p)), p)
        @test h(p, fockstates, jw) == phf(fockstates, subinds, N)
    end

    N = 7
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    partitions = [[[3, 2, 7, 5, 1], [4, 6]], [[7, 3, 2], [1, 5], [4, 6]]]
    for p in partitions
        subinds = map(p -> Tuple(getindices(jw, p)), p)
        local N = length(jw)
        @test h(p, fockstates, jw) == phf(fockstates, subinds, N)
    end

end

function phase_factor_l(f1, f2, X, Xbar, jw)::Int
    #(123b)
    phase = 1
    Xmask = focknbr_from_site_labels(X, jw)
    masked_f1 = Xmask & f1
    masked_f2 = Xmask & f2
    for li in Xbar
        i = getindex(jw, li)
        if xor(_bit(f1, i), _bit(f2, i))
            phase *= jwstring_anti(i, masked_f1) * jwstring_anti(i, masked_f2)
        end
    end
    return phase
end
function phase_factor_l(f1, f2, partition, jw)::Int
    #(126b)
    X = partition
    phase = 1
    for (s, Xs) in enumerate(X)
        for Xr in Iterators.drop(X, s)
            mask = focknbr_from_site_labels(Iterators.flatten((Xs, Xr)), jw)
            phase *= phase_factor_l(mask & f1, mask & f2, Xs, Xr, jw)
        end
    end
    return phase
end

@testitem "Phase factor l" begin
    # Appendix B.6
    import FermionicHilbertSpaces: phase_factor_l, bits
    N = 2
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    lX(p1, p2, fockstates, jw) = [phase_factor_l(f1, f2, p1, p2, jw) for f1 in fockstates, f2 in fockstates]
    lξ(p, fockstates, jw) = [phase_factor_l(f1, f2, p, jw) for f1 in fockstates, f2 in fockstates]

    p = [[1], [2]]
    @test lX(p..., fockstates, jw) == lξ(p, fockstates, jw) == [1 1 1 1;
              1 1 1 1;
              1 1 1 1;
              1 1 1 1]

    p = [[2], [1]]
    @test lX(p..., fockstates, jw) == lξ(p, fockstates, jw) == [1 1 1 -1;
              1 1 -1 1;
              1 -1 1 1;
              -1 1 1 1]
    #
    N = 3
    jw = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))

    p = [[1], [2, 3]]
    @test lX(p..., fockstates, jw) == lξ(p, fockstates, jw) == ones(Int, 2^N, 2^N)

    p = [[1, 2], [3]]
    @test lX(p..., fockstates, jw) == lξ(p, fockstates, jw) == ones(Int, 2^N, 2^N)

    p = [[2], [1, 3]]
    @test lX(p..., fockstates, jw) == lξ(p, fockstates, jw) ==
          [1 1 1 1 1 1 -1 -1;
              1 1 1 1 1 1 -1 -1;
              1 1 1 1 -1 -1 1 1;
              1 1 1 1 -1 -1 1 1;
              1 1 -1 -1 1 1 1 1;
              1 1 -1 -1 1 1 1 1;
              -1 -1 1 1 1 1 1 1;
              -1 -1 1 1 1 1 1 1]

    @test lξ([[1], [2], [3]], fockstates, jw) == ones(Int, 2^N, 2^N)
    @test lξ([[2], [1], [3]], fockstates, jw) == lξ([[2], [1, 3]], fockstates, jw)
    @test lξ([[2], [3], [1]], fockstates, jw) == lξ([[2, 3], [1]], fockstates, jw) == lX([2, 3], [1], fockstates, jw)
end
