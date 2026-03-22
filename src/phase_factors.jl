partial_trace_phase_factor(f1, f2, ::AbstractAtomicHilbertSpace) = 1


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


_find_position(n::L, labels::JordanWignerOrdering{L}) where L = get(labels.ordering, n, 0)
_find_position(n, v::AbstractVector) = (pos = findfirst(==(n), v); isnothing(pos) ? 0 : pos)

issubsystem(Hsub::AbstractHilbertSpace, H::AbstractFermionicClusterHilbertSpace) = isorderedsubsystem(Hsub, H)
function isorderedsubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace)
    positions = [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)]
    all(pos -> pos > 0, positions) || return false
    issorted(positions) || return false
    return true
end
function isorderedpartition(Hsubs, H::AbstractHilbertSpace)
    positions = map(Hsub -> [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)], Hsubs)
    isorderedpartition(positions, length(atomic_factors(H)))
end
function issubsystem(Hsub::AbstractHilbertSpace, H::AbstractHilbertSpace)
    positions = [_find_atom_position(atom, H) for atom in atomic_factors(Hsub)]
    all(pos -> pos > 0, positions)
end
function ispartition(partition, H::AbstractHilbertSpace)
    partition_inds = [_find_atom_position(atom, H) for part in partition for atom in atomic_factors(part)]
    ispartition(partition_inds, length(atomic_factors(H)))
end

function ispartition(partition, N::Int)
    covered = falses(N)
    for subsystem in partition
        for pos in subsystem
            pos == 0 && return false
            covered[pos] && return false
            covered[pos] = true
        end
    end
    return all(covered)
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

_find_atom_position(atom, H::AbstractClusterHilbertSpace) = _find_position(atom, H)
_find_atom_position(atom, H::AbstractHilbertSpace) = _find_position(atom, atomic_factors(H))

@testitem "partition" begin
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    order = 1:3
    ispart = Base.Fix2(ispartition, order)
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
    ispartvec = Base.Fix2(ispartition, order)
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
    isorderedpart = Base.Fix2(isorderedpartition, order)

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

# default_fock_type(jw::JordanWignerOrdering) = FockNumber{default_fock_representation(length(jw))}

function kron_phase_factor(state_splitter::FockMapper)
    inds = state_splitter.fermionpositions
    N = sum(length, inds)
    T = FockNumber{default_fock_representation(N)}
    masks = map(Xp -> T(focknbr_from_site_indices(Xp)), inds)
    function pfh(f1, f2)
        phase_factor_h(f1, f2, inds, masks)
    end
end


function phase_factor_h(partition, jw::JordanWignerOrdering)
    isorderedpartition(partition, jw) || error("Partition is not ordered")
    T = default_fock_type(jw)
    masks = map(Xp -> focknbr_from_site_labels(Xp, jw)::T, partition)
    inds = map(partition) do X
        [getindex(jw, li) for li in X]
    end
    function pfh(f1, f2)
        # phase_factor_h(f1, f2, partition, masks, inds)
        phase_factor_h(f1, f2, inds, masks)
    end
end

# phase_factor_h(f1, f2, Hs, H::AbstractFockHilbertSpace) = phase_factor_h(f1, f2, map(modes, flat_fock_spaces(Hs)), H.jw)
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

function phase_factor_h(f1::AbstractFockState, f2::AbstractFockState, partition, masks)::Int
    # this assumes that partition is a list of list of indices
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
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, getindices, bits

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
    import FermionicHilbertSpaces: phase_factor_h, phase_factor_f, getindices, bits
    N = 2
    ordering = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    h(p, fockstates, ordering) = [phase_factor_h(f1, f2, p, ordering) for f1 in fockstates, f2 in fockstates]
    h_fast(p, fockstates, ordering) = begin
        _h = phase_factor_h(p, ordering)
        [_h(f1, f2) for f1 in fockstates, f2 in fockstates]
    end

    @fermions a
    import FermionicHilbertSpaces: state_splitter, kron_phase_factor
    h_splitter(p, fockstates, ordering) = begin
        H = hilbert_space(a, ordering)
        Hs = map(p) do part
            hilbert_space(a, part)
        end
        splitter = state_splitter(H, Hs)
        _h = kron_phase_factor(splitter)
        [_h(f1, f2) for f1 in fockstates, f2 in fockstates]
    end

    result = [1 1 1 -1;
        1 1 -1 1;
        1 1 1 -1;
        1 1 -1 1]
    @test h([[1], [2]], fockstates, ordering) == result
    @test h_fast([[1], [2]], fockstates, ordering) == result
    @test h_splitter([[1], [2]], fockstates, ordering) == result

    phf(f1, f2, subinds, N) = prod(s -> phase_factor_f(f1, f2, s), subinds) * phase_factor_f(f1, f2, N)
    phf(fockstates, subinds, N) = [phf(f1, f2, subinds, N) for f1 in fockstates, f2 in fockstates]
    let part = [[1], [2]]
        subinds = map(p -> Tuple(getindices(ordering, p)), part)
        @test h(part, fockstates, ordering) == phf(fockstates, subinds, N)
    end
    #
    N = 3
    ordering = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    result = [1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1;
        1 1 1 -1 1 1 -1 1;
        1 1 -1 1 1 1 1 -1;
        1 1 1 -1 -1 -1 1 -1;
        1 1 -1 1 -1 -1 -1 1]
    @test h([[1, 3], [2]], fockstates, ordering) == result
    @test h_fast([[1, 3], [2]], fockstates, ordering) == result
    @test h_splitter([[1, 3], [2]], fockstates, ordering) == result

    partitions = [[[1], [2], [3]], [[2], [1], [3]], [[2], [3], [1]],
        [[1, 3], [2]], [[2, 3], [1]], [[3, 2], [1]], [[2, 1], [3]],
        [[1], [2, 3]], [[3], [2, 1]], [[2], [1, 3]],
        [[1, 2, 3]], [[2], [1], [3]]]
    for p in partitions
        subinds = map(p -> Tuple(getindices(ordering, p)), p)
        @test h(p, fockstates, ordering) == phf(fockstates, subinds, N)
    end

    N = 7
    ordering = JordanWignerOrdering(1:N)
    fockstates = sort(map(FockNumber, 0:2^N-1), by=Base.Fix2(bits, N))
    partitions = [[[3, 2, 7, 5, 1], [4, 6]], [[7, 3, 2], [1, 5], [4, 6]]]
    for p in partitions
        subinds = map(p -> Tuple(getindices(ordering, p)), p)
        @test h(p, fockstates, ordering) == phf(fockstates, subinds, N)
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
