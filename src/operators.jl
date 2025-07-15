"""
    removefermion(digitposition, f::FockNumber)

Return (newfocknbr, fermionstatistics) where `newfocknbr` is the state obtained by removing a fermion at `digitposition` from `f` and `fermionstatistics` is the phase from the Jordan-Wigner string, or 0 if the operation is not allowed.
"""
function removefermion(digitposition, f::FockNumber)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ f
    allowed = !iszero(cdag & f)
    fermionstatistics = jwstring(digitposition, f)
    return allowed * newfocknbr, allowed * fermionstatistics
end

"""
    parityoperator(H)

Return the fermionic parity operator for the Hilbert space `H`.
"""
function parityoperator(H::AbstractFockHilbertSpace)
    sparse_fockoperator(f -> (f, parity(f)), H)
end


"""
    numberoperator(H)

Return the number operator for the Hilbert space `H`.
"""
function numberoperator(H::AbstractFockHilbertSpace)
    sparse_fockoperator(f -> (f, fermionnumber(f)), H)
end

function togglefermions(digitpositions, daggers, focknbr::FockNumber)
    newfocknbr = 0
    allowed = true
    fermionstatistics = 1
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = (1 << (digitpos - 1)) #2^(digitpos - 1) but faster
        if dagger
            newfocknbr = op | focknbr
            # Check if there already was a fermion at the site.
            allowed = iszero(op & focknbr)
        else
            newfocknbr = op ⊻ focknbr
            # Check if the site was empty.
            allowed = !iszero(op & focknbr)
        end
        # return directly if we create/annihilate an occupied/empty state
        if !allowed
            return newfocknbr, allowed * fermionstatistics
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    # fermionstatistics better way?
    return newfocknbr, allowed * fermionstatistics
end

function togglemajoranas(digitpositions, daggers, focknbr)
    newfocknbr = zero(focknbr)
    allowed = true
    fermionstatistics = 1
    amp = Complex{Int}(1)
    for (digitpos, dagger) in zip(digitpositions, daggers)
        op = FockNumber(1 << (digitpos - 1))
        newfocknbr = op ⊻ focknbr
        if dagger
            occupied = iszero(op & focknbr)
            amp *= 1im * (occupied ? -1 : 1)
        end
        fermionstatistics *= jwstring(digitpos, focknbr)
        focknbr = newfocknbr
    end
    return newfocknbr, allowed * fermionstatistics * amp
end


"""
    fermion_sparse_matrix(fermion_number, H::AbstractFockHilbertSpace)

Constructs a sparse matrix of size representing a fermionic annihilation operator at bit position `fermion_number` on the Hilbert space H. 
"""
function fermion_sparse_matrix(fermion_number, H::AbstractFockHilbertSpace)
    sparse_fockoperator(Base.Fix1(removefermion, fermion_number), H)
end


function sparse_fockoperator(op, H::AbstractFockHilbertSpace)
    fs = basisstates(H)
    N = length(fs)
    amps = Int[]
    ininds = Int[]
    outinds = Int[]
    sizehint!(amps, N)
    sizehint!(ininds, N)
    sizehint!(outinds, N)
    for f in fs
        n = state_index(f, H)
        newfockstate, amp = op(f)
        if !iszero(amp)
            push!(amps, amp)
            push!(ininds, n)
            push!(outinds, state_index(newfockstate, H))
        end
    end
    return SparseArrays.sparse!(outinds, ininds, amps, N, N)
end


@testitem "Parity and number operator" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: FockHilbertSpace, parityoperator, numberoperator, SymmetricFockHilbertSpace, fermion_sparse_matrix
    numopvariant(H) = sum(l -> fermion_sparse_matrix(l, H)' * fermion_sparse_matrix(l, H), 1:2)
    H = FockHilbertSpace(1:2)
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    H = SymmetricFockHilbertSpace(1:2, ParityConservation())
    @test parityoperator(H) == Diagonal([-1, -1, 1, 1])
    @test numberoperator(H) == Diagonal([1, 1, 0, 2]) == numopvariant(H)

    H = SymmetricFockHilbertSpace(1:2, FermionConservation())
    @test parityoperator(H) == Diagonal([1, -1, -1, 1])
    @test numberoperator(H) == Diagonal([0, 1, 1, 2]) == numopvariant(H)

    ## Truncated Hilbert space
    basisstates = map(FockNumber, 0:2)
    H = FockHilbertSpace(1:2, basisstates)
    @test parityoperator(H) == Diagonal([1, -1, -1])
    @test numberoperator(H) == Diagonal([0, 1, 1])

    basisstates = map(FockNumber, 2:2)
    H = FockHilbertSpace(1:2, basisstates)
    @test parityoperator(H) == Diagonal([-1])
    @test numberoperator(H) == Diagonal([1])
end

"""
    fermions(H)

Return a dictionary of fermionic annihilation operators for the Hilbert space `H`.
"""
function fermions(H::AbstractFockHilbertSpace)
    M = length(H.jw)
    labelvec = keys(H)
    reps = [fermion_sparse_matrix(n, H) for n in 1:M]
    OrderedDict(zip(labelvec, reps))
end

"""
    majoranas(H)

Return a dictionary of Majorana operators for the Hilbert space `H`.
"""
function majoranas(H::AbstractFockHilbertSpace, labels=Base.product(keys(H), (:-, :+)))
    fs = values(fermions(H))
    majA = map(f -> f + f', fs)
    majB = map(f -> 1im * (f - f'), fs)
    majs = vcat(majA, majB)
    OrderedDict(zip(labels, majs))
end

@testitem "CAR" begin
    using LinearAlgebra
    for qn in [NoSymmetry(), ParityConservation(), FermionConservation()]
        c = fermions(hilbert_space(1:2, qn))
        @test c[1] * c[1] == 0I
        @test c[1]' * c[1] + c[1] * c[1]' == I
        @test c[1]' * c[2] + c[2] * c[1]' == 0I
        @test c[1] * c[2] + c[2] * c[1] == 0I
    end
end

@testitem "Majorana operators" begin
    using LinearAlgebra
    using FermionicHilbertSpaces: majoranas
    H = hilbert_space(1:2)
    γ = majoranas(H)
    # There should be 4 Majorana operators for 2 modes
    @test length(γ) == 4
    # Test Hermiticity: γ₁ = c + c†, γ₂ = i(c - c†) are Hermitian
    for op in values(γ)
        @test op ≈ op'
    end
    # Test anticommutation: {γ_i, γ_j} = 2δ_{ij}I
    γops = values(γ)
    for γ1 in γops, γ2 in γops
        anticom = γ1 * γ2 + γ2 * γ1
        @test anticom ≈ 2I * (γ1 == γ2)
    end
end
