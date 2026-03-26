struct GenericHilbertSpace{B,L,S} <: AbstractAtomicHilbertSpace{B}
    label::L
    basisstates::S
    state_index::Dict{B,Int}
    function GenericHilbertSpace(label, basisstates, state_index=Dict(reverse(pair) for pair in enumerate(basisstates)))
        B = eltype(basisstates)
        new{B,eltype(label),typeof(basisstates)}(label, basisstates, state_index)
    end
end
Base.:(==)(H1::GenericHilbertSpace, H2::GenericHilbertSpace) = H1 === H2 || (H1.label == H2.label && H1.basisstates == H2.basisstates)
Base.hash(H::GenericHilbertSpace, h::UInt) = hash((H.label, H.basisstates), h)
basisstates(H::GenericHilbertSpace) = H.basisstates
basisstate(ind, H::GenericHilbertSpace) = H.basisstates[ind]
Base.keys(H::GenericHilbertSpace) = (H.label,)
state_index(state, H::GenericHilbertSpace) = get(H.state_index, state, missing)
symbolic_group(H::GenericHilbertSpace) = H.label

function Base.show(io::IO, H::GenericHilbertSpace)
    if get(io, :compact, false)
        print(io, "GenericHilbertSpace(", H.label, ")")
    else
        print(io, "$(dim(H))-dimensional GenericHilbertSpace\n")
        print(io, "Label: ", H.label)
    end
end
Base.show(io::IO, ::MIME"text/plain", H::AbstractHilbertSpace) = show(io, H)


@testitem "GenericHilbertSpace, ProductSpace" begin
    using FermionicHilbertSpaces: GenericHilbertSpace
    using LinearAlgebra
    H1 = GenericHilbertSpace(:A, [:a, :b])
    H2 = GenericHilbertSpace(:B, [:c, :d])
    P = tensor_product(H1, H2)
    @test dim(P) == dim(H1) * dim(H2)
    @test 2 * I(2) == partial_trace(1.0 * I(4), P => H1)

    @test embed(I, H1 => P) == I(4)

    @fermions f
    Hf = hilbert_space(f, 1:2)
    P2 = tensor_product(Hf, H1, H2)
    @test dim(P2) / dim(H1) * I(2) == partial_trace(1.0 * I(dim(P2)), P2 => H1)

    P3 = tensor_product(Hf, P)
    @test P3 == P2
    @test_throws ArgumentError tensor_product(Hf, P2)
end


"""
    dim(H)

Return the Hilbert-space dimension of `H`, i.e. the number of basis states.
"""
dim(H::AbstractHilbertSpace) = Int(length(basisstates(H)))


@testitem "Hilbert space subsystem and ordering" begin
    import FermionicHilbertSpaces: isorderedsubsystem, issubsystem
    @fermions f
    # Simple Hilbert spaces
    H = hilbert_space(f, [1, 2, 3])
    Hsub1 = hilbert_space(f, [1, 2])
    Hsub2 = hilbert_space(f, [2, 3])
    Hsub3 = hilbert_space(f, [2, 1])
    Hsub4 = hilbert_space(f, [3, 2])
    Hsub5 = hilbert_space(f, [4])

    # issubsystem
    @test issubsystem(Hsub1, H)
    @test issubsystem(Hsub2, H)
    @test !issubsystem(Hsub5, H)

    # isorderedsubsystem
    @test isorderedsubsystem(Hsub1, H)
    @test !isorderedsubsystem(Hsub3, H)
    @test !isorderedsubsystem(Hsub4, H)
    @test !isorderedsubsystem(Hsub5, H)

    # ispartition and isorderedpartition
    import FermionicHilbertSpaces: ispartition, isorderedpartition
    # Partition of Hilbert spaces (as required by isorderedpartition(Hs, H))
    H = hilbert_space(f, [1, 2, 3])
    Hs1 = [hilbert_space(f, [1]), hilbert_space(f, [2, 3])]
    Hs2 = [hilbert_space(f, [2]), hilbert_space(f, [1, 3])]
    Hs3 = [hilbert_space(f, [1, 2, 3])]
    Hs4 = [hilbert_space(f, [1]), hilbert_space(f, [2])]
    @test ispartition(Hs1, H)
    @test ispartition(Hs2, H)
    @test !ispartition(Hs4, H)
    @test isorderedpartition(Hs1, H)
    @test isorderedpartition(Hs3, H)
    @test !isorderedpartition(Hs4, H)
end


statetype(::AbstractHilbertSpace{F}) where F = F
statetype(::Nothing) = Nothing

@testitem "Subregion of fermionic spaces with conservation laws" begin
    using FermionicHilbertSpaces: atomic_factors
    @fermions f
    H = hilbert_space(f, [1, 2, 3])
    Hsub = hilbert_space(f, [1, 2])
    Hsubcons = subregion(Hsub, H)
    @test atomic_factors(Hsubcons) == atomic_factors(Hsub)

    focks = [FockNumber(1), FockNumber(3)]
    HF = hilbert_space(f, [1, 2], focks)
    HFsub = subregion(hilbert_space(f, [1]), HF)
    @test basisstates(HFsub) == [FockNumber(1)]

    constraint = ParityConservation()
    HS = hilbert_space(f, [1, 2], constraint)
    HSsub = subregion(hilbert_space(f, [1]), HS)
    @test all(in([FockNumber(0), FockNumber(1)]), basisstates(HSsub))
    # Error on non-subsystem
    @test_throws ArgumentError subregion(hilbert_space(f, [4]), H)
end


@testitem "Hilbert space printing" begin
    # Check that printing of Hilbert spaces doesn't error
    using FermionicHilbertSpaces
    io = IOBuffer()
    @fermions f
    H_simple = hilbert_space(f, 1:3)
    @test begin
        show(io, H_simple)
        true
    end
    # FockHilbertSpace
    focks = [FockNumber(0), FockNumber(1), FockNumber(2)]
    H_fock = hilbert_space(f, 1:2, focks)
    @test begin
        show(io, H_fock)
        true
    end
    # SymmetricFockHilbertSpace
    qn = ParityConservation()
    H_sym = hilbert_space(f, 1:2, qn)
    @test begin
        show(io, H_sym)
        true
    end
end

@testitem "Complementary system" begin
    using FermionicHilbertSpaces: complementary_subsystem
    @fermions f
    H = hilbert_space(f, 1:4)
    H1 = hilbert_space(f, 1:2, NumberConservation(1))
    H2 = complementary_subsystem(H, H1)
    H20 = hilbert_space(f, 3:4, NumberConservation(1))
    H2qn = constrain_space(H2, NumberConservation(1))
    @test dim(H2) == 4
    @test dim(H20) == 2
    @test H20 == H2qn
end
