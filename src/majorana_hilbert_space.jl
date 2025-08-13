struct MajoranaHilbertSpace{L,H} <: AbstractFockHilbertSpace
    majoranaindices::L
    parent::H
end
Base.size(H::MajoranaHilbertSpace) = size(H.parent)
mode_ordering(H::MajoranaHilbertSpace) = mode_ordering(H.parent)
modes(H::MajoranaHilbertSpace) = modes(H.parent)
Base.:(==)(H1::MajoranaHilbertSpace, H2::MajoranaHilbertSpace) = H1.majoranaindices == H2.majoranaindices && H1.parent == H2.parent
basisstates(m::MajoranaHilbertSpace) = basisstates(m.parent)
Base.parent(H::MajoranaHilbertSpace) = H.parent

function majoranas(H::MajoranaHilbertSpace)
    @majoranas γ
    OrderedDict(l => matrix_representation(γ[l], H) for l in keys(H.majoranaindices))
end
function majorana_hilbert_space(labels, qn=NoSymmetry())
    iseven(length(labels)) || throw(ArgumentError("Must be an even number of Majoranas to define a Hilbert space."))
    pairs = [(labels[i], labels[i+1]) for i in 1:2:length(labels)-1]
    H = hilbert_space(pairs, qn)
    # majorana_position = OrderedDict(label => div(n + 1, 2) for (n, label) in enumerate(labels))
    majorana_position = OrderedDict(label => n for (n, label) in enumerate(labels))
    MajoranaHilbertSpace(majorana_position, H)
end
Base.show(io::IO, m::MajoranaHilbertSpace) = (println(io, "MajoranaHilbertSpace:"); show(io, m.parent))

function subregion(modes, H::MajoranaHilbertSpace)
    iseven(length(modes)) || throw(ArgumentError("Must be an even number of Majoranas to define a subregion."))
    pairs = [(modes[i], modes[i+1]) for i in 1:2:length(modes)-1]
    majorana_position = OrderedDict(label => n for (n, label) in enumerate(modes))
    MajoranaHilbertSpace(majorana_position, subregion(pairs, H.parent))
end
partial_trace!(mout, m::AbstractMatrix, H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace, phase_factors::Bool=true, complement::MajoranaHilbertSpace=simple_complementary_subsystem(H, Hsub)) = partial_trace!(mout, m, H.parent, Hsub.parent, phase_factors, complement.parent)
function simple_complementary_subsystem(H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace)
    complement_labels = setdiff(keys(H.majoranaindices), keys(Hsub.majoranaindices))
    complement_fermionic_space = simple_complementary_subsystem(H.parent, Hsub.parent)
    majorana_position = OrderedDict(label => n for (n, label) in enumerate(complement_labels))
    MajoranaHilbertSpace(majorana_position, complement_fermionic_space)
end
function complementary_subsystem(H::MajoranaHilbertSpace, Hsub::MajoranaHilbertSpace)
    complement_labels = setdiff(keys(H.majoranaindices), keys(Hsub.majoranaindices))
    complement_fermionic_space = complementary_subsystem(H.parent, Hsub.parent)
    majorana_position = OrderedDict(label => n for (n, label) in enumerate(complement_labels))
    MajoranaHilbertSpace(majorana_position, complement_fermionic_space)
end
isorderedpartition(Hs, H::MajoranaHilbertSpace) = isorderedpartition(map(parent, Hs), H.parent)
embedding(m, H::MajoranaHilbertSpace, Hnew::MajoranaHilbertSpace; kwargs...) = embedding(m, H.parent, Hnew.parent; kwargs...)
function tensor_product(H1::MajoranaHilbertSpace, H2::MajoranaHilbertSpace)
    Hf = tensor_product(H1.parent, H2.parent)
    majoranaindices = OrderedDict(mapreduce((ntup) -> [ntup[2][1] => 2ntup[1] - 1, ntup[2][2] => 2ntup[1]], vcat, enumerate(keys(Hf))))
    MajoranaHilbertSpace(majoranaindices, Hf)
end
## Define matrix representations of symbolic majorana operators on Majorana Hilbert spaces.
matrix_representation(op, H::MajoranaHilbertSpace) = matrix_representation(op, H.majoranaindices, basisstates(H), basisstates(H))
matrix_representation(op::Number, H::MajoranaHilbertSpace) = matrix_representation(op, H.parent)

function operator_inds_amps_generic!((outinds, ininds, amps), op::FermionMul{C,F}, label_to_site, outstates, instates, fock_to_outind) where {C,F<:AbstractMajoranaSym}
    majoranadigitpositions = Iterators.reverse(label_to_site[f.label] for f in op.factors)
    daggers = collect(iseven(pos) for pos in majoranadigitpositions)
    digitpositions = map(n -> div(n + 1, 2), majoranadigitpositions)
    mc = -op.coeff
    mic = -1im * op.coeff
    pc = op.coeff
    pic = 1im * op.coeff
    for (n, f) in enumerate(instates)
        newfockstate, amp = togglemajoranas(digitpositions, daggers, f)
        if !iszero(amp)
            push!(outinds, fock_to_outind[newfockstate])
            if amp == 1
                push!(amps, pc)
            elseif amp == -1
                push!(amps, mc)
            elseif amp == 1im
                push!(amps, pic)
            elseif amp == -1im
                push!(amps, mic)
            end
            push!(ininds, n)
        end
    end
    return (outinds, ininds, amps)
end

@testitem "Majorana matrix representations" begin
    H = FermionicHilbertSpaces.majorana_hilbert_space(1:2)
    Hf = H.parent
    @majoranas γ
    @fermions f

    @test parityoperator(H.parent) == matrix_representation(1im * γ[1] * γ[2], H)
    y1 = matrix_representation(γ[1], H)
    y2 = matrix_representation(γ[2], H)
    @test y1 * y2 == matrix_representation(γ[1] * γ[2], H)

    y(f) = f.creation ? -1im * f + hc : f + hc
    @test matrix_representation(γ[1], H) == matrix_representation(y(f[(1, 2)]), Hf)
    @test matrix_representation(γ[2], H) == matrix_representation(y(f[(1, 2)]'), Hf)
    @test matrix_representation(1, H) == matrix_representation(1, Hf)
    @test matrix_representation(γ[1] * γ[2], H) == matrix_representation(y(f[(1, 2)]) * y(f[(1, 2)]'), Hf)
    @test matrix_representation(1 + γ[1] + 1im * γ[2] + 0.2 * γ[1] * γ[2], H) ==
          matrix_representation(1 + y(f[(1, 2)]) + 1im * y(f[(1, 2)]') + 0.2 * y(f[(1, 2)]) * y(f[(1, 2)]'), Hf)
end

@testitem "Majorana hilbert space" begin
    using FermionicHilbertSpaces: majorana_hilbert_space
    H = majorana_hilbert_space(1:4, ParityConservation(1))
    Hsub = subregion(1:2, H)
    Hf = H.parent
    Hfsub = subregion([(1, 2)], Hf)
    m = rand(size(H)...)
    @test partial_trace(m, H => Hsub) == partial_trace(m, Hf => Hfsub)
    Hsub2 = subregion(3:4, H)
    Hfsub2 = subregion([(3, 4)], Hf)

    Hprod = tensor_product(Hsub, Hsub2)
    @test parent(Hprod) == tensor_product(Hfsub, Hfsub2)

    m1 = rand(size(Hsub)...)
    m2 = rand(size(Hsub2)...)
    @test tensor_product((m1, m2), (Hsub, Hsub2), Hprod) == tensor_product((m1, m2), (Hfsub, Hfsub2) => parent(Hprod))
end