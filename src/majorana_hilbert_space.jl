struct MajoranaHilbertSpace{L,H} <: AbstractHilbertSpace
    siteindex::L
    parent::H
end
Base.size(m::MajoranaHilbertSpace) = size(m.parent)
mode_ordering(m::MajoranaHilbertSpace) = mode_ordering(m.parent)
Base.:(==)(m1::MajoranaHilbertSpace, m2::MajoranaHilbertSpace) = m1.pairing == m2.pairing && m1.parent == m2.parent
focknumbers(m::MajoranaHilbertSpace) = focknumbers(m.parent)

function majorana_hilbert_space(labels, qn=NoSymmetry())
    iseven(length(labels)) || throw(ArgumentError("Must be an even number of Majoranas to define a Hilbert space."))
    pairs = [(labels[i], labels[i+1]) for i in 1:2:length(labels)-1]
    H = hilbert_space(pairs, qn)
    # majorana_position = OrderedDict(label => div(n + 1, 2) for (n, label) in enumerate(labels))
    majorana_position = OrderedDict(label => n for (n, label) in enumerate(labels))
    MajoranaHilbertSpace(majorana_position, H)
end
Base.show(io::IO, m::MajoranaHilbertSpace) = (println(io, "MajoranaHilbertSpace:"); show(io, m.parent))
## Define matrix representations of symbolic majorana operators on Majorana Hilbert spaces.

matrix_representation(op, H::MajoranaHilbertSpace) = matrix_representation(op, H.siteindex, focknumbers(H), focknumbers(H))
matrix_representation(op::Number, H::MajoranaHilbertSpace) = matrix_representation(op, H.parent)

function operator_inds_amps!((outinds, ininds, amps), op::FermionMul{C,F}, label_to_site, outstates, instates; fock_to_outind=Dict(map(reverse, enumerate(outstates)))) where {C,F<:AbstractMajoranaSym}
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