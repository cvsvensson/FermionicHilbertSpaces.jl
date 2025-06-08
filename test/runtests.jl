
using TestItemRunner
@run_package_tests

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

@testitem "Basis" begin
    using SparseArrays, LinearAlgebra, Random
    Random.seed!(1234)
    N = 2
    H = hilbert_space(1:N)
    B = fermions(H)
    # @test FermionicHilbertSpaces.nbr_of_modes(B) == N
    Hspin = hilbert_space(Base.product(1:N, (:↑, :↓)), FermionConservation())
    Bspin = fermions(Hspin)
    # @test FermionicHilbertSpaces.nbr_of_modes(Bspin) == 2N
    @test B[1] isa SparseMatrixCSC
    @test Bspin[1, :↑] isa SparseMatrixCSC
    @test parityoperator(H) isa SparseMatrixCSC
    @test parityoperator(Hspin) isa SparseMatrixCSC

    H = hilbert_space(1:3)
    a = fermions(H)
    Hs = (hilbert_space(1:1), hilbert_space(2:2), hilbert_space(3:3))
    Hw = tensor_product(Hs)
    @test FermionicHilbertSpaces.isfermionic(Hw)

    v = [FermionicHilbertSpaces.indtofock(i, H) for i in 1:8]
    t1 = reshape(v, H, Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    qn = ParityConservation()
    H1 = hilbert_space(2:2, qn)
    H2 = hilbert_space((1, 3), qn)
    H = hilbert_space(1:3, qn)
    v = [FermionicHilbertSpaces.indtofock(i, H) for i in 1:8]
    t1 = reshape(v, H, Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    using LinearMaps
    ptmap = LinearMap(rhovec -> vec(partial_trace(reshape(rhovec, size(H)), H, H1)), prod(size(H1)), prod(size(H)))
    embeddingmap = LinearMap(rhovec -> vec(embedding(reshape(rhovec, size(H1)), H1, H)), prod(size(H)), prod(size(H1)))
    @test Matrix(ptmap) ≈ Matrix(embeddingmap)'

    H = hilbert_space(Base.product(1:2, (:a, :b)))
    # c = fermions(H)
    Hparity = hilbert_space(Base.product(1:2, (:a, :b)), ParityConservation())
    # cparity = fermions(Hparity)
    ρ = Matrix(Hermitian(rand(2^4, 2^4) .- 0.5))
    ρ = ρ / tr(ρ)
    function bilinears(H, labels)
        c = fermions(H)
        ops = reduce(vcat, [[c[l], c[l]'] for l in labels])
        return [op1 * op2 for (op1, op2) in Base.product(ops, ops)]
    end
    function bilinear_equality(H, Hsub, ρ)
        subsystem = Tuple(keys(Hsub))
        ρsub = partial_trace(ρ, H, Hsub)
        @test tr(ρsub) ≈ 1
        all((tr(op1 * ρ) ≈ tr(op2 * ρsub)) for (op1, op2) in zip(bilinears(H, subsystem), bilinears(Hsub, subsystem)))
    end
    function get_subsystems(c, N)
        t = collect(Base.product(ntuple(i -> keys(c), N)...))
        (t[I] for I in CartesianIndices(t) if issorted(Tuple(I)) && allunique(Tuple(I)))
    end
    for N in 1:4
        @test all(bilinear_equality(H, hilbert_space(subsystem), ρ) for subsystem in get_subsystems(H, N))
        @test all(bilinear_equality(H, hilbert_space(subsystem, ParityConservation()), ρ) for subsystem in get_subsystems(H, N))
        @test all(bilinear_equality(H, hilbert_space(subsystem, ParityConservation()), ρ) for subsystem in get_subsystems(Hparity, N))
        @test all(bilinear_equality(H, hilbert_space(subsystem), ρ) for subsystem in get_subsystems(Hparity, N))
    end
end


@testitem "Fermionic trace" begin
    using LinearAlgebra
    N = 4
    Hs = [hilbert_space(n:n) for n in 1:N]
    H = hilbert_space(1:4)
    ops = [rand(ComplexF64, 2, 2) for _ in 1:N]
    op = fermionic_kron(ops, Hs, H)
    @test tr(op) ≈ prod(tr, ops)

    op = fermionic_kron(ops, Hs[[3, 2, 1, 4]], H)
    @test tr(op) ≈ prod(tr, ops)
end


@testitem "Fermionic partial trace" begin
    using LinearAlgebra, LinearMaps

    function test_adjoint(Hsub, H)
        pt = partial_trace(H => Hsub)
        embed = embedding(Hsub => H)
        ptmap = LinearMap(rhovec -> vec(pt(reshape(rhovec, size(H)))), prod(size(Hsub)), prod(size(H)))
        embeddingmap = LinearMap(rhovec -> vec(embed(reshape(rhovec, size(Hsub)))), prod(size(H)), prod(size(Hsub)))
        @test Matrix(ptmap) ≈ Matrix(embeddingmap)'
    end
    qns = [NoSymmetry(), ParityConservation(), FermionConservation()]
    for qn in qns
        H = hilbert_space(1:3, qn)
        H1 = hilbert_space(1:1, qn)
        H2 = hilbert_space(2:2, qn)
        H12 = hilbert_space(1:2, qn)
        H13 = hilbert_space(1:3, qn)
        H23 = hilbert_space(2:3, qn)
        c = fermions(H)
        c1 = fermions(H1)
        c2 = fermions(H2)
        c12 = fermions(H12)
        c13 = fermions(H13)
        c23 = fermions(H23)

        γ = Hermitian([0I rand(ComplexF64, 4, 4); rand(ComplexF64, 4, 4) 0I])
        f = c[1]
        @test tr(c1[1] * partial_trace(γ, H, H1)) ≈ tr(f * γ)
        @test tr(c12[1] * partial_trace(γ, H, H12)) ≈ tr(f * γ)
        @test tr(c13[1] * partial_trace(γ, H, H13)) ≈ tr(f * γ)

        f = c[2]
        @test tr(c2[2] * partial_trace(γ, H, H2)) ≈ tr(f * γ)
        @test tr(c12[2] * partial_trace(γ, H, H12)) ≈ tr(f * γ)
        @test tr(c23[2] * partial_trace(γ, H, H23)) ≈ tr(f * γ)

        test_adjoint(H1, H)
        test_adjoint(H12, H)
        test_adjoint(H13, H)
        test_adjoint(H23, H)
    end
end

@testitem "blocks" begin
    using SparseArrays, LinearAlgebra
    N = 2
    H = hilbert_space(1:N, ParityConservation())
    a = fermions(H)
    parityop = blocks(parityoperator(H), H)
    numberop = blocks(numberoperator(H), H)
end
