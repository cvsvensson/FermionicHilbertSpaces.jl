# Runs on JLArrays, a CPU implementation of the GPU array interface that disallows scalar indexing.
# Not part of the main suite: `julia --project=test/gpu test/gpu/runtests.jl`.
using FermionicHilbertSpaces, JLArrays, Test
using FermionicHilbertSpaces: FixedNumberFockState, NoScalarIndexingAlg, change_basis
JLArrays.allowscalar(false)

@testset "partial_trace of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    for (H, Hsub) in ((hilbert_space(f, 1:4), hilbert_space(f, [1, 3])), (Hcons, subregion(hilbert_space(f, 1:2), Hcons)))
        for x in (rand(ComplexF64, dim(H), dim(H)), rand(ComplexF64, dim(H)))
            out = partial_trace(JLArray(x), H => Hsub; alg)
            @test out isa JLArray
            @test Array(out) ≈ partial_trace(x, H => Hsub)
        end
    end
    @test_throws ArgumentError partial_trace(JLArray(rand(dim(Hcons), dim(Hcons))), Hcons => hilbert_space(f, 1:2, NumberConservation(1)); alg)

    # States 1 and 2 share the entry (K, b) = (1, 1), with weights 1 and -1.
    ψ = rand(ComplexF64, 3)
    A = [ψ[1] - ψ[2], ψ[3]]
    splits = [((1, 1, 1),), ((1, 1, -1),), ((2, 1, 1),)]
    @test Array(FermionicHilbertSpaces._gather_partial_trace(JLArray(ψ), splits, 2, 1)) ≈ A * A'
end

@testset "change_basis of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    H = hilbert_space(f, 1:4)
    Hsec = hilbert_space(f, 1:4, NumberConservation(2))
    for x in (rand(ComplexF64, dim(Hsec)), rand(ComplexF64, dim(Hsec), dim(Hsec)))
        out = change_basis(JLArray(x), Hsec => H; alg)
        @test out isa JLArray
        @test Array(out) == change_basis(x, Hsec => H)
    end
    ρ = rand(ComplexF64, dim(H), dim(H))
    @test Array(partial_trace(JLArray(ρ), H => Hsec; alg, projection = true)) == partial_trace(ρ, H => Hsec; projection = true)
    @test_throws ArgumentError change_basis(JLArray(ρ), H => Hsec; alg)
end

@testset "embed of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    Hcons = hilbert_space(f, 1:4, NumberConservation(2))
    for (Hsub, H) in ((hilbert_space(f, [1, 3]), hilbert_space(f, 1:4)), (subregion(hilbert_space(f, 1:2), Hcons), Hcons))
        m = rand(ComplexF64, dim(Hsub), dim(Hsub))
        out = embed(JLArray(m), Hsub => H; alg, projection = true)
        @test out isa JLArray
        @test Array(out) ≈ embed(m, Hsub => H; projection = true)
    end
    Hpar = hilbert_space(f, 1:4, ParityConservation())
    v = rand(ComplexF64, 8)
    @test Array(embed(JLArray(v), sector(-1, Hpar) => Hpar; alg)) ≈ embed(v, sector(-1, Hpar) => Hpar)
    @test_throws ArgumentError embed(JLArray(rand(4, 4)), hilbert_space(f, 1:2) => hilbert_space(f, 1:2, NumberConservation(0:1)); alg)
end

@testset "reshape of GPU arrays" begin
    alg = NoScalarIndexingAlg()
    @fermions f
    H1, H2 = hilbert_space(f, 1:2), hilbert_space(f, 3:4)
    H = hilbert_space(f, 1:4, NumberConservation(2))
    for x in (rand(ComplexF64, dim(H)), rand(ComplexF64, dim(H), dim(H)))
        T = reshape(JLArray(x), H => (H1, H2); alg)
        @test T isa JLArray
        @test Array(T) == reshape(x, H => (H1, H2))
        @test Array(reshape(T, (H1, H2) => H; alg)) == x
    end
    @test_throws ArgumentError reshape(JLArray(rand(dim(H1), dim(H2))), (H1, H2) => H; alg)
end

@testset "representation as GPU arrays" begin
    @fermions f
    @boson b
    @spin s 1 // 2
    hopping = sum(f[i]' * f[i + 1] + hc for i in 1:3) + sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:3)
    mixed = f[1]' * f[2] * s[:+] + 1im * b' * s[:-] + hc + f[1]' * f[1] * b' * b + s[:z] + 2
    Hmixed = tensor_product(hilbert_space(f, 1:2, NumberConservation(1)), hilbert_space(s), hilbert_space(b, 3))
    Hfixed = hilbert_space(f, 1:3, FixedNumberFockState{1}.(basisstates(hilbert_space(f, 1:3, NumberConservation(1)))))
    cases = (
        (hopping, hilbert_space(f, 1:4)),
        (hopping, hilbert_space(f, 1:4, NumberConservation(2))),
        (b' * b + b + b', hilbert_space(b, 4)),
        (s[:+] + s[:-] + 0.5 * s[:z], hilbert_space(s)),
        (mixed, Hmixed),
        (f[1]' * f[2] + hc, hilbert_space(f, 1:10, NumberConservation(1))),
        (2, hilbert_space(f, 1:2)),
        (f[1]' * f[2] * s[:+] + hc + 2im, tensor_product(hilbert_space(f, 1:2), hilbert_space(s))),
        (f[1]' * f[2] * s[:+] + hc, tensor_product(Hfixed, hilbert_space(s))),
    )
    for (op, H) in cases, format in (JLArray, JLSparseMatrixCSC, JLSparseMatrixCSR{ComplexF32})
        M = representation(op, H, format)
        @test M isa format
        @test collect(M) ≈ representation(op, H, Matrix)
    end
    @test_throws ArgumentError representation(f[1], hilbert_space(f, 1:4, NumberConservation(2)), JLArray)
    @test_throws ArgumentError representation(hopping, hilbert_space(f, 1:4), JLSparseMatrixCSC{Float64, Int8})
    @majoranas γ
    @test_throws ArgumentError representation(1im * γ[1] * γ[2], hilbert_space(γ, 1:4), JLArray)
end

@testset "representation of Hubbard, Holstein and t-J chains as GPU arrays" begin
    @fermions c
    @bosons b
    n = 4
    t, U, ω, g = 1 + 1im, 4.0, 1.0, 0.5
    function check(op, H, T; projection = false)
        @test collect(representation(op, H, JLSparseMatrixCSC{T}; projection)) ≈ representation(op, H, Matrix{T}; projection)
        @test collect(representation(op, H, JLMatrix{T}; projection)) ≈ representation(op, H, Matrix{T}; projection)
    end
    hubbard = sum(-t * c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(n - 1)) +
        sum(U * c[i, :↑]' * c[i, :↑] * c[i, :↓]' * c[i, :↓] for i in 1:n)
    labels(σ) = [(i, σ) for i in 1:n]
    updown(k) = tensor_product(hilbert_space(c, labels(:↑), NumberConservation(k)), hilbert_space(c, labels(:↓), NumberConservation(k)))
    check(hubbard, hilbert_space(c, [(i, σ) for i in 1:n for σ in (:↑, :↓)]), ComplexF32)
    check(hubbard, updown(2), ComplexF32)

    holstein = -sum(t * c[i]' * c[i + 1] + hc for i in 1:(n - 1)) + ω * sum(b[i]' * b[i] for i in 1:n) +
        g * sum(c[i]' * c[i] * (b[i]' + b[i]) for i in 1:n)
    check(holstein, tensor_product(hilbert_space(c, 1:n, NumberConservation(2)), hilbert_space(b, 1:n, 3)), ComplexF32)

    nn(i, σ) = c[i, σ]' * c[i, σ]
    splus(i) = c[i, :↑]' * c[i, :↓]
    tj = sum(-c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(n - 1)) +
        sum(0.15 * splus(i) * splus(i + 1)' + hc - 0.15 * (nn(i, :↑) * nn(i + 1, :↓) + nn(i, :↓) * nn(i + 1, :↑)) for i in 1:(n - 1))
    no_double = prod(NumberConservation(0:1, [c[i, :↑], c[i, :↓]]) for i in 1:n)
    check(tj, constrain_space(updown(1), no_double), Float32; projection = true)
end
