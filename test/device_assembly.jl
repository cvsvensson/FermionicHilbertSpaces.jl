@testmodule DeviceAssembly begin
    using FermionicHilbertSpaces, LinearAlgebra, SparseArrays, Test
    using AcceleratedKernels, Adapt
    using KernelAbstractions: CPU
    export backend, check

    # On KernelAbstractions 0.10 this backend compiles kernels with GPUCompiler, so it rejects non-isbits arguments,
    # dynamic dispatch and runtime allocation as a GPU backend would.
    const backend = CPU()

    function check(op, H; T = Float64, projection = false)
        ref = representation(op, H, Matrix{T}; projection)
        @test Array(representation(op, H, DeviceDenseRepr{T}(backend); projection)) ≈ ref
        D = representation(op, H, DeviceCSCRepr{T}(backend); projection)
        S = SparseMatrixCSC(D)
        @test S ≈ ref
        @test all(j -> issorted(view(rowvals(S), nzrange(S, j))), axes(S, 2))
        x = T.(1:size(ref, 2))
        return @test mul!(similar(x), D, x) ≈ ref * x
    end
end

@testitem "Device assembly: fermions" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    using SparseArrays: SparseMatrixCSC
    @fermions f
    H = hilbert_space(f, 1:6)
    op = sum(f[i]' * f[i + 1] + hc for i in 1:5) + 2.0 * sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:5) +
        0.3 * f[1]' * f[3]' + hc + 1.5
    check(op, H)
    check(op, H; T = Float32)
    check(1im * f[1]' * f[2] + hc, H; T = ComplexF64)
    check(f[2]' * f[4] * f[5], H)
    check(f[3], H)
    check(2, H)
    @test Array(representation(op, H, DeviceDenseRepr(backend))) ≈ representation(op, H, Matrix)
    @test SparseMatrixCSC(representation(f[1]' * f[2], H, DeviceCSCRepr(backend))) == representation(f[1]' * f[2], H)
    @test_throws ArgumentError representation(1im * f[1], H, DeviceDenseRepr{Float64}(backend))
end

@testitem "Device assembly: number conservation" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    @fermions f
    hop(N) = sum(f[i]' * f[i + 1] + hc for i in 1:(N - 1))
    # An index table
    check(hop(6) + 2.0 * sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:5) + 0.5, hilbert_space(f, 1:6, NumberConservation(3)))
    # An index keyed by the position in the parent space
    H = hilbert_space(f, 1:12, NumberConservation(1))
    check(hop(12) + 0.1 * f[1]' * f[12], H)
    check(f[1], H; projection = true)
    # Asynchronous backends throw when the result is used.
    @test_throws Exception Array(representation(f[1], H, DeviceDenseRepr(backend)))
end

@testitem "Device assembly: parity, sectors and listed states" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    using FermionicHilbertSpaces: FixedNumberFockState
    @fermions f
    op = sum(f[i]' * f[i + 1] + 0.4 * f[i]' * f[i + 1]' + hc for i in 1:5) + sum(f[i]' * f[i] for i in 1:6)
    check(op, hilbert_space(f, 1:6, ParityConservation(1)))
    check(op, hilbert_space(f, 1:6, ParityConservation()))
    hop = sum(f[i]' * f[i + 1] + hc for i in 1:5)
    check(hop, hilbert_space(f, 1:6, NumberConservation(1:3)))
    H = hilbert_space(f, 1:4)
    check(f[1]' * f[2] + hc, constrain_space(H, collect(basisstates(H))[[1, 2, 3, 5, 9]]); projection = true)
    states = FixedNumberFockState{2}.(basisstates(hilbert_space(f, 1:6, NumberConservation(2))))
    Hx = hilbert_space(f, 1:6, states)
    check(hop + 0.5 * f[1]' * f[2]' * f[4] * f[3] + hc, Hx)
    check(f[1]' * f[2]' * f[3], Hx; projection = true)
end

@testitem "Device assembly: bosons" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    @bosons b
    @boson a
    check(a' * a' * a + 0.5 * a + hc, hilbert_space(a, 5))
    op = sum(b[i]' * b[i + 1] + hc for i in 1:2) + 0.5 * sum(b[i]' * b[i]' * b[i] * b[i] for i in 1:3) + 0.2 * b[1]' + 1
    check(op, hilbert_space(b, 1:3, 4))
    check(sum(b[i]' * b[i + 1] + hc for i in 1:2) + b[2]' * b[2], hilbert_space(b, 1:3, 4, NumberConservation(3)))
end

@testitem "Device assembly: spins" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    @spin S 1
    check(S[:+]^2 + S[:z] + hc, hilbert_space(S))
    @spins s 1 // 2
    H = hilbert_space(s, 1:6)
    heisenberg = sum(s[i][:x] * s[i + 1][:x] + s[i][:y] * s[i + 1][:y] + 0.7 * s[i][:z] * s[i + 1][:z] for i in 1:5)
    check(heisenberg + 0.3 * s[1][:x], H; T = ComplexF64)
    check(sum(s[i][:+] * s[i + 1][:-] + hc for i in 1:5) + s[2][:z], H)
    Hm = hilbert_space(s, 1:6, AdditiveConstraint(3, st -> st.m + 1 // 2))
    check(sum(s[i][:+] * s[i + 1][:-] + hc for i in 1:5) + sum(s[i][:z] * s[i + 1][:z] for i in 1:5), Hm)
end

@testitem "Device assembly: fermions, spins and bosons" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    @fermions f
    @spin S 1 // 2
    @boson b
    Hf = hilbert_space(f, 1:3)
    Hs = hilbert_space(S)
    Hb = hilbert_space(b, 4)
    op = sum(f[i]' * f[i + 1] + hc for i in 1:2) + f[1]' * f[1] * S[:z] + 0.5 * (b + b') * S[:x] +
        0.2 * f[2]' * f[2] * b' * b + f[1] * S[:+] + hc + 0.4
    check(op, tensor_product(Hf, Hs, Hb); T = ComplexF64)
    conserving = sum(f[i]' * f[i + 1] + hc for i in 1:2) + f[1]' * f[1] * S[:z] + 0.5 * (b + b') * S[:x]
    check(conserving, tensor_product(Hf, Hs, Hb; constraint = NumberConservation(1, [Hf])); T = ComplexF64)
    check(conserving, tensor_product(hilbert_space(f, 1:3, NumberConservation(2)), Hs, Hb); T = ComplexF64)
end

@testitem "Device assembly: Hubbard, Holstein and t-J chains" setup = [DeviceAssembly] begin
    using .DeviceAssembly
    @fermions c
    @bosons b
    n = 4
    t, U, ω, g = 1 + 1im, 4.0, 1.0, 0.5
    hubbard = sum(-t * c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(n - 1)) +
        sum(U * c[i, :↑]' * c[i, :↑] * c[i, :↓]' * c[i, :↓] for i in 1:n)
    labels(σ) = [(i, σ) for i in 1:n]
    updown(k) = tensor_product(hilbert_space(c, labels(:↑), NumberConservation(k)), hilbert_space(c, labels(:↓), NumberConservation(k)))
    check(hubbard, hilbert_space(c, [(i, σ) for i in 1:n for σ in (:↑, :↓)]); T = ComplexF32)
    check(hubbard, updown(2); T = ComplexF32)

    nf(i) = c[i]' * c[i]
    holstein = -sum(t * c[i]' * c[i + 1] + hc for i in 1:(n - 1)) + ω * sum(b[i]' * b[i] for i in 1:n) +
        g * sum(nf(i) * (b[i]' + b[i]) for i in 1:n)
    check(holstein, tensor_product(hilbert_space(c, 1:n, NumberConservation(2)), hilbert_space(b, 1:n, 3)); T = ComplexF32)

    nn(i, σ) = c[i, σ]' * c[i, σ]
    splus(i) = c[i, :↑]' * c[i, :↓]
    tj = sum(-c[i, σ]' * c[i + 1, σ] + hc for σ in (:↑, :↓) for i in 1:(n - 1)) +
        sum(0.15 * splus(i) * splus(i + 1)' + hc - 0.15 * (nn(i, :↑) * nn(i + 1, :↓) + nn(i, :↓) * nn(i + 1, :↑)) for i in 1:(n - 1))
    no_double = prod(NumberConservation(0:1, [c[i, :↑], c[i, :↓]]) for i in 1:n)
    check(tj, constrain_space(updown(1), no_double); T = Float32, projection = true)
end
