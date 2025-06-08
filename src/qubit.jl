function qubit_lowering_sparse_matrix(qubit_number, H::AbstractFockHilbertSpace)
    sparse_fockoperator(f -> lower_qubit(qubit_number, f), H)
end

function lower_qubit(digitposition, statefocknbr)
    cdag = focknbr_from_site_index(digitposition)
    newfocknbr = cdag ⊻ statefocknbr
    allowed = !iszero(cdag & statefocknbr)
    return allowed * newfocknbr, allowed
end

struct QubitOp{S} end
function QubitOperators(H::AbstractFockHilbertSpace)
    M = length(H.jw)
    labelvec = keys(H.jw)
    reps = [1.0 * complex(qubit_lowering_sparse_matrix(n, H)) for n in 1:M]
    ops = OrderedDict{Any,eltype(reps)}(zip(labelvec, reps))
    for (k, op) in Base.product(labelvec, (:Z, :X, :Y, :H))
        ops[k, op] = qubit_operator(ops[k], QubitOp{op}())
    end
    ops
end
qubit_operator(c, ::QubitOp{:Z}) = -2c'c + I
qubit_operator(c, ::QubitOp{:X}) = c + c'
qubit_operator(c, ::QubitOp{:Y}) = 1im * (c' - c)
qubit_operator(c, ::QubitOp{:I}) = 0c + I
qubit_operator(c, ::QubitOp{:H}) = 1 / sqrt(2) * (qubit_operator(c, QubitOp{:Z}()) + qubit_operator(c, QubitOp{:X}())) # Hadamard operator


@testitem "Qubit operators" begin
    using SparseArrays, Random, LinearAlgebra
    using FermionicHilbertSpaces: QubitOperators
    Random.seed!(1234)

    N = 2
    H = hilbert_space(1:N)
    B = QubitOperators(H)
    @test B[1] isa SparseMatrixCSC
    @test B[1] + B[1]' ≈ B[1, :X]
    @test I - 2B[1]'B[1] ≈ B[1, :Z]
    @test 1im * (B[1]' - B[1]) ≈ B[1, :Y]

    H = hilbert_space(1:3)
    a = QubitOperators(H)
    Hs = (hilbert_space(1:1), hilbert_space(2:2), hilbert_space(3:3))
    v = [FermionicHilbertSpaces.indtofock(i, H) for i in 1:8]
    t1 = reshape(v, H, Hs)
    t2 = [i1 + 2i2 + 4i3 for i1 in (0, 1), i2 in (0, 1), i3 in (0, 1)]
    @test t1 == FockNumber.(t2)

    v2 = rand(8)
    H1 = hilbert_space(1:1)
    H2 = hilbert_space(2:3)
    H = tensor_product(H1, H2)
    @test sort(svdvals(reshape(v2, H, (H1, H2), false)) .^ 2) ≈ eigvals(partial_trace(v2 * v2', H, H1, false))

    ## Test that partial trace is the adjoint of embedding
    using LinearMaps
    ptmap = LinearMap(rhovec -> vec(partial_trace(reshape(rhovec, size(H)), H, H1, false)), prod(size(H1)), prod(size(H)))
    embeddingmap = LinearMap(rhovec -> vec(embedding(reshape(rhovec, size(H1)), H1, H, false)), prod(size(H)), prod(size(H1)))
    @test Matrix(ptmap) ≈ Matrix(embeddingmap)'

end