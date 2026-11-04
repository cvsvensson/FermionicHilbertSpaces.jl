# Loaded by the stage scripts. The backend is CUDA if `using CUDA` ran before, the KernelAbstractions CPU backend
# if ENV["FHS_GPU_BACKEND"] == "cpu", and JLArrays otherwise.
using FermionicHilbertSpaces, SparseArrays, LinearAlgebra, Test
using KernelAbstractions: KernelAbstractions
include(joinpath(@__DIR__, "FHSGPU.jl"))
using .FHSGPU: device_matrix, device_csc

if isdefined(Main, :CUDA)
    const backend = Main.CUDA.CUDABackend()
elseif get(ENV, "FHS_GPU_BACKEND", "") == "cpu"
    const backend = KernelAbstractions.CPU()
else
    using JLArrays
    const backend = JLArrays.JLBackend()
end

host_csc(S) = SparseMatrixCSC(S.m, S.n, Int.(Array(S.colptr)), Int.(Array(S.rowval)), Array(S.nzval))

# Compares the dense and sparse device matrices with the CPU matrix.
function check(op, H; T=Float64, projection=false)
    ref = representation(op, H, Matrix{T}; projection)
    @test Array(device_matrix(op, H, backend, T; projection)) ≈ ref
    S = host_csc(device_csc(op, H, backend, T; projection))
    @test S ≈ ref
    @test all(j -> issorted(view(rowvals(S), nzrange(S, j))), axes(S, 2))
end
