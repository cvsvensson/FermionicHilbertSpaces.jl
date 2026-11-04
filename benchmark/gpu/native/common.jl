# Loaded by the stage scripts. The backend is CUDA or JLArrays if `using CUDA` or `using JLArrays` ran before, and
# the KernelAbstractions CPU backend otherwise.
using FermionicHilbertSpaces, SparseArrays, LinearAlgebra, Test
using KernelAbstractions: KernelAbstractions
include(joinpath(@__DIR__, "FHSGPU.jl"))
using .FHSGPU: DeviceDenseRepr, DeviceCSCRepr

const backend = isdefined(Main, :CUDA) ? Main.CUDA.CUDABackend() :
                isdefined(Main, :JLArrays) ? Main.JLArrays.JLBackend() : KernelAbstractions.CPU()

host_csc(S) = SparseMatrixCSC(S.m, S.n, Int.(Array(S.colptr)), Int.(Array(S.rowval)), Array(S.nzval))

# Compares the dense and sparse device matrices with the CPU matrix.
function check(op, H; T=Float64, projection=false)
    ref = representation(op, H, Matrix{T}; projection)
    @test Array(representation(op, H, DeviceDenseRepr{T}(backend); projection)) ≈ ref
    S = host_csc(representation(op, H, DeviceCSCRepr{T}(backend); projection))
    @test S ≈ ref
    @test all(j -> issorted(view(rowvals(S), nzrange(S, j))), axes(S, 2))
end
