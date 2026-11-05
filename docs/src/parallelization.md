# Parallelization

Matrix construction can be parallelized by passing a chunking strategy to `representation`: `TermChunking(scheduler)` or `StateChunking(scheduler)`, with an [OhMyThreads](https://github.com/JuliaFolds2/OhMyThreads.jl) scheduler that has a fixed number of chunks and splits consecutively, for example `StaticScheduler(; nchunks)` or `DynamicScheduler(; nchunks)`. A scheduler without a fixed `nchunks` runs on a single chunk. Each chunk has a cost proportional to the dimension of the space, so the number of chunks should be about the number of threads. For sparse and dense matrices the two strategies do the same: the columns are split into one range per chunk, and the ranges are built in parallel. On a product space the terms are built in parallel instead. The result equals the serial one. For lazy operators, `TermChunking` splits the product with a vector by terms and `StateChunking` by basis states.

```@example parallelization
using FermionicHilbertSpaces, OhMyThreads
using FermionicHilbertSpaces: TermChunking, StateChunking
@fermions f
H = hilbert_space(f, 1:4)
op = f[1]' * f[2] + 1im * f[2]' * f[1] + f[3]' * f[4] + f[4]' * f[3] + 2

scheduler = StaticScheduler(; nchunks=4)
M1 = representation(op, H; chunking=TermChunking(scheduler))
M2 = representation(op, H; chunking=StateChunking(scheduler))
```

For distributed sparse matrices, use `PartitionedSparseRepr` with a [PartitionedArrays](https://github.com/PartitionedArrays/PartitionedArrays.jl) backend:
```julia
using FermionicHilbertSpaces, PartitionedArrays
using FermionicHilbertSpaces: PartitionedSparseRepr
@fermions f
space = hilbert_space(f, 1:3)
op = f[1]' * f[2] + (1 + 2im) * f[3]' * f[1] + 2.0

rep = PartitionedSparseRepr(; backend=DebugArray)
M = representation(op, space, rep)
```
To use [MPI](https://github.com/JuliaParallel/MPI.jl), start julia with MPI (`mpiexec -n 8 julia`) and do
```julia
using MPI
MPI.Init()
rep = PartitionedSparseRepr(; backend=distribute_with_mpi, nparts=MPI.Comm_size(MPI.COMM_WORLD))
representation(op, space, rep)
```

## GPU

`DeviceCSCRepr` and `DeviceDenseRepr` assemble the matrix on a [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) backend, with the same per-state code as on the CPU. They need AcceleratedKernels in the environment, alongside the GPU package.
```julia
using FermionicHilbertSpaces, CUDA, AcceleratedKernels, LinearAlgebra
@fermions f
H = hilbert_space(f, 1:20, NumberConservation(10))
op = sum(f[i]' * f[i+1] + hc for i in 1:19)
S = representation(op, H, DeviceCSCRepr{ComplexF32}(CUDABackend()))
x = CUDA.rand(ComplexF32, dim(H))
y = mul!(similar(x), S, x)
```
On CUDA, `S` is a `CuSparseMatrixCSC`. A backend without a device sparse type gives a `SparseMatrixCSC` on the host, assembled on the device.

Supported are fermions, bosons, spins and their products, with number and parity conservation, sectors, listed states and constraints. Not supported, with an error that names the field without a device version: `GenericHilbertSpace`, Majoranas, BdG and single-particle spaces, `TransposedSpace` and symbolic coefficients. Also not supported are `Axes(Hout, Hin)` and products whose groups have no `internal_rep`.

- With `projection=false`, an operator that leaves the space throws inside the kernel, and on an asynchronous backend the error appears when the result is used, without the message of the CPU version.
- The sparse assembly allocates one row per term, plus one for a scalar term, for every column, so many terms on a large space can run out of device memory.
- Boson and spin amplitudes are computed in `Float64`, so a backend without `Float64` support fails on them.
- The kernels inline more fully with `CUDABackend(always_inline=true)`, which makes product spaces with spins or bosons faster.
- The index and states of the space are uploaded on every call.
