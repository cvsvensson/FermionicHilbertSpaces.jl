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

## GPUs

With a GPU package loaded, a GPU array type as the format assembles the matrix on the GPU, with one thread per basis
state. The type selects the backend, the matrix type and, optionally, the element type:
```julia
using FermionicHilbertSpaces, CUDA
@fermions f
H = hilbert_space(f, 1:20, NumberConservation(10))
op = sum(f[i]' * f[i + 1] + hc for i in 1:19) + sum(f[i]' * f[i] * f[i + 1]' * f[i + 1] for i in 1:19)
M = representation(op, H, cuSPARSE.CuSparseMatrixCSR{ComplexF32})
Md = representation(op, H, CuArray)
```
This works for fermions, bosons, spins, their products and constrained subspaces of them. Other spaces and operators,
products of groups without an integer internal representation, `Axes(Hout, Hin)` and symbolic coefficients throw an
`ArgumentError`. Sparse formats index with their index type (`Int32` by default) and throw an `ArgumentError` when the
matrix may need more entries than it can count. Operator coefficients keep their type on the GPU, usually `Float64`, and
boson and spin matrix elements are computed in `Float64`, which Apple GPUs do not support.

`partial_trace`, `partial_transpose`, `embed`, `reshape`, `change_basis`, `generalized_kron` and `tensor_product` of dense GPU
arrays also run on the GPU. GPU sparse arrays throw
an `ArgumentError` there.
