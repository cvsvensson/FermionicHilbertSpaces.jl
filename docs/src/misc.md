
# Misc


## Parallelization

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