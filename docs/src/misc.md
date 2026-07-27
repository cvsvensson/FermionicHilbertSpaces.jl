
# Misc

## Subregion
When the Hilbert space has a restricted set of fock states, the Hilbert space of the subregion will only include fock states compatible with this restriction. In the example below, the total number of particles 1, and the subregion will have three possible states: (1,0), (0,1), and (0,0).
```@example subregion
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(1))
Hsub = subregion(hilbert_space(f, 1:2), H)
basisstates(Hsub)
``` 

## Parallelization

Matrix construction can be parallelized by passing a chunking strategy to `matrix_representation`. The two supported options are `TermChunking(scheduler)`, which splits work by operator terms, and `StateChunking(scheduler)`, which splits work by basis states. Any [OhMyThreads](https://github.com/JuliaFolds2/OhMyThreads.jl) scheduler can be used, for example `StaticScheduler`.

```@example parallelization
using FermionicHilbertSpaces, OhMyThreads
using FermionicHilbertSpaces: TermChunking, StateChunking
@fermions f
H = hilbert_space(f, 1:4)
op = f[1]' * f[2] + 1im * f[2]' * f[1] + f[3]' * f[4] + f[4]' * f[3] + 2

scheduler = StaticScheduler(; nchunks=4)
M1 = matrix_representation(op, H; chunking=TermChunking(scheduler))
M2 = matrix_representation(op, H; chunking=StateChunking(scheduler))
```

For distributed sparse matrices, use `PartitionedSparseRepr` with a [PartitionedArrays](https://github.com/PartitionedArrays/PartitionedArrays.jl) backend:
```julia
using FermionicHilbertSpaces, PartitionedArrays
using FermionicHilbertSpaces: PartitionedSparseRepr
@fermions f
space = hilbert_space(f, 1:3)
op = f[1]' * f[2] + (1 + 2im) * f[3]' * f[1] + 2.0

rep = PartitionedSparseRepr(; backend=DebugArray)
M = matrix_representation(op, space, rep)
```
To use [MPI](https://github.com/JuliaParallel/MPI.jl), start julia with MPI (`mpiexec -n 8 julia`) and do
```julia
using MPI
MPI.Init()
rep = PartitionedSparseRepr(; backend=distribute_with_mpi, nparts=MPI.Comm_size(MPI.COMM_WORLD))
matrix_representation(op, space, rep)
```


## Reshaping

`reshape` converts arrays by splitting or combining axes according to Hilbert-space mappings. Mappings are applied left-to-right, and each mapping consumes the next consecutive group of input axes.

```julia
reshape(t::AbstractArray, mappings...; repeat=false)
```

Supported mapping forms:

- `H => (H1, H2, ...)`: split one axis into several
- `(H1, H2, ...) => H`: combine several consecutive axes into one
- `H => H`: keep one axis unchanged

Many-to-many mappings are not supported directly. Use a combine step followed by a split step.

### Multiple mappings

Provide one mapping per axis group:

```julia
reshape(A, H1 => (H1a, H1b), (H2a, H2b) => H2, H3 => H3)
```

This splits the first axis, combines the next two, and keeps the last unchanged.

### Single-mapping shorthand

If exactly one mapping is provided, it is expanded automatically:

- If it consumes all axes of `t`, it is applied once.
- If `t` has twice as many axes as the mapping consumes, the mapping is applied to the first half and again to the second half.

This is convenient for square operator matrices:

```julia
reshape(v, H => (H1, H2))
reshape(m, H => (H1, H2))
reshape(T, (H1, H2) => H)
```

For asymmetric operators, specify both sides explicitly:

```julia
reshape(m, Hout => (H1, H2), Hin => (K1, K2))
```

### Repeating one mapping

With `repeat=true`, one mapping is applied to every consecutive compatible axis group:

```julia
reshape(A, H => (H1, H2); repeat=true)
```

### Pair syntax and curried form

Mappings use `Pair` syntax and can be passed directly:

```julia
reshape(m, H => (H1, H2))
reshape(T, (H1, H2) => H)
```

`reshape` also accepts mappings first and returns a function:

```julia
to_tensor = reshape(H => (H1, H2))
to_matrix = reshape((H1, H2) => H)
```
This does not provide any performance benefit at the moment, but might in the future.

### Practical usage

For worked examples, see [docs/src/literate_output/open_system_lindblad.md](literate_output/open_system_lindblad.md).


## State mapper interface

Internal tensor/reshape/partial-trace routines use a common mapper protocol:

- `state_mapper(H, Hs)` returns a mapper object.
- `split_state(state, mapper)` returns a tuple with one entry per target subsystem.
- Each tuple entry is a weighted collection `((substate, weight), ...)`.
- `combine_states(substates, mapper)` returns a weighted collection `((state, weight), ...)`.

This package does not require a single concrete container type for weighted collections; callers should treat them as iterable collections of `(state, weight)` outcomes.