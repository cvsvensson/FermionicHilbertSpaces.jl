# Native device assembly: notes

Status: all six stages pass on the KernelAbstractions 0.10 CPU backend, on JLArrays and on CUDA.

Run with `julia --project=benchmark/gpu/native` and `include("benchmark/gpu/native/runall.jl")`. The backend is CUDA
or JLArrays if `using CUDA` or `using JLArrays` ran before the include, and the KernelAbstractions CPU backend
otherwise. JLArrays 0.4 requires KernelAbstractions 0.9, so it needs its own environment; CUDA 6 and the CPU backend
use 0.10. Each stage script compares `representation(op, H, DeviceDenseRepr{T}(backend))` and
`DeviceCSCRepr{T}(backend)` with `representation(op, H, Matrix{T})`.

## Entry point

`DeviceDenseRepr(backend)` and `DeviceCSCRepr(backend)` are formats for `representation`, with an optional element
type as in `DeviceDenseRepr{Float32}(backend)`. They plug into FHS through `_canonicalize_repr`, `_repr_eltype` and
`_terms_matrix`, which FHS calls with the terms of a sum, already split by group on product spaces. Two more methods
cover what FHS builds without `_terms_matrix`: `_term_matrix_representation` for a single term, which forwards to
`_terms_matrix`, and `_convert_repr_eltype` for a number, which uploads `c * I` from the host. Widening the
`_terms_matrix` hook in FHS to single terms would remove the first. On a product space, a coefficient of wider type
than the terms makes FHS add two results, which the CSC named tuple does not support; an explicit `T` avoids it.

The KernelAbstractions 0.10 CPU backend runs kernels through OpenCL (PoCL) and compiles them with GPUCompiler, so it
rejects non-isbits arguments, dynamic dispatch and runtime allocation as a GPU backend would. JLArrays runs the
kernels as plain Julia and mainly checks the array and `Adapt` plumbing of a GPUArrays backend.

## The generic pattern

- One kernel, `column_kernel!`, with one thread per column. For each term the thread calls
  `FHS.chunked_operator_indices_and_amplitudes_generic!` on the range `j:j`, the CPU loop itself: `basisstate`,
  `_apply_local_operators`, `state_index`, `_skip_outside`, `push_inds_amps!`. Each term gives at most one entry per
  column, so term `k` writes slot `k` of column `j` and threads never share memory.
- The output is a sink with a `push_inds_amps!` method: `DenseSink` adds into a dense device matrix, and `SlotSink`
  fills an `nslots × n` slot matrix. A second kernel sorts each column and sums duplicates in place, and a third copies
  the columns into CSC arrays after a prefix sum of the column counts.
- Terms arrive from FHS already split by group on spaces with several groups. On the host their factors are made into tuples with the existing `TupleConcretizer`, and the per-factor precomputation from
  `precomputation_before_operator_application`. Terms of equal type go into one device vector, and the kernel
  recurses over the tuple of these vectors.
- Device data comes from `Adapt.adapt_structure`. Each struct keeps the fields its methods read on the device and
  replaces the others by `nothing`: labels, bases, `Symbol`s, `Dict`s and mode lists only identify modes, which the
  precomputation has already resolved. Arrays become device arrays.
- Every `AbstractSubspace` becomes a `ConstrainedSpace` of its parent, states and index, since `basisstate` and
  `state_index` of a subspace use only that interface. `IndexTable` adapts as it is. `RootIndexDict` and the state
  `Dict` become a `SortedIndex` searched by binary search.
- Before upload, every term and the space are checked to be isbits once arrays are dropped. A type without a device
  version gives an `ArgumentError` that names the offending field.

## Generic hooks and type-specific pieces

Generic: the kernels, the sinks, term preparation, the subspace rule, the indices and the isbits check. They work for
any space and algebra whose structs adapt to isbits.

Type-specific: one `adapt_structure` method per operator and space type (FermionSym, BosonSym, SpinSym,
FermionicSpace, TruncatedBosonicHilbertSpace, SpinSpace, ProductSpace). This is data, not algebra: each method only
says which fields to keep. They would move to an FHS Adapt extension. No algebra has a device copy of its
`apply_local_operator(s)`.

## FHS src changes

- `FermionicSpace`, `ProductSpace`, `ProductOperator`, `IndexTable`: fields that held a concrete `Vector`, `Dict` or
  `OrderedDict` are type parameters, so a device copy can replace them. `FermionicSpace` and `ProductSpace` get an
  inner constructor that takes the fields as they are.
- `TruncatedBosonicHilbertSpace`: the field `sym::BosonSym{L,B}` was not concretely typed, so the struct could not be
  isbits. It now has the third type parameter of `BosonSym`.
- `SpinSpace`: the unused `state_index` Dict and the `basisstates` Vector are removed; `basisstate` and `dim` are
  arithmetic, as `state_index` already was.
- `SpinSym`: `op` is a type parameter, and `apply_local_operator` compares `_spin_op_char(op.op)`, so a device copy can
  store a `Char`. Its unreachable `throw` with an interpolated string is removed, because device code cannot build
  strings.
- `basisstate` of fermion, boson and spin spaces calls `Base.throw_boundserror`, which GPU backends override, instead of
  `throw(BoundsError(H, i))`, which boxes the space.

## Per stage

1. Fermions: `FermionSym` keeps `creation`; the precomputation already holds the bit mask.
2. `NumberConservation`: needed the subspace rule and `IndexTable`; large spaces need `SortedIndex` for `RootIndexDict`.
3. Parity, several sectors, listed states: no new code, since `SectorHilbertSpace` goes through the subspace rule.
   `FixedNumberFockState`: no new code; its precomputation is already isbits (sites and `Val(Δ)`), and its state
   `Dict` becomes a `SortedIndex` keyed by states.
4. Bosons: `BosonSym` keeps `exp`; the space needed the concrete `sym` field. Several modes form a `ProductSpace`,
   which needed the `ProductSpace` and `ProductOperator` changes.
5. Spins: the `SpinSym` and `SpinSpace` changes.
6. Mixed products: no new code. A constrained group inside a product works because the group spaces a
   `ProductOperator` carries are only used for dispatch and scalar fields, so their arrays are dropped.

## What still fails or is missing

- `projection=false` with an operator that leaves the space throws inside the kernel. On CUDA this surfaces when
  the result is used, as a kernel exception without the FHS message.
- Unsupported, with an isbits error: `GenericHilbertSpace`, Majoranas, BdG and single-particle spaces, `TransposedSpace`
  and symbolic coefficients. `Axes(Hout, Hin)` is not supported. Products whose groups have no `internal_rep` take the
  slow product path, which indexes a heterogeneous tuple at run time and will not compile on a GPU.
- The adjoint of a device `SpinSym` would be wrong; only the transposed action needs it, and the kernel does not use it.
- The slot matrix has `nterms + 1` rows for every column, which is too much memory for many terms on a large space;
  columns would need to be assembled in chunks.
- Boson and spin amplitudes are computed in `Float64`, so Metal would fail.

## Open questions

- Performance. On an NVIDIA T1000 (weak in `Float64`), `DeviceCSCRepr` for a number-conserving hopping model with
  interactions takes about as long as single-threaded `representation` on the host, both for 16 and for 20 modes at
  half filling. Host preparation is negligible; the column kernel and the compaction of the slot matrix take
  comparable time.
