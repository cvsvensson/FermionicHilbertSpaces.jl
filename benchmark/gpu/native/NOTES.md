# Native device assembly: notes

Status: untested. Nothing here has been run yet, on any backend.

Run with `julia --project=benchmark/gpu/native` and `include("benchmark/gpu/native/runall.jl")`. The default backend is
JLArrays; `ENV["FHS_GPU_BACKEND"] = "cpu"` selects the KernelAbstractions CPU backend, and `using CUDA` before the
include selects CUDA (add CUDA to the environment first). Each stage script compares the dense and the CSC device
matrices with `representation(op, H, Matrix{T})`.

## The generic pattern

- One kernel, `column_kernel!`, with one thread per column. For each term the thread calls
  `FHS.chunked_operator_indices_and_amplitudes_generic!` on the range `j:j`, the CPU loop itself: `basisstate`,
  `_apply_local_operators`, `state_index`, `_skip_outside`, `push_inds_amps!`. Each term gives at most one entry per
  column, so term `k` writes slot `k` of column `j` and threads never share memory.
- The output is a sink with a `push_inds_amps!` method: `DenseSink` adds into a dense device matrix, and `SlotSink`
  fills an `nslots × n` slot matrix. A second kernel sorts each column and sums duplicates in place, and a third copies
  the columns into CSC arrays after a prefix sum of the column counts.
- Terms are prepared on the host as on the CPU: split by group with `partition_product` on spaces with several groups,
  factors made into tuples with the existing `TupleConcretizer`, and the per-factor precomputation from
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

- `projection=false` with an operator that leaves the space throws inside the kernel. On CUDA this surfaces as a
  kernel exception without the FHS message.
- Unsupported, with an isbits error: `GenericHilbertSpace`, Majoranas, BdG and single-particle spaces, `TransposedSpace`
  and symbolic coefficients. `Axes(Hout, Hin)` is not supported. Products whose groups have no `internal_rep` take the
  slow product path, which indexes a heterogeneous tuple at run time and will not compile on a GPU.
- The adjoint of a device `SpinSym` would be wrong; only the transposed action needs it, and the kernel does not use it.
- The slot matrix has `nterms + 1` rows for every column, which is too much memory for many terms on a large space;
  columns would need to be assembled in chunks.
- Boson and spin amplitudes are computed in `Float64`, so Metal would fail.

## Likely first failures

- Exceptions compiled into device code: `_skip_outside`'s `ArgumentError`, and `throw_boundserror` with a space
  argument on backends other than CUDA.
- `_check_state_kind` for a term that changes the state type (`FixedNumberFockState` with another particle number)
  relies on `parentmodule` and `nameof` of types folding to constants.
- Inference of the recursion over term groups and of the product fast path inside a kernel, which must be free of
  dynamic dispatch.
- `adapt(JLBackend(), ::Array)` and KA kernels on JLArrays, which depend on the JLArrays version.
