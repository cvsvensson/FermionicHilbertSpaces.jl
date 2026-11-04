# Operator matrices on the GPU

Untested prototypes that compare ways to get a matrix from `representation` onto an NVIDIA GPU. Run from this directory:

    julia --project -e 'using Pkg; Pkg.instantiate()'
    julia --project -t auto baseline.jl [--large]
    julia --project -t auto hybrid.jl [--large] [--cpu]

- `cases.jl` holds the operators and spaces (full Fock space, number-conserving sector, fermions ⊗ boson with and without a constraint); `--large` picks bigger sizes.
- `baseline.jl` times `representation` on the CPU (serial and with `StateChunking`), the IJV generation and a single `sparse` on those entries, and the upload with `CuSparseMatrixCSR(M)` or `CuArray(M)` for small spaces.
- `hybrid.jl` defines `DeviceSparseRepr`, which generates the entries on the CPU, uploads them with `Int32` indices and builds the CSR matrix on the device, with cuSPARSE's `sparse(I, J, V, m, n; fmt = :csr)` or with an AcceleratedKernels sort and segment sum. It times each step and checks the result against `representation(op, H)`. `--cpu` runs only the AcceleratedKernels assembly, on host arrays.
- On a product space without a constraint, `representation` uses Kronecker products, while `DeviceSparseRepr` and the IJV timings use the loop over basis states.
