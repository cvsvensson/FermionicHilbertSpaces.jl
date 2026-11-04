# Untested prototype: written without being run.
# Times the CPU matrix representation and its upload to the GPU as separate steps.
#     julia --project -t auto baseline.jl [--large]
include("cases.jl")
using CUDA
using CUDA.cuSPARSE: CuSparseMatrixCSR

# A dense matrix of this dimension takes 2 GiB in Float64
const MAX_DENSE_DIM = 2^14

function run_case((; name, op, H, kwargs))
    println("\n", name, ", dim ", dim(H))
    M = representation(op, H; kwargs...)
    report("CPU sparse, serial", besttime(() -> representation(op, H; kwargs...)))
    report("CPU sparse, $(Threads.nthreads()) threads", besttime(() -> representation(op, H; chunking = threaded(), kwargs...)))
    # FHS builds the matrix block by block of columns (or by kron on a product space), so these two do not add up to
    # the serial time; they show how the work of a single sort/merge on all entries would split.
    rows, cols, vals = host_ijv(op, H; kwargs...)
    report("  IJV generation, $(length(vals)) entries", besttime(() -> host_ijv(op, H; kwargs...)))
    report("  sparse on that IJV", besttime(() -> sparse(rows, cols, vals, size(M)...)))
    report("upload as CuSparseMatrixCSR", besttime(() -> CUDA.@sync CuSparseMatrixCSR(M)))
    if dim(H) <= MAX_DENSE_DIM
        D = representation(op, H, Matrix; kwargs...)
        report("CPU dense, serial", besttime(() -> representation(op, H, Matrix; kwargs...)))
        report("upload as CuArray", besttime(() -> CUDA.@sync CuArray(D)))
    end
    GC.gc()
    CUDA.reclaim()
end

foreach(run_case, cases("--large" in ARGS))
