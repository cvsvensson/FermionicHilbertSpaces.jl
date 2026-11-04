# Untested prototype: written without being run.
# Generates the (I, J, V) entries on the CPU, uploads them, and assembles the CSR matrix on the device, with
# cuSPARSE's `sparse` or with AcceleratedKernels. Checks both against `representation(op, H)`.
#     julia --project -t auto hybrid.jl [--large] [--cpu]
# `--cpu` runs the AcceleratedKernels assembly on host arrays instead, to check it without a GPU.
include("cases.jl")
using CUDA
using CUDA.cuSPARSE: CuSparseMatrixCSR
import AcceleratedKernels as AK

"""
    DeviceSparseRepr(A, build)

Format for `representation` that collects the entries on the host, uploads them as arrays of type `A` with `Int32`
indices, and returns `build(I, J, V, m, n)`. All spaces go through the basis-state loop, also product spaces.
"""
struct DeviceSparseRepr{A, F}
    build::F
end
DeviceSparseRepr(A, build) = DeviceSparseRepr{A, typeof(build)}(build)

FHS._canonicalize_repr(repr::DeviceSparseRepr) = repr
function FHS._matrix_representation(op::FHS.NCAdd, bases, space, repr::DeviceSparseRepr{A}, chunking; kwargs...) where {A}
    return repr.build(upload(A, host_ijv(op, bases, space, chunking; kwargs...))..., dim(space), dim(space))
end

upload(A, (rows, cols, vals)) = (A(Int32.(rows)), A(Int32.(cols)), A(vals))

# CSR arrays (rowptr, colval, nzval) of the m×n matrix with the entries `V` at `(I, J)`, duplicates summed.
# Uses only AcceleratedKernels and broadcasting, so it runs on any KernelAbstractions backend; a GPUArrays v13
# sparse CSR type could wrap the result in place of the backend-specific one.
function ak_csr(I, J, V, m, n)
    keys = (I .- 1) .* Int64(n) .+ (J .- 1)
    vals = copy(V)
    # Stable, so duplicates are summed in the order of the terms as on the CPU
    AK.sort_by_key!(keys, vals)
    head = similar(keys, Bool)
    AK.foreachindex(head) do k
        head[k] = k == 1 || keys[k] != keys[k - 1]
    end
    starts = AK.findall(head)
    L, nnz = length(keys), length(starts)
    colval = similar(I, nnz)
    nzval = similar(V, nnz)
    # An entry has at most one duplicate per term, so each entry sums its run serially
    AK.foreachindex(nzval) do s
        k = starts[s]
        stop = s == nnz ? L : starts[s + 1] - 1
        acc = vals[k]
        for q in (k + 1):stop
            acc += vals[q]
        end
        nzval[s] = acc
        colval[s] = keys[k] % n + 1
    end
    rows = keys[starts] .÷ n .+ 1
    queries = similar(rows, m + 1)
    AK.foreachindex(queries) do r
        queries[r] = r
    end
    rowptr = similar(I, m + 1)
    AK.searchsortedfirst!(rowptr, rows, queries)
    return rowptr, colval, nzval
end

# CSR arrays of a matrix are the CSC arrays of its transpose
host_csr_matrix((rowptr, colval, nzval), m, n) =
    copy(transpose(SparseMatrixCSC(n, m, Vector{Int}(rowptr), Vector{Int}(colval), Vector(nzval))))

const ON_CPU = "--cpu" in ARGS
const sync = ON_CPU ? Returns(nothing) : CUDA.synchronize
const FORMATS = if ON_CPU
    ["AK on host arrays" => DeviceSparseRepr(Array, (I, J, V, m, n) -> host_csr_matrix(ak_csr(I, J, V, m, n), m, n))]
else
    [
        "cuSPARSE sparse" => DeviceSparseRepr(CuArray, (I, J, V, m, n) -> sparse(I, J, V, m, n; fmt = :csr)),
        "AK" => DeviceSparseRepr(CuArray, (I, J, V, m, n) -> CuSparseMatrixCSR{eltype(V), Int32}(ak_csr(I, J, V, m, n)..., (m, n))),
    ]
end
const A = ON_CPU ? Array : CuArray

# cuSPARSE's `sparse` may sort its inputs in place, so each run gets fresh copies
devtime(build, ijv, d) = @belapsed ($build(xs..., $d, $d); $sync()) setup = (xs = map(copy, $ijv)) samples = 5 evals = 1 seconds = 30

function run_case((; name, op, H, kwargs))
    d = dim(H)
    println("\n", name, ", dim ", d)
    Mref = representation(op, H; kwargs...)
    report("host IJV, serial", besttime(() -> host_ijv(op, H; kwargs...)))
    report("host IJV, $(Threads.nthreads()) threads", besttime(() -> host_ijv(op, H; chunking = threaded(), kwargs...)))
    ijv = host_ijv(op, H; kwargs...)
    report("upload Int32 IJV, $(length(last(ijv))) entries", besttime(() -> (upload(A, ijv); sync())))
    dijv = upload(A, ijv)
    for (label, repr) in FORMATS
        report("  device build, $label", devtime(repr.build, dijv, d))
        M = representation(op, H, repr; chunking = threaded(), kwargs...)
        SparseMatrixCSC(M) ≈ Mref || error("$label differs from representation(op, H) on $name")
        report("end to end with threads, $label", besttime(() -> (representation(op, H, repr; chunking = threaded(), kwargs...); sync())))
    end
    GC.gc()
    ON_CPU || CUDA.reclaim()
end

foreach(run_case, cases("--large" in ARGS))
