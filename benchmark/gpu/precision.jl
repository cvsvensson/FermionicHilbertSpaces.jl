# Times the CPU build plus upload, the hybrid routes and the native device assembly of gpu-native-prototype, in single
# and double precision. Needs an environment where FermionicHilbertSpaces is the gpu-native-prototype branch:
#     julia --project=<env> precision.jl <path to benchmark/gpu/native/FHSGPU.jl> [--large]
include("hybrid.jl")
using KernelAbstractions: KernelAbstractions
include(ARGS[1])
using .FHSGPU: DeviceCSCRepr

function precision_cases(large)
    @fermions f
    @boson b
    hubbard_nc(N) = (; name = "hubbard half filling, $(2N) modes", op = hubbard(f, N), H = hilbert_space(f, hubbard_modes(N), NumberConservation(N)), kwargs = (;))
    return [
        hubbard_nc(8), hubbard_nc(10), (large ? [hubbard_nc(11), hubbard_nc(12)] : [])...,
        (; name = "fermions ⊗ boson, K = 12, nmax = 16", op = holstein(f, b, 12), H = tensor_product(hilbert_space(f, 1:12), hilbert_space(b, 16)), kwargs = (;)),
    ]
end

# Fewer runs than `besttime` and `devtime`, to keep the whole script short
best(f) = @belapsed $f() samples = 3 evals = 1 seconds = 30
best_build(build, ijv, d) = @belapsed ($build(xs..., $d, $d); CUDA.synchronize()) setup = (xs = map(copy, $ijv)) samples = 3 evals = 1 seconds = 30

# Device time of the kernels whose names contain each of `parts`, from the profiler, minimum over runs
function kernel_times(f, parts; runs = 3)
    tmin = fill(Inf, length(parts))
    for _ in 1:runs
        d = (CUDA.@profile f()).device
        tmin .= min.(tmin, [sum(d.stop[k] - d.start[k] for k in eachindex(d.name) if contains(d.name[k], p); init = 0.0) for p in parts])
    end
    return tmin
end

host_csc(S) = SparseMatrixCSC(S.m, S.n, Int.(Array(S.colptr)), Int.(Array(S.rowval)), Array(S.nzval))

function run_case((; name, op, H, kwargs), T)
    println("\n", name, ", dim ", dim(H), ", ", T)
    Mref = representation(op, H; kwargs...)
    M = representation(op, H, SparseMatrixCSC{T}; kwargs...)
    check(label, X) = isapprox(SparseMatrixCSC(X), Mref; rtol = sqrt(eps(real(T)))) || @warn "$label differs from representation(op, H)"
    tcpu = best(() -> representation(op, H, SparseMatrixCSC{T}; kwargs...))
    tup = best(() -> CUDA.@sync CuSparseMatrixCSR(M))
    report("CPU serial", tcpu)
    report("baseline: CPU serial + upload CSR", tcpu + tup)
    tijv = best(() -> host_ijv(op, H, T; kwargs...))
    ijv = host_ijv(op, H, T; kwargs...)
    tupijv = best(() -> CUDA.@sync upload(CuArray, ijv))
    dijv = upload(CuArray, ijv)
    report("hybrid: host IJV + upload, $(length(ijv[3])) entries", tijv + tupijv)
    ijv = nothing
    for (label, repr) in FORMATS
        try
            check("hybrid $label", repr.build(map(copy, dijv)..., dim(H), dim(H)))
            report("  + device build, $label", best_build(repr.build, dijv, dim(H)))
        catch err
            err isa CUDA.OutOfGPUMemoryError || rethrow()
            report("  + device build, $label: out of memory", NaN)
        end
        GC.gc(); CUDA.reclaim()
    end
    dijv = nothing; GC.gc(); CUDA.reclaim()
    native = DeviceCSCRepr{T}(CUDA.CUDABackend())
    nat() = representation(op, H, native; kwargs...)
    check("native", host_csc(nat()))
    report("native end to end", best(() -> (nat(); CUDA.synchronize())))
    tcol, tcompact, tgather = kernel_times(nat, ("column_kernel", "compact_kernel", "gather_kernel"))
    report("  column kernel", tcol)
    report("  compaction (sort/merge + gather)", tcompact + tgather)
    GC.gc(); CUDA.reclaim()
end

for case in precision_cases("--large" in ARGS), T in (Float64, Float32)
    run_case(case, FHS.mat_eltype(case.op) <: Complex ? complex(T) : T)
end
