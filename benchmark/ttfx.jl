# Time-to-first-use benchmarks. Each sample starts a fresh Julia process with
# the active project and runs a short script in it, so the times include
# Julia's own startup and one sample takes several seconds. These benchmarks
# are kept out of `SUITE` (and therefore out of AirspeedVelocity runs);
# `compare.jl run --suite=ttfx` runs them.
using BenchmarkTools

const TTFX_SUITE = BenchmarkGroup()

const TTFX_LOAD = "using FermionicHilbertSpaces"
const TTFX_FIRST_USE = """
using FermionicHilbertSpaces
@fermions f
H = hilbert_space(f, 1:4, NumberConservation(2))
op = sum(f[n]' * f[n+1] + hc for n in 1:3) + sum(f[n]' * f[n] for n in 1:4)
m = matrix_representation(op, H)
partial_trace(Matrix(m), H => subregion(hilbert_space(f, 1:2), H))
@spins S 1//2
Hs = hilbert_space(S, 1:4)
matrix_representation(sum(S[k][:z] * S[k+1][:z] for k in 1:3), Hs)
"""

ttfx_command(code) = pipeline(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`; stdout=devnull)

# `run` on a benchmark first evaluates it once untimed, so any precompilation
# of the active project happens outside the samples.
for (key, code) in ("load" => TTFX_LOAD, "first_use" => TTFX_FIRST_USE)
    TTFX_SUITE[key] = @benchmarkable run($(ttfx_command(code))) samples = 3 evals = 1 seconds = 120
end
