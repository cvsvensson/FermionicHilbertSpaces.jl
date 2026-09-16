# pairing_crossover.jl
#
# Exact diagonalization of the finite constant-pairing Hamiltonian
#
#     H = Σ_iσ ε_i n_iσ − G P†P ,    P† = Σ_i c†_{i↑} c†_{i↓}
#
# Seniority ν (the number of levels holding exactly one fermion) is conserved,
# so each block is diagonalized separately:
#
#     (N,   ν=0)  fully paired even  → ground state + pair vibration
#     (N,   ν=2)  one broken pair    → pair-breaking gap
#     (N∓1, ν=1)  odd neighbours     → three-point odd-even gap
#
# Also used: sparse operators, KrylovKit lowest eigenpairs, warm starts between
# couplings, online fidelity, and the package-native partial trace for S_A.
#
# Dependencies: FermionicHilbertSpaces, KrylovKit, LinearAlgebra, Printf, Plots

using FermionicHilbertSpaces
using FermionicHilbertSpaces: BranchConstraint
using KrylovKit, LinearAlgebra, Printf, Plots

# ------------------------------------------------------------------ parameters
L, N, d, CUT = 12, 6, 1.0, 2                       # levels, particles, spacing, |A|
ε  = d .* collect(-(L - 1) / 2:(L - 1) / 2)       # symmetric picket fence
Gs = collect(range(0.0, 1.5d, length = 41))
TOL, MAXIT, KDIM1, KDIM2 = 1e-8, 1000, 40, 60
@assert iseven(N) && 0 < N < 2L && 1 <= CUT < L

E_FG = 2sum(ε[1:N ÷ 2])                           # frozen Fermi sea, G = 0

# ----------------------------------------------------------------------- modes
# Level i holds the time-reversed pair of modes (2i−1, 2i). The ordering
# up₁, dn₁, up₂, dn₂, … is exactly what the seniority constraint relies on.
up(i) = 2i - 1
dn(i) = 2i
ALL  = collect(1:2L)
SUBA = collect(1:2CUT)
@fermions c

# --------------------------------------------------------- seniority constraint
"""
    seniority(ν)

BranchConstraint retaining states with exactly ν singly occupied levels. A level
counts only once both of its modes have been assigned, i.e. at even depth.
"""
function seniority(ν)
    return BranchConstraint() do partials, depth, spaces
        nlevels = length(spaces) ÷ 2
        done    = depth ÷ 2
        ν_now   = 0
        for i in 1:done
            ν_now += xor(count_ones(partials[2i - 1]) == 1,
                         count_ones(partials[2i]) == 1)
        end
        ν_now > ν && return false                        # already exceeded ν
        ν_now + (nlevels - done) < ν && return false     # cannot reach ν
        return depth == length(spaces) ? ν_now == ν : true
    end
end

"""Analytic dimension of the (Np, ν) sector."""
function sector_dim(L, Np, ν)
    @assert 0 <= ν <= L && ν <= Np && iseven(Np - ν) && (Np - ν) ÷ 2 <= L - ν
    return binomial(big(L), ν) * big(2)^ν * binomial(big(L - ν), (Np - ν) ÷ 2)
end

# ------------------------------------------------------------ Hilbert spaces
sector(Np, ν) = hilbert_space(c, ALL, NumberConservation(Np) * seniority(ν))

HN,  HN2 = sector(N, 0),     sector(N, 2)          # even: paired, broken pair
HNm, HNp = sector(N - 1, 1), sector(N + 1, 1)      # odd neighbours
HA = subregion(hilbert_space(c, SUBA), HN)                        # unrestricted subsystem
# HA = subregion([c[up(i)] for i in 1:L], HN)                        # unrestricted subsystem

names    = ("N ν=0", "N ν=2", "N−1 ν=1", "N+1 ν=1")
dims     = [dim(H) for H in (HN, HN2, HNm, HNp)]
expected = [sector_dim(L, N, 0), sector_dim(L, N, 2),
            sector_dim(L, N - 1, 1), sector_dim(L, N + 1, 1)]
@assert dims == expected
for (nm, D, e) in zip(names, dims, expected)
    @printf("%-8s dim = %4d   expected = %4d\n", nm, D, e)
end

# --------------------------------------------------------- symbolic operators
num(m)  = c[m]' * c[m]
pair(i) = c[up(i)]' * c[dn(i)]'

P    = sum(pair(i) for i in 1:L)                        # collective pair creation
H0   = sum(ε[i] * (num(up(i)) + num(dn(i))) for i in 1:L)
V    = -(P * P')                                        # H(G) = H0 + G V
Dp   = sum(pair(i) * pair(i)' for i in 1:L)             # diagonal part of P†P
NA   = sum(num(m) for m in SUBA)                        # particles in subsystem A
nlev = [0.5 * (num(up(i)) + num(dn(i))) for i in 1:L]   # 0 ≤ ⟨n_i⟩ ≤ 1

# ------------------------------------------------------ sparse representations
sm(op, H) = representation(op, H, :sparse)

H0N,  VN  = sm(H0, HN),  sm(V, HN)
H0N2, VN2 = sm(H0, HN2), sm(V, HN2)
H0Nm, VNm = sm(H0, HNm), sm(V, HNm)
H0Np, VNp = sm(H0, HNp), sm(V, HNp)
DpN, NAN  = sm(Dp, HN),  sm(NA, HN)
occ_mats  = [sm(nlev[i], HN) for i in 1:L]

DN, DN2, DNm, DNp = size(H0N, 1), size(H0N2, 1), size(H0Nm, 1), size(H0Np, 1)

# ------------------------------------------------------------------ utilities
expect(ψ, O) = real(dot(ψ, O * ψ))

function entropy_bits(ρ, tol = 1e-12)
    λ = max.(real.(eigvals(Hermitian(Matrix(ρ)))), 0.0)
    λ ./= sum(λ)
    return -sum(x * log2(x) for x in λ if x > tol)
end

# KrylovKit returns eigenvectors either as a Vector of vectors or as an n×k
# matrix, depending on version; normalise both to a Vector of unit vectors.
function asvectors(vs)
    v = ndims(vs) == 2 ? [vs[:, j] for j in 1:size(vs, 2)] : vs
    return [normalize(x) for x in v]
end

"""Lowest `nev` eigenpairs of Hermitian H, ascending in energy."""
function lowest(H, x0, nev, krylovdim)
    vals, vecs, _ = eigsolve(H, x0, nev, :SR; ishermitian = true, tol = TOL,
                             krylovdim = max(nev + 2, krylovdim), maxiter = MAXIT)
    p = sortperm(real.(vals))
    return real.(vals[p]), asvectors(vecs)[p]
end

lowest1(H, x0) = ((vals, vecs) = lowest(H, x0, 1, KDIM1); (vals[1], vecs[1]))
randvec(D)     = (v = randn(ComplexF64, D); v / norm(v))

# ---------------------------------------------------------------------- sweep
ptmap = partial_trace(HN => HA)
NG    = length(Gs)

E0    = zeros(NG); Δvib = zeros(NG); Δbrk = zeros(NG); Δoe = zeros(NG)
Ppair = zeros(NG); Pdiag = zeros(NG); Pcoh = zeros(NG)
S_A   = zeros(NG); nA_mean = zeros(NG); nA_var = zeros(NG)
occ   = zeros(NG, L)
Gmid  = 0.5 .* (Gs[1:end-1] .+ Gs[2:end])
chiF  = zeros(NG - 1)

ψ0_prev = ψ2_prev = ψm_prev = ψp_prev = nothing
ptmap = partial_trace(HN => HA)
@profview for (k, G) in enumerate(Gs)
    # Warm starts are read here and rewritten at the end of the iteration, so
    # they must be declared global inside this loop.
    global ψ0_prev, ψ2_prev, ψm_prev, ψp_prev

    start0 = ψ0_prev === nothing ? randvec(DN) : ψ0_prev
    vals, (ψ0, ψvib) = lowest(H0N + G * VN, start0, 2, KDIM2)
    E0[k], Δvib[k]   = vals[1], vals[2] - vals[1]

    if k > 1
        δG = G - Gs[k - 1]
        F  = clamp(abs(dot(ψ0_prev, ψ0)), 0.0, 1.0)
        chiF[k - 1] = 2 * (1 - F) / δG^2
    end

    Ppair[k] = -expect(ψ0, VN)                 # ⟨P†P⟩ = −⟨V⟩
    Pdiag[k] = expect(ψ0, DpN)                 # Σ_i ⟨P_i†P_i⟩
    Pcoh[k]  = Ppair[k] - Pdiag[k]             # off-diagonal pair transfer
    for i in 1:L
        occ[k, i] = expect(ψ0, occ_mats[i])
    end

    NAψ = NAN * ψ0                             # ⟨N_A²⟩ = ‖N_Aψ‖²
    nA_mean[k] = real(dot(ψ0, NAψ))
    nA_var[k]  = max(real(dot(NAψ, NAψ)) - nA_mean[k]^2, 0.0)

    S_A[k] = entropy_bits(ptmap(ψ0))           # package-native partial trace
    # S_A[k] = entropy_bits(partial_trace(ψ0, HN => HA))           # package-native partial trace

    E2, ψ2 = lowest1(H0N2 + G * VN2, ψ2_prev === nothing ? randvec(DN2) : ψ2_prev)
    Em, ψm = lowest1(H0Nm + G * VNm, ψm_prev === nothing ? randvec(DNm) : ψm_prev)
    Ep, ψp = lowest1(H0Np + G * VNp, ψp_prev === nothing ? randvec(DNp) : ψp_prev)
    Δbrk[k] = E2 - E0[k]
    Δoe[k]  = 0.5 * (Ep + Em - 2*E0[k])

    global ψ0_prev = ψ0
    global ψ2_prev = ψ2
    global ψm_prev = ψm
    global ψp_prev = ψp

    if k == 1 || k % div(length(Gs),5) == 0 || k == NG
        @printf("G/d = %5.3f   E0/d = %10.6f   S_A = %7.4f bits   Δbrk/d = %7.4f\n",
                G / d, E0[k] / d, S_A[k], Δbrk[k] / d)
    end
end

# ------------------------------------------------------------------- analysis
xc   = Gmid[argmax(chiF)] / d
corr = [E0[k] - (E_FG - Gs[k] * (N ÷ 2)) for k in 1:NG]     # vs frozen Fermi sea

# Hellmann–Feynman check: dE0/dG = −⟨P†P⟩, central differences in the interior.
dEdG   = [2 <= k <= NG - 1 ? -(E0[k + 1] - E0[k - 1]) / (Gs[k + 1] - Gs[k - 1]) :
                             NaN for k in 1:NG]
hf_err = maximum(abs.(dEdG[2:end-1] .- Ppair[2:end-1]))
n_err  = maximum(abs.(2 .* vec(sum(occ, dims = 2)) .- N))

@printf("\ncrossover Gc/d = %.4f   max χ_F d² = %.4f\n", xc, maximum(chiF) * d^2)
@printf("max |particle number error| = %.2e   max |Hellmann–Feynman error| = %.2e\n\n",
        n_err, hf_err)

@printf("%8s %12s %10s %10s %10s %10s\n",
        "G/d", "E0/d", "Δvib/d", "Δbrk/d", "Δ(3)/d", "C_pair")
for k in unique(round.(Int, range(1, NG, length = 7)))
    @printf("%8.3f %12.6f %10.6f %10.6f %10.6f %10.6f\n",
            Gs[k] / d, E0[k] / d, Δvib[k] / d, Δbrk[k] / d, Δoe[k] / d, Pcoh[k])
end

# ------------------------------------------------------------------- plotting
default(linewidth = 2.2, framestyle = :box, gridalpha = 0.25, legendfontsize = 8,
        guidefontsize = 10, tickfontsize = 8, titlefontsize = 10)

x   = Gs ./ d
cols = palette(:viridis, L)
mark!(p) = vline!(p, [xc]; label = "", color = :gray, linestyle = :dot, linewidth = 1.2)

p1 = plot(x, S_A; label = "S_A", color = :navy, xlabel = "G/d", ylabel = "bits",
          title = "Entanglement / number fluctuations")
plot!(p1, x, nA_var; label = "Var(N_A)", color = :darkorange, linestyle = :dash); mark!(p1)
plot!(p1, x[1:end-1], diff(S_A) ./ diff(x); label = "∂S_A", color = :navy, linestyle = :dash)

p2 = plot(x, Pcoh; label = "off-diagonal", color = :crimson, xlabel = "G/d",
          ylabel = "pair expectation", title = "Pair correlations")
plot!(p2, x, Ppair; label = "⟨P†P⟩", color = :purple, linestyle = :dash)
plot!(p2, x, dEdG;  label = "−dE₀/dG", color = :black, linestyle = :dot); mark!(p2)

p3 = plot(x, Δvib ./ d; label = "ν = 0 vibration", color = :forestgreen, xlabel = "G/d",
          ylabel = "gap / d", title = "Excitation gaps")
plot!(p3, x, Δbrk ./ d; label = "ν = 2 breaking", color = :royalblue, linestyle = :dashdot)
plot!(p3, x, Δoe ./ d;  label = "Δ⁽³⁾(N)", color = :red, linestyle = :dash); mark!(p3)

p4 = plot(x, corr ./ d; label = "E₀ − E_frozen", color = :darkcyan, xlabel = "G/d",
          ylabel = "energy / d", title = "Correlation energy")
hline!(p4, [0.0]; label = "", color = :black, linestyle = :dot); mark!(p4)

p5 = plot(xlabel = "G/d", ylabel = "occupation per mode", legend = :right,
          ylims = (-0.03, 1.03), title = "Single-particle occupations")
for i in 1:L
    plot!(p5, x, occ[:, i]; label = "ε$(i)/d = $(ε[i] / d)", color = cols[i])
end
mark!(p5)

p6 = plot(Gmid ./ d, chiF .* d^2; label = "χ_F", color = :magenta4, fill = (0, 0.15),
          xlabel = "G/d", ylabel = "χ_F d²", title = "Fidelity susceptibility")
vline!(p6, [xc]; label = "G_c/d = $(@sprintf("%.3f", xc))", color = :black, linestyle = :dash)

summary = plot(p1, p2, p3, p4, p5, p6; layout = (3, 2), size = (1100, 1100), margin = 4Plots.mm)
savefig(summary, "pairing_crossover.png")
display(summary)

# ---------------------------------------------------------------- CSV output
open("pairing_crossover_data.csv", "w") do io
    println(io, "G,E0,dvib,dbrk,doe,pair,pair_diag,pair_coh,S_A,NA_mean,NA_var,",
            join("occ_" .* string.(1:L), ","))
    for k in 1:NG
        println(io, join((Gs[k], E0[k], Δvib[k], Δbrk[k], Δoe[k], Ppair[k], Pdiag[k],
                          Pcoh[k], S_A[k], nA_mean[k], nA_var[k], occ[k, :]...), ","))
    end
end
