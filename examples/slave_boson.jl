#
# Slave-boson (holon-spinon) representation of the U = ∞ Hubbard / t-J model
# on an L-site ring, built with FermionicHilbertSpaces.jl
#
#   c_{iσ} = b_i† f_{iσ}        subject to the LOCAL constraint
#   Q_i = n^f_{i↑} + n^f_{i↓} + n^b_i = 1     at every site i
#
#   H = -t Σ_<ij>σ ( f†_{iσ} b_i b†_j f_{jσ} + h.c. )
#       + J Σ_<ij> ( S_i·S_j - n_i n_j / 4 )
#
# Demonstrates: local (Gauss-law) constraints, constraint composition,
# mixed fermion-boson spaces, validation against a projected-fermion model,
# graded partial traces, species vs spatial entanglement.
#
using FermionicHilbertSpaces, LinearAlgebra

# ----------------------------------------------------------------- parameters
L = 12          # sites on a ring
t = 1.0
J = 0.2        # set J = 0 for the pure U = ∞ Hubbard model
NHOLE = 1          # number of holes  ->  N_electrons = L - NHOLE
NUP = 5          # up-spin count (fixes S_z = (2*NUP - (L-NHOLE))/2)

bonds = [(i, mod1(i + 1, L)) for i in 1:L]

vn(λs) = -sum(λ -> λ > 1e-12 ? λ * log(λ) : 0.0, λs)   # von Neumann entropy

# ============================================================================
# 1.  SLAVE-BOSON HILBERT SPACE
# ============================================================================
@fermions f                                  # spinons: spin, no charge
@bosons b
Hup(i) = hilbert_space(f, [(i, :↑)])
Hdn(i) = hilbert_space(f, [(i, :↓)])
Hhol(i) = hilbert_space(b[i], 2)         # occupations 0..1

# --- the local Gauss law, one constraint per site ---------------------------
gauss(i) = NumberConservation(1, [Hup(i), Hdn(i), Hhol(i)])

# --- global sectors ---------------------------------------------------------
gauss_law = prod(gauss(i) for i in 1:L)                       # L local constraints
hole_num = NumberConservation(NHOLE, [Hhol(i) for i in 1:L]) # total charge
sz_sector = NumberConservation(NUP, [Hup(i) for i in 1:L]) # total S_z

factors = collect(Iterators.flatten((Hup(i), Hdn(i), Hhol(i)) for i in 1:L))

Hsb_full = tensor_product(factors; constraint=gauss_law)
Hsb = tensor_product(factors; constraint=gauss_law * hole_num * sz_sector)

println("unconstrained dimension        : ", 8^L)
println("Gauss law only  (expect 3^L)   : ", dim(Hsb_full), "  vs  ", 3^L)
println("+ charge & S_z sector          : ", dim(Hsb),
    "  vs  ", L * binomial(L - NHOLE, NUP))

# ============================================================================
# 2.  HAMILTONIAN  (written generically so it works for f and for c)
# ============================================================================
nn(o, i) = o[i, :↑]' * o[i, :↑] + o[i, :↓]' * o[i, :↓]
sz(o, i) = (o[i, :↑]' * o[i, :↑] - o[i, :↓]' * o[i, :↓]) / 2
sp(o, i) = o[i, :↑]' * o[i, :↓]
sm(o, i) = o[i, :↓]' * o[i, :↑]

spin_exchange(o) = J * sum(
    sz(o, i) * sz(o, j) +
        (sp(o, i) * sm(o, j) + sm(o, i) * sp(o, j)) / 2 -
        nn(o, i) * nn(o, j) / 4
    for (i, j) in bonds)

# correlated hop: a spinon and a holon must move together
hop_sb = -t * sum(f[i, σ]' * b[i] * b[j]' * f[j, σ] + hc
                  for (i, j) in bonds for σ in (:↑, :↓))

ham_sb = Hermitian(representation(hop_sb + spin_exchange(f), Hsb, :dense))
E_sb, V_sb = eigen(ham_sb)
println("\nslave-boson ground energy      : ", E_sb[1])

# ============================================================================
# 3.  VALIDATION: same physics with projected physical fermions, no bosons
# ============================================================================
@fermions c
Hcup(i) = hilbert_space(c, [(i, :↑)])
Hcdn(i) = hilbert_space(c, [(i, :↓)])

no_double = prod(NumberConservation(0:1, [Hcup(i), Hcdn(i)]) for i in 1:L)
cfactors = Tuple(Iterators.flatten((Hcup(i), Hcdn(i)) for i in 1:L))

Hph = tensor_product(cfactors;
    constraint=no_double *
        NumberConservation(L - NHOLE) *
        NumberConservation(NUP, [Hcup(i) for i in 1:L]))

hop_ph = -t * sum(c[i, σ]' * c[j, σ] + hc
                  for (i, j) in bonds for σ in (:↑, :↓))

E_ph = eigvals(Hermitian(representation(hop_ph + spin_exchange(c), Hph, :dense; projection=true)))

println("dim(slave-boson) == dim(phys)  : ", dim(Hsb) == dim(Hph))
println("spectra agree                  : ", isapprox(sort(E_sb), sort(E_ph); atol=1e-9))
println("max |ΔE|                       : ", maximum(abs, sort(E_sb) .- sort(E_ph)))

# ============================================================================
# 4.  SPECIES ENTANGLEMENT  (spinons vs holons)  -- and why it is trivial
# ============================================================================
ψ = V_sb[:, 1]
ρ = ψ * ψ'

Hspinons = subregion([f[i, σ] for i in 1:L for σ in (:↑, :↓)], Hsb)
ρ_spin = partial_trace(ρ, Hsb => Hspinons)
ρ_spin ./= tr(ρ_spin)                       # normalisation guard

S_species = vn(eigvals(Hermitian(ρ_spin)))

# The constraint makes n^b_i = 1 - n^f_i an operator identity, so the Schmidt
# coefficients are just the hole-position probabilities.  Check it:
p_hole = [real(ψ' * Matrix(matrix_representation(b[i]' * b[i], Hsb)) * ψ)
          for i in 1:L]

println("\n--- species (spinon:holon) entanglement ---")
println("S_species                      : ", S_species)
println("Shannon entropy of hole positions: ", vn(p_hole))
println("log(L)                         : ", log(L))
println("hole distribution              : ", round.(p_hole; digits=6))
# => all three numbers coincide, for ANY t and J.  Purely kinematic.

# ============================================================================
# 5.  SPATIAL ENTANGLEMENT  (this one carries physics)
#     region A = sites 1..LA, including both spinons and holons there
# ============================================================================
LA = 3
modesA = vcat([f[i, σ] for i in 1:LA for σ in (:↑, :↓)], [b[i] for i in 1:LA])
HA = subregion(modesA, Hsb)

ρA = partial_trace(ρ, Hsb => HA)
ρA ./= tr(ρA)
S_A = vn(eigvals(Hermitian(ρA)))

# number-resolved split: S_A = S_num + S_conf
NA = Hermitian(representation(sum(nn(f, i) for i in 1:LA), Hsb, :dense))
λ, W = eigen(NA)
amps = abs2.(W' * ψ)
pN = Dict{Int,Float64}()
for (val, w) in zip(round.(Int, real.(λ)), amps)
    pN[val] = get(pN, val, 0.0) + w
end
S_num = vn(collect(values(pN)))
S_conf = S_A - S_num

println("\n--- spatial cut (sites 1..$LA) ---")
println("S_A                            : ", S_A)
println("S_num  (charge fluctuations)   : ", S_num)
println("S_conf (spin/configurational)  : ", S_conf)
println("P(N_A)                         : ", sort(collect(pN)))

# ============================================================================
# 6.  DYNAMICS: quench a localised holon and watch charge and spin spread
# ============================================================================
## project the ground state onto "hole on site 1", renormalise, then evolve
# i0 = div(L, 2)
# P1 = representation(b[i0]' * b[i0], Hsb)
# ψ0 = P1 * ψ;
# ψ0 ./= norm(ψ0)
# nb = [representation(b[i]' * b[i], Hsb) for i in 1:L]
# szi = [representation(sz(f, i), Hsb) for i in 1:L]


# ##
# chs = []
# sps = []
# τs = range(0, 10, 100)
# for τ in τs
#     ψt = V_sb * (cis.(-E_sb * τ) .* (V_sb' * ψ0))
#     ch = [real(ψt' * nb[i] * ψt) for i in 1:L]
#     sp_ = [real(ψt' * szi[i] * ψt) for i in 1:L]
#     push!(chs, ch)
#     push!(sps, sp_)
# end

# ## plot the spreading holon and spin
# using Plots
# fig1 = heatmap(1:L, τs, hcat(chs...)', title="hole density ⟨n_b⟩", xlabel="site", ylabel="τ", clims=(0, 1))
# fig2 = heatmap(1:L, τs, hcat(sps...)', title="spin ⟨Sz⟩", xlabel="site", ylabel="τ", clims=(-0.5, 0.5), c=:balance)
# plot(fig1, fig2)

##
using Plots

# Ground state
ψ = V_sb[:, 1]
E0 = E_sb[1]

# Local operators
nb = [representation(b[i]' * b[i], Hsb) for i in 1:L]
szi = [representation(sz(f, i), Hsb) for i in 1:L]

# Put the excitation near the middle of the chain/ring
i0 = div(L, 2)

τs = range(0.0, 20.0; length=301)
Nt = length(τs)

# --------------------------------------------------------------------------
# A. Charge propagation: project the hole onto site i0
# --------------------------------------------------------------------------

ψhole0 = nb[i0] * ψ
@assert norm(ψhole0) > 1e-12
ψhole0 ./= norm(ψhole0)

# Expansion coefficients in the energy eigenbasis
ahole = V_sb' * ψhole0

hole_density = zeros(Float64, Nt, L)

for (it, τ) in enumerate(τs)
    ψt = V_sb * (cis.(-E_sb .* τ) .* ahole)

    for i in 1:L
        hole_density[it, i] = real(dot(ψt, nb[i] * ψt))
    end
end

# --------------------------------------------------------------------------
# B. Spin propagation: connected longitudinal spin correlator
#
# Czz(i,t) = Re <GS| δSz_i exp[-i(H-E0)t] δSz_i0 |GS>
#
# This stays in the same total-Sz sector, so no new Hilbert space is needed.
# --------------------------------------------------------------------------
# Local ground-state magnetization
sz_gs = [real(dot(ψ, szi[i] * ψ)) for i in 1:L]

# δSz_i0 |GS>
ϕ0 = szi[i0] * ψ - sz_gs[i0] * ψ
#ϕ0 = ψhole0
spin_weight = real(dot(ϕ0, ϕ0))
@assert spin_weight > 1e-12

aspin = V_sb' * ϕ0

spin_corr = zeros(ComplexF64, Nt, L)
spin_response = zeros(Float64, Nt, L)

for (it, τ) in enumerate(τs)
    ϕt = V_sb * (cis.(-E_sb .* τ) .* aspin)
    phase0 = cis(E0 * τ)

    for i in 1:L
        # <GS|δSz_i(t)δSz_i0(0)|GS>
        # Sink subtraction is unnecessary analytically because
        # <GS|ϕt> = 0, but can be included explicitly if desired.
        C = phase0 * dot(ψ, szi[i] * ϕt) #/ spin_weight

        spin_corr[it, i] = C

        # Retarded longitudinal spin response
        spin_response[it, i] = 2 * imag(C)
    end
end


# --------------------------------------------------------------------------
# Optional: remove the alternating antiferromagnetic sign.
#
# The signed Czz plot is physically complete, but neighboring sites often
# have opposite signs. Multiplying by (-1)^(i-i0) makes the envelope/front
# easier to recognize.
# --------------------------------------------------------------------------
#
# The signed Czz plot is physically complete, but neighboring sites often
# have opposite signs. Multiplying by (-1)^(i-i0) makes the envelope/front
# easier to recognize.
# --------------------------------------------------------------------------

spin_staggered = similar(spin_corr)

for i in 1:L
    spin_staggered[:, i] .= (-1)^(i - i0) .* spin_corr[:, i]
end

spin_response_staggered = similar(spin_response)

for i in 1:L
    spin_response_staggered[:, i] .=
        (-1)^(i - i0) .* spin_response[:, i]
end


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
##
fig_charge = heatmap(
    1:L, τs, hole_density;
    xlabel="site",
    ylabel="τ",
    title="hole density  ⟨nᵇᵢ(τ)⟩",
    clims=(0.0, 0.5),
    color=:inferno,
)

# Saturating the large signal at the initial point makes the weaker
# propagating part visible. Try 0.15--0.30 if necessary.
spin_lim = 0.25

# fig_spin = heatmap(
#     1:L, τs, spin_staggered;
#     xlabel="site",
#     ylabel="τ",
#     title="staggered connected spin propagator",
#     clims=(-spin_lim, spin_lim),
#     color=:balance,
# )

fig_spin_abs = heatmap(
    1:L, τs, abs.(spin_corr);
    xlabel="site",
    ylabel="τ",
    title="|Cᶻᶻ(i,τ)|",
    clims=(0.0, spin_lim),
    color=:viridis,
)

spin_lim = maximum(abs, spin_response_staggered[2:end, :])

fig_response = heatmap(
    1:L, τs, spin_response_staggered;
    xlabel="site",
    ylabel="τ",
    title="retarded spin response  χᶻᶻ(i,τ)",
    color=:balance,
    clims=(-spin_lim, spin_lim),
)

fig_response_abs = heatmap(
    1:L, τs, abs.(spin_response);
    xlabel="site",
    ylabel="τ",
    title="|χᶻᶻ(i,τ)|",
    color=:viridis,
    clims=(0, spin_lim),
)

# plot(fig_response, fig_response_abs;
#  layout=(1, 2), size=(950, 400))


plot(fig_charge, fig_response, fig_response_abs;
    layout=(1, 3), size=(1350, 400))