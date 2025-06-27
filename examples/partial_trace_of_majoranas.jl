## Partial traces of many-body Majoranas
# In this example, we will look at partial traces of many-body Majorana operators

using FermionicHilbertSpaces: majoranas
using Combinatorics

# First, let's define the full Hilbert space `H`, and the subregion `R`,
H = hilbert_space(1:5)
R = hilbert_space((1, 3, 4))
# and the Majorana operators in `H`.
Γ = majoranas(H)
# Then let's test a couple of properties. If the Majorana extends outside of `R`, it vanishes under the partial trace.
γRbar = Γ[2, :+] * Γ[5, :-] * Γ[2, :-]
@test norm(partial_trace(γRbar, H => R)) ≈ 0
γRRbar = Γ[1, :-] * Γ[3, :+] * Γ[2, :-]
@test norm(partial_trace(γRRbar, H => R)) ≈ 0

# If the Majorana is contained in `R`, its partial trace is the same as if expressed in the smaller basis `ΓR`, up to multiplication by dim(R̄).
ΓR = majoranas(R)
labels_combs = combinations(collect(keys(ΓR)))
for labels in labels_combs
    γR = mapreduce(l -> Γ[l...], *, labels)
    γR2 = mapreduce(l -> ΓR[l...], *, labels)
    @test partial_trace(γR, H => R) ≈ 4 * γR2 # dim(R̄) = 4
end





