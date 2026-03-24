using FermionicHilbertSpaces
using FermionicHilbertSpaces: GenericHilbertSpace
include("floquet_algebra.jl")
##
@spin σ
floquet_basis = FloquetBasis(0)
F = FloquetLadder(floquet_basis)
N = FloquetNumber(1, floquet_basis)
##
function Heff(ħω, V)
    h0 = σ[:z]
    hp1 = V * σ[:x] * F
    hm1 = V * σ[:x] * F'
    h0 + ħω * N + 1 / ħω * (hp1 * hm1 - hm1 * hp1) + 1 / ħω * (h0 * hp1 - hp1 * h0) - 1 / ħω * (h0 * hm1 - hm1 * h0)
end
##
Hfloc = GenericHilbertSpace(floquet_basis, FloquetState.(-1:1))
Hspin = hilbert_space(σ, 1 // 2)
H = tensor_product((Hspin, Hfloc))
ham = Heff(10, 1)
mat = matrix_representation(Heff(10, 1), H; projection=true)
partial_trace(mat, H => Hfloc)