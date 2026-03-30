# # SYK Model Tutorial

# In this example, we study the Sachdev-Ye-Kitaev (SYK) model for Majorana fermions.
# We build a random all-to-all 4-body Hamiltonian, diagonalize it in one parity sector,
# and compute simple diagnostics of chaotic behavior and entanglement.
using FermionicHilbertSpaces
using FermionicHilbertSpaces.NonCommutativeProducts: add!!
using LinearAlgebra, Random, Statistics

function syk_hamiltonian(g, M; J=1.0)
    iseven(M) || throw(ArgumentError("M must be even"))
    prefactor = sqrt(6) * J / M^(3 / 2)
    ham = 0
    for i in 1:M-3, j in i+1:M-2, k in j+1:M-1, l in k+1:M
        ham = add!!(ham, prefactor * randn() * g[i] * g[j] * g[k] * g[l])
    end
    return ham
end

function adjacent_gap_ratio(energies)
    spacings = diff(energies)
    rs = [min(s1, s2) / max(s1, s2) for (s1, s2) in zip(spacings[1:end-1], spacings[2:end]) if max(s1, s2) > 1e-12]
    mean(rs)
end

function entanglement_entropy(psi, Hsub, H)
    rho = partial_trace(psi * psi', H => Hsub)
    vals = filter(>(1e-12), eigvals(Hermitian(Matrix(rho))))
    -sum(vals .* log.(vals))
end

M = 8 * 2
@majoranas g

H = hilbert_space(g, 1:M, ParityConservation(1))
Random.seed!(11)
symham = syk_hamiltonian(g, M)
ham = Hermitian(Matrix(matrix_representation(symham, H)))
vals_even, vecs_even = eigen(ham)

##
Hleft = subregion(hilbert_space(g, 1:M÷2), H)

println("SYK model with M = $M Majoranas")
println("Hilbert-space dimension (even sector): $(dim(H))")
println("Ground-state energy (even): $(vals_even[1])")
println("Mean adjacent gap ratio: $(adjacent_gap_ratio(vals_even))") #should be roughly 0.53
println("Half-system entanglement entropy: $(entanglement_entropy(vecs_even[:,1], Hleft, H))") # should be close to 0.33841M/2
