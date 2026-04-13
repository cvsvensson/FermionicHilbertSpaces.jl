"""
    Jaynes-Cummings-Hubbard Model Tutorial

A conversion of the QuTiP JCHM tutorial to FermionicHilbertSpaces.jl.

The Jaynes-Cummings-Hubbard Model describes an array of optical cavities,
each containing a two-level atom, with photons able to tunnel between
neighboring cavities.

Hamiltonian:
H = ωc Σᵢ aᵢ† aᵢ + (ωa/2) Σᵢ σᵢᶻ + g Σᵢ (aᵢ† σᵢ⁻ + aᵢ σᵢ⁺) - J Σᵢ (aᵢ† aᵢ₊₁ + h.c.)

where:
  - ωc: cavity frequency
  - ωa: atomic transition frequency
  - g: atom-cavity coupling strength
  - J: hopping strength between adjacent cavities
  - aᵢ, aᵢ†: photon annihilation/creation operators
  - σᵢᶻ, σᵢ⁺, σᵢ⁻: Pauli and ladder operators
"""

using FermionicHilbertSpaces
using LinearAlgebra, Plots
using Statistics

# ============================================================================
# Helper: Create JCHM Hamiltonian for N sites
# ============================================================================
function create_jchm_hamiltonian(b, s, N_sites::Int, N_fock::Int, ωc, ωa, g, J)
    """
    Create the Hamiltonian for an N-site Jaynes-Cummings-Hubbard model.
    
    Args:
        N_sites: Number of cavity-atom sites
        N_fock: Number of Fock states per cavity
        ωc: Cavity frequency
        ωa: Atomic transition frequency
        g: Atom-cavity coupling strength
        J: Hopping strength between cavities
        
    Returns:
        H_sym: Symbolic Hamiltonian
        Hilbert_spaces: Dict of necessary Hilbert spaces
        operators: Dict of measurement operators
    """
    # Create boson and spin fields 
    ham = 0
    for i in 1:N_sites
        ham += ωc * b[i]' * b[i] # Cavity energy
        ham += (ωa / 2) * s[i][:z] # Atomic energy (σᶻ/2)
        ham += 2 * g * (b[i]'s[i][:-] + hc) # Atom-cavity coupling
    end
    for i in 1:(N_sites-1)
        ham += -J * (b[i]'b[i+1] + hc)
    end

    symoperators = Dict(
        "cavity_n" => [b[i]'b[i] for i in 1:N_sites],  # Photon numbers
        "atom_z" => [s[i][:z] for i in 1:N_sites],          # Atomic z-components
        "atom_e" => [s[i][:+]'s[i][:+] for i in 1:N_sites],  # Excitation (|↑⟩⟨↑|)
        #"cavity_a" => [mat(b[i]) for i in 1:N_sites],           # Field operators   
    )
    # operators = Dict(
    #     "cavity_n" => [mat(b[i]' * b[i]) for i in 1:N_sites],  # Photon numbers
    #     "atom_z" => [mat(s[i][:z]) for i in 1:N_sites],          # Atomic z-components
    #     "atom_e" => [mat(s[i][:+]' * s[i][:+]) for i in 1:N_sites],  # Excitation (|↑⟩⟨↑|)
    #     #"cavity_a" => [mat(b[i]) for i in 1:N_sites],           # Field operators   
    # )

    return ham, symoperators
end



# ============================================================================
# Helper: Compute ground state
# ============================================================================

function compute_ground_state(symham, H)
    ham = matrix_representation(symham, H)
    evals, evecs = eigen(Hermitian(Matrix(ham)))
    E0 = evals[1]
    ψ0 = evecs[:, 1]
    return E0, ψ0, evals
end
function sector_compute_ground_state(symham, H)
    data = map(sectors(H)) do H
        ham = matrix_representation(symham, H)
        evals, evecs = eigen(Hermitian(Matrix(ham)))
        E0 = evals[1]
        ψ0 = evecs[:, 1]
        return E0, ψ0, evals
    end
    E0, qnind = findmin(first, data)
    return E0, data[qnind][2], data[qnind][3], quantumnumbers(H)[qnind]
end


# ============================================================================
# Helper: Compute order parameters
# ============================================================================

function compute_order_parameters(symham, H, ops, N_sites)
    """
    Compute order parameters that characterize the ground state.
    
    Returns:
        delta_n_avg: Average photon number fluctuation (signature of superfluid phase)
        alpha_avg: Average cavity field amplitude (zero in finite systems)
        photon_numbers: Per-site photon numbers
        atom_excitations: Per-site atom excitations
    """
    E0, ψ0, _ = compute_ground_state(symham, H)
    # Compute quantities for each site
    mat = op -> matrix_representation(op, H)
    nop = map(mat, ops["cavity_n"])
    eop = map(mat, ops["atom_e"])
    zop = map(mat, ops["atom_z"])
    photon_numbers = [real(dot(ψ0, op, ψ0)) for op in nop]
    delta_n_list = [sqrt(real(dot(ψ0, op^2, ψ0) - photon_numbers[i]^2)) for (i, op) in enumerate(nop)]
    # alpha_list = [abs(dot(ψ0, ops["cavity_a"][i], ψ0)) for i in 1:N_sites]
    atom_excitations = [real(dot(ψ0, op, ψ0)) for op in eop]
    zlist = [real(dot(ψ0, op, ψ0)) for op in zop]
    # Average values
    delta_n_avg = mean(delta_n_list)
    # alpha_avg = mean(alpha_list)

    return delta_n_avg, photon_numbers, atom_excitations, E0, zlist
end


# ============================================================================
# SECTION A: Ground State Analysis
# ============================================================================

println("="^70)
println("JCHM Tutorial: Jaynes-Cummings-Hubbard Model")
println("="^70)

# System parameters (from QuTiP tutorial)
N_sites = 3              # Number of cavity-atom sites
N_fock = 5               # Fock dimension per cavity
ωc = 1.0                 # Cavity frequency (energy scale)
ωa = 1.0                 # Atomic transition frequency (resonant with cavity)
g = 0.3                  # Atom-cavity coupling strength
J_baseline = 0.2         # Baseline hopping strength

@bosons b
@spins s 1 // 2
# Each site has a cavity (Fock space) and an atom (2-level system)
Ha = hilbert_space(b, 1:N_sites, N_fock)
Hs = hilbert_space(s, 1:N_sites)
_spin_qn(s) = (sum(s -> Int(s.m + 1 // 2), s.states))
_qn(s) = sum(FermionicHilbertSpaces.particle_number, s.states)
# constraint = FermionicHilbertSpaces.AdditiveConstraint(-100:100, [qn, spin_qn])
H = tensor_product(Ha, Hs)
constraint2 = FermionicHilbertSpaces.SectorConstraint([Ha, Hs], [_qn, _spin_qn], qns -> begin
    N = sum(qns)
end)
H = constrain_space(H, constraint2)
# println(H.qn_to_states)


println("\n>>> SECTION A: Ground State Properties")
println("System Parameters:")
println("  N_sites = $N_sites")
println("  N_fock = $N_fock")
println("  ωc = $ωc")
println("  ωa = $ωa")
println("  g = $g")
println("  J = $J_baseline")
println("  total Hilbert space dimension: $(N_fock * 2)^$N_sites = $((N_fock * 2)^N_sites)")
println("  sector Hilbert space dimension: $(dim(H))")

# Create Hamiltonian
ham, ops = create_jchm_hamiltonian(b, s, N_sites, N_fock, ωc, ωa, g, J_baseline)

#Compute ground state
E0, ψ0, evals, qn = sector_compute_ground_state(ham, H)

println("\nGround State Analysis:")
println("  Ground state energy: E₀ = $(round(E0, digits=6))")

# Compute expectation values
delta_n, photon_nums, atom_excits, E0, zlist = compute_order_parameters(ham, sector(qn, H), ops, N_sites)

println("\n  Per-site photon numbers:")
for i in 1:N_sites
    println("    Site $i: ⟨n⟩ = $(round(photon_nums[i], digits=4))")
end

println("\n  Per-site atomic excitations:")
for i in 1:N_sites
    println("    Site $i: ⟨σ⁺σ⁻⟩ = $(round(atom_excits[i], digits=4))")
end

println("\n  Order parameters:")
println("    Avg. photon fluctuation (Δn): $(round(delta_n, digits=6))")
# println("    Avg. cavity field amplitude (α): $(round(alpha, digits=6))")


# ============================================================================
# SECTION B: Phase Transition via Order Parameters
# ============================================================================

println("\n" * "="^70)
println(">>> SECTION B: Phase Transition Signatures")
println("="^70)

# Vary hopping strength to explore phase transition
J_values = range(0.01, 0.5, length=20)
delta_n_values = Float64[]
# alpha_values = Float64[]

println("\nComputing order parameters for varying J...")

for (idx, J_val) in enumerate(J_values)
    # Create new Hamiltonian with current J
    H_sym_J, ops_J = create_jchm_hamiltonian(b, s, N_sites, N_fock, ωc, ωa, g, J_val)

    # Compute order parameters
    delta_n, photon_nums, atom_excits, E0 = compute_order_parameters(H_sym_J, H, ops_J, N_sites)
    push!(delta_n_values, delta_n)
    # push!(alpha_values, alpha)

    print("\r  J = $(round(J_val, digits=3)): Δn = $(round(delta_n, digits=4))")
end

println("\n✓ Calculations complete!")

# Plot order parameters
fig = plot(
    J_values,
    delta_n_values,
    marker=:circle,
    linewidth=2,
    label="Photon number fluctuation (Δn)",
    xlabel="Hopping strength J",
    ylabel="Order parameter",
    title="Mott Insulator → Superfluid Phase Transition",
    legend=:topleft,
    size=(600, 400),
    ylims=(0, 1.1)
)

# Add phase boundary marker
vline!(fig, [0.2], linestyle=:dash, color=:red, label="Approximate critical J", alpha=0.7)

savefig(fig, joinpath(@__DIR__, "jchm_order_parameters.png"))
println("\n✓ Plot saved to jchm_order_parameters.png")

# Print interpretation
println("\n🔬 Physical Interpretation:")
println("  • Mott insulator phase (small J): Photons localized, low Δn")
println("  • Superfluid phase (large J): Photons delocalized, increasing Δn")
println("  • Finite size effects: α remains ~0 (symmetry preserved)")
println("  • The transition around J ≈ 0.2 marks the crossover region")

println("\n" * "="^70)
println("Tutorial Complete!")
println("="^70)
