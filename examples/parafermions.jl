# # Boundary modes of an open Z_p parafermion chain
#

using FermionicHilbertSpaces
import FermionicHilbertSpaces as FHS

using LinearAlgebra
using SparseArrays
using Printf
using Plots

trace_norm(A) = sum(svdvals(Matrix(A)))

# ---------------------------------------------------------------------
# Model conventions
#
# Local clock operators:
#
#   X |n> = |n+1 mod p>
#   Z |n> = ω^n |n>
#   Z X = ω X Z
#
# Construct X from the nilpotent local Fock annihilator c:
#
#   X = c† + c^(p-1).
#
# With the extension's sigma convention, embedding X produces
#
#   α_j = (∏_{k<j} Z_k^sigma) X_j.
#
# Define its partner
#
#   β_j = κ α_j Z_j^sigma,
#   κ   = exp[-iπ sigma (p-1)/p],
#
# so α_j^p = β_j^p = I.
#
# The open-chain Hamiltonian is
#
#   H = -J ∑_j (b β_j† α_{j+1} + h.c.)
#       -h ∑_j (Z_j + Z_j†),
#
#   b = κ ω^(-sigma).
#
# In clock variables this is exactly
#
#   H = -J ∑_j (X_j† X_{j+1} + h.c.)
#       -h ∑_j (Z_j + Z_j†).
#
# Thus the model is independent of the chosen braiding convention.
# At h=0, α_1 and β_N commute exactly with H.
#
# The default p=3, h/J=0.18 lies near the ordered, parafermionic
# topological limit. Do not assume an identical phase diagram for
# arbitrary p or arbitrary additional interactions.
# ---------------------------------------------------------------------

function build_parafermion_chain(;
    p::Int=3,
    N::Int=6,
    sigma::Int=1,
    J::Real=1.0,
)
    @assert p >= 2
    @assert N >= 2
    @assert sigma in (-1, 1)
    @assert J > 0

    ω = cis(2π / p)
    κ = cis(-π * sigma * (p - 1) / p)
    bondphase = κ * ω^(-sigma)

    c = FHS.parafermion_basis(:c, p; sigma=sigma)
    Hspace = hilbert_space(c, collect(1:N))
    Hsites = [hilbert_space(c, [j]) for j in 1:N]

    D = Int(dim(Hspace))
    identity_full = spdiagm(0 => ones(ComplexF64, D))
    identity_local = Matrix{ComplexF64}(I, p, p)

    # Obtain the cyclic shift from the local Fock representation.
    c_local = sparse(representation(c[1], Hsites[1]))
    Xlocal = sparse(c_local' + c_local^(p - 1))
    Zlocal = spdiagm(0 => ComplexF64[ω^n for n in 0:(p-1)])

    @assert isapprox(
        Matrix(Xlocal^p), identity_local; atol=1e-11
    )
    @assert isapprox(
        Zlocal * Xlocal, ω * Xlocal * Zlocal; atol=1e-11
    )

    # IMPORTANT: use the existing graded embedding.
    # An ordinary Kronecker product would omit the strings.
    α = [
        sparse(FHS.embed(Xlocal, Hsites[j], Hspace))
        for j in 1:N
    ]
    Z = [
        sparse(FHS.embed(Zlocal, Hsites[j], Hspace))
        for j in 1:N
    ]

    β = [
        sparse(κ * α[j] * (sigma == 1 ? Z[j] : Z[j]'))
        for j in 1:N
    ]

    for j in (1, N)
        @assert isapprox(α[j]^p, identity_full; atol=1e-10)
        @assert isapprox(β[j]^p, identity_full; atol=1e-10)
    end

    Hbond = spzeros(ComplexF64, D, D)
    for j in 1:(N-1)
        term = bondphase * β[j]' * α[j+1]
        Hbond -= J * (term + term')
    end

    Hfield = spzeros(ComplexF64, D, D)
    for j in 1:N
        Hfield -= Z[j] + Z[j]'
    end

    # Build charge-sector indices from the actual basis ordering.
    # No parafermionic conservation-law wrapper is assumed here.
    states = collect(FHS.basisstates(Hspace))
    charges = [FHS.parafermion_charge(state) for state in states]
    sector_indices = [
        findall(==(q), charges) for q in 0:(p-1)
    ]
    Q = spdiagm(0 => ComplexF64[ω^q for q in charges])

    # These assertions also catch inconsistent embedding phases.
    scale = max(norm(Hbond), 1.0)
    @assert norm(Hbond * α[1] - α[1] * Hbond) < 1e-10 * scale
    @assert norm(Hbond * β[end] - β[end] * Hbond) < 1e-10 * scale

    @assert norm(Q * α[1] - ω * α[1] * Q) < 1e-10 * sqrt(D)
    @assert norm(Q * β[end] - ω * β[end] * Q) < 1e-10 * sqrt(D)

    return (;
        p, N, sigma, J, ω,
        Hspace, Hsites, D,
        Hbond, Hfield, Q, sector_indices,
        left_seed=α[1],
        right_seed=β[end],
    )
end

# ---------------------------------------------------------------------
# Lowest state in each Z_p charge sector
#
# We diagonalize each block separately and lift its ground state into
# the full space, just as the Kitaev example does for parity sectors.
#
# At the default size the blocks are only 243 × 243.
# ---------------------------------------------------------------------

function charge_ground_states(model, ham)
    (; p, D, sector_indices) = model

    energies = zeros(Float64, p)
    next_energies = zeros(Float64, p)
    G = zeros(ComplexF64, D, p)

    for q in 0:(p-1)
        inds = sector_indices[q+1]
        block = Matrix(ham[inds, inds])
        solution = eigen(Hermitian(block))

        energies[q+1] = solution.values[1]
        next_energies[q+1] = solution.values[2]
        G[inds, q+1] = solution.vectors[:, 1]
    end

    @assert isapprox(G' * G, Matrix{ComplexF64}(I, p, p); atol=1e-10)

    # Positive separation means the selected p states form the
    # entire lowest band, below every other state.
    bandwidth = maximum(energies) - minimum(energies)
    separation = minimum(next_energies) - maximum(energies)

    return (; G, energies, bandwidth, separation)
end

# ---------------------------------------------------------------------
# Ground-manifold boundary transitions
#
# Let P = G G†. Define
#
#   Γ_L = P α_1 P,
#   Γ_R = P β_N P.
#
# Unlike constructing arbitrary sums |q+1><q|, this prescription is
# independent of eigenvector phase choices and fixes the two boundary
# transitions through their physical endpoint operators.
#
# No Hermitian conjugate is added: for p>2 these charged transitions
# are not Majorana operators.
# ---------------------------------------------------------------------

function boundary_profile(model, G, energies, seed)
    (; Hspace, Hsites, N) = model

    small_operator = G' * seed * G
    normalization = trace_norm(small_operator)

    normalization > 1e-12 ||
        error("The endpoint operator has negligible ground-band overlap.")

    # Same trace norm as the full projected operator, because G is
    # an isometry. Normalize so profiles are comparable.
    Γ = G * small_operator * G' / normalization

    reductions = [partial_trace(Γ, Hspace => Hsites[j]) for j in 1:N]
    profile = trace_norm.(reductions)

    # Exact commutator norm within the selected invariant band.
    # At finite size it need not vanish because the sectors split.
    E = Diagonal(energies)
    commutator = norm(E * small_operator - small_operator * E) /
        max(norm(small_operator), eps(Float64))

    return (; profile, commutator)
end

# ---------------------------------------------------------------------
# Local distinguishability
#
# For each site j:
#
#   LD_j = max_{q<r} 1/2 ||ρ_q^(j) - ρ_r^(j)||_1.
#
# A small value means no single-site observable distinguishes the
# different charge-sector ground states well.
# ---------------------------------------------------------------------

function local_distinguishability(model, G)
    (; p, N, Hspace, Hsites) = model

    reduced_states = [
        Vector{Matrix{ComplexF64}}(undef, N) for _ in 1:p
    ]

    for q in 1:p
        ψ = G[:, q]
        for j in 1:N
            reduced_states[q][j] = partial_trace(ψ, Hspace => Hsites[j])

        end
    end

    LD = zeros(Float64, N)
    for j in 1:N, q in 1:p, r in (q+1):p
        LD[j] = max(LD[j],
            trace_norm(reduced_states[q][j] - reduced_states[r][j]) / 2)
    end

    return LD
end

function analyze_boundary_modes(model; h::Real)
    (; Hbond, Hfield, Q, left_seed, right_seed, J) = model

    ham = Hbond + h * Hfield
    scale = max(norm(ham), 1.0)

    @assert norm(ham - ham') < 1e-11 * scale
    @assert norm(ham * Q - Q * ham) < 1e-10 * scale

    ground = charge_ground_states(model, ham)
    (; G, energies, bandwidth, separation) = ground

    if separation <= 0
        @warn "The selected charge-sector states are not an isolated lowest band." h separation
    end

    left = boundary_profile(model, G, energies, left_seed)
    right = boundary_profile(model, G, energies, right_seed)
    LD = local_distinguishability(model, G)

    @printf("\nh/J = %.3f\n", h / J)
    for q in 0:(model.p-1)
        @printf("  E(q=%d) = %.12f\n", q, energies[q+1])
    end
    @printf("  Ground-band splitting:          %.4e\n", bandwidth)
    @printf("  Separation above ground band:   %.4e\n", separation)
    @printf("  Relative left commutator norm:  %.4e\n", left.commutator)
    @printf("  Relative right commutator norm: %.4e\n", right.commutator)
    @printf("  Maximum local distinguishability: %.4e\n", maximum(LD))

    return (;
        h, energies, bandwidth, separation,
        left=left.profile,
        right=right.profile,
        LD,
    )
end

# ---------------------------------------------------------------------
# Run at the exactly localized point and nearby in the same phase.
# ---------------------------------------------------------------------

model = build_parafermion_chain(;
    p=3,
    N=6,
    sigma=1,       # Also works with the conjugate convention sigma=-1.
    J=1.0,
)

sweet = analyze_boundary_modes(model; h=0.0)
topological = analyze_boundary_modes(model; h=0.18 * model.J)

# At the exactly localized point:
# - the left transition reduces only onto the first site;
# - the right transition reduces only onto the last site;
# - different ground states are locally indistinguishable.
#
# These checks are useful end-to-end tests of the complex phase hooks.
tol = 1e-8
@assert abs(sweet.left[1] - 1) < tol
@assert maximum(sweet.left[2:end]) < tol
@assert abs(sweet.right[end] - 1) < tol
@assert maximum(sweet.right[1:(end-1)]) < tol
@assert maximum(sweet.LD) < tol

# ---------------------------------------------------------------------
# Plot, following the style of kitaev_chain.jl.
# ---------------------------------------------------------------------

function locality_panel(model, result; title)
    sites = 1:model.N

    fig = plot(
        sites, result.left;
        label="Left boundary transition",
        xlabel="Site",
        ylabel="Normalized reduction trace norm",
        title=title,
        frame=:box,
        lw=3,
        marker=:circle,
        xticks=sites,
        ylims=(-0.04, 1.08),
        legend=:top,
    )

    plot!(
        fig, sites, result.right;
        label="Right boundary transition",
        lw=3,
        marker=:diamond,
    )

    plot!(
        fig, sites, result.LD;
        label="Local distinguishability",
        lw=2,
        marker=:square,
        linestyle=:dash,
    )

    return fig
end

fig = plot(
    locality_panel(
        model, sweet;
        title="Exactly localized: h/J = 0",
    ),
    locality_panel(
        model, topological;
        title="Topological regime: h/J = 0.18",
    );
    layout=(1, 2),
    size=(1100, 400),
    margin=5Plots.mm,
)
