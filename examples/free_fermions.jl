# # Free fermions on a 2D grid
# For free fermions the single-particle picture scales with the number of sites instead of exponentially: on a disc of thousands of sites, `SingleParticle()` turns a quadratic hamiltonian into a sparse matrix whose lowest eigenstates are found with Arpack. This example builds such a hamiltonian with a spiral potential, and plots the density and the angular momentum density of its lowest states.
using FermionicHilbertSpaces
using FermionicHilbertSpaces.FreeFermions
using Plots, LinearAlgebra
import FermionicHilbertSpaces: add!! # Import add!! for efficient Hamiltonian construction
using Arpack # For sparse eigenvalue decomposition
# ## Define a grid 
# We'll look at a system defined on a disc. Let's define a square grid and then cut out a disc in the middle
@fermions f
N = 40
xs, ys = -N:N, -N:N
indomain(xy) = norm(xy) < N
square_grid = [indomain(xy) ? xy : missing for xy in Iterators.product(xs, ys)]
disc = collect(skipmissing(square_grid))
shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
neighbours(Nx, Ny) = ((Nx + dx, Ny + dy) for (dx, dy) in shifts if indomain((Nx + dx, Ny + dy)))
H = hilbert_space(f, disc, SingleParticle());
# For plotting heatmaps later, we need a function to map a vector over the disc to a matrix on the 2d grid
function vec_to_square_grid(v::AbstractVector{T}) where T
    count = 0
    vals = Vector{Union{T,Missing}}(undef, length(square_grid))
    for (n, xy) in enumerate(square_grid)
        if ismissing(xy)
            vals[n] = missing
        else
            count += 1
            vals[n] = v[count]
        end
    end
    return reshape(vals, size(square_grid))
end
# ## Construct hamiltonian
# Let's define a quadratic hamiltonian with a spiral chemical potential and hopping
function potential(xy)
    θ = atan(xy...)
    r = norm(xy) / N
    5 * cos(2 * (θ - 2 * r))^2 * exp(r)
end
hopping(xy1, xy2) = N
# The hamiltonian has thousands of terms. `sum` would copy the partial sum at every step, so we accumulate the terms in place with `add!!`, starting from `0`.
site_terms(xy) = potential(xy) * f[xy]' * f[xy] + sum(hopping(nbr, xy) * f[nbr]' * f[xy] for nbr in neighbours(xy...))
ham = foldl(add!!, (site_terms(xy) for xy in disc); init = 0)
# And get a matrix representation of it on the single particle hilbert space
mat = representation(ham, H);

# ## Compute eigenstates and momentum operators and plot results
# Compute a few eigenvalues/eigenvectors (lowest energy states)
vals, vecs = eigs(mat; nev=3^2, which=:SR, v0=map(x -> eltype(mat)(first(x)), disc)[:]);
# Calculate momentum operators px, py and define a function to calculate angular momentum density
momentum(shift) = foldl(add!!, (1im * f[xy .+ shift]' * f[xy] + hc for xy in disc if xy .+ shift in disc); init = 0)
px = momentum((1, 0))
py = momentum((0, 1))
pxmat = representation(px, H)
pymat = representation(py, H);
function angular_momentum(v)
    px = pxmat * v
    py = pymat * v
    vec_to_square_grid(map((px, py, ψ, r) -> imag(conj(ψ) * dot(r, (-py, px))), px, py, v, disc))
end
# Plot the number density and angular momentum density of the lowest energy quasiparticles
kwargs = (; ticks=0, cbar=false, aspectratio=1, margins=-2Plots.mm, background_color=:black, size=400 .* (1, 1))
p_density = plot([heatmap(xs, ys, vec_to_square_grid(map(abs2, v))) for v in eachcol(vecs)]...; kwargs...)
p_angular = plot([heatmap(xs, ys, angular_momentum(v); c=:twilight, clims=1.5 * N^(-1.5) .* (-1, 1)) for v in eachcol(vecs)]...; kwargs...)
plot(p_density, p_angular; layout=(1, 2), size=400 .* (2, 1))
