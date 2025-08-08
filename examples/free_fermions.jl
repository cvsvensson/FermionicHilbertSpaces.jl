# # Free fermions on a 2D grid
# For free fermions, one can work in the single particle picture to get a better scaling with the size of the system. FermionicHilbertSpaces.jl contains some features to help with this.
using FermionicHilbertSpaces, Plots, LinearAlgebra
import FermionicHilbertSpaces: add!                # Import add! for efficient Hamiltonian construction
using Arpack                                        # For sparse eigenvalue decomposition
# ## Define a grid 
# We'll look at a system defined on a disc. Let's define a square grid and then cut out a disc in the middle
N = 40
xs, ys = -N:N, -N:N
indomain(xy) = norm(xy) < N
square_grid = [indomain(xy) ? xy : missing for xy in Iterators.product(xs, ys)]
disc = collect(skipmissing(square_grid))
shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
neighbours(Nx, Ny) = [(Nx + dx, Ny + dy) for (dx, dy) in shifts if (Nx + dx, Ny + dy) in disc]
# Utility: Map a vector of values back to the 2D grid (for plotting)
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
## 
# Let's define a quadratic hamiltonian with a spiral chemical potential and hopping
function potential(xy)
    θ = atan(xy...)
    r = norm(xy) / N
    5 * cos(2 * (θ - 2 * r))^2 * exp(r)
end
hopping(xy1, xy2) = N
@fermions f

# Since we are dealing with many fermions, symbolic sums may take a long time. To get better performance, we will use the function `add!` to update the symbolic hamiltonian in place. For this, it is important to initialize the Hamiltonian with the correct type like
ham = zero(1.0 * f[0, 0] + 1.0)

# Build the Hamiltonian: sum onsite potential and hopping terms
for xy in disc
    add!(ham, potential(xy) * f[xy]' * f[xy])
    for nbr in neighbours(xy...)
        if nbr in disc
            add!(ham, hopping(nbr, xy) * f[nbr]' * f[xy] + hc)
        end
    end
end

# Then we define a single particle hilbert space on the disc
H = single_particle_hilbert_space(disc)
# And get a matrix representation of the hamiltonian
mat = matrix_representation(ham, H)

# Compute a few eigenvalues/eigenvectors (lowest energy states)
vals, vecs = eigs(mat; nev=3^2, which=:SR, v0=map(x -> eltype(mat)(first(x)), disc)[:]);
## Let's define momentum operators
# Initialize momentum operators px, py
px = zero(1im * ham)
py = zero(px)
for xy in disc
    xy .+ (1, 0) in disc && add!(px, 1im * f[xy.+(1, 0)]' * f[xy] + hc)
    xy .+ (0, 1) in disc && add!(py, 1im * f[xy.+(0, 1)]' * f[xy] + hc)
end
# Matrix representations of momentum operators
pxmat = matrix_representation(px, H)
pymat = matrix_representation(py, H)

# Function to compute angular momentum density for a state vector
function angular_momentum(v)
    px = pxmat * v
    py = pymat * v
    vec_to_square_grid(map((px, py, ψ, r) -> imag(conj(ψ) * dot(r, (-py, px))), px, py, v, disc))
end
## Plotting
kwargs = (; ticks=0, cbar=false, aspectratio=1, margins=-2Plots.mm, background_color=:black, size=400 .* (1, 1))
p_density = plot([heatmap(xs, ys, vec_to_square_grid(map(abs2, v))) for v in eachcol(vecs)]...; kwargs...)
p_angular = plot([heatmap(xs, ys, angular_momentum(v); c=:twilight, clims=1.5 * N^(-1.5) .* (-1, 1)) for v in eachcol(vecs)]...; kwargs...)
plot(p_density, p_angular; layout=(1, 2), size=400 .* (2, 1))
