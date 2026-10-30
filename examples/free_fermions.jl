# # Free fermions on a 2D grid
# For free fermions the single-particle picture scales with the number of sites instead of exponentially: on a disc of thousands of sites, `SingleParticle()` turns a quadratic hamiltonian into a sparse matrix whose lowest eigenstates are found with Arpack. This example builds such a hamiltonian with a spiral potential, and plots the density and the angular momentum density of its lowest states.
using FermionicHilbertSpaces
using FermionicHilbertSpaces.FreeFermions
using Plots, LinearAlgebra
import FermionicHilbertSpaces: add!! # for efficient in-place construction of symbolic operators
using Arpack # for sparse eigenvalue decomposition

# We cut a disc out of a square grid. Since symbolic sums over many terms are slow, `add!!` updates the hamiltonian in place, which requires that it has the right type: `zero` of a simple hamiltonian gives an empty one of a matching type.
@fermions f
N = 40
xs, ys = -N:N, -N:N
indomain(xy) = norm(xy) < N
square_grid = [indomain(xy) ? xy : missing for xy in Iterators.product(xs, ys)]
disc = collect(skipmissing(square_grid))
shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
neighbours(Nx, Ny) = ((Nx + dx, Ny + dy) for (dx, dy) in shifts if indomain((Nx + dx, Ny + dy)))
H = hilbert_space(f, disc, SingleParticle());

function potential(xy)
    θ = atan(xy...)
    r = norm(xy) / N
    5 * cos(2 * (θ - 2 * r))^2 * exp(r)
end
hopping(xy1, xy2) = N

ham = zero(1.0 * f[0, 0] * f[1, 1] + 1.0)
for xy in disc
    add!!(ham, potential(xy) * f[xy]' * f[xy])
    for nbr in neighbours(xy...)
        add!!(ham, hopping(nbr, xy) * f[nbr]' * f[xy])
    end
end
mat = representation(ham, H);

# The lowest eigenstates are the lowest energy quasiparticles. The momentum operators give an angular momentum density, ``\mathrm{Im}(\psi^* \, \mathbf{r} \times \mathbf{p}\psi)``, and `vec_to_square_grid` maps a vector over the disc to a matrix on the grid for plotting heatmaps.
vals, vecs = eigs(mat; nev=3^2, which=:SR, v0=map(x -> eltype(mat)(first(x)), disc)[:]);

px = zero(1im * ham)
py = zero(px)
for xy in disc
    xy .+ (1, 0) in disc && add!!(px, 1im * f[xy.+(1, 0)]' * f[xy] + hc)
    xy .+ (0, 1) in disc && add!!(py, 1im * f[xy.+(0, 1)]' * f[xy] + hc)
end
pxmat = representation(px, H)
pymat = representation(py, H);

function vec_to_square_grid(v::AbstractVector{T}) where {T}
    count = 0
    vals = Vector{Union{T, Missing}}(undef, length(square_grid))
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
function angular_momentum(v)
    px = pxmat * v
    py = pymat * v
    vec_to_square_grid(map((px, py, ψ, r) -> imag(conj(ψ) * dot(r, (-py, px))), px, py, v, disc))
end

kwargs = (; ticks=0, cbar=false, aspectratio=1, margins=-2Plots.mm, background_color=:black, size=400 .* (1, 1))
p_density = plot([heatmap(xs, ys, vec_to_square_grid(map(abs2, v))) for v in eachcol(vecs)]...; kwargs...)
p_angular = plot([heatmap(xs, ys, angular_momentum(v); c=:twilight, clims=1.5 * N^(-1.5) .* (-1, 1)) for v in eachcol(vecs)]...; kwargs...)
plot(p_density, p_angular; layout=(1, 2), size=400 .* (2, 1))
