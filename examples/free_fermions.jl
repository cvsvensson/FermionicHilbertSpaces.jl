using FermionicHilbertSpaces, Plots
import FermionicHilbertSpaces: add!
using Arpack
using Pkg
##
Pkg.activate("examples\\");
using Arpack;
Pkg.activate(".")
# Let's define a quadratic hamiltonian with a chemical potential and hopping
## Define a grid and 
N = 40
xs, ys = -N:N, -N:N
indomain(xy) = norm(xy) < N #&& !iszero(norm(xy))
square_grid = [indomain(xy) ? xy : missing for xy in Iterators.product(xs, ys)]
disc = filter(!ismissing, square_grid)
shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
neighbours(Nx, Ny) = [(Nx + dx, Ny + dy) for (dx, dy) in shifts if (Nx + dx, Ny + dy) in disc]
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
function potential(xy)
    θ = atan(xy...)
    r = norm(xy) / N
    5 * cos(2 * (θ - 2 * r))^2 * exp(r)
end
hopping(xy1, xy2) = N
##
@fermions f
ham = zero(1.0 * f[disc[1]]' * f[disc[1]] + hopping(disc[1], disc[2]) * f[disc[1]]' * f[disc[2]] + hc) # To get the type of the hamiltonian right
@time for xy in disc
    add!(ham, potential(xy) * f[xy]' * f[xy])
    for nbr in neighbours(xy...)
        if nbr in disc
            add!(ham, hopping(nbr, xy) * f[nbr]' * f[xy] + hc)
        end
    end
end
px = zero(1im * ham)
py = zero(px)
for xy in disc
    xy .+ (1, 0) in disc && add!(px, 1im * f[xy.+(1, 0)]' * f[xy] + hc)
    xy .+ (0, 1) in disc && add!(py, 1im * f[xy.+(0, 1)]' * f[xy] + hc)
end
H = single_particle_hilbert_space(disc)
@time mat = matrix_representation(ham, H)
@time vals, vecs = eigs(mat; nev=3^2, which=:SR, v0=map(x -> eltype(mat)(first(x)), disc)[:]);
pxmat = matrix_representation(px, H)
pymat = matrix_representation(py, H)
function angular_momentum(v)
    px = pxmat * v
    py = pymat * v
    vec_to_square_grid(map((px, py, ψ, r) -> imag(conj(ψ) * dot(r, (-py, px))), px, py, v, disc))
end
function radial_momentum(v)
    px = pxmat * v
    py = pymat * v
    vec_to_square_grid(map((px, py, ψ, r) -> imag(conj(ψ) * (dot(r, (px, py)) + ψ / norm(r))), px, py, v, disc))
end
## Plotting
kwargs = (; ticks=0, cbar=false, aspectratio=1, margins=-1.9Plots.mm, size=600 .* (1, 1), background_color=:black)
prot = plot([heatmap(xs, ys, angular_momentum(v); c=:twilight, clims=1.5 * N^(-1.5) .* (-1, 1), kwargs...) for v in eachcol(vecs)]..., size=600 .* (1, 1), background_color=:black)
prad = plot([heatmap(xs, ys, radial_momentum(v); c=:twilight, clims=2 * N^(-1.5) .* (-1, 1), kwargs...) for v in eachcol(vecs)]..., size=600 .* (1, 1), background_color=:black)
pabs = plot([heatmap(xs, ys, vec_to_square_grid(map(abs2, v)); kwargs...) for v in eachcol(vecs)]..., size=600 .* (1, 1), background_color=:black)
plot(pabs, prot, layout=(1, 2), size=(1200, 600), background_color=:black)
plot(pabs, prot, prad, layout=(1, 3), size=(1800, 600), background_color=:black)
