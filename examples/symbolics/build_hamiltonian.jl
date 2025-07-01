using FermionicHilbertSpaces

# This tutorial shows how to build a Hamiltonian and speed it up with Symbolics.jl
# Let's start by defining a kitaev chain hamiltonian
N = 8
H = hilbert_space(1:N)
kitaev_chain(f, μ, t, Δ, U) = sum(t * f[i]' * f[i+1] + hc for i in 1:N-1) +
                              sum(Δ * f[i] * f[i+1] + hc for i in 1:N-1) +
                              sum(U * f[i]' * f[i] * f[i+1]' * f[i+1] for i in 1:N-1) +
                              sum(μ * f[i]' * f[i] for i in 1:N)
params = rand(4)

# The most straightforward way is purely numerical:
@fermions f
@time hmat = matrix_representation(kitaev_chain(f, params...), H) #  1.064 ms (12779 allocations: 1.10 MiB)

# We can get a symbolic matrix by using symbolic variables 
using Symbolics
symparams = @variables μ, t, Δ, U
@time hsym = matrix_representation(kitaev_chain(f, symparams...), H) #  6.228 ms (63336 allocations: 2.99 MiB)

# We can compile it, although for large N (around 12), the compilation might take a very long time.
@time fast_ham, fast_ham! = build_function(hsym, symparams; expression=Val{false});
@time s = fast_ham(params)  #   59.000 μs (2062 allocations: 100.56 KiB)
@time fast_ham!(s, params) #  480.513 ns (2 allocations: 96 bytes)


function substitute!(s, params, hsym)
    sub = Dict(zip([μ, t, Δ, U], params))
    foreach((n, v) -> s.nzval[n] = substitute(v, sub).val, eachindex(hsym.nzval), hsym.nzval)
end
@time substitute!(s, params, hsym)
@btime substitute!($s, $params, $hsym) #  5.404 ms (80591 allocations: 2.64 MiB)

# @time hmat = kitaev_chain(fmat, μ, t, Δ, U) #very slow
# @btime hmat_sym = matrix_representation($hsym, $H);
# @time hmat_sym = matrix_representation(hsym, H);
# @profview hmat_sym = matrix_representation(hsym, H)
# @time hnum = kitaev_chain(fmat, 1.0, 1.0, 1.0, 1.0)
# @time hsymnum = kitaev_chain(f, 1.0, 1.0, 1.0, 1.0)
# @btime hmatsymnum = matrix_representation(hsymnum, H);

@btime substitute($hsym, $sub)
@btime matrix_representation(substitute($hsym, $sub), $H)
foo(m, sub) = map(Base.Fix2(substitute, sub), m)
foo2(m, sub) = [substitute(v, sub) for v in m]
@btime foo($hmat_sym, $sub);
@btime foo2($hmat_sym, $sub);
# @time ex = build_function(hmat_sym, [μ, t, Δ, U]; expression=Val{true}, target=Symbolics.CTarget());



function complex_build_function(m, x)
    fr, fr! = build_function(real(m), x; expression=Val{false})
    fi, fi! = build_function(imag(m), x; expression=Val{false})
    f(x) = fr(x) .+ 1im .* fi(x)
    function f!(mout, x, cache=similar(mout))
        fr!(cache, x)
        mout .= cache
        fi!(cache, x)
        mout .+= 1im .* cache
    end
    return f, f!
end

