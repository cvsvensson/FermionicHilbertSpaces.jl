
function print_state(v::AbstractVector, H::SymmetricFockHilbertSpace; digits=3)
    N = length(keys(H))
    printstyled("labels = |", bold=true)
    for (n, k) in enumerate(keys(H))
        printstyled(k, bold=true)
        n < N && print(",")
    end
    printstyled(">", bold=true)
    println()
    for (n, qn) in enumerate(keys(H.symmetry.qntofockstates))
        print("QN = ", qn)
        println()
        states = H.symmetry.qntofockstates[qn]
        for f in states
            ind = H.symmetry.state_indexdict[f]
            print(" |", Int.(bits(f, N))..., ">")
            x = v[ind] isa LinearAlgebra.BlasFloat ? round(v[ind]; digits) : v[ind]
            println(" : ", x)
        end
    end
end
