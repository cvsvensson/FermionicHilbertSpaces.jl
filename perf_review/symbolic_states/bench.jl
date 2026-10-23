using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.NonCommutativeProducts: terms
import KrylovKit.VectorInterface: inner

t(f) = (f(); s = @timed f(); (round(s.time * 1e3; digits=2), round(s.bytes / 2^20; digits=1)))

@fermions c
for L in (16, 32)
    H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
    vac = Ket("0"^(2L), H)
    ψ = sum(c[i, :↑]' * c[j, :↓]' * vac for i in 1:L for j in 1:L)   # L² terms
    n = length(ψ.dict)
    ham = sum(-c[j, s]' * c[mod1(j + 1, L), s] + hc for j in 1:L for s in (:↑, :↓)) +
          sum(c[j, :↑]' * c[j, :↑] * c[j, :↓]' * c[j, :↓] for j in 1:L)
    hterms = collect(terms(ham))
    println("L=$L  terms(ψ)=$n  terms(H)=$(length(hterms))")
    println("  keytype: ", keytype(ψ.dict))
    println("  inner(ψ,ψ)           ms,MiB = ", t(() -> inner(ψ, ψ)))
    # dictionary baseline for the same inner product
    d = Dict(only(k.factors) => v for (k, v) in ψ.dict)
    println("  dict inner baseline  ms,MiB = ", t(() -> sum(conj(v) * get(d, k, 0) for (k, v) in d)))
    println("  ham * ψ              ms,MiB = ", t(() -> ham * ψ))
    println("  Σ h*ψ term by term   ms,MiB = ", t(() -> sum(h * ψ for h in hterms)))
    Hψ = ham * ψ
    println("  keytype(Hψ): ", keytype(Hψ.dict), "  terms=", length(Hψ.dict))
    one_term = first(hterms)
    k1 = first(keys(ψ.dict))
    println("  one op term * one ket µs = ", round(1e3 * first(t(() -> one_term * k1)); digits=1))
end
