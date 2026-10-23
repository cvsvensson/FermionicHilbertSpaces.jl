using FermionicHilbertSpaces, LinearAlgebra
using FermionicHilbertSpaces.NonCommutativeProducts: terms
const FHS = FermionicHilbertSpaces

t(f) = (f(); s = @timed f(); (round(s.time * 1e3; digits=2), round(s.bytes / 2^20; digits=1)))

# Apply an operator to a ket sum by looping over (operator term, basis state) pairs and accumulating in a Dict
# keyed by the basis state. One precomputation per operator term, no symbolic sorting.
function apply_direct(ham, ψ::Dict{B,T}, space) where {B,T}
    out = Dict{B,T}()
    for opterm in terms(ham)
        precomp = FHS.precomputation_before_operator_application(opterm, space)
        for (s, v) in ψ
            newstate, amp = FHS._apply_local_operators(opterm, s, space, precomp)
            iszero(amp) && continue
            out[newstate] = get(out, newstate, zero(T)) + amp * v
        end
    end
    filter!(kv -> !iszero(last(kv)), out)
end

@fermions c
for L in (16, 32, 100)
    H = hilbert_space(c, [(j, s) for j in 1:L for s in (:↑, :↓)])
    vac = Ket("0"^(2L), H)
    ψ = sum(c[i, :↑]' * c[j, :↓]' * vac for i in 1:min(L, 32) for j in 1:min(L, 32))
    d = Dict(only(k.factors).ket => float(v) for (k, v) in ψ.dict)
    ham = sum(-c[j, s]' * c[mod1(j + 1, L), s] + hc for j in 1:L for s in (:↑, :↓)) +
          sum(c[j, :↑]' * c[j, :↑] * c[j, :↓]' * c[j, :↓] for j in 1:L)
    println("L=$L  terms(ψ)=$(length(d))  statetype=$(eltype(keys(d)))")
    r = apply_direct(ham, d, H)
    println("  direct H*ψ  ms,MiB = ", t(() -> apply_direct(ham, d, H)), "  terms out=", length(r))
    if L <= 32
        Hψ = ham * ψ
        ok = all(only(k.factors).ket in keys(r) && r[only(k.factors).ket] ≈ v for (k, v) in Hψ.dict) && length(r) == length(Hψ.dict)
        println("  agrees with ham*ψ: ", ok)
    end
end
