using LinearAlgebra
using FermionicHilbertSpaces
include(string(@__DIR__, "/heom_bosonic.jl"))
## Params
omega = 1.1
m_max = 2
gamma = [1 + 0.5im, 0.5 + 0.1im]
eta = [0.4 - 0.2im, 0.15 + 0.05im]

## FermionicHilbertSpaces
@spin s 1 // 2
ham = omega * s[:z]
V = 2s[:x]
bath_fhs = HEOMBosonicBath(1, m_max, gamma, eta)
M_fhs_sym = heom_generator(ham, V, bath_fhs)
Hs, Hleft, Hright, left, right = open_system(s)
Haux = heom_bosonic_aux_space(bath_fhs)
Hfull = tensor_product((Hs, Haux))
M_fhs = matrix_representation(M_fhs_sym, Hfull);
## HierarchicalEOM.jl
using HierarchicalEOM
H_q = 0.5 * omega * sigmaz()
V_q = sigmax()
bath_h = BosonBath(V_q, eta, gamma)
M_h_obj = M_Boson(H_q, m_max, bath_h; threshold=0.0, verbose=false)
M_h = HierarchicalEOM.concretize(M_h_obj.data)
##
M_h - transpose(M_fhs) |> norm

## Constraint
using FermionicHilbertSpaces: SectorConstraint, parity
spin_parity(s) = (-1)^(iseven(Int(s.m + 1 // 2)))
ADOparity(s) = (-1)^(iseven(sum(s.n)))
even_parity(ps) = begin
    p = prod(ps)
    p == 1 ? p : missing
end
constraint = SectorConstraint(
    [Hleft, Hright, Haux],
    [spin_parity, spin_parity, ADOparity], even_parity
)
Hcons = constrain_space(Hfull, constraint)
matrix_representation(M_fhs_sym, Hcons)

