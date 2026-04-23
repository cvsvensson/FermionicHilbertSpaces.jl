using LinearAlgebra
using FermionicHilbertSpaces
using FermionicHilbertSpaces: HEOMBosonicBath, heom_bosonic_aux_space, heom_generator

using HierarchicalEOM
using SciMLOperators: concretize
##
# -----------------------------------------------------------------------------
# Shared model parameters (minimal TLS + explicit multi-exponential bosonic bath)
# -----------------------------------------------------------------------------

omega = 1.1
n_exp = 5
m_max = 5

gamma = ComplexF64[0.7+0.0im, 1.3+0.0im, 2, 4, 1im][1:n_exp]
eta = ComplexF64[0.40-0.20im, 0.15+0.05im, 2, 4, 1im][1:n_exp]

# -----------------------------------------------------------------------------
# FermionicHilbertSpaces matrix
# -----------------------------------------------------------------------------
using FermionicHilbertSpaces: SectorConstraint, parity

@spin s 1 // 2

ham = omega * s[:z]
V = 2s[:x]

bath_fhs = HEOMBosonicBath(1, n_exp, m_max, gamma, eta)
M_fhs_sym = heom_generator(ham, V, bath_fhs)
Hs, Hleft, Hright, left, right = open_system(s)
Haux = heom_bosonic_aux_space(bath_fhs)
Hfull = tensor_product((Hs, Haux))

M_fhs = matrix_representation(M_fhs_sym, Hfull)
M_fhs_sys = matrix_representation(1im * (left(ham) - right(ham)), Hs)

# Constraint
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

##
# -----------------------------------------------------------------------------
# HierarchicalEOM matrix
# -----------------------------------------------------------------------------

H_q = 0.5 * omega * sigmaz()
V_q = sigmax()

bath_h = BosonBath(V_q, eta, gamma)
M_h_obj = M_Boson(H_q, m_max, bath_h; threshold=0.0, verbose=false)
M_h = concretize(M_h_obj.data)

M_h_sys = concretize(M_S(H_q; verbose=false).data)
##
M_h - transpose(M_fhs) |> norm
