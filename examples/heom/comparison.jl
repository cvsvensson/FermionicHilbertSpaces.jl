using LinearAlgebra
## Params
omega = 1.1
m_max = 5
gamma = [1, 0.5, 0.2] # decay rates
eta = [0.4 - 0.2im, 0.15 + 0.05im, 0.1] # amplitudes 

## FermionicHilbertSpaces
using FermionicHilbertSpaces
include(joinpath(dirname(Base.active_project()), "heom_bosonic.jl"))
@spin s 1 // 2
ham = omega * s[:z]
V = 2s[:x]
bath_fhs = HEOMBosonicBath(:bath, m_max, gamma, eta)
M_fhs_sym = heom_generator(ham, V, bath_fhs)
Hs, Hleft, Hright, left, right = open_system(s)
Haux = heom_bosonic_aux_space(bath_fhs)
Hfull = tensor_product(Hs, Haux)
M_fhs = representation(M_fhs_sym, Hfull)

## HierarchicalEOM.jl
using HierarchicalEOM
H_q = 0.5 * omega * sigmaz()
V_q = sigmax()
bath_h = BosonBath(V_q, eta, gamma)
M_h_obj = M_Boson(H_q, m_max, bath_h; threshold=0.0, verbose=false)
M_h = HierarchicalEOM.concretize(M_h_obj.data)
## Compare the two representations 
M_h - transpose(M_fhs) |> norm
