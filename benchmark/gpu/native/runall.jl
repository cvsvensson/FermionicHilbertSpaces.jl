for stage in ("stage1_fermions", "stage2_number_conservation", "stage3_constraints", "stage4_bosons", "stage5_spins", "stage6_mixed")
    include(joinpath(@__DIR__, stage * ".jl"))
end
