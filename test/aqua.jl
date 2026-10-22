@testitem "Aqua" begin
    using Aqua
    # `+(m, ::HC)` is ambiguous with `+(::InitialValue, x)` from InitialValues,
    # which is loaded through NonCommutativeProducts. Resolving it would need a
    # dependency on InitialValues.
    Aqua.test_all(FermionicHilbertSpaces; ambiguities=(exclude=[Base.:+],))
end
