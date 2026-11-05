@testitem "ExplicitImports" begin
    using ExplicitImports
    # Load every weak dependency so that all package extensions are analyzed.
    using Adapt, Combinatorics, GPUArrays, KernelAbstractions, OhMyThreads, PartitionedArrays
    # Package extensions import internals of FermionicHilbertSpaces to add methods to them.
    internal_names = Tuple(filter(n -> !Base.ispublic(FermionicHilbertSpaces, n),
        names(FermionicHilbertSpaces; all=true)))
    nonpublic_imports = (internal_names...,
        # Not public in NonCommutativeProducts.
        :NCterms,
        # OhMyThreads scheduler internals.
        :Consecutive, :chunking_args,
        # GPUArrays does not declare its sparse types `public`.
        :AbstractGPUSparseMatrixCSC, :AbstractGPUSparseMatrixCSR,
    )
    nonpublic_qualified = (
        # NonCommutativeProducts interface functions that this package extends or calls.
        :grade, :grades, :grading, :mul_grade, :adjoint_grade,
        :isscalar, :contraction_key, :mul_terms!!, :termpairs, :split_by_group, :atom_group,
        # Base internals.
        :promote_op, :promote_eltype, :setindex, :product, :isiterable,
        :show_delim_array, :HasEltype, :broadcasted, :instantiate, :inferencebarrier, :mightalias,
        # SparseArrays internals.
        :getcolptr, :nonzeroinds, :sparse!, :AbstractSparseVecOrMat,
        :HigherOrderFns, :is_supported_sparse_broadcast,
        # TupleTools and BitPermutations do not declare their API `public`.
        :flatten, :permute, :sort, :pext, :pdep,
        # Nor do GPUArrays (its sparse interface) and Adapt.
        :dense_array_type, :sparse_array_type, :csc_type, :csr_type, Symbol("@adapt_structure"),
    )
    test_explicit_imports(FermionicHilbertSpaces;
        all_explicit_imports_are_public=(; ignore=nonpublic_imports),
        all_qualified_accesses_are_public=(; ignore=nonpublic_qualified),
        # The @testitem blocks in src/ run in their own modules and must qualify
        # package internals; ExplicitImports reports those as self-qualified.
        no_self_qualified_accesses=false)
end
