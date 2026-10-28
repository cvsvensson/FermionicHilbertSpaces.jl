
_indent(s, p) = join([(i > 1 ? p : "") * l for (i, l) in enumerate(split(s, '\n'))], '\n')
_parity(p) = p == [-1, 1] ? "any" : p == [1] ? "even" : p == [-1] ? "odd" : string(p)

function Base.show(io::IO, nc::NumberConservation{T,H,W}) where {T,H,W}
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(nc.total) ? nothing : sprint(show, nc.total),
            ismissing(nc.spaces) ? nothing : "subspaces: $(length(nc.spaces))",
            ismissing(nc.weights) ? nothing : "weighted"])
        print(io, "NumberConservation(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(nc.total) ? nothing : "total: " * (isa(nc.total, AbstractVector) && length(nc.total) == 1 ? string(only(nc.total)) : sprint(show, nc.total)),
            ismissing(nc.spaces) ? nothing : "subspaces: $(length(nc.spaces))",
            ismissing(nc.weights) ? nothing : "weights: " * sprint(show, nc.weights)])
        isempty(lines) ? print(io, "NumberConservation()") : print(io, "NumberConservation(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, pc::ParityConservation{H}) where {H}
    compact = get(io, :compact, false)
    if compact
        parts = [_parity(pc.allowed_parities)]
        !ismissing(pc.spaces) && push!(parts, "subspaces: $(length(pc.spaces))")
        print(io, "ParityConservation(", join(parts, ", "), ")")
    else
        lines = ["allowed_parities: $(_parity(pc.allowed_parities))"]
        !ismissing(pc.spaces) && push!(lines, "subspaces: $(length(pc.spaces))")
        print(io, "ParityConservation(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, ac::AdditiveConstraint{T,H,F}) where {T,H,F}
    nf = isa(ac.maps, Tuple) ? length(ac.maps) : 1
    if get(io, :compact, false)
        pre = ismissing(ac.allowed_values) ? "" : sprint(show, ac.allowed_values) * ", "
        print(io, "AdditiveConstraint(", pre, "$nf function(s))")
    else
        lines = ["  allowed_values: " * (ismissing(ac.allowed_values) ? "missing" : sprint(show, ac.allowed_values)),
            "  maps: $nf function(s)"]
        !ismissing(ac.spaces) && push!(lines, "  subspaces: $(length(ac.spaces))")
        print(io, "AdditiveConstraint(\n", join(lines, "\n"), "\n)")
    end
end

function Base.show(io::IO, fc::FilterConstraint)
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(fc.spaces) ? nothing : "subspaces: $(length(fc.spaces))",
            ismissing(fc.maps) ? nothing : "$(isa(fc.maps, Tuple) ? length(fc.maps) : 1) function(s)"])
        print(io, "FilterConstraint(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(fc.spaces) ? nothing : "subspaces: $(length(fc.spaces))",
            ismissing(fc.maps) ? nothing : "maps: $(isa(fc.maps, Tuple) ? length(fc.maps) : 1) function(s)",
            "reducer: $(fc.reducer)"])
        print(io, "FilterConstraint(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, bc::SectorConstraint)
    fc = bc.filter
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(fc.spaces) ? nothing : "subspaces: $(length(fc.spaces))",
            ismissing(fc.maps) ? nothing : "$(isa(fc.maps, Tuple) ? length(fc.maps) : 1) function(s)"])
        print(io, "SectorConstraint(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(fc.spaces) ? nothing : "subspaces: $(length(fc.spaces))",
            ismissing(fc.maps) ? nothing : "maps: $(isa(fc.maps, Tuple) ? length(fc.maps) : 1) function(s)",
            "reducer: $(fc.reducer)"])
        print(io, "SectorConstraint(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, pc::ProductConstraint{C}) where {C}
    nconstraints = length(pc.constraints)
    nshow = min(nconstraints, 6)
    if get(io, :compact, false)
        print(io, "ProductConstraint(")
        for (i, c) in enumerate(pc.constraints[1:nshow])
            i > 1 && print(io, " * ")
            show(IOContext(io, :compact => true), c)
        end
        nconstraints > nshow && print(io, " * ... ($(nconstraints - nshow) more)")
        print(io, ")")
    else
        print(io, "ProductConstraint:\n")
        for (i, c) in enumerate(pc.constraints[1:nshow])
            is_last = (i == nshow) && (nconstraints == nshow)
            prefix = is_last ? "  └─ " : "  ├─ "
            if isa(c, ProductConstraint)
                nested = join(split(sprint(show, c), '\n')[2:end], '\n')
                print(io, prefix, "ProductConstraint:\n", _indent(nested, is_last ? "     " : "  │  "))
            else
                print(io, prefix, sprint(show, c; context=:compact => true))
            end
            is_last || print(io, "\n")
        end
        if nconstraints > nshow
            nshow > 0 && print(io, "\n")
            print(io, "  └─ ... (", nconstraints - nshow, " more)")
        end
    end
end

block_idx(::PieceIndex{b,s}) where {b,s} = b
slot_idx(::PieceIndex{b,s}) where {b,s} = s
function Base.show(io::IO, sm::ProductSpaceMapper)
    if get(io, :compact, false)
        print(io, "ProductSpaceMapper(")
        for (i, ts) in enumerate(sm.target_spaces)
            i > 1 && print(io, " ⊗ ")
            show(IOContext(io, :compact => true), ts)
        end
        print(io, ")")
        return
    end
    println(io, "ProductSpaceMapper:")
    print(io, "  Targets: ")
    for (i, ts) in enumerate(sm.target_spaces)
        i > 1 && print(io, " ⊗ ")
        show(IOContext(io, :compact => true), ts)
    end
    for (fi, (leaf, route)) in enumerate(zip(sm.leaves, sm.factor_routes))
        println(io)
        if leaf isa Uncovered
            print(io, "  Factor $fi (uncovered)")
        else
            dests = join(["target $(block_idx(ref))[$(slot_idx(ref))]" for ref in route], ", ")
            if leaf isa Passthrough
                print(io, "  Factor $fi → $dests  (passthrough)")
            else
                n = length(route)
                print(io, "  Factor $fi → $dests  ($n piece$(n > 1 ? "s" : ""))")
            end
        end
    end
end



function Base.show(io::IO, H::AbstractSubspace)
    name = nameof(typeof(H))
    if get(io, :compact, false)
        print(io, name, "(")
        show(IOContext(io, :compact => true), parent(H))
        print(io, ", $(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional $name\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
        _show_details(io, H)
    end
end
_show_details(io, H) = nothing

function _show_details(io, H::SectorHilbertSpace)
    qns = collect(keys(H.qn_to_states))
    nsectors = length(qns)

    if !isempty(qns)
        max_printed_sectors = 5
        edge_sectors = 1

        if nsectors > max_printed_sectors
            print(io, "\nSectors: $nsectors total [")

            for (i, qn) in enumerate(qns[1:edge_sectors])
                i > 1 && print(io, ", ")
                print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
            end

            print(io, ", ..., ")

            for (i, qn) in enumerate(qns[end-edge_sectors+1:end])
                i > 1 && print(io, ", ")
                print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
            end

            print(io, "]")
        else
            print(io, "\nSectors: [")
            for (i, qn) in enumerate(qns)
                i > 1 && print(io, ", ")
                print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
            end
            print(io, "]")

        end
    end
end


function Base.show(io::IO, H::BdGHilbertSpace)
    if get(io, :compact, false)
        print(io, "BdGHilbertSpace($(dim(H))-dim, $(div(dim(H), 2)) modes)")
    else
        print(io, "$(dim(H))-dimensional BdGHilbertSpace\n")
        print(io, "Physical modes: ", div(dim(H), 2), "\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
    end
end

function Base.show(io::IO, H::SingleParticleHilbertSpace)
    if get(io, :compact, false)
        print(io, "SingleParticleHilbertSpace($(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional SingleParticleHilbertSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
    end
end

function Base.show(io::IO, c::FermionicSpace)
    max_modes = 10
    if get(io, :compact, false)
        _compact_fermionic_modes(io, c; max_groups=4, edge_groups=2, max_labels_per_group=6, edge_labels=2)
    else
        n = nbr_of_modes(c)
        print(io, "$(dim(c))-dimensional FermionicSpace\nModes: ")
        n > max_modes && print(io, "$n total ")
        _compact_fermionic_modes(io, c; (n > max_modes ? (max_groups=6, edge_groups=3, max_labels_per_group=8, edge_labels=3) : (;))...)
    end
end

function Base.show(io::IO, H::TransposedSpace)
    if get(io, :compact, false)
        show(io, parent(H))
        print(io, "^T")
    else
        lines = split(sprint(show, parent(H); context=io), '\n')
        lines[1] *= " (transposed)"
        print(io, join(lines, '\n'))
    end
end

function Base.show(io::IO, H::GenericHilbertSpace)
    if get(io, :compact, false)
        print(io, "GenericHilbertSpace(", H.label, ")")
    else
        print(io, "$(dim(H))-dimensional GenericHilbertSpace\n")
        print(io, "Label: ", H.label)
    end
end

function Base.show(io::IO, H::TruncatedBosonicHilbertSpace)
    sym = H.sym
    lbl = sym.basis isa Nothing ? string(sym.label) : "$( _symbolic_name_with_tags(sym.basis.name, sym.basis) )[$(sym.label)]"
    if get(io, :compact, false)
        print(io, "Bosons(", lbl, ", dim=", dim(H), ")")
    else
        print(io, "$(dim(H))-dimensional TruncatedBosonicHilbertSpace\n")
        print(io, "Label: ", lbl, ", dimension: ", dim(H))
    end
end

Base.show(io::IO, state::BosonicState) = print(io, "BosonicState(", state.n, ")")

function Base.show(io::IO, H::SpinSpace)
    J = H.spin
    sym = H.sym
    lbl = sym.field isa Nothing ? sym.label : "$( _symbolic_name_with_tags(sym.field.name, sym) )[$(sym.label)]"
    if get(io, :compact, false)
        print(io, "Spin{", J, "}(", lbl, ")")
    else
        print(io, "$(dim(H))-dimensional Spin{", J, "}")
        print(io, "(", lbl, ")")
    end
end
Base.show(io::IO, s::SpinState) = print(io, "SpinState(", s.m, ")")

Base.show(io::IO, m::MajoranaHilbertSpace) = (println(io, "MajoranaHilbertSpace: ", m.sym); show(IOContext(io, :compact => true), m.parent))


Base.show(io::IO, f::FockNumber) = print(io, "FockNumber(", f.f, ")")



function Base.show(io::IO, H::ProductSpace)
    if get(io, :compact, false)
        print(io, "ProductSpace($(dim(H))-dim, $(length(H.factors)) factors)")
    else
        print(io, "$(dim(H))-dimensional ProductSpace: ")
        dims = map(dim, H.factors)
        println(io, "", join(dims, "×"), ")")
        for (i, c) in enumerate(H.factors)
            i > 1 && print(io, " ⨯ ")
            show(IOContext(io, :compact => true), c)
        end
    end
end
function Base.show(io::IO, state::ProductState)
    print(io, "ProductState((")
    join(io, (sprint(show, s; context=io) for s in state.states), ", ")
    length(state.states) == 1 && print(io, ",")
    print(io, "))")
end

# A short label for a basis state of `space`. Fock states are labeled by one occupation digit per mode.
_state_label(state, space) = _state_label_atomic(state, _unwrap_space(space))
_unwrap_space(H::Union{TransposedSpace,AbstractSubspace}) = _unwrap_space(parent(H))
_unwrap_space(H) = H
const _FermionModeSpace = Union{FermionicSpace,MajoranaHilbertSpace}
_state_label_atomic(f::FockNumber, H::_FermionModeSpace) = join(digits(Int, f.f; base=2, pad=nbr_of_modes(H)))
_state_label_atomic(f::FixedNumberFockState, H::_FermionModeSpace) = _state_label_atomic(FockNumber(f), H)
_state_label_atomic(s::NambuState, H::_FermionModeSpace) = (s.hole ? "h" : "p") * _state_label_atomic(s.state, H)
_state_label_atomic(s::BosonicState, H) = string(s.n)
_state_label_atomic(s::SpinState, H::SpinSpace) = H.spin == 1 // 2 ? (s.m > 0 ? "↑" : "↓") : string(s.m)
_state_label_atomic(s::ProductState, H::ProductSpace) = "(" * join(map(_state_label, s.states, H.factors), ", ") * ")"
_state_label_atomic(s, H) = sprint(print, s)

function Base.show(io::IO, bs::BasisStates)
    max_items = 20
    n = length(bs)
    print(io, "[")
    if get(io, :limit, false) && n > max_items
        join(io, (_state_label(bs[i], bs.space) for i in 1:max_items÷2), ", ")
        print(io, "  …  ")
        join(io, (_state_label(bs[i], bs.space) for i in n-max_items÷2+1:n), ", ")
    else
        join(io, (_state_label(s, bs.space) for s in bs), ", ")
    end
    print(io, "]")
end
function Base.show(io::IO, ::MIME"text/plain", bs::BasisStates)
    max_rows = 8
    n = length(bs)
    print(io, n, "-element basis of ")
    show(IOContext(io, :compact => true), bs.space)
    print(io, ":")
    rows = get(io, :limit, false) && n > max_rows ? vcat(1:max_rows÷2, 0, n-max_rows÷2+1:n) : 1:n
    for i in rows
        print(io, "\n ", iszero(i) ? "⋮" : _state_label(bs[i], bs.space))
    end
end


function _show_mapping_summary(io::IO, mappings; max_mappings=3)
    nm = length(mappings)
    nshow = min(nm, max_mappings)
    for i in 1:nshow
        i > 1 && print(io, ", ")
        m = mappings[i]
        show(IOContext(io, :compact => true), first(m))
        print(io, " => ")
        show(IOContext(io, :compact => true), last(m))
    end
    nm > nshow && print(io, ", ... (", nm - nshow, " more)")
end

function Base.show(io::IO, op::PartialTransposeMap)
    in_dim = dim(op.H)
    out_dim = dim(op.Hout)
    if get(io, :compact, false)
        print(io, "PartialTransposeMap(", in_dim, "^2 → ", out_dim, "^2, transpose=")
        show(IOContext(io, :compact => true), op.Hsub)
        length(op.kwargs) > 0 && print(io, ", ", length(op.kwargs), " kw")
        print(io, ")")
    else
        print(io, "PartialTransposeMap\n")
        print(io, "  Input space: ")
        show(IOContext(io, :compact => true), op.H)
        print(io, "\n  Transposed subsystem: ")
        show(IOContext(io, :compact => true), op.Hsub)
        print(io, "\n  Output space: ")
        show(IOContext(io, :compact => true), op.Hout)
        print(io, "\n  Superoperator size: ", out_dim^2, " x ", in_dim^2)
        print(io, "\n  Stored map nnz: ", nnz(op.map))
        if length(op.kwargs) > 0
            print(io, "\n  kwargs: ")
            show(io, op.kwargs)
        end
    end
end

function Base.show(io::IO, op::PartialTraceMap)
    in_dim = dim(op.H)
    out_dim = dim(op.Hsub)
    if get(io, :compact, false)
        print(io, "PartialTraceMap(", in_dim, "^2 → ", out_dim, "^2")
        length(op.kwargs) > 0 && print(io, ", ", length(op.kwargs), " kw")
        print(io, ")")
    else
        print(io, "PartialTraceMap\n")
        print(io, "  Input space: ")
        show(IOContext(io, :compact => true), op.H)
        print(io, "\n  Output space: ")
        show(IOContext(io, :compact => true), op.Hsub)
        print(io, "\n  Superoperator size: ", out_dim^2, " x ", in_dim^2)
        print(io, "\n  Stored map nnz: ", nnz(op.map))
        if length(op.kwargs) > 0
            print(io, "\n  kwargs: ")
            show(io, op.kwargs)
        end
    end
end

function Base.show(io::IO, op::EmbedMap)
    in_dim = dim(op.Hsub)
    out_dim = dim(op.H)
    if get(io, :compact, false)
        print(io, "EmbedMap(", in_dim, "^2 → ", out_dim, "^2")
        length(op.kwargs) > 0 && print(io, ", ", length(op.kwargs), " kw")
        print(io, ")")
    else
        print(io, "EmbedMap\n")
        print(io, "  Input space: ")
        show(IOContext(io, :compact => true), op.Hsub)
        print(io, "\n  Output space: ")
        show(IOContext(io, :compact => true), op.H)
        print(io, "\n  Superoperator size: ", out_dim^2, " x ", in_dim^2)
        print(io, "\n  Stored map nnz: ", nnz(op.map))
        if length(op.kwargs) > 0
            print(io, "\n  kwargs: ")
            show(io, op.kwargs)
        end
    end
end

function Base.show(io::IO, op::ReshapeMap)
    dims_in = map(p -> Tuple(map(dim, p.Hsin)), op.precomputed)
    dims_out = map(p -> Tuple(map(dim, p.Hsout)), op.precomputed)

    if get(io, :compact, false)
        print(io, "ReshapeMap(")
        if length(dims_in) == 1
            print(io, dims_in[1], " -> ", dims_out[1])
            op.repeat && print(io, ", repeat=true")
        else
            pairs = map(((din, dout),) -> string(din, " -> ", dout), zip(dims_in, dims_out))
            print(io, "[", join(pairs, ", "), "]")
        end
        print(io, ")")
    else
        print(io, "ReshapeMap(")
        if length(dims_in) == 1
            print(io, "consumes ", dims_in[1], ", outputs ", dims_out[1])
            op.repeat && print(io, ", repeat=true")
        else
            pairs = map(((din, dout),) -> string(din, " -> ", dout), zip(dims_in, dims_out))
            print(io, "mappings=", "[", join(pairs, ", "), "]")
        end
        print(io, ")")
    end
end

@testitem "Basis state show" begin
    @fermions f
    @boson b
    @spin s 1 // 2
    H = hilbert_space(f, 1:4, NumberConservation(1))
    @test repr(basisstates(H)) == "[1000, 0100, 0010, 0001]"
    @test endswith(repr(MIME"text/plain"(), basisstates(H)), ":\n 1000\n 0100\n 0010\n 0001")
    Hp = tensor_product(hilbert_space(f, 1:2), hilbert_space(b, 3))
    @test occursin("(10, 2)", repr(basisstates(Hp)))
    @test repr(basisstates(hilbert_space(s))) == "[↓, ↑]"
    long = sprint(show, MIME"text/plain"(), basisstates(hilbert_space(f, 1:6)); context=:limit => true)
    @test occursin("⋮", long) && countlines(IOBuffer(long)) == 10
    @test repr(basisstate(2, H)) == "FockNumber(2)"
    @test repr(basisstate(6, Hp)) == "ProductState((FockNumber(1), BosonicState(1)))"
    @test repr(basisstate(2, hilbert_space(s))) == "SpinState(1/2)"
    Hn = hilbert_space(f, 1:3, NumberConservation())
    for space in (hilbert_space(f, 1:3), hilbert_space(b, 3), hilbert_space(s), Hp, H, sector(1, Hn))
        bs = basisstates(space)
        @test bs == collect(bs) == [basisstate(i, space) for i in 1:dim(space)]
        @test length(bs) == dim(space)
    end

    Hg = FermionicHilbertSpaces.GenericHilbertSpace(:x, [FockNumber(0), FockNumber(3)])
    @test repr(basisstates(Hg)) == "[FockNumber(0), FockNumber(3)]"
    @test occursin("FockNumber(3)", repr(MIME"text/plain"(), basisstates(Hg)))
    @majoranas γ
    @test repr(basisstates(hilbert_space(γ, 1:4, ParityConservation(1)))) == "[00, 11]"

    bs6 = basisstates(hilbert_space(f, 1:6))
    @test repr(bs6) == "[" * join(map(n -> join(digits(n; base=2, pad=6)), 0:63), ", ") * "]"
    limited = sprint(show, bs6; context=:limit => true)
    @test startswith(limited, "[000000, 100000, ") && endswith(limited, ", 011111, 111111]")
    @test count(", ", limited) == 18 && occursin("  …  ", limited)
    @test sprint(show, basisstates(H); context=:limit => true) == "[1000, 0100, 0010, 0001]"

    @test repr(basisstates(hilbert_space(f, 1:3, FermionicHilbertSpaces.SingleParticle()))) == "[100, 010, 001]"
    @test repr(basisstates(hilbert_space(f, 1:2, FermionicHilbertSpaces.BdG()))) == "[h10, h01, p10, p01]"
end
