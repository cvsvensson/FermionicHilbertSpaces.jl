
_indent(s, p) = join([(i > 1 ? p : "") * l for (i, l) in enumerate(split(s, '\n'))], '\n')
_parity(p) = p == [-1, 1] ? "any" : p == [1] ? "even" : p == [-1] ? "odd" : string(p)

function Base.show(io::IO, nc::NumberConservation{T,H,W}) where {T,H,W}
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(nc.total) ? nothing : sprint(show, nc.total),
            ismissing(nc.subspaces) ? nothing : "subspaces: $(length(nc.subspaces))",
            ismissing(nc.weights) ? nothing : "weighted"])
        print(io, "NumberConservation(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(nc.total) ? nothing : "total: " * (isa(nc.total, AbstractVector) && length(nc.total) == 1 ? string(only(nc.total)) : sprint(show, nc.total)),
            ismissing(nc.subspaces) ? nothing : "subspaces: $(length(nc.subspaces))",
            ismissing(nc.weights) ? nothing : "weights: " * sprint(show, nc.weights)])
        isempty(lines) ? print(io, "NumberConservation()") : print(io, "NumberConservation(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, pc::ParityConservation{H}) where {H}
    compact = get(io, :compact, false)
    if compact
        parts = [_parity(pc.allowed_parities)]
        !ismissing(pc.subspaces) && push!(parts, "subspaces: $(length(pc.subspaces))")
        print(io, "ParityConservation(", join(parts, ", "), ")")
    else
        lines = ["allowed_parities: $(_parity(pc.allowed_parities))"]
        !ismissing(pc.subspaces) && push!(lines, "subspaces: $(length(pc.subspaces))")
        print(io, "ParityConservation(", join(lines, ""), ")")
    end
end

function Base.show(io::IO, ac::AdditiveConstraint{T,H,F}) where {T,H,F}
    nf = isa(ac.functions, Tuple) ? length(ac.functions) : 1
    if get(io, :compact, false)
        pre = ismissing(ac.allowed_values) ? "" : sprint(show, ac.allowed_values) * ", "
        print(io, "AdditiveConstraint(", pre, "$nf function(s))")
    else
        lines = ["  allowed_values: " * (ismissing(ac.allowed_values) ? "missing" : sprint(show, ac.allowed_values)),
            "  functions: $nf function(s)"]
        !ismissing(ac.subspaces) && push!(lines, "  subspaces: $(length(ac.subspaces))")
        print(io, "AdditiveConstraint(\n", join(lines, "\n"), "\n)")
    end
end

function Base.show(io::IO, fc::FilterConstraint)
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(fc.subspaces) ? nothing : "subspaces: $(length(fc.subspaces))",
            ismissing(fc.subspace_functions) ? nothing : "$(isa(fc.subspace_functions, Tuple) ? length(fc.subspace_functions) : 1) function(s)"])
        print(io, "FilterConstraint(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(fc.subspaces) ? nothing : "subspaces: $(length(fc.subspaces))",
            ismissing(fc.subspace_functions) ? nothing : "subspace_functions: $(isa(fc.subspace_functions, Tuple) ? length(fc.subspace_functions) : 1) function(s)",
            "reducer: $(fc.reducer)"])
        print(io, "FilterConstraint(", join(lines, ", "), ")")
    end
end

function Base.show(io::IO, bc::SectorConstraint)
    fc = bc.filter
    if get(io, :compact, false)
        parts = filter(!isnothing, [
            ismissing(fc.subspaces) ? nothing : "subspaces: $(length(fc.subspaces))",
            ismissing(fc.subspace_functions) ? nothing : "$(isa(fc.subspace_functions, Tuple) ? length(fc.subspace_functions) : 1) function(s)"])
        print(io, "SectorConstraint(", join(parts, ", "), ")")
    else
        lines = filter(!isnothing, [
            ismissing(fc.subspaces) ? nothing : "subspaces: $(length(fc.subspaces))",
            ismissing(fc.subspace_functions) ? nothing : "subspace_functions: $(isa(fc.subspace_functions, Tuple) ? length(fc.subspace_functions) : 1) function(s)",
            "reducer: $(fc.reducer)"])
        print(io, "SectorConstraint(", join(lines, ""), ")")
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
    for (ci, (mapper, piece_targets)) in enumerate(zip(sm.factor_mappers, sm.factor_piece_targets))
        println(io)
        if isnothing(mapper)
            print(io, "  Group $ci: (unmapped)")
        else
            destinations = join(["target $(ti)[$(si)]" for (ti, si) in piece_targets], ", ")
            print(io, "  Group $ci → $destinations: ")
            show(IOContext(io, :compact => true), mapper)
        end
    end
end


function Base.show(io::IO, H::SectorHilbertSpace)
    if get(io, :compact, false)
        print(io, "SectorHilbertSpace(")
        show(IOContext(io, :compact => true), H.parent)
        print(io, ", $(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional SectorHilbertSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), H.parent)

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
                print(io, "\nSectors: ")
                for (i, qn) in enumerate(qns)
                    i > 1 && print(io, ", ")
                    print(io, qn, " (", length(H.qn_to_states[qn]), "-dim)")
                end
            end
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

function Base.show(io::IO, H::ConstrainedSpace)
    if get(io, :compact, false)
        print(io, "ConstrainedSpace(")
        show(IOContext(io, :compact => true), parent(H))
        print(io, ", $(dim(H))-dim)")
    else
        print(io, "$(dim(H))-dimensional ConstrainedSpace\n")
        print(io, "Parent: ")
        show(IOContext(io, :compact => true), parent(H))
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

function Base.show(io::IO, state::BosonicState)
    print(io, "|", state.n, "⟩")
end

function Base.show(io::IO, H::SpinSpace{J}) where J
    sym = H.sym
    lbl = sym.field isa Nothing ? sym.label : "$( _symbolic_name_with_tags(sym.field.name, sym) )[$(sym.label)]"
    if get(io, :compact, false)
        print(io, "Spin{", J, "}(", lbl, ")")
    else
        print(io, "$(dim(H))-dimensional Spin{", J, "}")
        print(io, "(", lbl, ")")
    end
end
function Base.show(io::IO, s::SpinState)
    print(io, "|", s.m, "⟩")
end

Base.show(io::IO, m::MajoranaHilbertSpace) = (println(io, "MajoranaHilbertSpace: ", m.sym); show(IOContext(io, :compact => true), m.parent))


function Base.show(io::IO, f::FockNumber)
    print(io, "|",)
    digs = digits(Int, f.f, base=2)
    n = min(5, length(digs))
    join(io, digs[1:n], "")
    if length(digs) > 5
        print(io, "…")
    end
    print(io, "⟩")
end
function Base.show(io::IO, ::MIME"text/plain", f::FockNumber{T}) where T
    get(io, :compact, false) && return show(io, f)
    print(io, "|",)
    join(io, digits(Int, f.f, base=2), "")
    print(io, "⟩")
end



function Base.show(io::IO, H::ProductSpace)
    if get(io, :compact, false)
        print(io, "ProductSpace($(dim(H))-dim, $(length(H.factors)) factors)")
    else
        print(io, "$(dim(H))-dimensional ProductSpace: ")
        dims = map(dim, H.factors)
        println(io, "(", join(dims, "×"), ")")
        for (i, c) in enumerate(H.factors)
            i > 1 && print(io, " ⨯ ")
            show(IOContext(io, :compact => true), c)
        end
    end
end
function Base.show(io::IO, state::ProductState)
    print(io, "")
    for (n, substate) in enumerate(state.states)
        n > 1 && print(io, "⨯")
        show(IOContext(io, :compact => true), substate)
    end
    print(io, "")
end