module Filtration

using Ripserer
using ..TopoTS
using ..Embedding: TakensEmbedding

include(joinpath(@__DIR__, "..", "cech", "CechFiltration.jl"))

export persistent_homology, DiagramCollection

# ─────────────────────────────────────────────────────────────────────────────
# DiagramCollection
# ─────────────────────────────────────────────────────────────────────────────

struct DiagramCollection
    diagrams   :: Vector
    dim_max    :: Int
    filtration :: Symbol
    n_points   :: Int
end

Base.getindex(dc::DiagramCollection, k::Int) = dc.diagrams[k]
Base.length(dc::DiagramCollection)           = length(dc.diagrams)

function Base.show(io::IO, dc::DiagramCollection)
    println(io, "DiagramCollection ($(dc.filtration), $(dc.n_points) pts):")
    for (k, dgm) in enumerate(dc.diagrams)
        println(io, "  H$(k - 1) : $(length(dgm)) points")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

function persistent_homology(emb;
                             dim_max    :: Int    = 2,
                             filtration          = :rips,
                             threshold  :: Real  = Inf,
                             modulus    :: Int   = 2)
    pts = _pts(emb)
    n   = size(pts, 1)

    # Singleton case:
    # all filtrations coincide and Ripserer overflows on nv = 1 in this setup.
    # Return the trivial persistence object directly.
    if n == 1
        return _singleton_diagram_collection(Symbol(filtration))
    end

    # With n points, homology above dimension n-2 is impossible.
    dim_eff = min(dim_max, max(n - 2, 0))

    result = _compute(pts, filtration, dim_eff, threshold, modulus)
    return DiagramCollection(collect(result), dim_eff, Symbol(filtration), n)
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-filtration dispatch
# ─────────────────────────────────────────────────────────────────────────────

function _compute(pts, filtration, dim_max, threshold, modulus)
    if filtration == :rips
        pointcloud = _pointcloud(pts; unique_only = true)
        n_eff = length(pointcloud)
        dim_eff = min(dim_max, max(n_eff - 2, 0))

        rips_kw = (dim_max = dim_eff, modulus = modulus)
        if !isinf(threshold)
            rips_kw = merge(rips_kw, (threshold = Float64(threshold),))
        end

        return ripserer(pointcloud; rips_kw...)

    elseif filtration == :alpha
        size(pts, 2) <= 3 || @warn(
            "Alpha filtration is recommended for ≤3D; got $(size(pts, 2))D")

        pointcloud = _pointcloud(pts; unique_only = true)
        n_eff = length(pointcloud)
        dim_eff = min(dim_max, max(n_eff - 2, 0))

        rips_kw = (dim_max = dim_eff, modulus = modulus)
        if !isinf(threshold)
            rips_kw = merge(rips_kw, (threshold = Float64(threshold),))
        end

        return ripserer(Alpha, pointcloud; rips_kw...)

    elseif filtration == :edge_collapsed
        pointcloud = _pointcloud(pts; unique_only = true)
        n_eff = length(pointcloud)
        dim_eff = min(dim_max, max(n_eff - 2, 0))

        rips_kw = (dim_max = dim_eff, modulus = modulus)
        if !isinf(threshold)
            rips_kw = merge(rips_kw, (threshold = Float64(threshold),))
        end

        return ripserer(EdgeCollapsedRips, pointcloud; rips_kw...)

    elseif filtration == :cubical
        signal = pts[:, 1]
        return ripserer(Cubical(signal);
                        dim_max = min(dim_max, 1),
                        modulus = modulus)

    elseif filtration == :cech
        return _compute_cech(pts, dim_max, threshold, modulus)

    else
        throw(ArgumentError(
            "filtration must be :rips, :alpha, :cech, :edge_collapsed, or " *
            ":cubical; got :$filtration"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Čech via libcech
# ─────────────────────────────────────────────────────────────────────────────

function _compute_cech(pts, dim_max, threshold, modulus)
    libpath = TopoTS._CECH_LIB[]

    if isempty(libpath) || !isfile(libpath)
        error("""
        Čech filtration: libcech not found.

        Install the pre-built binary (recommended):
            ]add CechCore_jll

        Or build locally:
            cd <package_root>/csrc && make
            # (or: using Pkg; Pkg.build("TopoTS"))

        Or point to an existing build:
            ENV["TOPOTS_LIBCECH"] = "/path/to/libcech.so"
        """)
    end

    n = size(pts, 1)

    # With n points, homology above dimension n-2 is impossible.
    dim_eff = min(dim_max, max(n - 2, 0))

    thr = isinf(threshold) ? 1e18 : Float64(threshold)

    filt = CechFiltration.build_cech_filtration(
        pts;
        dim_max   = dim_eff,
        threshold = thr,
        libpath   = libpath,
    )

    return ripserer(filt; dim_max = dim_eff, modulus = modulus)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_pts(emb::TakensEmbedding) = emb.points
_pts(pts::AbstractMatrix)  = Matrix{Float64}(pts)
_pts(emb)                  = Matrix{Float64}(emb.points)

function _pointcloud(pts::AbstractMatrix; unique_only::Bool = false)
    pc = [Tuple(@view pts[i, :]) for i in axes(pts, 1)]
    return unique_only ? unique(pc) : pc
end

function _singleton_diagram_collection(filtration::Symbol)
    # One connected component born at 0, no higher-dimensional features.
    # We represent the single H0 interval minimally; this is sufficient for
    # counting, printing, and basic sanity checks.
    diagrams = Any[
        [(0.0, Inf)]
    ]
    return DiagramCollection(diagrams, 0, filtration, 1)
end

end # module Filtration