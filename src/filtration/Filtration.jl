"""
    Filtration

Persistent homology from embedded point clouds.

Supported filtrations:
  :rips           — Vietoris–Rips (default)
  :alpha          — Alpha / Delaunay (≤3D recommended)
  :cech           — Čech (exact miniball; requires libcech)
  :edge_collapsed — Edge-collapsed Rips (same diagram, faster)
  :cubical        — Cubical sublevel-set (1-D signal, no embedding needed)

The Čech filtration is provided by a C++ shared library loaded via ccall.
Library resolution order:
  1. CechCore_jll  (installed via ]add CechCore_jll — preferred)
  2. TOPOTS_LIBCECH environment variable
  3. deps/lib/libcech.*  (local build via csrc/Makefile)

Rips / Čech interleaving guarantee (Euclidean space):
  Čech_ε  ⊆  Rips_ε  ⊆  Čech_{√2·ε}
Diagrams agree up to a √2 rescaling of birth/death coordinates.
"""
module Filtration

using Ripserer
using ..Embedding: TakensEmbedding

export persistent_homology, DiagramCollection

# ─────────────────────────────────────────────────────────────────────────────
# DiagramCollection
# ─────────────────────────────────────────────────────────────────────────────

"""
    DiagramCollection

Container for persistence diagrams, one per homological dimension.

`dc[1]` = H₀,  `dc[2]` = H₁,  etc.
"""
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
        println(io, "  H$(k-1) : $(length(dgm)) points")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    persistent_homology(emb;
                        dim_max    :: Int    = 2,
                        filtration          = :rips,
                        threshold  :: Real  = Inf,
                        modulus    :: Int   = 2) -> DiagramCollection

Compute persistent homology. The `filtration` keyword selects the complex:

- `:rips`           — Vietoris–Rips (default; fast; any dimension)
- `:alpha`          — Alpha/Delaunay (O(n log n); exact; ≤3D)
- `:cech`           — Čech (exact miniball; install `CechCore_jll` or build locally)
- `:edge_collapsed` — Edge-collapsed Rips (same diagram as `:rips`, faster for n>500)
- `:cubical`        — Sublevel-set on the first coordinate (1-D signal, H₀ only)

All five return a `DiagramCollection`; every downstream function
(`landscape`, `changepoint_score`, `crocker`, `topo_features`, …) works
identically regardless of filtration type.
"""
function persistent_homology(emb;
                              dim_max    :: Int    = 2,
                              filtration          = :rips,
                              threshold  :: Real  = Inf,
                              modulus    :: Int   = 2)
    pts = _pts(emb)
    n   = size(pts, 1)
    result = _compute(pts, filtration, dim_max, threshold, modulus)
    return DiagramCollection(collect(result), dim_max, Symbol(filtration), n)
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-filtration dispatch
# ─────────────────────────────────────────────────────────────────────────────

function _compute(pts, filtration, dim_max, threshold, modulus)
    rips_kw = (dim_max = dim_max, modulus = modulus)
    if !isinf(threshold)
        rips_kw = merge(rips_kw, (threshold = Float64(threshold),))
    end

    if filtration == :rips
        return ripserer(pts; rips_kw...)

    elseif filtration == :alpha
        size(pts, 2) <= 3 || @warn(
            "Alpha filtration is recommended for ≤3D; got $(size(pts,2))D")
        return ripserer(Alpha(pts); rips_kw...)

    elseif filtration == :edge_collapsed
        ecr = EdgeCollapsedRips(pts;
                  threshold = isinf(threshold) ? nothing : Float64(threshold))
        return ripserer(ecr; dim_max = dim_max, modulus = modulus)

    elseif filtration == :cubical
        signal = pts[:, 1]
        return ripserer(Cubical(signal);
                        dim_max = min(dim_max, 1), modulus = modulus)

    elseif filtration == :cech
        return _compute_cech(pts, dim_max, threshold, modulus)

    else
        throw(ArgumentError(
            "filtration must be :rips, :alpha, :cech, :edge_collapsed, or " *
            ":cubical; got :$filtration"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Čech via libcech (ccall)
# ─────────────────────────────────────────────────────────────────────────────

# Cached flag: CechFiltration.jl has been included into this module
const _CECH_INCLUDED = Ref(false)

function _compute_cech(pts, dim_max, threshold, modulus)
    # Resolve the library path from the package-level registry
    # (set by __init__, the JLL extension, or TOPOTS_LIBCECH)
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

    # Include CechFiltration.jl once (defines the CechFiltration module
    # and the CechFilt type that wraps the ccall)
    if !_CECH_INCLUDED[]
        cech_jl = joinpath(@__DIR__, "..", "cech", "CechFiltration.jl")
        isfile(cech_jl) || error("CechFiltration.jl not found at $cech_jl")
        include(cech_jl)
        _CECH_INCLUDED[] = true
    end

    # Pass the resolved library path to the filtration builder
    thr  = isinf(threshold) ? 1e18 : Float64(threshold)
    filt = CechFiltration.build_cech_filtration(
               pts;
               dim_max   = dim_max,
               threshold = thr,
               libpath   = libpath)

    return ripserer(filt; dim_max = dim_max, modulus = modulus)
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_pts(emb::TakensEmbedding) = emb.points
_pts(pts::AbstractMatrix)  = Matrix{Float64}(pts)
_pts(emb)                  = Matrix{Float64}(emb.points)  # MultivariateEmbedding etc.

end # module Filtration
