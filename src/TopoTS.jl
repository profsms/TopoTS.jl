"""
    TopoTS

A Julia package for Topological Data Analysis of time series.

Implements the full pipeline:
  1.  Embedding       — Takens delay embedding and parameter selection (AMI, FNN)
  2.  Filtration      — Five filtration types via Ripserer.jl + native Čech (C++)
  3.  Vectorisations  — Betti curves, persistence landscapes, persistence images,
                        topological summary statistics
  4.  Statistics      — Mean landscapes, bootstrap confidence bands,
                        two-sample hypothesis tests
  5.  Windowed PH     — Sliding-window persistent homology for non-stationary series
  6.  Change points   — Bottleneck, Wasserstein, and landscape-based scores
                        with automatic threshold detection
  7.  CROCKER plots   — Betti number surface β_k(ε, t) over scale × window time
  8.  Sublevel-set PH — Direct PH of functions without embedding
  9.  Multivariate    — Joint delay embedding of k-channel time series
  10. Kernels         — PSS, PWG, sliced-Wasserstein diagram kernels
  11. Features        — Flat feature vector extraction for ML pipelines
  12. Visualisations  — Makie-based plots (requires CairoMakie / GLMakie)

# Čech filtration

The Čech filtration (`:cech`) is provided by a native C++17 extension
(`csrc/cech_core.cpp`, Welzl miniball algorithm).  Two delivery modes:

**Mode 1 — JLL (preferred, zero compilation):**
```julia
]add CechCore_jll   # registered in the General registry
using TopoTS        # extension loads automatically
persistent_homology(emb; filtration=:cech)
```

**Mode 2 — environment variable (CI / custom builds):**
```julia
ENV["TOPOTS_LIBCECH"] = "/path/to/libcech.so"
using TopoTS
```
"""
module TopoTS

using LinearAlgebra
using Statistics
using StatsBase
using Libdl

# ── Čech library path registry ────────────────────────────────────────────────
# Populated by (in priority order):
#   1. TopoTSCechExt.__init__()  (called when CechCore_jll is loaded)
#   2. TOPOTS_LIBCECH env var    (developer / CI override)
const _CECH_LIB = Ref{String}("")

"""
    _register_cech_lib!(path::String)

Register the path to `libcech`. Called by `TopoTSCechExt` (the JLL extension)
on startup, or directly when using the env-var override.
Internal — not part of the public API.
"""
function _register_cech_lib!(path::String)
    isfile(path) || error("_register_cech_lib!: not a file: '$path'")
    _CECH_LIB[] = path
end

"""
    cech_lib_path() -> String

Return the path to the `libcech` shared library currently registered,
or an empty string if none is available.
"""
cech_lib_path() = _CECH_LIB[]

"""
    cech_available() -> Bool

Return `true` if a `libcech` binary has been located and passes a
quick runtime check.
"""
function cech_available()
    p = _CECH_LIB[]
    isempty(p) && return false
    isfile(p)  || return false

    local lib
    try
        lib = Libdl.dlopen(p)
        sym = Libdl.dlsym(lib, :cech_version)
        ver = unsafe_string(ccall(sym, Ptr{Cchar}, ()))
        return startswith(ver, "TopoTS-CechCore")
    catch
        return false
    finally
        if @isdefined(lib)
            try
                Libdl.dlclose(lib)
            catch
            end
        end
    end
end

function __init__()
    # Env-var override (developer / CI)
    if haskey(ENV, "TOPOTS_LIBCECH")
        p = ENV["TOPOTS_LIBCECH"]
        isfile(p) && (_CECH_LIB[] = p; return)
        @warn "TOPOTS_LIBCECH='$p' is not a file; ignoring"
    end
    # Otherwise _CECH_LIB[] stays ""; TopoTSCechExt.__init__() sets it
    # after this returns when CechCore_jll is present in the environment.
end

# ── Sub-modules ───────────────────────────────────────────────────────────────
include("embedding/Embedding.jl")
include("filtration/Filtration.jl")
include("vectorisations/BettiCurves.jl")
include("vectorisations/Landscapes.jl")
include("vectorisations/PersistenceImages.jl")
include("vectorisations/TopoStats.jl")
include("statistics/Bootstrap.jl")
include("statistics/HypothesisTests.jl")
include("windowed/Windowed.jl")
include("windowed/CROCKER.jl")
include("changepoint/ChangePoint.jl")
include("sublevel/Sublevel.jl")
include("multivariate/Multivariate.jl")
include("kernels/DiagramKernels.jl")
include("features/Features.jl")
include("visualisations/Visualisations.jl")

using .Embedding
using .Filtration
using .BettiCurves
using .Landscapes
using .PersistenceImages
using .TopoStats
using .Bootstrap
using .HypothesisTests
using .Windowed
using .CROCKER
using .ChangePoint
using .Sublevel
using .Multivariate
using .DiagramKernels
using .Features
using .Visualisations

# ── Public API ────────────────────────────────────────────────────────────────
export
    # Čech availability query
    cech_available, cech_lib_path, _register_cech_lib!,
    # Embedding
    embed, TakensEmbedding, optimal_lag, optimal_dim, ami_lag, fnn_dim,
    embed_multivariate, MultivariateEmbedding,
    # Filtration
    persistent_homology, DiagramCollection,
    # Sublevel-set PH
    sublevel_ph, SublevelDiagram, windowed_sublevel_ph,
    periodogram_ph, windowed_periodogram_ph,
    # Betti curves
    betti_curve, BettiCurve,
    # Landscapes
    landscape, PersistenceLandscape, mean_landscape, landscape_norm,
    # Persistence images
    persistence_image, PersistenceImage,
    # Summary statistics
    total_persistence, persistent_entropy, amplitude,
    # Bootstrap
    bootstrap_landscape, confidence_band,
    # Hypothesis tests
    permutation_test, landscape_ttest,
    # Windowed PH
    windowed_ph, WindowedDiagrams,
    # CROCKER
    crocker, CROCKERPlot,
    # Change-point detection
    changepoint_score, ChangePointResult,
    bottleneck_score, wasserstein_score, landscape_score,
    detect_changepoints, ChangePointEvent,
    # Kernels
    pss_kernel, pwg_kernel, sliced_wasserstein_kernel, kernel_matrix,
    wasserstein_distance,
    # Features
    topo_features, TopoFeatureSpec, feature_names,
    # Visualisations
    plot_diagram, plot_diagram_multi,
    plot_barcode,
    plot_landscape,
    plot_betti_curve,
    plot_crocker, plot_crocker_multi,
    plot_changepoint_score

end # module TopoTS