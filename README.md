# TopoTS.jl

**Topological Data Analysis for Time Series in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Julia 1.9+](https://img.shields.io/badge/Julia-1.9%2B-blueviolet)](https://julialang.org)

TopoTS.jl implements the complete pipeline from raw time series to topological inference:

```
time series  →  Takens embedding  →  persistent homology  →  vectorisation  →  statistics
```

---

## Features

| Module | Functions |
|---|---|
| **Embedding** | `embed`, `optimal_lag` (AMI), `optimal_dim` (FNN), `embed_multivariate` |
| **Filtration** | `persistent_homology` — five filtration types (see below) |
| **Betti curves** | `betti_curve` |
| **Landscapes** | `landscape`, `mean_landscape`, `landscape_norm` |
| **Persistence images** | `persistence_image` |
| **Summary stats** | `total_persistence`, `persistent_entropy`, `amplitude` |
| **Bootstrap** | `bootstrap_landscape`, `confidence_band` |
| **Hypothesis tests** | `permutation_test`, `landscape_ttest` |
| **Windowed PH** | `windowed_ph`, `WindowedDiagrams` |
| **CROCKER plots** | `crocker`, `CROCKERPlot` |
| **Change-point detection** | `changepoint_score`, `detect_changepoints`, `detect_changepoints_windowed`, `andrews_supF` |
| **Sublevel-set PH** | `sublevel_ph`, `windowed_sublevel_ph`, `periodogram_ph`, `windowed_periodogram_ph` |
| **Diagram kernels** | `pss_kernel`, `pwg_kernel`, `sliced_wasserstein_kernel`, `kernel_matrix`, `wasserstein_distance` |
| **Feature extraction** | `topo_features`, `TopoFeatureSpec`, `feature_names` |

---

## Filtrations

| Symbol | Description | Notes |
|---|---|---|
| `:rips` | Vietoris–Rips | Default; any dimension; fast |
| `:alpha` | Alpha (Delaunay) | O(n log n); exact; ≤3D recommended |
| `:cech` | **Čech (native C++)** | Exact miniball criterion; requires `CechCore_jll` |
| `:edge_collapsed` | Edge-collapsed Rips | Same diagram as `:rips`; faster for n > 500 |
| `:cubical` | Cubical sublevel-set | 1-D signal; no embedding needed |

The Čech filtration is implemented as a native C++17 shared library
(Welzl's miniball algorithm) shipped via
[CechCore_jll](https://github.com/profsms/CechCore_jll), called from Julia
via `ccall`. It implements the `AbstractFiltration` interface from Ripserer.jl,
so **all downstream functions work identically regardless of filtration type**.

### Rips/Čech interleaving

In Euclidean space:  `Čech_ε ⊆ Rips_ε ⊆ Čech_{√2·ε}`

The diagrams agree up to a factor of √2. Use `:cech` for geometric exactness;
`:rips` or `:edge_collapsed` for speed.

---

## Installation

```julia
using Pkg; Pkg.add(url="https://github.com/profsms/TopoTS.jl")
```

### Čech filtration

`CechCore_jll` is a weak dependency — install it separately to enable `:cech`:

```julia
using Pkg; Pkg.add("CechCore_jll")
```

Pre-built binaries are provided for all major platforms (Linux, macOS, Windows,
FreeBSD — x86\_64, aarch64, riscv64). No compiler required.

---

## Quick Start

```julia
using TopoTS

ts = sin.(range(0, 20π, length=2000)) .+ 0.05 .* randn(2000)

# Standard pipeline (Rips)
τ   = optimal_lag(ts)
emb = embed(ts; dim=3, lag=τ)
dgms = persistent_homology(emb; dim_max=1)
λ    = landscape(dgms, 1)

# Čech filtration (exact)
dgms_cech = persistent_homology(emb; filtration=:cech, threshold=2.0)

# Change-point detection — one-call convenience
wd = windowed_ph(ts; window=200, step=10, dim=2, lag=τ)
events, sc = detect_changepoints_windowed(wd; dim=1, method=:andrews)
println("Change points at t = ", [e.time for e in events])

# Spectral change-point detection (frequency-domain)
wd_spec = windowed_periodogram_ph(ts; window=200, step=10)
events_spec, _ = detect_changepoints_windowed(wd_spec; dim=0, method=:cusum_mad)

# CROCKER plot
cp = crocker(wd; dim=1)    # heatmap of β₁(ε, t)
```

---

## Change-point detection methods

`detect_changepoints` accepts a `method` keyword covering six approaches:

| `method` | Type | Description |
|---|---|---|
| `:cusum_mad` | online | Threshold = median + `n_mad`×MAD (default) |
| `:cusum_3sigma` | online | Threshold = μ + `n_sigma`×σ over burn-in |
| `:cusum_sustained` | online | `k` consecutive exceedances required |
| `:percentile` | online | 99th percentile of burn-in period |
| `:cusum_adaptive` | online | Sliding-window baseline of width `win` |
| `:rss` | offline | RSS minimisation — single best breakpoint |
| `:andrews` | offline | Andrews (1993) sup-F test |

```julia
# Raw score vector → events
score = changepoint_score(wd, 1).landscape.scores
events = detect_changepoints(score; method=:andrews, alpha=0.05)

# Or skip the intermediate step entirely
events, score_vec = detect_changepoints_windowed(wd; dim=1, method=:rss)
```

---

## Documentation

- `docs/src/vignette.md` — full tutorial
- `docs/src/cech_implementation.md` — technical notes on the C++ implementation

---

## Citation

```
Halkiewicz, S. M. S. (2026). Topological Data Analysis for Time Series.
Master's Thesis, AGH University of Cracow.
```

## License

MIT
