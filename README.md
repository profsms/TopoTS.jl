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
| **Change-point detection** | `changepoint_score`, `detect_changepoints`, `ChangePointEvent` |
| **Sublevel-set PH** | `sublevel_ph`, `windowed_sublevel_ph` |
| **Diagram kernels** | `pss_kernel`, `pwg_kernel`, `sliced_wasserstein_kernel`, `kernel_matrix` |
| **Feature extraction** | `topo_features`, `TopoFeatureSpec`, `feature_names` |

---

## Filtrations

| Symbol | Description | Notes |
|---|---|---|
| `:rips` | Vietoris–Rips | Default; any dimension; fast |
| `:alpha` | Alpha (Delaunay) | O(n log n); exact; ≤3D recommended |
| `:cech` | **Čech (native C++)** | Exact miniball criterion; requires `make` in `csrc/` |
| `:edge_collapsed` | Edge-collapsed Rips | Same diagram as `:rips`; faster for n > 500 |
| `:cubical` | Cubical sublevel-set | 1-D signal; no embedding needed |

The Čech filtration is implemented as a native C++17 shared library
(`csrc/cech_core.cpp`) using Welzl's miniball algorithm, called from Julia
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

### Building libcech (for the Čech filtration)

```julia
using Pkg; Pkg.build("TopoTS")   # auto-detects compiler and builds
```

Or manually:
```bash
cd csrc && make          # Linux / macOS
cd csrc && make windows  # Windows (MinGW)
cd csrc && make test     # run C++ smoke tests first
```

Requires g++ ≥ 9 or clang++ ≥ 10 with C++17.

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

# Change-point detection
wd     = windowed_ph(ts; window=200, step=10, dim=2, lag=τ)
scores = changepoint_score(wd, 1)
events = detect_changepoints(scores.landscape)
println("Change points at t = ", [e.time for e in events])

# CROCKER plot
cp = crocker(wd; dim=1)    # heatmap of β₁(ε, t)
```

---

## Documentation

- `docs/src/vignette.md` — full tutorial (thesis appendix / JSS submission)
- `docs/src/cech_implementation.md` — technical notes on the C++ implementation

---

## Citation

```
Halkiewicz, S. M. S. (2026). Topological Data Analysis for Time Series.
Master's Thesis, AGH University of Cracow.
```

## License

MIT
