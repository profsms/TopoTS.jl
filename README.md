# TopoTS.jl

**Topological Data Analysis for Time Series in Julia**

---

## Overview

TopoTS.jl provides a complete pipeline for applying topological data analysis (TDA) to time series:

**time series → embedding → persistent homology → vectorisation → statistical inference**

The package is designed for research applications in nonlinear dynamics, econometrics, and machine learning.

---

## Key Features

### Core Pipeline

* Takens embedding (AMI, FNN)
* Persistent homology (multiple filtrations)
* Vectorisation (landscapes, Betti curves, persistence images)

### Statistical Tools

* Topological summary statistics
* Bootstrap and confidence bands
* Hypothesis testing

### Time-dependent Analysis

* Sliding-window persistent homology
* CROCKER plots
* Change-point detection

### Machine Learning

* Topological feature extraction
* Kernel methods for persistence diagrams

---

## Supported Filtrations

| Symbol | Description | Notes |
|---|---|---|
| `:rips` | Vietoris–Rips | Default; any dimension; fast |
| `:alpha` | Alpha (Delaunay) | O(n log n); exact; ≤3D recommended |
| `:cech` | **Čech (native C++)** | Exact miniball criterion |
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

Requires g++ ≥ 12 or clang++ ≥ 10 with C++17.

---

## Example Workflow

```julia
using TopoTS

τ   = optimal_lag(ts)
emb = embed(ts; dim=3, lag=τ)
dgms = persistent_homology(emb)
λ = landscape(dgms, 1)
```

---

## Visualisation

Requires Makie backend:

```julia
using CairoMakie
fig = plot_crocker(cp)
```

---

## License

GNU General Public License (GPL-3 or later)

---

## Author

Stanisław M. S. Halkiewicz
AGH University of Cracow
