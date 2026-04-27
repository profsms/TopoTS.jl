# TopoTS.jl — Complete User Guide

**TopoTS** is a Julia package for Topological Data Analysis (TDA) of time series. It implements the full pipeline from raw signal to machine-learning-ready feature vectors, including change-point detection, statistical tests, and publication-quality visualisations.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Conceptual Overview](#2-conceptual-overview)
3. [Embedding — Takens Delay Reconstruction](#3-embedding--takens-delay-reconstruction)
4. [Filtration — Persistent Homology](#4-filtration--persistent-homology)
5. [Vectorisations](#5-vectorisations)
   - 5.1 [Betti Curves](#51-betti-curves)
   - 5.2 [Persistence Landscapes](#52-persistence-landscapes)
   - 5.3 [Persistence Images](#53-persistence-images)
   - 5.4 [Summary Statistics](#54-summary-statistics)
6. [Statistics — Bootstrap and Hypothesis Tests](#6-statistics--bootstrap-and-hypothesis-tests)
7. [Windowed Persistent Homology](#7-windowed-persistent-homology)
8. [CROCKER Plots](#8-crocker-plots)
9. [Change-Point Detection](#9-change-point-detection)
10. [Sublevel-Set Persistent Homology](#10-sublevel-set-persistent-homology)
11. [Multivariate Embedding](#11-multivariate-embedding)
12. [Diagram Kernels](#12-diagram-kernels)
13. [Feature Extraction for ML Pipelines](#13-feature-extraction-for-ml-pipelines)
14. [Visualisations](#14-visualisations)
15. [End-to-End Examples](#15-end-to-end-examples)
16. [Performance Notes and Caveats](#16-performance-notes-and-caveats)

---

## 1. Installation

```julia
using Pkg
Pkg.add("TopoTS")
```

TopoTS depends on [Ripserer.jl](https://github.com/mtsch/Ripserer.jl) for persistent homology and [PersistenceDiagrams.jl](https://github.com/mtsch/PersistenceDiagrams.jl) for diagram operations. These are pulled in automatically.

**Optional — Čech filtration (`:cech`):**

The Čech filtration uses a native C++ library. Install the pre-built JLL:

```julia
Pkg.add("CechCore_jll")
using TopoTS   # the extension loads automatically
```

Or point to a custom build at runtime:

```julia
ENV["TOPOTS_LIBCECH"] = "/path/to/libcech.so"
using TopoTS
```

Check availability before using:

```julia
cech_available()    # → true / false
cech_lib_path()     # → "/path/to/libcech.so" or ""
```

**Optional — Visualisations:**

All `plot_*` functions require a Makie backend loaded first. No hard dependency is imposed:

```julia
using CairoMakie    # for file output (PDF, PNG, SVG)
# or
using GLMakie       # for interactive windows
# or
using WGLMakie      # for Pluto / Jupyter notebooks
```

---

## 2. Conceptual Overview

The standard TDA pipeline for time series is:

```
raw signal x[1..N]
    │
    ▼  Step 1: Embedding
embed(x; dim=d, lag=τ)
    │  → point cloud in ℝᵈ
    ▼  Step 2: Filtration
persistent_homology(pts)
    │  → DiagramCollection (birth-death pairs per dimension)
    ▼  Step 3: Vectorisation
landscape / betti_curve / topo_features / ...
    │  → Vector{Float64}
    ▼  Step 4: Statistics / ML
permutation_test / kernel SVM / random forest / ...
```

**Why topological features?**

Persistent homology captures shape — loops, voids, connected components — in data that persists across a range of scales. Features like "the time series traces a torus in phase space" or "there are two dominant frequency bands" correspond to H₁ and H₀ bars in the persistence diagram and are provably stable under small perturbations of the signal.

---

## 3. Embedding — Takens Delay Reconstruction

> Module: `Embedding` | Key exports: `embed`, `ami_lag`, `fnn_dim`, `optimal_lag`, `optimal_dim`

### Background

Takens' theorem (1981) guarantees that, for a generic smooth dynamical system on a compact manifold of dimension $m$, a delay embedding with dimension $d \ge 2m+1$ recovers the topology of the attractor from a single observable $x(t)$. In practice, $d$ is selected by the False Nearest Neighbours criterion and $\tau$ by Average Mutual Information.

### `ami_lag` — lag selection

```julia
ami_lag(x; max_lag=50, nbins=32) -> Int
```

Estimates $\text{AMI}(\tau) = \sum_{i,j} p_{ij}(\tau) \log \frac{p_{ij}(\tau)}{p_i p_j}$ via a 2-D histogram and returns the first local minimum. At the first local minimum, successive observations are maximally independent, which is the recommended criterion of Fraser & Swinney (1986).

```julia
using TopoTS

ts = sin.(range(0, 20π, length=2000)) .+ 0.05 .* randn(2000)

lag = ami_lag(ts)
# e.g. 16  (depends on the signal; the first zero-crossing of AMI)
```

**Caveats:**
- `max_lag` must be ≥ the true optimal lag. For slowly-varying signals, increase it.
- `nbins` controls AMI resolution; 16–64 is typical. Too few bins → coarse estimate; too many → sparse histogram.
- If no local minimum exists within `max_lag`, the global minimum is returned as a fallback.
- For periodic signals, AMI has a minimum at approximately a quarter-period.

### `fnn_dim` — dimension selection

```julia
fnn_dim(x; lag, max_dim=10, rtol=10.0, atol=2.0) -> Int
```

Counts False Nearest Neighbours (FNN) at each candidate dimension. A neighbour is *false* if adding a coordinate causes the distance to that neighbour to jump by a factor `rtol`, or by more than `atol × σ(x)`. Returns the first dimension where FNN fraction drops below 1%.

```julia
dim = fnn_dim(ts; lag=lag)
# e.g. 3
```

**Caveats:**
- Uses brute-force O(n²) nearest-neighbour search. For `length(ts) > 5000`, subsample first.
- `rtol=10.0` and `atol=2.0` are standard defaults (Kennel et al., 1992). For noisy data, increase both slightly to avoid overestimating dimension.
- FNN can underestimate dimension for heavily noisy signals (noise adds false dimensions).

### `embed` — point cloud construction

```julia
embed(x; dim, lag) -> TakensEmbedding
```

Produces the point cloud $X_i = (x_i, x_{i+\tau}, \ldots, x_{i+(d-1)\tau})$ as an $(n \times d)$ matrix, where $n = N - (d-1)\tau$.

```julia
emb = embed(ts; dim=dim, lag=lag)
# TakensEmbedding{Float64}: 1970 points in ℝ3 (lag=16)

emb.points    # (n × d) Matrix{Float64}
emb.dim       # 3
emb.lag       # 16
emb.n_orig    # 2000
size(emb)     # (1970, 3)
length(emb)   # 1970  (number of embedded points)
```

**Convenience wrappers:**

```julia
lag = optimal_lag(ts)          # identical to ami_lag
dim = optimal_dim(ts; lag=lag) # identical to fnn_dim
```

**Caveats:**
- Both `dim ≥ 1` and `lag ≥ 1` are required; otherwise an `ArgumentError` is thrown.
- The embedded series is shorter than the original: $n = N - (d-1)\tau$. For $d=3, \tau=20, N=200$: $n=160$ — a significant loss. Keep `lag` small when data is short.
- `TakensEmbedding` plugs directly into `persistent_homology` and `topo_features`.

---

## 4. Filtration — Persistent Homology

> Module: `Filtration` | Key exports: `persistent_homology`, `DiagramCollection`

### `persistent_homology`

```julia
persistent_homology(pts;
    dim_max    = 2,
    filtration = :rips,
    threshold  = Inf,
    modulus    = 2) -> DiagramCollection
```

**Inputs accepted:**
- `TakensEmbedding` or `MultivariateEmbedding` (uses `.points` field automatically)
- Any `AbstractMatrix` of size $(n \times d)$ (rows = points, columns = coordinates)

**Filtration types:**

| Symbol             | Description | Notes |
|--------------------|-------------|-------|
| `:rips`            | Vietoris–Rips complex (default) | General purpose; works in any dimension |
| `:alpha`           | Alpha complex | Exact for ≤ 3D; faster than Rips in low dims |
| `:edge_collapsed`  | Edge-collapsed Rips | Faster for large point clouds; same output as `:rips` |
| `:cubical`         | Cubical complex on signal values | Uses only first column; for 1-D functions |
| `:cech`            | Čech complex (requires `CechCore_jll`) | Exact by definition; slower; for ≤ 3D |

```julia
# Rips filtration (most common)
dgms = persistent_homology(emb; dim_max=1, filtration=:rips)
# DiagramCollection (rips, 1970 pts):
#   H0 : 1969 points
#   H1 : 47 points

# Alpha filtration (faster for 3D embeddings)
dgms_alpha = persistent_homology(emb; dim_max=1, filtration=:alpha)

# With a threshold to speed up computation
dgms = persistent_homology(pts; dim_max=2, filtration=:rips, threshold=2.0)
```

### `DiagramCollection` — the result type

```julia
dgms[1]   # H₀ diagram (connected components), 1-indexed
dgms[2]   # H₁ diagram (loops)
dgms[3]   # H₂ diagram (voids), if dim_max ≥ 2

length(dgms)       # number of homological dimensions computed
dgms.dim_max       # maximum dimension
dgms.filtration    # :rips / :alpha / ...
dgms.n_points      # number of input points
```

Each `dgms[k+1]` is a Ripserer diagram object. Individual points support `birth(p)` and `death(p)`. To get raw pairs:

```julia
using PersistenceDiagrams: birth, death

h1 = dgms[2]
pairs = [(birth(p), death(p)) for p in h1]
lifetimes = [death(p) - birth(p) for p in h1 if isfinite(death(p))]
```

**Caveats:**
- Rips filtration on $n$ points has worst-case complexity $O(n^{d_{\max}+1})$ in both time and memory. **Subsample point clouds** before computing PH on large embeddings. Typically 200–1000 points is safe.
- `:alpha` is only reliable in 2D and 3D (Delaunay triangulation is exact there).
- `:cubical` treats the input as a 1-D function and uses only the first column of the matrix. It computes H₀ and H₁ only (max_dim clamped to 1).
- Infinite bars: every connected point cloud has exactly one H₀ bar `(0, Inf)` (the global component). H₀ bars correspond to connected components; H₁ bars to loops; H₂ bars to enclosed voids.
- `modulus=2` is the default field for homology coefficients. Change to a prime for homology over $\mathbb{F}_p$.

---

## 5. Vectorisations

Persistence diagrams are point clouds in $\mathbb{R}^2$ and cannot be directly used as fixed-length feature vectors. The vectorisation functions solve this.

### 5.1 Betti Curves

> Module: `BettiCurves` | Exports: `betti_curve`, `BettiCurve`

The $k$-th Betti curve counts the number of $k$-dimensional topological features alive at each filtration scale $\varepsilon$:

$$\beta_k(\varepsilon) = \#\{(b,d) \in PH_k : b \le \varepsilon \le d\}$$

```julia
betti_curve(dgms, dim; tgrid=nothing, n_grid=500) -> BettiCurve
```

```julia
bc0 = betti_curve(dgms, 0; n_grid=200)   # connected components
bc1 = betti_curve(dgms, 1; n_grid=200)   # loops

bc1.values    # Vector{Int} of length 200
bc1.tgrid     # scale grid (AbstractRange)
bc1.dim       # 1

# Plot manually
using CairoMakie
lines(bc1.tgrid, bc1.values; axis=(xlabel="ε", ylabel="β₁(ε)"))
```

**Shared grid for comparison across series:**

```julia
shared_grid = range(0.0, 3.0; length=300)
bc1_a = betti_curve(dgms_a, 1; tgrid=shared_grid)
bc1_b = betti_curve(dgms_b, 1; tgrid=shared_grid)

# Now bc1_a.values and bc1_b.values are directly comparable
diff = bc1_a.values .- bc1_b.values
```

**Caveats:**
- Points with infinite death time are excluded (they would make the curve nonzero forever). For H₀ this means the single global component is not counted.
- Auto-grid extends 5% beyond the maximum finite death time. If you compare series with different diagram extents, use an explicit `tgrid`.
- Betti curves are integer-valued and not differentiable. For statistical tests, use landscapes instead.

---

### 5.2 Persistence Landscapes

> Module: `Landscapes` | Exports: `landscape`, `PersistenceLandscape`, `mean_landscape`, `landscape_norm`

Persistence landscapes (Bubenik, 2015) map each persistence diagram to a sequence of piecewise-linear functions in $L^2(\mathbb{R})$. This makes averaging, variance, and hypothesis testing mathematically rigorous.

For a diagram $D = \{(b_i, d_i)\}$, the tent function is:
$$f_{(b,d)}(t) = \max(0, \min(t-b, d-t))$$

The $k$-th landscape layer $\lambda_k(t)$ is the $k$-th largest tent value at each $t$.

```julia
landscape(dgms, dim; tgrid=nothing, n_grid=500, n_layers=5) -> PersistenceLandscape
```

```julia
λ = landscape(dgms, 1; n_layers=3, n_grid=200)
# PersistenceLandscape(H1, 3 layers × 200 grid points)

λ.layers       # (3 × 200) Matrix{Float64}
λ.layers[1, :] # first (dominant) layer — the largest tent at each t
λ.tgrid        # scale grid
λ.dim          # 1

# The first layer captures the most persistent feature;
# the k-th layer captures the k-th most persistent.
```

**Mean landscape over an ensemble:**

```julia
# Compute landscapes for each series in an ensemble
λs = [landscape(persistent_homology(embed(ts_i; dim=3, lag=lag), dim_max=1), 1)
      for ts_i in ensemble]

# All landscapes must share the same grid and n_layers
λ_mean = mean_landscape(λs)   # PersistenceLandscape
```

**Landscape norms:**

```julia
landscape_norm(λ, 2)    # L² norm (default), sum over layers
landscape_norm(λ, 1)    # L¹ norm
landscape_norm(λ, Inf)  # sup-norm (max absolute value)
```

**Arithmetic:**

```julia
diff = λ1 - λ2             # pointwise difference
combined = 0.5 * λ1 + 0.5 * λ2   # weighted average
```

**Caveats:**
- `mean_landscape` requires all landscapes to have the same grid and the same number of layers. Always set an explicit `tgrid` and `n_layers` when aggregating.
- The first layer captures the single most persistent feature. If the diagram has fewer features than `n_layers`, the higher layers are zero.
- The landscape norm is computed via the trapezoidal rule on the stored grid. Finer grids give more accurate norms.

---

### 5.3 Persistence Images

> Module: `PersistenceImages` | Exports: `persistence_image`, `PersistenceImage`

A persistence image (Adams et al., 2017) maps a diagram to a fixed-size matrix by Gaussian-smoothing the diagram in birth–persistence coordinates.

```julia
persistence_image(dgms, dim;
    n_pixels = 20,
    sigma    = nothing,    # default: 10% of persistence range
    weight   = :linear,   # or :constant
    b_range  = nothing,   # (b_min, b_max); auto-detected
    p_range  = nothing    # (p_min, p_max); auto-detected
) -> PersistenceImage
```

```julia
img = persistence_image(dgms, 1; n_pixels=20)
# PersistenceImage(H1, 20×20 pixels, σ=0.12)

img.pixels    # (20 × 20) Matrix{Float64} (normalised to sum to 1)
img.sigma     # bandwidth used
size(img)     # (20, 20)

# Flatten to a feature vector for ML:
feat = vec(img)    # 400-dimensional
```

**Consistent range across series (required for comparison):**

```julia
# Compute a global birth and persistence range from training data
imgs_train = [persistence_image(dgm, 1) for dgm in train_dgms]
b_min = minimum(i.b_range[1] for i in imgs_train)
b_max = maximum(i.b_range[2] for i in imgs_train)
p_max = maximum(i.p_range[2] for i in imgs_train)

# Apply fixed range to all (train + test)
imgs_all = [persistence_image(dgm, 1;
                n_pixels=20, sigma=0.1,
                b_range=(b_min, b_max), p_range=(0.0, p_max))
            for dgm in all_dgms]
```

**Caveats:**
- Auto-detected ranges differ across diagrams. **Always fix `b_range` and `p_range`** when comparing images across time series or across train/test splits.
- Auto-sigma is 10% of the persistence range. Use a consistent `sigma` across series.
- `:linear` weight (default) down-weights low-persistence noise; `:constant` treats all points equally.
- The output is normalised (sum to 1). Set `n_pixels` consistently across datasets.

---

### 5.4 Summary Statistics

> Module: `TopoStats` | Exports: `total_persistence`, `persistent_entropy`, `amplitude`

These are scalar summaries of a persistence diagram, useful as features or diagnostic values.

#### `total_persistence`

$$\text{TotalPers}_p(D) = \sum_{(b,d) \in D} (d - b)^p$$

```julia
total_persistence(dgms, 1; p=1)   # sum of lifetimes (p=1)
total_persistence(dgms, 1; p=2)   # sum of squared lifetimes (p=2)
```

For $p=1$: total "mass" of all features. For $p=2$: emphasises long-lived features quadratically.

#### `persistent_entropy`

$$E(D) = -\sum_i \frac{\ell_i}{L} \log\frac{\ell_i}{L}, \quad \ell_i = d_i - b_i, \quad L = \sum_i \ell_i$$

```julia
persistent_entropy(dgms, 1)
```

Low entropy: one dominant long-lived feature. High entropy: many equally persistent features. Returns 0 for empty diagrams.

#### `amplitude`

$$A_p(D) = \left(\sum_{(b,d)} \left(\frac{d-b}{2}\right)^p\right)^{1/p}$$

For $p = \infty$: half-persistence of the most prominent feature (equals the bottleneck distance to the empty diagram).

```julia
amplitude(dgms, 1; p=Inf)   # max half-persistence
amplitude(dgms, 1; p=2)     # L² amplitude
```

**Caveats:**
- All three statistics ignore infinite-death bars (unpaired H₀ component).
- They are stable with respect to diagram perturbations but are scalar — they lose information. Use as additional features, not replacements for landscapes or images.

---

## 6. Statistics — Bootstrap and Hypothesis Tests

> Module: `Bootstrap`, `HypothesisTests`
> Exports: `bootstrap_landscape`, `confidence_band`, `permutation_test`, `landscape_ttest`

### Bootstrap confidence bands

```julia
bootstrap_landscape(λs;
    n_boot = 1000,
    alpha  = 0.05,
    stat   = :mean) -> NamedTuple
```

Draws `n_boot` bootstrap samples from `λs` (with replacement), computes the mean landscape for each, and returns pointwise quantile confidence bands.

```julia
λs = [landscape(dgms_i, 1; n_grid=200, n_layers=3) for dgms_i in all_dgms]

result = bootstrap_landscape(λs; n_boot=1000, alpha=0.05)
result.mean        # observed mean landscape
result.lower       # 2.5th percentile landscape
result.upper       # 97.5th percentile landscape
result.boot_means  # all 1000 bootstrap means

# Convenience: just mean + band
band = confidence_band(λs; n_boot=500, alpha=0.05)
band.mean; band.lower; band.upper
```

**Caveats:**
- All `λs` must have the same grid and layer count. See §5.2.
- `n_boot=1000` is standard; for quick exploration, 200–300 suffices.
- The band is pointwise, not uniform. For simultaneous coverage, multiply by a Bonferroni-type correction or use the permutation test.

### Permutation test for two groups

```julia
permutation_test(λs1, λs2;
    n_perm = 999,
    stat   = :l2_mean_diff) -> NamedTuple
```

Tests $H_0: E[\lambda_1] = E[\lambda_2]$ using the $L^2$ distance between sample mean landscapes as the test statistic. Labels are permuted `n_perm` times under $H_0$.

```julia
# Do periodic and chaotic series have different H₁ topology?
result = permutation_test(λs_periodic, λs_chaotic; n_perm=999)
result.pvalue       # permutation p-value (e.g. 0.002)
result.statistic    # observed L² distance between means
result.null_dist    # 999 permuted values
```

**Caveats:**
- Both groups must share the same landscape grid.
- `:l2_mean_diff` is the default; use `:l1_mean_diff` for robustness to outliers.
- The p-value formula `(count(null ≥ observed) + 1) / (n_perm + 1)` is the standard unbiased estimator; the smallest achievable p-value is $1/(n\_perm + 1)$.
- Group sizes need not be equal.

### Pointwise t-test

```julia
landscape_ttest(λs1, λs2; layer=1) -> NamedTuple
```

Runs Welch's two-sample t-test at each grid point of landscape layer `layer`. Useful for identifying *where* on the scale axis the two groups differ.

```julia
result = landscape_ttest(λs1, λs2; layer=1)
result.pvalues   # vector of raw pointwise p-values
result.tstats    # vector of t-statistics
result.tgrid     # scale grid

# Find the scale range where the groups differ
sig_region = result.tgrid[result.pvalues .< 0.05]
```

**Caveats:**
- Raw p-values are not corrected for multiple testing. Apply Bonferroni or BH correction before drawing conclusions.
- For a single global test, prefer `permutation_test`. Use `landscape_ttest` only for localisation.
- The normal approximation to the t-distribution is used; it is accurate for large sample sizes but can be anti-conservative for tiny groups.

---

## 7. Windowed Persistent Homology

> Module: `Windowed` | Exports: `windowed_ph`, `WindowedDiagrams`

Sliding-window PH applies the full embedding + PH pipeline to each overlapping window of a time series, producing a time series of persistence diagrams. This is the foundation for CROCKER plots and change-point detection.

```julia
windowed_ph(x;
    window,                    # window length W (samples)
    step      = window ÷ 4,   # stride between windows
    dim       = 2,             # Takens embedding dimension
    lag       = 1,             # Takens lag
    dim_max   = 1,             # max homological dimension
    filtration = :rips,
    threshold  = Inf,
    verbose    = false
) -> WindowedDiagrams
```

```julia
ts = vcat(sin.(range(0, 20π, 500)), randn(500))  # periodic → noise

wd = windowed_ph(ts;
    window = 150,
    step   = 10,
    dim    = 2,
    lag    = 8,
    dim_max = 1,
    verbose = true)

length(wd)       # number of windows computed
wd[1]            # DiagramCollection for the first window
wd.times         # centre sample of each window (Float64 vector)
wd.positions     # start index of each window

# Iterate
for (i, dgms) in enumerate(wd)
    # dgms :: DiagramCollection
end
```

**Choosing window and step:**
- `window` must satisfy $W > (d-1)\tau + 1$ (enough samples for the embedding).
- Smaller `step` → smoother score curves but longer computation. Rule of thumb: `step = window ÷ 4` to `window ÷ 10`.
- `wd.times[i]` is the centre of window $i$, useful for plotting scores against the original time axis.

**Caveats:**
- Each window is embedded and PH is run independently; this is $O(n\_windows \times n^{d_{max}+1})$ in the worst case. Subsampling within each window (via `threshold`) can dramatically reduce cost.
- For long series, use `:edge_collapsed` or a `threshold` to speed up PH.
- The embedding dimension and lag are the same for all windows. For very non-stationary signals, consider adaptive embedding parameters (not yet automated).

---

## 8. CROCKER Plots

> Module: `CROCKER` | Exports: `crocker`, `CROCKERPlot`

A **CROCKER plot** (Topaz et al., 2015) is a heatmap of the Betti number $\beta_k(\varepsilon, t)$ as a function of both filtration scale $\varepsilon$ (vertical axis) and window time $t$ (horizontal axis):

$$\text{CROCKER}[i,j] = \beta_k(\varepsilon_i, t_j) = \#\{(b,d) \in PH_k(W_j) : b \le \varepsilon_i \le d\}$$

```julia
crocker(wd;
    dim     = 1,     # homological dimension
    n_scale = 100,   # number of scale grid points
    scales  = nothing  # explicit scale grid (overrides n_scale)
) -> CROCKERPlot
```

```julia
wd = windowed_ph(ts; window=150, step=10, dim=2, lag=8, dim_max=1)

cp0 = crocker(wd; dim=0, n_scale=50)   # connected components
cp1 = crocker(wd; dim=1, n_scale=50)   # loops

cp1.surface    # (n_scale × n_windows) Int matrix
cp1.scales     # filtration scale axis
cp1.times      # time axis (window centres)
cp1.dim        # 1

# Manual heatmap
using CairoMakie
heatmap(cp1.times, collect(cp1.scales), cp1.surface';
        colormap = :viridis,
        axis = (xlabel="time", ylabel="scale ε", title="CROCKER H₁"))
```

**Reading a CROCKER plot:**
- A horizontal band of high $\beta_1$ at a particular scale indicates a persistent loop at that scale across many windows.
- A vertical discontinuity (sudden colour change across a column) signals a topological change — a candidate change point.
- Low scales: noise-level topology. High scales: coarse-grained structure.

**Caveats:**
- `cp.surface'` transposes to (time × scale) for standard `heatmap` conventions. The raw `.surface` is (scale × time) for direct indexing.
- Scales are auto-detected from the maximum finite death time across all windows. If a new test series has larger diagram extent, provide explicit `scales`.
- CROCKER plots are purely visual/descriptive. For formal change-point detection, use `landscape_score` or `bottleneck_score`.

---

## 9. Change-Point Detection

> Module: `ChangePoint`
> Exports: `bottleneck_score`, `wasserstein_score`, `landscape_score`, `changepoint_score`,
> `detect_changepoints`, `detect_changepoints_windowed`, `andrews_supF`, `ChangePointResult`, `ChangePointEvent`

### Score functions

Three complementary change-point scores are available, all measuring how much consecutive persistence diagrams differ.

#### `bottleneck_score`

$$\text{score}(i) = d_\infty(PH_k(W_i),\, PH_k(W_{i+1}))$$

Sensitive to **single large topological events** (one feature appears or disappears suddenly).

```julia
bn = bottleneck_score(wd, 1)   # H₁ bottleneck scores
bn.scores    # Vector{Float64} of length (n_windows - 1)
bn.times     # time of the later window for each score
```

#### `wasserstein_score`

$$\text{score}(i) = W_p(PH_k(W_i),\, PH_k(W_{i+1}))$$

Accumulates over all matched features; sensitive to **broad distributional shifts**.

```julia
ws = wasserstein_score(wd, 1; p=1)
```

#### `landscape_score`

$$\text{score}(i) = \|\lambda(PH_k(W_i)) - \lambda(PH_k(W_{i+1}))\|_{L^p}$$

Smoothest score; statistically well-founded (1-Lipschitz in diagram distance). Recommended for **statistical hypothesis testing**.

```julia
ls = landscape_score(wd, 1; n_grid=200, n_layers=3, norm_p=2)
```

#### `changepoint_score` — all at once

```julia
scores = changepoint_score(wd, 1; method=:all)
scores.bottleneck    # ChangePointResult
scores.wasserstein   # ChangePointResult
scores.landscape     # ChangePointResult

# Or a single method:
ls = changepoint_score(wd, 1; method=:landscape)
```

### `detect_changepoints` — thresholding

From a `ChangePointResult`:

```julia
detect_changepoints(result;
    threshold = :mad,    # :mad, :sigma, or a positive Float64
    min_gap   = 5,       # minimum samples between detections
    n_mad     = 3.0      # multiplier for automatic threshold
) -> Vector{ChangePointEvent}
```

```julia
events = detect_changepoints(ls; threshold=:mad, n_mad=3.0)
for ev in events
    println("Change point at t=$(ev.time), score=$(ev.score)")
end
```

From a raw score vector (more control over method):

```julia
detect_changepoints(score::Vector{Float64};
    method     = :cusum_mad,   # see table below
    n_mad      = 3.0,
    n_sigma    = 3.0,
    k          = 10,           # consecutive hits for :cusum_sustained
    win        = 50,           # window for :cusum_adaptive
    alpha      = 0.05,         # for :andrews
    r0         = 0,            # burn-in / trim (0 = auto 15%)
    min_gap    = 5,
    times      = nothing,
    score_type = :unknown,
    dim        = -1
) -> Vector{ChangePointEvent}
```

**Available detection methods:**

| Method             | Description |
|--------------------|-------------|
| `:cusum_mad`       | `median + n_mad × MAD` (robust, recommended for online use) |
| `:cusum_3sigma`    | `mean + n_sigma × σ` from burn-in period |
| `:cusum_sustained` | Must exceed threshold for `k` consecutive windows |
| `:percentile`      | 99th percentile of burn-in period as threshold |
| `:cusum_adaptive`  | Sliding-window baseline of width `win` |
| `:rss`             | Offline RSS minimisation — returns single best breakpoint |
| `:andrews`         | Andrews (1993) sup-F test — single breakpoint if significant |

```julia
# Robust online detection
events = detect_changepoints(ls.scores;
    method     = :cusum_mad,
    n_mad      = 2.5,
    min_gap    = 10,
    times      = ls.times,
    score_type = :landscape,
    dim        = 1)

# Offline econometric test (Andrews 1993)
events = detect_changepoints(ls.scores;
    method = :andrews,
    alpha  = 0.05,
    times  = ls.times)
```

### `detect_changepoints_windowed` — full pipeline in one call

```julia
events, score_vec = detect_changepoints_windowed(wd;
    dim      = 1,
    score    = :landscape,   # :landscape, :wasserstein, :bottleneck
    method   = :andrews,     # any detect_changepoints method
    n_grid   = 50,
    n_layers = 2,
    kwargs...)               # forwarded to detect_changepoints
```

### Andrews sup-F test

```julia
andrews_supF(score;
    r0    = 0,      # trimming fraction (0 = auto 15%)
    alpha = 0.05    # significance level
) -> NamedTuple
```

```julia
res = andrews_supF(ls.scores; alpha=0.05)
res.r_star      # index of the most likely breakpoint
res.sup_F       # sup-F statistic value
res.significant # true if sup_F > critical value
res.cv          # critical value from Andrews (1993) Table 1
```

Supported significance levels: 0.10, 0.05, 0.01.

**Caveats on change-point detection:**
- Bottleneck distances can be large for diagrams with many very small features (noise). Apply a persistence `threshold` when computing `persistent_homology` to reduce noise sensitivity.
- `:mad` is robust to outliers; `:cusum_3sigma` is appropriate when the score distribution is approximately Gaussian.
- `min_gap` prevents detecting the same event multiple times (a single structural shift typically elevates the score for several consecutive windows). Set it to roughly the expected duration of transition.
- Andrews sup-F is designed for a **single** structural break. For multiple breaks, run iteratively or use `:cusum_mad`.
- The Wasserstein distance in `wasserstein_score` uses a greedy augmented-assignment approximation (not exact Hungarian). It is accurate for the moderate-sized diagrams typical in windowed PH but may be slightly suboptimal for very large diagrams.

---

## 10. Sublevel-Set Persistent Homology

> Module: `Sublevel`
> Exports: `sublevel_ph`, `SublevelDiagram`, `windowed_sublevel_ph`, `periodogram_ph`, `windowed_periodogram_ph`

An alternative to embedding-based PH: treat the time series directly as a piecewise-linear function on $\{1, \ldots, N\}$ and compute the PH of its sublevel sets.

**Key advantages over Rips PH:**
- No embedding parameters (no `dim` or `lag` to choose)
- O(N log N) per window instead of O(N^{d+1})
- H₀ bars correspond directly to local minima; H₁ (extended) to oscillatory features
- Suitable for very long series or real-time applications

**Limitation:** restricted to H₀ and H₁; no higher-dimensional topology.

### `sublevel_ph`

```julia
sublevel_ph(x; extended=false) -> SublevelDiagram
```

```julia
f   = sin.(range(0, 4π, length=400)) .+ 0.1 .* randn(400)
dgm = sublevel_ph(f)

dgm.H0   # H₀ pairs: local minima ↔ saddles (finite)
dgm.H1   # H₁ pairs: empty unless extended=true
dgm.n    # 400

# With extended persistence (H₁ captures oscillatory features)
dgm_ext = sublevel_ph(f; extended=true)
dgm_ext.H1   # pairs corresponding to local maxima ↔ minima

# Indexing (1-based, consistent with DiagramCollection):
dgm[1]   # H₀ pairs
dgm[2]   # H₁ pairs
```

**Reading H₀ bars:**
- Each pair $(b, d)$ represents a connected component born at level $b$ (local minimum) and dying when it merges with an older component at level $d$.
- The most persistent H₀ bar (largest $d - b$) corresponds to the deepest local minimum (most prominent oscillation trough).
- One H₀ bar always has $d = \infty$ (the global minimum component; excluded from `H0_finite`).

### `windowed_sublevel_ph`

```julia
wd = windowed_sublevel_ph(ts; window=200, step=20, extended=false)
# WindowedDiagrams{SublevelDiagram}

# Plug into change-point detection:
sc = landscape_score(wd, 0)   # H₀ score
events = detect_changepoints(sc; threshold=:mad)
```

### `periodogram_ph` — spectral TDA

```julia
periodogram_ph(signal; bw=5, fs=1.0) -> SublevelDiagram
```

Applies sublevel-set PH to the smoothed power spectral density. Prominent spectral peaks appear as long-lived H₀ bars.

```julia
sig  = sin.(range(0, 20π, length=500)) .+ 0.2 .* randn(500)
dgm  = periodogram_ph(sig; bw=5)
dgm.H0  # H₀ pairs of spectral peaks
# The most persistent bar's birth ≈ noise floor, death ≈ peak height
```

**Windowed spectral TDA (two forms):**

```julia
# Sliding-window on a continuous signal
wd = windowed_periodogram_ph(sig; window=256, step=32, bw=5)

# Per-trial (e.g. EEG epochs, bearing fault trials)
trials = [randn(512) for _ in 1:50]
wd = windowed_periodogram_ph(trials; bw=5)

# Both return WindowedDiagrams{SublevelDiagram}
sc = changepoint_score(wd, 0; method=:landscape)
```

**Caveats:**
- Extended H₁ pairs from sublevel PH do not have the same meaning as H₁ from Rips filtration. They capture oscillatory amplitude, not topological loops.
- `periodogram_ph` uses a moving-average smoother with half-bandwidth `bw`. Larger `bw` → smoother spectrum → fewer but more reliable H₀ bars.
- Sublevel PH on the time domain is sensitive to amplitude changes. For frequency-domain changes, prefer `periodogram_ph`.

---

## 11. Multivariate Embedding

> Module: `Multivariate` | Exports: `embed_multivariate`, `MultivariateEmbedding`

Generalises the Takens embedding to $k$-channel time series by interleaving delay coordinates from all channels.

```julia
embed_multivariate(X; dim, lag) -> MultivariateEmbedding
```

**Input forms:**

```julia
# Matrix input: N×k (rows = time, columns = channels)
X   = hcat(x1, x2, x3)   # 5000 × 3 matrix
emb = embed_multivariate(X; dim=2, lag=5)
# MultivariateEmbedding{Float64}: 4995 points in ℝ6 (3 channels × dim=2, lag=5)

# Vector-of-vectors input:
emb = embed_multivariate([x1, x2, x3]; dim=2, lag=5)
```

Row $i$ of `emb.points` is:
$(x^1_i, x^1_{i+\tau}, \ldots, x^1_{i+(d-1)\tau},\ x^2_i, \ldots,\ x^k_i, \ldots, x^k_{i+(d-1)\tau})$

```julia
emb.n_channels    # k = 3
emb.dim           # per-channel embedding dimension
emb.lag           # per-channel lag
size(emb)         # (n_points, k*dim) = (4995, 6)

# Pass directly to persistent_homology
dgms = persistent_homology(emb; dim_max=2)
```

**Caveats:**
- The same `dim` and `lag` are used for all channels. To use different lags per channel, construct the embedding manually and pass the resulting matrix.
- Total point-cloud dimension is `k × dim`. With 5 channels and dim=3, you get 15-dimensional points. PH in high dimensions is computationally expensive — subsample aggressively.
- All channels must have the same length.

---

## 12. Diagram Kernels

> Module: `DiagramKernels`
> Exports: `pss_kernel`, `pwg_kernel`, `sliced_wasserstein_kernel`, `kernel_matrix`, `wasserstein_distance`

Kernel functions between persistence diagrams enable kernel-based classification, regression, and PCA on topological features. All three implemented kernels are positive semi-definite.

### Persistence Scale-Space Kernel (PSS)

```julia
pss_kernel(dgm1, dgm2; sigma=1.0) -> Float64
```

$$k_\sigma(D_1, D_2) = \frac{1}{8\pi\sigma} \sum_{p \in D_1} \sum_{q \in D_2} \left[\exp\!\left(-\frac{\|p-q\|^2}{8\sigma}\right) - \exp\!\left(-\frac{\|p-\bar{q}\|^2}{8\sigma}\right)\right]$$

where $\bar{q}$ is the reflection of $q$ across the diagonal.

```julia
k = pss_kernel(dgms1[2], dgms2[2]; sigma=0.5)   # H₁ diagrams
```

### Persistence Weighted Gaussian Kernel (PWG)

```julia
pwg_kernel(dgm1, dgm2; sigma=1.0, C=1.0) -> Float64
```

Weights diagram points by their persistence via $w(b,d) = \tanh(C(d-b))$, suppressing noise-level features.

```julia
k = pwg_kernel(dgms1[2], dgms2[2]; sigma=0.3, C=2.0)
```

### Sliced Wasserstein Kernel

```julia
sliced_wasserstein_kernel(dgm1, dgm2; sigma=1.0, n_directions=100) -> Float64
```

Approximates the Wasserstein distance via random projections, then exponentiates. Efficient: O(M · n log n).

```julia
k = sliced_wasserstein_kernel(dgms1[2], dgms2[2]; sigma=0.5, n_directions=200)
```

### Gram matrix for a collection

```julia
kernel_matrix(dgms, dim;
    kernel  = :pss,   # :pss, :pwg, or :sliced_wasserstein
    kwargs...         # passed to the kernel function
) -> Matrix{Float64}
```

```julia
K = kernel_matrix(all_dgms, 1; kernel=:pss, sigma=0.3)
# Symmetric (n × n) PSD matrix; use with kernel SVM, kernel PCA, GP regression
```

### Exact Wasserstein distance

```julia
wasserstein_distance(d1, d2; p=2) -> Float64
```

Computes the p-Wasserstein distance between two diagrams given as `Vector{Tuple{Float64,Float64}}` using the Hungarian algorithm (exact).

```julia
dgm1 = periodogram_ph(sig1).H0
dgm2 = periodogram_ph(sig2).H0
d = wasserstein_distance(dgm1, dgm2; p=2)
```

**Caveats:**
- `pss_kernel` and `pwg_kernel` take `dgm1`, `dgm2` as Ripserer diagram objects (elements of a `DiagramCollection`); use `dgms[k+1]` for dimension $k$.
- `wasserstein_distance` takes `Vector{Tuple{Float64,Float64}}` directly (e.g., `sublevel_ph(x).H0`).
- All kernels ignore infinite-death bars.
- `sigma` controls the scale of similarity. Too small: all diagrams appear distinct. Too large: all appear identical. Cross-validate.
- `kernel_matrix` is O(n²) kernel evaluations; for large collections, this can be slow.

---

## 13. Feature Extraction for ML Pipelines

> Module: `Features` | Exports: `topo_features`, `TopoFeatureSpec`, `feature_names`

`topo_features` runs the complete pipeline (embed → PH → vectorise) and returns a single `Float64` vector. It is the highest-level function in the package.

### `TopoFeatureSpec` — configuration

```julia
spec = TopoFeatureSpec(
    # Embedding
    dim_max      = 1,         # max PH dimension
    dim          = 2,         # Takens embedding dimension
    lag          = 1,         # Takens lag
    filtration   = :rips,
    threshold    = Inf,
    # Landscape
    use_landscape          = true,
    n_landscape_layers     = 3,
    n_landscape_grid       = 50,
    # Betti curve
    use_betti              = true,
    n_betti_grid           = 50,
    # Summary statistics
    use_stats              = true,    # total_pers (p=1,2), entropy, amplitude (∞,2)
    # Persistence image
    use_image              = false,
    n_image_pixels         = 10,
)
```

### `topo_features` — from a time series

```julia
topo_features(x; spec=TopoFeatureSpec(), tgrid_landscape=nothing, tgrid_betti=nothing)
    -> Vector{Float64}
```

```julia
spec = TopoFeatureSpec(dim=3, lag=10, dim_max=1, n_landscape_grid=50)
feat = topo_features(ts; spec=spec)
length(feat)   # depends on spec; use feature_names to check

# Feature matrix for a collection
X = reduce(hcat, topo_features(ts_i; spec=spec) for ts_i in ensemble)'
# X is (n_series × n_features)
```

### `topo_features` — from an existing `DiagramCollection`

```julia
dgms = persistent_homology(emb; dim_max=1)
feat = topo_features(dgms; spec=spec)
```

Use this form when you want to separate PH computation (expensive) from feature extraction (cheap), e.g., when trying multiple `spec` configurations.

### `feature_names` — label the feature vector

```julia
names = feature_names(spec)
length(names) == length(feat)  # always true

Dict(zip(names, feat))   # named feature map

# Example names:
# "landscape_H0_L1_t1", "landscape_H0_L1_t2", ...
# "betti_H0_t1", ...
# "total_pers_H0_p1", "total_pers_H0_p2",
# "entropy_H0", "amplitude_H0_inf", "amplitude_H0_p2"
```

### Shared grids for consistent features

When computing features for multiple series to compare, use a fixed grid:

```julia
spec = TopoFeatureSpec(dim=3, lag=10, dim_max=1)

# Compute one reference embedding/PH to determine scale range
ref_emb  = embed(ref_ts; dim=spec.dim, lag=spec.lag)
ref_dgms = persistent_homology(ref_emb; dim_max=spec.dim_max)
ref_λ    = landscape(ref_dgms, 1; n_grid=spec.n_landscape_grid)
ref_bc   = betti_curve(ref_dgms, 1; n_grid=spec.n_betti_grid)

tgrid_L = ref_λ.tgrid
tgrid_B = ref_bc.tgrid

# Now use shared grids for all series
features = [topo_features(ts_i; spec=spec,
                           tgrid_landscape=tgrid_L,
                           tgrid_betti=tgrid_B)
            for ts_i in all_series]
```

**Caveats:**
- Without shared grids, different series will have different scale ranges and the feature vectors cannot be compared directly.
- `topo_features` from a time series always uses the full embedding. If you have already subsampled the point cloud, call `persistent_homology` separately, then call `topo_features(dgms; spec=spec)`.
- Feature vector length: for `dim_max=1`, `n_landscape_layers=3`, `n_landscape_grid=50`, `n_betti_grid=50`, `use_stats=true`, `use_image=false`:
  - Landscape: 2 dimensions × 3 layers × 50 = 300
  - Betti: 2 × 50 = 100
  - Stats: 2 × 5 = 10
  - Total: **410 features**

---

## 14. Visualisations

> Module: `Visualisations`
> Exports: `plot_diagram`, `plot_diagram_multi`, `plot_barcode`, `plot_landscape`, `plot_betti_curve`, `plot_crocker`, `plot_crocker_multi`, `plot_changepoint_score`

All functions require a Makie backend loaded first (`using CairoMakie` etc.) and return a `Makie.Figure`.

### Persistence diagram

```julia
plot_diagram(dgms, dim=1;
    colormap       = :plasma,
    point_size     = 10,
    diagonal       = true,
    diagonal_color = :gray60,
    title          = "auto",
    infinite_label = true,    # show ∞ marker for unpaired bars
    threshold      = 0.0,     # hide points with persistence < threshold
    figsize        = (500, 500)
) -> Figure
```

```julia
using CairoMakie
fig = plot_diagram(dgms, 1; threshold=0.05)
save("diagram_H1.pdf", fig)
```

### Multi-dimension diagram grid

```julia
fig = plot_diagram_multi(dgms;
    colormap  = :plasma,
    threshold = 0.0,
    figsize   = (480 * length(dgms), 480))
```

### Barcode

```julia
fig = plot_barcode(dgms, 1;
    bar_height = 0.7,
    inf_extend = 0.05,    # how far to extend infinite bars beyond max
    threshold  = 0.0,
    figsize    = (600, 400))
```

### Persistence landscape

```julia
fig = plot_landscape(λ;
    n_layers   = size(λ.layers, 1),
    colormap   = :tab10,
    linewidth  = 2,
    fill_first = true,     # shade under the first layer
    figsize    = (700, 350))
```

### Betti curve

```julia
fig = plot_betti_curve(bc;
    color   = :steelblue,
    fill    = true,
    figsize = (650, 300))
```

### CROCKER plot

```julia
fig = plot_crocker(cp;
    colormap    = :viridis,
    vlines      = [ev.time for ev in events],  # mark detected events
    vline_color = :red,
    figsize     = (800, 400))
```

### Stacked CROCKER plots

```julia
fig = plot_crocker_multi([cp0, cp1];
    titles = ["H₀ — connected components", "H₁ — loops"],
    vlines = Float64[],
    figsize = (800, 650))
```

### Change-point score plot

```julia
using StatsBase: mad, median

threshold_val = median(ls.scores) + 2 * mad(ls.scores)

fig = plot_changepoint_score(ls;
    events    = events,
    threshold = threshold_val,
    color     = :steelblue,
    figsize   = (800, 300))
```

**Note:** `plot_changepoint_score` references `result.method` in the title. `ChangePointResult` stores `score_type` (`:bottleneck`, `:wasserstein`, or `:landscape`), not a `.method` field. If the title looks odd, override it with the `title` keyword argument.

**Caveats:**
- All `plot_*` functions error at runtime (not import time) if no Makie backend is loaded.
- Figures can be saved with `save("file.pdf", fig)`, `save("file.png", fig; px_per_unit=2)`, etc.
- For interactive exploration, use `GLMakie`; for publication output, `CairoMakie` produces vector PDFs.

---

## 15. End-to-End Examples

### Example 1: Periodic signal — verifying that H₁ detects a loop

```julia
using TopoTS
using Random; Random.seed!(1)

# Generate a noisy sinusoid
ts = sin.(range(0, 20π, length=2000)) .+ 0.1 .* randn(2000)

# Step 1: Embedding parameters
lag = ami_lag(ts)          # e.g. 16
dim = fnn_dim(ts; lag=lag) # e.g. 3

# Step 2: Embed and compute PH
emb  = embed(ts; dim=dim, lag=lag)
# Subsample for speed
idx  = round.(Int, range(1, length(emb), length=500))
dgms = persistent_homology(emb.points[idx, :]; dim_max=1)

# Step 3: Verify H₁ loop
h1 = dgms[2]
lifetimes = [death(p) - birth(p) for p in h1 if isfinite(death(p))]
# The most persistent H₁ bar corresponds to the loop traced by the periodic orbit

# Step 4: Vectorise
λ = landscape(dgms, 1; n_layers=3, n_grid=200)
println("Landscape norm: ", landscape_norm(λ, 2))

# Step 5: Visualise
using CairoMakie
fig1 = plot_diagram(dgms, 1; title="Sinusoid — H₁")
fig2 = plot_landscape(λ)
save("sinusoid_diagram.pdf", fig1)
save("sinusoid_landscape.pdf", fig2)
```

---

### Example 2: Change-point detection in an EEG-like signal

```julia
using TopoTS
using Random; Random.seed!(42)

N    = 2000
t    = range(0, 4π, length=N)
seg1 = sin.(10 .* t) .+ 0.5 .* sin.(20 .* t) .+ 0.2 .* randn(N)
seg2 = 2.0 .* sin.(2 .* t) .+ 0.8 .* randn(N)
ts   = vcat(seg1[1:N÷2], seg2[N÷2+1:end])
# True change point at sample 1000

# Windowed PH
wd = windowed_ph(ts;
    window = 100,
    step   = 20,
    dim    = 3,
    lag    = 3,
    dim_max = 1)

# Landscape change-point score
ls     = landscape_score(wd, 1; n_grid=100, n_layers=2)
events = detect_changepoints(ls; threshold=:mad, n_mad=2.5)

println("True CP at sample 1000")
for ev in events
    println("Detected: t=$(ev.time)  (sample ≈ $(round(Int, ev.time)))")
end

# CROCKER plots
cp1 = crocker(wd; dim=1, n_scale=50)

# Andrews sup-F for a single-breakpoint formal test
res = andrews_supF(ls.scores; alpha=0.05)
println("Andrews sup-F=$(round(res.sup_F, digits=2)), significant=$(res.significant)")
println("Estimated break at window $(res.r_star) ≈ sample $(round(Int, wd.times[res.r_star]))")

# Visualise
using CairoMakie
vl  = Float64[ev.index for ev in events]
fig = plot_crocker(cp1; vlines=vl, title="EEG CROCKER H₁")
save("eeg_crocker.pdf", fig)
```

---

### Example 3: Regime classification with kernel SVM

```julia
using TopoTS
using Random; Random.seed!(7)

# Generate two classes: GBM (trending) and OU (mean-reverting)
gbm(n) = cumsum(0.001 .+ 0.02 .* randn(n))
ou(n)  = (x = zeros(n); for i in 2:n; x[i] = x[i-1] + 0.1*(0.0 - x[i-1]) + 0.02*randn(); end; x)

function compute_dgm(ts)
    lag  = max(1, ami_lag(ts))
    d    = max(2, min(fnn_dim(ts; lag=lag), 4))
    emb  = embed(ts; lag=lag, dim=d)
    n    = size(emb.points, 1)
    idx  = round.(Int, range(1, n, length=min(300, n)))
    persistent_homology(emb.points[idx, :]; dim_max=1, filtration=:rips)
end

n_per_class = 20
dgms_gbm = [compute_dgm(gbm(300)) for _ in 1:n_per_class]
dgms_ou  = [compute_dgm(ou(300))  for _ in 1:n_per_class]
all_dgms = vcat(dgms_gbm, dgms_ou)

# Kernel matrix (PSS kernel on H₁)
K = kernel_matrix(all_dgms, 1; kernel=:pss, sigma=0.3)

# Labels
y = vcat(fill(1, n_per_class), fill(-1, n_per_class))

# The Gram matrix K can be passed to any kernel SVM implementation,
# e.g., LIBSVM.jl or scikit-learn via PythonCall.
# Simple centroid classifier for illustration:
K_train = K[1:2n_per_class, 1:2n_per_class]
c_gbm   = mean(K_train[1:n_per_class, :], dims=1)[1, :]
c_ou    = mean(K_train[n_per_class+1:end, :], dims=1)[1, :]

pred = [argmax([dot(K_train[i, :], c_gbm), dot(K_train[i, :], c_ou)]) == 1 ? 1 : -1
        for i in 1:2n_per_class]
acc  = mean(pred .== y)
println("Centroid accuracy: $(round(100acc, digits=1))%")
```

---

### Example 4: Sublevel-set PH for fast spectral monitoring

```julia
using TopoTS

# Simulated bearing signal: gradual frequency increase
N  = 10_000
t  = (1:N) ./ 1000.0
f0 = 50.0
signal = sin.(2π .* (f0 .+ 0.005 .* (1:N)) .* t)

# Windowed spectral TDA (no embedding parameters needed)
wd = windowed_periodogram_ph(signal; window=512, step=64, bw=3)

events, sc = detect_changepoints_windowed(wd;
    dim    = 0,
    score  = :landscape,
    method = :cusum_adaptive,
    win    = 20)

println("$(length(events)) frequency-shift events detected")
for ev in events
    println("  sample ≈ $(round(Int, ev.time))")
end
```

---

## 16. Performance Notes and Caveats

### Point cloud size

Rips PH is the computational bottleneck. Practical limits:

| Points | dim_max=1 | dim_max=2 |
|--------|-----------|-----------|
| 200    | < 1 s     | ~1–5 s    |
| 500    | ~1–5 s    | very slow |
| 1000   | ~30–120 s | rarely feasible |

**Subsample** the embedded point cloud before computing PH:

```julia
n    = size(emb.points, 1)
idx  = round.(Int, range(1, n, length=500))   # evenly spaced
pts  = emb.points[idx, :]
dgms = persistent_homology(pts; dim_max=1)
```

Alternatively, use a `threshold` to truncate the filtration:

```julia
dgms = persistent_homology(pts; dim_max=1, threshold=2.0)
# Only features born before scale 2.0 are computed
```

### Choosing `dim_max`

- `dim_max=1` is almost always sufficient for time series: H₁ captures loops (oscillatory behaviour), H₀ captures connected components (regime structure).
- `dim_max=2` (voids) is rarely informative for 1-D signals and is computationally expensive.
- FNN selects the embedding dimension of the attractor. For PH, you typically only need `dim_max = embedding_dim - 1`.

### Windowed PH speed

- Use `:edge_collapsed` or `:alpha` instead of `:rips` for point clouds in ≤ 3D.
- Sublevel-set `windowed_sublevel_ph` is 10–100× faster than `windowed_ph` and works well for detecting amplitude and frequency changes.

### Numerical stability

- All diagram distances (`_bottleneck`, `_wasserstein`) use the L∞ ground metric. Results are exact up to floating-point precision.
- The Wasserstein distance in `ChangePoint` uses a greedy augmented assignment, which may be slightly suboptimal for very large diagrams. For exact results, use `wasserstein_distance` from `DiagramKernels` (Hungarian algorithm).

### Reproducibility

- `ami_lag` and `fnn_dim` are deterministic.
- `sliced_wasserstein_kernel` uses random projections; fix `Random.seed!` for reproducibility.
- `bootstrap_landscape` and `permutation_test` use Julia's global RNG; fix `Random.seed!` before calling.

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ArgumentError: dim=1 not computed` | `dim_max` too low | Increase `dim_max` |
| `window=50 too small for dim=3, lag=10` | Window < $(d-1)\tau+2$ | Increase `window` or decrease `dim`/`lag` |
| `need at least 2 landscapes` | Too few series for bootstrap | Collect more series |
| `Čech filtration: libcech not found` | CechCore_jll not installed | `Pkg.add("CechCore_jll")` |
| `plot_* functions require a Makie backend` | No backend loaded | Add `using CairoMakie` |
| `DimensionMismatch` in `mean_landscape` | Landscapes on different grids | Use explicit shared `tgrid` |

---

*TopoTS.jl is designed to be modular: each step of the pipeline can be run independently and the results passed between functions. The full pipeline (embed → PH → features) is available in a single call via `topo_features`, while individual steps are exposed for research and experimentation.*
