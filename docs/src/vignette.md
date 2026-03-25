# TopoTS.jl: Topological Data Analysis for Time Series
## Package Vignette

**Author:** Stanisław M. S. Halkiewicz  
**Version:** 0.1.0

---

## 1. Introduction

TopoTS.jl is a Julia package implementing the full pipeline for Topological
Data Analysis (TDA) of time series. It bridges the gap between raw sequential
observations and the persistent-homology machinery developed in Chapters 2–4
of this thesis, providing:

- **Takens delay embedding** with automatic parameter selection (AMI lag, FNN dimension)
- **Persistent homology** via Ripserer.jl with a clean interface
- **Vectorisations**: Betti curves, persistence landscapes, and persistence images
- **Statistical inference**: mean landscapes, bootstrap confidence bands, and
  permutation tests

The package is designed so that the four-step pipeline — embed → compute → vectorise → test
— can be expressed in a few lines of Julia, while each component remains independently
accessible for custom workflows.

---

## 2. Installation

```julia
using Pkg
Pkg.add(url="https://github.com/shalkiewicz/TopoTS.jl")
```

Or with the registry (once registered):

```julia
Pkg.add("TopoTS")
```

---

## 3. Quick Start

```julia
using TopoTS

# Generate a noisy sinusoidal time series
N  = 2000
ts = sin.(range(0, 20π, length=N)) .+ 0.05 .* randn(N)

# Step 1: Takens embedding with automatic parameter selection
τ   = optimal_lag(ts)           # first minimum of AMI
d   = optimal_dim(ts; lag=τ)    # FNN criterion
emb = embed(ts; dim=d, lag=τ)

# Step 2: Persistent homology (H₀ and H₁)
dgms = persistent_homology(emb; dim_max=1)

# Step 3: Vectorise
bc = betti_curve(dgms, 1)               # Betti curve
λ  = landscape(dgms, 1; n_layers=3)    # Persistence landscape
img = persistence_image(dgms, 1)        # Persistence image (20×20)

# Step 4: Summary statistics
println("Total persistence (H₁): ", total_persistence(dgms, 1))
println("Persistent entropy (H₁): ", persistent_entropy(dgms, 1))
println("Amplitude (H₁):          ", amplitude(dgms, 1))
```

---

## 4. Filtrations

TopoTS supports five filtration types, all accessible through the same
`filtration=` keyword argument.

```julia
emb = embed(ts; dim=3, lag=10)

dgms_rips  = persistent_homology(emb; filtration=:rips)           # default
dgms_alpha = persistent_homology(emb; filtration=:alpha)          # Delaunay-based
dgms_cech  = persistent_homology(emb; filtration=:cech)           # exact (C++)
dgms_ec    = persistent_homology(emb; filtration=:edge_collapsed) # fast Rips
dgms_cub   = persistent_homology(emb; filtration=:cubical)        # 1-D signal
```

| Filtration        | Criterion                      | Notes                                    |
|-------------------|--------------------------------|------------------------------------------|
| `:rips`           | all pairwise dists ≤ 2ε        | Default; fast; any dimension             |
| `:alpha`          | circumradius via Delaunay      | O(n log n); exact; ≤3D only              |
| `:cech`           | circumradius of miniball ≤ ε   | Exact; requires compiled `libcech`       |
| `:edge_collapsed` | same as Rips, smaller complex  | Drop-in for `:rips`; faster for n > 500  |
| `:cubical`        | sublevel-set on 1-D signal     | No embedding needed; H₀ only            |

### Čech vs Rips

The Čech complex is the *exact* nerve of a ball cover and satisfies the
Nerve Theorem. In Euclidean space, Čech and Rips are related by:

```
Čech_ε  ⊆  Rips_ε  ⊆  Čech_{√2·ε}
```

So their diagrams agree up to a √2 rescaling. Use Čech when geometric
exactness matters; use Rips or EdgeCollapsedRips for speed.

### Building libcech

The Čech filtration requires a compiled C++ library:

```julia
using Pkg; Pkg.build("TopoTS")   # runs deps/build.jl automatically
```

Or manually:
```bash
cd <package_root>/csrc && make
```

Requires g++ ≥ 9 or clang++ ≥ 10 with C++17 support.

---

## 5. Detailed Usage

### 5.1 Takens Embedding

The `embed` function constructs the delay-embedding matrix from a scalar
time series. Given a series x₁, …, xₙ, the embedding produces the point cloud

```
Xᵢ = (xᵢ, xᵢ₊τ, xᵢ₊₂τ, …, xᵢ₊(d-1)τ),   i = 1, …, N - (d-1)τ
```

**Parameter selection:**

```julia
# Average Mutual Information for lag selection
τ = ami_lag(ts; max_lag=50, nbins=32)

# False Nearest Neighbours for dimension selection
d = fnn_dim(ts; lag=τ, max_dim=10)

# Or use convenience wrappers:
τ = optimal_lag(ts)
d = optimal_dim(ts; lag=τ)
```

The `TakensEmbedding` struct stores the point cloud and parameters:

```julia
emb = embed(ts; dim=3, lag=τ)
emb.points    # (n_points × 3) matrix
emb.dim       # 3
emb.lag       # τ
emb.n_orig    # N
```

### 5.2 Persistent Homology

```julia
dgms = persistent_homology(emb; dim_max=2, filtration=:rips)

# Access diagrams by dimension
dgms[1]   # H₀ (connected components)
dgms[2]   # H₁ (loops)
dgms[3]   # H₂ (voids)

# Alpha filtration (for 2D or 3D embeddings)
dgms_alpha = persistent_homology(emb; filtration=:alpha)

# Threshold to speed up computation on large point clouds
dgms_thresh = persistent_homology(emb; threshold=2.0)
```

The `DiagramCollection` wraps Ripserer's output and stores the filtration type
and number of input points for provenance tracking.

### 5.3 Betti Curves

The Betti curve β_k(ε) counts the number of k-dimensional features alive at
scale ε:

```julia
bc0 = betti_curve(dgms, 0; n_grid=500)     # connected components
bc1 = betti_curve(dgms, 1; n_grid=500)     # loops

# Custom grid
tgrid = range(0, 3, length=300)
bc1 = betti_curve(dgms, 1; tgrid=tgrid)

# Access values
bc1.values    # Vector{Int}
bc1.tgrid     # scale grid
```

### 5.4 Persistence Landscapes

Persistence landscapes (Bubenik, 2015) map each diagram to a sequence of
piecewise-linear functions in L²(ℝ), enabling averaging and hypothesis testing:

```julia
λ = landscape(dgms, 1;
              n_grid   = 500,    # grid resolution
              n_layers = 5)      # number of landscape functions

λ.layers      # (5 × 500) matrix; layers[k, :] = λ_k(t)
λ.tgrid       # scale grid

# Norm
landscape_norm(λ, 2)    # L² norm (summed over layers)
landscape_norm(λ, Inf)  # sup norm
```

**Mean landscape across an ensemble:**

```julia
λs = [landscape(persistent_homology(embed(ts_i; dim=d, lag=τ)), 1)
      for ts_i in time_series_collection]

λ_mean = mean_landscape(λs)
```

### 5.5 Persistence Images

Persistence images (Adams et al., 2017) provide fixed-size feature vectors
for machine-learning pipelines:

```julia
img = persistence_image(dgms, 1;
                        n_pixels = 20,      # 20×20 grid
                        sigma    = 0.1,     # Gaussian bandwidth
                        weight   = :linear) # w(b,p) = p/p_max

# Feature vector for classification / regression
feat = vec(img)    # 400-dimensional Float64 vector
```

### 5.6 Summary Statistics

```julia
# Total persistence: Σ (dᵢ - bᵢ)^p
total_persistence(dgms, 1; p=1)   # L¹ total persistence
total_persistence(dgms, 1; p=2)   # L² total persistence

# Persistent entropy: -Σ ℓᵢ/L · log(ℓᵢ/L)
persistent_entropy(dgms, 1)

# Amplitude: max half-persistence (= bottleneck distance to empty diagram)
amplitude(dgms, 1; p=Inf)
amplitude(dgms, 1; p=2)
```

---

## 6. Statistical Inference

### 5.1 Bootstrap Confidence Bands

```julia
# Assume λs is a Vector{PersistenceLandscape} from an ensemble
result = bootstrap_landscape(λs; n_boot=1000, alpha=0.05)

result.mean        # PersistenceLandscape — observed mean
result.lower       # PersistenceLandscape — 2.5th percentile band
result.upper       # PersistenceLandscape — 97.5th percentile band
result.boot_means  # all bootstrap mean landscapes

# Quick version (mean + band only)
band = confidence_band(λs; n_boot=500)
```

### 5.2 Two-Sample Permutation Test

Test whether two groups of time series have the same topological structure:

```julia
# H₀: E[λ₁] = E[λ₂]
result = permutation_test(λs_group1, λs_group2;
                          n_perm = 999,
                          stat   = :l2_mean_diff)

result.pvalue      # permutation p-value
result.statistic   # observed ‖mean(λ₁) - mean(λ₂)‖₂
result.null_dist   # 999 permuted statistic values
```

### 5.3 Pointwise t-test on Landscape Layers

For identifying *where* on the scale axis two groups differ:

```julia
result = landscape_ttest(λs_group1, λs_group2; layer=1)

result.pvalues    # pointwise p-values (n_grid vector)
result.tstats     # pointwise t-statistics
result.tgrid      # scale grid
```

---

## 7. Complete Example: Periodic vs. Noise Detection

```julia
using TopoTS
using Random
Random.seed!(42)

# Generate two classes of time series
n_samples = 20
N = 1000

# Class 1: periodic (circle attractor → β₁ = 1)
periodic = [sin.(range(0, 20π, length=N)) .+ 0.08 .* randn(N)
            for _ in 1:n_samples]

# Class 2: random noise (no topology → β₁ ≈ 0)
noise = [randn(N) for _ in 1:n_samples]

# Pipeline
τ = 12   # fixed lag for comparability
d = 3    # embedding dimension

function to_landscape(ts)
    emb  = embed(ts; dim=d, lag=τ)
    dgms = persistent_homology(emb; dim_max=1, threshold=3.0)
    landscape(dgms, 1; n_grid=200, n_layers=3)
end

λs_periodic = to_landscape.(periodic)
λs_noise    = to_landscape.(noise)

# Permutation test
test_result = permutation_test(λs_periodic, λs_noise; n_perm=999)
println("p-value: ", test_result.pvalue)
# Expected: p-value ≈ 0.001 (strong evidence against H₀)

# Bootstrap confidence bands
band_periodic = confidence_band(λs_periodic; n_boot=500)
band_noise    = confidence_band(λs_noise;    n_boot=500)

# The mean landscape for periodic data has a prominent peak;
# for noise it is flat near zero.
```

---

## 8. Design Notes

**Dependencies.** TopoTS.jl has a minimal dependency footprint:
Ripserer.jl (core PH computation), Distances.jl, Statistics, and StatsBase.
Plotting is intentionally not bundled; users can plot `PersistenceLandscape.layers`
directly with CairoMakie.jl, Makie.jl, or Plots.jl.

**Coefficient field.** Persistent homology is computed over ℤ/2ℤ by default
(as is standard). The `modulus` keyword to `persistent_homology` allows other
prime fields.

**Performance.** The FNN computation in `fnn_dim` uses a brute-force
O(n²) nearest-neighbour search, sufficient for typical time series lengths
(N ≤ 10,000). For longer series, provide `lag` and `dim` manually and bypass
automatic parameter selection.

---

## 9. References

- Takens, F. (1981). Detecting strange attractors in turbulence. *Lecture Notes in Mathematics*, 898.
- Bubenik, P. (2015). Statistical TDA using persistence landscapes. *JMLR*, 16(1), 77–102.
- Adams, H. et al. (2017). Persistence images. *JMLR*, 18(1), 218–252.
- Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence barcodes. *JACT*, 1(1), 41–60.
- Chazal, F. et al. (2015). Subsampling methods for persistent homology. *ICML*.
- Fraser, A. M. & Swinney, H. L. (1986). Independent coordinates for strange attractors from mutual information. *Phys. Rev. A*, 33(2), 1134.
- Kennel, M. B. et al. (1992). Determining embedding dimension for phase-space reconstruction. *Phys. Rev. A*, 45(6), 3403.
