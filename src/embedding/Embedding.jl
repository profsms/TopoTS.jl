"""
    Embedding

Takens delay embedding and parameter selection for time series.

Provides:
- `embed`      — construct the delay-embedding matrix
- `ami_lag`    — select lag via Average Mutual Information
- `fnn_dim`    — select embedding dimension via False Nearest Neighbours
- `optimal_lag`, `optimal_dim` — convenience wrappers
"""
module Embedding

using Statistics
using StatsBase: autocor, Histogram, fit

export embed, TakensEmbedding,
       optimal_lag, optimal_dim,
       ami_lag, fnn_dim

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    TakensEmbedding{T}

Result of a delay embedding. Stores the point cloud as an `(n_points × dim)`
matrix together with the parameters used to construct it.

# Fields
- `points :: Matrix{T}`  — embedded point cloud, rows are points in ℝᵈ
- `dim    :: Int`         — embedding dimension
- `lag    :: Int`         — time delay (in samples)
- `n_orig :: Int`         — length of the original time series
"""
struct TakensEmbedding{T<:Real}
    points :: Matrix{T}
    dim    :: Int
    lag    :: Int
    n_orig :: Int
end

Base.size(e::TakensEmbedding) = size(e.points)
Base.length(e::TakensEmbedding) = size(e.points, 1)

function Base.show(io::IO, e::TakensEmbedding{T}) where T
    n, d = size(e.points)
    print(io, "TakensEmbedding{$T}: $n points in ℝ$d (lag=$(e.lag))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Core embedding
# ─────────────────────────────────────────────────────────────────────────────

"""
    embed(x::AbstractVector; dim::Int, lag::Int) -> TakensEmbedding

Construct the Takens delay-embedding of a scalar time series `x`.

Given a time series ``x_1, x_2, \\ldots, x_N``, the embedding produces
the point cloud
```
X_i = (x_i,  x_{i+lag},  x_{i+2·lag},  …,  x_{i+(dim-1)·lag}),
      i = 1, …, N - (dim-1)·lag.
```

# Arguments
- `x`   — scalar time series (any `AbstractVector{<:Real}`)
- `dim` — embedding dimension ``d \\geq 1``
- `lag` — time delay ``\\tau \\geq 1`` (in samples)

# Returns
A `TakensEmbedding` whose `.points` field is an ``n \\times d`` matrix
with ``n = N - (d-1)\\tau`` rows.

# Example
```julia
ts  = sin.(range(0, 20π, length=2000)) .+ 0.05 .* randn(2000)
emb = embed(ts, dim=3, lag=15)
# TakensEmbedding{Float64}: 1970 points in ℝ3 (lag=15)
```

# References
Takens, F. (1981). Detecting strange attractors in turbulence.
*Lecture Notes in Mathematics*, 898, 366–381.
"""
function embed(x::AbstractVector{T}; dim::Int, lag::Int) where T<:Real
    dim ≥ 1 || throw(ArgumentError("dim must be ≥ 1, got $dim"))
    lag ≥ 1 || throw(ArgumentError("lag must be ≥ 1, got $lag"))
    N = length(x)
    n = N - (dim - 1) * lag
    n > 0 || throw(ArgumentError(
        "Time series too short: need N > (dim-1)*lag = $((dim-1)*lag), got N=$N"))
    pts = Matrix{T}(undef, n, dim)
    @inbounds for i in 1:n, j in 1:dim
        pts[i, j] = x[i + (j-1)*lag]
    end
    return TakensEmbedding(pts, dim, lag, N)
end

# ─────────────────────────────────────────────────────────────────────────────
# Lag selection: Average Mutual Information
# ─────────────────────────────────────────────────────────────────────────────

"""
    ami_lag(x::AbstractVector; max_lag::Int=50, nbins::Int=32) -> Int

Select the embedding lag as the first local minimum of the Average Mutual
Information (AMI) function.

The AMI at lag ``\\tau`` is estimated via a histogram approximation:
```
AMI(τ) = ∑_{i,j} p_{ij}(τ) log( p_{ij}(τ) / (p_i · p_j) )
```
where ``p_{ij}(\\tau)`` is the joint probability of ``(x_t, x_{t+\\tau})``
in a 2-D histogram and ``p_i``, ``p_j`` are the marginal probabilities.

The first local minimum identifies the lag at which successive observations
are maximally independent, following the recommendation of
Fraser & Swinney (1986).

# Arguments
- `x`       — scalar time series
- `max_lag` — maximum lag to search
- `nbins`   — number of histogram bins per axis

# Returns
Optimal lag (Int); defaults to 1 if no local minimum is found.

# References
Fraser, A. M., & Swinney, H. L. (1986). Independent coordinates for strange
attractors from mutual information. *Physical Review A*, 33(2), 1134.
"""
function ami_lag(x::AbstractVector; max_lag::Int=50, nbins::Int=32)
    N = length(x)
    ami_vals = Float64[]
    xmin, xmax = minimum(x), maximum(x)
    # marginal histogram
    edges = range(xmin, xmax; length=nbins+1)
    p_marg = _hist1d(x, edges, N)
    for τ in 1:max_lag
        n_joint = N - τ
        n_joint ≤ 0 && break
        p_joint = _hist2d(x[1:n_joint], x[τ+1:N], edges, n_joint)
        ami = 0.0
        for i in 1:nbins, j in 1:nbins
            pij = p_joint[i, j]
            pi  = p_marg[i]
            pj  = p_marg[j]
            if pij > 0 && pi > 0 && pj > 0
                ami += pij * log(pij / (pi * pj))
            end
        end
        push!(ami_vals, ami)
    end
    # first local minimum
    for i in 2:length(ami_vals)-1
        if ami_vals[i] < ami_vals[i-1] && ami_vals[i] < ami_vals[i+1]
            return i
        end
    end
    # fallback: global minimum
    return argmin(ami_vals)
end

function _hist1d(x, edges, N)
    nbins = length(edges) - 1
    counts = zeros(nbins)
    for xi in x
        k = searchsortedlast(edges, xi)
        k = clamp(k, 1, nbins)
        counts[k] += 1
    end
    return counts ./ N
end

function _hist2d(x1, x2, edges, N)
    nbins = length(edges) - 1
    counts = zeros(nbins, nbins)
    for (a, b) in zip(x1, x2)
        i = clamp(searchsortedlast(edges, a), 1, nbins)
        j = clamp(searchsortedlast(edges, b), 1, nbins)
        counts[i, j] += 1
    end
    return counts ./ N
end

# ─────────────────────────────────────────────────────────────────────────────
# Dimension selection: False Nearest Neighbours
# ─────────────────────────────────────────────────────────────────────────────

"""
    fnn_dim(x::AbstractVector; lag::Int, max_dim::Int=10,
            rtol::Real=10.0, atol::Real=2.0) -> Int

Select the embedding dimension via the False Nearest Neighbours (FNN)
criterion of Kennel et al. (1992).

A nearest neighbour in dimension ``d`` is *false* if it ceases to be a
neighbour in dimension ``d+1``: the relative distance increase exceeds
`rtol`, or the absolute distance in the new coordinate exceeds `atol`
times the attractor size.

Searches from `dim=1` up to `max_dim` and returns the first dimension at
which the FNN fraction drops below 1%.

# Arguments
- `x`       — scalar time series
- `lag`     — embedding lag (use `ami_lag` output)
- `max_dim` — maximum dimension to test
- `rtol`    — relative-distance threshold (default 10.0)
- `atol`    — absolute-distance threshold in units of σ(x) (default 2.0)

# Returns
Optimal embedding dimension (Int).

# References
Kennel, M. B., Brown, R., & Abarbanel, H. D. I. (1992).
Determining embedding dimension for phase-space reconstruction using a
geometrical construction. *Physical Review A*, 45(6), 3403.
"""
function fnn_dim(x::AbstractVector; lag::Int, max_dim::Int=10,
                 rtol::Real=10.0, atol::Real=2.0)
    Ra = std(x)   # attractor size estimate
    for d in 1:max_dim-1
        pts_d  = embed(x; dim=d,   lag=lag).points
        pts_d1 = embed(x; dim=d+1, lag=lag).points
        n = min(size(pts_d, 1), size(pts_d1, 1))
        n_false = 0
        for i in 1:n
            # brute-force nearest neighbour (sufficient for typical ts lengths)
            best_j, best_dist = 0, Inf
            for j in 1:n
                j == i && continue
                dist = sqrt(sum((pts_d[i, :] .- pts_d[j, :]).^2))
                if dist < best_dist
                    best_dist = dist
                    best_j = j
                end
            end
            best_j == 0 && continue
            new_coord_diff = abs(pts_d1[i, d+1] - pts_d1[best_j, d+1])
            if (new_coord_diff / best_dist) > rtol || (new_coord_diff / Ra) > atol
                n_false += 1
            end
        end
        fnn_frac = n_false / n
        fnn_frac < 0.01 && return d + 1
    end
    return max_dim
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

"""
    optimal_lag(x; kwargs...) -> Int

Wrapper for `ami_lag`. Returns the first local minimum of the AMI function.
"""
optimal_lag(x; kwargs...) = ami_lag(x; kwargs...)

"""
    optimal_dim(x; lag, kwargs...) -> Int

Wrapper for `fnn_dim`. Returns the FNN-optimal embedding dimension.
"""
optimal_dim(x; lag, kwargs...) = fnn_dim(x; lag=lag, kwargs...)

end # module Embedding
