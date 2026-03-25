"""
    Multivariate

Joint delay embedding for multivariate time series.

Generalises the Takens embedding to ``k``-variate inputs by interleaving
the delay coordinates of all channels into a single high-dimensional
point cloud.

# Reference
Pecora, L. M., Moniz, L., Nichols, J., & Carroll, T. L. (2007).
A unified approach to attractor reconstruction.
*Chaos*, 17(1), 013110.
"""
module Multivariate

using ..Embedding: TakensEmbedding

export embed_multivariate, MultivariateEmbedding

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    MultivariateEmbedding{T}

Result of a joint multivariate delay embedding.

# Fields
- `points   :: Matrix{T}` — ``n \\times (k \\cdot d)`` point cloud,
                             where ``k`` is the number of channels and
                             ``d`` is the embedding dimension per channel
- `dim      :: Int`       — per-channel embedding dimension
- `lag      :: Int`       — per-channel time lag (same for all channels)
- `n_channels :: Int`     — number of input channels ``k``
- `n_orig   :: Int`       — original series length

# Notes
The total embedding dimension is `n_channels × dim`.
Row `i` of `points` is
```
(x¹ᵢ, x¹ᵢ₊τ, …, x¹ᵢ₊(d-1)τ,  x²ᵢ, x²ᵢ₊τ, …,  xᵏᵢ, …, xᵏᵢ₊(d-1)τ)
```
"""
struct MultivariateEmbedding{T<:Real}
    points     :: Matrix{T}
    dim        :: Int
    lag        :: Int
    n_channels :: Int
    n_orig     :: Int
end

Base.size(e::MultivariateEmbedding) = size(e.points)
Base.length(e::MultivariateEmbedding) = size(e.points, 1)

function Base.show(io::IO, e::MultivariateEmbedding{T}) where T
    n, D = size(e.points)
    print(io, "MultivariateEmbedding{$T}: $n points in ℝ$D ",
              "($(e.n_channels) channels × dim=$(e.dim), lag=$(e.lag))")
end

# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    embed_multivariate(X::AbstractMatrix;
                       dim :: Int,
                       lag :: Int) -> MultivariateEmbedding

Construct a joint delay embedding from a multivariate time series.

# Arguments
- `X`   — ``N \\times k`` matrix; rows are time steps, columns are channels
- `dim` — embedding dimension per channel (``d \\geq 1``)
- `lag` — time delay per channel (``\\tau \\geq 1``)

# Returns
A `MultivariateEmbedding` with ``n = N - (d-1)\\tau`` rows and
``k \\cdot d`` columns.

# Example
```julia
# Lorenz system: three coupled channels
X = lorenz_trajectory(5000)   # 5000 × 3
emb = embed_multivariate(X; dim=2, lag=5)
# MultivariateEmbedding{Float64}: 4995 points in ℝ6 (3 channels × dim=2, lag=5)

dgms = persistent_homology(emb; dim_max=2)
```
"""
function embed_multivariate(X::AbstractMatrix{T};
                             dim :: Int,
                             lag :: Int) where T <: Real
    N, k = size(X)
    dim ≥ 1 || throw(ArgumentError("dim must be ≥ 1"))
    lag ≥ 1 || throw(ArgumentError("lag must be ≥ 1"))
    n = N - (dim - 1) * lag
    n > 0 || throw(ArgumentError(
        "Series too short: need N > (dim-1)*lag = $((dim-1)*lag), got N=$N"))

    D   = k * dim
    pts = Matrix{T}(undef, n, D)

    @inbounds for i in 1:n
        col = 0
        for ch in 1:k
            for j in 1:dim
                col += 1
                pts[i, col] = X[i + (j-1)*lag, ch]
            end
        end
    end

    return MultivariateEmbedding(pts, dim, lag, k, N)
end

"""
    embed_multivariate(xs::AbstractVector{<:AbstractVector};
                       dim :: Int,
                       lag :: Int) -> MultivariateEmbedding

Convenience overload accepting a vector of channel vectors.

# Example
```julia
emb = embed_multivariate([x1, x2, x3]; dim=2, lag=5)
```
"""
function embed_multivariate(xs::AbstractVector{<:AbstractVector{T}};
                              dim :: Int,
                              lag :: Int) where T <: Real
    k = length(xs)
    k ≥ 1 || throw(ArgumentError("need at least one channel"))
    N = length(xs[1])
    all(length(x) == N for x in xs) ||
        throw(DimensionMismatch("all channels must have the same length"))
    X = reduce(hcat, xs)   # N × k
    return embed_multivariate(X; dim=dim, lag=lag)
end

# ── Make MultivariateEmbedding work with persistent_homology ─────────────────
# persistent_homology accepts any object with a .points field of type Matrix.
# MultivariateEmbedding satisfies this, so no additional dispatch is needed.

end # module Multivariate
