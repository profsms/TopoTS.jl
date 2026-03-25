"""
    Bootstrap

Bootstrap methods for uncertainty quantification of persistence landscapes.

Provides confidence bands for the mean landscape and related statistics,
following the nonparametric bootstrap approach of Bubenik (2015) and
Chazal et al. (2015).
"""
module Bootstrap

using Statistics
using ..Landscapes: PersistenceLandscape, mean_landscape, landscape_norm

export bootstrap_landscape, confidence_band

"""
    bootstrap_landscape(λs::AbstractVector{PersistenceLandscape};
                        n_boot   :: Int  = 1000,
                        alpha    :: Real = 0.05,
                        stat            = :mean) -> NamedTuple

Bootstrap confidence band for the mean persistence landscape.

Draws `n_boot` bootstrap samples (with replacement) from `λs`, computes
the mean landscape for each, and returns a pointwise confidence band at
level ``1 - \\alpha``.

# Arguments
- `λs`     — vector of `PersistenceLandscape` objects (one per time series)
- `n_boot` — number of bootstrap replications (default 1000)
- `alpha`  — significance level (default 0.05 → 95% band)
- `stat`   — `:mean` (pointwise CI on mean) or `:norm` (CI on L² norm)

# Returns
A `NamedTuple` with fields:
- `mean`       — the observed mean landscape
- `lower`      — lower confidence band (PersistenceLandscape)
- `upper`      — upper confidence band (PersistenceLandscape)
- `boot_means` — all bootstrap mean landscapes

# Example
```julia
result = bootstrap_landscape(λs; n_boot=500, alpha=0.05)
plot(result.mean.tgrid, result.mean.layers[1, :];
     ribbon=(result.mean.layers[1,:] .- result.lower.layers[1,:],
             result.upper.layers[1,:] .- result.mean.layers[1,:]))
```

# References
Bubenik, P. (2015). Statistical topological data analysis using persistence
landscapes. *JMLR*, 16(1), 77–102.
"""
function bootstrap_landscape(λs::AbstractVector{PersistenceLandscape};
                              n_boot :: Int  = 1000,
                              alpha  :: Real = 0.05,
                              stat          = :mean)
    n = length(λs)
    n ≥ 2 || throw(ArgumentError("need at least 2 landscapes"))

    observed_mean = mean_landscape(λs)
    boot_means = Vector{PersistenceLandscape}(undef, n_boot)

    for b in 1:n_boot
        idx = rand(1:n, n)   # resample with replacement
        boot_means[b] = mean_landscape(λs[idx])
    end

    # pointwise quantiles
    K, N = size(observed_mean.layers)
    lower_layers = zeros(K, N)
    upper_layers = zeros(K, N)

    q_lo = alpha / 2
    q_hi = 1 - alpha / 2

    for k in 1:K, j in 1:N
        vals = [bm.layers[k, j] for bm in boot_means]
        lower_layers[k, j] = quantile(vals, q_lo)
        upper_layers[k, j] = quantile(vals, q_hi)
    end

    lower = PersistenceLandscape(lower_layers, observed_mean.tgrid, observed_mean.dim)
    upper = PersistenceLandscape(upper_layers, observed_mean.tgrid, observed_mean.dim)

    return (mean=observed_mean, lower=lower, upper=upper, boot_means=boot_means)
end

"""
    confidence_band(λs::AbstractVector{PersistenceLandscape};
                    n_boot::Int = 1000, alpha::Real = 0.05) -> NamedTuple

Convenience wrapper: returns only `(mean, lower, upper)` from `bootstrap_landscape`.
"""
function confidence_band(λs::AbstractVector{PersistenceLandscape};
                         n_boot::Int = 1000, alpha::Real = 0.05)
    result = bootstrap_landscape(λs; n_boot=n_boot, alpha=alpha)
    return (mean=result.mean, lower=result.lower, upper=result.upper)
end

end # module Bootstrap
