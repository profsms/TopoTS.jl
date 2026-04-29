"""
    HypothesisTests

Two-sample hypothesis tests for persistence diagrams, implemented via
landscape statistics. Tests whether two collections of time series have
the same topological structure.
"""
module HypothesisTests

using Statistics
using SpecialFunctions: erfc
using ..Landscapes: PersistenceLandscape, mean_landscape, landscape_norm

export permutation_test, landscape_ttest

"""
    permutation_test(λs1::AbstractVector{PersistenceLandscape},
                     λs2::AbstractVector{PersistenceLandscape};
                     n_perm   :: Int = 999,
                     stat            = :l2_mean_diff) -> NamedTuple

Permutation test for equality of mean landscapes between two groups.

Tests ``H_0: E[\\lambda_1] = E[\\lambda_2]`` against the two-sided alternative.

The test statistic is the ``L^2`` distance between the two sample mean landscapes.
Under ``H_0`` the group labels are exchangeable, so the null distribution
is approximated by randomly permuting labels `n_perm` times.

# Arguments
- `λs1`, `λs2` — landscapes from two groups (need not be the same size)
- `n_perm`     — number of permutations (default 999)
- `stat`       — test statistic: `:l2_mean_diff` (default) or `:l1_mean_diff`

# Returns
`NamedTuple` with:
- `pvalue`      — permutation p-value
- `statistic`   — observed test statistic value
- `null_dist`   — vector of permuted statistic values

# Example
```julia
# Do periodic and noisy time series have different H₁ topology?
result = permutation_test(λs_periodic, λs_noise; n_perm=999)
result.pvalue   # e.g. 0.001 → reject H₀
```
"""
function permutation_test(λs1::AbstractVector{PersistenceLandscape},
                           λs2::AbstractVector{PersistenceLandscape};
                           n_perm :: Int = 999,
                           stat          = :l2_mean_diff)

    p_norm = stat == :l2_mean_diff ? 2 :
             stat == :l1_mean_diff ? 1 :
             throw(ArgumentError("stat must be :l2_mean_diff or :l1_mean_diff"))

    n1, n2 = length(λs1), length(λs2)
    all_λs = vcat(λs1, λs2)
    N      = n1 + n2

    _stat(a, b) = landscape_norm(mean_landscape(a) - mean_landscape(b), p_norm)

    observed = _stat(λs1, λs2)
    null_dist = Vector{Float64}(undef, n_perm)

    for i in 1:n_perm
        perm = randperm(N)
        null_dist[i] = _stat(all_λs[perm[1:n1]], all_λs[perm[n1+1:N]])
    end

    pvalue = (sum(null_dist .≥ observed) + 1) / (n_perm + 1)
    return (pvalue=pvalue, statistic=observed, null_dist=null_dist)
end

"""
    landscape_ttest(λs1::AbstractVector{PersistenceLandscape},
                    λs2::AbstractVector{PersistenceLandscape};
                    layer::Int = 1) -> NamedTuple

Pointwise two-sample ``t``-test on a single landscape layer.

For each grid point ``t``, tests whether the mean of ``\\lambda_k(t)`` differs
between the two groups, using Welch's unequal-variance ``t``-test.
Returns the vector of ``p``-values (raw, not corrected for multiplicity).

# Arguments
- `λs1`, `λs2` — landscape collections
- `layer`      — which landscape layer to test (default 1, the dominant layer)

# Returns
`NamedTuple` with:
- `pvalues`  — vector of pointwise p-values (length = n_grid)
- `tstats`   — vector of t-statistics
- `tgrid`    — scale grid shared by the landscapes

# Notes
For a global test, prefer `permutation_test` which controls the family-wise
error rate. This function is useful for identifying *where* on the scale
axis the two groups differ.
"""
function landscape_ttest(λs1::AbstractVector{PersistenceLandscape},
                          λs2::AbstractVector{PersistenceLandscape};
                          layer :: Int = 1)

    n1, n2 = length(λs1), length(λs2)
    n1 ≥ 2 && n2 ≥ 2 || throw(ArgumentError("each group needs ≥ 2 landscapes"))

    ref = first(λs1)
    N = size(ref.layers, 2)

    # extract layer values: (n_obs × n_grid)
    vals1 = reduce(vcat, transpose(λ.layers[layer, :]) for λ in λs1)
    vals2 = reduce(vcat, transpose(λ.layers[layer, :]) for λ in λs2)

    pvals = Vector{Float64}(undef, N)
    tstats = Vector{Float64}(undef, N)

    for j in 1:N
        x1, x2 = vals1[:, j], vals2[:, j]
        m1, m2 = mean(x1), mean(x2)
        s1, s2 = var(x1), var(x2)
        se = sqrt(s1/n1 + s2/n2)
        if se ≈ 0
            tstats[j] = 0.0
            pvals[j]  = 1.0
        else
            t = (m1 - m2) / se
            # Welch–Satterthwaite degrees of freedom
            df = (s1/n1 + s2/n2)^2 / ((s1/n1)^2/(n1-1) + (s2/n2)^2/(n2-1))
            # two-sided p-value via normal approximation (valid for large df)
            pvals[j]  = 2 * (1 - _normal_cdf(abs(t)))
            tstats[j] = t
        end
    end

    return (pvalues=pvals, tstats=tstats, tgrid=ref.tgrid)
end

# ── utilities ─────────────────────────────────────────────────────────────────

function randperm(n::Int)
    v = collect(1:n)
    for i in n:-1:2
        j = rand(1:i)
        v[i], v[j] = v[j], v[i]
    end
    return v
end

# normal CDF via erfc
function _normal_cdf(x::Real)
    return 0.5 * erfc(-x / sqrt(2))
end

end # module HypothesisTests
