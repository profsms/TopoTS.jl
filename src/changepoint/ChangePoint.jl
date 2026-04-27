"""
    ChangePoint

Topological change-point detection for time series.

Detects structural changes by measuring how persistence diagrams evolve
across a sliding window. Three complementary scores are provided:

- `bottleneck_score`  — maximum diagram displacement (sensitive to single
                         large topological events)
- `wasserstein_score` — cumulative diagram displacement (sensitive to
                         broad distributional shifts)
- `landscape_score`   — L² distance between mean landscapes (smooth,
                         statistically well-founded)

All scores produce a time-indexed vector that can be thresholded or
passed to a peak-finding routine.

# References
- Gidea, M. (2017). Topological data analysis of critical transitions in
  financial networks. *In: Economic Complexity and Evolution*, 47–59.
- Gidea, M., & Katz, Y. (2018). Topological data analysis of financial
  time series: Landscapes of crashes. *Physica A*, 491, 820–834.
- Chazal, F. et al. (2015). Subsampling methods for persistent homology.
  *ICML*.
"""
module ChangePoint

using Statistics
using ..Windowed:    WindowedDiagrams
using ..Landscapes:  PersistenceLandscape, landscape, landscape_norm
using ..Filtration:  DiagramCollection
using PersistenceDiagrams: birth, death

export changepoint_score, ChangePointResult,
       detect_changepoints, detect_changepoints_windowed, ChangePointEvent,
       bottleneck_score, wasserstein_score, landscape_score,
       andrews_supF

# ─────────────────────────────────────────────────────────────────────────────
# Diagram distances (self-contained; avoids importing Ripserer internals)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _diagram_pairs(dgm) -> Vector{Tuple{Float64,Float64}}

Extract finite off-diagonal (b, d) pairs from a Ripserer diagram.
"""
function _diagram_pairs(dgm)
    [(Float64(birth(p)), Float64(death(p))) for p in dgm if isfinite(death(p))]
end

"""
    _bottleneck(pairs1, pairs2) -> Float64

Bottleneck distance between two diagrams given as vectors of (b,d) pairs.
Uses the standard O(n²) assignment approach (sufficient for typical
diagram sizes encountered in windowed PH of time series).
"""
function _bottleneck(pairs1::Vector{<:Tuple}, pairs2::Vector{<:Tuple})
    # cost of matching a point to the diagonal
    diag_cost(b, d) = (d - b) / 2.0

    # augment with diagonal projections
    pts1 = [(b, d) for (b, d) in pairs1]
    pts2 = [(b, d) for (b, d) in pairs2]

    n1, n2 = length(pts1), length(pts2)

    # degenerate cases
    if n1 == 0 && n2 == 0
        return 0.0
    elseif n1 == 0
        return maximum(diag_cost(b, d) for (b, d) in pts2)
    elseif n2 == 0
        return maximum(diag_cost(b, d) for (b, d) in pts1)
    end

    # ℓ∞ cost between two off-diagonal points
    pt_cost(b1, d1, b2, d2) = max(abs(b1 - b2), abs(d1 - d2))

    # greedy upper bound via min-cost bipartite matching approximation:
    # For the bottleneck, we use the exact formulation:
    # try all candidate thresholds δ (the sorted set of all pairwise costs
    # and diagonal costs) and find the smallest δ under which a perfect
    # matching exists using a greedy check.
    #
    # Sizes: treat unmatched points as matched to diagonal.
    # Total points in augmented problem = n1 + n2 (each side).

    candidates = Float64[]
    for (b1, d1) in pts1
        push!(candidates, diag_cost(b1, d1))
        for (b2, d2) in pts2
            push!(candidates, pt_cost(b1, d1, b2, d2))
        end
    end
    for (b2, d2) in pts2
        push!(candidates, diag_cost(b2, d2))
    end
    sort!(candidates)
    unique!(candidates)

    for δ in candidates
        if _feasible(pts1, pts2, δ)
            return δ
        end
    end
    return maximum(candidates)
end

# Check if a perfect matching exists with cost ≤ δ (augmented Hopcroft-Karp
# would be optimal; we use a simple greedy DFS which is correct for small n).
function _feasible(pts1, pts2, δ::Float64)
    n1, n2 = length(pts1), length(pts2)
    diag_cost(b, d) = (d - b) / 2.0
    pt_cost(b1, d1, b2, d2) = max(abs(b1 - b2), abs(d1 - d2))

    # build adjacency: each pt in pts1 can match each pt in pts2 (if cost≤δ)
    # or match its own diagonal projection (always available if diag_cost≤δ)
    # Same for pts2.
    # Simple greedy: match pts with minimum cost first.
    matched2 = fill(false, n2)
    for (b1, d1) in pts1
        # try to match to a point in pts2
        best_j = -1
        best_c = δ + 1
        for (j, (b2, d2)) in enumerate(pts2)
            !matched2[j] || continue
            c = pt_cost(b1, d1, b2, d2)
            c ≤ δ && c < best_c && (best_c = c; best_j = j)
        end
        if best_j > 0
            matched2[best_j] = true
        else
            # must match to diagonal
            diag_cost(b1, d1) ≤ δ || return false
        end
    end
    # unmatched pts2 must match their diagonal
    for (j, (b2, d2)) in enumerate(pts2)
        matched2[j] && continue
        diag_cost(b2, d2) ≤ δ || return false
    end
    return true
end

"""
    _wasserstein(pairs1, pairs2; p=1) -> Float64

p-Wasserstein distance between two diagrams.
Uses the same augmented matching formulation as `_bottleneck`,
but minimises the sum of costs^p.
"""
function _wasserstein(pairs1::Vector{<:Tuple}, pairs2::Vector{<:Tuple}; p::Real=1)
    diag_cost(b, d) = (d - b) / 2.0
    pt_cost(b1, d1, b2, d2) = max(abs(b1 - b2), abs(d1 - d2))

    n1, n2 = length(pairs1), length(pairs2)
    if n1 == 0 && n2 == 0
        return 0.0
    elseif n1 == 0
        return sum(diag_cost(b, d)^p for (b, d) in pairs2)^(1/p)
    elseif n2 == 0
        return sum(diag_cost(b, d)^p for (b, d) in pairs1)^(1/p)
    end

    # Hungarian-style: build cost matrix (n1 + n2) × (n1 + n2)
    # columns: pts2[1..n2], then diagonal copies for pts1
    # rows:    pts1[1..n1], then diagonal copies for pts2
    N = n1 + n2
    C = fill(Inf, N, N)

    for (i, (b1, d1)) in enumerate(pairs1)
        for (j, (b2, d2)) in enumerate(pairs2)
            C[i, j] = pt_cost(b1, d1, b2, d2)^p
        end
        # match to diagonal: columns n2+1 .. n2+n1
        C[i, n2 + i] = diag_cost(b1, d1)^p
    end
    for (j, (b2, d2)) in enumerate(pairs2)
        # diagonal rows n1+1 .. n1+n2
        C[n1 + j, j] = diag_cost(b2, d2)^p
        # cross-diagonal entries = 0
        for i2 in 1:n1
            C[n1 + j, n2 + i2] = 0.0
        end
    end

    # greedy assignment (exact for p=1 Wasserstein on typical small diagrams)
    assignment = _greedy_assign(C, N)
    total = sum(C[i, assignment[i]] for i in 1:N if !isinf(C[i, assignment[i]]))
    return total^(1/p)
end

function _greedy_assign(C::Matrix{Float64}, N::Int)
    # Candidate columns for each row, sorted by increasing cost.
    candidates = Vector{Vector{Int}}(undef, N)
    for i in 1:N
        js = [j for j in 1:N if isfinite(C[i, j])]
        sort!(js, by = j -> C[i, j])
        isempty(js) && error("No feasible assignment found in _greedy_assign; cost matrix row $i has no finite entries.")
        candidates[i] = js
    end

    # Process the most constrained rows first.
    row_order = sortperm(1:N; by = i -> length(candidates[i]))

    # match_col[j] = row currently assigned to column j, or 0 if free
    match_col = zeros(Int, N)

    function augment(i::Int, seen::Vector{Bool})
        for j in candidates[i]
            seen[j] && continue
            seen[j] = true
            if match_col[j] == 0 || augment(match_col[j], seen)
                match_col[j] = i
                return true
            end
        end
        return false
    end

    for i in row_order
        seen = fill(false, N)
        augment(i, seen) || error("No feasible assignment found in _greedy_assign; could not match row $i.")
    end

    # Convert column->row matching into row->column assignment.
    assignment = zeros(Int, N)
    for j in 1:N
        i = match_col[j]
        i != 0 && (assignment[i] = j)
    end

    any(==(0), assignment) && error("No feasible assignment found in _greedy_assign; incomplete matching.")
    return assignment
end

# ─────────────────────────────────────────────────────────────────────────────
# Score types
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChangePointResult

Time-indexed change-point score vector, with metadata.

# Fields
- `scores     :: Vector{Float64}` — score at each window transition
- `times      :: Vector{Float64}` — time axis (centre of the *later* window)
- `score_type :: Symbol`          — `:bottleneck`, `:wasserstein`, or `:landscape`
- `dim        :: Int`             — homological dimension used
- `threshold  :: Float64`         — detection threshold (if set)
- `peaks      :: Vector{Int}`     — indices of detected change points

Use `detect_changepoints` to populate `peaks` and `threshold`.
"""
struct ChangePointResult
    scores     :: Vector{Float64}
    times      :: Vector{Float64}
    score_type :: Symbol
    dim        :: Int
    threshold  :: Float64
    peaks      :: Vector{Int}
end

function Base.show(io::IO, r::ChangePointResult)
    println(io, "ChangePointResult ($(r.score_type), H$(r.dim)):")
    println(io, "  $(length(r.scores)) scores over t ∈ [$(round(first(r.times),digits=1)), $(round(last(r.times),digits=1))]")
    println(io, "  max score = $(round(maximum(r.scores), sigdigits=4))")
    if !isempty(r.peaks)
        print(io, "  detected CPs at t = $(round.(r.times[r.peaks], digits=1))")
    else
        print(io, "  no change points detected (threshold=$(round(r.threshold,sigdigits=3)))")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Individual scores
# ─────────────────────────────────────────────────────────────────────────────

"""
    bottleneck_score(wd::WindowedDiagrams, dim::Int) -> ChangePointResult

Compute the bottleneck distance between consecutive persistence diagrams.

For window indices ``i`` and ``i+1``:
```
score(i) = d_b(PH_k(W_i),  PH_k(W_{i+1}))
```

The bottleneck distance is sensitive to *individual* large topological
events — the birth or death of a single prominent feature.

# Arguments
- `wd`  — output of `windowed_ph`
- `dim` — homological dimension (0, 1, 2, …)
"""
function bottleneck_score(wd::WindowedDiagrams, dim::Int) :: ChangePointResult
    n = length(wd)
    scores = Vector{Float64}(undef, n - 1)
    for i in 1:n-1
        p1 = _diagram_pairs(wd[i][dim + 1])
        p2 = _diagram_pairs(wd[i+1][dim + 1])
        scores[i] = _bottleneck(p1, p2)
    end
    # time: centre of the later window
    times = wd.times[2:end]
    return ChangePointResult(scores, times, :bottleneck, dim, NaN, Int[])
end

"""
    wasserstein_score(wd::WindowedDiagrams, dim::Int; p::Real = 1) -> ChangePointResult

Compute the p-Wasserstein distance between consecutive persistence diagrams.

```
score(i) = W_p(PH_k(W_i),  PH_k(W_{i+1}))
```

Unlike the bottleneck distance, the Wasserstein distance accumulates
over *all* matched points, making it more sensitive to broad
distributional shifts in the diagram.

# Arguments
- `wd`  — output of `windowed_ph`
- `dim` — homological dimension
- `p`   — Wasserstein exponent (default 1)
"""
function wasserstein_score(wd::WindowedDiagrams, dim::Int;
                           p::Real = 1) :: ChangePointResult
    n = length(wd)
    scores = Vector{Float64}(undef, n - 1)
    for i in 1:n-1
        p1 = _diagram_pairs(wd[i][dim + 1])
        p2 = _diagram_pairs(wd[i+1][dim + 1])
        scores[i] = _wasserstein(p1, p2; p=p)
    end
    times = wd.times[2:end]
    return ChangePointResult(scores, times, :wasserstein, dim, NaN, Int[])
end

"""
    landscape_score(wd::WindowedDiagrams, dim::Int;
                    n_grid    :: Int  = 200,
                    n_layers  :: Int  = 3,
                    norm_p    :: Real = 2) -> ChangePointResult

Compute the L^p distance between consecutive persistence landscapes.

```
score(i) = ‖λ(PH_k(W_i)) - λ(PH_k(W_{i+1}))‖_{L^p}
```

This score is the smoothest of the three: because the landscape map
is 1-Lipschitz and the L² norm averages over the entire scale grid,
it is less affected by isolated outlier features and has the best
statistical properties for hypothesis testing.

# Arguments
- `wd`       — output of `windowed_ph`
- `dim`      — homological dimension
- `n_grid`   — landscape grid resolution (default 200)
- `n_layers` — number of landscape layers to use (default 3)
- `norm_p`   — Lp norm exponent (default 2)

# References
Bubenik, P. (2015). Statistical TDA using persistence landscapes.
*JMLR*, 16(1), 77–102.
"""
function landscape_score(wd::WindowedDiagrams, dim::Int;
                         n_grid   :: Int  = 200,
                         n_layers :: Int  = 3,
                         norm_p   :: Real = 2) :: ChangePointResult
    n = length(wd)

    # compute all landscapes with a shared grid
    # find global scale range first
    all_pairs = [_diagram_pairs(wd[i][dim + 1]) for i in 1:n]
    t_max = maximum(
        (isempty(p) ? 0.0 : maximum(d for (_, d) in p))
        for p in all_pairs; init=0.0)
    tgrid = range(0.0, t_max * 1.05 + 1e-8; length=n_grid)

    λs = [landscape(wd[i], dim; tgrid=tgrid, n_layers=n_layers)
          for i in 1:n]

    scores = Vector{Float64}(undef, n - 1)
    for i in 1:n-1
        diff = λs[i] - λs[i+1]
        scores[i] = landscape_norm(diff, norm_p)
    end
    times = wd.times[2:end]
    return ChangePointResult(scores, times, :landscape, dim, NaN, Int[])
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute all three scores at once
# ─────────────────────────────────────────────────────────────────────────────

"""
    changepoint_score(wd::WindowedDiagrams, dim::Int;
                      method  = :all,
                      p       :: Real = 1,
                      n_grid  :: Int  = 200,
                      n_layers:: Int  = 3) -> Union{ChangePointResult, NamedTuple}

Compute topological change-point score(s) from windowed persistence diagrams.

# Arguments
- `wd`     — output of `windowed_ph`
- `dim`    — homological dimension
- `method` — `:bottleneck`, `:wasserstein`, `:landscape`, or `:all` (default)
- `p`      — Wasserstein exponent (used when `method ∈ (:wasserstein, :all)`)
- others   — passed to the individual score functions

# Returns
- If `method` is a single symbol: a `ChangePointResult`
- If `method == :all`: `NamedTuple` with fields `bottleneck`, `wasserstein`, `landscape`

# Example
```julia
wd     = windowed_ph(ts; window=150, step=10, dim=2, lag=8)
scores = changepoint_score(wd, 1)

# plot bottleneck score
using CairoMakie
lines(scores.bottleneck.times, scores.bottleneck.scores)
```
"""
function changepoint_score(wd::WindowedDiagrams, dim::Int;
                            method   = :all,
                            p        :: Real = 1,
                            n_grid   :: Int  = 200,
                            n_layers :: Int  = 3)
    if method == :bottleneck
        return bottleneck_score(wd, dim)
    elseif method == :wasserstein
        return wasserstein_score(wd, dim; p=p)
    elseif method == :landscape
        return landscape_score(wd, dim; n_grid=n_grid, n_layers=n_layers)
    elseif method == :all
        bn = bottleneck_score(wd, dim)
        ws = wasserstein_score(wd, dim; p=p)
        ls = landscape_score(wd, dim; n_grid=n_grid, n_layers=n_layers)
        return (bottleneck=bn, wasserstein=ws, landscape=ls)
    else
        throw(ArgumentError("method must be :bottleneck, :wasserstein, :landscape, or :all"))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Detection: threshold + peak finding
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChangePointEvent

A detected topological change point.

# Fields
- `time       :: Float64`               — time (window centre) of the event
- `index      :: Int`                   — index into the score vector
- `score      :: Float64`               — score value at the event
- `score_type :: Symbol`                — which score was used
- `dim        :: Int`                   — homological dimension
- `rstar_R    :: Union{Float64,Nothing}` — relative position r*/R in [0,1]
                                           (`:rss` and `:andrews` only)
- `sup_F      :: Union{Float64,Nothing}` — Andrews sup-F statistic
                                           (`:andrews` only)
"""
struct ChangePointEvent
    time       :: Float64
    index      :: Int
    score      :: Float64
    score_type :: Symbol
    dim        :: Int
    rstar_R    :: Union{Float64, Nothing}
    sup_F      :: Union{Float64, Nothing}
end

function Base.show(io::IO, e::ChangePointEvent)
    s = "ChangePoint(t=$(round(e.time,digits=2)), score=$(round(e.score,sigdigits=3)), $(e.score_type), H$(e.dim)"
    !isnothing(e.sup_F)   && (s *= ", sup_F=$(round(e.sup_F,sigdigits=4))")
    !isnothing(e.rstar_R) && (s *= ", r*/R=$(round(e.rstar_R,digits=3))")
    print(io, s * ")")
end

"""
    detect_changepoints(result::ChangePointResult;
                        threshold  :: Union{Real, Symbol} = :mad,
                        min_gap    :: Int = 5,
                        n_mad      :: Real = 3.0) -> Vector{ChangePointEvent}

Detect change points from a `ChangePointResult` score vector.

# Arguments
- `result`    — output of `changepoint_score` (single score)
- `threshold` — detection threshold:
  - `:mad`  (default) — `median + n_mad × MAD` (robust, recommended)
  - `:sigma`          — `mean + n_mad × std`
  - a positive real   — explicit threshold
- `min_gap`   — minimum number of samples between consecutive detections
                 (suppresses double-detections from a single event)
- `n_mad`     — multiplier for the automatic threshold (default 3.0)

# Returns
`Vector{ChangePointEvent}`, sorted by time. An updated `ChangePointResult`
with `peaks` and `threshold` filled is stored in the events.

# Example
```julia
scores = changepoint_score(wd, 1; method=:landscape)
events = detect_changepoints(scores; threshold=:mad, n_mad=3.0)
for ev in events
    println("Change point at t = \$(ev.time)")
end
```
"""
function detect_changepoints(result::ChangePointResult;
                              threshold  :: Union{Real, Symbol} = :mad,
                              min_gap    :: Int  = 5,
                              n_mad      :: Real = 3.0) :: Vector{ChangePointEvent}
    s = result.scores
    # compute threshold
    τ = if threshold == :mad
        med = median(s)
        mad = median(abs.(s .- med))
        med + n_mad * 1.4826 * mad   # 1.4826 makes MAD consistent with σ
    elseif threshold == :sigma
        mean(s) + n_mad * std(s)
    else
        Float64(threshold)
    end

    # find local maxima above threshold with minimum gap
    n = length(s)
    events = ChangePointEvent[]
    last_peak = -min_gap - 1

    # collect all above-threshold peaks, sorted by score (largest first),
    # then enforce min_gap
    candidates = [(s[i], i) for i in 1:n if s[i] ≥ τ]
    sort!(candidates; rev=true)

    selected = Int[]
    for (_, i) in candidates
        all(abs(i - j) ≥ min_gap for j in selected) && push!(selected, i)
    end
    sort!(selected)

    for i in selected
        push!(events, ChangePointEvent(
            result.times[i], i, s[i], result.score_type, result.dim,
            nothing, nothing))
    end

    return events
end

# ─────────────────────────────────────────────────────────────────────────────
# Andrews (1993) critical values — Table 1, 1 restriction, 15% trimming
# ─────────────────────────────────────────────────────────────────────────────

const ANDREWS_CV = Dict(
    0.10 => 7.12,
    0.05 => 8.85,
    0.01 => 12.16,
)

# ─────────────────────────────────────────────────────────────────────────────
# Offline helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _rss_changepoint(score; r0) -> (r_star::Int, best_rss::Float64)

Offline single-breakpoint detection via RSS minimisation.
Returns the index `r_star` that minimises within-segment RSS and the
minimum RSS value. O(n) via prefix sums.
"""
function _rss_changepoint(score::Vector{Float64}; r0::Int = 0)
    R  = length(score)
    r0 = r0 == 0 ? round(Int, 0.15 * R) : r0
    r0 = max(r0, 1)

    S  = cumsum(score)
    S2 = cumsum(score .^ 2)

    best_rss = Inf
    best_r   = div(R, 2)

    for r in r0 : R - r0
        n1  = r;      n2  = R - r
        mu1 = S[r] / n1
        mu2 = (S[R] - S[r]) / n2
        rss = (S2[r] - n1 * mu1^2) + ((S2[R] - S2[r]) - n2 * mu2^2)
        if rss < best_rss
            best_rss = rss
            best_r   = r
        end
    end

    return best_r, best_rss
end

"""
    andrews_supF(score; r0, alpha)
    -> NamedTuple (r_star, sup_F, significant, cv)

Andrews (1993) sup-F test for a single structural break in `score`.

Computes the F-statistic at every candidate breakpoint in the trimmed
interior `[r0, R-r0]` and returns the maximum (sup-F). Compares against
the Andrews (1993) Table 1 critical value for 1 restriction at significance
level `alpha` (supported: 0.10, 0.05, 0.01).

# Reference
Andrews, D.W.K. (1993). Tests for Parameter Instability and Structural
Change with Unknown Change Point. *Econometrica*, 61(4), 821–856.
"""
function andrews_supF(score::Vector{Float64};
                      r0    :: Int     = 0,
                      alpha :: Float64 = 0.05)
    R  = length(score)
    r0 = r0 == 0 ? round(Int, 0.15 * R) : r0
    r0 = max(r0, 1)

    S  = cumsum(score)
    S2 = cumsum(score .^ 2)

    total_mean = S[R] / R
    total_rss  = S2[R] - R * total_mean^2

    best_F = -Inf
    best_r = r0

    for r in r0 : R - r0
        n1 = r;   n2 = R - r
        mu1 = S[r] / n1
        mu2 = (S[R] - S[r]) / n2
        rss_within = (S2[r] - n1 * mu1^2) + ((S2[R] - S2[r]) - n2 * mu2^2)

        # F = (RSS_null − RSS_alt) / (RSS_alt / (R − 2))
        F = (total_rss - rss_within) / (rss_within / max(R - 2, 1))

        if F > best_F
            best_F = F
            best_r = r
        end
    end

    cv          = get(ANDREWS_CV, alpha, ANDREWS_CV[0.05])
    significant = best_F > cv

    return (r_star=best_r, sup_F=best_F, significant=significant, cv=cv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Multi-method detect_changepoints (raw score vector dispatch)
# ─────────────────────────────────────────────────────────────────────────────

"""
    detect_changepoints(score::Vector{Float64};
                        method   :: Symbol  = :cusum_mad,
                        n_mad    :: Float64 = 3.0,
                        n_sigma  :: Float64 = 3.0,
                        k        :: Int     = 10,
                        win      :: Int     = 50,
                        alpha    :: Float64 = 0.05,
                        r0       :: Int     = 0,
                        min_gap  :: Int     = 5,
                        times    :: Union{Vector{Float64}, Nothing} = nothing,
                        score_type :: Symbol = :unknown,
                        dim        :: Int    = -1)
    -> Vector{ChangePointEvent}

Detect change points from a raw score vector using one of six methods.
`times`, `score_type`, and `dim` are metadata attached to returned events;
when `times` is `nothing`, event times default to their index.

| `method`           | Description                                        |
|--------------------|----------------------------------------------------|
| `:cusum_mad`       | Threshold = median + `n_mad` × MAD  (default)     |
| `:cusum_3sigma`    | Threshold = μ + `n_sigma` × σ  (burn-in = `r0`)   |
| `:cusum_sustained` | `:cusum_3sigma` threshold, `k` consecutive hits    |
| `:percentile`      | 99th percentile of burn-in period                  |
| `:cusum_adaptive`  | Sliding-window baseline of width `win`             |
| `:rss`             | Offline RSS minimisation — single best breakpoint  |
| `:andrews`         | Andrews (1993) sup-F — single breakpoint if signif |
"""
function detect_changepoints(score::Vector{Float64};
                              method     :: Symbol  = :cusum_mad,
                              n_mad      :: Float64 = 3.0,
                              n_sigma    :: Float64 = 3.0,
                              k          :: Int     = 10,
                              win        :: Int     = 50,
                              alpha      :: Float64 = 0.05,
                              r0         :: Int     = 0,
                              min_gap    :: Int     = 5,
                              times      :: Union{Vector{Float64}, Nothing} = nothing,
                              score_type :: Symbol  = :unknown,
                              dim        :: Int     = -1)

    n     = length(score)
    times = isnothing(times) ? Float64.(1:n) : times
    _ev(i, rstar_R=nothing, sup_F=nothing) =
        ChangePointEvent(times[i], i, score[i], score_type, dim, rstar_R, sup_F)

    # ── offline methods ───────────────────────────────────────────────────────
    if method == :rss
        r, _ = _rss_changepoint(score; r0=r0)
        return [_ev(r, r / n)]

    elseif method == :andrews
        res = andrews_supF(score; r0=r0, alpha=alpha)
        res.significant || return ChangePointEvent[]
        return [_ev(res.r_star, res.r_star / n, res.sup_F)]
    end

    # ── online methods — compute threshold, then find peaks ──────────────────
    burn = r0 == 0 ? max(2, round(Int, 0.15 * n)) : max(r0, 2)

    τ = if method == :cusum_mad
        med = median(score)
        mad = median(abs.(score .- med))
        med + n_mad * 1.4826 * mad

    elseif method == :cusum_3sigma
        μ = mean(score[1:burn]); σ = std(score[1:burn])
        μ + n_sigma * σ

    elseif method == :cusum_sustained || method == :percentile
        if method == :percentile
            quantile(score[1:min(burn, n)], 0.99)
        else
            μ = mean(score[1:burn]); σ = std(score[1:burn])
            μ + n_sigma * σ
        end

    elseif method == :cusum_adaptive
        # placeholder threshold for first pass — overridden per-point below
        0.0

    else
        throw(ArgumentError("Unknown method :$method. Choose from " *
            ":cusum_mad, :cusum_3sigma, :cusum_sustained, :percentile, " *
            ":cusum_adaptive, :rss, :andrews"))
    end

    # ── sustained-exceedance (M2) ─────────────────────────────────────────────
    if method == :cusum_sustained
        events   = ChangePointEvent[]
        run_len  = 0
        last_cp  = -min_gap - 1
        for i in 1:n
            if score[i] ≥ τ
                run_len += 1
                if run_len == k && i - last_cp ≥ min_gap
                    push!(events, _ev(i - k + 1))
                    last_cp = i
                    run_len = 0
                end
            else
                run_len = 0
            end
        end
        return events
    end

    # ── adaptive sliding-window (M4) ─────────────────────────────────────────
    if method == :cusum_adaptive
        events  = ChangePointEvent[]
        last_cp = -min_gap - 1
        for i in 1:n
            lo   = max(1, i - win)
            hi   = i - 1
            if hi < lo
                continue        # not enough history yet
            end
            seg = score[lo:hi]
            μ_w = mean(seg); σ_w = std(seg)
            τ_i = μ_w + n_sigma * σ_w
            if score[i] ≥ τ_i && i - last_cp ≥ min_gap
                push!(events, _ev(i))
                last_cp = i
            end
        end
        return events
    end

    # ── standard threshold + peak-finding (cusum_mad, cusum_3sigma, percentile)
    candidates = [(score[i], i) for i in 1:n if score[i] ≥ τ]
    sort!(candidates; rev=true)

    selected = Int[]
    for (_, i) in candidates
        all(abs(i - j) ≥ min_gap for j in selected) && push!(selected, i)
    end
    sort!(selected)

    return [_ev(i) for i in selected]
end

# ─────────────────────────────────────────────────────────────────────────────
# Higher-level convenience: score + detect in one call
# ─────────────────────────────────────────────────────────────────────────────

"""
    detect_changepoints_windowed(wd::WindowedDiagrams;
                                 dim      :: Int    = 1,
                                 score    :: Symbol = :landscape,
                                 method   :: Symbol = :andrews,
                                 n_grid   :: Int    = 50,
                                 n_layers :: Int    = 2,
                                 kwargs...)
    -> (events::Vector{ChangePointEvent}, score_vec::Vector{Float64})

Full change-point pipeline in one call: compute a topological score from
`wd`, then detect breakpoints using the chosen `method`.

Returns both the events and the raw score vector so the caller can plot them.

# Arguments
- `dim`      — homological dimension
- `score`    — `:landscape`, `:wasserstein`, or `:bottleneck`
- `method`   — any method accepted by `detect_changepoints`
                (default `:andrews` for single-experiment offline use;
                 use `:cusum_mad` for online / streaming use)
- `n_grid`, `n_layers` — passed to `landscape_score` when `score=:landscape`
- `kwargs`   — forwarded to `detect_changepoints` (e.g. `alpha`, `n_mad`)

# Example
```julia
wd = windowed_ph(ts; window=100, step=10, dim=2, lag=8, dim_max=1)
events, sc = detect_changepoints_windowed(wd; dim=1, method=:andrews)
```
"""
function detect_changepoints_windowed(wd::WindowedDiagrams;
                                       dim      :: Int    = 1,
                                       score    :: Symbol = :landscape,
                                       method   :: Symbol = :andrews,
                                       n_grid   :: Int    = 50,
                                       n_layers :: Int    = 2,
                                       kwargs...)

    cr = if score == :landscape
        landscape_score(wd, dim; n_grid=n_grid, n_layers=n_layers)
    elseif score == :wasserstein
        wasserstein_score(wd, dim)
    elseif score == :bottleneck
        bottleneck_score(wd, dim)
    else
        throw(ArgumentError("score must be :landscape, :wasserstein, or :bottleneck"))
    end

    events = detect_changepoints(cr.scores;
                                  method     = method,
                                  times      = cr.times,
                                  score_type = cr.score_type,
                                  dim        = cr.dim,
                                  kwargs...)

    return (events=events, score_vec=cr.scores)
end

end # module ChangePoint
