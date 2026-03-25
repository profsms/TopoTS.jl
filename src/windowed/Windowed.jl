"""
    Windowed

Sliding-window persistent homology for non-stationary time series.

This is the core computational engine for change-point detection and
CROCKER plots. For each window position, a Takens embedding is built
and persistent homology is computed, producing a time-indexed sequence
of persistence diagrams.

# Main function
- `windowed_ph`   — compute a diagram at each window position
- `WindowedDiagrams` — the result type (iterable, indexable)
"""
module Windowed

using ..Embedding: TakensEmbedding, embed
using ..Filtration: DiagramCollection, persistent_homology

export windowed_ph, WindowedDiagrams

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    WindowedDiagrams

Result of sliding-window persistent homology computation.

Each element `wd[i]` is the `DiagramCollection` computed from the
sub-series starting at sample `wd.positions[i]`.

# Fields
- `diagrams   :: Vector{DiagramCollection}` — one per window
- `positions  :: Vector{Int}`               — start index of each window
- `times      :: Vector{Float64}`           — centre time of each window
                                               (in sample units)
- `window     :: Int`   — window length in samples
- `step       :: Int`   — step size between windows
- `dim        :: Int`   — Takens embedding dimension
- `lag        :: Int`   — Takens embedding lag
- `dim_max    :: Int`   — maximum homological dimension
- `n_orig     :: Int`   — original time series length

# Iteration
```julia
for dgms in wd
    # dgms :: DiagramCollection
end
```
"""
struct WindowedDiagrams
    diagrams  :: Vector{DiagramCollection}
    positions :: Vector{Int}
    times     :: Vector{Float64}
    window    :: Int
    step      :: Int
    dim       :: Int
    lag       :: Int
    dim_max   :: Int
    n_orig    :: Int
end

Base.length(wd::WindowedDiagrams)          = length(wd.diagrams)
Base.getindex(wd::WindowedDiagrams, i)     = wd.diagrams[i]
Base.iterate(wd::WindowedDiagrams, s=1)    = s > length(wd) ? nothing : (wd.diagrams[s], s+1)
Base.eachindex(wd::WindowedDiagrams)       = eachindex(wd.diagrams)

function Base.show(io::IO, wd::WindowedDiagrams)
    println(io, "WindowedDiagrams:")
    println(io, "  windows  : $(length(wd)) × $(wd.window) samples (step=$(wd.step))")
    println(io, "  embedding: dim=$(wd.dim), lag=$(wd.lag)")
    print(io,   "  H₀…H$(wd.dim_max) computed per window")
end

# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

"""
    windowed_ph(x::AbstractVector;
                window    :: Int,
                step      :: Int        = window ÷ 4,
                dim       :: Int        = 2,
                lag       :: Int        = 1,
                dim_max   :: Int        = 1,
                filtration              = :rips,
                threshold :: Real       = Inf,
                verbose   :: Bool       = false) -> WindowedDiagrams

Compute persistent homology on a sliding window over a time series.

For each window ``x[t : t + W - 1]``, constructs the Takens delay
embedding and computes persistent homology up to dimension `dim_max`.
Returns a `WindowedDiagrams` container with one `DiagramCollection`
per window position.

# Arguments
- `x`          — scalar time series (length N)
- `window`     — window length W in samples; must satisfy
                  ``W > (\\texttt{dim} - 1) \\cdot \\texttt{lag}``
- `step`       — step size between consecutive windows (default W÷4)
- `dim`        — Takens embedding dimension (default 2)
- `lag`        — Takens embedding lag (default 1)
- `dim_max`    — maximum homological dimension to compute (default 1)
- `filtration` — `:rips` (default) or `:alpha`
- `threshold`  — maximum filtration value (default `Inf`)
- `verbose`    — print progress (default false)

# Returns
A `WindowedDiagrams` object. Access individual diagrams via indexing
or iteration; the `.times` field gives the centre of each window.

# Example
```julia
ts = vcat(sin.(range(0, 20π, 500)), randn(500))   # periodic then noise

wd = windowed_ph(ts;
    window = 150,
    step   = 10,
    dim    = 2,
    lag    = 8,
    dim_max = 1)

length(wd)    # number of windows
wd[1]         # DiagramCollection for first window
wd.times      # centre sample of each window
```

# Notes
The minimum window size is ``(\\texttt{dim} - 1) \\cdot \\texttt{lag} + 2``.
If the window is too small for the chosen embedding parameters, an
`ArgumentError` is thrown.

The `times` field stores the 1-based index of the window **centre**,
which is natural for plotting change-point scores against the original
time axis.

# References
Perea, J. A., & Harer, J. (2015). Sliding windows and persistence:
An application of topological methods to signal analysis.
*Foundations of Computational Mathematics*, 15(3), 799–838.
"""
function windowed_ph(x::AbstractVector{T};
                     window    :: Int,
                     step      :: Int        = max(1, window ÷ 4),
                     dim       :: Int        = 2,
                     lag       :: Int        = 1,
                     dim_max   :: Int        = 1,
                     filtration              = :rips,
                     threshold :: Real       = Inf,
                     verbose   :: Bool       = false) where T <: Real

    N = length(x)
    min_win = (dim - 1) * lag + 2
    window ≥ min_win || throw(ArgumentError(
        "window=$window too small for dim=$dim, lag=$lag; need ≥ $min_win"))
    step ≥ 1 || throw(ArgumentError("step must be ≥ 1"))
    window ≤ N || throw(ArgumentError("window=$window exceeds series length N=$N"))

    # collect window start positions
    positions = collect(1 : step : N - window + 1)
    n_windows = length(positions)

    diagrams = Vector{DiagramCollection}(undef, n_windows)
    times    = Float64[p + window / 2 for p in positions]

    for (i, t0) in enumerate(positions)
        verbose && print("\rwindowed_ph: window $i / $n_windows")
        seg = x[t0 : t0 + window - 1]
        emb = embed(seg; dim=dim, lag=lag)
        diagrams[i] = persistent_homology(emb;
                                          dim_max    = dim_max,
                                          filtration = filtration,
                                          threshold  = threshold)
    end
    verbose && println()

    return WindowedDiagrams(diagrams, positions, times,
                            window, step, dim, lag, dim_max, N)
end

end # module Windowed
