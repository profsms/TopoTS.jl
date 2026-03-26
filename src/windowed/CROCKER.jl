"""
    CROCKER

CROCKER (Contour Realization Of Computed k-dimensional hole Evolution in
the Rips complex) plots for time series.

A CROCKER plot is a heatmap of the Betti number β_k(ε, t) as a function
of both filtration scale ε and window time t. It provides a global visual
summary of how the topology of a time series evolves across scales and time.

# Reference
Topaz, C. M., Ziegelmeier, L., & Halverson, T. (2015).
Topological data analysis of biological aggregation models.
*PLOS ONE*, 10(5), e0126383.
"""
module CROCKER

using ..Windowed: WindowedDiagrams
using PersistenceDiagrams: birth, death

export crocker, CROCKERPlot

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    CROCKERPlot

A CROCKER plot: the Betti number surface β_k(ε, t).

# Fields
- `surface   :: Matrix{Int}`     — `(n_scale × n_time)` matrix;
                                    `surface[i,j]` = β_k(ε_i, t_j)
- `scales    :: AbstractRange`   — filtration scale axis ε
- `times     :: Vector{Float64}` — time axis (window centres)
- `dim       :: Int`             — homological dimension k
- `window    :: Int`             — window length used
- `step      :: Int`             — window step used

# Access
```julia
cp = crocker(wd, dim=1)
cp.surface    # (n_scale × n_time) Int matrix  — rows=scale, cols=time
cp.scales     # scale grid
cp.times      # time grid (window centres)
```
"""
struct CROCKERPlot
    surface :: Matrix{Int}
    scales  :: AbstractRange
    times   :: Vector{Float64}
    dim     :: Int
    window  :: Int
    step    :: Int
end

function Base.show(io::IO, c::CROCKERPlot)
    ns, nt = size(c.surface)
    println(io, "CROCKERPlot (H$(c.dim)):")
    println(io, "  scale axis : $ns points ∈ [$(round(first(c.scales),digits=3)), $(round(last(c.scales),digits=3))]")
    print(io,   "  time  axis : $nt windows, t ∈ [$(round(first(c.times),digits=1)), $(round(last(c.times),digits=1))]")
end

# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    crocker(wd::WindowedDiagrams;
            dim     :: Int = 1,
            n_scale :: Int = 100,
            scales        = nothing) -> CROCKERPlot

Compute the CROCKER plot from a `WindowedDiagrams` object.

The CROCKER surface at position ``(i, j)`` is
```
β_k(ε_i, t_j) = #{(b, d) ∈ PH_k(W_j) : b ≤ ε_i ≤ d}
```
i.e., the number of k-dimensional features alive at scale ε_i
in window W_j.

# Arguments
- `wd`      — output of `windowed_ph`
- `dim`     — homological dimension (default 1, loops)
- `n_scale` — number of scale grid points (default 100)
- `scales`  — explicit scale grid (overrides `n_scale`)

# Returns
A `CROCKERPlot` with `surface[i, j]` = β_k(ε_i, t_j).

# Example
```julia
wd = windowed_ph(ts; window=150, step=10, dim=2, lag=8)
cp = crocker(wd; dim=1, n_scale=100)

# heatmap with CairoMakie
using CairoMakie
heatmap(cp.times, collect(cp.scales), cp.surface';
        colormap=:viridis,
        axis=(xlabel="time", ylabel="scale ε", title="CROCKER plot H₁"))
```

# Notes
Columns of `cp.surface` correspond to windows (time); rows correspond
to scale values. When transposing for plotting with (time, scale) axes,
use `cp.surface'`.

# Reference
Topaz, C. M., Ziegelmeier, L., & Halverson, T. (2015).
Topological data analysis of biological aggregation models.
*PLOS ONE*, 10(5), e0126383.
"""
function crocker(wd::WindowedDiagrams;
                 dim     :: Int = 1,
                 n_scale :: Int = 100,
                 scales         = nothing) :: CROCKERPlot

    n_win = length(wd)
    n_win ≥ 1 || throw(ArgumentError("WindowedDiagrams is empty"))
    0 ≤ dim ≤ wd.dim_max || throw(ArgumentError(
        "dim=$dim not in 0…$(wd.dim_max)"))

    # collect all finite birth-death pairs across all windows to build scale grid
    all_pairs = [
        [(Float64(birth(p)), Float64(death(p)))
         for p in wd[j][dim + 1] if isfinite(death(p))]
        for j in 1:n_win
    ]

    if isnothing(scales)
        t_max = maximum(
            (isempty(ps) ? 0.0 : maximum(d for (_, d) in ps))
            for ps in all_pairs; init=0.0)
        scales = range(0.0, t_max * 1.02 + 1e-8; length=n_scale)
    end

    n_scale_actual = length(scales)
    surface = zeros(Int, n_scale_actual, n_win)

    for j in 1:n_win
        pairs_j = all_pairs[j]
        isempty(pairs_j) && continue
        for (i, ε) in enumerate(scales)
            surface[i, j] = count(b ≤ ε ≤ d for (b, d) in pairs_j)
        end
    end

    return CROCKERPlot(surface, scales, wd.times, dim, wd.window, wd.step)
end

end # module CROCKER
