"""
    Landscapes

Persistence landscape computation, averaging, and norms.

Implements the persistence landscape of Bubenik (2015), which maps each
persistence diagram to a sequence of piecewise-linear functions in L²(ℝ),
enabling mean, variance, and hypothesis-test computation on diagrams.
"""
module Landscapes

using Statistics
using ..Filtration: DiagramCollection
using PersistenceDiagrams: birth, death

export landscape, PersistenceLandscape, mean_landscape, landscape_norm

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    PersistenceLandscape

A discretised persistence landscape.

# Fields
- `layers :: Matrix{Float64}` — `(n_layers × n_grid)` matrix;
                                 `layers[k, :]` is the k-th landscape function
- `tgrid  :: AbstractRange`  — scale grid
- `dim    :: Int`             — homological dimension

# Access
```julia
λ = landscape(dgms, dim=1)
λ.layers[1, :]   # first (dominant) landscape function
λ.layers[2, :]   # second landscape function
```
"""
struct PersistenceLandscape
    layers :: Matrix{Float64}    # (n_layers × n_grid)
    tgrid  :: AbstractRange
    dim    :: Int
end

Base.size(λ::PersistenceLandscape) = size(λ.layers)

function Base.show(io::IO, λ::PersistenceLandscape)
    K, N = size(λ.layers)
    print(io, "PersistenceLandscape(H$(λ.dim), $K layers × $N grid points)")
end

# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    landscape(dgms::DiagramCollection, dim::Int;
              tgrid = nothing, n_grid::Int = 500,
              n_layers::Int = 5) -> PersistenceLandscape

Compute the persistence landscape of the ``k``-th diagram.

For a persistence diagram ``D = \\{(b_i, d_i)\\}``, the tent function is
```
f_{(b,d)}(t) = max(0, min(t-b, d-t))
```
which peaks at height ``(d-b)/2`` at the midpoint ``(b+d)/2``.
The ``k``-th landscape function is the ``k``-th order statistic of
``\\{f_{(b_i,d_i)}(t)\\}_i`` at each ``t``.

# Arguments
- `dgms`     — `DiagramCollection` from `persistent_homology`
- `dim`      — homological dimension
- `tgrid`    — explicit evaluation grid (overrides `n_grid`)
- `n_grid`   — number of grid points (default 500)
- `n_layers` — number of landscape layers to compute (default 5)

# Returns
`PersistenceLandscape` with `layers[k, t]` = ``λ_k(t)``.

# Example
```julia
λ = landscape(dgms, dim=1, n_layers=3)
# layer 1 captures the most persistent feature
plot(λ.tgrid, λ.layers[1, :])
```

# References
Bubenik, P. (2015). Statistical topological data analysis using persistence
landscapes. *Journal of Machine Learning Research*, 16(1), 77–102.
"""
function landscape(dgms::DiagramCollection, dim::Int;
                   tgrid    = nothing,
                   n_grid   :: Int = 500,
                   n_layers :: Int = 5)

    1 ≤ dim + 1 ≤ length(dgms.diagrams) || throw(ArgumentError(
        "dim=$dim not computed; available 0…$(dgms.dim_max)"))

    dgm = dgms[dim + 1]

    # collect finite pairs
    pairs = [(Float64(birth(p)), Float64(death(p))) for p in dgm
             if isfinite(death(p))]

    # build grid
    if isnothing(tgrid)
        if isempty(pairs)
            tgrid = range(0.0, 1.0; length=n_grid)
        else
            t_min = minimum(b for (b, _) in pairs)
            t_max = maximum(d for (_, d) in pairs)
            tgrid = range(t_min, t_max * 1.02; length=n_grid)
        end
    end

    N = length(tgrid)
    layers = zeros(n_layers, N)

    # for each grid point, collect tent values and take order statistics
    for (j, t) in enumerate(tgrid)
        tent_vals = [max(0.0, min(t - b, d - t)) for (b, d) in pairs]
        sort!(tent_vals; rev=true)
        for k in 1:min(n_layers, length(tent_vals))
            layers[k, j] = tent_vals[k]
        end
    end

    return PersistenceLandscape(layers, tgrid, dim)
end

# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic and statistics on landscapes
# ─────────────────────────────────────────────────────────────────────────────

"""
    mean_landscape(λs::AbstractVector{PersistenceLandscape}) -> PersistenceLandscape

Compute the pointwise mean landscape from a collection of landscapes.

All landscapes must share the same grid and number of layers.
This is the natural estimator of the population mean landscape ``E[λ]``,
which is consistent by the law of large numbers in L²(ℝ) (Bubenik, 2015).

# Example
```julia
landscapes = [landscape(persistent_homology(embed(ts_i; dim=3, lag=10)),
                         dim=1) for ts_i in ensemble]
λ_mean = mean_landscape(landscapes)
```
"""
function mean_landscape(λs::AbstractVector{PersistenceLandscape})
    isempty(λs) && throw(ArgumentError("need at least one landscape"))
    ref = first(λs)
    all(size(λ.layers) == size(ref.layers) for λ in λs) || throw(DimensionMismatch(
        "all landscapes must have the same size"))
    all(λ.dim == ref.dim for λ in λs) || throw(ArgumentError(
        "all landscapes must have the same homological dimension"))

    mean_layers = mean(λ.layers for λ in λs)
    return PersistenceLandscape(mean_layers, ref.tgrid, ref.dim)
end

"""
    landscape_norm(λ::PersistenceLandscape, p::Real = 2) -> Float64

Compute the ``L^p`` norm of the landscape, summed over layers:
```
‖λ‖_p = (∑_k ∫ |λ_k(t)|^p dt)^{1/p}
```
approximated via the trapezoidal rule on the stored grid.

# Arguments
- `λ` — a `PersistenceLandscape`
- `p` — norm exponent (default 2; use `Inf` for sup-norm)
"""
function landscape_norm(λ::PersistenceLandscape, p::Real = 2)
    dt = step(λ.tgrid)
    if isinf(p)
        return maximum(abs, λ.layers)
    end
    total = 0.0
    for k in axes(λ.layers, 1)
        total += sum(abs(λ.layers[k, j])^p for j in axes(λ.layers, 2)) * dt
    end
    return total^(1/p)
end

# allow pointwise arithmetic for bootstrap / test statistics
Base.:+(a::PersistenceLandscape, b::PersistenceLandscape) =
    PersistenceLandscape(a.layers .+ b.layers, a.tgrid, a.dim)
Base.:-(a::PersistenceLandscape, b::PersistenceLandscape) =
    PersistenceLandscape(a.layers .- b.layers, a.tgrid, a.dim)
Base.:*(c::Real, λ::PersistenceLandscape) =
    PersistenceLandscape(c .* λ.layers, λ.tgrid, λ.dim)
Base.:/(λ::PersistenceLandscape, c::Real) =
    PersistenceLandscape(λ.layers ./ c, λ.tgrid, λ.dim)

end # module Landscapes
