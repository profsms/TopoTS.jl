"""
    BettiCurves

Betti curve (persistent Betti number function) computation and utilities.
"""
module BettiCurves

using ..Filtration: DiagramCollection

export betti_curve, BettiCurve

"""
    BettiCurve{T}

A Betti curve for a fixed homological dimension and scale grid.

# Fields
- `values :: Vector{Int}`  — β_k(ε) at each grid point
- `tgrid  :: AbstractRange` — scale grid ε₁ < ε₂ < … < εₙ
- `dim    :: Int`            — homological dimension k

# Example
```julia
bc = betti_curve(dgms, dim=1, tgrid=range(0, 3, length=300))
plot(bc.tgrid, bc.values)
```
"""
struct BettiCurve
    values :: Vector{Int}
    tgrid  :: AbstractRange
    dim    :: Int
end

Base.length(bc::BettiCurve) = length(bc.values)

function Base.show(io::IO, bc::BettiCurve)
    print(io, "BettiCurve(H$(bc.dim), $(length(bc.tgrid)) points, ",
          "ε ∈ [$(round(first(bc.tgrid),digits=3)), ",
          "$(round(last(bc.tgrid),digits=3))])")
end

"""
    betti_curve(dgms::DiagramCollection, dim::Int;
                tgrid = nothing, n_grid::Int = 500) -> BettiCurve

Compute the ``k``-th Betti curve from a `DiagramCollection`.

The Betti curve is defined as
```
β_k(ε) = #{(b,d) ∈ PH_k : b ≤ ε ≤ d}
```
the number of ``k``-dimensional homological features alive at scale ``ε``.

# Arguments
- `dgms`   — output of `persistent_homology`
- `dim`    — homological dimension (0, 1, 2, …)
- `tgrid`  — explicit scale grid (overrides `n_grid`)
- `n_grid` — number of grid points if `tgrid` is not given (default 500)

# Returns
A `BettiCurve` with `values[i]` = β_k(tgrid[i]).

# Example
```julia
bc0 = betti_curve(dgms, 0)   # connected components
bc1 = betti_curve(dgms, 1)   # loops
```

# Notes
Points with infinite death time (the single infinite H₀ class in a connected
space) are treated as dying at `last(tgrid)`.
"""
function betti_curve(dgms::DiagramCollection, dim::Int;
                     tgrid = nothing,
                     n_grid::Int = 500)

    1 ≤ dim + 1 ≤ length(dgms.diagrams) || throw(ArgumentError(
        "dim=$dim not computed; available dims 0…$(dgms.dim_max)"))

    dgm = dgms[dim + 1]   # 1-indexed storage

    # collect finite birth-death pairs
    pairs = [(Float64(birth(p)), Float64(death(p))) for p in dgm
             if isfinite(death(p))]

    # auto-grid if not provided
    if isnothing(tgrid)
        if isempty(pairs)
            tgrid = range(0.0, 1.0; length=n_grid)
        else
            t_max = maximum(d for (_, d) in pairs)
            tgrid = range(0.0, t_max * 1.05; length=n_grid)
        end
    end

    vals = [count(b ≤ t ≤ d for (b, d) in pairs) for t in tgrid]
    return BettiCurve(vals, tgrid, dim)
end

end # module BettiCurves
