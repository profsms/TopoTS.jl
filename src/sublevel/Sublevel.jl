"""
    Sublevel

Sublevel-set persistent homology for scalar functions and time series.

Unlike the embedding-based approach (Takens → Rips), sublevel-set PH
treats the time series ``x_1, \\ldots, x_N`` directly as a piecewise-linear
function on ``\\{1, \\ldots, N\\}`` and computes the PH of its sublevel sets.

This is:
  - computationally cheaper (O(N log N) vs O(N^{d+1}))
  - free of embedding parameters (no dim, lag to choose)
  - limited to H₀ and H₁ by construction on 1-D functions
  - directly interpretable: features correspond to local extrema

# Reference
Edelsbrunner, H., & Harer, J. (2010). Computational Topology.
AMS. Chapter VII.
"""
module Sublevel

export sublevel_ph, SublevelDiagram

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

"""
    SublevelDiagram

Result of sublevel-set persistent homology on a 1-D function.

# Fields
- `H0 :: Vector{Tuple{Float64,Float64}}` — H₀ birth-death pairs
                                            (merging of connected components
                                             as the level rises)
- `H1 :: Vector{Tuple{Float64,Float64}}` — H₁ birth-death pairs
                                            (local minima creating loops
                                             filled by local maxima)
- `f  :: Vector{Float64}`               — original function values
- `n  :: Int`                           — number of samples
"""
struct SublevelDiagram
    H0 :: Vector{Tuple{Float64,Float64}}
    H1 :: Vector{Tuple{Float64,Float64}}
    f  :: Vector{Float64}
    n  :: Int
end

function Base.show(io::IO, d::SublevelDiagram)
    println(io, "SublevelDiagram (n=$(d.n)):")
    println(io, "  H₀: $(length(d.H0)) pairs")
    print(io,   "  H₁: $(length(d.H1)) pairs")
end

# ─────────────────────────────────────────────────────────────────────────────
# Union-Find
# ─────────────────────────────────────────────────────────────────────────────

mutable struct UF
    parent :: Vector{Int}
    rank   :: Vector{Int}
    birth  :: Vector{Float64}
end

UF(n::Int, f::AbstractVector) =
    UF(collect(1:n), zeros(Int, n), Float64.(f))

function find!(uf::UF, x::Int)
    while uf.parent[x] != x
        uf.parent[x] = uf.parent[uf.parent[x]]
        x = uf.parent[x]
    end
    return x
end

# returns (root kept, root merged, merge_level), elder rule: keep older (lower birth)
function union!(uf::UF, x::Int, y::Int, level::Float64)
    rx, ry = find!(uf, x), find!(uf, y)
    rx == ry && return (rx, -1, level)   # already same component

    # elder rule: the component born earlier (lower value) survives
    older, younger = uf.birth[rx] ≤ uf.birth[ry] ? (rx, ry) : (ry, rx)

    if uf.rank[older] < uf.rank[younger]
        older, younger = younger, older
    end
    uf.parent[younger] = older
    uf.rank[older] == uf.rank[younger] && (uf.rank[older] += 1)
    return (older, younger, level)
end

# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────

"""
    sublevel_ph(x::AbstractVector; extended::Bool = false) -> SublevelDiagram

Compute sublevel-set persistent homology of a 1-D time series / function.

The function ``f : \\{1,\\ldots,N\\} \\to \\mathbb{R}`` is treated as
piecewise-linear on the path graph ``1 - 2 - \\cdots - N``.

**H₀ (connected components):**  
Components are born at local minima and die when two components merge
at a saddle (the older component survives, elder rule). The single
component born at the global minimum is the unpaired generator; its
death time is recorded as `Inf`.

**H₁ (loops):**  
On a path graph there are no topological loops, but the *extended
persistence* pairing (set `extended=true`) produces H₁ pairs that
correspond to local maxima paired with local minima, capturing
oscillations.

# Arguments
- `x`        — scalar time series or sampled function values
- `extended` — compute extended persistence (H₁ pairs from up-down
                filtration); default false

# Returns
A `SublevelDiagram` with fields `H0` and `H1` (pairs of Float64).

# Example
```julia
f   = sin.(range(0, 4π, length=400)) .+ 0.1 .* randn(400)
dgm = sublevel_ph(f)

# H₀: each local minimum paired with the saddle that merges it
dgm.H0   # [(b₁,d₁), (b₂,d₂), …]
# H₁: oscillatory features (with extended=true)
dgm = sublevel_ph(f; extended=true)
```

# Notes
This is equivalent to the 0-th persistence of the Rips filtration
on a path graph, which is computable in O(N log N) via a single
union-find sweep — far cheaper than the full Rips complex for
large N.

# Reference
Edelsbrunner, H., & Harer, J. (2010). *Computational Topology*, AMS.
"""
function sublevel_ph(x::AbstractVector; extended::Bool = false) :: SublevelDiagram
    N = length(x)
    N ≥ 2 || throw(ArgumentError("need at least 2 samples"))
    f = Float64.(x)

    # ── H₀ via union-find on sorted vertices ─────────────────────────────────
    order = sortperm(f)               # process vertices in increasing f order
    uf    = UF(N, f)
    H0    = Tuple{Float64,Float64}[]

    for v in order
        fv = f[v]
        # check left neighbour
        if v > 1 && f[v-1] ≤ fv
            r1, younger, _ = union!(uf, v, v-1, fv)
            younger > 0 && push!(H0, (uf.birth[younger], fv))
        end
        # check right neighbour
        if v < N && f[v+1] ≤ fv
            r2, younger, _ = union!(uf, v, v+1, fv)
            younger > 0 && push!(H0, (uf.birth[younger], fv))
        end
    end

    # the oldest component is unpaired
    root = find!(uf, order[1])
    push!(H0, (uf.birth[root], Inf))

    # filter out zero-persistence pairs
    H0_finite = [(b, d) for (b, d) in H0 if isfinite(d) && d > b]

    # ── H₁ via extended persistence (superlevel union-find) ──────────────────
    H1 = Tuple{Float64,Float64}[]
    if extended
        # superlevel-set sweep: sort descending, pair local maxima with minima
        order_desc = sortperm(f; rev=true)
        uf2 = UF(N, -f)   # negate so union-find still uses "lower birth"

        for v in order_desc
            fv = f[v]
            if v > 1 && f[v-1] ≥ fv
                r, younger, _ = union!(uf2, v, v-1, -fv)
                if younger > 0
                    b_h1 = -uf2.birth[younger]   # birth in superlevel = local max
                    d_h1 = fv                     # death = where it merges
                    b_h1 > d_h1 && push!(H1, (d_h1, b_h1))
                end
            end
            if v < N && f[v+1] ≥ fv
                r, younger, _ = union!(uf2, v, v+1, -fv)
                if younger > 0
                    b_h1 = -uf2.birth[younger]
                    d_h1 = fv
                    b_h1 > d_h1 && push!(H1, (d_h1, b_h1))
                end
            end
        end
    end

    return SublevelDiagram(H0_finite, H1, f, N)
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: windowed sublevel PH
# ─────────────────────────────────────────────────────────────────────────────

"""
    windowed_sublevel_ph(x::AbstractVector;
                         window   :: Int,
                         step     :: Int = window ÷ 4,
                         extended :: Bool = false)
    -> NamedTuple{(:diagrams, :times), Tuple{Vector{SublevelDiagram}, Vector{Float64}}}

Apply `sublevel_ph` to each window of a time series.
Returns the vector of diagrams and the corresponding time axis.

Cheaper than `windowed_ph` (no embedding, O(W log W) per window)
and particularly useful for detecting amplitude/frequency changes
in a univariate series.

# Example
```julia
result = windowed_sublevel_ph(ts; window=200, step=20)
# result.diagrams :: Vector{SublevelDiagram}
# result.times    :: Vector{Float64}
```
"""
function windowed_sublevel_ph(x::AbstractVector;
                               window   :: Int,
                               step     :: Int  = max(1, window ÷ 4),
                               extended :: Bool = false)
    N = length(x)
    positions = collect(1 : step : N - window + 1)
    times     = Float64[p + window / 2 for p in positions]
    diagrams  = SublevelDiagram[]

    for t0 in positions
        seg = x[t0 : t0 + window - 1]
        push!(diagrams, sublevel_ph(seg; extended=extended))
    end

    return (diagrams=diagrams, times=times)
end

end # module Sublevel
