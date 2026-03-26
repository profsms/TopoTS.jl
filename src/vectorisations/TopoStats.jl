"""
    TopoStats

Scalar summary statistics derived from persistence diagrams.

All statistics are well-defined for finite diagrams and are
stable with respect to the bottleneck or Wasserstein metrics.
"""
module TopoStats

using ..Filtration: DiagramCollection
using PersistenceDiagrams: birth, death

export total_persistence, persistent_entropy, amplitude

"""
    total_persistence(dgms::DiagramCollection, dim::Int; p::Real = 1) -> Float64

Compute the total persistence (``L^p`` amplitude):
```
TotalPers_p(D) = ∑_{(b,d) ∈ D} (d - b)^p
```

For ``p=1`` this is the total lifetime of all homological features.
For ``p=2`` it equals the squared Frobenius norm of the off-diagonal mass.

# Arguments
- `dgms` — `DiagramCollection`
- `dim`  — homological dimension
- `p`    — exponent (default 1)
"""
function total_persistence(dgms::DiagramCollection, dim::Int; p::Real = 1)
    dgm = _get_dim(dgms, dim)
    return sum((Float64(death(pt)) - Float64(birth(pt)))^p
               for pt in dgm if isfinite(death(pt)); init=0.0)
end

"""
    persistent_entropy(dgms::DiagramCollection, dim::Int) -> Float64

Compute the persistent entropy of a diagram:
```
E(D) = -∑_i ℓ_i/L · log(ℓ_i/L),   ℓ_i = d_i - b_i,   L = ∑_i ℓ_i
```

Persistent entropy measures the spread of persistence values.
A diagram dominated by a single long-lived feature has low entropy;
a diagram with many equally-persistent features has high entropy.

# References
Atienza, N., González-Díaz, R., & Soriano-Trigueros, M. (2020).
On the stability of persistent entropy and new summary functions for TDA.
*Entropy*, 22(6), 601.
"""
function persistent_entropy(dgms::DiagramCollection, dim::Int)
    dgm = _get_dim(dgms, dim)
    lifetimes = [Float64(death(pt)) - Float64(birth(pt))
                 for pt in dgm if isfinite(death(pt))]
    isempty(lifetimes) && return 0.0
    L = sum(lifetimes)
    L ≈ 0 && return 0.0
    return -sum(l/L * log(l/L) for l in lifetimes if l > 0)
end

"""
    amplitude(dgms::DiagramCollection, dim::Int; p::Real = Inf) -> Float64

Compute the ``p``-amplitude (distance to the empty diagram):
```
A_p(D) = (∑_{(b,d)} ((d-b)/2)^p)^{1/p}
```
For ``p = Inf`` this equals ``max_i (d_i - b_i)/2``, the half-persistence
of the most prominent feature.

The amplitude is related to the bottleneck distance:
``d_b(D, ∅) = A_∞(D)``.
"""
function amplitude(dgms::DiagramCollection, dim::Int; p::Real = Inf)
    dgm = _get_dim(dgms, dim)
    half_pers = [(Float64(death(pt)) - Float64(birth(pt))) / 2
                 for pt in dgm if isfinite(death(pt))]
    isempty(half_pers) && return 0.0
    if isinf(p)
        return maximum(half_pers)
    else
        return sum(h^p for h in half_pers)^(1/p)
    end
end

# ── internal helper ───────────────────────────────────────────────────────────

function _get_dim(dgms::DiagramCollection, dim::Int)
    1 ≤ dim + 1 ≤ length(dgms.diagrams) || throw(ArgumentError(
        "dim=$dim not in range 0…$(dgms.dim_max)"))
    return dgms[dim + 1]
end

end # module TopoStats
