"""
    Features

Flat feature vector extraction from persistence diagrams for
machine-learning pipelines.

`topo_features` runs the full pipeline (embed → PH → vectorise)
and returns a single `Float64` vector combining:
  - persistence landscape values (layers 1–K, on grid)
  - Betti curve values
  - scalar summary statistics (total persistence, entropy, amplitude)

The output is compatible with any tabular ML framework (MLJ.jl,
scikit-learn via PythonCall, etc.).
"""
module Features

using ..Embedding:         TakensEmbedding, embed
using ..Filtration:        DiagramCollection, persistent_homology
using ..Landscapes:        landscape, landscape_norm
using ..BettiCurves:       betti_curve
using ..PersistenceImages: persistence_image
using ..TopoStats:         total_persistence, persistent_entropy, amplitude

export topo_features, TopoFeatureSpec, feature_names

# ─────────────────────────────────────────────────────────────────────────────
# Feature specification
# ─────────────────────────────────────────────────────────────────────────────

"""
    TopoFeatureSpec

Configuration object controlling which features are extracted.

# Fields (all have sensible defaults)
- `dim_max      :: Int`   — maximum PH dimension (default 1)
- `dim          :: Int`   — embedding dimension (default 2)
- `lag          :: Int`   — embedding lag (default 1)
- `filtration   :: Symbol` — `:rips` or `:alpha` (default `:rips`)
- `threshold    :: Float64` — filtration threshold (default `Inf`)
# Landscape parameters
- `use_landscape :: Bool`  — include landscape features (default true)
- `n_landscape_layers :: Int` — (default 3)
- `n_landscape_grid   :: Int` — (default 50)
# Betti curve parameters
- `use_betti     :: Bool`  — include Betti curve features (default true)
- `n_betti_grid  :: Int`   — (default 50)
# Summary statistics
- `use_stats     :: Bool`  — include scalar statistics (default true)
# Persistence image parameters
- `use_image     :: Bool`  — include persistence image features (default false)
- `n_image_pixels :: Int`  — (default 10)

# Example
```julia
spec = TopoFeatureSpec(dim=3, lag=10, n_landscape_grid=100)
feat = topo_features(ts; spec=spec)
```
"""
Base.@kwdef struct TopoFeatureSpec
    dim_max      :: Int     = 1
    dim          :: Int     = 2
    lag          :: Int     = 1
    filtration   :: Symbol  = :rips
    threshold    :: Float64 = Inf
    # landscape
    use_landscape          :: Bool = true
    n_landscape_layers     :: Int  = 3
    n_landscape_grid       :: Int  = 50
    # betti curve
    use_betti              :: Bool = true
    n_betti_grid           :: Int  = 50
    # scalar stats
    use_stats              :: Bool = true
    # persistence image
    use_image              :: Bool = false
    n_image_pixels         :: Int  = 10
end

# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

"""
    topo_features(x::AbstractVector;
                  spec :: TopoFeatureSpec = TopoFeatureSpec(),
                  tgrid_landscape = nothing,
                  tgrid_betti     = nothing) -> Vector{Float64}

Extract a flat topological feature vector from a time series.

Runs the full pipeline:
1. Takens embedding with parameters from `spec`
2. Persistent homology up to `spec.dim_max`
3. For each homological dimension 0…dim_max:
   - Persistence landscape (if `spec.use_landscape`)
   - Betti curve (if `spec.use_betti`)
   - Summary statistics: total persistence, entropy, amplitude
     (if `spec.use_stats`)
   - Persistence image (if `spec.use_image`)
4. Concatenate all features into a single `Float64` vector.

# Arguments
- `x`    — scalar time series
- `spec` — `TopoFeatureSpec` controlling which features are extracted
- `tgrid_landscape` — shared landscape grid (for consistency across
   multiple series); auto-computed if `nothing`
- `tgrid_betti` — shared Betti curve grid

# Returns
`Vector{Float64}` of length determined by `spec`.
Use `feature_names(spec)` to get matching feature labels.

# Example
```julia
spec = TopoFeatureSpec(dim=3, lag=10, dim_max=1)

# Single series
feat = topo_features(ts; spec=spec)

# Feature matrix for a collection
X = reduce(hcat, topo_features(ts_i; spec=spec) for ts_i in ensemble)'
# X is (n_series × n_features)
```
"""
function topo_features(x::AbstractVector;
                        spec            :: TopoFeatureSpec = TopoFeatureSpec(),
                        tgrid_landscape = nothing,
                        tgrid_betti     = nothing) :: Vector{Float64}

    emb  = embed(x; dim=spec.dim, lag=spec.lag)
    dgms = persistent_homology(emb;
                               dim_max    = spec.dim_max,
                               filtration = spec.filtration,
                               threshold  = spec.threshold)
    return _extract(dgms, spec;
                    tgrid_landscape=tgrid_landscape,
                    tgrid_betti=tgrid_betti)
end

"""
    topo_features(dgms::DiagramCollection;
                  spec :: TopoFeatureSpec = TopoFeatureSpec(),
                  tgrid_landscape = nothing,
                  tgrid_betti     = nothing) -> Vector{Float64}

Extract features from an already-computed `DiagramCollection`.
Use when you want to separate the PH computation from vectorisation.
"""
function topo_features(dgms::DiagramCollection;
                        spec            :: TopoFeatureSpec = TopoFeatureSpec(),
                        tgrid_landscape = nothing,
                        tgrid_betti     = nothing) :: Vector{Float64}
    return _extract(dgms, spec;
                    tgrid_landscape=tgrid_landscape,
                    tgrid_betti=tgrid_betti)
end

function _extract(dgms::DiagramCollection, spec::TopoFeatureSpec;
                  tgrid_landscape, tgrid_betti) :: Vector{Float64}
    feats = Float64[]

    for k in 0:spec.dim_max
        if spec.use_landscape
            λ = landscape(dgms, k;
                          tgrid    = tgrid_landscape,
                          n_grid   = spec.n_landscape_grid,
                          n_layers = spec.n_landscape_layers)
            append!(feats, vec(λ.layers))
        end

        if spec.use_betti
            bc = betti_curve(dgms, k;
                             tgrid  = tgrid_betti,
                             n_grid = spec.n_betti_grid)
            append!(feats, Float64.(bc.values))
        end

        if spec.use_stats
            push!(feats, total_persistence(dgms, k; p=1))
            push!(feats, total_persistence(dgms, k; p=2))
            push!(feats, persistent_entropy(dgms, k))
            push!(feats, amplitude(dgms, k; p=Inf))
            push!(feats, amplitude(dgms, k; p=2))
        end

        if spec.use_image
            img = persistence_image(dgms, k; n_pixels=spec.n_image_pixels)
            append!(feats, vec(img.pixels))
        end
    end

    return feats
end

# ─────────────────────────────────────────────────────────────────────────────
# Feature names
# ─────────────────────────────────────────────────────────────────────────────

"""
    feature_names(spec::TopoFeatureSpec) -> Vector{String}

Return a vector of descriptive feature names matching the output of
`topo_features` for a given `TopoFeatureSpec`.

Useful for constructing data frames or reporting feature importances.

# Example
```julia
spec  = TopoFeatureSpec(dim_max=1, n_landscape_grid=10, n_betti_grid=5)
names = feature_names(spec)
feat  = topo_features(ts; spec=spec)
Dict(zip(names, feat))
```
"""
function feature_names(spec::TopoFeatureSpec) :: Vector{String}
    names = String[]
    for k in 0:spec.dim_max
        if spec.use_landscape
            for layer in 1:spec.n_landscape_layers
                for t in 1:spec.n_landscape_grid
                    push!(names, "landscape_H$(k)_L$(layer)_t$(t)")
                end
            end
        end
        if spec.use_betti
            for t in 1:spec.n_betti_grid
                push!(names, "betti_H$(k)_t$(t)")
            end
        end
        if spec.use_stats
            push!(names, "total_pers_H$(k)_p1")
            push!(names, "total_pers_H$(k)_p2")
            push!(names, "entropy_H$(k)")
            push!(names, "amplitude_H$(k)_inf")
            push!(names, "amplitude_H$(k)_p2")
        end
        if spec.use_image
            n = spec.n_image_pixels
            for i in 1:n, j in 1:n
                push!(names, "image_H$(k)_$(i)_$(j)")
            end
        end
    end
    return names
end

end # module Features
