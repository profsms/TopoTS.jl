"""
    PersistenceImages

Persistence image computation (Adams et al., 2017).

Maps a persistence diagram to a fixed-size matrix by convolving
a weighted point cloud in birth–persistence coordinates with a Gaussian kernel.
"""
module PersistenceImages

using ..Filtration: DiagramCollection

export persistence_image, PersistenceImage

"""
    PersistenceImage

A persistence image: a discretised density on the birth–persistence plane.

# Fields
- `pixels   :: Matrix{Float64}` — image array (n_pixels × n_pixels)
- `b_range  :: Tuple{Float64,Float64}` — birth axis range
- `p_range  :: Tuple{Float64,Float64}` — persistence axis range
- `sigma    :: Float64` — Gaussian bandwidth used
- `dim      :: Int`    — homological dimension
"""
struct PersistenceImage
    pixels  :: Matrix{Float64}
    b_range :: Tuple{Float64,Float64}
    p_range :: Tuple{Float64,Float64}
    sigma   :: Float64
    dim     :: Int
end

Base.size(pi::PersistenceImage) = size(pi.pixels)
Base.vec(pi::PersistenceImage) = vec(pi.pixels)

function Base.show(io::IO, img::PersistenceImage)
    n = size(img.pixels, 1)
    print(io, "PersistenceImage(H$(img.dim), $(n)×$(n) pixels, σ=$(img.sigma))")
end

"""
    persistence_image(dgms::DiagramCollection, dim::Int;
                      n_pixels::Int   = 20,
                      sigma::Real     = nothing,
                      weight          = :linear,
                      b_range         = nothing,
                      p_range         = nothing) -> PersistenceImage

Compute the persistence image of the ``k``-th diagram.

# Algorithm
1. Transform to birth–persistence coordinates: ``(b_i, p_i) = (b_i, d_i - b_i)``.
2. Weight each point: ``w_i = w(b_i, p_i)`` (see `weight` argument).
3. Convolve with a 2-D isotropic Gaussian of bandwidth ``\\sigma``.
4. Evaluate on an ``n_\\text{pixels} \\times n_\\text{pixels}`` grid.

# Arguments
- `dgms`      — `DiagramCollection`
- `dim`       — homological dimension
- `n_pixels`  — grid resolution per axis (default 20)
- `sigma`     — Gaussian bandwidth (default: 10% of persistence range)
- `weight`    — `:linear` (``w = p/p_\\max``) or `:constant` (``w = 1``)
- `b_range`   — birth axis limits `(b_min, b_max)`; auto-detected if `nothing`
- `p_range`   — persistence axis limits `(p_min, p_max)`; auto-detected

# Returns
A `PersistenceImage`; call `vec(img)` to get a flat feature vector.

# Example
```julia
img = persistence_image(dgms, 1; n_pixels=20)
feat = vec(img)   # 400-dimensional feature vector for ML
```

# References
Adams, H., et al. (2017). Persistence images: A stable vector representation
of persistent homology. *JMLR*, 18(1), 218–252.
"""
function persistence_image(dgms::DiagramCollection, dim::Int;
                            n_pixels :: Int  = 20,
                            sigma    :: Union{Real,Nothing} = nothing,
                            weight           = :linear,
                            b_range          = nothing,
                            p_range          = nothing)

    1 ≤ dim + 1 ≤ length(dgms.diagrams) || throw(ArgumentError(
        "dim=$dim not computed; available 0…$(dgms.dim_max)"))

    dgm = dgms[dim + 1]

    # finite pairs in birth–persistence coords
    bp_pairs = [(Float64(birth(p)), Float64(death(p)) - Float64(birth(p)))
                for p in dgm if isfinite(death(p))]

    # handle empty diagram
    if isempty(bp_pairs)
        br = isnothing(b_range) ? (0.0, 1.0) : Float64.(b_range)
        pr = isnothing(p_range) ? (0.0, 1.0) : Float64.(p_range)
        return PersistenceImage(zeros(n_pixels, n_pixels), br, pr,
                                isnothing(sigma) ? 0.1 : Float64(sigma), dim)
    end

    bs = [bp[1] for bp in bp_pairs]
    ps = [bp[2] for bp in bp_pairs]

    # auto-range
    br = isnothing(b_range) ? (minimum(bs), maximum(bs) + 1e-8) : Float64.(b_range)
    pr = isnothing(p_range) ? (0.0, maximum(ps) * 1.05 + 1e-8) : Float64.(p_range)

    # auto-sigma: 10% of persistence range
    σ = isnothing(sigma) ? 0.1 * (pr[2] - pr[1]) : Float64(sigma)
    σ > 0 || throw(ArgumentError("sigma must be positive"))

    p_max = maximum(ps)

    # grid centres
    b_grid = range(br[1], br[2]; length=n_pixels)
    p_grid = range(pr[1], pr[2]; length=n_pixels)

    img = zeros(n_pixels, n_pixels)
    inv2s2 = 1.0 / (2σ^2)

    for (b_i, p_i) in bp_pairs
        # weight
        w = if weight == :linear
            p_max > 0 ? p_i / p_max : 1.0
        elseif weight == :constant
            1.0
        else
            throw(ArgumentError("weight must be :linear or :constant"))
        end

        # add Gaussian contribution to each pixel
        for (jb, bg) in enumerate(b_grid)
            db2 = (bg - b_i)^2
            for (jp, pg) in enumerate(p_grid)
                dp2 = (pg - p_i)^2
                img[jp, jb] += w * exp(-(db2 + dp2) * inv2s2)
            end
        end
    end

    # normalise so columns sum to 1 (optional, aids comparability)
    s = sum(img)
    s > 0 && (img ./= s)

    return PersistenceImage(img, br, pr, σ, dim)
end

end # module PersistenceImages
