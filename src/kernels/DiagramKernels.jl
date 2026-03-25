"""
    DiagramKernels

Kernel functions between persistence diagrams, enabling kernel-based
classification and regression (SVM, kernel PCA, GP regression) on
topological features.

# Implemented kernels
- `persistence_scale_space_kernel` — Reininghaus et al. (2015)
- `persistence_weighted_gaussian_kernel` — Kusano et al. (2016)
- `sliced_wasserstein_kernel` — Carrière et al. (2017)

All kernels are positive semi-definite and stable with respect to
the Wasserstein distance between diagrams.

# Reference
Reininghaus, J., Huber, S., Bauer, U., & Kwitt, R. (2015).
A stable multi-scale kernel for topological machine learning.
*CVPR*, 4741–4748.
"""
module DiagramKernels

export pss_kernel, pwg_kernel, sliced_wasserstein_kernel,
       kernel_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function _pairs(dgm) :: Vector{Tuple{Float64,Float64}}
    [(Float64(birth(p)), Float64(death(p)))
     for p in dgm if isfinite(death(p))]
end

_reflect(b, d) = (d, b)   # reflection across diagonal

# ─────────────────────────────────────────────────────────────────────────────
# Persistence Scale-Space Kernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    pss_kernel(dgm1, dgm2; sigma::Real = 1.0) -> Float64

Persistence Scale-Space (PSS) kernel between two persistence diagrams.

```
k_σ(D₁, D₂) = 1/(8πσ) ∑_{p∈D₁} ∑_{q∈D₂}
    [exp(-‖p-q‖²/(8σ)) - exp(-‖p-q̄‖²/(8σ))]
```
where ``q̄`` is the reflection of ``q`` across the diagonal.

The PSS kernel is positive semi-definite and stable:
``|k_σ(D₁,D₂) - k_σ(D₁',D₂')| ≤ C_σ · W_∞(D₁,D₁')``
for a constant ``C_σ`` depending only on ``\\sigma``.

# Arguments
- `dgm1`, `dgm2` — Ripserer persistence diagrams (elements of a `DiagramCollection`)
- `sigma`        — bandwidth parameter (default 1.0)

# Example
```julia
k = pss_kernel(dgms1[2], dgms2[2]; sigma=0.5)  # H₁ diagrams
```

# Reference
Reininghaus, J., Huber, S., Bauer, U., & Kwitt, R. (2015).
A stable multi-scale kernel for topological machine learning. *CVPR*.
"""
function pss_kernel(dgm1, dgm2; sigma::Real = 1.0)
    σ = Float64(sigma)
    σ > 0 || throw(ArgumentError("sigma must be positive"))

    p1 = _pairs(dgm1)
    p2 = _pairs(dgm2)
    (isempty(p1) || isempty(p2)) && return 0.0

    inv8s = 1.0 / (8σ)
    k = 0.0
    for (b1, d1) in p1
        for (b2, d2) in p2
            # direct term
            k += exp(-((b1-b2)^2 + (d1-d2)^2) * inv8s)
            # reflected term (p2 reflected across diagonal)
            k -= exp(-((b1-d2)^2 + (d1-b2)^2) * inv8s)
        end
    end
    return k / (8π * σ)
end

# ─────────────────────────────────────────────────────────────────────────────
# Persistence Weighted Gaussian Kernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    pwg_kernel(dgm1, dgm2; sigma::Real = 1.0, C::Real = 1.0) -> Float64

Persistence Weighted Gaussian (PWG) kernel.

```
k(D₁, D₂) = ∑_{p∈D₁} ∑_{q∈D₂} w(p) w(q) exp(-‖p-q‖²/(2σ²))
```
where ``w(b, d) = \\tanh(C \\cdot (d-b))`` weights points by their persistence.

# Arguments
- `sigma` — Gaussian bandwidth (default 1.0)
- `C`     — persistence weight steepness (default 1.0; higher → sharper threshold)

# Reference
Kusano, G., Fukumizu, K., & Hiraoka, Y. (2016).
Persistence weighted Gaussian kernel for topological data analysis. *ICML*.
"""
function pwg_kernel(dgm1, dgm2; sigma::Real = 1.0, C::Real = 1.0)
    σ  = Float64(sigma)
    c  = Float64(C)
    σ > 0 || throw(ArgumentError("sigma must be positive"))

    p1 = _pairs(dgm1)
    p2 = _pairs(dgm2)
    (isempty(p1) || isempty(p2)) && return 0.0

    inv2s2 = 1.0 / (2σ^2)
    w(b, d) = tanh(c * (d - b))

    k = 0.0
    for (b1, d1) in p1
        w1 = w(b1, d1)
        for (b2, d2) in p2
            k += w1 * w(b2, d2) * exp(-((b1-b2)^2 + (d1-d2)^2) * inv2s2)
        end
    end
    return k
end

# ─────────────────────────────────────────────────────────────────────────────
# Sliced Wasserstein Kernel
# ─────────────────────────────────────────────────────────────────────────────

"""
    sliced_wasserstein_kernel(dgm1, dgm2;
                              sigma     :: Real = 1.0,
                              n_directions :: Int = 100) -> Float64

Sliced Wasserstein kernel between two persistence diagrams.

Approximates the Wasserstein distance by averaging 1-D Wasserstein
distances along `n_directions` random projections, then exponentiates:
```
k(D₁, D₂) = exp(-SW(D₁, D₂) / (2σ²))
```
where ``SW(D₁, D₂) = \\frac{1}{M} \\sum_{\\theta_i} W_1(\\pi_{\\theta_i}(D₁), \\pi_{\\theta_i}(D₂))``.

This is positive semi-definite and computationally efficient:
O(M · n log n) where n is the diagram size.

# Arguments
- `sigma`        — kernel bandwidth (default 1.0)
- `n_directions` — number of random projection directions (default 100)

# Reference
Carrière, M., Cuturi, M., & Oudot, S. (2017).
Sliced Wasserstein kernel for persistence diagrams. *ICML*.
"""
function sliced_wasserstein_kernel(dgm1, dgm2;
                                    sigma        :: Real = 1.0,
                                    n_directions :: Int  = 100)
    σ = Float64(sigma)
    σ > 0 || throw(ArgumentError("sigma must be positive"))

    sw = _sliced_wasserstein(dgm1, dgm2; n_directions=n_directions)
    return exp(-sw / (2σ^2))
end

function _sliced_wasserstein(dgm1, dgm2; n_directions::Int = 100)
    # augment each diagram with diagonal projections of the other
    p1 = _pairs(dgm1)
    p2 = _pairs(dgm2)

    # augment: add diagonal projections (midpoints on diagonal)
    aug1 = vcat(p1, [((b+d)/2, (b+d)/2) for (b, d) in p2])
    aug2 = vcat(p2, [((b+d)/2, (b+d)/2) for (b, d) in p1])

    n  = length(aug1)
    sw = 0.0

    for _ in 1:n_directions
        θ  = rand() * π
        cθ = cos(θ)
        sθ = sin(θ)
        proj1 = sort([b*cθ + d*sθ for (b, d) in aug1])
        proj2 = sort([b*cθ + d*sθ for (b, d) in aug2])
        sw += sum(abs(proj1[i] - proj2[i]) for i in 1:n) / n
    end

    return sw / n_directions
end

# ─────────────────────────────────────────────────────────────────────────────
# Gram matrix construction
# ─────────────────────────────────────────────────────────────────────────────

"""
    kernel_matrix(dgms::AbstractVector, dim::Int;
                  kernel  = :pss,
                  kwargs...) -> Matrix{Float64}

Compute the ``n \\times n`` Gram matrix for a collection of persistence diagrams.

# Arguments
- `dgms`   — vector of `DiagramCollection` objects
- `dim`    — homological dimension to use
- `kernel` — `:pss` (default), `:pwg`, or `:sliced_wasserstein`
- `kwargs` — passed to the kernel function (e.g. `sigma=0.5`)

# Returns
Symmetric positive semi-definite ``n \\times n`` matrix ``K`` with
``K[i,j] = k(D_i, D_j)``.

# Example
```julia
K = kernel_matrix(all_dgms, 1; kernel=:pss, sigma=0.3)
# Use K with a kernel SVM, kernel PCA, etc.
```
"""
function kernel_matrix(dgms::AbstractVector, dim::Int;
                        kernel = :pss,
                        kwargs...)
    n  = length(dgms)
    K  = zeros(n, n)
    kf = if kernel == :pss
        (a, b) -> pss_kernel(a, b; kwargs...)
    elseif kernel == :pwg
        (a, b) -> pwg_kernel(a, b; kwargs...)
    elseif kernel == :sliced_wasserstein
        (a, b) -> sliced_wasserstein_kernel(a, b; kwargs...)
    else
        throw(ArgumentError("kernel must be :pss, :pwg, or :sliced_wasserstein"))
    end

    for i in 1:n
        K[i, i] = kf(dgms[i][dim+1], dgms[i][dim+1])
        for j in i+1:n
            v = kf(dgms[i][dim+1], dgms[j][dim+1])
            K[i, j] = v
            K[j, i] = v
        end
    end
    return K
end

end # module DiagramKernels
