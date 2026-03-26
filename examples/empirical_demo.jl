# =============================================================================
# TopoTS.jl — Empirical Illustrations
# Covers: Lorenz attractor, EEG (synthetic), Financial time series
# Analyses: Takens embedding + PH, CROCKER plots, change-point detection,
#           classification with topological features
# =============================================================================

using TopoTS
using Statistics, StatsBase, LinearAlgebra, Random

Random.seed!(42)

# =============================================================================
# 1. SYNTHETIC: Lorenz attractor
# =============================================================================

function integrate_lorenz(; σ=10.0, ρ=28.0, β=8/3, dt=0.01, T=50.0)
    u = [1.0, 0.0, 0.0]
    N = round(Int, T / dt)
    xs = zeros(N)

    for i in 1:N
        k1 = [σ * (u[2] - u[1]),
              u[1] * (ρ - u[3]) - u[2],
              u[1] * u[2] - β * u[3]]

        u2 = u .+ 0.5 * dt .* k1
        k2 = [σ * (u2[2] - u2[1]),
              u2[1] * (ρ - u2[3]) - u2[2],
              u2[1] * u2[2] - β * u2[3]]

        u2 = u .+ 0.5 * dt .* k2
        k3 = [σ * (u2[2] - u2[1]),
              u2[1] * (ρ - u2[3]) - u2[2],
              u2[1] * u2[2] - β * u2[3]]

        u2 = u .+ dt .* k3
        k4 = [σ * (u2[2] - u2[1]),
              u2[1] * (ρ - u2[3]) - u2[2],
              u2[1] * u2[2] - β * u2[3]]

        u .+= (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        xs[i] = u[1]
    end

    return xs[1000:end]
end

println("=" ^ 60)
println("1. LORENZ ATTRACTOR")
println("=" ^ 60)

lorenz_x = integrate_lorenz()
println("Lorenz x-component: $(length(lorenz_x)) samples")

lag = ami_lag(lorenz_x)
dim = fnn_dim(lorenz_x; lag=lag)
println("Optimal lag: $lag, optimal dim: $dim")

emb = embed(lorenz_x; lag=lag, dim=dim)
println("Embedded: $(size(emb.points)) point cloud")

max_pts = 500
npts = size(emb.points, 1)
idx = round.(Int, range(1, npts, length=min(max_pts, npts)))
pts_sub = emb.points[idx, :]

println("Subsampled for PH: $(size(pts_sub)) point cloud")

dgms = persistent_homology(pts_sub; dim_max=1, filtration=:rips)
println(dgms)

h1 = length(dgms) >= 2 ? dgms[2] : []
pers = [d[2] - d[1] for d in h1 if isfinite(d[2])]

if !isempty(pers)
    println("H1 persistence values (top 5): ", sort(pers, rev=true)[1:min(5, length(pers))])
else
    println("No prominent H1 classes detected.")
end

spec = TopoFeatureSpec(dim_max=1)
feats = topo_features(dgms; spec=spec)
println("Feature vector length: $(length(feats))")
println("Features: ", round.(feats, digits=3))

# =============================================================================
# 2. EEG-LIKE SIGNAL: two-state oscillator (awake vs sleep analogue)
# =============================================================================

println()
println("=" ^ 60)
println("2. EEG-LIKE SIGNAL — CHANGE-POINT DETECTION")
println("=" ^ 60)

N_eeg = 2000
t = range(0, 4π, length=N_eeg)

state1 = sin.(10 .* t) .+ 0.5 .* sin.(20 .* t) .+ 0.2 .* randn(N_eeg)
state2 = 2.0 .* sin.(2 .* t) .+ 0.8 .* randn(N_eeg)

eeg = vcat(state1[1:N_eeg ÷ 2], state2[N_eeg ÷ 2 + 1:end])
println("EEG signal: $(length(eeg)) samples, change point at $(N_eeg ÷ 2)")

wd_eeg = windowed_ph(
    eeg;
    window  = 100,
    step    = 50,
    dim     = 3,
    lag     = 3,
    dim_max = 1,
)

cp_result = changepoint_score(wd_eeg, 1; method=:bottleneck)

events = detect_changepoints(cp_result; threshold=:mad, n_mad=2.0)
println("Detected $(length(events)) change point(s):")
for ev in events
    println("  t ≈ $(ev.index * 50) (score=$(round(ev.score, digits=3)))")
end

println("\nComputing CROCKER plot...")
crk = crocker(wd_eeg; dim=1, n_scale=15)
println("CROCKER shape: $(size(crk.surface))  (scale × time)")

# =============================================================================
# 3. FINANCIAL TIME SERIES — CLASSIFICATION
# =============================================================================

println()
println("=" ^ 60)
println("3. FINANCIAL TIME SERIES — REGIME CLASSIFICATION")
println("=" ^ 60)

function gbm(n; μ=0.001, σ=0.02)
    r = μ .+ σ .* randn(n)
    return cumsum(r)
end

function ou(n; θ=0.1, μ=0.0, σ=0.02)
    x = zeros(n)
    for i in 2:n
        x[i] = x[i - 1] + θ * (μ - x[i - 1]) + σ * randn()
    end
    return x
end

N_fin = 300
n_train = 20
n_test  = 10

println("Generating $n_train trending + $n_train mean-reverting series...")

function series_features(ts)
    lag = max(1, ami_lag(ts))
    d   = max(2, min(fnn_dim(ts; lag=lag), 4))
    emb = embed(ts; lag=lag, dim=d)

    npts = size(emb.points, 1)
    max_pts = 200
    idx = round.(Int, range(1, npts, length=min(max_pts, npts)))
    pts_sub = emb.points[idx, :]

    dgm = persistent_homology(pts_sub; dim_max=1, filtration=:rips)
    spec = TopoFeatureSpec(dim_max=1)
    return topo_features(dgm; spec=spec)
end

X_trend = [series_features(gbm(N_fin)) for _ in 1:n_train]
X_mr    = [series_features(ou(N_fin))  for _ in 1:n_train]

F_trend = hcat(X_trend...)
F_mr    = hcat(X_mr...)

println("Feature matrix size (trending): $(size(F_trend))")
println("Feature matrix size (mean-reverting): $(size(F_mr))")

c_trend = vec(mean(F_trend, dims=2))
c_mr    = vec(mean(F_mr, dims=2))

function classify(x, c_trend, c_mr)
    return norm(x .- c_trend) < norm(x .- c_mr) ? :trending : :mean_reverting
end

trend_correct = sum(classify(series_features(gbm(N_fin)), c_trend, c_mr) == :trending for _ in 1:n_test)
mr_correct    = sum(classify(series_features(ou(N_fin)),  c_trend, c_mr) == :mean_reverting for _ in 1:n_test)

correct = trend_correct + mr_correct
acc = correct / (2 * n_test)

println("Nearest-centroid accuracy on $(2 * n_test) test series: $(round(100 * acc, digits=1))%")

println()
println("All empirical illustrations completed successfully.")