# =============================================================================
# TopoTS.jl — Empirical Illustrations
#
# Three datasets:
#   1. Lorenz attractor      — Takens embedding + PH + feature extraction
#   2. EEG-like signal       — change-point detection + CROCKER plot
#   3. Financial time series — regime classification with topo features
#
# Plots are saved to plots/ if CairoMakie is available.
# =============================================================================

using TopoTS
using Statistics, StatsBase, LinearAlgebra, Random

Random.seed!(42)

# Try to load CairoMakie for plots (optional)
const PLOT = try
    @eval using CairoMakie
    isdir("plots") || mkdir("plots")
    true
catch
    @info "CairoMakie not available — skipping plot output"
    false
end

# =============================================================================
# 1. LORENZ ATTRACTOR
# =============================================================================
println("=" ^ 60)
println("1. LORENZ ATTRACTOR")
println("=" ^ 60)

function integrate_lorenz(; σ=10.0, ρ=28.0, β=8/3, dt=0.01, T=50.0)
    u = [1.0, 0.0, 0.0]
    N = round(Int, T / dt)
    xs = zeros(N)
    for i in 1:N
        k1 = [σ*(u[2]-u[1]),  u[1]*(ρ-u[3])-u[2],  u[1]*u[2]-β*u[3]]
        u2 = u .+ 0.5dt .* k1
        k2 = [σ*(u2[2]-u2[1]), u2[1]*(ρ-u2[3])-u2[2], u2[1]*u2[2]-β*u2[3]]
        u2 = u .+ 0.5dt .* k2
        k3 = [σ*(u2[2]-u2[1]), u2[1]*(ρ-u2[3])-u2[2], u2[1]*u2[2]-β*u2[3]]
        u2 = u .+ dt .* k3
        k4 = [σ*(u2[2]-u2[1]), u2[1]*(ρ-u2[3])-u2[2], u2[1]*u2[2]-β*u2[3]]
        u .+= (dt/6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        xs[i] = u[1]
    end
    return xs[1000:end]   # discard transient
end

lorenz_x = integrate_lorenz()
println("Lorenz x-component: $(length(lorenz_x)) samples")

lag = ami_lag(lorenz_x)
dim = fnn_dim(lorenz_x; lag=lag)
println("Optimal lag: $lag,  optimal dim: $dim")

emb = embed(lorenz_x; lag=lag, dim=dim)
println("Embedded: $(size(emb.points)) point cloud")

# Subsample for PH (full cloud is too large for Rips)
max_pts = 500
npts    = size(emb.points, 1)
idx     = round.(Int, range(1, npts, length=min(max_pts, npts)))
pts_sub = emb.points[idx, :]
println("Subsampled: $(size(pts_sub)) points")

dgms = persistent_homology(pts_sub; dim_max=1, filtration=:rips)
println(dgms)

h1   = dgms[2]
pers = [d[2] - d[1] for d in h1 if isfinite(d[2])]
if !isempty(pers)
    println("H1 top-5 persistence: ", round.(sort(pers, rev=true)[1:min(5,length(pers))], digits=4))
end

spec  = TopoFeatureSpec(dim_max=1)
feats = topo_features(dgms; spec=spec)
println("Feature vector ($(length(feats)) dims): ", round.(feats, digits=3))

# Betti curve
bc = betti_curve(dgms, 1; n_grid=200)

# Landscape
λ = landscape(dgms, 1; n_layers=3, n_grid=200)
println("Landscape norm (λ₁): ", round(landscape_norm(λ, 1), digits=4))

if PLOT
    fig1a = plot_diagram(dgms, 1;
        title      = "Lorenz attractor — H₁ persistence diagram",
        figsize = (520, 520))
    save("plots/lorenz_pd_H1.pdf", fig1a)

    fig1b = plot_diagram_multi(dgms;
        title      = "Lorenz attractor",
        figsize = (900, 450))
    save("plots/lorenz_pd_multi.pdf", fig1b)

    fig1c = plot_barcode(dgms, 1;
        title      = "Lorenz attractor — H₁ barcode",
        figsize = (650, 400))
    save("plots/lorenz_barcode_H1.pdf", fig1c)

    fig1d = plot_landscape(λ;
        title      = "Lorenz attractor — H₁ landscape",
        figsize = (700, 320))
    save("plots/lorenz_landscape_H1.pdf", fig1d)

    fig1e = plot_betti_curve(bc;
        title      = "Lorenz attractor — H₁ Betti curve",
        figsize = (700, 300))
    save("plots/lorenz_betti_H1.pdf", fig1e)

    println("Saved Lorenz plots to plots/")
end

# =============================================================================
# 2. EEG-LIKE SIGNAL — CHANGE-POINT DETECTION + CROCKER
# =============================================================================
println()
println("=" ^ 60)
println("2. EEG-LIKE SIGNAL — CHANGE-POINT DETECTION")
println("=" ^ 60)

N_eeg = 2000
t     = range(0, 4π, length=N_eeg)
state1 = sin.(10 .* t) .+ 0.5 .* sin.(20 .* t) .+ 0.2 .* randn(N_eeg)
state2 = 2.0 .* sin.(2 .* t) .+ 0.8 .* randn(N_eeg)
eeg    = vcat(state1[1:N_eeg÷2], state2[N_eeg÷2+1:end])
println("EEG signal: $(length(eeg)) samples, true change-point at $(N_eeg÷2)")

wd_eeg = windowed_ph(eeg; window=100, step=50, dim=3, lag=3, dim_max=1)
println("Windows computed: $(length(wd_eeg))")

# Change-point detection
ls_score = landscape_score(wd_eeg, 1; n_grid=50, n_layers=2)
events   = detect_changepoints(ls_score; threshold=:mad, n_mad=2.0)
println("Detected $(length(events)) change-point(s):")
for ev in events
    println("  window $(ev.index)  ≈ sample $(ev.index * 50)  (score=$(round(ev.score, digits=3)))")
end

# CROCKER plot
println("\nComputing CROCKER plots (H₀ and H₁)...")
cp0 = crocker(wd_eeg; dim=0, n_scale=40)
cp1 = crocker(wd_eeg; dim=1, n_scale=40)
println("CROCKER H₀ shape: $(size(cp0.surface))  (scale × time)")
println("CROCKER H₁ shape: $(size(cp1.surface))  (scale × time)")

if PLOT
    # Detected change-point times for vlines
    vl = isempty(events) ? Float64[] : [Float64(ev.index) for ev in events]

    fig2a = plot_changepoint_score(ls_score;
        events      = events,
        threshold   = median(ls_score.scores) + 2*mad(ls_score.scores),
        title       = "EEG — landscape change-point score",
        figsize = (800, 300))
    save("plots/eeg_changepoint_score.pdf", fig2a)

    fig2b = plot_crocker(cp1;
        title       = "EEG — CROCKER plot H₁",
        vlines      = vl,
        figsize = (800, 380))
    save("plots/eeg_crocker_H1.pdf", fig2b)

    fig2c = plot_crocker_multi([cp0, cp1];
        titles      = ["H₀ — connected components", "H₁ — loops"],
        vlines      = vl,
        figsize = (800, 650))
    save("plots/eeg_crocker_multi.pdf", fig2c)

    println("Saved EEG plots to plots/")
end

# =============================================================================
# 3. FINANCIAL TIME SERIES — REGIME CLASSIFICATION
# =============================================================================
println()
println("=" ^ 60)
println("3. FINANCIAL TIME SERIES — REGIME CLASSIFICATION")
println("=" ^ 60)

function gbm(n; μ=0.001, σ=0.02)
    cumsum(μ .+ σ .* randn(n))
end

function ou(n; θ=0.1, μ=0.0, σ=0.02)
    x = zeros(n)
    for i in 2:n
        x[i] = x[i-1] + θ*(μ - x[i-1]) + σ*randn()
    end
    x
end

N_fin   = 300
n_train = 20
n_test  = 10

println("Generating $n_train trending + $n_train mean-reverting training series...")

function series_features(ts)
    lag  = max(1, ami_lag(ts))
    d    = max(2, min(fnn_dim(ts; lag=lag), 4))
    emb  = embed(ts; lag=lag, dim=d)
    npts = size(emb.points, 1)
    idx  = round.(Int, range(1, npts, length=min(200, npts)))
    dgm  = persistent_homology(emb.points[idx, :]; dim_max=1, filtration=:rips)
    spec = TopoFeatureSpec(dim_max=1)
    return topo_features(dgm; spec=spec)
end

X_trend = [series_features(gbm(N_fin))  for _ in 1:n_train]
X_mr    = [series_features(ou(N_fin))   for _ in 1:n_train]

F_trend = hcat(X_trend...)
F_mr    = hcat(X_mr...)

println("Feature matrix (trending):     $(size(F_trend))")
println("Feature matrix (mean-reverting): $(size(F_mr))")

c_trend = vec(mean(F_trend, dims=2))
c_mr    = vec(mean(F_mr,    dims=2))

classify(x) = norm(x .- c_trend) < norm(x .- c_mr) ? :trending : :mean_reverting

trend_acc = mean(classify(series_features(gbm(N_fin))) == :trending    for _ in 1:n_test)
mr_acc    = mean(classify(series_features(ou(N_fin)))  == :mean_reverting for _ in 1:n_test)
acc       = (trend_acc + mr_acc) / 2

println("Trending accuracy:       $(round(100trend_acc, digits=1))%")
println("Mean-reverting accuracy: $(round(100mr_acc,    digits=1))%")
println("Overall accuracy:        $(round(100acc,       digits=1))%")

if PLOT && acc > 0
    # Visualise one example diagram from each class
    dgm_trend = persistent_homology(
        begin
            ts = gbm(N_fin); lag = max(1, ami_lag(ts)); d = max(2, min(fnn_dim(ts; lag=lag), 4))
            emb = embed(ts; lag=lag, dim=d); emb.points[1:min(200,end), :]
        end; dim_max=1)
    dgm_mr = persistent_homology(
        begin
            ts = ou(N_fin); lag = max(1, ami_lag(ts)); d = max(2, min(fnn_dim(ts; lag=lag), 4))
            emb = embed(ts; lag=lag, dim=d); emb.points[1:min(200,end), :]
        end; dim_max=1)

    fig3a = plot_diagram(dgm_trend, 1;
        title      = "Trending (GBM) — H₁ diagram",
        figsize = (480, 480))
    save("plots/finance_pd_trending.pdf", fig3a)

    fig3b = plot_diagram(dgm_mr, 1;
        title      = "Mean-reverting (OU) — H₁ diagram",
        figsize = (480, 480))
    save("plots/finance_pd_mr.pdf", fig3b)

    println("Saved finance plots to plots/")
end

println()
println("=" ^ 60)
println("All empirical illustrations completed.")
PLOT && println("Plots saved to plots/")
println("=" ^ 60)
