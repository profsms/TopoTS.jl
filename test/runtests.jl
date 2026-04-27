using Test
using Random
using TopoTS
using PersistenceDiagrams: birth, death

try
    @eval using CechCore_jll
catch
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function circle_series(n; noise=0.0)
    t = range(0, 2π, length=n + 1)[1:n]
    x = sin.(t)
    noise > 0 && (x .+= noise .* sin.(17 .* t .+ 1.3))
    return x
end

function piecewise_series(n; cp=nothing)
    cp = isnothing(cp) ? n ÷ 2 : cp
    t1 = range(0, 10π, length=cp)
    t2 = range(0, 10π, length=n - cp) .+ 5π
    return vcat(sin.(t1), sin.(t2) .* 0.3)
end

# ─────────────────────────────────────────────────────────────────────────────
# Core package tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "TopoTS.jl" begin

    @testset "Embedding" begin
        ts  = circle_series(500)
        emb = embed(ts; dim=3, lag=10)
        @test size(emb.points) == (480, 3)
        @test_throws ArgumentError embed(ts; dim=0, lag=1)
        @test 1 ≤ optimal_lag(ts; max_lag=30) ≤ 30
    end

    @testset "Filtration baseline" begin
        emb  = embed(circle_series(200; noise=0.05); dim=2, lag=12)
        dgms = persistent_homology(emb; dim_max=1)
        @test length(dgms) == 2
        @test dgms.filtration == :rips
    end

    @testset "Vectorisations" begin
        dgms = persistent_homology(
            embed(circle_series(300; noise=0.03); dim=2, lag=12);
            dim_max=1
        )

        bc = betti_curve(dgms, 1; n_grid=100)
        @test length(bc) == 100
        @test all(bc.values .≥ 0)

        λ = landscape(dgms, 1; n_layers=3, n_grid=100)
        @test size(λ.layers) == (3, 100)

        img = persistence_image(dgms, 1; n_pixels=10)
        @test length(vec(img)) == 100
    end

    @testset "TopoStats" begin
        dgms = persistent_homology(
            embed(circle_series(300; noise=0.03); dim=2, lag=12);
            dim_max=1
        )
        @test total_persistence(dgms, 1) ≥ 0
        @test persistent_entropy(dgms, 1) ≥ 0
        @test amplitude(dgms, 1) ≥ 0
    end

    @testset "Windowed PH" begin
        ts = circle_series(600; noise=0.05)
        wd = windowed_ph(ts; window=100, step=20, dim=2, lag=8, dim_max=1)
        @test wd[1] isa DiagramCollection
        @test length(wd.times) == length(wd)
        @test_throws ArgumentError windowed_ph(ts; window=2, step=1, dim=3, lag=5)
    end

    @testset "CROCKER" begin
        ts = circle_series(400; noise=0.05)
        wd = windowed_ph(ts; window=80, step=20, dim=2, lag=6, dim_max=1)
        cp = crocker(wd; dim=1, n_scale=50)
        @test size(cp.surface) == (50, length(wd))
        @test all(cp.surface .≥ 0)
    end

    @testset "ChangePoint scores" begin
        ts = piecewise_series(600)
        wd = windowed_ph(ts; window=100, step=15, dim=2, lag=6, dim_max=1)

        bn = bottleneck_score(wd, 1)
        @test bn.score_type == :bottleneck
        @test all(bn.scores .≥ 0)

        ws = wasserstein_score(wd, 1)
        @test length(ws.scores) == length(bn.scores)

        ls = landscape_score(wd, 1; n_grid=50, n_layers=2)
        @test length(ls.scores) == length(bn.scores)

        all_s = changepoint_score(wd, 1)
        @test haskey(all_s, :bottleneck)
        @test haskey(all_s, :wasserstein)
        @test haskey(all_s, :landscape)
    end

    @testset "detect_changepoints" begin
        ts = piecewise_series(600; cp=300)
        wd = windowed_ph(ts; window=100, step=15, dim=2, lag=6, dim_max=1)
        ls = landscape_score(wd, 1; n_grid=50, n_layers=2)
        events = detect_changepoints(ls; threshold=:mad, n_mad=2.0)
        @test events isa Vector{ChangePointEvent}
        @test all(e -> e isa ChangePointEvent, events)
        if length(events) > 1
            idx = [e.index for e in events]
            @test all(diff(idx) .≥ 5)
        end
    end

    @testset "Sublevel PH" begin
        f   = vcat(sin.(range(0, 4π, length=200)), cos.(range(0, 4π, length=200)))
        dgm = sublevel_ph(f)
        @test !isempty(dgm.H0)
        @test all(b < d for (b, d) in dgm.H0)

        dgm_ext = sublevel_ph(f; extended=true)
        @test !isempty(dgm_ext.H1)

        result = windowed_sublevel_ph(f; window=80, step=20)
        @test length(result.diagrams) == length(result.times)
    end

    @testset "Multivariate embedding" begin
        N = 500
        X = hcat(sin.(range(0, 6π, length=N)), cos.(range(0, 6π, length=N)))
        emb = embed_multivariate(X; dim=2, lag=5)
        @test emb.n_channels == 2
        @test size(emb.points, 2) == 4
        dgms = persistent_homology(emb; dim_max=1)
        @test dgms isa DiagramCollection
    end

    @testset "Diagram kernels" begin
        dgms = persistent_homology(
            embed(circle_series(300; noise=0.03); dim=2, lag=12);
            dim_max=1
        )
        dgm = dgms[2]
        @test pss_kernel(dgm, dgm; sigma=0.5) ≥ 0
        @test pwg_kernel(dgm, dgm; sigma=0.5) ≥ 0
        @test sliced_wasserstein_kernel(dgm, dgm; sigma=0.5, n_directions=20) ≈ 1.0 atol=1e-4
        K = kernel_matrix([dgms], 1; kernel=:pss, sigma=0.5)
        @test size(K) == (1, 1)
    end

    @testset "Feature extraction" begin
        ts = circle_series(300; noise=0.03)
        spec = TopoFeatureSpec(
            dim=2, lag=10, dim_max=1,
            n_landscape_grid=10, n_landscape_layers=2,
            n_betti_grid=5, use_image=false
        )
        feat  = topo_features(ts; spec=spec)
        names = feature_names(spec)
        @test length(feat) == length(names)
        @test !any(isnan, feat)

        emb  = embed(ts; dim=2, lag=10)
        dgms = persistent_homology(emb; dim_max=1)
        @test topo_features(dgms; spec=spec) ≈ feat
    end

    # ── Visualisation smoke tests (no rendering, just object creation) ────────
    @testset "Visualisations (no backend)" begin
        dgms = persistent_homology(
            embed(circle_series(200; noise=0.03); dim=2, lag=10);
            dim_max=1
        )
        ts = circle_series(400; noise=0.05)
        wd = windowed_ph(ts; window=80, step=20, dim=2, lag=6, dim_max=1)
        cp = crocker(wd; dim=1, n_scale=30)

        # Without a Makie backend loaded these should throw a clear error
        @test_throws ErrorException plot_diagram(dgms, 1)
        @test_throws ErrorException plot_barcode(dgms, 1)
        @test_throws ErrorException plot_crocker(cp)

        ls = landscape_score(wd, 1; n_grid=30, n_layers=2)
        events = detect_changepoints(ls; threshold=:mad, n_mad=2.0)
        @test_throws ErrorException plot_changepoint_score(ls; events=events)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Filtration variety / geometry sanity checks
# ─────────────────────────────────────────────────────────────────────────────

@testset "Filtration types" begin
    ts  = circle_series(300; noise=0.03)
    emb = embed(ts; dim=2, lag=12)

    dgms_rips = persistent_homology(emb; dim_max=1)
    @test dgms_rips.filtration == :rips

    dgms_alpha = persistent_homology(emb; dim_max=1, filtration=:alpha)
    @test dgms_alpha.filtration == :alpha
    @test length(dgms_alpha) == 2

    dgms_ec = persistent_homology(emb; dim_max=1, filtration=:edge_collapsed)
    @test dgms_ec.filtration == :edge_collapsed
    @test length(dgms_ec[1]) == length(dgms_rips[1])

    dgms_cub = persistent_homology(emb; dim_max=1, filtration=:cubical)
    @test dgms_cub.filtration == :cubical
    @test !isempty(dgms_cub[1])

    @test_throws ArgumentError persistent_homology(emb; filtration=:unknown)
end

@testset "Small point-cloud geometry" begin
    pts2 = [0.0 0.0; 1.0 0.0]
    pts3 = [0.0 0.0; 1.0 0.0; 0.5 0.866]

    ph2_r = persistent_homology(pts2; filtration=:rips, dim_max=1)
    @test ph2_r.filtration == :rips
    @test length(ph2_r[1]) == 2

    ph3_r = persistent_homology(pts3; filtration=:rips, dim_max=1)
    ph3_a = persistent_homology(pts3; filtration=:alpha, dim_max=1)

    @test length(ph3_r[1]) == 3
    @test length(ph3_r[2]) == 0
    @test length(ph3_a[1]) == 3
    @test length(ph3_a[2]) == 1
end

@testset "Duplicate points" begin
    dup  = [0.0 0.0; 0.0 0.0; 1.0 0.0]
    ph_r = persistent_homology(dup; filtration=:rips, dim_max=1)
    @test ph_r isa DiagramCollection
end

@testset "Square loop geometry" begin
    sq   = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]
    ph_r = persistent_homology(sq; filtration=:rips, dim_max=1)
    ph_a = persistent_homology(sq; filtration=:alpha, dim_max=1)
    @test length(ph_r[2]) ≥ 1
    @test length(ph_a[2]) ≥ 1
end

# ─────────────────────────────────────────────────────────────────────────────
# Čech filtration tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "Čech filtration (libcech / extension)" begin
    if !TopoTS.cech_available()
        @warn "CechCore_jll/libcech not available — skipping Čech tests."
        @test_skip true
    else
        libpath = TopoTS.cech_lib_path()
        @test isfile(libpath)

        pts1 = [0.0 0.0]
        ph1  = persistent_homology(pts1; filtration=:cech, dim_max=1)
        @test ph1.filtration == :cech
        @test ph1.n_points == 1
        @test length(ph1) == 1
        @test length(ph1[1]) == 1

        pts2 = [0.0 0.0; 1.0 0.0]
        ph2  = persistent_homology(pts2; filtration=:cech, dim_max=1)
        @test ph2.filtration == :cech
        @test length(ph2[1]) == 2

        pts3    = [0.0 0.0; 1.0 0.0; 0.5 0.866]
        ph3_c   = persistent_homology(pts3; filtration=:cech,  dim_max=1)
        ph3_r   = persistent_homology(pts3; filtration=:rips,  dim_max=1)
        ph3_a   = persistent_homology(pts3; filtration=:alpha, dim_max=1)

        @test length(ph3_c[1]) == 3
        @test length(ph3_c[2]) == 1
        @test length(ph3_r[1]) == 3
        @test length(ph3_r[2]) == 0
        @test length(ph3_a[1]) == 3
        @test length(ph3_a[2]) == 1

        import Libdl
        lib    = Libdl.dlopen(libpath)
        ver_fn = Libdl.dlsym(lib, :cech_version)
        ver    = unsafe_string(ccall(ver_fn, Ptr{Cchar}, ()))
        @test startswith(ver, "TopoTS-CechCore")

        cr_fn     = Libdl.dlsym(lib, :cech_circumradius)
        pts2_flat = Float64[0.0, 0.0, 2.0, 0.0]
        v2        = Int32[0, 1]
        r2        = ccall(cr_fn, Cdouble,
                          (Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Cint),
                          pts2_flat, 2, 2, v2, 2)
        @test abs(r2 - 1.0) < 1e-9

        pts3_flat = Float64[0.0, 0.0, 2.0, 0.0, 1.0, sqrt(3.0)]
        v3        = Int32[0, 1, 2]
        r3        = ccall(cr_fn, Cdouble,
                          (Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Cint),
                          pts3_flat, 3, 2, v3, 3)
        @test abs(r3 - 2.0 / sqrt(3.0)) < 1e-7
        Libdl.dlclose(lib)

        ts      = circle_series(200; noise=0.05)
        emb     = embed(ts; dim=2, lag=10)
        dgms_c  = persistent_homology(emb; dim_max=1, filtration=:cech, threshold=3.0)
        dgms_r  = persistent_homology(emb; dim_max=1, threshold=3.0)
        @test dgms_c.filtration == :cech
        @test !isempty(dgms_c[1])
        @test count(p -> !isfinite(death(p)), dgms_c[1]) == 1
        @test count(p -> !isfinite(death(p)), dgms_r[1]) == 1

        λ = landscape(dgms_c, 1; n_grid=50, n_layers=2)
        @test size(λ.layers) == (2, 50)

        bc = betti_curve(dgms_c, 1; n_grid=50)
        @test length(bc) == 50

        img = persistence_image(dgms_c, 1; n_pixels=8)
        @test size(img.pixels) == (8, 8)

        feat = topo_features(dgms_c; spec=TopoFeatureSpec(
            dim_max=1, n_landscape_grid=10, n_landscape_layers=2,
            n_betti_grid=5, use_image=false))
        @test length(feat) > 0
        @test !any(isnan, feat)

        ts_w = circle_series(400; noise=0.05)
        wd   = windowed_ph(ts_w; window=80, step=20, dim=2, lag=6,
                           dim_max=1, filtration=:cech, threshold=3.0)
        @test length(wd) > 0
        @test wd[1] isa DiagramCollection

        ls = landscape_score(wd, 1; n_grid=30, n_layers=2)
        @test length(ls.scores) == length(wd) - 1
        @test all(ls.scores .≥ 0)

        cp = crocker(wd; dim=1, n_scale=30)
        @test size(cp.surface, 1) == 30
    end
end

include("test_spectral_changepoint.jl")
include("test_changepoint.jl")
