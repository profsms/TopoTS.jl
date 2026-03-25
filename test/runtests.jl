using Test
using TopoTS

function circle_series(n; noise=0.0)
    t = range(0, 2π, length=n+1)[1:n]
    x = sin.(t)
    noise > 0 && (x .+= noise .* sin.(17 .* t .+ 1.3))
    return x
end

function piecewise_series(n; cp=nothing)
    cp = isnothing(cp) ? n÷2 : cp
    t1 = range(0, 10π, length=cp)
    t2 = range(0, 10π, length=n-cp) .+ 5π
    vcat(sin.(t1), sin.(t2) .* 0.3)
end

@testset "TopoTS.jl" begin

    @testset "Embedding" begin
        ts  = circle_series(500)
        emb = embed(ts; dim=3, lag=10)
        @test size(emb.points) == (480, 3)
        @test_throws ArgumentError embed(ts; dim=0, lag=1)
        @test 1 ≤ optimal_lag(ts; max_lag=30) ≤ 30
    end

    @testset "Filtration" begin
        emb  = embed(circle_series(200; noise=0.05); dim=2, lag=12)
        dgms = persistent_homology(emb; dim_max=1)
        @test length(dgms) == 2
        @test dgms.filtration == :rips
    end

    @testset "Vectorisations" begin
        dgms = persistent_homology(embed(circle_series(300; noise=0.03); dim=2, lag=12); dim_max=1)
        bc   = betti_curve(dgms, 1; n_grid=100)
        @test length(bc) == 100 && all(bc.values .≥ 0)
        λ    = landscape(dgms, 1; n_layers=3, n_grid=100)
        @test size(λ.layers) == (3, 100)
        img  = persistence_image(dgms, 1; n_pixels=10)
        @test length(vec(img)) == 100
    end

    @testset "TopoStats" begin
        dgms = persistent_homology(embed(circle_series(300; noise=0.03); dim=2, lag=12); dim_max=1)
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
        @test bn.score_type == :bottleneck && all(bn.scores .≥ 0)
        ws = wasserstein_score(wd, 1)
        @test length(ws.scores) == length(bn.scores)
        ls = landscape_score(wd, 1; n_grid=50, n_layers=2)
        @test length(ls.scores) == length(bn.scores)
        all_s = changepoint_score(wd, 1)
        @test haskey(all_s, :bottleneck)
    end

    @testset "detect_changepoints" begin
        ts = piecewise_series(600; cp=300)
        wd = windowed_ph(ts; window=100, step=15, dim=2, lag=6, dim_max=1)
        ls = landscape_score(wd, 1; n_grid=50, n_layers=2)
        events = detect_changepoints(ls; threshold=:mad, n_mad=2.0)
        @test events isa Vector{ChangePointEvent}
        length(events) ≥ 2 && @test abs(events[2].index - events[1].index) ≥ 5
    end

    @testset "Sublevel PH" begin
        f   = vcat(sin.(range(0, 4π, length=200)), cos.(range(0, 4π, length=200)))
        dgm = sublevel_ph(f)
        @test !isempty(dgm.H0)
        @test all(b < d for (b,d) in dgm.H0)
        dgm_ext = sublevel_ph(f; extended=true)
        @test !isempty(dgm_ext.H1)
        result = windowed_sublevel_ph(f; window=80, step=20)
        @test length(result.diagrams) == length(result.times)
    end

    @testset "Multivariate embedding" begin
        N  = 500
        X  = hcat(sin.(range(0, 6π, length=N)), cos.(range(0, 6π, length=N)))
        emb = embed_multivariate(X; dim=2, lag=5)
        @test emb.n_channels == 2
        @test size(emb.points, 2) == 4
        dgms = persistent_homology(emb; dim_max=1)
        @test dgms isa DiagramCollection
    end

    @testset "Diagram kernels" begin
        dgms = persistent_homology(embed(circle_series(300; noise=0.03); dim=2, lag=12); dim_max=1)
        dgm  = dgms[2]
        @test pss_kernel(dgm, dgm; sigma=0.5) ≥ 0
        @test pwg_kernel(dgm, dgm; sigma=0.5) ≥ 0
        @test sliced_wasserstein_kernel(dgm, dgm; sigma=0.5, n_directions=20) ≈ 1.0 atol=1e-5
        K = kernel_matrix([dgms], 1; kernel=:pss, sigma=0.5)
        @test size(K) == (1, 1)
    end

    @testset "Feature extraction" begin
        ts   = circle_series(300; noise=0.03)
        spec = TopoFeatureSpec(dim=2, lag=10, dim_max=1,
                               n_landscape_grid=10, n_landscape_layers=2,
                               n_betti_grid=5, use_image=false)
        feat  = topo_features(ts; spec=spec)
        names = feature_names(spec)
        @test length(feat) == length(names)
        @test !any(isnan, feat)
        emb  = embed(ts; dim=2, lag=10)
        dgms = persistent_homology(emb; dim_max=1)
        @test topo_features(dgms; spec=spec) ≈ feat
    end

end

# ── Filtration variety tests ──────────────────────────────────────────────────
@testset "Filtration types" begin

    ts  = circle_series(300; noise=0.03)
    emb = embed(ts; dim=2, lag=12)

    # :rips — baseline
    dgms_rips = persistent_homology(emb; dim_max=1)
    @test dgms_rips.filtration == :rips

    # :alpha — should produce a valid diagram
    dgms_alpha = persistent_homology(emb; dim_max=1, filtration=:alpha)
    @test dgms_alpha.filtration == :alpha
    @test length(dgms_alpha) == 2

    # :edge_collapsed — same number of H₀ classes as :rips
    dgms_ec = persistent_homology(emb; dim_max=1, filtration=:edge_collapsed)
    @test dgms_ec.filtration == :edge_collapsed
    @test length(dgms_ec[1]) == length(dgms_rips[1])

    # :cubical — H₀ only; each local minimum yields a class
    dgms_cub = persistent_homology(emb; dim_max=1, filtration=:cubical)
    @test dgms_cub.filtration == :cubical
    @test !isempty(dgms_cub[1])

    # invalid symbol
    @test_throws ArgumentError persistent_homology(emb; filtration=:unknown)
end

# ── Čech filtration tests ─────────────────────────────────────────────────────
@testset "Čech filtration (libcech)" begin
    libpath = joinpath(dirname(@__DIR__), "deps", "lib",
                       Sys.iswindows() ? "cech.dll" :
                       Sys.isapple()   ? "libcech.dylib" : "libcech.so")

    if !isfile(libpath)
        @warn "libcech not found at $libpath — skipping Čech tests.\n" *
              "Build with: cd csrc && make"
        @test_skip true
    else
        # ── ABI sanity via Libdl ──────────────────────────────────────────
        import Libdl
        lib = Libdl.dlopen(libpath)

        ver_fn = Libdl.dlsym(lib, :cech_version)
        ver    = unsafe_string(ccall(ver_fn, Ptr{Cchar}, ()))
        @test startswith(ver, "TopoTS-CechCore")

        # circumradius: two points at distance 2 → r = 1
        cr_fn = Libdl.dlsym(lib, :cech_circumradius)
        pts2  = Float64[0.0, 0.0, 2.0, 0.0]
        v2    = Int32[0, 1]
        r = ccall(cr_fn, Cdouble,
                  (Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Cint),
                  pts2, 2, 2, v2, 2)
        @test abs(r - 1.0) < 1e-9

        # circumradius: equilateral triangle, side=2 → r = 2/√3
        pts3 = Float64[0.0,0.0, 2.0,0.0, 1.0,sqrt(3.0)]
        v3   = Int32[0, 1, 2]
        r3 = ccall(cr_fn, Cdouble,
                   (Ptr{Cdouble}, Cint, Cint, Ptr{Cint}, Cint),
                   pts3, 3, 2, v3, 3)
        @test abs(r3 - 2.0/sqrt(3.0)) < 1e-7

        Libdl.dlclose(lib)

        # ── Via persistent_homology API ────────────────────────────────────
        ts   = circle_series(200; noise=0.05)
        emb  = embed(ts; dim=2, lag=10)

        dgms_cech = persistent_homology(emb;
                                         dim_max=1,
                                         filtration=:cech,
                                         threshold=3.0)
        @test dgms_cech.filtration == :cech
        @test length(dgms_cech) == 2

        # Čech / Rips interleaving: H₀ classes should agree exactly
        dgms_rips = persistent_homology(emb; dim_max=1, threshold=3.0)
        @test length(dgms_cech[1]) == length(dgms_rips[1])

        # H₁: Čech birth times ≤ Rips birth times (Čech is tighter)
        if !isempty(dgms_cech[2]) && !isempty(dgms_rips[2])
            max_cech_birth = maximum(Float64(birth(p)) for p in dgms_cech[2])
            max_rips_birth = maximum(Float64(birth(p)) for p in dgms_rips[2])
            @test max_cech_birth <= max_rips_birth * sqrt(2) + 1e-6
        end

        # Čech should plug into the full pipeline unchanged
        λ  = landscape(dgms_cech, 1; n_grid=50, n_layers=2)
        @test size(λ.layers) == (2, 50)

        bc = betti_curve(dgms_cech, 1; n_grid=50)
        @test length(bc) == 50

        img = persistence_image(dgms_cech, 1; n_pixels=8)
        @test size(img.pixels) == (8, 8)

        feat = topo_features(dgms_cech;
                              spec=TopoFeatureSpec(dim_max=1,
                                                   n_landscape_grid=10,
                                                   n_landscape_layers=2,
                                                   n_betti_grid=5,
                                                   use_image=false))
        @test length(feat) > 0 && !any(isnan, feat)
    end

    # ── Čech in windowed pipeline ─────────────────────────────────────────
    if isfile(libpath)
        ts = circle_series(400; noise=0.05)
        wd = windowed_ph(ts; window=80, step=20, dim=2, lag=6,
                          dim_max=1, filtration=:cech, threshold=3.0)
        @test wd.filtration == :cech
        @test length(wd) > 0
        @test wd[1] isa DiagramCollection

        # change-point scores should work through Čech diagrams
        ls = landscape_score(wd, 1; n_grid=30, n_layers=2)
        @test length(ls.scores) == length(wd) - 1
        @test all(ls.scores .>= 0)

        cp = crocker(wd; dim=1, n_scale=30)
        @test size(cp.surface, 1) == 30
    end
end
