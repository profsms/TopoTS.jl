@testset "Spectral changepoint detection" begin

    # ── shared fixture ────────────────────────────────────────────────────────
    sig = sin.(range(0, 20π, length=500)) .+ 0.01 .* randn(500)

    # ── 1. periodogram_ph returns SublevelDiagram ─────────────────────────────
    @testset "periodogram_ph" begin
        dgm = periodogram_ph(sig; bw=5)
        @test dgm isa SublevelDiagram
        @test length(dgm.H0) > 0
        @test all(isfinite(d) for (_, d) in dgm.H0)
    end

    # ── 2. wasserstein_distance ───────────────────────────────────────────────
    @testset "wasserstein_distance" begin
        dgm = periodogram_ph(sig; bw=5)

        # identity: W(D, D) = 0
        @test wasserstein_distance(dgm.H0, dgm.H0) ≈ 0.0 atol=1e-10

        # non-negative
        dgm2 = periodogram_ph(randn(500); bw=5)
        d12 = wasserstein_distance(dgm.H0, dgm2.H0)
        @test d12 >= 0.0

        # empty diagram edge cases
        @test wasserstein_distance(Tuple{Float64,Float64}[], dgm.H0) >= 0.0
        @test wasserstein_distance(dgm.H0, Tuple{Float64,Float64}[]) >= 0.0
        @test wasserstein_distance(Tuple{Float64,Float64}[],
                                   Tuple{Float64,Float64}[]) == 0.0

        # p=1 and p=2 are both valid
        @test wasserstein_distance(dgm.H0, dgm2.H0; p=1) >= 0.0
        @test wasserstein_distance(dgm.H0, dgm2.H0; p=2) >= 0.0
    end

    # ── 3. windowed_periodogram_ph (signal form) ──────────────────────────────
    @testset "windowed_periodogram_ph (signal)" begin
        wd = windowed_periodogram_ph(sig; window=100, step=10)
        @test wd isa WindowedDiagrams
        @test length(wd) > 0
        @test wd.times isa Vector{Float64}

        # plugs directly into changepoint_score without errors
        sc = changepoint_score(wd, 0)
        @test sc.wasserstein isa ChangePointResult
        @test sc.bottleneck  isa ChangePointResult
        @test sc.landscape   isa ChangePointResult
        @test length(sc.wasserstein.scores) == length(wd) - 1
    end

    # ── 4. windowed_periodogram_ph (trials form) ──────────────────────────────
    @testset "windowed_periodogram_ph (trials)" begin
        trials = [randn(500) for _ in 1:20]
        wd2 = windowed_periodogram_ph(trials)
        @test wd2 isa WindowedDiagrams
        @test length(wd2) == 20
        @test wd2.times == Float64.(1:20)

        sc2 = changepoint_score(wd2, 0)
        @test sc2.wasserstein isa ChangePointResult
        @test length(sc2.wasserstein.scores) == 19
    end

    # ── 5. windowed_sublevel_ph now returns WindowedDiagrams ─────────────────
    @testset "windowed_sublevel_ph return type" begin
        wd3 = windowed_sublevel_ph(sig; window=100, step=10)
        @test wd3 isa WindowedDiagrams
        @test length(wd3) > 0

        sc3 = changepoint_score(wd3, 0)
        @test sc3.wasserstein isa ChangePointResult
        @test sc3.bottleneck  isa ChangePointResult
        @test sc3.landscape   isa ChangePointResult
        @test length(sc3.wasserstein.scores) == length(wd3) - 1
    end

end
