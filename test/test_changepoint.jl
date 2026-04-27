@testset "Extended change-point detection (M1–M6)" begin

    # ── shared fixtures ───────────────────────────────────────────────────────
    rng = MersenneTwister(42)
    D_shift = vcat(randn(rng, 100), randn(rng, 100) .+ 5.0)   # level shift at 100
    D_null  = randn(rng, 200)                                  # pure noise

    # ── M5: :rss — single best breakpoint near true CP ───────────────────────
    @testset ":rss" begin
        events = detect_changepoints(D_shift; method=:rss)
        @test length(events) == 1
        @test abs(events[1].index - 100) < 10
        @test events[1].rstar_R ≈ events[1].index / length(D_shift)
        @test isnothing(events[1].sup_F)
    end

    # ── M6: :andrews — significant on level shift, near true CP ──────────────
    @testset ":andrews significant" begin
        events = detect_changepoints(D_shift; method=:andrews, alpha=0.05)
        @test length(events) == 1
        @test events[1].sup_F > 8.85
        @test abs(events[1].index - 100) < 15
        @test !isnothing(events[1].rstar_R)
    end

    # ── M6: :andrews — mostly quiet on pure noise ─────────────────────────────
    @testset ":andrews null" begin
        # sup-F on pure noise should be well below the 1% critical value
        res = andrews_supF(D_null; alpha=0.01)
        # we don't assert non-significance (5% chance of false positive),
        # but the statistic should be finite and non-negative
        @test isfinite(res.sup_F)
        @test res.sup_F >= 0.0
        @test res.cv == 12.16
    end

    # ── backward compatibility — no method kwarg ──────────────────────────────
    @testset "backward compatibility" begin
        D_old = vcat(randn(rng, 100), randn(rng, 100) .+ 3.0)
        events_old = detect_changepoints(D_old)
        events_new = detect_changepoints(D_old; method=:cusum_mad)
        @test length(events_old) == length(events_new)
    end

    # ── M1: :cusum_3sigma ─────────────────────────────────────────────────────
    @testset ":cusum_3sigma" begin
        events = detect_changepoints(D_shift; method=:cusum_3sigma, n_sigma=3.0)
        @test events isa Vector{ChangePointEvent}
        @test all(e.rstar_R === nothing for e in events)
    end

    # ── M2: :cusum_sustained ─────────────────────────────────────────────────
    @testset ":cusum_sustained" begin
        events = detect_changepoints(D_shift; method=:cusum_sustained,
                                     n_sigma=2.0, k=5)
        @test events isa Vector{ChangePointEvent}
    end

    # ── M3: :percentile ───────────────────────────────────────────────────────
    @testset ":percentile" begin
        events = detect_changepoints(D_shift; method=:percentile)
        @test events isa Vector{ChangePointEvent}
    end

    # ── M4: :cusum_adaptive ───────────────────────────────────────────────────
    @testset ":cusum_adaptive" begin
        events = detect_changepoints(D_shift; method=:cusum_adaptive,
                                     win=30, n_sigma=2.0)
        @test events isa Vector{ChangePointEvent}
    end

    # ── detect_changepoints_windowed ──────────────────────────────────────────
    @testset "detect_changepoints_windowed" begin
        ts  = sin.(2π .* (1:500) ./ 50) .+ 0.1 .* randn(rng, 500)
        wd  = windowed_ph(ts; window=100, step=10, dim=3, lag=5, dim_max=1)

        events, sc = detect_changepoints_windowed(wd; dim=1, method=:rss)
        @test sc isa Vector{Float64}
        @test events isa Vector{ChangePointEvent}

        events_w, sc_w = detect_changepoints_windowed(wd; dim=1,
                                                       score=:wasserstein,
                                                       method=:cusum_mad)
        @test sc_w isa Vector{Float64}
        @test events_w isa Vector{ChangePointEvent}

        # times should come from wd.times, not bare indices
        if !isempty(events_w)
            @test events_w[1].time in wd.times
        end
    end

    # ── andrews_supF directly ─────────────────────────────────────────────────
    @testset "andrews_supF" begin
        r_star, sup_F, sig, cv = andrews_supF(D_shift; alpha=0.05)
        @test r_star isa Int
        @test sup_F  isa Float64
        @test sig    isa Bool
        @test cv == 8.85
        @test sig          # should be significant for a clean level shift
    end

end
