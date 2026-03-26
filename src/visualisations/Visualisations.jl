"""
    Visualisations

Makie-based plotting functions for TopoTS objects.

All functions require a Makie backend to be loaded **before** calling:
```julia
using CairoMakie   # or GLMakie / WGLMakie
```
No hard Makie dependency is declared — detection is done at runtime.

# Exported functions

| Function                     | Input                  | Output                         |
|------------------------------|------------------------|--------------------------------|
| `plot_diagram`               | `DiagramCollection`    | Persistence diagram scatter    |
| `plot_diagram_multi`         | `DiagramCollection`    | Grid of diagrams H₀…Hₖ        |
| `plot_barcode`               | `DiagramCollection`    | Barcode (horizontal bars)      |
| `plot_landscape`             | `PersistenceLandscape` | Landscape function layers      |
| `plot_betti_curve`           | `BettiCurve`           | Betti curve step function      |
| `plot_crocker`               | `CROCKERPlot`          | Heatmap β_k(ε, t)              |
| `plot_crocker_multi`         | `Vector{CROCKERPlot}`  | Stacked CROCKER heatmaps       |
| `plot_changepoint_score`     | `ChangePointResult`    | Score + detected events        |
"""
module Visualisations

using ..Filtration:   DiagramCollection
using ..Landscapes:   PersistenceLandscape
using ..BettiCurves:  BettiCurve
using ..CROCKER:      CROCKERPlot
using ..ChangePoint:  ChangePointResult, ChangePointEvent
using PersistenceDiagrams: birth, death

export plot_diagram, plot_diagram_multi,
       plot_barcode,
       plot_landscape,
       plot_betti_curve,
       plot_crocker, plot_crocker_multi,
       plot_changepoint_score

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

function _makie()
    for pkg in (:CairoMakie, :GLMakie, :WGLMakie)
        isdefined(Main, pkg) && return getfield(Main, pkg)
    end
    error("plot_* functions require a Makie backend. " *
          "Please run `using CairoMakie` (or GLMakie / WGLMakie) first.")
end

_dim_label(k) = k == 0 ? "H₀" : k == 1 ? "H₁" : k == 2 ? "H₂" : "H$k"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Persistence diagram
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_diagram(dgms::DiagramCollection, dim::Int = 1; kwargs...) -> Figure

Scatter plot of a persistence diagram for homological dimension `dim`.

Off-diagonal points are coloured by persistence `d - b`.
Infinite death values are drawn at the top of the axis.

# Keyword arguments
- `colormap`       — point colormap (default `:plasma`)
- `point_size`     — marker size (default `10`)
- `diagonal`       — draw the diagonal line (default `true`)
- `diagonal_color` — diagonal colour (default `:gray60`)
- `title`          — plot title (default auto-generated)
- `infinite_label` — show "∞" marker on the death axis (default `true`)
- `threshold`      — only show points with persistence ≥ threshold (default `0`)
- `figsize`        — figure size (default `(500, 500)`)
"""
function plot_diagram(dgms::DiagramCollection, dim::Int = 1;
                      colormap       = :plasma,
                      point_size     = 10,
                      diagonal       = true,
                      diagonal_color = :gray60,
                      title          = "$(_dim_label(dim)) persistence diagram ($(dgms.filtration))",
                      infinite_label = true,
                      threshold      = 0.0,
                      figsize        = (500, 500))

    Mk = _makie()
    1 ≤ dim + 1 ≤ length(dgms) || throw(ArgumentError(
        "dim=$dim not available in this DiagramCollection"))

    raw = dgms[dim + 1]

    finite = [(Float64(birth(p)), Float64(death(p)))
              for p in raw if isfinite(death(p)) &&
                              Float64(death(p)) - Float64(birth(p)) ≥ threshold]
    infpts = [Float64(birth(p)) for p in raw if !isfinite(death(p))]

    bs = isempty(finite) ? Float64[] : first.(finite)
    ds = isempty(finite) ? Float64[] : last.(finite)
    ps = ds .- bs

    lo = isempty(bs) ? 0.0 : min(0.0, minimum(bs) - 0.05)
    hi_d = isempty(ds) ? 1.0 : maximum(ds)
    hi_b = isempty(bs) ? 1.0 : maximum(bs)
    hi   = max(hi_d, hi_b) * 1.08
    inf_y = hi * 0.97

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title      = title,
                  xlabel     = "birth",
                  ylabel     = "death",
                  titlesize  = 14,
                  xlabelsize = 12,
                  ylabelsize = 12,
                  aspect     = Mk.DataAspect())

    if diagonal
        Mk.lines!(ax, [lo, hi], [lo, hi];
                  color     = diagonal_color,
                  linewidth = 1,
                  linestyle = :dash)
    end

    if !isempty(finite)
        sc = Mk.scatter!(ax, bs, ds;
                         color       = ps,
                         colormap    = colormap,
                         markersize  = point_size,
                         strokewidth = 0.5,
                         strokecolor = :black)
        Mk.Colorbar(fig[1, 2], sc;
                    label     = "persistence",
                    labelsize = 11,
                    width     = 12)
    end

    if !isempty(infpts) && infinite_label
        Mk.scatter!(ax, infpts, fill(inf_y, length(infpts));
                    color      = :gray40,
                    marker     = :utriangle,
                    markersize = point_size,
                    label      = "d = ∞")
        Mk.hlines!(ax, [inf_y]; color = :gray70, linestyle = :dot, linewidth = 1)
        Mk.text!(ax, hi * 0.02, inf_y; text = "∞", fontsize = 11, color = :gray40)
    end

    Mk.xlims!(ax, lo, hi)
    Mk.ylims!(ax, lo, hi)
    Mk.colgap!(fig.layout, 8)

    return fig
end


"""
    plot_diagram_multi(dgms::DiagramCollection; kwargs...) -> Figure

Plot persistence diagrams for all available homological dimensions
in a horizontal grid.
"""
function plot_diagram_multi(dgms::DiagramCollection;
                            colormap   = :plasma,
                            point_size = 10,
                            diagonal   = true,
                            title      = nothing,
                            threshold  = 0.0,
                            figsize    = (480 * length(dgms), 480))

    Mk = _makie()
    n   = length(dgms)
    fig = Mk.Figure(; size = figsize)

    for dim in 0:(n - 1)
        raw = dgms[dim + 1]
        finite = [(Float64(birth(p)), Float64(death(p)))
                  for p in raw if isfinite(death(p)) &&
                                  Float64(death(p)) - Float64(birth(p)) ≥ threshold]
        infpts = [Float64(birth(p)) for p in raw if !isfinite(death(p))]

        bs = isempty(finite) ? Float64[] : first.(finite)
        ds = isempty(finite) ? Float64[] : last.(finite)
        ps = ds .- bs

        lo = 0.0
        hi = isempty([bs; ds]) ? 1.0 : max(maximum([bs; ds]) * 1.08, 0.1)
        inf_y = hi * 0.96

        ax = Mk.Axis(fig[1, dim + 1];
                     title      = isnothing(title) ? "$(_dim_label(dim))  ($(length(raw)) pts)" : string(title),
                     xlabel     = "birth",
                     ylabel     = dim == 0 ? "death" : "",
                     titlesize  = 13,
                     xlabelsize = 11,
                     ylabelsize = 11,
                     aspect     = Mk.DataAspect())

        if diagonal
            Mk.lines!(ax, [lo, hi], [lo, hi];
                      color = :gray60, linewidth = 1, linestyle = :dash)
        end

        if !isempty(finite)
            Mk.scatter!(ax, bs, ds;
                        color       = ps,
                        colormap    = colormap,
                        markersize  = point_size,
                        strokewidth = 0.4,
                        strokecolor = :black)
        end

        if !isempty(infpts)
            Mk.scatter!(ax, infpts, fill(inf_y, length(infpts));
                        color = :gray40, marker = :utriangle,
                        markersize = point_size)
            Mk.hlines!(ax, [inf_y]; color = :gray70, linestyle = :dot, linewidth = 1)
        end

        Mk.xlims!(ax, lo, hi)
        Mk.ylims!(ax, lo, hi)
    end

    Mk.colgap!(fig.layout, 10)
    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 2. Barcode
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_barcode(dgms::DiagramCollection, dim::Int = 1; kwargs...) -> Figure

Barcode plot — horizontal bars `[birth, death]` for each feature,
sorted by birth time.
"""
function plot_barcode(dgms::DiagramCollection, dim::Int = 1;
                      colormap   = :plasma,
                      bar_height = 0.7,
                      inf_extend = 0.05,
                      threshold  = 0.0,
                      title      = "$(_dim_label(dim)) barcode ($(dgms.filtration))",
                      figsize    = (600, 400))

    Mk = _makie()
    raw = dgms[dim + 1]

    pairs = sort([(Float64(birth(p)),
                   isfinite(death(p)) ? Float64(death(p)) : Inf)
                  for p in raw
                  if (isfinite(death(p)) ? Float64(death(p)) - Float64(birth(p)) : Inf) ≥ threshold],
                 by = first)

    isempty(pairs) && @warn "No features to plot for H$dim"

    finite_deaths = [d for (_, d) in pairs if isfinite(d)]
    d_max = isempty(finite_deaths) ? 1.0 : maximum(finite_deaths)
    x_max = d_max * (1.0 + inf_extend)

    bs = first.(pairs)
    ds = [isfinite(d) ? d : x_max for (_, d) in pairs]
    ps = ds .- bs

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title               = title,
                  xlabel              = "filtration scale ε",
                  ylabel              = "feature index",
                  titlesize           = 14,
                  xlabelsize          = 12,
                  ylabelsize          = 12,
                  yticksvisible       = false,
                  yticklabelsvisible  = false)

    cmap = Mk.cgrad(colormap)
    denom = isempty(ps) ? 1.0 : max(maximum(ps), eps())

    for (i, (b, d)) in enumerate(pairs)
        pers = isfinite(d) ? (d - b) : (x_max - b)
        col  = cmap[clamp(pers / denom, 0, 1)]
        Mk.lines!(ax, [b, ds[i]], [i, i];
                  color = col,
                  linewidth = 6 * bar_height)
    end

    if any(!isfinite(d) for (_, d) in pairs)
        Mk.text!(ax, x_max, length(pairs) + 0.5;
                 text = "∞ →",
                 align = (:right, :bottom),
                 fontsize = 11,
                 color = :gray40)
    end

    Mk.xlims!(ax, 0, x_max)
    Mk.ylims!(ax, 0, max(length(pairs) + 1, 2))

    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 3. Persistence landscape
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_landscape(λ::PersistenceLandscape; kwargs...) -> Figure

Plot the persistence landscape layers λ₁, λ₂, … as overlaid curves.
"""
function plot_landscape(λ::PersistenceLandscape;
                        n_layers   = Base.size(λ.layers, 1),
                        colormap   = :tab10,
                        linewidth  = 2,
                        fill_first = true,
                        title      = "$(_dim_label(λ.dim)) persistence landscape",
                        figsize    = (700, 350))

    Mk    = _makie()
    K     = min(n_layers, Base.size(λ.layers, 1))
    tgrid = collect(λ.tgrid)
    cmap  = Mk.cgrad(colormap, K; categorical = true)

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title      = title,
                  xlabel     = "scale t",
                  ylabel     = "λₖ(t)",
                  titlesize  = 14,
                  xlabelsize = 12,
                  ylabelsize = 12)

    for k in 1:K
        vals = λ.layers[k, :]
        col  = cmap[k]
        if k == 1 && fill_first
            Mk.band!(ax, tgrid, zeros(length(tgrid)), vals;
                     color = (col, 0.15))
        end
        Mk.lines!(ax, tgrid, vals;
                  color     = col,
                  linewidth = linewidth,
                  label     = "λ$k")
    end

    K > 1 && Mk.axislegend(ax; position = :rt, framevisible = false,
                                labelsize = 11)
    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 4. Betti curve
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_betti_curve(bc::BettiCurve; kwargs...) -> Figure

Step-function plot of the Betti curve β_k(ε).
"""
function plot_betti_curve(bc::BettiCurve;
                          color      = :steelblue,
                          fill       = true,
                          title      = "$(_dim_label(bc.dim)) Betti curve",
                          figsize    = (650, 300))

    Mk    = _makie()
    tgrid = collect(bc.tgrid)
    vals  = Float64.(bc.values)

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title      = title,
                  xlabel     = "filtration scale ε",
                  ylabel     = "β_$(bc.dim)(ε)",
                  titlesize  = 14,
                  xlabelsize = 12,
                  ylabelsize = 12)

    if fill
        Mk.band!(ax, tgrid, zeros(length(tgrid)), vals;
                 color = (color, 0.15))
    end
    Mk.stairs!(ax, tgrid, vals; color = color, linewidth = 2)

    Mk.ylims!(ax, -0.1, max(maximum(vals) + 0.5, 1.5))
    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROCKER plot
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_crocker(cp::CROCKERPlot; kwargs...) -> Figure

Heatmap of the CROCKER surface β_k(ε, t).
"""
function plot_crocker(cp::CROCKERPlot;
                      colormap    = :viridis,
                      title       = "CROCKER — $(_dim_label(cp.dim)) (window=$(cp.window), step=$(cp.step))",
                      xlabel      = "time (window index)",
                      ylabel      = "filtration scale ε",
                      colorbar    = true,
                      vlines      = Float64[],
                      vline_color = :red,
                      vline_style = :dash,
                      figsize     = (800, 400))

    Mk     = _makie()
    times  = cp.times
    scales = collect(cp.scales)
    mat    = cp.surface'

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title      = title,
                  xlabel     = xlabel,
                  ylabel     = ylabel,
                  titlesize  = 14,
                  xlabelsize = 12,
                  ylabelsize = 12)

    hm = Mk.heatmap!(ax, times, scales, mat; colormap = colormap)

    if colorbar
        Mk.Colorbar(fig[1, 2], hm;
                    label     = "β_$(cp.dim)",
                    labelsize = 12,
                    width     = 14)
        Mk.colgap!(fig.layout, 8)
    end

    if !isempty(vlines)
        for vl in vlines
            Mk.vlines!(ax, [Float64(vl)];
                       color     = vline_color,
                       linestyle = vline_style,
                       linewidth = 2)
        end
    end

    return fig
end


"""
    plot_crocker_multi(cps::Vector{CROCKERPlot}; kwargs...) -> Figure

Vertically stacked CROCKER heatmaps sharing the same time axis.
"""
function plot_crocker_multi(cps::Vector{CROCKERPlot};
                            colormap    = :viridis,
                            titles      = ["CROCKER $(_dim_label(cp.dim))" for cp in cps],
                            xlabel      = "time (window index)",
                            ylabel      = "scale ε",
                            colorbar    = true,
                            vlines      = Float64[],
                            vline_color = :red,
                            vline_style = :dash,
                            figsize     = (800, 320 * length(cps)))

    Mk = _makie()
    n  = length(cps)
    fig = Mk.Figure(; size = figsize)

    for (i, cp) in enumerate(cps)
        mat = cp.surface'
        ax  = Mk.Axis(fig[i, 1];
                      title      = titles[i],
                      xlabel     = i == n ? xlabel : "",
                      ylabel     = ylabel,
                      titlesize  = 13,
                      xlabelsize = 11,
                      ylabelsize = 11)

        hm = Mk.heatmap!(ax, cp.times, collect(cp.scales), mat; colormap = colormap)

        if colorbar
            Mk.Colorbar(fig[i, 2], hm;
                        label     = "β_$(cp.dim)",
                        labelsize = 11,
                        width     = 12)
        end

        if !isempty(vlines)
            for vl in vlines
                Mk.vlines!(ax, [Float64(vl)];
                           color     = vline_color,
                           linestyle = vline_style,
                           linewidth = 2)
            end
        end
    end

    Mk.colgap!(fig.layout, 8)
    Mk.rowgap!(fig.layout, 6)
    return fig
end


# ─────────────────────────────────────────────────────────────────────────────
# 6. Change-point score
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_changepoint_score(result::ChangePointResult; kwargs...) -> Figure

Line plot of the change-point score over time, with detected events
marked as vertical lines.
"""
function plot_changepoint_score(result::ChangePointResult;
                                events      = ChangePointEvent[],
                                color       = :steelblue,
                                event_color = :red,
                                threshold   = nothing,
                                title       = "Change-point score ($(result.method))",
                                figsize     = (800, 300))

    Mk     = _makie()
    times  = result.times
    scores = result.scores

    fig = Mk.Figure(; size = figsize)
    ax  = Mk.Axis(fig[1, 1];
                  title      = title,
                  xlabel     = "time (window index)",
                  ylabel     = "score",
                  titlesize  = 14,
                  xlabelsize = 12,
                  ylabelsize = 12)

    Mk.lines!(ax, times, scores; color = color, linewidth = 2)

    if !isnothing(threshold)
        Mk.hlines!(ax, [Float64(threshold)];
                   color     = :gray50,
                   linestyle = :dash,
                   linewidth = 1.5,
                   label     = "threshold")
    end

    if !isempty(events)
        first_ev = events[1]
        Mk.vlines!(ax, [Float64(first_ev.index)];
                   color     = event_color,
                   linestyle = :dash,
                   linewidth = 2,
                   label     = "detected")

        for ev in events[2:end]
            Mk.vlines!(ax, [Float64(ev.index)];
                       color     = event_color,
                       linestyle = :dash,
                       linewidth = 2)
        end

        Mk.axislegend(ax; position = :rt, framevisible = false, labelsize = 11)
    end

    Mk.ylims!(ax, 0, max(maximum(scores) * 1.1, 1.05))
    return fig
end


end # module Visualisations
