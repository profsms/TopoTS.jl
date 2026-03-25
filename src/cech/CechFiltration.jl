"""
    CechFiltration

Julia ccall wrapper for the libcech C++ shared library.

Implements Ripserer.jl's AbstractFiltration interface via CechFilt,
so `ripserer(filt)` works directly.

The library path is passed in at construction time (from Filtration.jl
which resolves it via the TopoTS._CECH_LIB registry).  This module
itself is stateless with respect to the library path — it holds no
global handle, making it safe to use from multiple tasks.
"""
module CechFiltration

using Ripserer

export CechFilt, build_cech_filtration

# ─────────────────────────────────────────────────────────────────────────────
# C struct mirror — must match CechSimplex in cech_core.hpp byte-for-byte
# ─────────────────────────────────────────────────────────────────────────────

# Layout (verified by ABI test in csrc/):
#   [0–31]  int32_t verts[8]  =  32 bytes
#   [32–35] int32_t dim       =   4 bytes
#   [36–39] int32_t _pad      =   4 bytes  (compiler alignment padding)
#   [40–47] double  birth     =   8 bytes
#   total: 48 bytes
struct CechSimplexC
    verts :: NTuple{8, Int32}
    dim   :: Int32
    _pad  :: Int32
    birth :: Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# ccall wrappers
# ─────────────────────────────────────────────────────────────────────────────

function _call_build_filtration(pts::Matrix{Float64},
                                 dim_max::Int,
                                 threshold::Float64,
                                 libpath::String) :: Vector{CechSimplexC}
    n_pts, d = size(pts)

    # Julia is column-major; C expects row-major (n_pts × d) flat array
    pts_flat = Float64[pts[i, j] for i in 1:n_pts, j in 1:d] |> vec

    out_ptr = Ref{Ptr{CechSimplexC}}(C_NULL)
    out_n   = Ref{Int64}(0)

    ret = ccall(
        (:cech_build_filtration, libpath),
        Cint,
        (Ptr{Cdouble}, Cint, Cint, Cint, Cdouble,
         Ptr{Ptr{CechSimplexC}}, Ptr{Int64}),
        pts_flat, Int32(n_pts), Int32(d), Int32(dim_max), threshold,
        out_ptr, out_n
    )
    ret == 0 || error("cech_build_filtration returned error $ret")

    n   = out_n[]
    raw = unsafe_wrap(Array, out_ptr[], n; own=false)
    result = copy(raw)

    ccall((:cech_free, libpath), Cvoid, (Ptr{CechSimplexC},), out_ptr[])
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# CechFilt — Ripserer AbstractFiltration implementation
# ─────────────────────────────────────────────────────────────────────────────

"""
    CechFilt{I, T}

A Čech filtration on a finite point cloud, implementing Ripserer.jl's
`AbstractFiltration` interface.

Construct via `build_cech_filtration(pts; ...)`.
"""
struct CechFilt{I<:Integer, T<:AbstractFloat}
    simplices  :: Vector{Any}   # simplices[k+1] = all k-simplices
    n_pts      :: Int
    dim_max    :: Int
    _threshold :: T
end

Ripserer.nv(f::CechFilt)        = f.n_pts
Ripserer.threshold(f::CechFilt) = f._threshold
Ripserer.dim_max(f::CechFilt)   = f.dim_max

function Ripserer.simplex_type(::Type{CechFilt{I, T}}, ::Val{D}) where {I, T, D}
    Simplex{D, T, I}
end

Ripserer.birth(::CechFilt{I, T}, ::Integer) where {I, T} = zero(T)

function Ripserer.columns_to_reduce(f::CechFilt, ::Val{D}) where D
    D + 1 <= length(f.simplices) || return Simplex{D, Float64, Int32}[]
    return f.simplices[D + 1]
end

# ─────────────────────────────────────────────────────────────────────────────
# Public constructor
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_cech_filtration(pts::AbstractMatrix;
                          dim_max   :: Int    = 2,
                          threshold :: Real   = Inf,
                          libpath   :: String) -> CechFilt

Build the Čech filtration from a point cloud.

`pts` is (n_points × d).  `libpath` is the absolute path to `libcech`
resolved by `Filtration._compute_cech`.
"""
function build_cech_filtration(pts::AbstractMatrix;
                                dim_max   :: Int    = 2,
                                threshold :: Real   = Inf,
                                libpath   :: String)
    isfile(libpath) || error("libcech not found: '$libpath'")

    pts64  = Matrix{Float64}(pts)
    n_pts, _ = size(pts64)
    thr64  = Float64(threshold)

    raw = _call_build_filtration(pts64, dim_max, thr64, libpath)

    # Convert CechSimplexC → Ripserer Simplex, grouped by dimension
    simplices = [Simplex{k, Float64, Int32}[] for k in 0:dim_max]

    for cs in raw
        k = Int(cs.dim)
        k > dim_max && continue
        # Convert 0-based C indices → 1-based Julia indices
        verts = ntuple(i -> cs.verts[i] + Int32(1), k + 1)
        push!(simplices[k + 1], Simplex{k, Float64, Int32}(verts, cs.birth))
    end

    return CechFilt{Int32, Float64}(simplices, n_pts, dim_max, thr64)
end

end # module CechFiltration
