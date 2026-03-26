"""
    CechFiltration

Julia ccall wrapper for the libcech C++ shared library.

Implements Ripserer.jl's AbstractCustomFiltration interface via CechFilt,
so `ripserer(filt)` works directly.
"""
module CechFiltration

using Ripserer
using Libdl
using SparseArrays

export CechFilt, build_cech_filtration

# ─────────────────────────────────────────────────────────────────────────────
# C struct mirror — must match CechSimplex in cech_core.hpp byte-for-byte
# ─────────────────────────────────────────────────────────────────────────────

struct CechSimplexC
    verts :: NTuple{8, Int32}
    dim   :: Int32
    _pad  :: Int32
    birth :: Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# ccall wrappers
# ─────────────────────────────────────────────────────────────────────────────

function _call_build_filtration(
    pts::Matrix{Float64},
    dim_max::Int,
    threshold::Float64,
    libpath::String,
)::Vector{CechSimplexC}
    n_pts, d = size(pts)

    # C expects row-major flattening
    pts_flat = Float64[pts[i, j] for i in 1:n_pts for j in 1:d]

    out_ptr = Ref{Ptr{CechSimplexC}}(C_NULL)
    out_n   = Ref{Int64}(0)

    local lib
    try
        lib = Libdl.dlopen(libpath)

        build_sym = Libdl.dlsym(lib, :cech_build_filtration)
        free_sym  = Libdl.dlsym(lib, :cech_free)

        ret = ccall(
            build_sym,
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

        ccall(free_sym, Cvoid, (Ptr{CechSimplexC},), out_ptr[])
        return result
    finally
        if @isdefined(lib)
            try
                Libdl.dlclose(lib)
            catch
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# CechFilt — Ripserer AbstractCustomFiltration implementation
# ─────────────────────────────────────────────────────────────────────────────

struct CechFilt{I<:Signed, T<:AbstractFloat} <: Ripserer.AbstractCustomFiltration{I, T}
    adj        :: SparseMatrixCSC{Bool, Int}
    dicts      :: Vector{Dict{I, T}}
    _threshold :: T
end

Ripserer.simplex_dicts(f::CechFilt)    = f.dicts
Ripserer.adjacency_matrix(f::CechFilt) = f.adj
Ripserer.threshold(f::CechFilt)        = f._threshold

# ─────────────────────────────────────────────────────────────────────────────
# Public constructor
# ─────────────────────────────────────────────────────────────────────────────

function build_cech_filtration(
    pts::AbstractMatrix;
    dim_max   :: Int  = 2,
    threshold :: Real = Inf,
    libpath   :: String,
)
    isfile(libpath) || error("libcech not found: '$libpath'")

    pts64    = Matrix{Float64}(pts)
    n_pts, _ = size(pts64)
    thr64    = Float64(threshold)

    raw = _call_build_filtration(pts64, dim_max, thr64, libpath)

    # dicts[k+1] stores births of k-simplices by simplex index
    dicts = [Dict{Int, Float64}() for _ in 0:dim_max]

    # adjacency for 0-dimensional reduction / sparse coboundary
    adj_i = Int[]
    adj_j = Int[]
    adj_v = Bool[]

    for cs in raw
        k = Int(cs.dim)
        k > dim_max && continue

        verts = ntuple(i -> Int(cs.verts[i]) + 1, k + 1)

        # Use the vertex-based constructor, then store by simplex index
        sx  = Simplex{k}(verts, cs.birth)
        idx = Ripserer.index(sx)

        # keep smallest birth if duplicate appears
        dicts[k + 1][idx] = min(cs.birth, get(dicts[k + 1], idx, Inf))

        if k == 1
            u, v = verts
            push!(adj_i, u); push!(adj_j, v); push!(adj_v, true)
            push!(adj_i, v); push!(adj_j, u); push!(adj_v, true)
        end
    end

    adj = sparse(adj_i, adj_j, adj_v, n_pts, n_pts)

    return CechFilt{Int, Float64}(adj, dicts, thr64)
end

end # module CechFiltration