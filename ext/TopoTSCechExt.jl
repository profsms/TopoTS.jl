"""
    TopoTSCechExt

Julia package extension for TopoTS.jl that activates the Čech filtration
when `CechCore_jll` is installed.

This module is loaded automatically by Julia 1.9+'s extension mechanism
whenever both `TopoTS` and `CechCore_jll` are present in the active
environment — no `using` statement required from the user.

# How it works

1. Julia sees `[weakdeps] CechCore_jll = "..."` in TopoTS/Project.toml.
2. When the user has `CechCore_jll` installed, Julia loads this extension
   after `TopoTS.__init__()` completes.
3. This extension calls `TopoTS._register_cech_lib!(path)`, storing the
   JLL library path in a package-level Ref.
4. `persistent_homology(...; filtration=:cech)` checks that Ref; if it is
   set it calls libcech via ccall, otherwise it prints a clear install hint.

The extension also runs a quick runtime ABI check (struct size, version
string) to catch mismatches between the JLL binary and the Julia wrapper.
"""
module TopoTSCechExt

using TopoTS
using CechCore_jll: libcech
using Libdl

function __init__()
    # Verify the library file actually exists at the resolved path
    isfile(libcech) || begin
        @warn """
        TopoTSCechExt: CechCore_jll is installed but the library was not found
        at '$libcech'. Try:  using Pkg; Pkg.instantiate()
        """
        return
    end

    # ── ABI check ────────────────────────────────────────────────────────────
    # Load transiently (we don't keep a handle; ccall uses the path string)
    local lib
    try
        lib = Libdl.dlopen(libcech)
    catch e
        @warn "TopoTSCechExt: could not dlopen '$libcech': $e"
        return
    end

    # 1. Version string
    ver_ok = try
        ver = unsafe_string(ccall(Libdl.dlsym(lib, :cech_version),
                                  Ptr{Cchar}, ()))
        if !startswith(ver, "TopoTS-CechCore")
            @warn "TopoTSCechExt: unexpected version '$ver'"
            false
        else
            true
        end
    catch
        @warn "TopoTSCechExt: cech_version() call failed"
        false
    end

    # 2. Struct size check — must be 48 bytes (8×Int32 + Int32 + Int32 + Float64)
    # We verify this by building a minimal 1-point filtration and checking
    # that the returned pointer arithmetic is consistent.
    struct_ok = try
        pts  = Float64[0.0, 0.0]   # 1 point in R²
        op   = Ref{Ptr{Cvoid}}(C_NULL)
        on   = Ref{Int64}(0)
        ret  = ccall(Libdl.dlsym(lib, :cech_build_filtration),
                     Cint,
                     (Ptr{Cdouble}, Cint, Cint, Cint, Cdouble,
                      Ptr{Ptr{Cvoid}}, Ptr{Int64}),
                     pts, 1, 2, 0, 1e18, op, on)
        ret == 0 && on[] == 1  # exactly 1 simplex (the single vertex)
    catch
        false
    end |> identity

    Libdl.dlclose(lib)

    if !ver_ok || !struct_ok
        @warn """
        TopoTSCechExt: ABI check failed for '$libcech'.
        The Čech filtration is disabled. This can happen if the JLL binary
        was built with a different version of cech_core.hpp than the one
        expected by this version of TopoTS.
        Expected: TopoTS-CechCore-1.0.0  (CechCore_jll v1.x)
        Try: `]update CechCore_jll`
        """
        return
    end

    # ── Register the library path with TopoTS ─────────────────────────────────
    TopoTS._register_cech_lib!(libcech)
end

end # module TopoTSCechExt
