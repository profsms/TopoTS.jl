"""
deps/build.jl  —  TopoTS local build fallback for libcech

This file is the FALLBACK build path for the Čech C++ library.
It is used when the proper JLL package (CechCore_jll) is not installed.

Execution order on `Pkg.build("TopoTS")`:
  1. Julia checks if CechCore_jll is installed in the active environment.
     If yes, the JLL provides the pre-compiled binary automatically —
     this build.jl does nothing.
  2. If CechCore_jll is NOT installed, this file compiles the library
     from source using whatever C++ compiler is available.

To use the JLL (recommended for production/sharing):
  pkg> add CechCore_jll   # available in the General registry

To build locally (development / offline use):
  julia --project deps/build.jl
  — or —
  pkg> build TopoTS
"""

using Libdl

const PKG_ROOT = dirname(@__DIR__)
const CSRC_DIR = joinpath(PKG_ROOT, "csrc")
const LIB_DIR  = joinpath(PKG_ROOT, "deps", "lib")
const LIB_EXT  = Sys.iswindows() ? "dll" : Sys.isapple() ? "dylib" : "so"
const LIB_NAME = Sys.iswindows() ? "cech.dll" : "libcech.$LIB_EXT"
const LIB_PATH = joinpath(LIB_DIR, LIB_NAME)

# ── Step 1: Check if CechCore_jll is already providing the library ────────────
function jll_available()
    try
        @eval Main begin using CechCore_jll end
        lib = Main.CechCore_jll.libcech
        return isfile(lib)
    catch
        return false
    end
end

if jll_available()
    println("CechCore_jll is installed — no local build needed.")
    println("The JLL provides pre-compiled binaries for all platforms.")
    exit(0)
end

# ── Step 2: Try to find a C++ compiler ───────────────────────────────────────
function find_cxx()
    for cxx in ("g++", "clang++", "c++")
        try
            success(pipeline(`$(cxx) --version`; stdout=devnull, stderr=devnull)) && return cxx
        catch; end
    end
    return nothing
end

cxx = find_cxx()
if isnothing(cxx)
    @warn """
    No C++ compiler found (tried g++, clang++, c++).
    The Čech filtration will not be available.

    To fix this:
      • Install a C++ compiler (apt install g++ / brew install llvm)
      • Or install the JLL:  pkg> add CechCore_jll
    """
    exit(0)
end

# ── Step 3: Check if rebuild is needed ───────────────────────────────────────
src = joinpath(CSRC_DIR, "cech_core.cpp")
isfile(src) || error("Source not found: $src")

if isfile(LIB_PATH)
    src_mtime = mtime(src)
    lib_mtime = mtime(LIB_PATH)
    if lib_mtime >= src_mtime
        println("libcech up to date at $LIB_PATH")
        exit(0)
    end
    println("Source newer than library — rebuilding...")
end

# ── Step 4: Compile ───────────────────────────────────────────────────────────
mkpath(LIB_DIR)
println("Compiling libcech with $cxx ...")

compile_cmd = if Sys.iswindows()
    `$cxx -O3 -std=c++17 -shared -DNDEBUG -o $LIB_PATH $src`
elseif Sys.isapple()
    # Try universal binary (x86_64 + arm64); fall back to native on failure
    cmd_uni = `$cxx -O3 -std=c++17 -shared -fPIC -DNDEBUG
                    -arch x86_64 -arch arm64 -o $LIB_PATH $src`
    cmd_nat = `$cxx -O3 -std=c++17 -shared -fPIC -DNDEBUG -o $LIB_PATH $src`
    try; run(cmd_uni); nothing; catch; run(cmd_nat); nothing; end
    nothing   # already ran above
else
    `$cxx -O3 -std=c++17 -shared -fPIC -DNDEBUG -o $LIB_PATH $src`
end

compile_cmd !== nothing && run(compile_cmd)

isfile(LIB_PATH) || error("Compilation appeared to succeed but $LIB_PATH not found")

kb = round(filesize(LIB_PATH) / 1024; digits=1)
println("Built: $LIB_PATH  ($kb KB)")

# ── Step 5: Verify ────────────────────────────────────────────────────────────
lib = Libdl.dlopen(LIB_PATH)
ver_sym = Libdl.dlsym(lib, :cech_version)
ver = unsafe_string(ccall(ver_sym, Ptr{Cchar}, ()))
println("Library version: $ver")
Libdl.dlclose(lib)
