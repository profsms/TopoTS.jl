# build_tarballs.jl — BinaryBuilder.jl recipe for CechCore_jll
#
# This is the Yggdrasil-compatible build recipe for the TopoTS Čech
# filtration C++ library.
#
# ── Quick start ───────────────────────────────────────────────────────────────
#
#   # Install BinaryBuilder once
#   ]add BinaryBuilder
#
#   # Build for all platforms (takes ~20 min, runs in Docker containers)
#   julia build_tarballs.jl --verbose
#
#   # Build for one platform only (fast, for testing)
#   julia build_tarballs.jl --verbose x86_64-linux-gnu
#
#   # Build and auto-deploy to your GitHub JLL repo
#   julia build_tarballs.jl --verbose --deploy=profsms/CechCore_jll.jl
#
# ── After building ────────────────────────────────────────────────────────────
#
#   BinaryBuilder writes a completed Artifacts.toml with real hashes.
#   Copy it to jll/CechCore_jll/Artifacts.toml (replacing the stubs).
#
# ── Yggdrasil PR ──────────────────────────────────────────────────────────────
#
#   To get CechCore_jll into the General registry:
#   1. Fork https://github.com/JuliaPackaging/Yggdrasil
#   2. Add C/CechCore/build_tarballs.jl (this file, renamed)
#   3. Open a PR — CI builds all platforms and publishes the JLL
#
# Reference: https://docs.binarybuilder.org/stable/building/

using BinaryBuilder, Pkg

# ── Package metadata ──────────────────────────────────────────────────────────

name    = "CechCore"
version = v"1.0.0"

# ── Sources ───────────────────────────────────────────────────────────────────
# BinaryBuilder downloads these into the build sandbox.
# For a real submission, point to a tagged GitHub release of your source.
# For local testing, use a local path source.

sources = [
    # Option A: GitHub release tag (use this for Yggdrasil PR)
    GitSource(
        "https://github.com/profsms/TopoTS.jl.git",
        "6230968b62a2c29cdc1e64d6016f7042e242aaef"   # git tag; create this when you tag the TopoTS release
    ),

    # Option B: local directory (for development without pushing)
    # DirectorySource("../../csrc")
]

# ── Build script ──────────────────────────────────────────────────────────────
# Runs inside a cross-compilation sandbox (musl or glibc toolchain).
# ${CXXFLAGS}, ${prefix}, ${libdir} are set by BinaryBuilder.
# Note: on Windows ${libdir} is `bin/` and the extension is `.dll`.

script = raw"""
cd ${WORKSPACE}/srcdir

# Sources land in srcdir; the csrc/ directory contains our C++ files.
# Adjust the path if using DirectorySource.
SRC="${WORKSPACE}/srcdir/TopoTS.jl/csrc"

# Confirm the source is present
ls "${SRC}/cech_core.cpp" || (echo "ERROR: cech_core.cpp not found"; exit 1)

# Run the C++ smoke tests first (native toolchain, fast)
${CXX} -O2 -std=c++17 -DCECH_TEST_MAIN \
    -o /tmp/cech_test "${SRC}/cech_test.cpp" && \
    /tmp/cech_test || (echo "ERROR: C++ tests failed"; exit 1)

# Build the shared library for the target platform
${CXX} -O3 -std=c++17 -shared -fPIC \
    ${CXXFLAGS} \
    -I"${SRC}" \
    -o "${libdir}/libcech.${dlext}" \
    "${SRC}/cech_core.cpp"

# Install the public header
install -Dm644 "${SRC}/cech_core.hpp" "${includedir}/cech_core.hpp"

# Sanity-check: the library must export cech_version
nm -D "${libdir}/libcech.${dlext}" | grep -q "cech_version" || \
    (echo "ERROR: cech_version not exported"; exit 1)

echo "Build succeeded: ${libdir}/libcech.${dlext}"
"""

# ── Supported platforms ───────────────────────────────────────────────────────
# All Tier-1 and Tier-2 Julia platforms. C++17 requires GCC ≥ 7 or Clang ≥ 5.
# BinaryBuilder's default GCC 9 shard handles all of these.

platforms = expand_cxxstring_abis(supported_platforms())

# Filter out platforms where the toolchain is known to lack C++17
# (typically old armv6 or i686 — uncomment if needed)
# filter!(p -> !( arch(p) == "i686" ), platforms)

# ── Produced products ─────────────────────────────────────────────────────────

products = [
    LibraryProduct("libcech", :libcech),
    FileProduct("include/cech_core.hpp", :cech_header),
]

# ── Dependencies ──────────────────────────────────────────────────────────────
# libcech has no runtime dependencies (pure C++17 stdlib).

dependencies = Dependency[]

# ── Run BinaryBuilder ─────────────────────────────────────────────────────────

build_tarballs(ARGS, name, version, sources, script, platforms, products,
               dependencies;
               julia_compat = "1.6",
               preferred_gcc_version = v"9")
