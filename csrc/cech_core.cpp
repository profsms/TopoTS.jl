/*
 * cech_core.cpp
 *
 * Compilation unit for the TopoTS Čech filtration C++ library.
 * All implementation lives in cech_core.hpp (header-only style)
 * to keep the build simple — a single .cpp → single .so.
 *
 * Build
 * -----
 *   Linux/macOS:
 *     g++ -O3 -march=native -std=c++17 -shared -fPIC \
 *         -o libcech.so cech_core.cpp
 *
 *   macOS (clang, universal):
 *     clang++ -O3 -std=c++17 -shared -fPIC \
 *         -arch x86_64 -arch arm64 \
 *         -o libcech.dylib cech_core.cpp
 *
 *   Windows (MSVC):
 *     cl /O2 /std:c++17 /LD cech_core.cpp /Fe:cech.dll
 *
 * Exported symbols (extern "C", no mangling)
 * -------------------------------------------
 *   cech_build_filtration   — main entry point
 *   cech_free               — release allocated memory
 *   cech_circumradius       — query a single circumradius
 *   cech_version            — version string
 */

#include "cech_core.hpp"
