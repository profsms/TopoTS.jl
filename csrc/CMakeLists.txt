cmake_minimum_required(VERSION 3.14)
project(cech_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Shared library ─────────────────────────────────────────────────────────
add_library(cech SHARED cech_core.cpp)

target_include_directories(cech PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(cech PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -O3 -Wall -Wextra -fvisibility=hidden>
    $<$<CXX_COMPILER_ID:MSVC>:
        /O2 /W3>
)

# Ensure symbols are exported on all platforms
set_target_properties(cech PROPERTIES
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    # macOS: don't prepend 'lib' on dylib — Julia looks for libcech
    PREFIX "lib"
    OUTPUT_NAME "cech"
)

# ── Install ────────────────────────────────────────────────────────────────
install(TARGETS cech
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib     # DLL on Windows goes here too
    ARCHIVE DESTINATION lib)

install(FILES cech_core.hpp DESTINATION include)

# ── Smoke test ────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build the smoke test" OFF)
if(BUILD_TESTS)
    add_executable(cech_test cech_core.cpp)
    target_compile_definitions(cech_test PRIVATE CECH_TEST_MAIN)
    target_include_directories(cech_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(cech_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2 -Wall>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>)
endif()
