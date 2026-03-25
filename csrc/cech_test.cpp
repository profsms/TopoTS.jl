/*
 * cech_test.cpp
 *
 * Standalone smoke test for cech_core.hpp.
 * Compile and run:
 *   g++ -O2 -std=c++17 -DCECH_TEST_MAIN -o cech_test cech_test.cpp && ./cech_test
 */

#define CECH_TEST_MAIN
#include "cech_core.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg)                                       \
    do {                                                       \
        if (cond) { printf("  PASS  %s\n", msg); ++n_pass; }  \
        else      { printf("  FAIL  %s\n", msg); ++n_fail; }  \
    } while(0)

#define CHECK_NEAR(a, b, tol, msg)                             \
    CHECK(std::fabs((a)-(b)) <= (tol), msg)

int main(void) {
    printf("=== cech_core smoke tests ===\n\n");

    // ── version ─────────────────────────────────────────────────────────
    printf("version: %s\n\n", cech_version());

    // ── circumradius: single point ───────────────────────────────────────
    {
        double pts[] = {1.0, 2.0};
        int32_t v[] = {0};
        double r = cech_circumradius(pts, 1, 2, v, 1);
        CHECK_NEAR(r, 0.0, 1e-10, "circumradius(single point) == 0");
    }

    // ── circumradius: two points in R^2 ─────────────────────────────────
    {
        // (0,0) and (2,0): circumradius = 1
        double pts[] = {0.0, 0.0,  2.0, 0.0};
        int32_t v[] = {0, 1};
        double r = cech_circumradius(pts, 2, 2, v, 2);
        CHECK_NEAR(r, 1.0, 1e-9, "circumradius(2 pts, dist=2) == 1");
    }

    // ── circumradius: equilateral triangle in R^2 ────────────────────────
    {
        // Side length 2, circumradius = 2/sqrt(3) ≈ 1.1547
        double s = 2.0;
        double pts[] = {
            0.0,         0.0,
            s,           0.0,
            s / 2.0,     s * std::sqrt(3.0) / 2.0
        };
        int32_t v[] = {0, 1, 2};
        double r = cech_circumradius(pts, 3, 2, v, 3);
        double expected = s / std::sqrt(3.0);
        CHECK_NEAR(r, expected, 1e-7, "circumradius(equilateral triangle)");
    }

    // ── circumradius: unit simplex in R^3 ────────────────────────────────
    {
        // Tetrahedron with vertices at (0,0,0),(1,0,0),(0,1,0),(0,0,1)
        // Circumradius = sqrt(3)/2 ≈ 0.8660
        double pts[] = {
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        };
        int32_t v[] = {0, 1, 2, 3};
        double r = cech_circumradius(pts, 4, 3, v, 4);
        double expected = std::sqrt(3.0) / 2.0;
        CHECK_NEAR(r, expected, 1e-6, "circumradius(unit tetrahedron)");
    }

    // ── build_filtration: 4 points on unit circle ────────────────────────
    {
        double pts[] = {
            1.0, 0.0,
            0.0, 1.0,
           -1.0, 0.0,
            0.0,-1.0
        };
        CechSimplex* simplices = nullptr;
        int64_t n = 0;
        int ret = cech_build_filtration(pts, 4, 2, 1, 1e100,
                                        &simplices, &n);
        CHECK(ret == 0, "build_filtration returns 0");
        CHECK(n > 0,    "build_filtration produces simplices");

        // All 4 vertices should be present at birth=0
        int n_vert = 0;
        for (int64_t i = 0; i < n; ++i)
            if (simplices[i].dim == 0) ++n_vert;
        CHECK(n_vert == 4, "4 vertices at birth=0");

        // All edges: there are C(4,2)=6 edges
        int n_edge = 0;
        for (int64_t i = 0; i < n; ++i)
            if (simplices[i].dim == 1) ++n_edge;
        CHECK(n_edge == 6, "C(4,2)=6 edges");

        // Birth times should be non-decreasing
        bool sorted = true;
        for (int64_t i = 1; i < n; ++i)
            if (simplices[i].birth < simplices[i-1].birth - 1e-12)
                sorted = false;
        CHECK(sorted, "filtration is sorted by birth time");

        cech_free(simplices);
    }

    // ── threshold works ───────────────────────────────────────────────────
    {
        double pts[] = {0.0, 0.0,  10.0, 0.0};  // very far apart
        CechSimplex* simplices = nullptr;
        int64_t n = 0;
        // threshold = 1.0 → edge (circumradius=5) should be excluded
        cech_build_filtration(pts, 2, 2, 1, 1.0, &simplices, &n);
        int n_edge = 0;
        for (int64_t i = 0; i < n; ++i)
            if (simplices[i].dim == 1) ++n_edge;
        CHECK(n_edge == 0, "threshold excludes long edge");
        cech_free(simplices);
    }

    // ── null safety ───────────────────────────────────────────────────────
    {
        int ret = cech_build_filtration(nullptr, 0, 2, 1, 1e9, nullptr, nullptr);
        CHECK(ret != 0, "null input returns error");
    }

    // ── summary ───────────────────────────────────────────────────────────
    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
