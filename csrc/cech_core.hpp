#pragma once
/*
 * cech_core.hpp
 *
 * Core data structures and algorithms for the Čech filtration.
 *
 * The Čech complex at scale ε contains a simplex σ if and only if
 * the smallest enclosing ball of the point set σ has radius ≤ ε.
 *
 * We expose a pure-C interface (extern "C") so Julia can call it
 * via ccall without C++ name-mangling complications.
 *
 * Algorithm
 * ---------
 * 1. For each candidate simplex σ ⊆ X (up to dim_max+1 vertices),
 *    compute the circumradius using Welzl's randomised miniball
 *    algorithm in O(d · |σ|) expected time.
 * 2. Sort all simplices by birth time (circumradius).
 * 3. Return the simplex list as a flat array of structs for Julia.
 *
 * References
 * ----------
 * Welzl, E. (1991). Smallest enclosing disks (balls and ellipsoids).
 *   New Results and New Trends in Computer Science, 359–370.
 * Edelsbrunner & Harer (2010). Computational Topology. AMS.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ── Miniball (Welzl) ─────────────────────────────────────────────────────────

/*
 * Compute the squared circumradius of the smallest enclosing ball
 * of a set of points in R^d using Welzl's algorithm.
 *
 * pts      : pointer to contiguous point data, row-major (n_pts × dim)
 * indices  : which rows of pts to use
 * boundary : points that must lie on the boundary of the ball
 * center   : output — centre of the miniball (length dim)
 *
 * Returns squared radius. Returns 0 for a single point (radius 0).
 */
static double miniball_sq(
    const double* pts, int dim,
    std::vector<int>& indices,        // points to enclose
    std::vector<int>& boundary,       // points on boundary
    double* center                    // output centre
) {
    // Base cases
    if (indices.empty() || boundary.size() == static_cast<size_t>(dim + 1)) {
        // Fit ball through boundary points
        int nb = (int)boundary.size();
        if (nb == 0) {
            std::fill(center, center + dim, 0.0);
            return 0.0;
        }
        if (nb == 1) {
            const double* p = pts + boundary[0] * dim;
            std::copy(p, p + dim, center);
            return 0.0;
        }
        // General case: circumball of nb points via linear system
        // Solve: for i=1..nb-1,  2*(p_i - p_0)^T c = |p_i|^2 - |p_0|^2
        // This is a (nb-1) x dim system; we use the pseudoinverse for dim >= nb-1.
        const double* p0 = pts + boundary[0] * dim;
        int n_eq = nb - 1;
        // For n_eq <= dim we solve by building the (n_eq x n_eq) normal equations
        // when n_eq <= dim, the system is under-determined so we pick the
        // minimum-norm solution (ball through all boundary points, smallest radius).
        std::vector<double> A(n_eq * dim), b_vec(n_eq);
        for (int i = 0; i < n_eq; ++i) {
            const double* pi = pts + boundary[i + 1] * dim;
            double bval = 0.0;
            for (int k = 0; k < dim; ++k) {
                double diff = pi[k] - p0[k];
                A[i * dim + k] = 2.0 * diff;
                bval += pi[k] * pi[k] - p0[k] * p0[k];
            }
            b_vec[i] = bval;
        }
        // Normal equations: (A A^T) lambda = b,  c = A^T lambda + p0
        std::vector<double> AAt(n_eq * n_eq, 0.0);
        for (int i = 0; i < n_eq; ++i)
            for (int j = 0; j < n_eq; ++j)
                for (int k = 0; k < dim; ++k)
                    AAt[i * n_eq + j] += A[i * dim + k] * A[j * dim + k];

        // Solve (AAt) lambda = b via Gauss elimination with pivoting
        std::vector<double> lambda(n_eq);
        std::vector<std::vector<double>> mat(n_eq, std::vector<double>(n_eq + 1));
        for (int i = 0; i < n_eq; ++i) {
            for (int j = 0; j < n_eq; ++j) mat[i][j] = AAt[i * n_eq + j];
            mat[i][n_eq] = b_vec[i];
        }
        for (int col = 0; col < n_eq; ++col) {
            int pivot = col;
            for (int row = col + 1; row < n_eq; ++row)
                if (std::fabs(mat[row][col]) > std::fabs(mat[pivot][col])) pivot = row;
            std::swap(mat[col], mat[pivot]);
            if (std::fabs(mat[col][col]) < 1e-12) continue;
            for (int row = 0; row < n_eq; ++row) {
                if (row == col) continue;
                double f = mat[row][col] / mat[col][col];
                for (int k = col; k <= n_eq; ++k) mat[row][k] -= f * mat[col][k];
            }
        }
        for (int i = 0; i < n_eq; ++i)
            lambda[i] = (std::fabs(mat[i][i]) > 1e-12) ? mat[i][i + n_eq /*augmented col*/] / mat[i][i] : 0.0;
        // Wait — augmented column is index n_eq
        for (int i = 0; i < n_eq; ++i)
            lambda[i] = (std::fabs(mat[i][i]) > 1e-12) ? mat[i][n_eq] / mat[i][i] : 0.0;

        // c = p0 + A^T lambda
        std::copy(p0, p0 + dim, center);
        for (int i = 0; i < n_eq; ++i)
            for (int k = 0; k < dim; ++k)
                center[k] += A[i * dim + k] * lambda[i];

        // squared radius = |c - p0|^2
        double r2 = 0.0;
        for (int k = 0; k < dim; ++k) {
            double d = center[k] - p0[k];
            r2 += d * d;
        }
        return r2;
    }

    // Pick last point from indices
    int p_idx = indices.back();
    indices.pop_back();

    // Recurse without p
    double r2 = miniball_sq(pts, dim, indices, boundary, center);

    // Check if p is inside the current ball
    const double* p = pts + p_idx * dim;
    double dist2 = 0.0;
    for (int k = 0; k < dim; ++k) {
        double d = p[k] - center[k];
        dist2 += d * d;
    }

    if (dist2 <= r2 * (1.0 + 1e-10)) {
        indices.push_back(p_idx);
        return r2;
    }

    // p must be on boundary
    boundary.push_back(p_idx);
    r2 = miniball_sq(pts, dim, indices, boundary, center);
    boundary.pop_back();
    indices.push_back(p_idx);
    return r2;
}

/*
 * Public entry: circumradius of the set of points given by vertex_indices.
 * pts: (n_pts × dim) row-major matrix.
 * Returns the circumradius (not squared).
 */
static double circumradius(
    const double* pts, int dim,
    const int* vertex_indices, int n_verts
) {
    std::vector<int> indices(vertex_indices, vertex_indices + n_verts);
    std::vector<int> boundary;
    std::vector<double> center(dim);

    // Randomise order for expected O(d·n) performance
    std::shuffle(indices.begin(), indices.end(), std::mt19937{42});

    double r2 = miniball_sq(pts, dim, indices, boundary, center.data());
    return std::sqrt(std::max(0.0, r2));
}

// ── Simplex struct (plain-C layout for ccall) ────────────────────────────────

#define TOPOTSS_MAX_VERTS 8   /* max simplex dimension = 7, i.e. k+1 ≤ 8 */

struct CechSimplex {
    int32_t  verts[TOPOTSS_MAX_VERTS];  /* vertex indices, -1 = unused     */
    int32_t  dim;                        /* simplex dimension (0 = vertex)  */
    double   birth;                      /* circumradius                    */
};

// ── Filtration builder ───────────────────────────────────────────────────────

/*
 * Build the full Čech filtration up to dimension dim_max.
 *
 * For n points in R^d:
 *   - All k-subsets of {0,…,n-1} for k = 1, …, dim_max+1 are considered.
 *   - Each subset's birth time = circumradius of the minimal enclosing ball.
 *   - Subsets with birth > threshold are excluded.
 *
 * Output
 * ------
 * out_simplices : caller-allocated array of CechSimplex, length *out_n.
 *                 Caller must free with cech_free().
 * out_n         : number of simplices written.
 *
 * Returns 0 on success, nonzero on error.
 */
extern "C" int cech_build_filtration(
    const double* pts,      /* (n_pts × dim) row-major               */
    int32_t n_pts,
    int32_t dim,
    int32_t dim_max,        /* max simplex dimension                 */
    double  threshold,      /* max birth time (Inf = no threshold)   */
    CechSimplex** out_simplices,
    int64_t* out_n
) {
    if (!pts || n_pts <= 0 || dim <= 0 || dim_max < 0 || !out_simplices || !out_n)
        return -1;

    int max_k = std::min(dim_max + 1, n_pts - 1);  /* max simplex size - 1 */
    std::vector<CechSimplex> result;
    result.reserve(n_pts * (dim_max + 1));

    // ── 0-simplices (vertices, birth = 0) ─────────────────────────────────
    for (int32_t i = 0; i < n_pts; ++i) {
        CechSimplex s;
        std::fill(std::begin(s.verts), std::end(s.verts), -1);
        s.verts[0] = i;
        s.dim      = 0;
        s.birth    = 0.0;
        result.push_back(s);
    }

    // ── k-simplices for k = 1 … dim_max ───────────────────────────────────
    std::vector<int32_t> combo;
    for (int k = 1; k <= max_k; ++k) {
        // Enumerate all (k+1)-subsets of {0,…,n_pts-1}
        combo.resize(k + 1);
        std::iota(combo.begin(), combo.end(), 0);

        while (true) {
            // Compute circumradius
            double r = circumradius(pts, dim, combo.data(), k + 1);

            if (r <= threshold + 1e-12) {
                CechSimplex s;
                std::fill(std::begin(s.verts), std::end(s.verts), -1);
                for (int j = 0; j <= k; ++j) s.verts[j] = combo[j];
                s.dim   = k;
                s.birth = r;
                result.push_back(s);
            }

            // Advance to next combination (Knuth's algorithm)
            int i = k;
            while (i >= 0 && combo[i] == n_pts - k - 1 + i) --i;
            if (i < 0) break;
            ++combo[i];
            for (int j = i + 1; j <= k; ++j) combo[j] = combo[j - 1] + 1;
        }
    }

    // ── Sort by birth time (stable within same birth, by dim then lex) ────
    std::stable_sort(result.begin(), result.end(),
        [](const CechSimplex& a, const CechSimplex& b) {
            if (a.birth != b.birth) return a.birth < b.birth;
            if (a.dim   != b.dim)   return a.dim   < b.dim;
            for (int i = 0; i <= std::min(a.dim, b.dim); ++i)
                if (a.verts[i] != b.verts[i]) return a.verts[i] < b.verts[i];
            return false;
        });

    // ── Copy to heap-allocated output ─────────────────────────────────────
    *out_n = (int64_t)result.size();
    *out_simplices = (CechSimplex*)std::malloc(result.size() * sizeof(CechSimplex));
    if (!*out_simplices) return -2;
    std::memcpy(*out_simplices, result.data(), result.size() * sizeof(CechSimplex));
    return 0;
}

/*
 * Free memory allocated by cech_build_filtration.
 */
extern "C" void cech_free(CechSimplex* ptr) {
    std::free(ptr);
}

/*
 * Query functions: return metadata about a single simplex.
 * Useful for validating the Julia wrapper.
 */
extern "C" double cech_circumradius(
    const double* pts, int32_t n_pts, int32_t dim,
    const int32_t* verts, int32_t n_verts
) {
    (void)n_pts;
    return circumradius(pts, dim, verts, n_verts);
}

/*
 * Version string for runtime verification.
 */
extern "C" const char* cech_version(void) {
    return "TopoTS-CechCore-1.0.0";
}
