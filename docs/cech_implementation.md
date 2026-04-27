# The Čech filtration in TopoTS.jl: implementation notes

## Overview

The Čech complex at scale ε contains a simplex σ if and only if the
**smallest enclosing ball** (miniball) of the vertex set σ has radius ≤ ε.
This is geometrically tighter than the Vietoris–Rips complex (which only
requires pairwise distances) and produces the same persistent homology as
the nerve of a ball cover — the setting of the Nerve Theorem.

No existing Julia package implements the Čech filtration natively.
TopoTS provides one via a C++17 shared library (`libcech`) called from Julia
through `ccall`, with zero Python or external tool dependencies.

---

## Algorithm

### 1. Miniball computation (Welzl's algorithm)

The core operation is computing the circumradius of a finite point set
in ℝᵈ. We implement Welzl's randomised algorithm (1991), which solves
this in expected O(d · n) time for n points in ℝᵈ.

The key insight is the following recursive structure. Let P be a set of
points and B a set of "boundary" points that must lie on the sphere.
The miniball of (P, B) satisfies:

- If P is empty or |B| = d+1, return the unique circumsphere of B.
- Pick a random point p from P.
- Recursively compute miniball of (P \ {p}, B).
- If p lies inside this ball, return it.
- Otherwise, p must lie on the boundary: return miniball of (P \ {p}, B ∪ {p}).

The circumsphere through |B| ≤ d+1 points is computed by solving the
normal equations of the linear system:

    2(pᵢ - p₀)ᵀ c = |pᵢ|² - |p₀|²,   i = 1, …, |B|-1

via Gaussian elimination with partial pivoting. This is numerically stable
for the simplex sizes encountered in TDA (typically |σ| ≤ d+2 ≤ 6).

### 2. Filtration construction

For n points in ℝᵈ, all k-subsets of {0,…,n-1} for k = 1,…,dim_max+1 are
enumerated using Knuth's combinatorial algorithm. Each subset's birth time
is its circumradius; subsets with birth > threshold are excluded.

The simplex list is sorted stably by birth time, then by dimension, then
lexicographically — matching the ordering expected by Ripserer's boundary
matrix reduction.

**Complexity:** O(C(n, dim_max+1) · d · (dim_max+1)) in the worst case.
In practice, the threshold eliminates most high-diameter simplices early.
For the typical setting in time-series TDA (n ≤ 300, dim_max = 1, d = 2–3)
this runs in milliseconds.

### 3. Boundary reduction (delegated to Ripserer.jl)

Once the simplex list with birth times is constructed by libcech, the
Julia wrapper `CechFiltration` implements Ripserer.jl's `AbstractFiltration`
interface. Ripserer then handles:

- Boundary matrix construction
- Column reduction (cohomology algorithm with apparent-pair optimisation)
- Persistence pairing → diagram

This design separates concerns cleanly: the C++ library owns only the
geometry (miniball computation, simplex enumeration, sorting); Ripserer
owns the algebra (homology computation). Neither needs to know about the
other's internals.

---

## Čech vs Rips: when to use which

In Euclidean space ℝᵈ, the two complexes satisfy the **2-interleaving**:

    Čech_ε  ⊆  Rips_ε  ⊆  Čech_{√2·ε}

This means their persistence diagrams are within a factor of √2 in the
bottleneck distance. Concretely:

| Property               | Čech                        | Rips                        |
|------------------------|-----------------------------|-----------------------------|
| Geometric criterion    | circumradius ≤ ε            | all pairwise dists ≤ 2ε     |
| Nerve theorem applies  | Yes (exact nerve of balls)  | No (approximation)          |
| Scale factor           | 1× (exact)                  | up to √2× inflation         |
| Computation cost       | O(n^{d+1}) miniball calls   | O(n²) distance comparisons  |
| Practical limit        | n ≤ ~400, dim ≤ 3           | n ≤ ~5000, any dim          |

**Recommendation for time-series TDA:**

- Use `:rips` (default) for exploratory analysis and large windows.
- Use `:cech` when geometric exactness matters: e.g. when comparing
  birth/death coordinates across filtrations, or when dim_max ≥ 2 and
  the exact circumradius is needed for a Čech-specific stability result.
- Use `:alpha` for 2D or 3D embeddings — it is as tight as Čech but
  O(n log n) via the Delaunay triangulation.
- Use `:edge_collapsed` as a drop-in replacement for `:rips` when the
  point cloud is large (n > 500) and speed matters.

---

## C++ / Julia interface

### Struct layout

The C++ `CechSimplex` struct has the following memory layout, which Julia's
`ccall` must match exactly:

```c
struct CechSimplex {
    int32_t  verts[8];   //  0–31  (32 bytes): vertex indices, -1 = unused
    int32_t  dim;        // 32–35  ( 4 bytes): simplex dimension
    int32_t  _pad;       // 36–39  ( 4 bytes): alignment padding
    double   birth;      // 40–47  ( 8 bytes): circumradius
};                       // total: 48 bytes
```

Julia mirrors this as:

```julia
struct CechSimplexC
    verts :: NTuple{8, Int32}
    dim   :: Int32
    _pad  :: Int32
    birth :: Float64
end
# sizeof(CechSimplexC) == 48  ✓
```

The padding field is required because C compilers align `double` members
to 8-byte boundaries. Without the explicit `_pad`, the Julia struct would
be 44 bytes and every `birth` field would be read from the wrong address.

### Memory management

`cech_build_filtration` heap-allocates the output array via `malloc` and
returns a pointer. Julia calls `cech_free` immediately after copying the
data into a Julia `Vector`, so there is no memory leak and no need for a
Julia finalizer.

### Row-major vs column-major

Julia stores matrices in column-major order; C expects row-major.
The bridge function converts explicitly:

```julia
pts_flat = Float64[pts[i,j] for i in 1:n_pts, j in 1:d] |> vec
```

This is a one-time O(n·d) copy at the start of each `persistent_homology`
call, negligible compared to the O(C(n,k)) filtration construction.

### Vertex index convention

C uses 0-based indices; Julia uses 1-based. The wrapper adds 1 to all
vertex indices when converting `CechSimplexC` to Ripserer `Simplex` objects:

```julia
verts = Tuple(cs.verts[i] + Int32(1) for i in 1:k+1)
```

---

## Building libcech

### Automatic (recommended)

```julia
using Pkg
Pkg.build("TopoTS")   # runs deps/build.jl
```

Requires g++ ≥ 9 or clang++ ≥ 10 with C++17 support.

### Manual

```bash
cd <package_root>/csrc

# Linux
make linux

# macOS (universal binary x86_64 + arm64)
make macos

# Windows (MinGW)
make windows

# Run C++ smoke tests first
make test
```

The shared library is placed in `deps/lib/libcech.{so,dylib,dll}`.

### Environment variable override

Set `TOPOTS_LIBCECH` to point to a custom build location:

```julia
ENV["TOPOTS_LIBCECH"] = "/path/to/libcech.so"
using TopoTS
```

---

## References

- Welzl, E. (1991). Smallest enclosing disks (balls and ellipsoids).
  *New Results and New Trends in Computer Science*, LNCS 555, 359–370.
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*.
  AMS. Chapter III (Čech and Rips complexes), Chapter VII (persistence).
- de Silva, V., & Ghrist, R. (2007). Coverage in sensor networks via persistent
  homology. *Algebraic & Geometric Topology*, 7, 339–358.
  (Establishes the Nerve Theorem connection for Čech.)
- Bauer, U. (2021). Ripser: efficient computation of Vietoris–Rips persistence
  barcodes. *Journal of Applied and Computational Topology*, 5(3), 391–423.
