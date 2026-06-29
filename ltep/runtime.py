"""
runtime_sensitivity.py
======================

Runtime (complexity / scalability) and sensitivity analyses for the MLP-topology
framework. Addresses the Associate Editor's request for "runtime and sensitivity
analyses" and Reviewer 1's points 3 (computational complexity, scalability,
sparsification sensitivity) and 6 (epsilon sensitivity, seed robustness,
reproducibility of scale choice).

This module is deliberately decoupled from the exact pipeline signature: the
sweep functions take a *callable* that runs your pipeline for a given input and
returns a small result dict. You supply the wiring; this module supplies the
measurement, the statistics, and the plots.

----------------------------------------------------------------------------
MEASUREMENT CAVEATS (read before reporting numbers in the paper)
----------------------------------------------------------------------------
* Wall-clock is measured with time.perf_counter (monotonic, high-resolution).
  Always report the MEDIAN over repeats, not a single run; warm up once first.
  Pin threads if you want reproducible timings (e.g. OMP_NUM_THREADS=1), and say
  so in the paper -- gudhi/numpy may use BLAS/OpenMP threads otherwise.

* Memory: tracemalloc only tracks *Python-level* allocations. gudhi builds the
  simplex tree in C++; that heap is largely invisible to tracemalloc. We therefore
  ALSO report process resident-set high-water mark via resource.getrusage, which
  does include C++ allocations but (i) is a high-water mark that never decreases
  within a process, so it must be measured in a FRESH SUBPROCESS per configuration
  to be meaningful, and (ii) has platform-dependent units (KiB on Linux, bytes on
  macOS) -- handled below. For trustworthy memory curves, use
  `run_in_subprocess=True`.

* Empirical complexity exponents are LEAST-SQUARES FITS over the tested range,
  not asymptotic proofs. The dominant cost term changes regime (distance matrix
  O(n^2) memory; Rips expansion can be exponential in the worst case but is
  bounded by the clique count in practice; boundary-matrix reduction is O(M^3)
  worst case in the number of simplices M, near-linear empirically). Report the
  fit with its R^2 and the n-range, and state these as empirical, regime-limited
  observations.

----------------------------------------------------------------------------
PIPELINE CONTRACT
----------------------------------------------------------------------------
Several functions expect a callable `pipeline_fn(**params) -> dict` returning a
subset of:
    {
      "simplex_trees_by_layer": list[gudhi.SimplexTree],
      "mlp_intervals_by_dim":   dict[int, (k,2) array]   # layer-indexed barcode
      "n_layers":               int,
      "diagrams_by_layer":      list[dict[int, (k,2) array]],   # optional
    }
You decide what `params` are (epsilon sequence, sparsification threshold, seed,
subsample size, ...). The sweep helpers pass the swept parameter through.
"""

import os
import sys
import time
import gc
import tracemalloc
from contextlib import contextmanager

import numpy as np

try:
    import gudhi as gd
    _HAS_GUDHI = True
except ImportError:  # pragma: no cover
    _HAS_GUDHI = False

# Reuse metric primitives where available.
try:
    from .metrics import (
        compute_persistence_diagrams,
        betti_curve,
        persistence_landscape,
        make_eps_grid,
        _finite_bars,
        topological_activity_from_mlp_barcode,
        saturation_depth,
    )
    _HAS_METRICS = True
except ImportError:  # pragma: no cover
    _HAS_METRICS = False


# ============================================================================
# 0. Low-level timing / memory primitives
# ============================================================================

def _maxrss_bytes():
    """
    Process resident-set high-water mark in bytes, or None if unavailable.
    Handles the Linux (KiB) vs macOS (bytes) unit difference in ru_maxrss.
    """
    try:
        import resource
    except ImportError:  # Windows
        return None
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru)            # bytes on macOS
    return int(ru) * 1024         # KiB on Linux/BSD


@contextmanager
def measure(track_python_mem=True):
    """
    Context manager yielding a mutable dict that, on exit, holds:
        seconds          : wall-clock (perf_counter)
        py_peak_bytes    : peak Python allocation during the block (tracemalloc)
        rss_after_bytes  : process RSS high-water AFTER the block (resource)
    Note rss_after is a high-water mark; only meaningful per-process (see caveats).
    """
    rec = {}
    gc.collect()
    if track_python_mem:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        rec["seconds"] = time.perf_counter() - t0
        if track_python_mem:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rec["py_peak_bytes"] = int(peak)
        else:
            rec["py_peak_bytes"] = None
        rec["rss_after_bytes"] = _maxrss_bytes()


def timeit_median(func, *args, repeats=5, warmup=1, **kwargs):
    """
    Median wall-clock of `func` over `repeats` runs (after `warmup` discarded
    runs). Returns (result_of_last_run, dict(median, mean, std, times)).
    """
    for _ in range(max(warmup, 0)):
        func(*args, **kwargs)
    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times)
    return result, dict(median=float(np.median(times)),
                        mean=float(times.mean()),
                        std=float(times.std()),
                        times=times)


# ============================================================================
# 1. Per-stage profiling of one configuration
# ============================================================================

def profile_pipeline_stages(points_by_layer,
                            vr_builder,
                            epsilons,
                            max_dimension=2,
                            persistence_fn=None,
                            pullback_fn=None,
                            trajectory_fn=None,
                            repeats=3):
    """
    Time (and Python-memory) each stage of the pipeline separately for ONE input,
    so the paper can show where cost concentrates rather than only a total.

    Stages timed (those whose callable is provided):
      - 'vr_build'      : building the per-layer VR complexes
      - 'persistence'   : computing persistence diagrams of each layer
      - 'pullback'      : building the pullback / layer-wise combinatorial complexes
      - 'trajectory'    : computing trajectories

    Parameters
    ----------
    points_by_layer : list[np.ndarray]
    vr_builder      : fn(points, epsilon, max_dimension) -> SimplexTree
    epsilons        : list[float], one per layer
    persistence_fn  : fn(SimplexTree) -> diagrams   (default: compute_persistence_diagrams)
    pullback_fn     : fn(points_by_layer, epsilons, simplex_trees) -> list[SimplexTree] or None
    trajectory_fn   : fn(points_by_layer, simplex_trees, epsilons) -> any or None

    Returns
    -------
    dict[str, dict]  stage -> timing record (median seconds, peak python bytes)
    """
    if persistence_fn is None:
        if not _HAS_METRICS:
            raise ImportError("topological_metrics not importable; pass persistence_fn.")
        persistence_fn = lambda st: compute_persistence_diagrams(st, max_dim=max_dimension)

    report = {}

    def build_all():
        return [vr_builder(P, e, max_dimension)
                for P, e in zip(points_by_layer, epsilons)]

    sts, t = timeit_median(build_all, repeats=repeats, warmup=1)
    report["vr_build"] = t

    def persist_all():
        return [persistence_fn(st) for st in sts]
    _, t = timeit_median(persist_all, repeats=repeats, warmup=1)
    report["persistence"] = t

    if pullback_fn is not None:
        def pull():
            return pullback_fn(points_by_layer, epsilons, sts)
        _, t = timeit_median(pull, repeats=repeats, warmup=1)
        report["pullback"] = t

    if trajectory_fn is not None:
        def traj():
            return trajectory_fn(points_by_layer, sts, epsilons)
        _, t = timeit_median(traj, repeats=repeats, warmup=1)
        report["trajectory"] = t

    # also record simplex counts -- the real driver of persistence cost
    report["_simplex_counts"] = [int(st.num_simplices()) for st in sts]
    return report


# ============================================================================
# 2. Scalability sweeps:  cost vs n, vs latent dimension, vs depth
# ============================================================================

def _empirical_exponent(x, y):
    """
    Least-squares slope of log(y) vs log(x) (the empirical complexity exponent),
    with R^2. Returns (exponent, r2). Strictly an in-range fit; see caveats.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope = coef[0]
    pred = A @ coef
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def scaling_sweep_samples(points, vr_builder, epsilon, sizes,
                          max_dimension=2, repeats=3, seed=0,
                          run_in_subprocess=False):
    """
    Cost vs sample count n: subsample the point cloud to each size in `sizes`,
    time VR build + persistence, record simplex count and (optionally) RSS.

    Reports an empirical exponent for time-vs-n AND for time-vs-#simplices; the
    latter is the more honest scaling variable, since persistence cost is driven
    by the number of simplices, not directly by n.

    `run_in_subprocess=True` measures peak RSS reliably (fresh process per size);
    requires this module to be importable by name in the child. Without it, RSS
    high-water marks across sizes are not comparable -- only Python peaks are.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, float)
    rows = []
    for n in sizes:
        n = int(min(n, len(points)))
        idx = rng.choice(len(points), size=n, replace=False)
        P = points[idx]
        if run_in_subprocess:
            rec = _measure_build_persist_subprocess(P, epsilon, max_dimension)
        else:
            with measure() as rec:
                st = vr_builder(P, epsilon, max_dimension)
                if _HAS_METRICS:
                    compute_persistence_diagrams(st, max_dim=max_dimension)
                else:
                    st.persistence()
                rec["num_simplices"] = int(st.num_simplices())
            # median timing on top of the single measured run
            _, t = timeit_median(
                lambda: (lambda s: (s.persistence(), None)[1])(
                    vr_builder(P, epsilon, max_dimension)),
                repeats=repeats, warmup=1)
            rec["seconds"] = t["median"]
        rec["n"] = n
        rows.append(rec)
    times = [r["seconds"] for r in rows]
    ns = [r["n"] for r in rows]
    msimp = [r.get("num_simplices", np.nan) for r in rows]
    exp_n, r2_n = _empirical_exponent(ns, times)
    exp_s, r2_s = _empirical_exponent(msimp, times)
    return dict(rows=rows,
                exponent_time_vs_n=(exp_n, r2_n),
                exponent_time_vs_simplices=(exp_s, r2_s),
                n_range=(min(ns), max(ns)))


def scaling_sweep_dimension(make_points_of_dim, vr_builder, epsilon, dims,
                            n_points, max_dimension=2, repeats=3):
    """
    Cost vs latent dimension d. `make_points_of_dim(d, n_points) -> (n,d) array`
    lets you either (a) take real latent reps padded/projected to dimension d, or
    (b) generate synthetic clouds. The binding cost of d is in the O(n^2 * d)
    distance-matrix computation, not in persistence (which sees only distances);
    this sweep makes that explicit.
    """
    rows = []
    for d in dims:
        P = np.asarray(make_points_of_dim(int(d), int(n_points)), float)
        _, t = timeit_median(
            lambda: vr_builder(P, epsilon, max_dimension),
            repeats=repeats, warmup=1)
        rows.append(dict(dim=int(d), seconds=t["median"], std=t["std"]))
    exp, r2 = _empirical_exponent([r["dim"] for r in rows],
                                  [r["seconds"] for r in rows])
    return dict(rows=rows, exponent_time_vs_dim=(exp, r2))


def scaling_sweep_depth(pipeline_fn, depths, repeats=3, **fixed_params):
    """
    Cost vs network depth m. `pipeline_fn(depth=m, **fixed_params) -> result dict`
    runs the WHOLE pipeline for an m-layer net (you train/extract inside it).
    Returns total wall-clock vs depth. Depth scaling is expected ~linear in m
    for the layer-wise construction; this verifies it empirically.
    """
    rows = []
    for m in depths:
        _, t = timeit_median(lambda: pipeline_fn(depth=int(m), **fixed_params),
                             repeats=repeats, warmup=0)
        rows.append(dict(depth=int(m), seconds=t["median"], std=t["std"]))
    exp, r2 = _empirical_exponent([r["depth"] for r in rows],
                                  [r["seconds"] for r in rows])
    return dict(rows=rows, exponent_time_vs_depth=(exp, r2))


# Subprocess RSS measurement -------------------------------------------------

def _measure_build_persist_subprocess(points, epsilon, max_dimension):
    """
    Build + persist in a fresh subprocess and return its peak RSS in bytes plus
    wall-clock. Falls back to in-process measurement if subprocess spawn fails.
    The child uses gudhi's RipsComplex directly to avoid pickling user builders.
    """
    import subprocess, json, tempfile
    npy = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(npy.name, points)
    npy.close()
    code = (
        "import sys, time, json, numpy as np, gudhi as gd\n"
        "import resource\n"
        f"P = np.load(r'{npy.name}')\n"
        "t0 = time.perf_counter()\n"
        f"st = gd.RipsComplex(points=P, max_edge_length={float(epsilon)})"
        f".create_simplex_tree(max_dimension={int(max_dimension)})\n"
        "st.persistence()\n"
        "sec = time.perf_counter() - t0\n"
        "ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "rss = ru if sys.platform=='darwin' else ru*1024\n"
        "print(json.dumps({'seconds': sec, 'rss_after_bytes': int(rss),"
        " 'num_simplices': int(st.num_simplices())}))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=3600)
        rec = json.loads(out.stdout.strip().splitlines()[-1])
        rec["py_peak_bytes"] = None
        return rec
    except Exception:
        with measure() as rec:
            st = gd.RipsComplex(points=points, max_edge_length=float(epsilon)) \
                .create_simplex_tree(max_dimension=max_dimension)
            st.persistence()
            rec["num_simplices"] = int(st.num_simplices())
        return rec
    finally:
        try:
            os.unlink(npy.name)
        except OSError:
            pass


# ============================================================================
# 3. Sensitivity:  epsilon band
# ============================================================================

def _safe_bottleneck(d1, d2):
    """gudhi bottleneck on finite bars, robust to empty diagrams."""
    if not _HAS_GUDHI:
        raise ImportError("gudhi required for bottleneck distance.")
    a = _finite_bars(np.asarray(d1).reshape(-1, 2)) if _HAS_METRICS \
        else np.asarray(d1).reshape(-1, 2)
    b = _finite_bars(np.asarray(d2).reshape(-1, 2)) if _HAS_METRICS \
        else np.asarray(d2).reshape(-1, 2)
    try:
        return float(gd.bottleneck_distance(a, b))
    except Exception:
        if a.size == 0 and b.size == 0:
            return 0.0
        nonempty = a if a.size else b
        return float(np.max((nonempty[:, 1] - nonempty[:, 0]) / 2.0))


def epsilon_sensitivity(pipeline_fn, eps_sequences, activity_dim=1,
                        threshold=0.5, reference_index=0):
    """
    Stability of conclusions under the epsilon choice (Reviewer 1, point 6).

    For each epsilon sequence in `eps_sequences`, run the pipeline, derive the
    cross-layer activity curve and the saturation depth d*, and the MLP barcode
    in dimension `activity_dim`. Report:
      * the set of saturation depths across the band (should be (near-)constant);
      * bottleneck distance of each run's barcode to a reference run;
      * coefficient of variation of d*.

    A near-constant d* across a non-trivial band is the evidence that the
    qualitative conclusion does NOT depend on the (previously visual) scale pick.

    `pipeline_fn(epsilons=seq) -> {mlp_intervals_by_dim, n_layers}`.
    """
    if not _HAS_METRICS:
        raise ImportError("topological_metrics required for activity / saturation.")
    runs = []
    for seq in eps_sequences:
        res = pipeline_fn(epsilons=seq)
        act = topological_activity_from_mlp_barcode(
            res["mlp_intervals_by_dim"], res["n_layers"])
        dstar = saturation_depth(act, threshold)
        runs.append(dict(epsilons=seq, activity=act, d_star=dstar,
                         barcode=res["mlp_intervals_by_dim"].get(
                             activity_dim, np.empty((0, 2)))))
    ref = runs[reference_index]["barcode"]
    for r in runs:
        r["bottleneck_to_ref"] = _safe_bottleneck(ref, r["barcode"])
    dstars = np.array([r["d_star"] for r in runs if r["d_star"] is not None],
                      dtype=float)
    cov = float(dstars.std() / dstars.mean()) if dstars.size and dstars.mean() else 0.0
    return dict(runs=runs,
                d_star_values=sorted(set(int(d) for d in dstars)) if dstars.size else [],
                d_star_cov=cov,
                max_bottleneck_to_ref=float(max(r["bottleneck_to_ref"] for r in runs)))


# ============================================================================
# 4. Sensitivity:  sparsification threshold (runtime vs distortion)
# ============================================================================

def sparsify_delta_net(points, delta, seed=0):
    """
    Greedy delta-net: keep a maximal set of points pairwise >= delta apart;
    every removed point lies within delta of a kept one. The Hausdorff distance
    between the net and the original cloud is therefore <= delta, which (by VR
    stability) bounds the bottleneck distortion of the diagram by <= delta.

    NOTE on the manuscript's wording: the paper sparsifies on SQUARED distance
    (">= 0.05") and states the diagram changes "by at most 0.05". The correct
    bound is the covering radius in ACTUAL distance (here `delta`), not the
    squared threshold. Use this function with `delta` = actual distance and
    measure the realised distortion below; restate the claim accordingly.

    Returns (kept_indices, removed_indices).
    """
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, float)
    n = len(pts)
    order = rng.permutation(n)
    kept = []
    kept_pts = np.empty((0, pts.shape[1]))
    for i in order:
        if kept_pts.shape[0] == 0:
            kept.append(i)
            kept_pts = pts[[i]]
            continue
        if cdist(pts[[i]], kept_pts).min() >= delta:
            kept.append(i)
            kept_pts = np.vstack([kept_pts, pts[[i]]])
    kept = np.array(sorted(kept))
    removed = np.array(sorted(set(range(n)) - set(kept.tolist())))
    return kept, removed


def sparsification_sensitivity(points, vr_builder, epsilon, deltas,
                               max_dimension=2, homology_dim=1, repeats=3,
                               seed=0):
    """
    Trade-off curve: sparsification threshold delta vs (runtime, realised diagram
    distortion). Validates Reviewer 1's point 4 (characterise the approximation
    error) and the manuscript's sparsification claim EMPIRICALLY.

    For each delta:
      - build the full diagram once (delta -> 0 reference, cached);
      - build the delta-net diagram, time the build+persist;
      - measure bottleneck(full, net) in `homology_dim` -> realised distortion;
      - record net size and covering radius (the theoretical bound).

    The key plot for the paper: realised bottleneck vs delta, with the line y=x
    overlaid (distortion should stay at or below the covering radius).
    """
    if not _HAS_METRICS:
        raise ImportError("topological_metrics required.")
    from scipy.spatial.distance import cdist
    pts = np.asarray(points, float)

    st_full = vr_builder(pts, epsilon, max_dimension)
    diag_full = compute_persistence_diagrams(st_full, max_dim=max(homology_dim, 1))
    ref = diag_full.get(homology_dim, np.empty((0, 2)))

    rows = []
    for delta in deltas:
        kept, removed = sparsify_delta_net(pts, delta, seed=seed)
        sub = pts[kept]
        # covering radius: max over removed points of distance to nearest kept
        if len(removed) and len(kept):
            cover = float(cdist(pts[removed], sub).min(axis=1).max())
        else:
            cover = 0.0
        _, t = timeit_median(lambda: vr_builder(sub, epsilon, max_dimension),
                             repeats=repeats, warmup=1)
        st_sub = vr_builder(sub, epsilon, max_dimension)
        diag_sub = compute_persistence_diagrams(st_sub, max_dim=max(homology_dim, 1))
        realised = _safe_bottleneck(ref, diag_sub.get(homology_dim,
                                                       np.empty((0, 2))))
        rows.append(dict(delta=float(delta),
                         n_kept=int(len(kept)),
                         covering_radius=cover,
                         seconds=t["median"],
                         bottleneck_distortion=realised,
                         num_simplices=int(st_sub.num_simplices())))
    return dict(rows=rows, reference_n=len(pts),
                reference_simplices=int(st_full.num_simplices()))


# ============================================================================
# 5. Sensitivity:  seeds / retraining (aggregated via stable landscapes)
# ============================================================================

def seed_sensitivity(train_and_extract_fn, seeds, homology_dim=1,
                     t_grid=None, num_landscapes=5, eps_grid=None):
    """
    Robustness across random seeds / retraining (Reviewer 1, point 6).

    `train_and_extract_fn(seed) -> {diagrams_by_layer, mlp_intervals_by_dim, n_layers}`
    retrains the network from scratch with the given seed and runs the pipeline.

    For each layer we aggregate the per-seed diagrams via PERSISTENCE LANDSCAPES
    (stable -> a meaningful mean and std band), not via raw Betti curves. We also
    report the cross-seed variability of the saturation depth d*, which is the
    decision-relevant scalar.

    Returns per-layer mean/std landscapes plus d* statistics.
    """
    if not _HAS_METRICS:
        raise ImportError("topological_metrics required.")
    per_seed = [train_and_extract_fn(s) for s in seeds]
    n_layers = per_seed[0]["n_layers"]

    if t_grid is None:
        all_diags = []
        for r in per_seed:
            for d in r["diagrams_by_layer"]:
                all_diags.append(d)
        t_grid = make_eps_grid(all_diags, is_diagrams=True,
                               max_dim=max(homology_dim, 1))

    # landscape per (seed, layer); aggregate across seeds within each layer
    layer_mean, layer_std = [], []
    for layer in range(n_layers):
        Ls = []
        for r in per_seed:
            diag = r["diagrams_by_layer"][layer].get(
                homology_dim, np.empty((0, 2)))
            Ls.append(persistence_landscape(diag, t_grid,
                                            num_landscapes=num_landscapes))
        stack = np.stack(Ls, axis=0)
        layer_mean.append(stack.mean(axis=0))
        layer_std.append(stack.std(axis=0))

    dstars = []
    for r in per_seed:
        act = topological_activity_from_mlp_barcode(
            r["mlp_intervals_by_dim"], r["n_layers"])
        d = saturation_depth(act, threshold=0.5)
        if d is not None:
            dstars.append(d)
    dstars = np.asarray(dstars, float)
    return dict(t_grid=t_grid,
                layer_landscape_mean=layer_mean,
                layer_landscape_std=layer_std,
                d_star_values=dstars.tolist(),
                d_star_mode=int(np.bincount(dstars.astype(int)).argmax())
                            if dstars.size else None,
                d_star_agreement=float(np.mean(dstars == np.round(np.median(dstars))))
                                 if dstars.size else 0.0)


# ============================================================================
# 6. Empirical test of the stability theorem
# ============================================================================

def empirical_stability_test(build_diagram_from_cover_fn, base_cover,
                             perturbations, eta, homology_dim=1):
    """
    Directly test Theorem (Stability of Pullback Cover Towers): the paper PROVES
    d_b(Dgm(U), Dgm(V)) <= eta for eta-interleaved towers but never tests it.

    Given a way to turn an output-layer cover into the resulting MLP-persistence
    diagram, perturb the base cover (e.g. shift cover centres, resplit components,
    reseed clustering) and verify the realised bottleneck distance stays within
    the interleaving bound eta.

    `build_diagram_from_cover_fn(cover) -> (k,2) diagram` in `homology_dim`.
    `base_cover` -> reference cover; `perturbations` -> iterable of covers that are
    (claimed) eta-interleaved with base_cover.

    Returns per-perturbation realised bottleneck and a pass flag (<= eta within a
    small numerical tolerance). A violation means either the interleaving estimate
    eta is wrong or the construction is not as stable as claimed -- both are
    findings worth reporting honestly.
    """
    ref = np.asarray(build_diagram_from_cover_fn(base_cover)).reshape(-1, 2)
    tol = 1e-9
    rows = []
    for i, cov in enumerate(perturbations):
        d = np.asarray(build_diagram_from_cover_fn(cov)).reshape(-1, 2)
        db = _safe_bottleneck(ref, d)
        rows.append(dict(index=i, bottleneck=db, within_bound=bool(db <= eta + tol)))
    realised = [r["bottleneck"] for r in rows]
    return dict(rows=rows, eta=float(eta),
                max_realised=float(max(realised)) if realised else 0.0,
                all_within_bound=bool(all(r["within_bound"] for r in rows)),
                slack=float(eta - max(realised)) if realised else float(eta))


# ============================================================================
# 7. Reporting helpers
# ============================================================================

def runtime_table(profile_report, units="ms"):
    """Format a per-stage profile (section 1) as ordered (stage, time) rows."""
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[units]
    rows = []
    for stage, rec in profile_report.items():
        if stage.startswith("_"):
            continue
        rows.append((stage, round(rec["median"] * scale, 3),
                     round(rec["std"] * scale, 3)))
    return rows  # [(stage, median, std), ...] in chosen units


def plot_sparsification_tradeoff(spars_result, ax=None):
    """
    Two-axis plot: delta vs runtime (left) and delta vs realised bottleneck
    distortion with the y=covering_radius bound overlaid (right). This is the
    figure that answers 'characterise the approximation error'.
    """
    import matplotlib.pyplot as plt
    rows = spars_result["rows"]
    deltas = [r["delta"] for r in rows]
    secs = [r["seconds"] for r in rows]
    dist = [r["bottleneck_distortion"] for r in rows]
    cover = [r["covering_radius"] for r in rows]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(deltas, secs, "o-", color="tab:blue", label="runtime (s)")
    ax.set_xlabel(r"sparsification threshold $\delta$")
    ax.set_ylabel("runtime (s)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(deltas, dist, "s-", color="tab:red", label="realised bottleneck")
    ax2.plot(deltas, cover, "k--", alpha=0.6, label="covering radius (bound)")
    ax2.set_ylabel("bottleneck distortion", color="tab:red")
    ax.set_title("Sparsification: runtime vs realised distortion")
    return ax, ax2
