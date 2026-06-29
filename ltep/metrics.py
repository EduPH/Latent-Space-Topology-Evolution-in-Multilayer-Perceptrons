"""
topological_metrics.py
=======================

Quantitative metrics to replace visual inference in the MLP-topology framework.
Designed to sit alongside the existing VR/trajectory module: every function here
consumes objects that module already produces (gudhi SimplexTrees, point clouds,
`communities_by_layer`, `trajectories`).

Metric design notes (kept here on purpose, for the referee response):

  * H1 is reported as a *Betti curve* and H0 as a *scalar total persistence*.
    These are the SAME object at two levels of aggregation, not two unrelated
    metrics:  for finite bars,   TP_k = sum (d - b) = \\int_0^infty beta_k(eps) d eps.
    We keep the full curve where loop evolution is the signal (H1) and collapse
    to the scalar where only connectivity matters (H0).

  * Betti curves are NOT stable in the bottleneck sense (one bar crossing a
    threshold shifts beta_k by 1 over an interval). For cross-seed statistics we
    therefore aggregate PERSISTENCE LANDSCAPES, which live in L^p, admit a
    meaningful mean, and are 1-Lipschitz w.r.t. bottleneck distance. Use Betti
    curves for figures, landscapes for variance reporting.

  * epsilon is NOT chosen to maximise feature counts (that rewards near-diagonal
    noise and is correlated with the conclusion). It is selected from:
        (a) a significance threshold tau estimated by subsampling bootstrap
            (Fasy et al. 2014, "Confidence sets for persistence diagrams"), then
        (b) the widest scale-plateau on which the significant features coexist.
    Both are determined by the full diagram up to tau, independently of the
    quantity (H1 evolution) whose behaviour we then report.

  * The beta_1(layer, eps) heatmap summarises LAYER persistence at all scales
    with zero selection freedom. It is a pointwise Betti function over the
    (layer, scale) grid; it does NOT encode the inter-layer simplicial maps
    phi_i. The horizontal maps are captured only by MLP persistence. Do not
    present the heatmap as the true 2-parameter persistence module.
"""

import numpy as np
import gudhi as gd

try:
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
    )
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ----------------------------------------------------------------------------
# 0. Persistence diagrams from a simplex tree
# ----------------------------------------------------------------------------

def compute_persistence_diagrams(simplex_tree, max_dim=2,
                                 homology_coeff_field=2, min_persistence=0.0):
    """
    Compute persistence and return diagrams as a dict {k: (n_k, 2) array}.

    Deaths of essential classes are returned as np.inf (e.g. the single
    H0 component that never dies). Downstream functions handle infinities
    explicitly; read their docstrings for the convention used.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
    max_dim : int
        Highest homology dimension to return (inclusive).
    homology_coeff_field : int
        Prime p for coefficients in Z_p (default 2).
    min_persistence : float
        Bars with (d - b) <= min_persistence are discarded by gudhi.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping dimension -> array of (birth, death) rows.
    """
    # persistence() must be called before persistence_intervals_in_dimension()
    simplex_tree.persistence(homology_coeff_field=homology_coeff_field,
                             min_persistence=min_persistence)
    diagrams = {}
    for k in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(k)
        if intervals is None or len(intervals) == 0:
            diagrams[k] = np.empty((0, 2), dtype=float)
        else:
            diagrams[k] = np.asarray(intervals, dtype=float).reshape(-1, 2)
    return diagrams


def _finite_bars(diagram):
    """Return only bars with finite death (drops essential classes)."""
    if diagram.size == 0:
        return diagram
    mask = np.isfinite(diagram[:, 1])
    return diagram[mask]


# ----------------------------------------------------------------------------
# 1. Betti curves  (primary descriptor for H1)
# ----------------------------------------------------------------------------

def betti_curve(diagram, eps_grid):
    """
    Betti number beta_k(eps) = #{ (b, d) : b <= eps < d } on a grid.

    The half-open convention [b, d) avoids double counting at endpoints.
    Essential classes (d = inf) contribute for all eps >= b, which is correct:
    they are alive at every scale beyond birth.

    Parameters
    ----------
    diagram : (n, 2) array of (birth, death), death may be inf.
    eps_grid : 1-D array of scales (increasing).

    Returns
    -------
    np.ndarray
        Integer Betti values, one per grid point.
    """
    eps_grid = np.asarray(eps_grid, dtype=float)
    if diagram.size == 0:
        return np.zeros_like(eps_grid, dtype=int)
    births = diagram[:, 0][:, None]      # (n, 1)
    deaths = diagram[:, 1][:, None]      # (n, 1)
    alive = (births <= eps_grid[None, :]) & (eps_grid[None, :] < deaths)
    return alive.sum(axis=0).astype(int)


def make_eps_grid(simplex_trees_or_diagrams, n_points=200, pad=0.05,
                  is_diagrams=False, max_dim=2):
    """
    Build a shared scale grid covering all finite birth/death values across a
    collection of layers, so Betti curves / heatmaps are directly comparable.

    Pass either a list of SimplexTrees (is_diagrams=False) or a list of
    {k: diagram} dicts (is_diagrams=True).
    """
    finite_vals = []
    for obj in simplex_trees_or_diagrams:
        diags = obj if is_diagrams else compute_persistence_diagrams(obj, max_dim)
        for k, d in diags.items():
            fb = _finite_bars(d)
            if fb.size:
                finite_vals.append(fb.ravel())
    if not finite_vals:
        return np.linspace(0.0, 1.0, n_points)
    vals = np.concatenate(finite_vals)
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-12)
    return np.linspace(lo, hi + pad * span, n_points)


# ----------------------------------------------------------------------------
# 2. Total persistence  (scalar descriptor for H0)
# ----------------------------------------------------------------------------

def total_persistence(diagram, order=1, exclude_infinite=True, cap=None):
    """
    Total persistence  sum (d - b)^order  over the diagram.

    Convention for essential classes (d = inf):
      * exclude_infinite=True (default): drop them. Then for order=1 the value
        equals the integral of the Betti curve restricted to finite bars,
        TP_1 = \\int_0^infty beta_k^{finite}(eps) d eps.
      * cap=<float>: replace inf deaths by `cap` (e.g. max filtration value),
        then the integral identity holds over [0, cap] including the essential
        class. Use this if you report TP and the windowed Betti integral together
        and want them to agree numerically.

    `exclude_infinite=True` and `cap=None` is the standard choice; for H0, where
    there is always one essential class, this measures the persistence of the
    *merging events only*, which is usually what you want.
    """
    if diagram.size == 0:
        return 0.0
    d = diagram.copy()
    if cap is not None:
        d[~np.isfinite(d[:, 1]), 1] = cap
    if exclude_infinite:
        d = _finite_bars(d)
    if d.size == 0:
        return 0.0
    lifetimes = d[:, 1] - d[:, 0]
    return float(np.sum(lifetimes ** order))


def persistence_entropy(diagram, exclude_infinite=True):
    """
    Persistence entropy  -sum p_i log p_i,  p_i = l_i / sum l_j,  l = d - b.
    A scale-free scalar summary; reported alongside total persistence.
    """
    d = _finite_bars(diagram) if exclude_infinite else diagram
    if d.size == 0:
        return 0.0
    lifetimes = d[:, 1] - d[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-np.sum(p * np.log(p)))


# ----------------------------------------------------------------------------
# 3. Persistence landscapes  (stable cross-seed aggregation)
# ----------------------------------------------------------------------------

def persistence_landscape(diagram, t_grid, num_landscapes=5):
    """
    Discretised persistence landscape (Bubenik 2015).

    For a bar (b, d) define the tent  f(t) = max(0, min(t - b, d - t)).
    The j-th landscape lambda_j(t) is the j-th largest tent value at t.

    Infinite bars are dropped (their tent is undefined). Landscapes are stable:
    || L(D) - L(D') ||_inf <= d_bottleneck(D, D'), so pointwise means across
    seeds are well defined and theoretically grounded.

    Returns
    -------
    np.ndarray, shape (num_landscapes, len(t_grid))
    """
    t_grid = np.asarray(t_grid, dtype=float)
    d = _finite_bars(diagram)
    if d.size == 0:
        return np.zeros((num_landscapes, t_grid.size))
    b = d[:, 0][:, None]
    de = d[:, 1][:, None]
    # tent values: (n_bars, n_t)
    tents = np.maximum(0.0, np.minimum(t_grid[None, :] - b, de - t_grid[None, :]))
    # sort descending along bars axis
    tents_sorted = np.sort(tents, axis=0)[::-1, :]
    out = np.zeros((num_landscapes, t_grid.size))
    k = min(num_landscapes, tents_sorted.shape[0])
    out[:k, :] = tents_sorted[:k, :]
    return out


def average_landscapes(landscapes):
    """
    Mean of a list of landscape arrays (e.g. one per seed) on a common t_grid.
    Arrays may have different numbers of layers; they are zero-padded to the
    max so the mean is taken in the same L^p space.
    """
    if not landscapes:
        raise ValueError("empty landscape list")
    max_l = max(L.shape[0] for L in landscapes)
    n_t = landscapes[0].shape[1]
    padded = []
    for L in landscapes:
        if L.shape[1] != n_t:
            raise ValueError("landscapes must share the same t_grid")
        if L.shape[0] < max_l:
            L = np.vstack([L, np.zeros((max_l - L.shape[0], n_t))])
        padded.append(L)
    stack = np.stack(padded, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def landscape_norm(landscape, p=2):
    """L^p norm of a (discretised) landscape; a stable scalar feature."""
    return float(np.linalg.norm(landscape.ravel(), ord=p))


# ----------------------------------------------------------------------------
# 4. Trajectory / clustering label agreement (per layer)
# ----------------------------------------------------------------------------

def _purity(pred, true):
    """Weighted cluster purity: sum_c max_class |c & class| / N."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    n = len(true)
    if n == 0:
        return 0.0
    total = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if members.size:
            _, counts = np.unique(members, return_counts=True)
            total += counts.max()
    return total / n


def layerwise_label_agreement(communities_by_layer, y):
    """
    Quantify how well each layer's clustering aligns with the true labels.

    For each layer, restrict to points that (i) received a community and
    (ii) have a label, then compute NMI, ARI and purity against y.

    `coverage` reports the fraction of labelled points that were assigned a
    community at that layer (isolated points get id -1 in the source module and
    are excluded). Report it: low coverage makes the agreement scores unreliable.

    NOTE: purity -> 1 trivially on linearly separable problems, so on the toy
    concentric-circles example these scores are uninformative by construction;
    they earn their keep on harder / multiclass datasets.

    Returns
    -------
    list[dict] with keys: nmi, ari, purity, n_clusters, coverage
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for NMI/ARI metrics.")
    y = np.asarray(y)
    results = []
    for comms in communities_by_layer:
        idx = [p for p, c in comms.items() if c != -1 and p < len(y)]
        if len(idx) == 0:
            results.append(dict(nmi=np.nan, ari=np.nan, purity=np.nan,
                                n_clusters=0, coverage=0.0))
            continue
        pred = np.array([comms[p] for p in idx])
        true = y[idx]
        results.append(dict(
            nmi=float(normalized_mutual_info_score(true, pred)),
            ari=float(adjusted_rand_score(true, pred)),
            purity=float(_purity(pred, true)),
            n_clusters=int(len(np.unique(pred))),
            coverage=float(len(idx) / np.count_nonzero(np.arange(len(y)) >= 0)),
        ))
    return results


# ----------------------------------------------------------------------------
# 5. Significance threshold and epsilon selection (NOT count-maximising)
# ----------------------------------------------------------------------------

def significance_threshold_fraction(diagram, alpha=0.5):
    """
    Simple threshold: tau = alpha * (max finite persistence).
    Bars with (d - b) > tau are deemed significant. Cheap and transparent,
    but `alpha` is a free knob -- prefer the bootstrap version when feasible.
    """
    d = _finite_bars(diagram)
    if d.size == 0:
        return 0.0
    return float(alpha * np.max(d[:, 1] - d[:, 0]))


def significance_threshold_bootstrap(points, dim=1, n_boot=100, alpha=0.05,
                                     max_dimension=2, rng=None,
                                     vr_builder=None):
    """
    Subsampling-bootstrap noise floor (Fasy et al. 2014).

    Resample the point cloud with replacement B times, recompute the diagram,
    and measure bottleneck distance to the full-data diagram. The (1 - alpha)
    quantile c is a confidence half-width around the diagonal; a feature is
    significant iff its distance to the diagonal exceeds c, i.e.
        (d - b) / 2 > c   <=>   (d - b) > 2c.
    We return tau = 2c, usable directly as a persistence cutoff.

    Infinite bars are stripped before bottleneck (relevant mostly for H>=1).

    `vr_builder` must be a function (points, epsilon, max_dimension) -> SimplexTree.
    Pass your module's `compute_vietoris_rips_complex` (with a large epsilon so the
    full filtration is built). If None, a Rips complex over the full distance range
    is built via gudhi directly.
    """
    rng = np.random.default_rng(rng)
    points = np.asarray(points, dtype=float)
    n = len(points)

    def diagram_of(pts):
        if vr_builder is not None:
            # build at an epsilon large enough to capture the full filtration
            from scipy.spatial.distance import pdist
            eps = float(pdist(pts).max()) if len(pts) > 1 else 1.0
            st = vr_builder(pts, eps, max_dimension)
        else:
            rips = gd.RipsComplex(points=pts)
            st = rips.create_simplex_tree(max_dimension=max_dimension)
        diags = compute_persistence_diagrams(st, max_dim=dim)
        return _finite_bars(diags.get(dim, np.empty((0, 2))))

    full = diagram_of(points)
    dists = []
    for _ in range(n_boot):
        sample = points[rng.integers(0, n, size=n)]
        try:
            db = gd.bottleneck_distance(full, diagram_of(sample))
        except Exception:
            # gudhi can choke on empty diagrams; treat as zero distance
            db = 0.0
        dists.append(db)
    c = float(np.quantile(dists, 1.0 - alpha))
    return 2.0 * c


def select_significant_features(diagram, tau):
    """Return the sub-diagram of bars with (d - b) > tau."""
    d = _finite_bars(diagram)
    if d.size == 0:
        return d
    return d[(d[:, 1] - d[:, 0]) > tau]


def select_epsilon_plateau(eps_grid, betti_values, target_value=None, eps_min=None):
    """
    Choose epsilon as the midpoint of the WIDEST scale-interval on which the
    Betti curve is constant -- i.e. the most stable resolution -- rather than
    the scale that maximises feature count.

    If `target_value` is given (e.g. the number of significant features from the
    bootstrap), only plateaus at that exact value are considered; otherwise the
    widest plateau at any positive value is used.

    `eps_min` (optional): restrict the search to epsilon STRICTLY ABOVE this value,
    so the chosen epsilon is guaranteed to lie outside the noise/confidence band
    (used for H0, where eps_min = tau, the noise floor scale).

    Returns
    -------
    (epsilon, plateau_value, (start_eps, end_eps))  or  (None, None, None)
        if no qualifying plateau exists.
    """
    eps_grid = np.asarray(eps_grid, dtype=float)
    betti = np.asarray(betti_values)
    if eps_grid.size == 0:
        return None, None, None

    if eps_min is not None:                 # epsilon must be out of the confidence band
        keep = eps_grid > float(eps_min)
        if not keep.any():
            return None, None, None
        eps_grid = eps_grid[keep]
        betti = betti[keep]

    best = None  # (width, value, i_start, i_end)
    i = 0
    n = len(betti)
    while i < n:
        j = i
        while j + 1 < n and betti[j + 1] == betti[i]:
            j += 1
        value = betti[i]
        qualifies = (value > 0) if target_value is None else (value == target_value)
        if qualifies:
            width = eps_grid[j] - eps_grid[i]
            if best is None or width > best[0]:
                best = (width, value, i, j)
        i = j + 1

    if best is None:
        return None, None, None
    _, value, a, b = best
    return float(0.5 * (eps_grid[a] + eps_grid[b])), int(value), \
        (float(eps_grid[a]), float(eps_grid[b]))


def select_epsilon_for_layer(points, dim=1, max_dimension=2, n_grid=200,
                             use_bootstrap=True, n_boot=100, alpha=0.05,
                             frac_alpha=0.5, rng=None, vr_builder=None):
    """
    End-to-end, non-visual epsilon selector for one latent point cloud:
      1. estimate the significance threshold tau (bootstrap, or fraction),
      2. count significant dim-features -> target_value,
      3. return the midpoint of the widest Betti-`dim` plateau at that value.

    Returns a dict with the chosen epsilon and all intermediate quantities, so
    the choice is fully auditable in the paper / supplement.
    """
    points = np.asarray(points, dtype=float)
    if vr_builder is not None:
        from scipy.spatial.distance import pdist
        eps_full = float(pdist(points).max()) if len(points) > 1 else 1.0
        st = vr_builder(points, eps_full, max_dimension)
    else:
        st = gd.RipsComplex(points=points).create_simplex_tree(
            max_dimension=max_dimension)

    diags = compute_persistence_diagrams(st, max_dim=max(dim, 1))
    diagram = diags.get(dim, np.empty((0, 2)))

    if use_bootstrap:
        tau = significance_threshold_bootstrap(
            points, dim=dim, n_boot=n_boot, alpha=alpha,
            max_dimension=max_dimension, rng=rng, vr_builder=vr_builder)
    else:
        tau = significance_threshold_fraction(diagram, alpha=frac_alpha)

    sig = select_significant_features(diagram, tau)
    target = len(sig)

    eps_grid = make_eps_grid([diags], is_diagrams=True, n_points=n_grid,
                             max_dim=max(dim, 1))

    # Three-tier ladder, identical for H0 and H1 (the only dimension-specific bit is
    # how tau enters: a SCALE for H0, a PERSISTENCE for H1):
    #   tier 1 -- significant features exist (persistence > tau): epsilon = midpoint of
    #             the widest plateau where exactly those features are alive.
    #   tier 2 -- none clears tau but dim-features exist: epsilon = alive-window midpoint
    #             of the single MOST PERSISTENT feature (flagged sub-threshold).
    #   tier 3 -- no dim-feature at all: epsilon = None; the caller applies a geometric
    #             heuristic (median pairwise distance).
    if target > 0:                                       # ---- tier 1 (significant) ----
        if dim == 0:
            betti = betti_curve(diagram, eps_grid)       # tau is a scale: eps > tau
            eps, plateau_val, interval = select_epsilon_plateau(
                eps_grid, betti, target_value=target, eps_min=tau)
        else:
            sig_arr = np.asarray(sig, float).reshape(-1, 2)
            betti = betti_curve(sig_arr, eps_grid)       # significant-only curve
            eps, plateau_val, interval = select_epsilon_plateau(
                eps_grid, betti, target_value=target)
        tier = 1
    else:
        betti = betti_curve(diagram, eps_grid)
        cap = float(eps_grid[-1]) if len(eps_grid) else 0.0
        D = np.asarray(diagram, float).reshape(-1, 2)
        if D.shape[0]:                                   # ---- tier 2 (most persistent) ----
            deaths = np.where(np.isfinite(D[:, 1]), D[:, 1], cap)
            births = D[:, 0]
            pers = deaths - births
            j = int(np.argmax(pers))
            if pers[j] > 0:
                eps, interval, tier = 0.5 * (births[j] + deaths[j]), (births[j], deaths[j]), 2
            else:
                eps, interval, tier = None, None, 3
        else:                                            # ---- tier 3 (no feature) ----
            eps, interval, tier = None, None, 3
        plateau_val = 0

    return dict(epsilon=eps, tau=tau, n_significant=target, tier=tier,
                plateau_value=plateau_val, plateau_interval=interval,
                eps_grid=eps_grid, betti_curve=betti, diagram=diagram)


# ----------------------------------------------------------------------------
# 6. Bigraded Betti heatmap  (scale-free bi-persistence summary)
# ----------------------------------------------------------------------------

def betti_layer_scale_grid(simplex_trees_by_layer, dim=1, eps_grid=None,
                           n_grid=200, max_dimension=2):
    """
    beta_dim(layer, eps) over the (layer, scale) grid: one Betti curve per layer,
    stacked into a (n_layers, len(eps_grid)) array.

    This is a pointwise Betti FUNCTION of the bigraded (layer, scale) parameter.
    It visualises layer persistence at all scales with no scale-selection freedom,
    which is its whole point. It does NOT encode the inter-layer simplicial maps
    phi_i; those are captured by MLP persistence. Label it accordingly.
    """
    diags_by_layer = [compute_persistence_diagrams(st, max_dim=max(dim, max_dimension))
                      for st in simplex_trees_by_layer]
    if eps_grid is None:
        eps_grid = make_eps_grid(diags_by_layer, is_diagrams=True,
                                 n_points=n_grid, max_dim=max(dim, max_dimension))
    heat = np.vstack([betti_curve(d.get(dim, np.empty((0, 2))), eps_grid)
                      for d in diags_by_layer])
    return heat, np.asarray(eps_grid)


def plot_betti_heatmap(heat, eps_grid, dim=1, ax=None, cmap="viridis"):
    """Render the beta_dim(layer, eps) heatmap. Layers on y, scale on x."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(heat, aspect="auto", origin="lower", cmap=cmap,
                   extent=[eps_grid[0], eps_grid[-1], -0.5, heat.shape[0] - 0.5])
    ax.set_xlabel(r"scale $\varepsilon$")
    ax.set_ylabel("layer")
    ax.set_yticks(range(heat.shape[0]))
    ax.set_title(rf"$\beta_{{{dim}}}(\mathrm{{layer}}, \varepsilon)$")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(rf"$\beta_{{{dim}}}$")
    return ax


# ----------------------------------------------------------------------------
# 7. Cross-layer topological-activity curve  (the depth-saturation signal)
# ----------------------------------------------------------------------------

def topological_activity_from_mlp_barcode(mlp_intervals_by_dim, n_layers):
    """
    Per-transition activity T_i = #(features born or dying between layer i and i+1),
    computed from the MLP-persistence barcode (where 'time' = layer index).

    This is the intrinsic cross-layer change signal: it reads births/deaths off
    the barcode that already carries the inter-layer maps, so it does NOT conflate
    metric rescaling with genuine reorganisation (unlike comparing independent
    per-layer diagrams, which must be diameter-normalised first).

    Parameters
    ----------
    mlp_intervals_by_dim : dict[int, (n, 2) array]
        Birth/death indices (in LAYER units) per homology dimension, as produced
        by running persistence on the combined layer-indexed filtration.
    n_layers : int

    Returns
    -------
    np.ndarray of length (n_layers - 1): activity per transition i -> i+1.
    """
    activity = np.zeros(max(n_layers - 1, 0))
    for _, intervals in mlp_intervals_by_dim.items():
        for b, d in np.asarray(intervals, dtype=float).reshape(-1, 2):
            # a birth at layer b contributes to transition (b-1 -> b)
            if np.isfinite(b) and 1 <= int(round(b)) <= n_layers - 1:
                activity[int(round(b)) - 1] += 1
            # a death at layer d contributes to transition (d-1 -> d)
            if np.isfinite(d) and 1 <= int(round(d)) <= n_layers - 1:
                activity[int(round(d)) - 1] += 1
    return activity


def saturation_depth(activity, threshold):
    """
    First transition index beyond which activity stays <= threshold (the depth
    past which the network stops reorganising topology). Returns the 1-based
    layer count to keep, or None if activity never settles.
    """
    activity = np.asarray(activity, dtype=float)
    for i in range(len(activity)):
        if np.all(activity[i:] <= threshold):
            return i + 1  # keep layers 0..i  -> i+1 layers / i transitions done
    return None
