#!/usr/bin/env python
# coding: utf-8
"""
pipeline.py
===========

THE single, shared policy for the topology-of-MLPs framework. Every experiment
driver (toy, cardiotocography, COIL-100, CIFAR) imports from here instead of
defining its own epsilon rule, its own pullback chain, or its own d* reading.
That is what makes "unified framework" true rather than asserted: the four
experiments become one method run at four difficulty levels, reporting the same
objects (tau, epsilon_i, the MLP-persistence barcode, the loop-resolution layers,
and d* with a stability gate).

ONE IDEA, TWO AXES, READ PER DIMENSION
--------------------------------------
The whole framework is a single significance idea -- a bootstrap confidence test
that separates genuine topological features from sampling noise -- applied on two
different filtration axes, and read separately for H0 and H1:

  * SCALE axis  -> epsilon selection.  Within one latent cloud, a bootstrap
    confidence band (Fasy et al.) gives a persistence cutoff tau; epsilon_i is the
    centre of the widest scale-interval on which the SIGNIFICANT generators of the
    relevant dimension are alive. (select_epsilon, Pillar 1.)

  * LAYER axis  -> where features are RESOLVED, and where the network goes inert.
    The MLP-persistence barcode is a monotone filtration over layer index
    (create_combined_filtration: a simplex enters at the first layer it appears).
    An H1 loop is therefore BORN where its cycle is complete with no filler and
    DIES at the layer where a filling triangle appears -- i.e. where the network
    contracts that loop. The DEATH LAYER is the signal: it tells you where a real
    loop is resolved. A loop that survives to the last layer (an essential bar) is
    the FAILURE case for H1, not a feature to keep.

    Significance on this axis is NOT a layer-lifespan threshold (an earlier version
    used "lifespan >= 2 layers", which wrongly discarded loops that die in a single
    transition -- exactly the fast reorganisations we care about). Instead a bar is
    genuine iff it RECURS across row-resamples of the data: resample rows
    (m-out-of-n), re-propagate through the fixed trained net, recompute the barcode,
    and keep the integer (birth_layer, death_layer) events that appear in at least
    (1 - alpha) of replicates. Because the layer axis is discrete, this is a clean
    frequency test over (birth, death) cells -- and a stable lifespan-1 loop death
    counts, while a drifting lifespan-2 noise bar does not.

  * H0 vs H1 mean different things (relevant_dimension, Pillar 3):
      - H1 (loops): which input loops the network resolves, and BY WHICH LAYER.
        Carried by the input/hidden layers; structurally absent at the ~1-D output.
      - H0 (components): clusters merging toward linear separability; the
        separation-settles-by layer. H0 near the output is dominated by the funnel
        collapse, so it is reported as its own signal.
    The carrier of the reorganisation signal is DATA-DEPENDENT: loop datasets
    (concentric circles, COIL rotations) are carried by H1; cluster-structured
    tabular data (e.g. cardiotocography) by H0. The pipeline reports both and lets
    the data decide -- it does not assume H1 always carries it.

PRUNING / CONVERGENCE
---------------------
d* = the last layer at which any GENUINE event (a born or a finitely-resolved
feature, over both dimensions) occurs. Layers beyond d* see no genuine topological
event -> inert / prunable. d* comes with a stability read (how much the last-event
layer wobbles across the same row-resamples); assert pruning only when stable.

PRE-COMMITTED PARAMETERS
------------------------
The constants below are fixed once and reported, never retuned after seeing an
accuracy curve. There is no free scale parameter on the layer axis: significance is
the bootstrap recurrence frequency at level (1 - ALPHA), the same ALPHA used by the
scale-axis band.

UNITS NOTE (carry into the manuscript)
--------------------------------------
Sparsification must be expressed and bounded in ACTUAL distance: the bottleneck
distortion of a delta-net is bounded by the covering radius delta, NOT by the
squared threshold. delta_net_sparsify returns kept indices for a delta given in
actual distance; restate the manuscript claim as the covering radius accordingly.

Depends on (same directory): topological_metrics.py, runtime_sensitivity.py,
VR_trajectories.py.
"""

from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from .vr import (
    compute_vietoris_rips_complex,
    get_maximal_simplices,
    vr_pullback,
    create_combined_filtration,
)
from .metrics import (
    select_epsilon_for_layer,
    compute_persistence_diagrams,
    _finite_bars,
)
from .runtime import _safe_bottleneck, sparsify_delta_net

# ----------------------------------------------------------------------------
# Pre-committed parameters (report these; do not retune)
# ----------------------------------------------------------------------------
ALPHA = 0.01            # confidence level for the scale-axis bootstrap band (tau).
                        # 0.01 (stricter) chosen for the paper: 0.05 left the H0
                        # epsilon too small (too many noise components). Significance
                        # now lives ONLY here -- the layer axis does no resampling.
N_BOOT = 100            # bootstrap replicates for the scale-axis significance band
USE_BOOTSTRAP = True    # principled band on the headline run; set False only for
                        # speed inside an inner sweep (then tau uses the fraction rule)
MAX_DIMENSION = 2       # VR expansion dimension
CONV_N_RESAMPLE = 50    # row-resamples for layer-axis significance + d* stability
                        # (>=40 needed to resolve recurrence freq near the 1-alpha
                        # threshold; 20 was too coarse and made borderline d* flicker)
CONV_SUBSAMPLE_FRAC = 0.8   # m-out-of-n subsample fraction
AGREEMENT_MIN = 0.8     # stability gate: assert pruning only if d* agreement >= this
TAU_FLOOR_FRAC = 1e-2   # tau below this fraction of the cloud diameter => degenerate
                        # layer (significance unreliable); flagged, not silently used

PARAMS = dict(ALPHA=ALPHA, N_BOOT=N_BOOT, USE_BOOTSTRAP=USE_BOOTSTRAP,
              MAX_DIMENSION=MAX_DIMENSION, CONV_N_RESAMPLE=CONV_N_RESAMPLE,
              CONV_SUBSAMPLE_FRAC=CONV_SUBSAMPLE_FRAC, AGREEMENT_MIN=AGREEMENT_MIN)


def set_alpha(a):
    """Set the scale-axis significance level at runtime (updates the global AND the
    PARAMS report). select_epsilon resolves ALPHA at call time, so this takes effect
    for every subsequent epsilon selection -- use it from an experiment's --alpha flag."""
    global ALPHA
    ALPHA = float(a)
    PARAMS["ALPHA"] = float(a)
    return ALPHA


# ============================================================================
# Pillar 3 -- the role rule (which homology dimension to read for a layer)
# ============================================================================

def relevant_dimension(Xi, is_output, max_hom_dim=1):
    """
    The homology dimension used to SELECT epsilon for one layer (scale axis).
    H0 for the output / any structurally <2-D cloud (cannot carry H1); H1 for
    loop-bearing interior layers. `max_hom_dim` caps the choice: pass 0 to force an
    H0-only analysis (e.g. cardio, where H1 is noise), 1 for the usual H0+H1. Note this
    governs epsilon selection only; on the LAYER axis the carrier of the reorganisation
    signal is decided empirically (see convergence_depth), not assumed.
    """
    X = np.asarray(Xi)
    if X.ndim == 1 or X.shape[1] < 2:
        return 0
    return min(0 if is_output else 1, max_hom_dim)


# ============================================================================
# Pillar 1 -- epsilon selection (confidence band on the scale axis)
# ============================================================================

def _median_pairwise(X):
    """Data-driven fallback scale: median pairwise distance (auditable, no hand-pick)."""
    X = np.asarray(X, float)
    if len(X) < 2:
        return 1.0
    d = pdist(X)
    return float(np.median(d)) if d.size else 1.0


def _h0_merge_scales(X):
    """
    The H0 single-linkage merge scales = MST edge weights = the finite H0 deaths.
    Sorted ascending. The largest is the merge-completion scale (Betti0 -> 1).
    """
    X = np.asarray(X, float)
    if len(X) < 2:
        return np.empty(0)
    mst = minimum_spanning_tree(squareform(pdist(X))).toarray()
    edges = mst[mst > 0]
    return np.sort(edges)


def _h0_separability_epsilon(X):
    """Epsilon for an H0 layer with NO band-significant component.

    The tier-2 rule "alive-window midpoint of the most persistent feature" is the
    right idea for H1 but degenerate for H0: the most persistent H0 component is the
    essential one (death = inf), so its alive-window midpoint reads noise-level
    fragmentation and floods the pullback tower with spurious components. The correct
    H0 analogue of "most persistent feature" is the most persistent SPLIT: the epsilon
    at the largest gap between consecutive H0 merge scales. That is the longest-lived
    multi-component configuration; it keeps the dominant separation ("separability
    information"), leaves >=2 components (never the fully-merged tail), and -- applied
    to the output anchor -- collapses the spurious extra clusters down to the genuine
    (e.g. class) split instead of letting them pull back and inflate d*.

    Returns None on a degenerate cloud (handled by the median fallback upstream).
    """
    edges = _h0_merge_scales(X)            # finite H0 deaths, ascending
    if edges.size == 0:
        return None
    if edges.size == 1:
        return float(edges[0]) * 0.5       # single merge: read just below it (2 comps)
    gaps = np.diff(edges)
    k = int(np.argmax(gaps))               # largest gap sits between edges[k], edges[k+1]
    return float(0.5 * (edges[k] + edges[k + 1]))


def _guard_h0_epsilon(X, eps, tau):
    """
    Stop the OUTPUT/H0 epsilon from landing in the fully-merged tail. The widest
    Betti0 plateau is the terminal 'one component' regime, which extends to the max
    distance, so its midpoint lands far past any structure (we saw eps ~ 20 on the
    logits). Cap eps at the largest SIGNIFICANT merge scale (the last H0 death whose
    persistence exceeds tau); beyond that no significant component is alive.

    Returns (eps_capped, capped_flag).
    """
    edges = _h0_merge_scales(X)
    if edges.size == 0 or eps is None:
        return eps, False
    sig = edges[edges > tau]
    cap = float(sig.max()) if sig.size else float(edges.max())
    if eps > cap:
        return cap, True
    return eps, False


def parse_manual_epsilons(spec, n_layers):
    """Parse a manual per-layer epsilon override (heuristic OFF).
    `spec` is a comma-separated string ('0.7,0.5,0.4,...') or a list of floats, one per
    layer, read off the layer-persistence diagrams by eye. Returns a list[float] of
    length n_layers. Raises ValueError on a length mismatch so a wrong spec fails loudly
    rather than silently misaligning layers."""
    if isinstance(spec, str):
        vals = [float(v) for v in spec.replace(" ", "").split(",") if v != ""]
    else:
        vals = [float(v) for v in spec]
    if len(vals) != n_layers:
        raise ValueError(f"manual epsilons: got {len(vals)} values for {n_layers} "
                         f"layers; supply exactly one epsilon per representation.")
    return vals


def select_epsilon(latents, use_bootstrap=USE_BOOTSTRAP, n_boot=None, alpha=None,
                   max_dimension=MAX_DIMENSION, exclude_output=True, max_hom_dim=1,
                   rng=None):
    """
    Non-visual, per-layer epsilon, selected by the SAME three-tier ladder for H0 and
    H1 (the only asymmetry is how tau enters: a SCALE for H0, a PERSISTENCE for H1).
    Per layer and per dimension k:
      tau    = bootstrap confidence-band cutoff (Fasy et al.) at level (1 - alpha);
      tier 1 = significant features (persistence > tau) exist -> eps = midpoint of the
               widest plateau where exactly those features are alive;
      tier 2 = none clears tau but dim-k features exist -> eps = alive-window midpoint
               of the single MOST PERSISTENT feature (flagged sub-threshold). This is
               what anchors the H1 pullback tower at a genuine H1 scale when no loop is
               significant -- there is NO borrowing of the H0 scale.
               H0 OVERRIDE: for components the most persistent feature is the essential
               one, so its alive-window midpoint is meaningless and reads noise-level
               fragmentation. When no H0 component clears tau we instead take the most
               persistent SPLIT -- eps at the largest H0 merge gap (h0_separability) --
               which keeps the coarsest genuine separation and stops the output anchor
               from injecting spurious components into the tower;
      tier 3 = no dim-k feature at all -> eps = median pairwise distance (geometric
               heuristic), flagged.

    `alpha` / `n_boot` default to the MODULE GLOBALS ALPHA / N_BOOT, resolved at CALL
    time (not import time), so setting pl.ALPHA / pl.N_BOOT at runtime -- e.g. from an
    experiment's --alpha flag -- takes effect here. Pass explicit values to override.

    Two further guards (both flagged, never silent):
      * degenerate: tau below TAU_FLOOR_FRAC * diameter => the cloud is near-collapsed
        and the significance test is unreliable (e.g. a saturated late layer).
      * capped: for a SIGNIFICANT (tier-1) H0 selection the eps is capped at the last
        significant merge so it cannot sit in the fully-merged tail.

    Returns
    -------
    dict(epsilons_H0=[...], epsilons_H1=[...], epsilons=<alias of H1>,
         per_layer=[dict(layer, eps_H0, eps_H1, eps_H0_used, eps_H1_used, tau_H0, tau_H1,
                         n_sig_H0, n_sig_H1, h0_tier, h1_tier, h1_significant,
                         h0_fallback, h1_fallback, capped, h0_separability, degenerate), ...])
    where h{0,1}_tier in {1,2,3}; h1_significant == (h1_tier == 1); the *_fallback flags
    mark tier 3 (geometric heuristic); h0_separability marks the H0 most-persistent-split
    fallback used when no component clears tau.
    """
    alpha = ALPHA if alpha is None else alpha       # resolve at call time
    n_boot = N_BOOT if n_boot is None else n_boot
    L = len(latents)
    eps_H0_seq, eps_H1_seq, per_layer = [], [], []
    for i, Xi in enumerate(latents):
        X = np.asarray(Xi, float)
        is_out = (i == L - 1 and exclude_output)
        diam = float(pdist(X).max()) if len(X) > 1 else 1.0

        def _layer_eps(dim):
            r = select_epsilon_for_layer(
                X, dim=dim, max_dimension=max_dimension,
                use_bootstrap=use_bootstrap, n_boot=n_boot, alpha=alpha,
                vr_builder=compute_vietoris_rips_complex, rng=rng)
            return r  # r["epsilon"] is None when no significant plateau exists

        # H0 epsilon.
        #   tier 1 (significant components AND a clean plateau): midpoint of the widest
        #     significant plateau, capped out of the fully-merged tail.
        #   otherwise (no significant component, OR significant but no clean plateau):
        #     read the COARSEST GENUINE SPLIT -- the epsilon at the largest H0 merge gap
        #     (the most persistent split). Keeps separability information instead of
        #     sub-tau fragmentation, and stops the output anchor from admitting spurious
        #     components that pull back and inflate d*. Falls through to the tier-2
        #     alive-window, then the median heuristic, on a degenerate cloud.
        r0 = _layer_eps(0)
        eps0, tau0, tier0 = r0["epsilon"], float(r0["tau"]), r0["tier"]
        n_sig0 = int(r0["n_significant"])
        capped0 = sep0 = False
        if n_sig0 > 0 and eps0 is not None:     # tier 1: significant components, clean plateau
            eps0_used, capped0 = _guard_h0_epsilon(X, eps0, tau0)
        else:                                   # no significant scale -> coarsest genuine split
            eps_sep = _h0_separability_epsilon(X)
            if eps_sep is not None:
                eps0_used, sep0 = float(eps_sep), True
            elif eps0 is not None:              # tier-2 alive-window if no split available
                eps0_used = float(eps0)
            else:                               # tier 3: degenerate cloud
                eps0_used, tier0 = _median_pairwise(X), 3
        if eps0_used is None:                   # safety net: never emit a None epsilon
            eps0_used, tier0 = _median_pairwise(X), 3

        # H1 epsilon (loop-bearing interior layers). Same ladder; NO borrow of the H0
        # scale -- tier 2 anchors on the most persistent loop's alive-window so the
        # pullback tower is anchored at a genuine H1 scale even when no loop clears tau.
        want_h1 = (max_hom_dim >= 1) and (not is_out) and (X.ndim > 1 and X.shape[1] >= 2)
        r1 = _layer_eps(1) if want_h1 else None
        eps1 = r1["epsilon"] if r1 is not None else None
        tier1 = r1["tier"] if r1 is not None else 3
        if eps1 is None:                       # tier 3 (no loop at all) -> heuristic
            eps1_used, tier1 = _median_pairwise(X), 3
        else:
            eps1_used = eps1

        # NO max-combination: keep the two scales separate so each barcode reads its
        # dimension at its OWN optimal scale (H0 components vs H1 loop alive-window).
        eps_H0_seq.append(float(eps0_used))
        eps_H1_seq.append(float(eps1_used))
        degenerate = bool(diam > 0 and tau0 < TAU_FLOOR_FRAC * diam)
        per_layer.append(dict(
            layer=i, eps_H0=eps0, eps_H1=eps1,
            eps_H0_used=float(eps0_used), eps_H1_used=float(eps1_used),
            tau_H0=tau0, tau_H1=(float(r1["tau"]) if r1 is not None else None),
            n_sig_H0=n_sig0,
            n_sig_H1=(int(r1["n_significant"]) if r1 is not None else 0),
            h0_tier=tier0, h1_tier=tier1,
            h1_significant=(tier1 == 1),
            h0_fallback=(tier0 == 3), h1_fallback=(tier1 == 3),
            capped=capped0, h0_separability=sep0, degenerate=degenerate))
    return dict(epsilons_H0=eps_H0_seq, epsilons_H1=eps_H1_seq,
                epsilons=eps_H1_seq,          # deprecated alias (prefer the explicit keys)
                per_layer=per_layer)


# ============================================================================
# Pullback tower / MLP persistence (consolidated; was duplicated per driver)
# ============================================================================

def mlp_persistence(latents, epsilons, max_dim=MAX_DIMENSION, augment_output=True,
                    exclude_output=True):
    """
    Depth-general pullback chain. Builds the last VR complex, pulls back layer
    by layer to the input, assembles the layer-indexed combined filtration, and
    returns the MLP-persistence barcode (births/deaths in LAYER units).

    latents  : [X0, X1, ..., X_{L-1}], output last.
    epsilons : list of length L (one scale per representation).

    exclude_output=True (default): the last layer is a ~1-D OUTPUT that cannot carry
      loops, so H1 is taken over layers 0..L-2 (the loop-bearing representations).
      Correct for softmax/scalar heads (CIFAR, the COIL classifier/regressor).
    exclude_output=False: the last layer is a genuine loop-bearing representation
      (e.g. an autoencoder bottleneck), so H1 is taken over ALL layers 0..L-1; the
      tower is anchored at a last complex whose loop is ALIVE (its epsilon must be an
      H1 scale), so the cycle threads the pullback instead of being filled at anchor.

    NOTE on augment_output: appends a thresholded label column to the output to
    distinguish the two sides of a (binary) decision boundary. Use augment_output=
    False for multiclass softmax and for non-output last layers (bottlenecks).
    """
    L = len(latents)
    assert len(epsilons) == L, "need one epsilon per representation"

    out = latents[-1]
    if augment_output:
        col0 = (np.asarray(out).reshape(len(out), -1)[:, :1] > 0.5).astype(float)
        out = np.c_[np.asarray(out, float).reshape(len(out), -1), col0]
    out = np.asarray(out, float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)

    trees = [None] * L
    trees[-1] = compute_vietoris_rips_complex(out, epsilons[-1], max_dimension=1)
    ms = get_maximal_simplices(trees[-1], epsilons[-1])
    for i in range(L - 2, -1, -1):
        ki = vr_pullback(np.asarray(latents[i], float), epsilons[i], ms,
                         max_dimension=max_dim)
        trees[i] = ki
        if i > 0:
            ms = get_maximal_simplices(ki, epsilons[i])
    # expand the layers that participate in H1 to admit triangles (loop fillings)
    h1_trees = trees if not exclude_output else trees[:-1]
    for t in h1_trees:
        t.expansion(max_dim)

    c0 = create_combined_filtration(trees)
    c0.compute_persistence()
    iv0 = np.asarray(c0.persistence_intervals_in_dimension(0)).reshape(-1, 2)

    min_layers = 2 if not exclude_output else 3
    if len(h1_trees) >= 2 and L >= min_layers:
        c1 = create_combined_filtration(h1_trees)
        c1.compute_persistence()
        iv1 = np.asarray(c1.persistence_intervals_in_dimension(1)).reshape(-1, 2)
    else:
        iv1 = np.empty((0, 2))

    return dict(mlp_intervals_by_dim={0: iv0, 1: iv1},
                n_layers=L, pullback_trees=trees)


def _cap_and_bars(intervals, cap):
    """Cap essential (infinite-death) bars at `cap` layers, return (n, 2) array."""
    arr = np.asarray(intervals, float).reshape(-1, 2)
    if arr.size == 0:
        return np.empty((0, 2))
    arr = arr.copy()
    arr[~np.isfinite(arr[:, 1]), 1] = float(cap)
    return arr


def _caps(L, exclude_output=True):
    """Layer-index caps per dimension. H0 spans 0..L-1. H1 spans 0..L-2 when the
    last layer is a true (~1-D) OUTPUT that cannot carry a loop; when the last layer
    is a genuine loop-bearing representation (e.g. an autoencoder bottleneck),
    exclude_output=False and H1 spans the full 0..L-1."""
    return {0: L - 1, 1: max(L - (2 if exclude_output else 1), 0)}


# ============================================================================
# Pillar 2 -- layer-axis significance via bootstrap recurrence + event reading
# ============================================================================

def _event_cells(replicate_bars, cap, alpha):
    """
    Bootstrap-recurrence significance over the DISCRETE (birth_layer, death_layer)
    grid. A cell is genuine iff at least one bar lands in it in >= (1 - alpha) of
    replicates. No lifespan filter -- a stable lifespan-1 death counts.

    replicate_bars : list over replicates of (n_k, 2) capped bar arrays for one dim.
    Returns a list of dict(birth, death, frequency, mean_count, essential), sorted.
    """
    R = len(replicate_bars)
    if R == 0:
        return []
    presence, counts = Counter(), Counter()
    for bars in replicate_bars:
        arr = np.asarray(bars, float).reshape(-1, 2)
        seen = set()
        for b, d in arr:
            cell = (int(round(b)), int(round(d)))
            counts[cell] += 1
            if cell not in seen:
                presence[cell] += 1
                seen.add(cell)
    thr = 1.0 - alpha
    cells = []
    for cell, p in presence.items():
        freq = p / R
        b, d = cell
        cells.append(dict(birth=b, death=d, frequency=float(freq),
                          mean_count=float(counts[cell] / R),
                          essential=(d >= cap), kept=(freq >= thr)))
    cells.sort(key=lambda c: (c["birth"], c["death"]))
    return cells


def _summarise_dim(cap, events):
    """
    Turn a list of event dicts (each with birth, death, essential, and -- when
    available -- frequency/mean_count) into the per-dimension report.

      resolution_events : (birth -> death) for features that ACTUALLY die (d < cap)
      resolved_by       : last layer at which a genuine feature is resolved
      unresolved        : count of genuine features that survive to the end
                          (essential). For H1 this is the FAILURE flag.
      onset_layers      : sorted distinct birth layers
    """
    finite = [e for e in events if not e["essential"]]
    essential = [e for e in events if e["essential"]]
    resolved_by = max((e["death"] for e in finite), default=None)
    return dict(
        n_events=len(events),
        resolution_events=[(e["birth"], e["death"]) for e in finite],
        resolved_by=resolved_by,
        unresolved=len(essential),
        unresolved_births=sorted(e["birth"] for e in essential),
        onset_layers=sorted({e["birth"] for e in events}),
        events=events)


def _dstar_from_dims(per_dim, L):
    """d* = last layer with any genuine birth or finite (real) death, over all dims."""
    last = []
    for s in per_dim.values():
        for e in s["events"]:
            last.append(e["birth"])
            if not e["essential"]:
                last.append(e["death"])
    d = int(max(last)) if last else None
    inert = list(range(d + 1, L)) if d is not None else []
    return d, inert


def _active_transitions(per_dim, L):
    """Layer indices i in 1..L-1 whose incoming transition (i-1 -> i) does
    topological WORK: a feature is BORN at i (i>=1) or has a FINITE death (merge /
    resolution) at i, in any tracked degree. The complementary transitions are
    inert -- the induced homology map is an isomorphism, so the two layers carry the
    same tracked topology."""
    active = set()
    for s in per_dim.values():
        for e in s["events"]:
            b, d = int(e["birth"]), int(e["death"])
            if b >= 1:
                active.add(b)
            if not e["essential"] and d >= 1:
                active.add(d)
    return sorted(a for a in active if 1 <= a <= L - 1)


def collapsible_blocks(per_dim, L, protect_output=True):
    """Reformulated convergence: partition layers 0..L-1 into topological STAGES
    separated by the active transitions, and report which stages can be COLLAPSED.

    Layers inside a stage are joined only by inert (isomorphism) transitions, so they
    carry identical tracked homology and collapse to a single representative; a stage
    of length >= 2 is a `collapsible block`. Unlike the tail-only d*, blocks may be
    INNER (a layer that merges nothing between two layers that do). Returns:

      active_transitions : layers where work happens (stage boundaries)
      stages             : (start, end) inclusive layer ranges, contiguous, covering 0..L-1
      collapsible_blocks : the stages with end > start (the prunable runs)
      redundancy         : sum(end-start) = removable layers = (L-1) - #active
      collapsed_depth    : number of stages (depth after collapsing every block)

    protect_output keeps the final ~1-D output as its own stage (a distinct
    representation, never merged into a hidden block) even when it is inert."""
    active = _active_transitions(per_dim, L)
    cuts = set(active)
    if protect_output and L >= 1:
        cuts.add(L - 1)                       # force a boundary before the output
    stages, start = [], 0
    for i in range(1, L):
        if i in cuts:                         # close the current stage at the boundary
            stages.append((start, i - 1))
            start = i
    stages.append((start, L - 1))
    blocks = [(p, q) for (p, q) in stages if q > p]
    return dict(active_transitions=active, stages=stages,
                collapsible_blocks=blocks,
                redundancy=int(sum(q - p for p, q in blocks)),
                collapsed_depth=len(stages))


def collapsed_hidden_widths(hidden_widths, blocks):
    """Reduce a hidden-width tuple by collapsing each block to its first layer.
    Representation i (1-based over latents) is hidden layer i-1, so within a block
    (p, q) we keep rep p and drop reps p+1..q. Output/input reps carry no hidden
    width and are unaffected. Returns the collapsed hidden-width tuple."""
    drop = set()
    for p, q in blocks:
        drop.update(range(p + 1, q + 1))      # keep the first layer of the block
    return tuple(w for j, w in enumerate(hidden_widths) if (j + 1) not in drop)


def _stability(values, agreement_min):
    if not values:
        return dict(mode=None, agreement=0.0, ci95=(None, None), values=[],
                    stable=False)
    arr = np.asarray(values, int)
    mode = int(np.bincount(arr).argmax())
    agreement = float(np.mean(arr == mode))
    return dict(mode=mode, agreement=agreement,
                ci95=(int(np.percentile(arr, 2.5)), int(np.percentile(arr, 97.5))),
                values=arr.tolist(), stable=agreement >= agreement_min)


def convergence_depth(latents, epsilons, significance=True,
                      n_resample=CONV_N_RESAMPLE, subsample_frac=CONV_SUBSAMPLE_FRAC,
                      alpha=ALPHA, max_dim=MAX_DIMENSION, augment_output=True,
                      exclude_output=True, agreement_min=AGREEMENT_MIN, rng=None):
    """
    Read the MLP-persistence barcode and report, PER DIMENSION, which features are
    resolved and by which layer, plus the convergence/pruning depth d*.

    significance=True (optional): bars are additionally qualified by bootstrap
      recurrence over row-resamples (re-propagated through the fixed net). This adds a
      layer-axis stability gate ON TOP of the per-layer epsilon significance.
    significance=False: a single full-data barcode with NO resampling. Significance
      already lives in the per-layer epsilon choice (the scale-axis tau band discards
      noisy cycles), so every cycle present at the chosen epsilon is real information
      from that layer and counts as a genuine event; MLP-persistence then just tracks
      how those cycles are born/shared/resolved across layers. d* and the prunable
      tail are read directly from these births/deaths.

    Returns
    -------
    dict(significance, n_layers, ref_bars (per dim, capped),
         per_dim={0: summary, 1: summary}, d_star, inert_layers,
         d_star_stability (None if significance=False), n_replicates_ok)
    where each summary is the output of _summarise_dim (resolution_events,
    resolved_by, unresolved, onset_layers, events).
    """
    rng = np.random.default_rng(rng)
    ref = mlp_persistence(latents, epsilons, max_dim=max_dim,
                          augment_output=augment_output, exclude_output=exclude_output)
    L = ref["n_layers"]
    cap = _caps(L, exclude_output=exclude_output)
    ref_bars = {k: _cap_and_bars(ref["mlp_intervals_by_dim"][k], cap[k])
                for k in (0, 1)}

    if not significance:
        per_dim = {}
        for k in (0, 1):
            evs = [dict(birth=int(round(b)), death=int(round(d)),
                        essential=(int(round(d)) >= cap[k]))
                   for b, d in ref_bars[k]]
            per_dim[k] = _summarise_dim(cap[k], evs)
        d_star, inert = _dstar_from_dims(per_dim, L)
        return dict(significance=False, n_layers=L, ref_bars=ref_bars,
                    per_dim=per_dim, d_star=d_star, inert_layers=inert,
                    redundancy=collapsible_blocks(per_dim, L),
                    d_star_stability=None, n_replicates_ok=0,
                    pullback_trees=ref["pullback_trees"])

    N = len(latents[0])
    m = max(int(round(subsample_frac * N)), 10)
    rep_bars = {0: [], 1: []}
    n_ok = 0
    for _ in range(n_resample):
        idx = rng.choice(N, size=min(m, N), replace=False)
        lat_b = [np.asarray(Xi, float)[idx] for Xi in latents]
        try:
            t = mlp_persistence(lat_b, epsilons, max_dim=max_dim,
                                augment_output=augment_output,
                                exclude_output=exclude_output)
        except Exception:
            continue                               # never let one draw kill the pass
        n_ok += 1
        rep_bars[0].append(_cap_and_bars(t["mlp_intervals_by_dim"][0], cap[0]))
        rep_bars[1].append(_cap_and_bars(t["mlp_intervals_by_dim"][1], cap[1]))

    all_cells = {k: _event_cells(rep_bars[k], cap[k], alpha) for k in (0, 1)}
    kept = {k: [c for c in all_cells[k] if c["kept"]] for k in (0, 1)}
    genuine_set = {k: {(c["birth"], c["death"]) for c in kept[k]} for k in (0, 1)}
    per_dim = {k: _summarise_dim(cap[k], kept[k]) for k in (0, 1)}

    # Headline d* and its stability are the SAME quantity: the per-replicate last
    # GENUINE event (its bars restricted to the recurrence-significant cells), so
    # the reported d* is the central tendency of the distribution its CI describes.
    # (Earlier the headline used the union's last genuine event while the CI used
    # the per-replicate last RAW event -- two different quantities that disagreed.)
    rep_last = []
    for r in range(len(rep_bars[0])):
        last = []
        for k in (0, 1):
            for b, d in np.asarray(rep_bars[k][r], float).reshape(-1, 2):
                bi, di = int(round(b)), int(round(d))
                if (bi, di) in genuine_set[k]:
                    last.append(bi)
                    if di < cap[k]:
                        last.append(di)
        if last:
            rep_last.append(int(max(last)))

    stability = _stability(rep_last, agreement_min)
    d_star = stability["mode"]                       # mode of the per-replicate distribution
    d_star_aggregate, _ = _dstar_from_dims(per_dim, L)   # last genuine event in the union
    inert = list(range(d_star + 1, L)) if d_star is not None else []
    return dict(significance=True, n_layers=L, n_replicates_ok=n_ok,
                ref_bars=ref_bars, per_dim=per_dim, event_cells=kept,
                event_cells_all=all_cells, d_star=d_star,
                d_star_aggregate=d_star_aggregate, inert_layers=inert,
                redundancy=collapsible_blocks(per_dim, L),
                d_star_stability=stability,
                pullback_trees=ref["pullback_trees"])


def signal_dimension(convergence_result):
    """
    Routing / DESCRIPTION: which homology degree carries genuine structure worth
    reading -- H1 if a genuine loop is PRESENT at all, whether RESOLVED (a finite
    bar) or PRESERVED (an essential bar that survives every layer), else H0. This is
    what tells you the MLP barcode is worth reading in H1; it does NOT by itself mean
    the network simplified anything. (e.g. an autoencoder: signal_dim = H1, because a
    strong loop is present and preserved throughout.)
    """
    h1 = convergence_result["per_dim"].get(1, {})
    present = (h1.get("resolved_by") is not None) or (h1.get("unresolved", 0) > 0)
    return 1 if present else 0


def simplification_dimension(convergence_result):
    """
    The METHOD'S PURPOSE: which degree actually undergoes simplification across depth
    -- H1 only if a genuine loop is RESOLVED (a real death), else H0. This is the
    degree tied to convergence and to changes through layers; a merely PRESERVED loop
    is a non-event here. (e.g. an autoencoder: simplification_dim = H0 -- it does not
    untangle the loop; a classifier that resolves the loop: simplification_dim = H1.)

    d* (convergence_depth) is driven by simplification: births and REAL deaths only;
    an essential/preserved bar contributes no late event, so it never inflates d*.
    """
    h1 = convergence_result["per_dim"].get(1, {})
    return 1 if h1.get("resolved_by") is not None else 0


# Backwards-compatible alias: historically 'carrier' meant the resolution locus, so
# it maps to the simplification dimension (the convergence-relevant one).
def carrier_dimension(convergence_result):
    """Deprecated name. Equals `simplification_dimension` (the convergence-relevant
    degree). Use `signal_dimension` for presence/description and
    `simplification_dimension` for what drives d*."""
    return simplification_dimension(convergence_result)


# ============================================================================
# Pillar 2 cross-check -- Lipschitz confirmation that the tail is quiescent
# ============================================================================

def cross_check_bottleneck(latents, d_star, homology_dim=1,
                           max_dimension=MAX_DIMENSION, exclude_last=True):
    """
    Stable, 1-Lipschitz confirmation of d*: bottleneck distance between consecutive,
    diameter-NORMALISED per-layer persistence diagrams of `homology_dim` (pass the
    carrier dimension). Normalising by each cloud's diameter makes this measure
    reorganisation rather than overall rescaling. Beyond d* the change should sit in
    the noise (~0); a large value past d* contradicts the barcode reading.

    exclude_last=True drops the final (last-hidden -> output) transition, which is
    the funnel COLLAPSE toward the decision layer, not reorganisation -- including it
    pollutes the "should be ~0 past d*" check (the output is ~1-D for binary nets and
    the softmax simplex for multiclass). Set False to inspect it explicitly.

    Returns a list of dict(transition, bottleneck, beyond_dstar).
    """
    diags = []
    for Xi in latents:
        X = np.asarray(Xi, float)
        if len(X) < 2:
            diags.append(np.empty((0, 2)))
            continue
        diam = float(pdist(X).max()) or 1.0
        Xn = X / diam
        eps = float(pdist(Xn).max()) or 1.0
        st = compute_vietoris_rips_complex(
            Xn, eps, max_dimension=max(homology_dim + 1, max_dimension))
        d = compute_persistence_diagrams(st, max_dim=homology_dim).get(
            homology_dim, np.empty((0, 2)))
        diags.append(_finite_bars(d))

    n_trans = len(diags) - 1
    upper = max(n_trans - 1, 0) if exclude_last else n_trans
    transitions = []
    for i in range(upper):
        b = _safe_bottleneck(diags[i], diags[i + 1])
        transitions.append(dict(
            transition=(i, i + 1), bottleneck=float(b),
            beyond_dstar=(d_star is not None and i >= d_star)))
    return transitions


# ============================================================================
# Pillar 3 -- output-layer H1 anomaly (actionable interpretability, scale axis)
# ============================================================================

def output_loop_anomaly(latents, alpha=ALPHA, n_boot=N_BOOT,
                        max_dimension=MAX_DIMENSION, rng=None):
    """
    Does the FINAL representation still contain a significant loop (in its own
    layer-persistence diagram)? Surviving H1 at the output means the network has not
    untangled the data into a linearly separable form -- a concrete failure signal.
    Complementary to the layer-axis 'unresolved loops' count from convergence_depth.
    """
    Xout = np.asarray(latents[-1], float)
    if Xout.ndim == 1 or Xout.shape[1] < 2:
        return dict(applicable=False, n_significant_H1=0, anomaly=False,
                    note="output is ~1-D; H1 structurally absent")
    res = select_epsilon_for_layer(
        Xout, dim=1, max_dimension=max_dimension, use_bootstrap=True,
        n_boot=n_boot, alpha=alpha, vr_builder=compute_vietoris_rips_complex, rng=rng)
    n_sig = int(res["n_significant"])
    return dict(applicable=True, n_significant_H1=n_sig, anomaly=n_sig > 0,
                tau=float(res["tau"]),
                note="surviving significant H1 at output => not linearly separated")


# ============================================================================
# Stage 0 helper -- units-correct sparsification
# ============================================================================

def delta_net_sparsify(points, delta, seed=0):
    """
    delta-net in ACTUAL distance. Returns (kept_points, kept_indices,
    covering_radius). The bottleneck distortion of the resulting diagram is bounded
    by `delta` (the covering radius), NOT by delta**2; restate the manuscript's
    sparsification claim in these (actual-distance) units.
    """
    pts = np.asarray(points, float)
    kept_idx, _ = sparsify_delta_net(pts, delta, seed=seed)
    return pts[kept_idx], kept_idx, float(delta)


# ============================================================================
# Stage 0 helper -- curse-of-dimensionality mitigation (Hiraoka et al.)
# ============================================================================

def pca_normalize(X, n_components=10, var_target=0.90, max_components=20,
                  normalize="whiten", min_components=2):
    """
    Project a high-dimensional cloud onto its principal subspace and normalise the
    scores, following Hiraoka, Imoto, Kanazawa & Liu, "Curse of dimensionality on
    persistence diagrams". In high dimension isotropic noise inflates pairwise
    distances and concentrates them, making persistence diagrams unreliable;
    projecting onto the signal subspace and normalising the scores recovers a
    well-scaled cloud on which to compute persistence.

      1. centre and PCA-project to k components,
      2. normalise the PC scores,
      3. (caller) compute persistence on the returned cloud.

    n_components : fixed k (DEFAULT, recommended). Keep it near the SIGNAL dimension.
                   If None, the smallest k explaining >= var_target of the variance,
                   capped at max_components -- but beware: under high-dimensional
                   noise the variance is spread across a long tail, so var_target
                   OVER-RETAINS noise directions, which `normalize='whiten'` then
                   rescales up. Prefer a fixed, conservative k.
    normalize    : 'whiten' (each retained PC to unit variance -- standardises scale
                   across layers; only safe when k is the signal dimension),
                   'global' (divide by the overall RMS scale -- preserves relative PC
                   magnitudes, so strong signal PCs keep dominating; safer if k may
                   include noise), or 'none'.

    Rows are preserved (only coordinates change), so the pullback tower stays valid.
    Returns the compressed, normalised cloud.
    """
    X = np.asarray(X, float)
    X = X.reshape(len(X), -1)
    n, d = X.shape
    if d < 2 or n < 2:
        return X.copy()
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    scores = U * S                                   # PC scores, shape (n, r)

    if n_components is None:
        var = S ** 2
        cum = np.cumsum(var) / max(float(var.sum()), 1e-12)
        k = int(np.searchsorted(cum, var_target) + 1)
    else:
        k = int(n_components)
    k = min(k, scores.shape[1], max_components)
    k = max(k, min(min_components, scores.shape[1]))
    Z = scores[:, :k]

    if normalize == "whiten":
        sd = Z.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z = Z / sd
    elif normalize == "global":
        rms = float(np.sqrt((Z ** 2).mean())) or 1.0
        Z = Z / rms
    elif normalize != "none":
        raise ValueError(f"unknown normalize={normalize!r}")
    return Z


def preprocess_latents(latents, method="raw", **kw):
    """
    Apply a Stage-0 transform to every representation, preserving row order.
      method='raw' : identity (faithful baseline).
      method='pca' : pca_normalize on each cloud (Hiraoka et al.); kw forwarded.
    Returns a new list of clouds.
    """
    if method == "raw":
        return [np.asarray(X, float).reshape(len(X), -1) for X in latents]
    if method == "pca":
        return [pca_normalize(X, **kw) for X in latents]
    raise ValueError(f"unknown method={method!r}")


# ============================================================================
# The one-call pipeline + a consistent reporter
# ============================================================================

def run_pipeline(latents, augment_output=True, compute_confidence=True,
                 n_resample=CONV_N_RESAMPLE, alpha=ALPHA,
                 use_bootstrap=USE_BOOTSTRAP, cross_check=True,
                 check_output_loop=True, rng=None):
    """
    Run the whole unified pipeline and return one structured result. The reported
    objects (epsilon audit, per-dimension resolution events + resolved-by layers,
    unresolved-loop flag, d* with stability gate, carrier-dimension cross-check,
    output anomaly) are identical across all four experiments by construction.

    compute_confidence=True  -> layer-axis bootstrap-recurrence significance (the
                                headline numbers).
    compute_confidence=False -> single-pass PROVISIONAL barcode (fast wiring check).
    """
    eps_res = select_epsilon(latents, use_bootstrap=use_bootstrap, alpha=alpha, rng=rng)
    epsilons = eps_res["epsilons"]

    conv = convergence_depth(latents, epsilons, significance=compute_confidence,
                             n_resample=n_resample, alpha=alpha,
                             augment_output=augment_output, rng=rng)

    signal = signal_dimension(conv)
    simpl = simplification_dimension(conv)
    # the cross-check corroborates d* (convergence), so it reads the dimension that
    # actually simplifies; if nothing simplifies in H1, fall back to H0 structure.
    xc = cross_check_bottleneck(latents, conv["d_star"],
                                homology_dim=simpl) if cross_check else None
    anom = output_loop_anomaly(latents, alpha=alpha, rng=rng) if check_output_loop else None

    return dict(params=dict(PARAMS), epsilons=epsilons,
                epsilon_audit=eps_res["per_layer"], convergence=conv,
                signal_dim=signal, simplification_dim=simpl,
                carrier_dim=simpl,                       # back-compat alias
                cross_check=xc, output_anomaly=anom)


def _fmt_events(summary):
    if not summary["events"]:
        return "  (none)"
    parts = []
    for e in summary["events"]:
        arrow = f"{e['birth']}->{e['death']}"
        if e["essential"]:
            arrow += "*"                                   # survives to end
        if "frequency" in e:
            arrow += f"@{e['frequency']:.2f}"
        parts.append(arrow)
    return "  " + ", ".join(parts)


def pretty_print(result):
    """Consistent textual report of a run_pipeline result, for every driver."""
    print("=" * 70)
    print("UNIFIED PIPELINE REPORT")
    print("=" * 70)
    print("pre-committed params:", result["params"])

    print("\n-- epsilon selection (scale axis): eps_H0 and eps_H1 per layer --")
    print(f"{'layer':>5} | {'eps_H0':>8} | {'eps_H1':>8} | {'tau_H0':>8} | "
          f"{'nH0':>4} | {'nH1':>4} | flags")
    for r in result["epsilon_audit"]:
        flags = [f for f, on in (("H0fallback", r.get("h0_fallback")),
                                 ("H0split", r.get("h0_separability")),
                                 ("capped", r.get("capped")),
                                 ("degenerate", r.get("degenerate"))) if on]
        e1 = "  -  " if not r.get("h1_significant") else f"{r['eps_H1_used']:8.4f}"
        print(f"{r['layer']:5d} | {r['eps_H0_used']:8.4f} | {e1:>8} | "
              f"{r['tau_H0']:8.4f} | {r['n_sig_H0']:4d} | {r.get('n_sig_H1',0):4d} | "
              f"{', '.join(flags) if flags else '-'}")

    c = result["convergence"]
    kind = "bootstrap-recurrence" if c["significance"] else "PROVISIONAL (no significance)"
    print(f"\n-- MLP-persistence events (layer axis; {kind}) --")
    if c["significance"]:
        print(f"  replicates used: {c['n_replicates_ok']}   "
              f"(birth->death, * = survives to end, @ = recurrence freq)")
    h1, h0 = c["per_dim"][1], c["per_dim"][0]
    print("  H1 (loops):")
    print("    events:" + _fmt_events(h1))
    print(f"    loops resolved by layer : {h1['resolved_by']}")
    print(f"    unresolved loops        : {h1['unresolved']}  "
          f"(births {h1['unresolved_births']})   [>0 = not untangled: FAILURE flag]")
    print("  H0 (components):")
    print("    events:" + _fmt_events(h0))
    print(f"    separation settles by   : {h0['resolved_by']}")
    print(f"    components to the end    : {h0['unresolved']}")

    sig = result.get("signal_dim", result.get("carrier_dim"))
    sim = result.get("simplification_dim", result.get("carrier_dim"))
    print(f"\n  signal dimension (present)      : H{sig}  "
          f"(degree carrying genuine structure; read the barcode here)")
    print(f"  simplification dimension        : H{sim}  "
          f"(degree actually resolved across depth -> drives d*)")
    if sig == 1 and sim == 0:
        print("    note: H1 loop is PRESENT but PRESERVED (essential) -- a non-event "
              "for convergence; it appears as a surviving bar in the MLP barcode.")
    print(f"  convergence d*           : {c['d_star']}  "
          f"(mode of per-replicate last genuine event)")
    if c.get("d_star_aggregate") is not None and c["significance"]:
        print(f"  d* (union, reference)    : {c['d_star_aggregate']}  "
              f"(last genuine event in the pooled barcode)")
    print(f"  inert / prunable layers  : {c['inert_layers']}")

    s = c.get("d_star_stability")
    if s is not None:
        print(f"  d* stability             : mode={s['mode']} "
              f"agreement={s['agreement']:.2f} CI={s['ci95']} stable={s['stable']}")
        if not s["stable"]:
            print("  [gate] d* unstable -> do NOT assert pruning; report instability.")

    xc = result.get("cross_check")
    if xc is not None:
        print(f"\n-- bottleneck cross-check (diameter-normalised H{result.get('simplification_dim', result['carrier_dim'])}) --")
        for t in xc:
            tag = "  (>d*)" if t["beyond_dstar"] else ""
            print(f"  layers {t['transition']}: {t['bottleneck']:.4f}{tag}")
        print("  layers beyond d* should be ~0; a large value there contradicts d*.")

    a = result.get("output_anomaly")
    if a is not None:
        print("\n-- output-layer H1 anomaly (scale axis) --")
        if not a["applicable"]:
            print(f"  {a['note']}")
        else:
            print(f"  significant H1 at output: {a['n_significant_H1']}  "
                  f"anomaly={a['anomaly']}")
            if a["anomaly"]:
                print(f"  [flag] {a['note']}")
    print("=" * 70)


if __name__ == "__main__":
    # Dependency-free smoke check of the role rule and the event readers.
    import numpy as _np
    assert relevant_dimension(_np.random.randn(20, 4), is_output=False) == 1
    assert relevant_dimension(_np.random.rand(20, 1), is_output=True) == 0
    assert relevant_dimension(_np.random.rand(20, 1), is_output=False) == 0

    # a stable lifespan-1 loop death (0->1) must now be KEPT as genuine
    reps = [_np.array([[0.0, 1.0]]) for _ in range(20)]   # (birth=0, death=1) each
    cells = _event_cells(reps, cap=4, alpha=0.05)
    kept = [c for c in cells if c["kept"]]
    assert len(kept) == 1 and kept[0]["birth"] == 0 and kept[0]["death"] == 1
    assert not kept[0]["essential"]
    summ = _summarise_dim(4, kept)
    assert summ["resolved_by"] == 1 and summ["unresolved"] == 0
    print("role rule + lifespan-1 loop-death retention: PASS")
