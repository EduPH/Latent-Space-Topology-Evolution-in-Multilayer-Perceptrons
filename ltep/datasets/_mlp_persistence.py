"""
mlp_persistence_convergence.py
==============================

Parameter-free convergence / inert-layer diagnostic read DIRECTLY from the single
MLP-persistence barcode (the layer-indexed combined filtration). No bottleneck
distance, no per-layer diagram comparison, no threshold parameter.

CONSTRUCTION RECAP (so the reading is unambiguous):
  create_combined_filtration assigns each simplex filtration value = the FIRST
  layer index at which it appears; layers are indexed 0 .. L-1 for L
  representations. The resulting persistence barcode has births and deaths in
  LAYER UNITS. An essential bar (death = inf) is a feature that survives to the
  last representation; we cap its death at L-1 (NOT the hardcoded 2 from the old
  plotting snippet, which is wrong for L > 3).

PARAMETER-FREE RULE:
  A bar (b, d) counts as a genuine topological EVENT iff its layer-lifespan
  d - b >= 2, i.e. the feature survives at least one full intermediate layer.
  A bar with d - b == 1 is born and dies within a single transition (a flicker);
  these are exactly what made the earlier 'activity count' unstable, so we ignore
  them -- with the smallest possible structural span as the cutoff, not a tuned
  value.

  Convergence layer d* = the last layer at which any genuine bar is BORN or DIES,
  taken over BOTH H0 and H1. Layers beyond d* produce no genuine topological
  event: the barcode is frozen there, so those layers are inert / prunable.

  Reported per dimension as well, because H0 convergence (clusters settling) and
  H1 convergence (loops settling) mean different things and may differ.

Stability note: 'last genuine event' is robust precisely because the >=2-layer
filter removes the flickering tail bars that 'last death of any bar' was sensitive
to. The diagnostic should be run through the same stability gate (recompute on
independent analysis subsamples) before any pruning claim; convergence in LAYER
units has no free parameter to tune, which is the point.
"""

import numpy as np


def _bars_by_dim(combined_st, n_layers, max_dim=1):
    """
    Compute persistence of the layer-indexed combined filtration and return, per
    homology dimension, an array of (birth_layer, death_layer) with essential
    deaths capped at L-1 = n_layers-1.
    """
    combined_st.compute_persistence(homology_coeff_field=2, min_persistence=-1.0)
    cap = float(n_layers - 1)
    out = {}
    for k in range(max_dim + 1):
        iv = combined_st.persistence_intervals_in_dimension(k)
        if iv is None or len(iv) == 0:
            out[k] = np.empty((0, 2), dtype=float)
            continue
        arr = np.asarray(iv, dtype=float).reshape(-1, 2)
        arr[~np.isfinite(arr[:, 1]), 1] = cap   # cap essentials at last layer
        out[k] = arr
    return out


def genuine_bars(bars, min_lifespan=2):
    """Bars whose layer-lifespan (death - birth) >= min_lifespan (=2, fixed)."""
    if bars.size == 0:
        return bars
    return bars[(bars[:, 1] - bars[:, 0]) >= min_lifespan]


def convergence_layer(combined_st, n_layers, max_dim=1):
    """
    Parameter-free convergence diagnostic from one MLP-persistence barcode.

    Returns a dict:
      d_star            : last layer with a genuine (lifespan>=2) birth or death,
                          over all dimensions (None if there are no genuine bars)
      d_star_by_dim     : same, per homology dimension
      n_genuine_by_dim  : count of genuine bars per dimension
      inert_layers      : list of layer indices > d_star (the prunable tail)
      bars_by_dim       : the (capped) bars per dimension, for inspection/plots
    """
    bars = _bars_by_dim(combined_st, n_layers, max_dim=max_dim)

    d_by_dim, n_by_dim = {}, {}
    last_events = []
    for k in range(max_dim + 1):
        g = genuine_bars(bars.get(k, np.empty((0, 2))))
        n_by_dim[k] = int(len(g))
        if len(g) == 0:
            d_by_dim[k] = None
            continue
        # last layer at which a genuine bar is born OR dies
        last_event_k = float(max(g[:, 0].max(), g[:, 1].max()))
        # a death capped at L-1 (essential) is NOT a real 'event' layer: the
        # feature never actually dies, so only count deaths strictly below the cap
        deaths = g[:, 1]
        real_deaths = deaths[deaths < (n_layers - 1)]
        births = g[:, 0]
        candidates = list(births)
        if real_deaths.size:
            candidates += list(real_deaths)
        last_event_k = float(max(candidates)) if candidates else float(births.max())
        d_by_dim[k] = int(round(last_event_k))
        last_events.append(d_by_dim[k])

    d_star = int(max(last_events)) if last_events else None
    inert = list(range(d_star + 1, n_layers)) if d_star is not None else []
    return dict(d_star=d_star, d_star_by_dim=d_by_dim,
                n_genuine_by_dim=n_by_dim, inert_layers=inert,
                bars_by_dim=bars)


def plot_mlp_barcode(bars_by_dim, n_layers, d_star=None, ax=None,
                     colors=("tab:blue", "tab:red")):
    """
    Plot the layer-indexed MLP-persistence barcode, H0 and H1, with the
    convergence layer d* marked. Genuine bars (lifespan>=2) solid; flickers
    (lifespan 1) faint, so the reader sees what was filtered and why.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    y = 0
    yticks, ylabels = [], []
    for k, bars in sorted(bars_by_dim.items()):
        for b, d in bars:
            lifespan = d - b
            faint = lifespan < 2
            ax.plot([b, d], [y, y], lw=2 if not faint else 1,
                    color=colors[k % len(colors)], alpha=0.35 if faint else 1.0,
                    solid_capstyle="butt")
            yticks.append(y); ylabels.append(f"H{k}")
            y += 1
    if d_star is not None:
        ax.axvline(d_star + 0.5, color="black", ls="--",
                   label=f"convergence layer d*={d_star}")
        ax.legend(loc="lower right")
    ax.set_xlabel("layer index")
    ax.set_xlim(-0.3, n_layers - 0.7)
    ax.set_xticks(range(n_layers))
    ax.set_yticks([])
    ax.set_title("MLP-persistence barcode (layer-indexed)")
    return ax


# ----------------------------------------------------------------------------
# Convenience: build the combined filtration from a list of per-layer trees and
# run the diagnostic in one call. The trees must already be the pullback /
# layer complexes (output of the pullback chain), NOT independent per-layer VR.
# ----------------------------------------------------------------------------

def diagnose_from_trees(pullback_trees, max_dim=1):
    """
    pullback_trees : list of SimplexTree, one per representation, sharing vertex
                     indices across layers (i.e. the pullback tower), ordered
                     input -> ... -> output.
    """
    from ..vr import create_combined_filtration
    L = len(pullback_trees)
    combined = create_combined_filtration(pullback_trees, max_dimension=max_dim)
    return convergence_layer(combined, n_layers=L, max_dim=max_dim)