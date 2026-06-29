#!/usr/bin/env python
# coding: utf-8
"""
pipeline_plots.py
=================

Visual sanity checks for the unified pipeline (pipeline.py):

  * plot_layer_persistence : one persistence DIAGRAM per layer, in DISTANCE units,
    with (a) the diagonal, (b) the bootstrap noise floor tau as a band parallel to
    the diagonal -- points above it are significant -- and (c) the SELECTED epsilon
    drawn as a green cross-hair plus the shaded "alive at epsilon" quadrant
    {birth <= epsilon <= death}. A good epsilon is one where the significant
    features of the layer's chosen dimension fall inside the green quadrant.

  * plot_mlp_persistence : the layer-indexed MLP-persistence BARCODE, in LAYER
    units, for H0 and H1, with kept (recurring/genuine) events bold, removed
    (non-recurring) events faint, essential bars (survive to the last layer) marked
    with a ">", and the convergence depth d* drawn as a dashed line. This shows the
    downstream effect of the epsilon choice on the cross-layer structure.

Depends on: VR_trajectories.py, topological_metrics.py (same directory).
Save PNGs by passing `path=...`; omit it to get the Figure back for interactive use.
"""

from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; comment out for interactive backends
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from .vr import compute_vietoris_rips_complex
from .metrics import compute_persistence_diagrams, _finite_bars

_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:purple"}
_MARKERS = {0: "o", 1: "^", 2: "s"}


# ----------------------------------------------------------------------------
# Layer persistence (distance units) -- the epsilon check
# ----------------------------------------------------------------------------

def _layer_diagrams(Xi, max_dimension=2):
    X = np.asarray(Xi, float)
    eps_full = float(pdist(X).max()) if len(X) > 1 else 1.0
    st = compute_vietoris_rips_complex(X, eps_full, max_dimension=max_dimension)
    return compute_persistence_diagrams(st, max_dim=max_dimension)


def _draw_layer_diagram(ax, diags_by_dim, eps, tau, title, eps2=None, tau2=None):
    """Draw one persistence diagram with the tau band(s) and the epsilon quadrant(s).
    eps  = eps_H0 (green, solid quadrant); eps2 = eps_H1 (orange dashed line), if any.
    tau  = H0 noise floor (grey band); tau2 = H1 noise floor (orange band) -- H1 loops
    are judged SIGNIFICANT against tau2, NOT tau, so the two bands generally differ and
    a loop above the grey H0 band can still be sub-threshold for H1."""
    finite = []
    for k, d in diags_by_dim.items():
        fb = _finite_bars(d)
        if fb.size:
            finite.append(fb[:, 1].max())
    hi = max(finite, default=(eps * 2 if eps else 1.0))
    hi = max(hi, eps if eps else 0.0, eps2 if eps2 else 0.0) * 1.05 + 1e-9

    ax.plot([0, hi], [0, hi], color="0.6", lw=1)                       # diagonal
    ax.plot([0, hi], [tau, hi + tau], color="0.6", lw=1, ls="--")     # H0 noise floor
    ax.fill_between([0, hi], [0, hi], [tau, hi + tau],
                    color="0.85", alpha=0.6, zorder=0)                 # H0 noise band
    if tau2 is not None and abs(tau2 - tau) > 1e-9:                    # H1 noise floor
        ax.plot([0, hi], [tau2, hi + tau2], color="tab:orange", lw=1, ls="--", alpha=0.7)
        ax.fill_between([0, hi], [0, hi], [tau2, hi + tau2],
                        color="tab:orange", alpha=0.06, zorder=0)      # H1 noise band

    if eps is not None:
        ax.axvline(eps, color="tab:green", lw=1.2, ls=":")
        ax.axhline(eps, color="tab:green", lw=1.2, ls=":")
        ax.add_patch(plt.Rectangle((0, eps), eps, hi - eps,
                                   color="tab:green", alpha=0.08, zorder=0))
    if eps2 is not None:                                               # eps_H1 marker
        ax.axvline(eps2, color="tab:orange", lw=1.2, ls="--")
        ax.axhline(eps2, color="tab:orange", lw=1.2, ls="--")

    for k, d in diags_by_dim.items():
        fb = _finite_bars(d)
        if fb.size:
            ax.scatter(fb[:, 0], fb[:, 1], s=24, c=_COLORS.get(k, "k"),
                       marker=_MARKERS.get(k, "o"), edgecolor="white", lw=0.4,
                       label=f"H{k}", zorder=3)

    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")


def plot_layer_embeddings(latents, layer_names=None, color=None, path=None,
                          method="isomap", title=None, cmap="twilight"):
    """2-D embedding of each layer's point cloud, one panel per layer, scatter coloured
    by `color` (e.g. rotation-angle index). For the COIL autoencoder this is the direct
    VISUAL test of loop preservation: a preserved rotation circle shows as a closed
    colour-cycling ring at the bottleneck. Uses Isomap (geodesic, best for a manifold
    loop); silently falls back to PCA if Isomap is unavailable or fails (e.g. tiny N).

    color : 1-D array aligned to the rows of every layer (shared row index), or None.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    L = len(latents)
    if L == 0:
        return None
    ncol = min(L, 5)
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.1 * nrow), squeeze=False)

    def _embed2d(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] <= 2:                       # already <=2-D: pad/return as-is
            return np.column_stack([X, np.zeros(len(X))])[:, :2]
        if method == "isomap":
            try:
                from sklearn.manifold import Isomap
                k = int(min(10, max(2, len(X) - 1)))
                return Isomap(n_neighbors=k, n_components=2).fit_transform(X)
            except Exception:
                pass                              # fall through to PCA
        Xc = X - X.mean(0)
        try:
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            return U[:, :2] * S[:2]
        except Exception:
            return Xc[:, :2]

    for i in range(L):
        ax = axes[i // ncol][i % ncol]
        Y = _embed2d(latents[i])
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cmap, s=18,
                        edgecolor="white", lw=0.3)
        nm = layer_names[i] if (layer_names and i < len(layer_names)) else f"layer {i}"
        ax.set_title(nm, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(L, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if color is not None:
        fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01, label="angle index")
    meth = "Isomap" if method == "isomap" else "PCA"
    fig.suptitle(title or f"Layer embeddings ({meth}, 2-D); a preserved loop = closed "
                 f"colour-cycling ring", fontsize=10)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig


def plot_layer_persistence(latents, eps_result, max_dimension=2, path=None,
                           layer_names=None):
    """Grid of per-layer persistence diagrams with the tau band and BOTH chosen
    epsilons drawn (eps_H0 and eps_H1), since the two barcodes read at different
    scales. eps_H1 is omitted on a panel where no loop was significant."""
    audit = eps_result["per_layer"]
    L = len(latents)
    ncol = min(3, L)
    nrow = int(np.ceil(L / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for i, Xi in enumerate(latents):
        ax = axes[i // ncol][i % ncol]
        a = audit[i]
        diags = _layer_diagrams(Xi, max_dimension=max_dimension)
        nm = layer_names[i] if (layer_names and i < len(layer_names)) else f"layer {i}"
        t1 = a.get("h1_tier", 3)
        flags = "".join(s for s, on in ((" [H0 fallback]", a.get("h0_fallback")),
                                        (" [capped]", a.get("capped")),
                                        (" [degenerate]", a.get("degenerate"))) if on)
        # show eps_H1 for tier 1 (significant) AND tier 2 (most-persistent, sub-threshold)
        e0 = a["eps_H0_used"]; e1 = a["eps_H1_used"] if t1 in (1, 2) else None
        h1n = a.get("n_sig_H1", 0)
        if e1 is None:
            h1txt = "epsH1=- (no loop)"
        elif t1 == 1:
            h1txt = f"epsH1={e1:.3f} (n_sig {h1n})"
        else:                                    # tier 2: below tau, most-persistent loop
            h1txt = f"epsH1={e1:.3f} (sub-tau)"
        title = f"{nm}  epsH0={e0:.3f} (n_sig {a['n_sig_H0']})  " + h1txt + flags
        # H0 epsilon = solid green line; H1 epsilon = dashed orange line
        _draw_layer_diagram(ax, diags, e0, a["tau_H0"], title, eps2=e1,
                            tau2=a.get("tau_H1"))
    for j in range(L, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Layer persistence (distance units): green = eps_H0, orange dashed = "
                 "eps_H1; grey band = tau_H0, orange band = tau_H1 (H1 loops are "
                 "significant only ABOVE the orange band)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig


# ----------------------------------------------------------------------------
# Betti-0 vs epsilon diagnostic -- "why so few components?"
# ----------------------------------------------------------------------------

def betti0_curve(Xi, n_eps=200):
    """
    Number of H0 components alive as a function of scale epsilon, for one cloud.
    Returns (eps_grid, betti0). betti0 starts at N (all points isolated at eps=0)
    and decays to 1 as everything merges. Computed from the H0 death times: a
    component dies (merges) at its death scale, so betti0(eps) = N - #(deaths <= eps).
    """
    X = np.asarray(Xi, float)
    n = len(X)
    if n < 2:
        return np.array([0.0, 1.0]), np.array([n, n])
    eps_full = float(pdist(X).max()) or 1.0
    st = compute_vietoris_rips_complex(X, eps_full, max_dimension=1)
    h0 = compute_persistence_diagrams(st, max_dim=1).get(0, np.empty((0, 2)))
    deaths = np.asarray(h0, float).reshape(-1, 2)[:, 1]
    deaths = np.sort(deaths[np.isfinite(deaths)])
    grid = np.linspace(0.0, eps_full, n_eps)
    # components alive at eps = N - (#finite deaths <= eps)
    betti = n - np.searchsorted(deaths, grid, side="right")
    return grid, betti


def plot_betti0_diagnostic(latents, eps_result, layer_indices=None,
                           layer_names=None, path=None):
    """
    For selected layers, plot the Betti-0(eps) curve with the SELECTED epsilon
    marked, to see how many components survive at the chosen scale -- i.e. whether
    the sparse MLP-persistence H0 barcode is because epsilon sits past the merge.

    layer_indices : which layers to show (default: first, middle, last).
    """
    L = len(latents)
    if layer_indices is None:
        layer_indices = sorted(set([0, L // 2, L - 1]))
    audit = eps_result["per_layer"]
    k = len(layer_indices)
    fig, axes = plt.subplots(1, k, figsize=(4.2 * k, 3.6), squeeze=False)
    for j, i in enumerate(layer_indices):
        ax = axes[0][j]
        grid, betti = betti0_curve(latents[i])
        n = len(latents[i])
        ax.plot(grid, betti, color="tab:blue", lw=1.6)
        eps = audit[i]["eps_H0_used"]
        b_at_eps = int(betti[np.searchsorted(grid, eps, side="right") - 1]) if eps else n
        ax.axvline(eps, color="tab:green", ls=":", lw=1.4,
                   label=f"sel eps={eps:.2f}\n-> {b_at_eps} comp.")
        ax.axhline(1, color="0.7", ls="--", lw=0.8)
        nm = layer_names[i] if (layer_names and i < len(layer_names)) else f"layer {i}"
        ax.set_title(f"{nm}  (N={n})", fontsize=9)
        ax.set_xlabel("epsilon (distance)")
        ax.set_ylabel("Betti-0 (components alive)")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Betti-0 vs epsilon: where the selected epsilon sits relative to "
                 "component merging", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig


# ----------------------------------------------------------------------------
# MLP persistence (layer units) -- the downstream barcode
# ----------------------------------------------------------------------------

def plot_mlp_persistence(conv_result, path=None, layer_names=None, title=None,
                         epsilons=None):
    """Layer-indexed MLP-persistence barcode (H0, H1) with kept/removed events + d*.

    layer_names : optional list of length n_layers to label the x-axis ticks
                  (e.g. ResNet block names ['layer1.0', ..., 'logits']); rotated.
    epsilons    : optional list of length n_layers; the chosen epsilon per layer is
                  appended under each x-tick (the scale at which that layer was read).
    title       : optional override for the figure title prefix.
    """
    L = conv_result["n_layers"]
    sig = conv_result["significance"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = 0

    if sig:
        all_cells = conv_result.get("event_cells_all", {})
        for k in (1, 0):                                   # H1 on top, H0 below
            for c in all_cells.get(k, []):
                kept = c["kept"]
                ax.plot([c["birth"], c["death"]], [y, y],
                        lw=3.0 if kept else 1.2, color=_COLORS[k],
                        alpha=1.0 if kept else 0.3, solid_capstyle="butt")
                if c["essential"]:
                    ax.plot(c["death"], y, ">", color=_COLORS[k],
                            alpha=1.0 if kept else 0.3, ms=6)
                ax.text(c["death"] + 0.05, y, f"{c['frequency']:.2f}",
                        fontsize=6, va="center",
                        alpha=1.0 if kept else 0.4)
                y += 1
        subtitle = ("bold=kept  faint=removed  >=essential  num=recurrence freq")
    else:
        ref = conv_result["ref_bars"]
        for k in (1, 0):
            cnt = Counter((int(round(b)), int(round(d))) for b, d in ref[k])
            for (b, d), n in sorted(cnt.items()):
                ax.plot([b, d], [y, y], lw=2.4, color=_COLORS[k],
                        solid_capstyle="butt")
                if n > 1:
                    ax.text(d + 0.05, y, f"x{n}", fontsize=6, va="center")
                y += 1
        subtitle = "all cycles at the chosen \u03b5 (per-layer); multiplicities as xN"

    d_star = conv_result["d_star"]
    if d_star is not None:
        ax.axvline(d_star + 0.5, color="k", ls="--", lw=1.2, label=f"d*={d_star}")
        ax.legend(loc="lower right", fontsize=8)

    # legend proxies for H0/H1
    ax.plot([], [], color=_COLORS[1], lw=3, label="H1 (loops)")
    ax.plot([], [], color=_COLORS[0], lw=3, label="H0 (components)")
    ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel("layer index")
    ax.set_xlim(-0.3, L - 0.7)
    ax.set_xticks(range(L))
    # base labels: names if given, else indices; optionally append the chosen epsilon
    base = (list(layer_names) if (layer_names is not None and len(layer_names) == L)
            else [str(i) for i in range(L)])
    if epsilons is not None and len(epsilons) == L:
        labels = [f"{b}\n\u03b5={float(e):.2f}" for b, e in zip(base, epsilons)]
        ax.set_xticklabels(labels, rotation=0 if layer_names is None else 45,
                           ha="center" if layer_names is None else "right", fontsize=7)
        ax.set_xlabel("layer index (with chosen \u03b5)" if layer_names is None
                      else "layer / block (with chosen \u03b5)")
    elif layer_names is not None and len(layer_names) == L:
        ax.set_xticklabels(base, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("layer / block")
    ax.set_yticks([])
    ax.set_ylim(-1, max(y, 1))
    prefix = title or "MLP persistence (layer units)"
    ax.set_title(f"{prefix}\n{subtitle}", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig


# ----------------------------------------------------------------------------
# Trajectory flow -- per-point H0-community flow across layers (alluvial)
# ----------------------------------------------------------------------------

def _tree_components(tree, n):
    """Connected components of the 1-skeleton of a gudhi SimplexTree over the shared
    vertex set [n], returned as an integer label per row 0..n-1. Rows absent from the
    tree (isolated vertices) become their own singleton component. This is exactly the
    H0 the MLP-persistence barcode reads at that layer: beta_0(K_i) = number of labels."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    rows, cols = [], []
    for simplex, _ in tree.get_skeleton(1):
        if len(simplex) == 2:
            a, b = simplex
            rows += [a, b]
            cols += [b, a]
    if rows:
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        A = csr_matrix((n, n))
    _, labels = connected_components(A, directed=False)
    return labels


def _trees_from_eps(latents, eps_result, augment_output=True, exclude_output=True):
    """Rebuild the clamped pullback tower from the H0 epsilon sequence, identical to the
    tower the barcode is read from. Used only when the caller does not already hand in
    the trees (preferred path: pass conv["pullback_trees"])."""
    from . import pipeline as pl
    audit = eps_result["per_layer"]
    epsilons = [audit[i]["eps_H0_used"] for i in range(len(latents))]
    ref = pl.mlp_persistence(latents, epsilons, augment_output=augment_output,
                             exclude_output=exclude_output)
    return ref["pullback_trees"], epsilons


def plot_trajectory_flow(latents, eps_result, labels=None, path=None,
                         layer_names=None, d_star=None, trees=None,
                         augment_output=True, exclude_output=True):
    """Alluvial flow of points through the H0 communities of the PULLBACK TOWER.

    Communities are the connected components of the clamped pullback complexes K_i
    (Definition: layer-wise VR tower), i.e. the very objects whose persistence is the
    MLP-persistence barcode -- NOT an independent per-layer re-clustering. Pass the
    tower explicitly via `trees=conv["pullback_trees"]` for an exact match with the
    barcode/d*; if `trees` is None it is rebuilt from each layer's eps_H0 in
    `eps_result` (with the same augment_output/exclude_output as the barcode).

    Because K_0 subseteq K_1 subseteq ... is nested, the flow is purely alluvial:
    components only merge with depth and never split (Proposition: trajectories), so
    the node count at layer i equals beta_0(K_i) and the merges between layers i and
    i+1 equal the H0 deaths at that transition. Line width is proportional to point
    count, nodes are sized by population and coloured by dominant class when `labels`
    is given; `d_star` draws a dashed marker at the convergence depth. Saves to `path`
    (returns it) or returns the Figure."""
    audit = eps_result["per_layer"]
    L = len(latents)
    n = len(latents[0])
    if trees is None:
        trees, epsilons = _trees_from_eps(latents, eps_result,
                                          augment_output=augment_output,
                                          exclude_output=exclude_output)
    else:
        epsilons = [audit[i]["eps_H0_used"] for i in range(L)]
    comms = [_tree_components(trees[i], n) for i in range(L)]

    present = [sorted(set(c.tolist())) for c in comms]
    maxc = max(len(p) for p in present)
    pos = {}
    for i, ids in enumerate(present):
        for j, c in enumerate(ids):
            pos[(i, c)] = (i, j - (len(ids) - 1) / 2.0)

    node_color = {}
    if labels is not None and len(np.asarray(labels)) == n:
        labels = np.asarray(labels)
        classes = np.unique(labels)
        cmap = plt.get_cmap("tab10", max(len(classes), 10))
        cidx = {c: k for k, c in enumerate(classes)}
        for i in range(L):
            for c in present[i]:
                m = comms[i] == c
                dom = np.bincount([cidx[v] for v in labels[m]],
                                  minlength=len(classes)).argmax()
                node_color[(i, c)] = cmap(dom)

    fig, ax = plt.subplots(figsize=(max(7, 1.7 * L), 6))

    trans = []
    for i in range(L - 1):
        tc = {}
        for p in range(n):
            k = (comms[i][p], comms[i + 1][p])
            tc[k] = tc.get(k, 0) + 1
        trans.append(tc)
    maxcount = max((max(tc.values()) for tc in trans if tc), default=1)
    for i, tc in enumerate(trans):
        for (s, t), cnt in sorted(tc.items(), key=lambda kv: kv[1]):
            if (i, s) not in pos or (i + 1, t) not in pos:
                continue
            x0, y0 = pos[(i, s)]
            x1, y1 = pos[(i + 1, t)]
            ax.plot([x0, x1], [y0, y1], "-", color=node_color.get((i, s), "0.6"),
                    alpha=0.55, lw=0.4 + 5.0 * cnt / maxcount, zorder=1,
                    solid_capstyle="round")
    for (i, c), (x, y) in pos.items():
        cnt = int((comms[i] == c).sum())
        ax.scatter(x, y, s=60 + 240 * cnt / n,
                   c=[node_color.get((i, c), "lightblue")],
                   edgecolor="black", linewidth=0.6, zorder=2)

    if d_star is not None:
        ax.axvline(d_star + 0.5, ls="--", color="black", lw=1.2, zorder=0,
                   label=f"d*={d_star}")
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    names = layer_names or [f"L{i}" for i in range(L)]
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-maxc / 2 - 0.8, maxc / 2 + 0.8)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{names[i]}\n\u03b5={epsilons[i]:.2f}" for i in range(L)],
                       fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Latent trajectory flow (H0 communities per layer"
                 + (", coloured by class)" if node_color else ")"))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    return fig
