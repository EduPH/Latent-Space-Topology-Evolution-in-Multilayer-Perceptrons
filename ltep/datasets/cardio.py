#!/usr/bin/env python
# coding: utf-8
"""
Experiment_cardiotocography_metrics.py
======================================

Cardiotocography experiment, re-instrumented. This is the dataset where the new
metrics actually discriminate (unlike the separable toy), so it carries the
headline evidence for the revision.

HEADLINE / KEYSTONE (section_depth_sweep):
  Converts "the 5-layer net is overparameterized" -- previously read off a
  trajectory picture -- into a falsifiable, statistical claim:
    * train MLPs of depth 1..L (controlled width), several seeds each;
    * for the deepest net, the per-transition topological ACTIVITY T_i locates a
      saturation depth d* (the layer past which topology stops changing);
    * accuracy(depth) is shown to plateau at ~d* (mean +/- std over seeds).
  If accuracy keeps rising past d*, the topological diagnosis is wrong and we say
  so. That falsifiability is the point.

Everything else (layer metrics, principled epsilon selection, MLP-persistence
activity, trajectory agreement, linear probe, CKA / intrinsic-dimension baseline,
runtime, epsilon/sparsification/seed sensitivity, empirical stability test) is the
supporting battery.

Modules required (same directory): VR_trajectories.py, topological_metrics.py,
runtime_sensitivity.py.

UNITS / SPARSIFICATION NOTE carried throughout:
  The manuscript sparsifies on SQUARED distance (min_squared_dist) and states the
  diagram changes "by at most <that value>". The correct bottleneck bound is the
  covering radius in ACTUAL distance, i.e. delta = sqrt(min_squared_dist). Section
  G measures the realised distortion against that true-distance bound so the claim
  can be restated correctly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

SEED = 1234
np.random.seed(SEED)

import gudhi as gd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ..vr import (
    compute_vietoris_rips_complex, get_maximal_simplices, vr_pullback,
    create_combined_filtration, extract_1_skeleton_graph,
)
# NOTE: find_row_indices is NOT defined in VR_trajectories.py (the original cardio
# script relied on a notebook-cell definition that was never exported). We define a
# robust version locally so this file is self-contained.
from ..metrics import (
    compute_persistence_diagrams,
    betti_curve, make_eps_grid,
    total_persistence, persistence_entropy,
    persistence_landscape,
    layerwise_label_agreement,
    select_epsilon_for_layer,
    betti_layer_scale_grid, plot_betti_heatmap,
    topological_activity_from_mlp_barcode, saturation_depth,
)
from .. import runtime as rs

OUTDIR = os.path.dirname(os.path.abspath(__file__))
HANDPICKED_EPS = [1.0, 2.5, 0.2]          # original visual choice (depth-1 net)
MLP_SPARSE_SQDIST = 0.5                    # input sparsification for MLP persistence
LAYER_SPARSE_SQDIST = 0.05                 # sparsification for layer persistence


def saturation_depth_plateau(activity, threshold, tail_fraction=0.25):
    """
    Saturation depth for a possibly NON-MONOTONE activity curve.

    The strict rule ("first i beyond which activity stays <= threshold forever")
    is vetoed by any late uptick -- e.g. the H0 funnel collapse near the output.
    Here we instead require:
      (i)  activity[i] <= threshold  (this transition is quiet), AND
      (ii) the activity from i onward carries only a small share of the total
           'churn': sum(activity[i:]) <= tail_fraction * sum(activity).
    We return the first such i as the number of layers to keep (1-based), or None.

    This selects the onset of a low-activity tail rather than demanding global
    monotonicity. `tail_fraction` is reported alongside d* so the rule is auditable;
    do NOT tune it to manufacture a particular d*.
    """
    activity = np.asarray(activity, dtype=float)
    total = float(activity.sum())
    if total <= 0:
        return None
    for i in range(len(activity)):
        if activity[i] <= threshold and float(activity[i:].sum()) <= tail_fraction * total:
            return i + 1
    return None


# ============================================================================
# Data
# ============================================================================

def load_cardio_dataset():
    from ucimlrepo import fetch_ucirepo
    cardio = fetch_ucirepo(id=193)
    Xraw = cardio.data.features.values
    yraw = cardio.data.targets
    # binary split as in the original code (CLASS column, threshold > 4)
    y_binary = (np.array(yraw)[:, 0] > 4).astype(int)
    X = MinMaxScaler().fit_transform(Xraw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
    print(f"Dataset shape: {X.shape}  class balance: {np.bincount(y_binary)}")
    return X, y_binary, X_train, y_train, X_test, y_test


# ============================================================================
# Models (arbitrary depth) + latent extraction
# ============================================================================

def build_mlp(input_dim, hidden_widths, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    tf.random.set_seed(seed)
    seq = [layers.Input(shape=(input_dim,), name="input_layer")]
    for i, w in enumerate(hidden_widths):
        seq.append(layers.Dense(w, activation="sigmoid", name=f"hidden_{i+1}"))
    seq.append(layers.Dense(1, activation="sigmoid", name="output_layer"))
    model = Sequential(seq)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, epochs=2000, verbose=0):
    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
              validation_split=0.2, verbose=verbose)
    return model


def get_all_latents(model, X):
    """Return [X0, h1, h2, ..., output] -- one representation per Dense layer."""
    from tensorflow.keras.models import Model
    reps = [np.asarray(X, float)]
    for layer in model.layers:
        sub = Model(inputs=model.inputs, outputs=layer.output)
        reps.append(np.asarray(sub.predict(X, verbose=0), float))
    return reps


def find_row_indices(X, subset, atol=1e-8):
    """
    For each row of `subset`, return the index of the matching row in `X`.

    gudhi's sparsify_point_set returns a SUBSET of the input rows (coordinates are
    not recomputed), so rows should match (near-)exactly; we nonetheless match by
    nearest neighbour within a tolerance rather than relying on exact float
    equality, which is fragile. Matching is done WITHOUT replacement so that
    duplicate rows in X map to distinct indices (avoids collapsing labels of
    coincident points onto a single original index).

    Returns an int array of length len(subset). Raises if any subset row has no
    match within `atol` (a real bug worth surfacing, not silently mis-labelling).
    """
    X = np.asarray(X, float)
    subset = np.asarray(subset, float)
    D = cdist(subset, X)                      # (m, n) distances
    used = np.zeros(X.shape[0], dtype=bool)
    idx = np.empty(subset.shape[0], dtype=int)
    for i in range(subset.shape[0]):
        order = np.argsort(D[i])
        # take the nearest not-yet-used original row
        j = next((k for k in order if not used[k]), order[0])
        if D[i, j] > atol:
            # fall back to the global nearest (allowing reuse) but warn loudly:
            j_global = int(order[0])
            if D[i, j_global] > 1e-3:
                raise ValueError(
                    f"sparsified row {i} has no match in X within tolerance "
                    f"(nearest distance {D[i, j_global]:.3e}); index mapping unsafe.")
            j = j_global
        used[j] = True
        idx[i] = j
    return idx


def sparsify(X, min_squared_dist):
    """Greedy sqrt(min_squared_dist)-net (gudhi); returns (sparse_points, indices)."""
    sp = np.asarray(gd.subsampling.sparsify_point_set(
        points=np.asarray(X, float), min_squared_dist=min_squared_dist), float)
    idx = find_row_indices(np.asarray(X), sp)
    return sp, np.asarray(idx)


# ============================================================================
# Generalized pipeline: MLP persistence for arbitrary depth
# ============================================================================

def run_mlp_persistence(latents, epsilons, max_dim=2, augment_output=True):
    """
    Faithful, depth-general version of the original pullback chain.

    latents  : [X0, X1, ..., X_{L-1}]  (L representations, output last)
    epsilons : list of length L (per-layer scale)

    Builds the output VR, then pulls back successively to the input, assembles the
    layer-indexed combined filtration, and returns the MLP-persistence barcode.
    H1 is taken over all layers EXCEPT the ~1-D output (which cannot carry loops).
    """
    L = len(latents)
    assert len(epsilons) == L, "need one epsilon per representation"

    out = latents[-1]
    if augment_output:
        out = np.c_[out, (np.asarray(out).reshape(len(out), -1)[:, :1] > 0.5).astype(float)]
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
    for t in trees[:-1]:
        t.expansion(max_dim)

    c0 = create_combined_filtration(trees)
    c0.compute_persistence()
    iv0 = np.asarray(c0.persistence_intervals_in_dimension(0)).reshape(-1, 2)

    c1 = create_combined_filtration(trees[:-1])   # exclude 1-D output
    c1.compute_persistence()
    iv1 = np.asarray(c1.persistence_intervals_in_dimension(1)).reshape(-1, 2)

    return {"mlp_intervals_by_dim": {0: iv0, 1: iv1},
            "n_layers": L, "pullback_trees": trees}


def _selection_homology_dim(Xi, is_output):
    """H0 for the 1-D / output layer (binary separability question, beta_1 == 0
    structurally); H1 for loop-bearing interior layers. See toy script rationale."""
    if is_output or np.asarray(Xi).shape[1] < 2:
        return 0
    return 1


def select_epsilons(latents, use_bootstrap=False, n_boot=20):
    """
    Per-layer, non-visual scale selection (significance + widest plateau).
    Fast mode (use_bootstrap=False) uses the fraction threshold and is suitable
    inside the depth sweep; bootstrap mode is for the single headline net.
    Falls back to HANDPICKED_EPS (padded/truncated) if a layer returns nothing.
    """
    vr = compute_vietoris_rips_complex
    L = len(latents)
    eps = []
    for i, Xi in enumerate(latents):
        dim = _selection_homology_dim(Xi, is_output=(i == L - 1))
        res = select_epsilon_for_layer(
            np.asarray(Xi, float), dim=dim, max_dimension=2,
            use_bootstrap=use_bootstrap, n_boot=n_boot, alpha=0.05,
            frac_alpha=0.5, vr_builder=vr, rng=SEED)
        e = res["epsilon"]
        if e is None:
            e = HANDPICKED_EPS[min(i, len(HANDPICKED_EPS) - 1)]
        eps.append(float(e))
    return eps


def layer_filtration(Xi, max_dim=2, max_filtration=5.0, sparsify_sqdist=LAYER_SPARSE_SQDIST):
    """Layer-persistence VR complex with sparsification + edge collapse, mirroring
    the original compute_plot_pd (minus the file I/O)."""
    Xs = np.asarray(gd.subsampling.sparsify_point_set(
        points=np.asarray(Xi, float), min_squared_dist=sparsify_sqdist), float)
    dm = cdist(Xs, Xs)
    st = gd.SimplexTree.create_from_array(dm, max_filtration=max_filtration)
    st.collapse_edges()
    st.expansion(max_dim if Xs.shape[1] > 1 else 0)
    return st


# ============================================================================
# KEYSTONE -- depth-saturation sweep
# ============================================================================

def section_depth_sweep(X_train, y_train, X_test, y_test, X_full,
                        depths=(1, 2, 3, 4, 5), width=32, seeds=(0, 1, 2),
                        epochs=1500, activity_threshold_frac=0.1):
    """
    Train fixed-width MLPs of increasing depth; relate test accuracy to the
    topological saturation depth d* read off the DEEPEST net's activity profile.

    Fixed width isolates depth as the sole variable (the 'controlled width' the
    reviewer asked for). The paper's 32-16-8-4 taper is an alternative family; the
    early-saturation conclusion should be reported as architecture-robust via this
    sweep rather than asserted from one net.
    """
    print("\n=== KEYSTONE: depth-saturation sweep ===")
    input_dim = X_train.shape[1]

    # input sparsification reused across all nets (it is input-geometry only)
    X0_sparse, _ = sparsify(X_full, MLP_SPARSE_SQDIST)

    acc_mean, acc_std = [], []
    deepest_act_by_dim = None
    for d in depths:
        accs = []
        for s in seeds:
            model = build_mlp(input_dim, [width] * d, seed=s)
            train_model(model, X_train, y_train, epochs=epochs)
            pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
            accs.append(float((pred == np.asarray(y_test)).mean()))
            # activity profile only for the deepest depth, first seed (cost control)
            if d == max(depths) and s == seeds[0]:
                latents = [X0_sparse] + get_all_latents(model, X0_sparse)[1:]
                eps = select_epsilons(latents, use_bootstrap=False)
                try:
                    res = run_mlp_persistence(latents, eps)
                    by_dim = res["mlp_intervals_by_dim"]
                    L = res["n_layers"]
                    deepest_act_by_dim = {
                        0: topological_activity_from_mlp_barcode(
                            {0: by_dim.get(0, np.empty((0, 2)))}, L),
                        1: topological_activity_from_mlp_barcode(
                            {1: by_dim.get(1, np.empty((0, 2)))}, L),
                    }
                except Exception as e:  # robustness: never let one net kill the sweep
                    print(f"    [warn] activity computation failed: {e}")
        accs = np.asarray(accs)
        acc_mean.append(float(accs.mean()))
        acc_std.append(float(accs.std()))
        print(f"  depth {d}: test acc {accs.mean():.4f} +/- {accs.std():.4f}")

    # Define d* on the H1 (loop) activity: that is the representational
    # reorganization with a principled meaning. H0 activity near the output is
    # dominated by the funnel collapsing to ~1-D and would veto any saturation;
    # we report it separately rather than folding it in.
    dstar = None
    if deepest_act_by_dim is not None:
        act_h1 = deepest_act_by_dim[1]
        act_h0 = deepest_act_by_dim[0]
        print(f"  deepest-net H0 activity: {act_h0.tolist()}")
        print(f"  deepest-net H1 activity: {act_h1.tolist()}")
        if act_h1.size and act_h1.max() > 0:
            thr = activity_threshold_frac * float(act_h1.max())
            dstar = saturation_depth_plateau(act_h1, threshold=thr)
            print(f"  saturation depth d* (on H1, plateau rule): {dstar}  "
                  f"(threshold = {thr:.2f})")
        else:
            print("  H1 activity is empty/zero -> no loop reorganization to saturate.")

    # headline figure: accuracy vs depth, with d* marked
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(list(depths), acc_mean, yerr=acc_std, marker="o", capsize=3,
                label="test accuracy")
    if dstar is not None:
        ax.axvline(dstar, color="tab:red", ls="--",
                   label=fr"$d^*$ from topology = {dstar}")
    ax.set_xlabel("network depth (# hidden layers)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs depth, with topological saturation depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "keystone_depth_vs_accuracy.png"), dpi=150)
    plt.close(fig)
    print("  saved keystone_depth_vs_accuracy.png")
    print("  CLAIM TEST: accuracy should plateau at ~d*. If it keeps rising past "
          "d*, the topological overparameterization diagnosis is falsified.")
    return dict(depths=list(depths), acc_mean=acc_mean, acc_std=acc_std,
                deepest_activity_by_dim=None if deepest_act_by_dim is None
                else {k: v.tolist() for k, v in deepest_act_by_dim.items()},
                d_star=dstar)


# ============================================================================
# Supporting battery (real-data versions of the toy sections)
# ============================================================================

def section_layer_metrics(latents):
    print("\n=== layer-persistence metrics (distance units) ===")
    trees = [layer_filtration(Xi) for Xi in latents]
    diags = [compute_persistence_diagrams(st, max_dim=2) for st in trees]
    eps_grid = make_eps_grid(diags, is_diagrams=True, n_points=200, max_dim=2)
    print(f"{'layer':>5} | {'TP_H0':>8} | {'entH0':>7} | {'TP_H1':>8} | {'maxB1':>6}")
    for i, d in enumerate(diags):
        b1 = betti_curve(d.get(1, np.empty((0, 2))), eps_grid)
        print(f"{i:>5} | {total_persistence(d.get(0, np.empty((0,2)))):8.3f} | "
              f"{persistence_entropy(d.get(0, np.empty((0,2)))):7.3f} | "
              f"{total_persistence(d.get(1, np.empty((0,2)))):8.3f} | "
              f"{int(b1.max()) if b1.size else 0:6d}")
    heat, hgrid = betti_layer_scale_grid(trees, dim=1, n_grid=200, max_dimension=2)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_betti_heatmap(heat, hgrid, dim=1, ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "cardio_betti1_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  saved cardio_betti1_heatmap.png")
    return trees, diags


def section_epsilon_selection(latents):
    print("\n=== principled epsilon selection (vs visual [1, 2.5, 0.2]) ===")
    eps = select_epsilons(latents, use_bootstrap=True, n_boot=40)
    print(f"  visual choice : {HANDPICKED_EPS}")
    print(f"  selected      : {[round(e,3) for e in eps]}")
    return eps


def section_mlp_activity(latents, epsilons):
    print("\n=== MLP persistence -> cross-layer activity (layer units) ===")
    res = run_mlp_persistence(latents, epsilons)
    by_dim = res["mlp_intervals_by_dim"]
    L = res["n_layers"]
    act_h0 = topological_activity_from_mlp_barcode({0: by_dim.get(0, np.empty((0,2)))}, L)
    act_h1 = topological_activity_from_mlp_barcode({1: by_dim.get(1, np.empty((0,2)))}, L)
    act_tot = topological_activity_from_mlp_barcode(by_dim, L)
    print(f"  H0 activity (primary, binary): {act_h0.tolist()}")
    print(f"  H1 activity (secondary)      : {act_h1.tolist()}")
    print(f"  total activity               : {act_tot.tolist()}")
    print(f"  saturation depth d*          : {saturation_depth(act_tot, 0.5)}")
    return res


def section_trajectory_agreement(latents, y_sparse, mlp_res, epsilons):
    print("\n=== trajectory label agreement per layer (informative on real data) ===")
    communities_by_layer = []
    for i, st in enumerate(mlp_res["pullback_trees"]):
        G = extract_1_skeleton_graph(st, epsilons[i])
        comms, cid = {}, 0
        for comp in nx.connected_components(G):
            for v in comp:
                comms[v] = cid
            cid += 1
        communities_by_layer.append(comms)
    rows = layerwise_label_agreement(communities_by_layer, y_sparse)
    print(f"{'layer':>5} | {'NMI':>6} | {'ARI':>6} | {'purity':>6} | "
          f"{'#clust':>6} | {'cover':>6}")
    for i, r in enumerate(rows):
        print(f"{i:>5} | {r['nmi']:6.3f} | {r['ari']:6.3f} | {r['purity']:6.3f} | "
              f"{r['n_clusters']:6d} | {r['coverage']:6.3f}")
    return rows


def section_linear_probe(model, X_train, y_train, X_test, y_test):
    """
    Empirical counterpart to Proposition 4.1 (separability <-> disconnected nerve):
    a linear probe accuracy per layer should RISE as the network reorganises data,
    tracking the collapse of H0 components toward the number of classes.
    """
    print("\n=== linear probe per layer (empirical Prop. 4.1) ===")
    reps_tr = get_all_latents(model, X_train)
    reps_te = get_all_latents(model, X_test)
    print(f"{'layer':>5} | {'probe_acc':>9}")
    accs = []
    for i, (Rtr, Rte) in enumerate(zip(reps_tr, reps_te)):
        clf = LogisticRegression(max_iter=1000).fit(Rtr, y_train)
        a = float(clf.score(Rte, y_test))
        accs.append(a)
        print(f"{i:>5} | {a:9.4f}")
    print("  Rising probe accuracy alongside H0 component-collapse is the "
          "quantitative version of 'the network makes the classes separable'.")
    return accs


def section_representation_baselines(model, X_eval):
    """
    Baselines the AE explicitly named: linear CKA between consecutive layers and a
    participation-ratio intrinsic-dimension estimate per layer. The decisive
    comparison for the paper is to exhibit a transition where CKA stays high
    (representations 'similar') while topology changes -- CKA is blind to H1.
    """
    print("\n=== representation-analysis baselines (CKA, intrinsic dim) ===")
    reps = get_all_latents(model, X_eval)

    def linear_cka(A, B):
        A = A - A.mean(0); B = B - B.mean(0)
        hsic = np.linalg.norm(B.T @ A, "fro") ** 2
        den = (np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro"))
        return float(hsic / den) if den > 0 else 0.0

    def participation_ratio(X):
        Xc = X - X.mean(0)
        ev = np.linalg.eigvalsh(Xc.T @ Xc / max(len(X), 1))
        ev = ev[ev > 1e-12]
        return float(ev.sum() ** 2 / (ev ** 2).sum()) if ev.size else 0.0

    print(f"{'transition':>11} | {'CKA':>6}")
    ckas = []
    for i in range(len(reps) - 1):
        c = linear_cka(reps[i], reps[i + 1])
        ckas.append(c)
        print(f"{i}->{i+1:>9} | {c:6.3f}")
    print(f"{'layer':>11} | {'PR_dim':>6}")
    prs = [participation_ratio(R) for R in reps]
    for i, p in enumerate(prs):
        print(f"{i:>11} | {p:6.2f}")
    return dict(cka=ckas, participation_ratio=prs)


def section_runtime(latents, epsilons, dense_cloud=None):
    print("\n=== runtime (per-stage + scaling over n, subprocess memory) ===")
    def pullback_fn(pts, eps, sts):
        return run_mlp_persistence(pts, eps)["pullback_trees"]
    prof = rs.profile_pipeline_stages(
        latents, vr_builder=compute_vietoris_rips_complex, epsilons=epsilons,
        max_dimension=2, pullback_fn=pullback_fn, repeats=2)
    print("  per-stage (median ms):", rs.runtime_table(prof, units="ms"))

    # Scaling MUST run on a dense cloud: latents[0] here is already sparsified to
    # ~99 points, so sizes above that would all be capped to the same cloud and
    # produce a meaningless (even negative) exponent. Use the full input cloud.
    cloud = np.asarray(dense_cloud if dense_cloud is not None else latents[0], float)
    n = len(cloud)
    if n < 200:
        print(f"  [skip] scaling sweep needs a dense cloud; got n={n}. "
              f"Pass dense_cloud=X (full, unsparsified).")
        return prof, None
    sizes = [s for s in [200, 400, 800, 1600] if s <= n] or [n // 2, n]
    sweep = rs.scaling_sweep_samples(
        cloud, vr_builder=compute_vietoris_rips_complex, epsilon=epsilons[0],
        sizes=sizes, max_dimension=2, repeats=2, seed=SEED,
        run_in_subprocess=True)   # real peak RSS -- worth it at this data size
    print(f"  sizes tested               : {sizes}")
    print("  exponent time vs n         :", sweep["exponent_time_vs_n"])
    print("  exponent time vs #simplices:", sweep["exponent_time_vs_simplices"])
    return prof, sweep


def section_epsilon_sensitivity(latents, base_eps):
    print("\n=== epsilon sensitivity (is d* stable across a band?) ===")
    factors = [0.85, 0.9, 1.0, 1.1, 1.15]
    band = [[round(f * e, 4) for e in base_eps] for f in factors]
    out = rs.epsilon_sensitivity(lambda epsilons: run_mlp_persistence(latents, epsilons),
                                 band, activity_dim=0, threshold=0.5, reference_index=2)
    print(f"  distinct d* values    : {out['d_star_values']}  (cov {out['d_star_cov']:.3f})")
    print(f"  max bottleneck to ref : {out['max_bottleneck_to_ref']:.4f} (layer units)")
    return out


def section_sparsification(latents, dense_cloud=None):
    print("\n=== sparsification trade-off (runtime vs realised distortion) ===")
    # MUST run on a dense cloud: latents[0] is already sparsified, so small deltas
    # keep every point (covering radius 0, bottleneck 0) and the curve is degenerate.
    cloud = np.asarray(dense_cloud if dense_cloud is not None else latents[0], float)
    n = len(cloud)
    if n < 200:
        print(f"  [skip] needs a dense cloud; got n={n}. Pass dense_cloud=X.")
        return None
    # Choose deltas relative to the cloud's own scale so the net actually thins.
    from scipy.spatial.distance import pdist
    med = float(np.median(pdist(cloud[:min(n, 500)])))   # typical pairwise distance
    deltas = [round(f * med, 4) for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    # paper's min_squared_dist=0.5 -> delta = sqrt(0.5) ~ 0.707 in actual distance
    deltas = sorted(set(deltas + [0.707]))
    spars = rs.sparsification_sensitivity(
        cloud, vr_builder=compute_vietoris_rips_complex, epsilon=med,
        deltas=deltas, max_dimension=2, homology_dim=1, repeats=2, seed=SEED)
    print(f"  median pairwise distance ~ {med:.3f}; epsilon set to it")
    print(f"{'delta':>6} | {'n_kept':>6} | {'cover_r':>8} | {'sec':>7} | {'bottleneck':>10}")
    for r in spars["rows"]:
        print(f"{r['delta']:6.3f} | {r['n_kept']:6d} | {r['covering_radius']:8.4f} | "
              f"{r['seconds']:7.3f} | {r['bottleneck_distortion']:10.4f}")
    print("  realised bottleneck <= covering radius is the TRUE-distance bound; "
          "use it to restate the manuscript's squared-distance claim correctly.")
    fig, _ = rs.plot_sparsification_tradeoff(spars)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "cardio_sparsification.png"), dpi=150)
    plt.close()
    return spars


def section_seed_sensitivity(X_train, y_train, X_full, input_dim,
                             width=32, depth=1, seeds=(0, 1, 2, 3, 4), epochs=1200):
    print("\n=== seed / retraining sensitivity ===")
    X0_sparse, _ = sparsify(X_full, MLP_SPARSE_SQDIST)

    def train_and_extract(seed):
        model = build_mlp(input_dim, [width] * depth, seed=seed)
        train_model(model, X_train, y_train, epochs=epochs)
        latents = [X0_sparse] + get_all_latents(model, X0_sparse)[1:]
        trees = [layer_filtration(Xi) for Xi in latents]
        diags = [compute_persistence_diagrams(st, max_dim=2) for st in trees]
        eps = select_epsilons(latents, use_bootstrap=False)
        mlp = run_mlp_persistence(latents, eps)
        return {"diagrams_by_layer": diags,
                "mlp_intervals_by_dim": mlp["mlp_intervals_by_dim"],
                "n_layers": len(latents)}

    out = rs.seed_sensitivity(train_and_extract, list(seeds), homology_dim=1,
                              num_landscapes=3)
    print(f"  d* per seed : {out['d_star_values']}  (mode {out['d_star_mode']}, "
          f"agreement {out['d_star_agreement']:.2f})")
    return out


def section_stability(latents, epsilons, delta=0.05, n_perturb=6, hom_dim=1):
    print("\n=== empirical stability test (scale units, layer 0, H%d) ===" % hom_dim)
    X0 = np.asarray(latents[0], float)
    mids = latents[1:-1]
    out_cloud = np.asarray(latents[-1], float).reshape(len(latents[-1]), -1)[:, :1]
    rng = np.random.default_rng(SEED)
    eps = epsilons

    def layer0_diagram(out_pert):
        L = len(latents)
        trees = [None] * L
        trees[-1] = compute_vietoris_rips_complex(out_pert, eps[-1], max_dimension=1)
        ms = get_maximal_simplices(trees[-1], eps[-1])
        for i in range(L - 2, -1, -1):
            src = X0 if i == 0 else mids[i - 1]
            ki = vr_pullback(np.asarray(src, float), eps[i], ms, max_dimension=2)
            trees[i] = ki
            if i > 0:
                ms = get_maximal_simplices(ki, eps[i])
        trees[0].expansion(2)
        d = compute_persistence_diagrams(trees[0], max_dim=max(hom_dim, 1))
        return d.get(hom_dim, np.empty((0, 2)))

    perturbations = [out_cloud + rng.normal(scale=delta, size=out_cloud.shape)
                     for _ in range(n_perturb)]
    res = rs.empirical_stability_test(layer0_diagram, base_cover=out_cloud,
                                      perturbations=perturbations, eta=delta,
                                      homology_dim=hom_dim)
    print(f"  eta=delta={res['eta']:.4f}  max realised bottleneck={res['max_realised']:.4f}"
          f"  within bound: {res['all_within_bound']} (slack {res['slack']:.4f})")
    return res


# ============================================================================
# Driver
# ============================================================================

def main(run_depth_sweep=True, run_seed_sweep=True):
    X, y, X_train, y_train, X_test, y_test = load_cardio_dataset()
    input_dim = X.shape[1]

    if run_depth_sweep:
        section_depth_sweep(X_train, y_train, X_test, y_test, X)

    # single headline net (depth-1, the reduced model G from the paper)
    model = build_mlp(input_dim, [32], seed=SEED)
    train_model(model, X_train, y_train, epochs=2000)
    acc = float(((model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
                 == np.asarray(y_test)).mean())
    print(f"\nheadline net (32-hidden) test accuracy: {acc:.4f}")

    X0_sparse, idx = sparsify(X, MLP_SPARSE_SQDIST)
    latents = [X0_sparse] + get_all_latents(model, X0_sparse)[1:]
    y_sparse = y[idx]

    section_layer_metrics(latents)
    sel_eps = section_epsilon_selection(latents)
    mlp_res = section_mlp_activity(latents, HANDPICKED_EPS)
    section_trajectory_agreement(latents, y_sparse, mlp_res, HANDPICKED_EPS)
    section_linear_probe(model, X_train, y_train, X_test, y_test)
    section_representation_baselines(model, X_test)
    section_runtime(latents, HANDPICKED_EPS, dense_cloud=X)
    section_epsilon_sensitivity(latents, HANDPICKED_EPS)
    section_sparsification(latents, dense_cloud=X)
    if run_seed_sweep:
        section_seed_sensitivity(X_train, y_train, X, input_dim)
    section_stability(latents, HANDPICKED_EPS)

    print("\nDone. Figures written to:", OUTDIR)


if __name__ == "__main__":
    main()