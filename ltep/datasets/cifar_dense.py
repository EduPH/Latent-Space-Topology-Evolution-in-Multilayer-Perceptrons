#!/usr/bin/env python
# coding: utf-8
"""
Experiment_cifar_dense.py
=========================

CIFAR-10 with a WEAK conv front and a DEEP dense head, so that dense depth does
genuine work and can therefore saturate -- the precondition for any meaningful
"prune the inert layers" claim. The topological depth diagnostic is computed ONCE
on the deepest net; an actual depth sweep is run ONCE, only to VALIDATE that the
cheap diagnostic predicts the accuracy plateau. After that validation a
practitioner would trust the diagnostic and skip the sweep -- that one-shot
diagnosis replacing an N-fold retraining sweep is the contribution.

DISCIPLINE ENFORCED IN CODE (this is the whole point):
  1. The d* extraction rule and its single parameter are PRE-COMMITTED below and
     must NOT be retuned after seeing the accuracy curve. Doing so would make the
     "prediction" circular and worthless.
  2. A STABILITY GATE recomputes d* on several independent analysis subsamples of
     the one cached net. The expensive depth sweep is SKIPPED unless d* is stable.
     An unstable predictor predicts nothing; we refuse to spend compute validating
     it, and report the instability honestly.

STABLE topological signal used (NOT the cross-layer MLP-persistence activity, which
we found unstable on real data): the bottleneck distance between consecutive,
diameter-normalised per-layer H1 persistence diagrams. Bottleneck is 1-Lipschitz
and the diagrams of a fixed network on resampled points are stable, so this change
signal is well behaved. Diameter-normalisation makes it measure REORGANISATION
rather than the layer's overall rescaling (caveat: normalisation removes global
scale, not every scale-related effect; we state this).

Requires (same dir): topological_metrics.py, runtime_sensitivity.py, VR_trajectories.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

SEED = 1234
np.random.seed(SEED)

import gudhi as gd
import networkx as nx
from sklearn.linear_model import LogisticRegression

from ..vr import (
    compute_vietoris_rips_complex, extract_1_skeleton_graph,
    get_maximal_simplices, vr_pullback, create_combined_filtration,
)
from ..metrics import (
    compute_persistence_diagrams, _finite_bars,
    betti_curve, make_eps_grid,
    total_persistence, persistence_entropy,
    layerwise_label_agreement,
    select_epsilon_plateau,
    significance_threshold_bootstrap,
)
from ..runtime import _safe_bottleneck
from .. import runtime as rs
from ._mlp_persistence import (
    convergence_layer, plot_mlp_barcode,
)

OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_CLASSES = 10
POINTS_PER_CLASS = 60                  # 60*10 = 600 analysed points (VR-tractable)
DENSE_DEPTH_MAX = 10                    # deepest dense head; sweep covers 1..this
DENSE_WIDTH = 128
NORM_MAXFILT = 0.6                     # cap normalised filtration (bounds expansion cost)

# ----- PARAMETER-FREE CONVERGENCE RULE (no tunable threshold) -------------------
# d* is read directly from the single MLP-persistence barcode (layer-indexed
# pullback tower): the last layer at which a GENUINE bar (layer-lifespan >= 2, the
# minimal structural span) is born or dies, over H0 and H1. Layers beyond d* are
# inert. There is no parameter to tune; the only stability question is whether the
# analysis SUBSAMPLE moves d*, which the gate checks.
PULLBACK_MAXDIM = 2
# --------------------------------------------------------------------------------


def model_path(depth, width=DENSE_WIDTH):
    return os.path.join(OUTDIR, f"cifar_cnn_d{depth}_w{width}.keras")


# ============================================================================
# Data
# ============================================================================

def load_cifar10():
    from tensorflow.keras.datasets import cifar10
    (Xtr, ytr), (Xte, yte) = cifar10.load_data()
    Xtr = Xtr.astype("float32") / 255.0
    Xte = Xte.astype("float32") / 255.0
    return Xtr, ytr.ravel().astype(int), Xte, yte.ravel().astype(int)


def class_balanced_subsample(X, y, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N_CLASSES):
        cls = np.where(y == c)[0]
        idx.extend(rng.choice(cls, size=min(per_class, len(cls)), replace=False).tolist())
    idx = np.array(sorted(idx))
    return X[idx], y[idx], idx


# ============================================================================
# Model: weak conv front + deep dense head
# ============================================================================

def head_layer_names(dense_depth):
    """Representations analysed, in order: gap entry, each dense layer, output."""
    return ["gap_features"] + [f"dense_{i+1}" for i in range(dense_depth)] + ["output"]


def build_cnn(dense_depth, width=DENSE_WIDTH, seed=SEED):
    """
    Deliberately WEAK conv front (one block) so the dense head has real work to do
    and dense depth can plausibly help, then saturate. A strong conv stack would
    leave the head inert and make any 'pruning' result trivial.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    tf.random.set_seed(seed)
    seq = [
        layers.Input(shape=(32, 32, 3), name="image_input"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(name="gap_features"),
    ]
    for i in range(dense_depth):
        seq.append(layers.Dense(width, activation="relu", name=f"dense_{i+1}"))
    seq.append(layers.Dense(N_CLASSES, activation="softmax", name="output"))
    model = Sequential(seq)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_or_load(Xtr, ytr, Xte, yte, dense_depth, epochs=30, force_retrain=False):
    from tensorflow.keras.models import load_model
    p = model_path(dense_depth)
    if os.path.exists(p) and not force_retrain:
        print(f"  loading cached: {os.path.basename(p)}")
        model = load_model(p)
    else:
        model = build_cnn(dense_depth)
        model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs,
                  batch_size=128, verbose=2)
        model.save(p)
    acc = float(model.evaluate(Xte, yte, verbose=0)[1])
    return model, acc


def head_representations(model, X, layer_names):
    from tensorflow.keras.models import Model
    outs = [model.get_layer(n).output for n in layer_names]
    extractor = Model(inputs=model.inputs, outputs=outs)
    reps = extractor.predict(X, verbose=0)
    if not isinstance(reps, list):
        reps = [reps]
    return [np.asarray(r, float) for r in reps]


def diameter_normalize(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P.copy()
    diam = float(pdist(P).max())
    return P / diam if diam > 0 else P.copy()


def layer_filtration(P, max_dim=2, maxfilt=NORM_MAXFILT):
    """VR complex of a diameter-normalised cloud, filtration capped to bound cost."""
    Q = diameter_normalize(P)
    dm = cdist(Q, Q)
    st = gd.SimplexTree.create_from_array(dm, max_filtration=maxfilt)
    st.collapse_edges()
    st.expansion(max_dim if Q.shape[1] > 1 else 0)
    return st


def layer_diagrams(reps, max_dim=2):
    return [compute_persistence_diagrams(layer_filtration(P, max_dim), max_dim=max_dim)
            for P in reps]


# ============================================================================
# MLP-persistence pullback tower -> parameter-free convergence diagnostic
# ============================================================================

def select_layer_epsilons(reps):
    """
    One epsilon per representation for the pullback tower. We use a geometry-driven,
    parameter-free choice: the median pairwise distance of each (diameter-normalised)
    cloud. This connects the cloud just past its typical nearest-neighbour scale --
    a standard, defensible default that needs no tuning and adapts per layer.
    """
    eps = []
    for P in reps:
        Q = diameter_normalize(P)
        dist = pdist(Q)
        eps.append(float(np.median(dist)) if dist.size else 0.3)
    return eps


def build_pullback_tower(reps, epsilons, max_dim=PULLBACK_MAXDIM):
    """
    Build the pullback tower (shared vertex indices across layers), output -> input,
    exactly as in the paper's MLP-persistence construction. reps are ordered
    input -> ... -> output; we diameter-normalise each before building so scales are
    comparable. Returns the list of per-layer SimplexTrees (the tower).
    """
    Q = [diameter_normalize(P) for P in reps]
    L = len(Q)
    trees = [None] * L
    trees[-1] = compute_vietoris_rips_complex(Q[-1], epsilons[-1], max_dimension=1)
    ms = get_maximal_simplices(trees[-1], epsilons[-1])
    for i in range(L - 2, -1, -1):
        ki = vr_pullback(Q[i], epsilons[i], ms, max_dimension=max_dim)
        trees[i] = ki
        if i > 0:
            ms = get_maximal_simplices(ki, epsilons[i])
    for t in trees[:-1]:
        t.expansion(max_dim)
    return trees


def mlp_persistence_convergence(reps, epsilons=None, max_dim=PULLBACK_MAXDIM):
    """
    Build the tower, assemble the layer-indexed combined filtration, and read the
    parameter-free convergence layer d* directly from the single barcode.
    Returns the diagnostic dict (see mlp_persistence_convergence.convergence_layer).
    """
    if epsilons is None:
        epsilons = select_layer_epsilons(reps)
    trees = build_pullback_tower(reps, epsilons, max_dim=max_dim)
    combined = create_combined_filtration(trees, max_dimension=max_dim)
    res = convergence_layer(combined, n_layers=len(trees), max_dim=min(max_dim, 1))
    res["epsilons"] = epsilons
    return res


def compute_dstar_for_net(model, Xpool, ypool, per_class, subsample_seed,
                          dense_depth):
    """One convergence-d* computation on a given net for a given analysis subsample."""
    names = head_layer_names(dense_depth)
    Xsub, ysub, _ = class_balanced_subsample(Xpool, ypool, per_class, subsample_seed)
    reps = head_representations(model, Xsub, names)   # includes output
    res = mlp_persistence_convergence(reps)
    return res["d_star"], res


# ============================================================================
# STABILITY GATE -- blocks the expensive sweep unless d* is invariant
# ============================================================================

def stability_gate(model, Xpool, ypool, dense_depth, per_class=POINTS_PER_CLASS,
                   k_repeats=5):
    print("\n=== STABILITY GATE: is the convergence d* invariant across subsamples? ===")
    print(f"  (parameter-free rule: last genuine birth/death in the MLP-persistence "
          f"barcode, lifespan>=2) -- recomputing d* on {k_repeats} independent "
          f"class-balanced subsamples of the SAME net")
    dstars, last = [], None
    for j in range(k_repeats):
        dstar, res = compute_dstar_for_net(
            model, Xpool, ypool, per_class, subsample_seed=1000 + j,
            dense_depth=dense_depth)
        dstars.append(dstar); last = res
        print(f"  subsample {j}: d*_by_dim={res['d_star_by_dim']}  "
              f"n_genuine={res['n_genuine_by_dim']}  -> d*={dstar}  "
              f"inert={res['inert_layers']}")
    valid = [d for d in dstars if d is not None]
    if not valid:
        print("  GATE FAILED: no genuine bars -> no convergence layer (empty H0/H1?).")
        return False, dstars, last
    mode = max(set(valid), key=valid.count)
    agreement = valid.count(mode) / len(dstars)
    stable = (agreement >= 0.8)
    print(f"  d* values: {dstars}  | mode={mode}  agreement={agreement:.2f}")
    print(f"  GATE {'PASSED' if stable else 'FAILED'} "
          f"(need >= 0.80 agreement to validate via the sweep)")
    if not stable:
        print("  -> Subsampling moves d*: the convergence layer is not robust on this "
              "net. We do NOT spend compute on the sweep; report this honestly.")
    return stable, dstars, last


# ============================================================================
# HEADLINE (a) FIXED: beta_0 plateau (honest cluster count) + H0 scalar summaries
# ============================================================================

def headline_separability(reps, layer_names):
    """
    Replaces the broken significant-H0-COUNT (which read distance-concentration
    noise, returning ~n/2). We instead report concentration-robust TOPOLOGICAL
    SUMMARIES per layer:
      * TP_H0, entropy_H0  -- stable scalars; TP_H0 falling = components consolidating
      * beta_0 at the WIDEST PLATEAU of its scale-curve -- the honest cluster count
        at the most stable resolution (vs N_CLASSES). 'None' if no plateau exists,
        which is itself an honest answer (no single scale resolves the clusters).
    """
    print("\n=== HEADLINE (a): separability via topological summaries ===")
    diags = layer_diagrams(reps, max_dim=1)
    eps_grid = make_eps_grid(diags, is_diagrams=True, n_points=200, max_dim=1)
    print(f"{'layer':>12} | {'TP_H0':>8} | {'entH0':>7} | {'b0@plateau':>10} | {'target':>6}")
    out = []
    for name, d in zip(layer_names, diags):
        b0 = betti_curve(d.get(0, np.empty((0, 2))), eps_grid)
        eps, plateau_val, _ = select_epsilon_plateau(eps_grid, b0, target_value=None)
        tp0 = total_persistence(d.get(0, np.empty((0, 2))))
        ent0 = persistence_entropy(d.get(0, np.empty((0, 2))))
        out.append(dict(layer=name, tp_h0=tp0, ent_h0=ent0, b0_plateau=plateau_val))
        print(f"{name:>12} | {tp0:8.3f} | {ent0:7.3f} | "
              f"{('--' if plateau_val is None else plateau_val):>10} | {N_CLASSES:>6d}")
    print("  TP_H0 should fall through the head (components consolidating); the "
          "beta_0 plateau value is the honest cluster count -- compare to "
          f"{N_CLASSES}. Concentration-robust (no significance thresholding).")
    return out, diags


# ============================================================================
# HEADLINE (b) FIXED: drop trivial maxB0; keep TP and maxB1
# ============================================================================

def headline_betti_collapse(reps, layer_names, max_dim=2):
    print("\n=== HEADLINE (b): persistence collapse across the head ===")
    diags = layer_diagrams(reps, max_dim=max_dim)
    eps_grid = make_eps_grid(diags, is_diagrams=True, n_points=200, max_dim=max_dim)
    print(f"{'layer':>12} | {'TP_H0':>8} | {'TP_H1':>8} | {'maxB1':>6}")
    for name, d in zip(layer_names, diags):
        b1 = betti_curve(d.get(1, np.empty((0, 2))), eps_grid)
        print(f"{name:>12} | {total_persistence(d.get(0, np.empty((0,2)))):8.3f} | "
              f"{total_persistence(d.get(1, np.empty((0,2)))):8.3f} | "
              f"{int(b1.max()) if b1.size else 0:6d}")
    print("  (maxB0 dropped: it is trivially n at the smallest scale and carries "
          "no information. TP_H1 collapse + maxB1 are the loop-destruction signal.)")
    return diags


# ============================================================================
# HEADLINE (c): CKA vs topological change
# ============================================================================

def headline_cka_vs_topology(reps, diags, layer_names):
    print("\n=== HEADLINE (c): CKA vs topological change ===")

    def linear_cka(A, B):
        A = A - A.mean(0); B = B - B.mean(0)
        num = np.linalg.norm(B.T @ A, "fro") ** 2
        den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
        return float(num / den) if den > 0 else 0.0

    tp1 = [total_persistence(d.get(1, np.empty((0, 2)))) for d in diags]
    print(f"{'transition':>22} | {'CKA':>6} | {'dTP_H1':>8} | {'btl_H1':>7}")
    rows = []
    for i in range(len(reps) - 1):
        c = linear_cka(reps[i], reps[i + 1])
        dtp = tp1[i + 1] - tp1[i]
        btl = _safe_bottleneck(diags[i].get(1, np.empty((0, 2))),
                               diags[i + 1].get(1, np.empty((0, 2))))
        label = f"{layer_names[i]}->{layer_names[i+1]}"
        rows.append((label, c, dtp, btl))
        print(f"{label:>22} | {c:6.3f} | {dtp:8.3f} | {btl:7.4f}")
    print("  HIGH CKA + large |dTP_H1| (or btl) = topology sees reorganisation CKA "
          "reports as mild. CKA is invariant to orthogonal maps and blind to H1.")
    return rows


# ============================================================================
# Probe (robust separability corroboration)
# ============================================================================

def supporting_linear_probe(model, Xtr, ytr, Xte, yte, layer_names):
    print("\n=== linear probe per layer (concentration-robust separability) ===")
    Rtr = head_representations(model, Xtr, layer_names)
    Rte = head_representations(model, Xte, layer_names)
    print(f"{'layer':>12} | {'probe_acc':>9}")
    accs = []
    for name, A, B in zip(layer_names, Rtr, Rte):
        clf = LogisticRegression(max_iter=2000).fit(A, ytr)
        a = float(clf.score(B, yte))
        accs.append(a); print(f"{name:>12} | {a:9.4f}")
    print("  Probe measures LINEAR separability (a weaker, permissive cousin of H0 "
          "disconnection). It CORROBORATES the separability trend; it does not prove "
          "topological disconnection. Keep that distinction in the paper.")
    return accs


# ============================================================================
# VALIDATION depth sweep -- GATED behind stability
# ============================================================================

def validation_depth_sweep(Xtr, ytr, Xte, yte, predicted_dstar,
                           depths=tuple(range(1, DENSE_DEPTH_MAX + 1)),
                           epochs=30, force_retrain=False):
    print("\n=== VALIDATION: depth sweep (does accuracy plateau at d*?) ===")
    # predicted_dstar is a CONVERGENCE LAYER INDEX into [gap=0, dense_1=1, ...,
    # dense_L=L, output=L+1]. The number of dense layers to keep is that index
    # minus the gap offset (gap is index 0, so dense layer k sits at index k).
    # A convergence layer d* therefore predicts keeping d* dense layers (clipped
    # to the valid range). output-index convergence would mean 'all layers needed'.
    if predicted_dstar is None:
        keep = None
    else:
        keep = int(np.clip(predicted_dstar, 1, DENSE_DEPTH_MAX))
    print(f"  convergence layer index = {predicted_dstar}  ->  predicted dense "
          f"layers to keep = {keep}")
    accs = []
    for d in depths:
        model, acc = train_or_load(Xtr, ytr, Xte, yte, dense_depth=d,
                                   epochs=epochs, force_retrain=force_retrain)
        accs.append(acc)
        print(f"  dense_depth {d}: test acc {acc:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(depths), accs, "o-", label="test accuracy")
    if keep is not None:
        ax.axvline(keep, color="tab:red", ls="--",
                   label=f"predicted dense depth $d^*$ = {keep}")
    ax.set_xlabel("dense-head depth"); ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs dense depth, with predicted d*")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "cifar_depth_validation.png"), dpi=150)
    plt.close(fig)
    print("  saved cifar_depth_validation.png")
    print("  CLAIM TEST: accuracy should be flat for dense_depth >= d*. If it keeps "
          "rising past d*, the topological diagnosis is FALSIFIED -- report that.")
    return dict(depths=list(depths), accs=accs, predicted_keep=keep)


# ============================================================================
# Driver
# ============================================================================

def main(epochs=60, force_retrain=False, run_sweep=True,
         per_class=POINTS_PER_CLASS):
    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"CIFAR-10: train {Xtr.shape}, test {Xte.shape}")

    # deepest net: the single training the diagnostic reads
    print(f"\n--- deepest net (dense_depth={DENSE_DEPTH_MAX}) ---")
    model, acc = train_or_load(Xtr, ytr, Xte, yte, dense_depth=DENSE_DEPTH_MAX,
                               epochs=epochs, force_retrain=force_retrain)
    print(f"  test accuracy: {acc:.4f}")
    names = head_layer_names(DENSE_DEPTH_MAX)

    # analysis subsample (test set, held out)
    Xsub, ysub, _ = class_balanced_subsample(Xte, yte, per_class, seed=SEED)
    reps = head_representations(model, Xsub, names)
    print(f"  analysis subsample: {len(Xsub)} pts; clouds:",
          {n: r.shape for n, r in zip(names, reps)})

    # headline (stable) triple
    sep_summary, _ = headline_separability(reps, names)
    diags_b = headline_betti_collapse(reps, names)
    headline_cka_vs_topology(reps, diags_b, names)

    # probe
    supporting_linear_probe(model, Xtr[:5000], ytr[:5000], Xte, yte, names)

    # cross-layer: MLP-persistence convergence (parameter-free), headline subsample
    print("\n=== MLP-persistence convergence (claim 1: inert layers) ===")
    conv = mlp_persistence_convergence(reps)
    print(f"  layer epsilons (median pairwise): {np.round(conv['epsilons'], 3).tolist()}")
    print(f"  d*_by_dim={conv['d_star_by_dim']}  n_genuine={conv['n_genuine_by_dim']}")
    print(f"  convergence layer d*={conv['d_star']}  inert layers={conv['inert_layers']}")
    print("  (d* read directly from the single layer-indexed barcode; layers beyond "
          "d* produce no genuine birth/death -> candidates for pruning.)")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_mlp_barcode(conv["bars_by_dim"], n_layers=len(names), d_star=conv["d_star"], ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "cifar_mlp_barcode.png"), dpi=150)
    plt.close(fig)
    print("  saved cifar_mlp_barcode.png (faint bars = lifespan-1 flickers, filtered)")

    # stability gate -> only then the expensive validation sweep
    stable, dstars, _ = stability_gate(model, Xte, yte, DENSE_DEPTH_MAX, per_class)
    predicted = max(set([d for d in dstars if d is not None]),
                    key=[d for d in dstars if d is not None].count) \
        if any(d is not None for d in dstars) else None

    if run_sweep and stable:
        validation_depth_sweep(Xtr, ytr, Xte, yte, predicted_dstar=predicted,
                               epochs=epochs, force_retrain=force_retrain)
    elif run_sweep and not stable:
        print("\n[sweep skipped by stability gate -- see above]")

    print("\nDone. Figures written to:", OUTDIR)


if __name__ == "__main__":
    main()