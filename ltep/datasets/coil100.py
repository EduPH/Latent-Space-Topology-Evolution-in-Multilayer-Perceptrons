#!/usr/bin/env python
# coding: utf-8
"""
Experiment_coil100.py
=====================

COIL-100 with VERIFIABLE topology. Each object is photographed at 72 angles over a
full 360 deg rotation, so each object's image set traces a CLOSED LOOP (a circle)
in representation space -- a genuine H1 generator we KNOW is present by
construction. This is the first dataset in the sequence where H1 is ground truth,
not hoped for.

We use 10 objects, ALL 72 angles each (720 points). Subsampling note: unlike CIFAR
we must NOT randomly subsample images -- that would destroy the loops. We keep
every angle per object so each rotation circle stays densely sampled.

THREE THINGS THIS EXPERIMENT ESTABLISHES:
  (V) VERIFICATION: per object, H1 at the input/GAP representation should show ONE
      strong, persistent loop. If the framework cannot recover a circle we KNOW is
      there, nothing downstream is trustworthy. This is the missing ground-truth
      check.
  (1) INERT LAYERS (claim 1): the parameter-free MLP-persistence convergence layer
      d*, read from the single layer-indexed barcode of the pullback tower.
  (2) FEATURE TRACKING (claim 2): track each object's rotation loop (its longest H1
      bar) through the dense head to locate WHERE rotation-invariance emerges, i.e.
      where the loop is destroyed. Tracking a feature whose existence is certified.

A classifier MUST become rotation-invariant to recognise objects, so the rotation
loop MUST die somewhere in the head. Finding where is the scientific payoff, and it
is verifiable because we put the loop there.

Requires (same dir): topological_metrics.py, mlp_persistence_convergence.py,
VR_trajectories.py.  Data: tensorflow_datasets 'coil100' (auto-download) or a local
COIL-100 image directory (obj{N}__{angle}.png).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

SEED = 1234
np.random.seed(SEED)

import gudhi as gd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ..vr import (
    compute_vietoris_rips_complex, get_maximal_simplices, vr_pullback,
    create_combined_filtration,
)
from ..metrics import (
    compute_persistence_diagrams, _finite_bars,
    betti_curve, make_eps_grid, total_persistence,
)
from ._mlp_persistence import convergence_layer, plot_mlp_barcode

OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_OBJECTS = 10
N_ANGLES = 72
IMG_SIZE = 64
DENSE_DEPTH = 4
DENSE_WIDTH = 128
RAW_PCA_DIM = 50
NORM_MAXFILT = 0.7
MODEL_PATH = os.path.join(OUTDIR, f"coil_cnn_{N_OBJECTS}obj.keras")


# ============================================================================
# Data
# ============================================================================

def load_coil100(n_objects=N_OBJECTS, img_size=IMG_SIZE):
    """
    Load COIL-100, keeping ALL angles for the first `n_objects` objects.
    Tries tensorflow_datasets first, then a local directory of obj{N}__{ang}.png.
    Returns X (N,img,img,3) float [0,1], obj_id (N,) int 0..n_objects-1,
    angle_idx (N,) int 0..71.
    """
    # --- try tensorflow_datasets ---
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        ds = tfds.load("coil100", split="train")
        imgs, objs, angs = [], [], []
        for ex in tfds.as_numpy(ds):
            oid = int(ex["object_id"])
            if oid >= n_objects:
                continue
            # angle_label is 0..71; fall back to angle/5 if absent
            if "angle_label" in ex:
                a = int(ex["angle_label"])
            else:
                a = int(ex["angle"]) // 5
            im = ex["image"].astype("float32") / 255.0
            im = tf.image.resize(im, (img_size, img_size)).numpy()
            imgs.append(im); objs.append(oid); angs.append(a)
        X = np.asarray(imgs, "float32")
        obj = np.asarray(objs, int); ang = np.asarray(angs, int)
        if len(X):
            print(f"loaded via tfds: {X.shape}, {n_objects} objects")
            return X, obj, ang
    except Exception as e:
        print(f"  tfds load failed ({e}); trying local directory")

    # --- local directory fallback ---
    import glob, re
    from PIL import Image
    candidates = [os.path.join(OUTDIR, "coil-100"), os.path.join(OUTDIR, "coil100"),
                  os.environ.get("COIL100_DIR", "")]
    root = next((c for c in candidates if c and os.path.isdir(c)), None)
    if root is None:
        raise FileNotFoundError(
            "COIL-100 not found. Install `tensorflow_datasets` or set COIL100_DIR "
            "to a folder of obj{N}__{angle}.png files.")
    imgs, objs, angs = [], [], []
    for f in sorted(glob.glob(os.path.join(root, "obj*__*.png"))):
        m = re.search(r"obj(\d+)__(\d+)\.png", os.path.basename(f))
        if not m:
            continue
        oid = int(m.group(1)) - 1            # files are 1-indexed
        if oid >= n_objects:
            continue
        a = int(m.group(2)) // 5
        im = Image.open(f).convert("RGB").resize((img_size, img_size))
        imgs.append(np.asarray(im, "float32") / 255.0); objs.append(oid); angs.append(a)
    X = np.asarray(imgs, "float32")
    print(f"loaded via directory: {X.shape}, {n_objects} objects")
    return X, np.asarray(objs, int), np.asarray(angs, int)


def split_by_angle(obj, ang, train_frac=0.8, seed=SEED):
    """
    Train/test split BY ANGLE within each object, so test angles are held out but
    every object is seen. Topology is still studied on ALL angles (dense loops).
    """
    rng = np.random.default_rng(seed)
    is_train = np.zeros(len(obj), dtype=bool)
    for o in np.unique(obj):
        idx = np.where(obj == o)[0]
        k = int(round(train_frac * len(idx)))
        is_train[rng.choice(idx, size=k, replace=False)] = True
    return is_train


# ============================================================================
# Model
# ============================================================================

def head_layer_names(dense_depth=DENSE_DEPTH):
    return ["gap_features"] + [f"dense_{i+1}" for i in range(dense_depth)] + ["output"]


def build_coil_cnn(n_classes, dense_depth=DENSE_DEPTH, width=DENSE_WIDTH,
                   img_size=IMG_SIZE, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    tf.random.set_seed(seed)
    seq = [
        layers.Input(shape=(img_size, img_size, 3), name="image_input"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(name="gap_features"),
    ]
    for i in range(dense_depth):
        seq.append(layers.Dense(width, activation="relu", name=f"dense_{i+1}"))
    seq.append(layers.Dense(n_classes, activation="softmax", name="output"))
    model = Sequential(seq)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_or_load(X, y, is_train, epochs=60, force_retrain=False):
    from tensorflow.keras.models import load_model
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print(f"  loading cached: {os.path.basename(MODEL_PATH)}")
        model = load_model(MODEL_PATH)
    else:
        model = build_coil_cnn(n_classes=int(y.max()) + 1)
        model.fit(X[is_train], y[is_train], validation_data=(X[~is_train], y[~is_train]),
                  epochs=epochs, batch_size=32, verbose=2)
        model.save(MODEL_PATH)
    acc = float(model.evaluate(X[~is_train], y[~is_train], verbose=0)[1])
    print(f"  test accuracy (held-out angles): {acc:.4f}")
    return model, acc


def head_representations(model, X, layer_names):
    from tensorflow.keras.models import Model
    outs = [model.get_layer(n).output for n in layer_names]
    reps = Model(inputs=model.inputs, outputs=outs).predict(X, verbose=0)
    if not isinstance(reps, list):
        reps = [reps]
    return [np.asarray(r, float) for r in reps]


def raw_pca_representation(X, n_components=RAW_PCA_DIM):
    """PCA of the flattened raw images -> low-dim 'input' cloud where the rotation
    loop is preserved but distance concentration is avoided."""
    flat = X.reshape(len(X), -1)
    k = min(n_components, flat.shape[1], len(flat) - 1)
    return PCA(n_components=k, random_state=SEED).fit_transform(flat)


# ============================================================================
# Topology helpers (diameter-normalised; loop persistence = longest H1 bar)
# ============================================================================

def diameter_normalize(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return P.copy()
    diam = float(pdist(P).max())
    return P / diam if diam > 0 else P.copy()


def layer_filtration(P, max_dim=2, maxfilt=NORM_MAXFILT):
    Q = diameter_normalize(P)
    dm = cdist(Q, Q)
    st = gd.SimplexTree.create_from_array(dm, max_filtration=maxfilt)
    st.collapse_edges()
    st.expansion(max_dim if Q.shape[1] > 1 else 0)
    return st


def longest_h1_bar(P, max_dim=2):
    """Persistence (death-birth) of the most persistent H1 feature = the loop's
    strength. 0 if no H1. Diameter-normalised so it is comparable across layers."""
    st = layer_filtration(P, max_dim=max_dim)
    d1 = _finite_bars(compute_persistence_diagrams(st, max_dim=max(max_dim, 1)).get(
        1, np.empty((0, 2))))
    if d1.size == 0:
        return 0.0
    return float(np.max(d1[:, 1] - d1[:, 0]))


def betti1_max(P, max_dim=2):
    st = layer_filtration(P, max_dim=max_dim)
    d1 = compute_persistence_diagrams(st, max_dim=max(max_dim, 1)).get(1, np.empty((0, 2)))
    grid = make_eps_grid([{1: d1}], is_diagrams=True, n_points=200, max_dim=1)
    b1 = betti_curve(d1, grid)
    return int(b1.max()) if b1.size else 0


# ============================================================================
# (V) VERIFICATION: does the framework recover the known per-object loop?
# ============================================================================

def verify_loops(raw_pca, gap, obj):
    print("\n=== (V) VERIFICATION: per-object rotation loop (known H1) ===")
    print(f"{'object':>6} | {'rawPCA: b1':>10} {'loop_pers':>10} | "
          f"{'GAP: b1':>8} {'loop_pers':>10}")
    rows = []
    for o in np.unique(obj):
        m = obj == o
        b1_raw = betti1_max(raw_pca[m]); lp_raw = longest_h1_bar(raw_pca[m])
        b1_gap = betti1_max(gap[m]);     lp_gap = longest_h1_bar(gap[m])
        rows.append((o, b1_raw, lp_raw, b1_gap, lp_gap))
        print(f"{o:>6} | {b1_raw:>10d} {lp_raw:>10.4f} | {b1_gap:>8d} {lp_gap:>10.4f}")
    lp_raw_mean = np.mean([r[2] for r in rows])
    print(f"  mean raw-PCA loop persistence = {lp_raw_mean:.4f}")
    print("  EXPECT: a clear, persistent single loop per object in raw-PCA (b1~1, "
          "large loop_pers). If present, the framework recovers topology we KNOW is "
          "there -- the missing ground-truth check. GAP shows how much survives the "
          "conv front.")
    return rows


# ============================================================================
# (2) FEATURE TRACKING: where does each object's loop die in the head?
# ============================================================================

def track_loop_destruction(reps, layer_names, obj):
    print("\n=== (2) FEATURE TRACKING: rotation loop persistence through layers ===")
    objs = np.unique(obj)
    # curve[layer, object] = longest H1 bar of that object's points at that layer
    curve = np.zeros((len(reps), len(objs)))
    for li, R in enumerate(reps):
        for oi, o in enumerate(objs):
            curve[li, oi] = longest_h1_bar(R[obj == o])
    mean = curve.mean(1); std = curve.std(1)
    print(f"{'layer':>12} | {'loop_pers (mean+/-std)':>22}")
    for name, m, s in zip(layer_names, mean, std):
        print(f"{name:>12} | {m:>10.4f} +/- {s:<8.4f}")
    # parameter-free landmark: transition with the largest DROP in mean loop persistence
    diffs = np.diff(mean)
    if diffs.size:
        k = int(np.argmin(diffs))   # most negative = biggest destruction
        print(f"  largest loop destruction at transition "
              f"{layer_names[k]} -> {layer_names[k+1]} (drop {(-diffs[k]):.4f})")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(reps)), mean, yerr=std, marker="o", capsize=3)
    ax.set_xticks(range(len(reps))); ax.set_xticklabels(layer_names, rotation=30, ha="right")
    ax.set_ylabel("rotation-loop persistence (longest H1 bar)")
    ax.set_title("Where the rotation loop dies (rotation-invariance emerges)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "coil_loop_tracking.png"), dpi=150)
    plt.close(fig)
    print("  saved coil_loop_tracking.png")
    return curve, layer_names


def plot_one_loop_pca(raw_pca, late_rep, obj, ang, which_obj=0):
    """Show one object's rotation circle in raw-PCA vs a late layer -- the loop and
    its destruction, made visible."""
    m = obj == which_obj
    order = np.argsort(ang[m])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, R, title in zip(axes, [raw_pca[m][order], late_rep[m][order]],
                            ["raw-PCA (input): rotation circle",
                             "late dense layer: loop collapsed?"]):
        P = PCA(n_components=2, random_state=SEED).fit_transform(R) if R.shape[1] > 2 else R
        ax.plot(P[:, 0], P[:, 1], "-o", ms=3)
        ax.plot([P[-1, 0], P[0, 0]], [P[-1, 1], P[0, 1]], "-", color="gray", alpha=0.5)
        ax.set_title(title); ax.set_aspect("equal", "datalim")
    fig.suptitle(f"object {which_obj}: rotation loop, input vs late layer")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "coil_loop_pca.png"), dpi=150)
    plt.close(fig)
    print("  saved coil_loop_pca.png")


# ============================================================================
# (1) MLP-persistence convergence (inert layers), full cloud
# ============================================================================

def select_layer_epsilons(reps):
    return [float(np.median(pdist(diameter_normalize(P)))) if len(P) > 1 else 0.3
            for P in reps]


def plot_layer_diagrams(reps, layer_names, epsilons, max_dim=2):
    """
    Per-layer persistence diagram (H0 + H1) on the diameter-NORMALISED clouds --
    the same clouds the pullback tower uses, so the epsilons are on the same scale.
    The chosen epsilon for each layer is drawn as crosshair lines: a GOOD epsilon
    sits among / just past the off-diagonal (persistent) points, NOT inside the
    near-diagonal noise band. Points are plotted in (birth, death); the diagonal is
    the zero-persistence line, and a faint band above it flags the noise zone.

    Reading guide for epsilon quality, printed to console too:
      * epsilon line crossing BELOW persistent off-diagonal points  -> those
        features are still alive at epsilon (captured). Good.
      * epsilon line ABOVE all points (only diagonal noise remains)  -> too large,
        everything has merged/filled. Too coarse.
      * epsilon line so small that nothing has been born yet         -> too small.
    """
    n = len(reps)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.2), squeeze=False)
    axes = axes[0]
    for ax, name, P, eps in zip(axes, layer_names, reps, epsilons):
        st = layer_filtration(P, max_dim=max_dim)
        diags = compute_persistence_diagrams(st, max_dim=max(max_dim, 1))
        d0 = _finite_bars(diags.get(0, np.empty((0, 2))))
        d1 = _finite_bars(diags.get(1, np.empty((0, 2))))
        # axis range from finite points (+ a margin); guard empties
        allpts = np.vstack([a for a in (d0, d1) if a.size]) if (d0.size or d1.size) \
            else np.array([[0.0, NORM_MAXFILT]])
        hi = float(max(allpts.max(), eps, NORM_MAXFILT)) * 1.05
        ax.plot([0, hi], [0, hi], color="0.7", lw=1, zorder=0)           # diagonal
        # faint near-diagonal noise band (purely visual aid, width = 5% of range)
        band = 0.05 * hi
        ax.fill_between([0, hi], [0, hi], [band, hi + band], color="0.9", zorder=0)
        if d0.size:
            ax.scatter(d0[:, 0], d0[:, 1], s=12, c="tab:blue", label="H0", alpha=0.7)
        if d1.size:
            ax.scatter(d1[:, 0], d1[:, 1], s=18, c="tab:red", marker="^",
                       label="H1", alpha=0.8)
        # chosen epsilon as crosshair (birth=eps vertical, death=eps horizontal)
        ax.axvline(eps, color="k", ls="--", lw=1, alpha=0.7)
        ax.axhline(eps, color="k", ls="--", lw=1, alpha=0.7)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_title(f"{name}\n$\\varepsilon$={eps:.3f}", fontsize=9)
        ax.set_xlabel("birth", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("death", fontsize=8); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Per-layer persistence diagrams (diameter-normalised) with chosen "
                 "$\\varepsilon$", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "coil_layer_diagrams.png"), dpi=150)
    plt.close(fig)
    print("  saved coil_layer_diagrams.png")

    # console summary: how many genuine (above-band) H1 features survive at eps
    print(f"{'layer':>12} | {'eps':>6} | {'H1 alive@eps':>12} | {'max H1 pers':>11}")
    for name, P, eps in zip(layer_names, reps, epsilons):
        st = layer_filtration(P, max_dim=max_dim)
        d1 = _finite_bars(compute_persistence_diagrams(
            st, max_dim=max(max_dim, 1)).get(1, np.empty((0, 2))))
        alive = int(np.sum((d1[:, 0] <= eps) & (eps < d1[:, 1]))) if d1.size else 0
        mp = float(np.max(d1[:, 1] - d1[:, 0])) if d1.size else 0.0
        print(f"{name:>12} | {eps:6.3f} | {alive:12d} | {mp:11.4f}")


def build_pullback_tower(reps, epsilons, max_dim=2):
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


def mlp_convergence(reps, layer_names):
    print("\n=== (1) MLP-persistence convergence (inert layers) ===")
    eps = select_layer_epsilons(reps)
    trees = build_pullback_tower(reps, eps)
    combined = create_combined_filtration(trees, max_dimension=2)
    res = convergence_layer(combined, n_layers=len(trees), max_dim=1)
    print(f"  layer epsilons: {np.round(eps,3).tolist()}")
    print(f"  d*_by_dim={res['d_star_by_dim']}  n_genuine={res['n_genuine_by_dim']}")
    print(f"  convergence layer d*={res['d_star']}  inert layers={res['inert_layers']}")
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_mlp_barcode(res["bars_by_dim"], n_layers=len(trees), d_star=res["d_star"], ax=ax)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "coil_mlp_barcode.png"), dpi=150)
    plt.close(fig)
    print("  saved coil_mlp_barcode.png")
    return res


# ============================================================================
# Supporting: per-layer object-classification probe
# ============================================================================

def supporting_probe(model, X, y, is_train, layer_names):
    print("\n=== object-classification probe per layer ===")
    Rtr = head_representations(model, X[is_train], layer_names)
    Rte = head_representations(model, X[~is_train], layer_names)
    print(f"{'layer':>12} | {'probe_acc':>9}")
    for name, A, B in zip(layer_names, Rtr, Rte):
        clf = LogisticRegression(max_iter=2000).fit(A, y[is_train])
        print(f"{name:>12} | {clf.score(B, y[~is_train]):9.4f}")


# ============================================================================
# Driver
# ============================================================================

def main(epochs=60, force_retrain=False):
    X, obj, ang = load_coil100()
    y = obj                                   # classify object identity
    is_train = split_by_angle(obj, ang)
    print(f"COIL-100 subset: {X.shape}, {N_OBJECTS} objects x {N_ANGLES} angles; "
          f"train {is_train.sum()} / test {(~is_train).sum()} (by angle)")

    model, acc = train_or_load(X, y, is_train, epochs=epochs, force_retrain=force_retrain)
    names = head_layer_names()

    # representations on ALL points (dense loops preserved)
    raw_pca = raw_pca_representation(X)
    net_reps = head_representations(model, X, names)         # [gap, dense.., output]
    track_names = ["raw_pca"] + names
    track_reps = [raw_pca] + net_reps

    # (V) verify the framework recovers the known loop
    verify_loops(raw_pca, net_reps[0], obj)

    # (2) track the loop through the head; visualise one object's circle
    track_loop_destruction(track_reps, track_names, obj)
    plot_one_loop_pca(raw_pca, net_reps[-2], obj, ang, which_obj=0)

    # (1) inert-layer convergence on the network representations
    mlp_convergence(net_reps, names)

    # supporting probe
    supporting_probe(model, X, y, is_train, names)

    print("\nDone. Figures written to:", OUTDIR)


if __name__ == "__main__":
    main()