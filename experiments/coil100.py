#!/usr/bin/env python
# coding: utf-8
"""
Experiment_coil100_tasks.py
===========================

COIL-100 as the H1 POSITIVE CONTROL, and as a test of *what* the convergence depth
measures. Each object is photographed at 72 angles over 360 deg, so each object's
views trace a closed rotation loop -- a genuine H1 generator known by construction.
The loops are WITHIN-class (one circle per object), not between classes.

We compare how three TASKS treat that known loop, which is the scientific point:

  (AE)  AUTOENCODER (reconstruct the image): the bottleneck MUST retain every angle
        to reconstruct, so the loop should be PRESERVED through the encoder
        (carrier H1, loop never resolved in the encoder).
  (CLF) CLASSIFIER (object identity): object recognition is rotation-INVARIANT, so
        all 72 views of an object must collapse to one logit -- but only AT THE END.
        Hypothesis (to TEST, not assume): because the loops are within-class and the
        classes are not entangled, the network keeps each loop intact through the
        head and collapses it only near the output.
  (REG) ANGLE REGRESSION (predict the rotation angle; optional): the angle is the
        task, and a scalar (cut) target forces the network to linearise the circle,
        so the loop should be RESOLVED EARLY. This is the contrast that shows d*
        tracks TASK-RELEVANT topology.

Preprocessing: RAW (mild downsample), diameter-normalised per layer -- NOT PCA. For
COIL the signal is an intrinsically curved 1-manifold; PCA keeps off-circle linear
directions and smears the S^1 (we show this in the input verification). This is the
opposite preprocessing choice from CIFAR, and that data-dependence is itself reported.

Pipeline of record: pipeline.py (same carrier / d* machinery as CIFAR/ResNet) for the
framework numbers; the diameter-normalised longest-H1-bar from Experiment_coil100.py
for the loop-strength tracking curve.

    python Experiment_coil100_tasks.py                 # AE + classifier
    python Experiment_coil100_tasks.py --tasks ae clf reg
    python Experiment_coil100_tasks.py --objects 5 --epochs 80

Reuses Experiment_coil100.py (loader, classifier, topology helpers) and pipeline.py.
"""

import os
import argparse
import numpy as np

# --- make the repo root importable whether run as a script or via -m ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# PARAMETERS FOR THIS DATASET (COIL-100) -- edit here, nowhere else
# ============================================================================
PARAMS = dict(
    tasks=["ae"],          # COIL is the AE / preserved-feature experiment; classification
                           # is trivial here and is carried by the ResNet experiment.
    n_objects=5,           # small set of classes; pooled clouds stay tractable
    epochs=500,
    enc_depth=3,
    dense_depth=4,
    bottleneck=32,         # AE bottleneck width (room for the per-object cycles)
    ae_width=256,
    group_size=1,          # 1 = per object (AE); >1 pools classes (classifier needs this)
    angles=24,             # EVENLY-SPACED angles kept per object (stratified, keeps S^1)
    subsample=1.0,         # random per-object thinning (used only when angles is None)
    alpha=0.05,            # COIL loops are weak -> 0.05 captures the significant ones
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

SEED = 1234
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = OUTDIR          # set per-run to COIL/<params>/ in main(); figures save here
IMG_SIZE = 64
WIDTH = 128
N_ANGLES_FULL = 72


# ----------------------------------------------------------------------------
# Models: MLP on the FLATTENED pixel vector (no CNN). The flattened input is the
# space where the rotation loop is known to show up under persistent homology, and
# it is tracked directly as the first representation ("input_flat"). With distinct
# objects the MLP reaches high accuracy, so the conv front is unnecessary and (by
# pooling / translation-invariance) would only attenuate the loop before the dense
# layers see it.
# ----------------------------------------------------------------------------

def _flat_input(img_size, layers):
    inp = layers.Input(shape=(img_size, img_size, 3), name="image_input")
    f = layers.Flatten(name="input_flat")(inp)          # raw flattened pixels = tracked input
    return inp, f


def build_mlp_classifier(n_classes, img_size=IMG_SIZE, dense_depth=4, width=WIDTH, seed=SEED):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    tf.random.set_seed(seed)
    inp, x = _flat_input(img_size, layers)
    for i in range(dense_depth):
        x = layers.Dense(width, activation="relu", name=f"dense_{i+1}")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_mlp_autoencoder(img_size=IMG_SIZE, enc_depth=3, width=WIDTH,
                          bottleneck_dim=8, seed=SEED):
    """Flatten -> dense encoder -> linear bottleneck -> dense decoder -> reconstruct.
    Encoder representations tracked: input_flat, enc_1..enc_{enc_depth}, bottleneck.
    The bottleneck must keep the loop to reconstruct all 72 angles."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    tf.random.set_seed(seed)
    inp, x = _flat_input(img_size, layers)
    for i in range(enc_depth):
        x = layers.Dense(width, activation="relu", name=f"enc_{i+1}")(x)
    b = layers.Dense(bottleneck_dim, activation="linear", name="bottleneck")(x)
    d = layers.Dense(width, activation="relu")(b)
    d = layers.Dense(img_size * img_size * 3, activation="sigmoid")(d)
    out = layers.Reshape((img_size, img_size, 3), name="reconstruction")(d)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def build_mlp_regressor(img_size=IMG_SIZE, dense_depth=4, width=WIDTH, seed=SEED):
    """Flatten -> dense head -> scalar angle in [0,1) (= angle_idx/72), MSE. The cut
    (non-periodic) target forces linearisation of the circle (loop destroyed)."""
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Model
    tf.random.set_seed(seed)
    inp, x = _flat_input(img_size, layers)
    for i in range(dense_depth):
        x = layers.Dense(width, activation="relu", name=f"dense_{i+1}")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def mlp_head_names(dense_depth=4):
    return ["input_flat"] + [f"dense_{i+1}" for i in range(dense_depth)] + ["output"]


def encoder_layer_names(enc_depth=3):
    return ["input_flat"] + [f"enc_{i+1}" for i in range(enc_depth)] + ["bottleneck"]


# ----------------------------------------------------------------------------
# Task dispatch: build, train, extract representations
# ----------------------------------------------------------------------------

def run_task_network(task, X, obj, ang, is_train, epochs, enc_depth, dense_depth,
                     bottleneck_dim=32, ae_width=256, force_retrain=False):
    """Train (or load) the network for `task` and return (reps, layer_names, metric).
    reps are per-layer representations on ALL points (dense loops preserved).

    The AUTOENCODER trains on ALL angles (no angle hold-out): its job is to embed the
    rotation manifold as densely as possible, and reconstruction carries no label to
    leak. The classifier/regressor keep the angle hold-out to test generalisation."""
    from ltep.datasets import coil100 as base
    from tensorflow.keras.models import load_model
    path = os.path.join(OUTDIR, f"coil_{task}_{len(np.unique(obj))}obj.keras")

    if task == "clf":
        names = mlp_head_names(dense_depth)
        builder = lambda: build_mlp_classifier(int(obj.max()) + 1, dense_depth=dense_depth)
        target = obj
        tr = is_train                                # hold out angles (generalisation)
    elif task == "ae":
        names = encoder_layer_names(enc_depth)
        builder = lambda: build_mlp_autoencoder(enc_depth=enc_depth, width=ae_width,
                                                bottleneck_dim=bottleneck_dim)
        target = X                                   # reconstruct the image
        tr = np.ones(len(X), bool)                   # use ALL angles (densest manifold)
    elif task == "reg":
        names = mlp_head_names(dense_depth)
        builder = lambda: build_mlp_regressor(dense_depth=dense_depth)
        target = ang.astype("float32") / 72.0
        tr = is_train
    else:
        raise ValueError(task)

    if os.path.exists(path) and not force_retrain:
        print(f"  loading cached: {os.path.basename(path)}")
        model = load_model(path)
    else:
        model = builder()
        # for the AE (tr = all) hold out a small random slice only to monitor val loss
        if task == "ae":
            val = base.split_by_angle(obj, ang, train_frac=0.9, seed=SEED)
            model.fit(X[val], target[val], validation_data=(X[~val], target[~val]),
                      epochs=epochs, batch_size=32, verbose=2)
        else:
            model.fit(X[tr], target[tr], validation_data=(X[~tr], target[~tr]),
                      epochs=epochs, batch_size=32, verbose=2)
        model.save(path)

    eval_mask = tr if task == "ae" else ~is_train
    metric = model.evaluate(X[eval_mask], target[eval_mask], verbose=0)
    if isinstance(metric, (list, tuple)):     # classifier returns [loss, accuracy]
        metric = metric[-1]
    metric = float(metric)
    label = {"clf": "test accuracy", "ae": "recon MSE (all angles)",
             "reg": "angle MSE"}.get(task, "metric")
    print(f"  [{task}] {label}: {metric:.4f}")

    # DIAGNOSTIC for the AE: per-object reconstruction error. A failing-loop object
    # that reconstructs BADLY -> embedding problem (training helps); one that
    # reconstructs FINE but shows no significant loop -> input/significance, not
    # training. This is what distinguishes the two explanations object-by-object.
    if task == "ae":
        pred = model.predict(X, verbose=0)
        print(f"  {'object':>6} | {'recon MSE':>10} | {'input loop':>10}")
        for o in np.unique(obj):
            m = obj == o
            mse_o = float(np.mean((pred[m] - X[m]) ** 2))
            lp = base.longest_h1_bar(X[m].reshape(m.sum(), -1))
            print(f"  {o:>6} | {mse_o:>10.5f} | {lp:>10.4f}")

    reps = base.head_representations(model, X, names)
    return reps, names, metric


# ----------------------------------------------------------------------------
# (V) Input-loop verification, and the raw-vs-PCA point
# ----------------------------------------------------------------------------

def input_verification(X, obj, ang):
    """Confirm the known rotation loop is recovered at the RAW input, and show PCA
    smears it. Plots one object's circle (raw 2D PCA-for-display vs distances)."""
    from ltep.datasets import coil100 as base
    raw = X.reshape(len(X), -1)
    raw_pca = base.raw_pca_representation(X)            # PCA(50) of flattened pixels
    print("\n=== (V) INPUT-LOOP VERIFICATION (known H1, per object) ===")
    print(f"{'object':>6} | {'raw b1':>7} {'raw loop':>9} | {'pca b1':>7} {'pca loop':>9}")
    raw_lp, pca_lp = [], []
    for o in np.unique(obj):
        m = obj == o
        b1r = base.betti1_max(raw[m]);     lpr = base.longest_h1_bar(raw[m])
        b1p = base.betti1_max(raw_pca[m]); lpp = base.longest_h1_bar(raw_pca[m])
        raw_lp.append(lpr); pca_lp.append(lpp)
        print(f"{o:>6} | {b1r:>7d} {lpr:>9.4f} | {b1p:>7d} {lpp:>9.4f}")
    print(f"  mean longest-H1 bar: RAW={np.mean(raw_lp):.4f}  PCA={np.mean(pca_lp):.4f}")
    print("  EXPECT: a single persistent loop per object (b1~1, large bar) in RAW; "
          "PCA tends to smear it (off-circle linear directions) -> raw is the COIL "
          "preprocessing of record.")

    # visual: one object's circle, raw vs PCA (2D projection for display only)
    from sklearn.decomposition import PCA
    o0 = int(np.unique(obj)[0]); m = obj == o0; order = np.argsort(ang[m])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, P, ttl in zip(axes,
                          [raw[m][order], raw_pca[m][order]],
                          ["raw pixels (2D PCA view)", "PCA-50 (2D PCA view)"]):
        V = PCA(n_components=2, random_state=SEED).fit_transform(P)
        ax.plot(V[:, 0], V[:, 1], "-o", ms=3)
        ax.plot([V[-1, 0], V[0, 0]], [V[-1, 1], V[0, 1]], color="gray", alpha=0.5)
        ax.set_title(ttl); ax.set_aspect("equal", "datalim")
    fig.suptitle(f"object {o0}: rotation circle (ground-truth H1)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "coil_input_loop.png"), dpi=150)
    plt.close(fig)
    print("  saved coil_input_loop.png")
    return float(np.mean(raw_lp))


# ----------------------------------------------------------------------------
# Loop-strength tracking across layers (the headline measurement)
# ----------------------------------------------------------------------------

def loop_strength_curve(reps, obj):
    """Per layer, the mean (over objects) longest diameter-normalised H1 bar -- how
    strongly each object's rotation loop survives at that layer."""
    from ltep.datasets import coil100 as base
    objs = np.unique(obj)
    curve = np.zeros(len(reps))
    spread = np.zeros(len(reps))
    for li, R in enumerate(reps):
        vals = [base.longest_h1_bar(R[obj == o]) for o in objs]
        curve[li] = float(np.mean(vals)); spread[li] = float(np.std(vals))
    return curve, spread


def framework_carrier(reps, obj, use_bootstrap, exclude_output=True, seed=SEED):
    """Per-object unified-pipeline readouts. Reports BOTH:
      * signal_dim         = H1 if a genuine loop is PRESENT (resolved or preserved)
      * simplification_dim = H1 only if a loop is RESOLVED (drives convergence/d*)
    and the H1 resolution layer (where the loop dies; None = preserved).

    exclude_output=False when the last representation is a loop-bearing bottleneck
    (autoencoder) rather than a collapsing softmax/scalar head (classifier/regressor).
    """
    from ltep.datasets import coil100 as base
    from ltep import pipeline as pl
    objs = np.unique(obj)
    signal, simpl, resolved = [], [], []
    for o in objs:
        Q = [base.diameter_normalize(R[obj == o]) for R in reps]
        eps = pl.select_epsilon(Q, use_bootstrap=use_bootstrap, n_boot=pl.N_BOOT,
                                exclude_output=exclude_output, rng=seed)["epsilons"]
        # no resampling: significance is already in the per-layer epsilon choice; the
        # MLP barcode just tracks the surviving cycles across layers.
        conv = pl.convergence_depth(Q, eps, significance=False,
                                    augment_output=False, exclude_output=exclude_output)
        signal.append(pl.signal_dimension(conv))
        simpl.append(pl.simplification_dimension(conv))
        resolved.append(conv["per_dim"][1]["resolved_by"])    # None = loop preserved
    n = len(objs)
    n_sig = sum(s == 1 for s in signal)
    n_sim = sum(s == 1 for s in simpl)
    # preserved = genuine loop PRESENT but not resolved (not merely 'no H1')
    n_pres = sum((sg == 1 and r is None) for sg, r in zip(signal, resolved))
    res_layers = sorted(r for r in resolved if r is not None)
    print(f"  signal H1 (loop PRESENT)       in {n_sig}/{n} objects")
    print(f"  simplification H1 (RESOLVED)   in {n_sim}/{n} objects")
    print(f"  loop PRESERVED (present, unresolved) in {n_pres}/{n}; "
          f"resolved-by layers: {res_layers}")
    return dict(signal=signal, simplification=simpl, resolved=resolved,
                n_signal_h1=n_sig, n_simpl_h1=n_sim, n_preserved=n_pres)


def embedding_loop_count(reps, names, obj, task, tag="", ang=None):
    """Betti1 of the cloud passed in (one object, a subset, or full) at each layer.
    For the AE the embedding (bottleneck) should carry the rotation loop(s); per
    object Betti1~1, pooled Betti1~n_objects. `tag` distinguishes outputs.
    If `ang` (rotation-angle index per row) is given, also save an Isomap embedding
    grid coloured by angle -- the direct VISUAL check that the loop is a real circle."""
    from ltep.datasets import coil100 as base
    from ltep.metrics import compute_persistence_diagrams, _finite_bars
    sfx = f"_{tag}" if tag else ""
    n_obj = len(np.unique(obj))
    print(f"  embedding loop count ({tag or 'full cloud'})   target Betti1 = {n_obj}")
    print(f"  {'layer':>12} | {'Betti1 (full cloud)':>19}")
    b1s = []
    for name, R in zip(names, reps):
        b1 = base.betti1_max(R)
        b1s.append(b1)
        print(f"  {name:>12} | {b1:>19d}")
    # persistence diagram of the embedding (last rep) on the full cloud
    st = base.layer_filtration(reps[-1])
    diags = compute_persistence_diagrams(st, max_dim=2)
    d0 = _finite_bars(diags.get(0, np.empty((0, 2))))
    d1 = _finite_bars(diags.get(1, np.empty((0, 2))))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    hi = float(max(d0.max() if d0.size else 0, d1.max() if d1.size else 0, 0.1)) * 1.05
    ax.plot([0, hi], [0, hi], color="0.7", lw=1)
    if d0.size:
        ax.scatter(d0[:, 0], d0[:, 1], s=14, c="tab:blue", alpha=0.6, label="$H_0$")
    if d1.size:
        ax.scatter(d1[:, 0], d1[:, 1], s=24, c="tab:red", marker="^",
                   alpha=0.8, label="$H_1$")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(f"COIL [{task}] embedding ({names[-1]}) full cloud\n"
                 f"Betti1={b1s[-1]} (target {n_obj})", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"coil_{task}{sfx}_embedding_diagram.png"), dpi=150)
    plt.close(fig)
    print(f"  saved coil_{task}{sfx}_embedding_diagram.png  "
          f"(embedding Betti1 = {b1s[-1]}, target {n_obj})")

    # visual loop check: Isomap of every layer, coloured by rotation angle
    if ang is not None:
        from ltep import plots as _plt
        p = _plt.plot_layer_embeddings(
            reps, layer_names=names, color=np.asarray(ang),
            path=os.path.join(FIGDIR, f"coil_{task}{sfx}_embedding_isomap.png"),
            method="isomap",
            title=f"COIL [{task}] {tag or 'full'} -- Isomap per layer (colour = angle); "
                  f"preserved loop = closed colour ring")
        print(f"  saved {os.path.basename(p)}  (visual loop check, coloured by angle)")
    return b1s


def classifier_prunability(conv, n_layers, t_eps, t_conv):
    """Report d*-based prunability + runtime for the classifier. PURE REPORTER: it reads
    the SAME conv computed in coil_topology_figures (diameter-normalized cloud, no
    resampling), so the d* here cannot diverge from the per-subgroup d* used elsewhere.
    H1 collapses early (within-class); convergence is the H0 geometric reorganisation
    toward class separability. d* = last layer at which a cycle/component is born or
    resolved; the trailing layers with no such event carry identical topology -> prunable."""
    d = conv["d_star"]; inert = conv["inert_layers"]
    print("\n=== classifier prunability (d*) + runtime ===")
    print(f"  d* = {d}  (of {n_layers-1} transitions)   "
          f"prunable tail (stable topology): {inert}")
    print(f"  verdict: {'PRUNABLE past d*' if inert else 'no inert tail -> not prunable'}"
          f"  (last {len(inert)} layer(s) carry identical cycles/components)")
    print(f"  runtime: epsilon band {t_eps:.2f}s | MLP-persistence {t_conv:.2f}s")
    return dict(d_star=d, inert=inert, t_eps=t_eps, t_conv=t_conv)


def make_groups(objs, group_size, seed=SEED):
    """Object groups for the heavy figures.
      group_size <= 1 -> per object (each object its own singleton).
      group_size >= n -> one group of all objects.
      otherwise       -> RANDOM partition of all classes into size-`group_size`
                         subgroups (ceil(n/size) of them), so EVERY class is covered.
                         If the last subgroup is short, it is padded with classes drawn
                         from the others (repetition across subgroups, which is allowed
                         here) so all subgroups have the requested size."""
    objs = np.asarray(objs); n = len(objs)
    if group_size <= 1:
        return [[int(o)] for o in objs]
    if group_size >= n:
        return [sorted(int(o) for o in objs)]
    rng = np.random.default_rng(seed)
    perm = [int(o) for o in rng.permutation(objs)]
    groups = []
    for i in range(0, n, group_size):
        chunk = perm[i:i + group_size]
        if len(chunk) < group_size:                       # pad with repeats from others
            pool = [o for o in perm if o not in chunk]
            extra = rng.choice(pool, size=group_size - len(chunk), replace=False).tolist()
            chunk = chunk + [int(x) for x in extra]
        groups.append(sorted(chunk))
    return groups


def ae_collapse_sweep(X, obj, enc_depths, bottleneck_dim, ae_width, epochs,
                      seeds=(0,), loop_frac=0.5, recon_tol=2.0, outdir="."):
    """Collapse the AE encoder over several seeds. Topology flags the WHOLE encoder as
    a single stage -- the rotation loop is PRESERVED at every layer (no births / finite
    deaths), so the depth is topologically redundant -- and a two-criterion retrain
    decides how far it collapses: keep the smallest depth that, ON AVERAGE, retains
    BOTH reconstruction (MSE within `recon_tol`x the best) AND the per-object loops
    (bottleneck longest-H1 bar >= `loop_frac` of the input bar, in >= 80% of the
    loop-bearing objects). All angles used. Saves a depth-sweep figure."""
    from ltep.datasets import coil100 as base
    objs = np.unique(obj)
    raw = X.reshape(len(X), -1)
    in_bar = {int(o): base.longest_h1_bar(raw[obj == o]) for o in objs}
    n_with = sum(v > 0 for v in in_bar.values())
    seeds = list(seeds)

    rec = {d: dict(recon=[], kept=[], bar=[]) for d in enc_depths}
    print("\n" + "=" * 70)
    print(f"AE ENCODER COLLAPSE -- {len(seeds)} seed(s) (retrain shallower; keep recon + loops)")
    print("=" * 70)
    for s in seeds:
        for d in enc_depths:
            model = build_mlp_autoencoder(enc_depth=d, width=ae_width,
                                          bottleneck_dim=bottleneck_dim, seed=s)
            model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
            recon = float(np.mean((model.predict(X, verbose=0) - X) ** 2))
            bott = base.head_representations(model, X, encoder_layer_names(d))[-1]
            bars = {int(o): base.longest_h1_bar(base.diameter_normalize(bott[obj == o]))
                    for o in objs}
            kept = sum(bars[o] >= loop_frac * in_bar[o] for o in objs if in_bar[o] > 0)
            mbar = float(np.mean(list(bars.values())))
            rec[d]["recon"].append(recon); rec[d]["kept"].append(kept); rec[d]["bar"].append(mbar)
            print(f"  seed {s} enc_depth {d}: recon {recon:.5f}  loops {kept}/{n_with}  bar {mbar:.4f}")

    # aggregate table
    print("\n  " + "=" * 64)
    print(f"  SUMMARY ({len(seeds)} seeds, mean+/-std)")
    print("  " + "=" * 64)
    print(f"  {'enc_depth':>9} | {'recon MSE':>18} | {'loops kept':>12} | {'mean bar':>9}")
    agg = {}
    for d in enc_depths:
        r = rec[d]
        agg[d] = dict(recon=float(np.mean(r["recon"])), recon_std=float(np.std(r["recon"])),
                      kept=float(np.mean(r["kept"])), bar=float(np.mean(r["bar"])), n=n_with)
        print(f"  {d:>9} | {agg[d]['recon']:.5f} +/- {agg[d]['recon_std']:.5f} | "
              f"{agg[d]['kept']:>5.1f}/{n_with:<5} | {agg[d]['bar']:>9.4f}")

    best = min(agg[d]["recon"] for d in enc_depths)
    ok = [d for d in enc_depths if agg[d]["kept"] >= 0.8 * n_with
          and agg[d]["recon"] <= recon_tol * best]
    rdepth = min(ok) if ok else None
    if rdepth is not None:
        safe = min((d for d in ok if agg[d]["recon"] <= 1.2 * best), default=rdepth)
        print(f"\n  RECOMMENDED encoder depth = {rdepth} of {max(enc_depths)} "
              f"(mean recon {agg[rdepth]['recon']:.5f} <= {recon_tol:.0f}x best {best:.5f}; "
              f"loops {agg[rdepth]['kept']:.1f}/{n_with}). "
              f"Conservative (recon within 1.2x best): depth {safe}. "
              f"Layers beyond are topologically inert AND empirically free.")
    else:
        print("\n  no shallower encoder kept both criteria on average -> depth is used.")

    # ---- depth-sweep figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = sorted(enc_depths)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.errorbar(ds, [agg[d]["recon"] for d in ds], yerr=[agg[d]["recon_std"] for d in ds],
                     marker="o", capsize=3, color="tab:red")
        ax1.set_xlabel("encoder depth"); ax1.set_ylabel("reconstruction MSE")
        ax1.set_title("Reconstruction vs encoder depth")
        ax2.plot(ds, [agg[d]["kept"] for d in ds], "o-", color="tab:blue", label="loops kept")
        ax2b = ax2.twinx()
        ax2b.plot(ds, [agg[d]["bar"] for d in ds], "s--", color="tab:green",
                  label="mean bottleneck bar")
        ax2.set_xlabel("encoder depth"); ax2.set_ylabel("loops kept (of %d)" % n_with, color="tab:blue")
        ax2b.set_ylabel("mean bottleneck H1 bar", color="tab:green")
        ax2.set_title("Loop preservation vs encoder depth")
        for ax in (ax1, ax2):
            if rdepth is not None:
                ax.axvline(rdepth, ls="--", color="k", lw=1.2)
            ax.set_xticks(ds)
        fig.suptitle("AE encoder collapse (dashed = recommended depth)")
        fig.tight_layout()
        p = os.path.join(outdir, "coil_ae_collapse_sweep.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"  saved {p}")
    except Exception as e:
        print(f"  (collapse figure skipped: {e})")
    return agg, rdepth


def _linear_cka(A, B):
    """Linear CKA between two representations on the SAME rows (feature dims may
    differ). 1 = identical up to rotation/scale; small = the layer changed a lot."""
    A = np.asarray(A, float); A = A - A.mean(0)
    B = np.asarray(B, float); B = B - B.mean(0)
    hsic = np.linalg.norm(B.T @ A, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(hsic / den) if den > 0 else 0.0


def ae_collapse_baselines(X, obj, agg, enc_depths, bottleneck_dim, ae_width, epochs,
                          recon_tol=2.0, loop_frac=0.5, seed=0, cka_thr=0.9, outdir="."):
    """Non-topological baselines for the encoder-depth choice (reviewer R1.2). The AE
    encoder is a UNIFORM-width stack, so random-position layer drop is degenerate
    (dropping any single width-{ae_width} layer yields the same architecture); the
    meaningful comparison is between DEPTH-SELECTION CRITERIA:

      (T) topological         -- smallest depth keeping reconstruction AND the per-object
                                 loops (the paper's rule);
      (R) reconstruction-only -- smallest depth within recon_tol x best MSE, IGNORING the
                                 loops (a purely functional pruning criterion);
      (C) representation CKA   -- collapse the run of encoder layers whose consecutive
                                 linear CKA stays >= cka_thr (representations 'unchanged'),
                                 the non-topological analogue of one inert stage.

    (T) and (R) reuse the already-computed sweep `agg` (no retraining); (C) trains one
    full-depth encoder to read its consecutive-layer CKA profile. Prints a comparison
    and saves coil_baselines.json."""
    from ltep.datasets import coil100 as base
    from ltep import runtime as rs
    import json
    n_with = next(iter(agg.values()))["n"]
    best = min(agg[d]["recon"] for d in agg)

    okT = [d for d in enc_depths if agg[d]["kept"] >= 0.8 * n_with
           and agg[d]["recon"] <= recon_tol * best]
    d_T = min(okT) if okT else None
    okR = [d for d in enc_depths if agg[d]["recon"] <= recon_tol * best]
    d_R = min(okR) if okR else None

    # (C) consecutive-layer CKA on the deepest trained encoder
    dmax = max(enc_depths)
    with rs.measure() as t_cka_train:
        model = build_mlp_autoencoder(enc_depth=dmax, width=ae_width,
                                      bottleneck_dim=bottleneck_dim, seed=seed)
        model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
    names = encoder_layer_names(dmax)
    reps = base.head_representations(model, X, names)   # [input_flat, enc_1..enc_dmax, bottleneck]
    enc_reps = reps[1:1 + dmax]                          # the width-ae_width encoder layers
    cka_consec = [_linear_cka(enc_reps[i], enc_reps[i + 1])
                  for i in range(len(enc_reps) - 1)]
    n_changes = sum(c < cka_thr for c in cka_consec)     # each below-thr pair = a real change
    d_C = 1 + n_changes

    def _loops(d):  return (f"{agg[d]['kept']:.1f}/{n_with}" if d in agg else "n/a")
    def _recon(d):  return (f"{agg[d]['recon']:.5f}" if d in agg else "n/a")

    print("\n" + "=" * 70)
    print("AE ENCODER-DEPTH BASELINES (criterion comparison; random drop is degenerate)")
    print("=" * 70)
    print(f"  full depth {dmax}; best recon {best:.5f}; recon_tol {recon_tol}x; "
          f"loop_frac {loop_frac}; CKA thr {cka_thr}")
    print(f"  consecutive encoder CKA (enc_i -> enc_i+1): {[round(c,3) for c in cka_consec]}")
    print(f"\n  {'criterion':>26} | {'rec. depth':>10} | {'loops kept':>12} | {'recon':>9}")
    print(f"  {'topological (recon+loops)':>26} | {str(d_T):>10} | {_loops(d_T):>12} | {_recon(d_T):>9}")
    print(f"  {'reconstruction-only':>26} | {str(d_R):>10} | {_loops(d_R):>12} | {_recon(d_R):>9}")
    print(f"  {'representation CKA':>26} | {str(d_C):>10} | {_loops(d_C):>12} | {_recon(d_C):>9}")
    agree = (d_T == d_R == d_C)
    print(f"\n  verdict: criteria {'AGREE' if agree else 'DIFFER'} "
          f"(topo={d_T}, recon-only={d_R}, CKA={d_C}). "
          + ("All three collapse to the same depth: the topological reading is "
             "corroborated by a functional and a representation-similarity criterion."
             if agree else
             "Where they differ, the loop criterion is what protects the preserved "
             "feature; recon-only / CKA can over-collapse and break loops."))
    print(f"  runtime: CKA full-depth train {t_cka_train['seconds']:.1f}s "
          f"(recon-only reuses the sweep, no extra training)")
    rec = dict(full_depth=dmax, best_recon=best, recon_tol=recon_tol,
               loop_frac=loop_frac, cka_thr=cka_thr, cka_consecutive=cka_consec,
               depth_topological=d_T, depth_recon_only=d_R, depth_cka=d_C,
               agree=bool(agree),
               runtime=dict(cka_train_s=float(t_cka_train["seconds"])),
               per_depth={int(d): dict(recon=agg[d]["recon"], kept=agg[d]["kept"],
                                       bar=agg[d]["bar"]) for d in agg})
    with open(os.path.join(outdir, "coil_baselines.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  saved {os.path.join(outdir, 'coil_baselines.json')}")
    return rec


def collapsed_ae_figures(X, obj, ang, enc_depth, *, bottleneck_dim, ae_width, epochs,
                         angles, use_bootstrap, seed=SEED):
    """Train the recommended collapsed AE and emit the per-object topology figures
    (layer-persistence diagrams, MLP-persistence H0/H1 barcodes, trajectory flow and
    the bottleneck embedding) -- so the collapsed encoder's preserved loops are shown
    with the actual persistence, not only the sweep curve."""
    from ltep.datasets import coil100 as base
    from ltep import runtime as rs, pipeline as pl
    print(f"\n=== topology figures for the collapsed AE (enc_depth={enc_depth}) ===")
    with rs.measure() as t_train:
        model = build_mlp_autoencoder(enc_depth=enc_depth, width=ae_width,
                                      bottleneck_dim=bottleneck_dim, seed=seed)
        model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
    names = encoder_layer_names(enc_depth)
    reps = base.head_representations(model, X, names)
    keep = subsample_by_angle(obj, ang, angles) if angles else np.ones(len(obj), bool)
    eps_secs, pers_secs, npts = [], [], []
    for o in np.unique(obj):
        m = (obj == o) & keep
        tag = f"d{enc_depth}_obj{int(o)}"
        reps_g = [R[m] for R in reps]
        _, t_eps, t_pers = coil_topology_figures(
            reps_g, names, obj[m], "ae", use_bootstrap=use_bootstrap,
            exclude_output=False, tag=tag)
        eps_secs.append(t_eps); pers_secs.append(t_pers); npts.append(int(m.sum()))
        embedding_loop_count(reps_g, names, obj[m], "ae", tag=tag, ang=ang[m])

    # --- per-stage time table (per-object topology analysis) ---
    print(f"\n  -- runtime (enc_depth={enc_depth}); {len(eps_secs)} objects, "
          f"~{int(np.mean(npts))} pts each, B={pl.PARAMS['N_BOOT']} bootstrap --")
    print(f"  {'stage':>16} | {'mean s':>8} | {'std s':>7}")
    print(f"  {'AE training':>16} | {t_train['seconds']:8.2f} | {'-':>7}")
    print(f"  {'epsilon band':>16} | {np.mean(eps_secs):8.2f} | {np.std(eps_secs):7.2f}")
    print(f"  {'MLP persistence':>16} | {np.mean(pers_secs):8.4f} | {np.std(pers_secs):7.4f}")
    print(f"  {'analysis/object':>16} | {np.mean(eps_secs)+np.mean(pers_secs):8.2f} | "
          f"   (epsilon + persistence; the topology cost)")


def select_top_loop_objects(X, obj, ang, n_keep):
    """From the loaded pool, keep the n_keep objects with the STRONGEST input rotation
    loop (longest diameter-normalised H1 bar on raw pixels), excluding near-symmetric
    objects whose 72 views barely move (no circle to track). Remaps the kept object
    ids to contiguous 0..n_keep-1 so the classifier's class count stays correct."""
    from ltep.datasets import coil100 as base
    raw = X.reshape(len(X), -1)
    strength = {int(o): base.longest_h1_bar(raw[obj == o]) for o in np.unique(obj)}
    ranked = sorted(strength, key=strength.get, reverse=True)[:n_keep]
    chosen = sorted(ranked)
    print("  object loop strengths (raw, longest H1):")
    for o in sorted(strength, key=strength.get, reverse=True):
        mark = "  <- kept" if o in chosen else ""
        print(f"     obj {o:>3}: {strength[o]:.4f}{mark}")
    keep = np.isin(obj, chosen)
    remap = {old: new for new, old in enumerate(chosen)}
    obj_new = np.array([remap[int(o)] for o in obj[keep]], int)
    print(f"  selected {len(chosen)} objects {chosen} -> remapped to 0..{len(chosen)-1}")
    return X[keep], obj_new, ang[keep]


def subsample_by_angle(obj, ang, n_keep, total_angles=N_ANGLES_FULL):
    """Keep n_keep EVENLY-SPACED angles per object (stratified by the known rotation
    label), so each circle stays uniformly sampled and the H1 loop survives at low
    point counts -- unlike random thinning, which leaves gaps that fragment the loop.
    All objects share the angle labels 0..71, so one evenly-spaced angle set applies
    to every object. Returns a boolean mask over all points."""
    if n_keep is None or n_keep >= total_angles:
        return np.ones(len(obj), bool)
    keep = np.unique(np.linspace(0, total_angles, n_keep, endpoint=False).astype(int))
    return np.isin(ang, keep)


def subsample_per_object(obj, frac=1.0, seed=SEED):
    """Random subsample WITHIN each object/class (keeps every object present, thins
    each circle). Random, not angle-stratified -- does not exploit the known angular
    structure. Floor of 12 points/object so a class is never starved. frac>=1 -> all.
    Returns a boolean mask over all points."""
    if frac >= 1.0:
        return np.ones(len(obj), bool)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(obj), bool)
    for o in np.unique(obj):
        idx = np.where(obj == o)[0]
        k = min(max(int(round(frac * len(idx))), 12), len(idx))
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def coil_topology_figures(reps, names, obj, task, use_bootstrap, exclude_output=True,
                          tag="", seed=SEED):
    """Layer-persistence diagrams and MLP-persistence barcode for the points passed in
    (a single object, a subset, or the full cloud, depending on the caller). `tag`
    distinguishes the output files/titles. Also prints the RAW (pre-recurrence) H1
    bars the pullback tower produces. exclude_output=False for the autoencoder (last
    layer is the loop-bearing bottleneck, not a collapsing output)."""
    from ltep import pipeline as pl, plots
    from ltep.datasets import coil100 as base
    from ltep import runtime as rs
    sfx = f"_{tag}" if tag else ""
    Q = [base.diameter_normalize(R) for R in reps]
    with rs.measure() as eps_t:
        eps_res = pl.select_epsilon(Q, use_bootstrap=use_bootstrap, n_boot=pl.N_BOOT,
                                    exclude_output=exclude_output, rng=seed)

    # per-layer epsilon audit (two scales, no max-combination):
    #   eps_H1 '~' suffix = sub-threshold (tier 2: most-persistent loop, not significant)
    #   eps_H1 '-'        = no loop at all (tier 3: geometric fallback)
    print(f"  epsilon audit [{tag or 'all'}]:")
    print(f"    {'layer':>11} | {'eps_H0':>7} | {'eps_H1':>7} | {'nH0':>4} | {'nH1':>4}")
    for nm, pli in zip(names, eps_res["per_layer"]):
        t1 = pli.get("h1_tier", 3)
        e1 = "  -  " if t1 == 3 else (f"{pli['eps_H1_used']:.3f}" + ("" if t1 == 1 else "~"))
        print(f"    {nm:>11} | {pli['eps_H0_used']:7.3f} | {e1:>7} | "
              f"{pli['n_sig_H0']:4d} | {pli.get('n_sig_H1',0):4d}")

    # layer-persistence diagrams (both epsilon lines drawn: green=H0, orange dashed=H1)
    plots.plot_layer_persistence(
        Q, eps_res, layer_names=names,
        path=os.path.join(FIGDIR, f"coil_{task}{sfx}_layer_diagrams.png"))

    # trajectory flow across layers (per-point H0-community flow; pass per-point object
    # or angle ids as `labels=` to colour it -- not in scope here, so uncoloured)
    plots.plot_trajectory_flow(
        Q, eps_res, layer_names=names,
        path=os.path.join(FIGDIR, f"coil_{task}{sfx}_trajectory_flow.png"))

    # TWO barcodes, no max-combination: one read at the H0 epsilon, one at the H1
    # epsilon. Each displays H0 and H1 at its own scale. NO resampling.
    convs, times = {}, {}
    for scheme, eps_seq in (("H0", eps_res["epsilons_H0"]),
                            ("H1", eps_res["epsilons_H1"])):
        with rs.measure() as ct:
            c = pl.convergence_depth(Q, eps_seq, significance=False,
                                     augment_output=False, exclude_output=exclude_output)
        convs[scheme] = c; times[scheme] = ct["seconds"]
        h1 = np.asarray(c["ref_bars"][1], float).reshape(-1, 2)
        print(f"  [{task} {tag or 'all'} | eps@{scheme}] H1 bars: {h1.tolist()}  "
              f"d*={c['d_star']}")
        plots.plot_mlp_persistence(
            c, layer_names=names, epsilons=eps_seq,
            title=f"COIL [{task}] MLP persistence -- epsilon@{scheme} ({tag or 'all'})",
            path=os.path.join(FIGDIR, f"coil_{task}{sfx}_mlp_barcode_{scheme}.png"))
    # the conv that drives d*/prunability: H1 for the loop-bearing AE, H0 otherwise
    primary = "H1" if not exclude_output else "H0"
    print(f"  saved coil_{task}{sfx} layer_diagrams + barcodes (H0,H1); "
          f"d* from eps@{primary}")
    return convs[primary], eps_t["seconds"], times[primary]


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def plot_task_tracking(task, curve, spread, layer_names):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(curve)), curve, yerr=spread, marker="o", capsize=3)
    diffs = np.diff(curve)
    if diffs.size:
        k = int(np.argmin(diffs))
        ax.annotate(f"largest drop\n{layer_names[k]}->{layer_names[k+1]}",
                    xy=(k + 1, curve[k + 1]), fontsize=8,
                    xytext=(k + 1, curve.max() * 0.6),
                    arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xticks(range(len(curve)))
    ax.set_xticklabels(layer_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("rotation-loop persistence (longest H1 bar)")
    ax.set_title(f"COIL [{task}]: where the rotation loop survives / dies")
    fig.tight_layout()
    p = os.path.join(FIGDIR, f"coil_{task}_loop_tracking.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  saved {os.path.basename(p)}")


def plot_overlay(curves):
    """Overlay loop-strength vs FRACTIONAL depth for all tasks (different depths)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    label = {"ae": "autoencoder (preserve)", "clf": "classifier (invariance)",
             "reg": "angle regression (linearise)"}
    for task, (curve, _sp, names) in curves.items():
        xs = np.linspace(0, 1, len(curve))
        ax.plot(xs, curve, "-o", ms=4, label=label.get(task, task))
    ax.set_xlabel("fractional depth (input $\\to$ output/bottleneck)")
    ax.set_ylabel("rotation-loop persistence (longest H1 bar)")
    ax.set_title("Does the network keep the known rotation loop? (by task)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "coil_loop_tracking_overlay.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\nsaved {os.path.basename(p)}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main(tasks=("ae", "clf"), n_objects=10, epochs=150, enc_depth=3, dense_depth=4,
         bottleneck_dim=32, ae_width=256, group_size=1, subsample=1.0, angles=24,
         select_by_loop=False, select_pool=None, collapse_ae=False, collapse_seeds=5,
         use_bootstrap=True, force_retrain=False, outdir=None, baselines=False):
    global FIGDIR
    from ltep.datasets import coil100 as base
    from ltep import pipeline as pl, output
    # when selecting by loop strength, scan a larger pool then keep the best n_objects
    pool = (select_pool or max(20, 4 * n_objects)) if select_by_loop else n_objects
    X, obj, ang = base.load_coil100(n_objects=pool, img_size=IMG_SIZE)
    if select_by_loop:
        print(f"\n=== object selection by loop strength (scan {len(np.unique(obj))} "
              f"-> keep {n_objects}) ===")
        X, obj, ang = select_top_loop_objects(X, obj, ang, n_objects)

    # AE encoder-collapse mode: retrain shallower encoders, keep recon + loops.
    if collapse_ae:
        FIGDIR = outdir or output.run_dir("coil100", tag=f"collapse_ae_bn{bottleneck_dim}")
        input_verification(X, obj, ang)
        enc_depths = list(range(enc_depth, 0, -1))
        agg, rdepth = ae_collapse_sweep(
            X, obj, enc_depths=enc_depths,
            bottleneck_dim=bottleneck_dim, ae_width=ae_width, epochs=epochs,
            seeds=range(collapse_seeds), outdir=FIGDIR)
        # non-topological depth-selection baselines (reconstruction-only, CKA)
        if baselines:
            ae_collapse_baselines(
                X, obj, agg, enc_depths=enc_depths, bottleneck_dim=bottleneck_dim,
                ae_width=ae_width, epochs=epochs, outdir=FIGDIR)
        # per-object topology figures for the collapsed encoder (and the full one to
        # contrast): layer diagrams, MLP-persistence barcodes, trajectory flow.
        for d in {rdepth, enc_depth} - {None}:
            collapsed_ae_figures(X, obj, ang, d, bottleneck_dim=bottleneck_dim,
                                 ae_width=ae_width, epochs=epochs, angles=angles,
                                 use_bootstrap=use_bootstrap)
        print("\nDone (AE collapse). Output:", FIGDIR)
        return

    is_train = base.split_by_angle(obj, ang, seed=SEED)
    print(f"COIL-100: {X.shape}, {len(np.unique(obj))} objects x 72 angles; "
          f"train {is_train.sum()} / test {(~is_train).sum()} (split by angle)")

    # all artefacts go to results/coil100/<params>_<timestamp>/ (figures + log + params)
    run_name = (f"gs{group_size}_a{angles or 72}_bn{bottleneck_dim}"
                f"_w{ae_width}_ep{epochs}")
    FIGDIR = outdir or output.run_dir("coil100", tag=run_name)
    output.save_params(FIGDIR, dict(
        tasks=list(tasks), n_objects=n_objects, epochs=epochs, enc_depth=enc_depth,
        dense_depth=dense_depth, bottleneck=bottleneck_dim, ae_width=ae_width,
        group_size=group_size, subsample=subsample, angles=angles, use_bootstrap=use_bootstrap,
        **dict(pl.PARAMS)))
    print(f"output -> {FIGDIR}")

    # (V) input-loop ground-truth check first -- anchors 'preserved' vs 'resolved'
    input_verification(X, obj, ang)

    curves = {}
    for task in tasks:
        print("\n" + "#" * 70)
        print(f"# TASK: {task}")
        print("#" * 70)
        reps, names, metric = run_task_network(
            task, X, obj, ang, is_train, epochs, enc_depth, dense_depth,
            bottleneck_dim=bottleneck_dim, ae_width=ae_width,
            force_retrain=force_retrain)
        curve, spread = loop_strength_curve(reps, obj)
        print(f"  loop strength per layer: "
              f"{dict(zip(names, np.round(curve, 3).tolist()))}")
        plot_task_tracking(task, curve, spread, names)
        excl = (task != "ae")            # AE last layer = bottleneck (a representation)
        framework_carrier(reps, obj, use_bootstrap=use_bootstrap, exclude_output=excl)

        # heavy figures (pullback tower + recurrence) run PER GROUP of objects so each
        # stays small and fast. group_size=1 -> per object (72 pts, all angles, clean
        # single loop); larger -> a pooled subset. Each object keeps all 72 angles.
        objs = np.unique(obj)
        groups = make_groups(objs, group_size)
        # stratified angle keep (preserves each circle); falls back to random thinning
        if angles is not None:
            keep = subsample_by_angle(obj, ang, angles)
        else:
            keep = subsample_per_object(obj, frac=subsample)
        d_stars = []
        for gi, g in enumerate(groups):
            m = np.isin(obj, g) & keep
            reps_g, obj_g, ang_g = [R[m] for R in reps], obj[m], ang[m]
            tag = (f"obj{int(g[0])}" if group_size <= 1
                   else f"grp{gi}_{'-'.join(str(int(x)) for x in g)}")
            note = (f" ({angles} stratified angles/object)" if angles is not None
                    else f" (random {subsample:.0%}/object)" if subsample < 1.0 else "")
            print(f"  --- topology figures [{tag}] on {m.sum()} points{note} ---")
            conv, t_eps, t_conv = coil_topology_figures(
                reps_g, names, obj_g, task, use_bootstrap=use_bootstrap,
                exclude_output=excl, tag=tag)
            d_stars.append((tag, conv["d_star"]))      # same conv used everywhere
            if task == "ae":             # does this object's embedding hold its loop?
                embedding_loop_count(reps_g, names, obj_g, task, tag=tag, ang=ang_g)
            if task == "clf":            # d*-prunability + runtime (reports SAME conv)
                classifier_prunability(conv, len(reps_g), t_eps, t_conv)

        # (3) d* statistics across subgroups + recommended (max) prune depth
        vals = [d for _, d in d_stars]
        if vals:
            mx = max(vals)
            worst = [t for t, d in d_stars if d == mx]
            print(f"\n=== [{task}] d* across {len(vals)} subgroups ===")
            print(f"  per-subgroup d*: {dict(d_stars)}")
            print(f"  min={min(vals)}  median={int(np.median(vals))}  max={mx}  "
                  f"mean={np.mean(vals):.2f}")
            print(f"  RECOMMENDED d* = {mx} (max across subgroups; set by {worst}). "
                  f"Prune layers after {mx}: the deepest subgroup still reorganises up "
                  f"to {mx}, so keeping fewer layers would cut a subgroup that needs them.")
        curves[task] = (curve, spread, names)

    if len(curves) >= 2:
        plot_overlay(curves)

    print("\n=== READING ===")
    print("  AE  : loop strength rises into the encoder and holds to the bottleneck")
    print("        -> loop PRESENT and PRESERVED (signal H1, no simplification).")
    print("  CLF : the rotation loop is PRESENT at the input and RESOLVED within the")
    print("        first few layers (barcode: H1 born at input_flat, dies by layers 2-3)")
    print("        -> the task-irrelevant within-class loop is simplified early, while")
    print("        the loop-strength curve's big drop at dense_1 marks where it starts.")
    print("  REG : loop strength drops early -> the circle is linearised (task needs it).")
    print("  The loop is SIMPLIFIED iff the task requires it -- validated on a loop we")
    print("  KNOW is present. (Barcode = presence/where-resolved; d* = prune depth.)")
    print("\nDone. Figures in:", FIGDIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=PARAMS["tasks"],
                    choices=["ae", "clf", "reg"])
    ap.add_argument("--objects", type=int, default=PARAMS["n_objects"])
    ap.add_argument("--epochs", type=int, default=PARAMS["epochs"])
    ap.add_argument("--enc-depth", type=int, default=PARAMS["enc_depth"])
    ap.add_argument("--dense-depth", type=int, default=PARAMS["dense_depth"])
    ap.add_argument("--bottleneck", type=int, default=PARAMS["bottleneck"],
                    help="AE bottleneck width (room for the 10 cycles)")
    ap.add_argument("--ae-width", type=int, default=PARAMS["ae_width"],
                    help="AE encoder/decoder hidden width")
    ap.add_argument("--group-size", type=int, default=PARAMS["group_size"],
                    help="objects per topology figure (1 = per object, fast & clean; "
                         "larger pools objects -- needed for the classifier's "
                         "between-class structure)")
    ap.add_argument("--subsample", type=float, default=PARAMS["subsample"],
                    help="random fraction of points kept PER OBJECT/CLASS (used only "
                         "when --angles is 0/unset). Stratified --angles is preferred.")
    ap.add_argument("--angles", type=int, default=PARAMS["angles"],
                    help=f"keep this many EVENLY-SPACED angles per object (default "
                         f"{PARAMS['angles']}; 0 = all 72). Stratified, so the rotation "
                         "loop survives at low point counts -- preferred over --subsample.")
    ap.add_argument("--select-by-loop", action="store_true",
                    help="scan a larger pool of objects and keep the n with the "
                         "STRONGEST input rotation loop (drops near-symmetric objects "
                         "that have no circle to track).")
    ap.add_argument("--select-pool", type=int, default=None,
                    help="how many objects to scan when --select-by-loop (default "
                         "max(20, 4*objects)).")
    ap.add_argument("--baselines", action="store_true",
                    help="with --collapse-ae: add non-topological depth-selection "
                         "baselines (reconstruction-only and CKA-similarity) and save "
                         "coil_baselines.json. Random-position drop is degenerate for a "
                         "uniform-width encoder, so it is not included.")
    ap.add_argument("--collapse-ae", action="store_true",
                    help="AE encoder-collapse mode: retrain encoders from --enc-depth "
                         "down to 1 and report the smallest that keeps reconstruction "
                         "AND the per-object loops (set --enc-depth high to start "
                         "over-provisioned, e.g. 5).")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds for --collapse-ae (default 5).")
    ap.add_argument("--alpha", type=float, default=None,
                    help=f"scale-axis significance level (dataset default {PARAMS['alpha']})")
    ap.add_argument("--fast", action="store_true", help="skip the bootstrap band")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    from ltep import pipeline as _pl, output
    _pl.set_alpha(args.alpha if args.alpha is not None else PARAMS["alpha"])
    angles = args.angles if args.angles and args.angles > 0 else None
    run_name = (f"gs{args.group_size}_a{angles or 72}_bn{args.bottleneck}"
                f"_w{args.ae_width}_ep{args.epochs}"
                + ("_sel" if args.select_by_loop else ""))
    rd = output.run_dir("coil100", tag=run_name)
    with output.capture(rd):
        main(tasks=tuple(args.tasks), n_objects=args.objects, epochs=args.epochs,
             enc_depth=args.enc_depth, dense_depth=args.dense_depth,
             bottleneck_dim=args.bottleneck, ae_width=args.ae_width,
             group_size=args.group_size, subsample=args.subsample, angles=angles,
             select_by_loop=args.select_by_loop, select_pool=args.select_pool,
             collapse_ae=args.collapse_ae, collapse_seeds=args.seeds,
             use_bootstrap=(not args.fast), force_retrain=args.retrain, outdir=rd,
             baselines=args.baselines)
