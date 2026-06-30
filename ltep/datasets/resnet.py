#!/usr/bin/env python
# coding: utf-8
"""
resnet_family_comparison.py
===========================

Run the unified pipeline across the CIFAR-10 ResNet family (20/32/44/56) to report
TWO things -- neither of which is a pruning claim:

  1. CONSISTENCY: d* vs total depth. Each model is SEPARATELY trained (these are not
     truncations of one another), so this asks whether the topological convergence
     depth behaves as a sensible, stable function of architecture depth. On compact
     ResNets we EXPECT d* to sit near the end (H0 cluster-merging continues almost to
     the logits) -- i.e. NO large inert block, so we explicitly do NOT claim pruning.

  2. RUNTIME / MEMORY SCALING of MLP-persistence vs depth (number of representations
     grows 10 -> 14 -> 18 -> 22 across the family). Per-stage seconds + peak MB, for
     the paper's complexity section.

Reuses extraction from diagnostic_pretrained_cifar.py and the pipeline from pipeline.py.

NO-RESAMPLING DESIGN (matches the COIL classifier): significance lives ONCE, in the
per-layer epsilon choice (the scale-axis bootstrap tau band). The layer axis
(MLP-persistence) is then pure tracking -- every cycle/component present at the chosen
epsilon is a real event, drawn and counted; d* and the stable-topology tail are read
directly from the single full-data barcode's births/deaths. The across-SEED variation
of d* (different data subsamples) is the robustness check; there is no within-run
recurrence gate.

Models (chenyaofo/pytorch-cifar-models, CIFAR-10): resnet20/32/44/56 = 6n+2, n=3/5/7/9.
Blocks per model: 3*n  -> reps = 3*n + 1 (logits).  Top-1 ~92.6/93.5/94.0/94.4%.

    python resnet_family_comparison.py                 # endpoints 20 & 56 (recommended start)
    python resnet_family_comparison.py --all           # 20/32/44/56
    python resnet_family_comparison.py --seeds 3        # average d* over seeds

Network: torch.hub download needs github.com (your machine; NOT the sandbox).
"""

import sys
import argparse
import time
from collections import Counter
import numpy as np

# --- make the repo root importable whether run as a script or via -m ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# all figures are written under this dir (set per-run in __main__ via ltep.output)
OUTDIR = "."
def _out(name):
    return os.path.join(OUTDIR, name)


def composed_h0_tower(latents, epsilons, read_idx=None):
    """Faithful, edge-level H0 MLP-persistence with per-layer eps_j AND layer
    composition (Proposal 2 done correctly).

    Composition is NOT 'jump and intersect endpoints' -- that re-admits edges an
    intermediate layer would forbid. By functoriality the composed map carries every
    intermediate constraint, which combinatorially is the FULL VR information across
    the span. We realise it at the edge level (the paper's effective-scale identity
    E_i = cap_{j>=i} VR_{eps_j}(X_j) cap E^out): an edge {r,s} survives the downward
    intersection at layer i iff dist_j(r,s) <= eps_j at EVERY layer j>=i. Equivalently,
    with the per-layer normalised distance D_j = dist_j/eps_j, the edge's birth layer is
    1 + (last layer j where D_j > 1) -- the layer just after its last violation, over ALL
    layers, kept or composed-over. No simplex tree is built: each layer contributes only
    an O(N^2) distance comparison (a logical AND on the surviving edges), and H0 is
    union-find on the resulting graph.

    `read_idx` (sorted layer indices, default all) are the layers at which the barcode
    is read; composing/striding reads at fewer indices while every intermediate VR still
    constrains the edges. Returns beta0 per read layer, the active transitions, the
    redundancy R and collapsed depth (at the read granularity), all in read-layer units.
    """
    import numpy as np
    L = len(latents)
    N = len(latents[0])
    if read_idx is None:
        read_idx = list(range(L))
    read_idx = sorted(set(int(i) for i in read_idx))

    # last layer at which each edge violates its own eps_j (-1 = never), over ALL layers
    last_violation = -np.ones((N, N), dtype=int)
    for j in range(L):
        Xj = np.asarray(latents[j], float)
        d = np.sqrt(np.maximum(((Xj[:, None, :] - Xj[None, :, :]) ** 2).sum(-1), 0.0))
        last_violation = np.where(d > epsilons[j], j, last_violation)
    birth_full = last_violation + 1                      # 0..L (L = never appears)

    # snap each edge's birth to the first READ layer >= birth_full
    read_arr = np.asarray(read_idx)
    pos = np.searchsorted(read_arr, birth_full)          # index into read_arr (N x N)
    appears = pos < len(read_arr)
    birth_read = np.where(appears, read_arr[np.clip(pos, 0, len(read_arr) - 1)], L + 1)

    # beta0 per read layer by incremental union-find (add edges as the read layer grows)
    iu = np.triu_indices(N, k=1)
    e_birth = birth_read[iu]
    rs_pairs = np.stack(iu, axis=1)
    order = np.argsort(e_birth, kind="stable")
    parent = np.arange(N)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    beta0, ptr, comps = [], 0, N
    for v in read_idx:
        while ptr < len(order) and e_birth[order[ptr]] <= v:
            r, s = rs_pairs[order[ptr]]; ptr += 1
            ra, sb = find(r), find(s)
            if ra != sb:
                parent[ra] = sb; comps -= 1
        beta0.append(comps)

    # stages = maximal runs of constant beta0 across read layers; R / collapsed depth
    stages = 1
    for a in range(1, len(beta0)):
        if beta0[a] != beta0[a - 1]:
            stages += 1
    active = [read_idx[a] for a in range(1, len(beta0)) if beta0[a] != beta0[a - 1]]
    n_read = len(read_idx)
    return dict(read_idx=read_idx, beta0=beta0, active_transitions=active,
                collapsed_depth=stages, redundancy=n_read - stages,
                n_read_layers=n_read)


def _read_indices(L, stride):
    """Read layers for a composed tower: keep endpoints, stride the interior."""
    if stride is None or stride <= 1:
        return list(range(L))
    keep = list(range(0, L, stride))
    if keep[-1] != L - 1:
        keep.append(L - 1)
    return keep


def _composed_epsilons(latents, read_idx, use_bootstrap, n_boot, rng):
    """Per-layer eps_j for the composed tower. Every layer keeps its OWN eps_j (the
    constraint), but the expensive bootstrap-significance band is run only at the READ
    layers; the composed-over (constraint-only) layers take the cheap heuristic eps_j.
    When read_idx is all layers this is just the bootstrap on every layer. Returns the
    full-length per-layer eps list (used as the AND-constraints in composed_h0_tower)."""
    # cheap heuristic eps_j for every layer (the constraint scale)
    eps = list(pl.select_epsilon(latents, use_bootstrap=False, max_hom_dim=0,
                                 n_boot=n_boot, rng=rng)["epsilons_H0"])
    if use_bootstrap:
        if len(read_idx) == len(latents):
            eps = list(pl.select_epsilon(latents, use_bootstrap=True, max_hom_dim=0,
                                         n_boot=n_boot, rng=rng)["epsilons_H0"])
        else:                                   # bootstrap only the read layers
            read_lat = [latents[i] for i in read_idx]
            eps_r = pl.select_epsilon(read_lat, use_bootstrap=True, max_hom_dim=0,
                                      n_boot=n_boot, rng=rng)["epsilons_H0"]
            for k, i in enumerate(read_idx):
                eps[i] = eps_r[k]
    return eps

# ============================================================================
# PARAMETERS FOR THIS DATASET (pretrained ResNet family on CIFAR) -- edit here
# ============================================================================
PARAMS = dict(
    depths_default=[20, 56],
    depths_all=[20, 32, 44, 56],
    seeds=1,
    prep="pca",            # Hiraoka-global PCA features (--raw for block features)
    alpha=0.01,            # ResNet carrier is H0 (H1 is noise)
)

from ltep import pipeline as pl
from ltep import runtime as rs
from ltep.datasets.resnet_features import (
    find_basic_blocks, extract_block_features, cifar_subsample,
)

DEPTH_TO_HUB = {20: "cifar10_resnet20", 32: "cifar10_resnet32",
                44: "cifar10_resnet44", 56: "cifar10_resnet56"}
PCA_K = 10
NORMALIZE = "global"
PER_CLASS = 20


def load_model(depth):
    import torch
    m = torch.hub.load("chenyaofo/pytorch-cifar-models",
                       DEPTH_TO_HUB[depth], pretrained=True)
    m.eval()
    return m


def linear_probe_per_layer(model, layer_names, d_star, depth,
                           per_class_train=100, per_class_test=50, seed=0):
    """
    Linear-probe BASELINE: per representation, fit multinomial logistic regression
    on RAW block features (full dim, the standard linear-separability probe) and
    report TEST accuracy. The probe-train set is drawn from the CIFAR TRAIN split and
    the probe-test set from the TEST split (disjoint). Saves a probe-acc-vs-layer plot
    with the topological d* overlaid, so one can compare where linear separability
    saturates against the topological convergence depth. Computed once per model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr_imgs, tr_y = cifar_subsample(per_class_train, seed=seed, train=True)
    te_imgs, te_y = cifar_subsample(per_class_test, seed=seed + 1, train=False)
    tr_reps, names = extract_block_features(model, tr_imgs)
    te_reps, _ = extract_block_features(model, te_imgs)

    accs = []
    for Xtr, Xte in zip(tr_reps, te_reps):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(sc.transform(Xtr), tr_y)
        accs.append(float(clf.score(sc.transform(Xte), te_y)))

    print("     linear-probe test accuracy per layer:")
    for nm, a in zip(names, accs):
        print(f"       {nm:<14}{a:6.3f}")
    # where does the probe saturate? first layer within 1% of the max
    amax = max(accs)
    sat = next((i for i, a in enumerate(accs) if a >= amax - 0.01), len(accs) - 1)
    print(f"     probe saturates by layer {sat} ({names[sat]}); "
          f"topological d*={d_star}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(range(len(accs)), accs, "o-", color="tab:purple", label="linear-probe acc")
    if d_star is not None:
        ax.axvline(d_star, color="k", ls="--", lw=1.2, label=f"topological d*={d_star}")
    ax.axvline(sat, color="tab:orange", ls=":", lw=1.2, label=f"probe saturates @{sat}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"ResNet-{depth}: linear-probe baseline vs topological d*", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = f"resnet{depth}_linear_probe.png"
    fig.savefig(_out(path), dpi=150)
    plt.close(fig)
    print(f"     saved {_out(path)}")
    return dict(names=names, accs=accs, saturates_at=sat, d_star=d_star)


def runtime_scaling(depth, per_class_values, prep="pca", seed=0,
                    path="resnet_runtime_vs_N.png"):
    """
    Empirical COMPLEXITY of the MLP-persistence stage: sweep the analysis sample size
    N and time convergence_depth (single full-data barcode, NO resampling -- the
    no-resampling design: significance lives in the per-layer epsilon, the layer axis
    is pure tracking). Also records d* at each N -> a subsample-robustness check. Uses
    the fast (non-bootstrap) epsilon so the timing isolates the MLP-persistence cost.
    Fits a log-log slope (complexity exponent). Run once, on the smallest model.
    """
    print("\n" + "=" * 70)
    print(f"RUNTIME / ROBUSTNESS vs N  (ResNet-{depth}, MLP-persistence stage)")
    print("=" * 70)
    model = load_model(depth)
    Ns, secs, dstars = [], [], []
    for pc in per_class_values:
        imgs, _ = cifar_subsample(pc, seed=seed)
        reps, _ = extract_block_features(model, imgs)
        latents = (pl.preprocess_latents(reps, "pca", n_components=PCA_K,
                                         normalize=NORMALIZE) if prep == "pca"
                   else pl.preprocess_latents(reps, "raw"))
        eps = pl.select_epsilon(latents, use_bootstrap=False, rng=seed)["epsilons_H0"]
        with rs.measure() as m:
            conv = pl.convergence_depth(latents, eps, significance=False,
                                        augment_output=False, rng=seed)
        Ns.append(len(imgs)); secs.append(m["seconds"]); dstars.append(conv["d_star"])
        print(f"  N={len(imgs):4d}: MLP-persistence {m['seconds']:7.2f}s   d*={conv['d_star']}")

    slope = float(np.polyfit(np.log(Ns), np.log(secs), 1)[0]) if len(Ns) > 1 else float("nan")
    print(f"  log-log slope (complexity exponent) ~ {slope:.2f}")
    print(f"  d* vs N: {dict(zip(Ns, dstars))}  (robustness: d* should be ~stable)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(Ns, secs, "s-", color="tab:red", label="measured")
    ax.set_xlabel("N (analysis points)")
    ax.set_ylabel("MLP-persistence seconds")
    ax.set_title(f"Runtime scaling (slope ~ {slope:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_out(path), dpi=150)
    plt.close(fig)
    print(f"  saved {path}")
    return dict(Ns=Ns, seconds=secs, dstars=dstars, slope=slope)


def subsample_sweep(depth, per_class_values, seeds, prep="pca", use_bootstrap=True,
                    n_boot=None, layer_stride=1, path="resnet_subsample_sweep.png"):
    """W3 robustness check. The ResNet diagnostic is OBSERVATIONAL: the weights are
    fixed (no retraining), so the only randomness is (i) WHICH class-balanced CIFAR-10
    images form the analysis cloud (cifar_subsample seed) and (ii) the scale-axis
    bootstrap. This sweep quantifies how the redundancy R and the collapsed depth
    STABILISE as the per-class subsample grows: for each per-class size it recomputes
    the H0 reading over several input subsamples (seeds) and reports the across-seed
    mean +/- std of R and collapsed depth. Choose the smallest per-class past which the
    std plateaus. With layer_stride>1 the reading uses the composed H0 tower
    (composed_h0_tower): every intermediate VR still constrains the edges (per-layer
    eps_j, via the normalised max), the barcode is read at the strided layers, and the
    bootstrap runs only at those read layers. Saves an error-bar plot and a JSON. Best
    run on the deepest (wobbliest) model."""
    import json
    print("\n" + "=" * 70)
    print(f"SUBSAMPLE ROBUSTNESS SWEEP  (ResNet-{depth}: R, collapsed depth vs N; "
          f"weights FIXED, {len(seeds)} input subsamples per N)")
    print("=" * 70)
    model = load_model(depth)
    n_blocks = len(find_basic_blocks(model))
    if layer_stride > 1:
        print(f"  composed H0 tower, layer-stride {layer_stride} (every intermediate "
              f"VR still constrains edges; bootstrap only at read layers)")
    rows = []
    for pc in per_class_values:
        Rs, colls, Ns, eps_secs, pers_secs = [], [], [], [], []
        for s in seeds:
            imgs, _ = cifar_subsample(pc, seed=s)
            reps, names = extract_block_features(model, imgs)
            latents = (pl.preprocess_latents(reps, "pca", n_components=PCA_K,
                                             normalize=NORMALIZE) if prep == "pca"
                       else pl.preprocess_latents(reps, "raw"))
            if layer_stride > 1:
                read_idx = _read_indices(len(latents), layer_stride)
                with rs.measure() as t_eps:
                    eps = _composed_epsilons(latents, read_idx, use_bootstrap, n_boot, s)
                with rs.measure() as t_pers:
                    ct = composed_h0_tower(latents, eps, read_idx=read_idx)
                R, coll = ct["redundancy"], ct["collapsed_depth"]
            else:
                with rs.measure() as t_eps:
                    eps = pl.select_epsilon(latents, use_bootstrap=use_bootstrap,
                                            max_hom_dim=0, n_boot=n_boot,
                                            rng=s)["epsilons_H0"]
                with rs.measure() as t_pers:
                    conv = pl.convergence_depth(latents, eps, significance=False,
                                                augment_output=False, rng=s)
                R = conv["redundancy"]["redundancy"]
                coll = conv["redundancy"]["collapsed_depth"]
            Rs.append(R); colls.append(coll); Ns.append(len(imgs))
            eps_secs.append(t_eps["seconds"]); pers_secs.append(t_pers["seconds"])
        rec = dict(per_class=pc, N=int(np.mean(Ns)), n_seeds=len(seeds),
                   layer_stride=layer_stride,
                   R_mean=float(np.mean(Rs)), R_std=float(np.std(Rs)),
                   R_vals=[int(r) for r in Rs],
                   collapsed_mean=float(np.mean(colls)),
                   collapsed_std=float(np.std(colls)),
                   collapsed_vals=[int(c) for c in colls],
                   epsilon_s_mean=float(np.mean(eps_secs)),
                   persistence_s_mean=float(np.mean(pers_secs)))
        rows.append(rec)
        print(f"  per_class={pc:>3} (N={rec['N']:>4}, {len(seeds)} seeds): "
              f"R={rec['R_mean']:5.1f}+/-{rec['R_std']:4.1f}  "
              f"collapsed={rec['collapsed_mean']:5.1f}+/-{rec['collapsed_std']:4.1f}  "
              f"[eps {rec['epsilon_s_mean']:.1f}s, pers {rec['persistence_s_mean']:.2f}s]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [r["N"] for r in rows]
    fig, (axR, axS) = plt.subplots(1, 2, figsize=(11, 4.2))
    axR.errorbar(Ns, [r["R_mean"] for r in rows], yerr=[r["R_std"] for r in rows],
                 fmt="o-", capsize=3, color="tab:red")
    axR.set_xlabel("N (analysis points)"); axR.set_ylabel("redundancy R")
    axR.set_title(f"ResNet-{depth}: R vs N (across-seed mean$\\pm$std)")
    axS.errorbar(Ns, [r["collapsed_mean"] for r in rows],
                 yerr=[r["collapsed_std"] for r in rows],
                 fmt="s-", capsize=3, color="tab:blue")
    axS.set_xlabel("N (analysis points)"); axS.set_ylabel("collapsed depth")
    axS.set_title("collapsed depth vs N")
    fig.tight_layout(); fig.savefig(_out(path), dpi=150); plt.close(fig)
    print(f"  saved {path}")
    jpath = f"resnet{depth}_subsample_sweep.json"
    with open(_out(jpath), "w") as f:
        json.dump(dict(depth=depth, n_reps=n_blocks + 1, prep=prep,
                       use_bootstrap=use_bootstrap,
                       n_boot=(n_boot if n_boot is not None else pl.N_BOOT),
                       seeds=list(seeds), rows=rows), f, indent=2)
    print(f"  saved {jpath}")
    return rows


def run_biband(depth, seed=0, prep="pca", t_grid=None, layer_stride=1, n_boot=None):
    """Bi-persistence band on ONE ResNet (single subsample -- it is a scale sweep on top
    of an already-costly pass). Restores the scale axis: at multiplier t the per-layer
    scales are t*eps_chosen[j] (confidence-set choice at t=1), read via the composed H0
    tower. Saves the beta0 surface + R(t) curve and a JSON with the stability scalar."""
    import json
    from ltep import bipersistence as bp
    print("\n" + "=" * 70)
    print(f"BI-PERSISTENCE BAND  (ResNet-{depth}, H0; single subsample seed {seed}, "
          f"layer-stride {layer_stride})")
    print("=" * 70)
    model = load_model(depth)
    imgs, _ = cifar_subsample(PER_CLASS, seed=seed)
    reps, names = extract_block_features(model, imgs)
    latents = (pl.preprocess_latents(reps, "pca", n_components=PCA_K, normalize=NORMALIZE)
               if prep == "pca" else pl.preprocess_latents(reps, "raw"))
    read_idx = _read_indices(len(latents), layer_stride)
    eps = _composed_epsilons(latents, read_idx, use_bootstrap=True,
                             n_boot=(n_boot or pl.N_BOOT), rng=seed)
    band = bp.bipersistence_band(latents, eps, t_grid=t_grid, read_idx=read_idx)
    rn = [names[i] for i in read_idx]
    p = bp.plot_bipersistence_band(band, _out(f"resnet{depth}_biband.png"),
                                   title=f"ResNet-{depth} bi-persistence band (H0)",
                                   layer_names=rn)
    with open(_out(f"resnet{depth}_biband.json"), "w") as f:
        json.dump(band, f, indent=2)
    print(f"  t grid     : {band['t_grid']}")
    print(f"  R(t)       : {band['R']}")
    print(f"  R at t=1   : {band['R_at_t1']}  (range over band [{band['R_min']},{band['R_max']}])")
    print(f"  stability  : {band['stability']:.2f} "
          f"(fraction of the t-band with the SAME active-transition set as t=1)")
    print(f"  saved {p} and resnet{depth}_biband.json")
    return band


def run_model(depth, seeds, prep="pca", full=True, layer_stride=1):
    model = load_model(depth)
    n_blocks = len(find_basic_blocks(model))
    print("\n" + "#" * 70)
    print(f"# ResNet-{depth}: {n_blocks} blocks -> {n_blocks + 1} representations"
          + (f"  [layer-stride {layer_stride}: functoriality coarsening]"
             if layer_stride > 1 else ""))
    print("#" * 70)

    dstars, carriers, timings_acc, total_times = [], [], None, []
    Rs, colls = [], []
    last_conv, last_names, last_latents, last_eps = None, None, None, None
    for s in seeds:
        imgs, _ = cifar_subsample(PER_CLASS, seed=s)

        # --- extraction (timed separately from persistence) ---
        with rs.measure() as m_ext:
            reps, names = extract_block_features(model, imgs)
        if prep == "pca":
            latents = pl.preprocess_latents(reps, "pca", n_components=PCA_K,
                                            normalize=NORMALIZE)
        else:
            latents = pl.preprocess_latents(reps, "raw")
        read_idx = _read_indices(len(latents), layer_stride)

        timings = {"extract": m_ext}
        t0 = time.perf_counter()

        with rs.measure() as m:
            eps_res = pl.select_epsilon(latents, use_bootstrap=full, max_hom_dim=0,
                                        n_boot=pl.N_BOOT, rng=s)   # H0-only scale band
        timings["epsilon"] = m
        eps = eps_res["epsilons_H0"]    # ResNet carrier is H0 (H1 is noise); read at eps_H0

        # NO resampling on the layer axis: significance is already in the epsilon
        # choice, so MLP-persistence just tracks how the surviving features span layers.
        with rs.measure() as m:
            conv = pl.convergence_depth(latents, eps, significance=False,
                                        augment_output=False, rng=s)
            # reported R/collapsed: composed H0 tower (faithful, all constraints kept);
            # at stride 1 this reads every layer, matching the simplex-tree tower.
            if layer_stride > 1:
                ct = composed_h0_tower(latents, eps, read_idx=read_idx)
                R_seed, coll_seed = ct["redundancy"], ct["collapsed_depth"]
            else:
                R_seed = conv["redundancy"]["redundancy"]
                coll_seed = conv["redundancy"]["collapsed_depth"]
        timings["mlp_persistence"] = m

        carrier = pl.carrier_dimension(conv)
        total = time.perf_counter() - t0
        last_conv, last_names = conv, names
        last_latents, last_eps = latents, eps_res

        dstars.append(conv["d_star"]); Rs.append(R_seed); colls.append(coll_seed)
        carriers.append(carrier)
        total_times.append(total)
        # accumulate per-stage seconds across seeds (report the median later)
        if timings_acc is None:
            timings_acc = {k: [] for k in timings}
        for k, rec in timings.items():
            timings_acc[k].append((rec["seconds"], (rec.get("py_peak_bytes") or 0) / 1e6))

        print(f"  seed {s}: R={R_seed} collapsed={coll_seed} carrier=H{carrier}  "
              f"total={total:.2f}s")

    # ---- per-model seed-summary statistics ----
    dvals = [d for d in dstars if d is not None]
    d_mode = int(np.bincount(dvals).argmax()) if dvals else None
    agreement = (sum(1 for d in dvals if d == d_mode) / len(dvals)) if dvals else 0.0
    d_mean = float(np.mean(dvals)) if dvals else float("nan")
    d_std = float(np.std(dvals)) if dvals else float("nan")
    carrier_counts = Counter(carriers)
    n_seeds = len(carriers)
    dom_carrier, dom_n = carrier_counts.most_common(1)[0]
    carrier_frac = {f"H{c}": f"{carrier_counts[c]}/{n_seeds}" for c in sorted(carrier_counts)}

    R_mean, R_std = float(np.mean(Rs)), float(np.std(Rs))
    coll_mean, coll_std = float(np.mean(colls)), float(np.std(colls))
    print(f"\n  -> ResNet-{depth}: redundancy R = {R_mean:.1f}+/-{R_std:.1f}, "
          f"collapsed depth = {coll_mean:.1f}+/-{coll_std:.1f} of {n_blocks + 1} reps "
          f"(read at layer-stride {layer_stride}; R is a scale-dependent count and is "
          f"noisy across input subsamples -- the stable reading is where merging "
          f"concentrates, corroborated by the linear probe).")
    print(f"     carrier stability: {carrier_frac}  "
          f"(dominant H{dom_carrier}: {dom_n}/{n_seeds})")
    print(f"     {'stage':<16}{'median_s':>10}{'med_MB':>9}")
    stage_med = {}
    for k, vals in timings_acc.items():
        secs = np.median([v[0] for v in vals])
        mb = np.median([v[1] for v in vals])
        stage_med[k] = secs
        print(f"     {k:<16}{secs:>10.2f}{mb:>9.1f}")

    # MLP-persistence barcode for this model (from the last seed), block-labelled
    from ltep import plots
    bc_path = f"resnet{depth}_mlp_barcode.png"
    plots.plot_mlp_persistence(last_conv, path=_out(bc_path), layer_names=last_names,
                               epsilons=last_eps["epsilons_H0"],
                               title=f"ResNet-{depth} MLP persistence")
    # Per-layer persistence diagrams -- the epsilon sanity check
    ld_path = f"resnet{depth}_layer_diagrams_{prep}.png"
    plots.plot_layer_persistence(last_latents, last_eps, path=_out(ld_path),
                                 layer_names=last_names)
    # Betti-0 vs epsilon -- "why so few components?" (first / middle / last block)
    b0_path = f"resnet{depth}_betti0_{prep}.png"
    plots.plot_betti0_diagnostic(last_latents, last_eps, layer_names=last_names,
                                 path=_out(b0_path))
    # Per-layer LINEAR-PROBE baseline (comparison ask): where does linear
    # separability saturate, vs the topological d*?  Computed once per model.
    probe = linear_probe_per_layer(model, last_names, d_mode, depth)

    print(f"     saved {bc_path}, {ld_path}, {b0_path}")
    return dict(depth=depth, n_reps=n_blocks + 1, d_star=d_mode,
                d_agreement=agreement, d_mean=d_mean, d_std=d_std,
                R_mean=R_mean, R_std=R_std, R_vals=[int(r) for r in Rs],
                collapsed_mean=coll_mean, collapsed_std=coll_std,
                collapsed_vals=[int(c) for c in colls], layer_stride=layer_stride,
                carriers=carriers, carrier_frac=carrier_frac,
                dstars=dvals, probe=probe,
                median_total=float(np.median(total_times)),
                stage_median_s=stage_med)


def plot_family(results, path="resnet_family_dstar_runtime.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    depths = [r["n_reps"] for r in results]
    dstars = [r["d_star"] for r in results]
    persist_s = [r["stage_median_s"].get("mlp_persistence", np.nan) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: d* vs depth (consistency), with the y=x "merges to the end" reference
    ax1.plot(depths, dstars, "o-", color="tab:blue", label="d* (mode)")
    ax1.plot(depths, [d - 1 for d in depths], "--", color="0.6",
             label="last layer (no inert tail)")
    ax1.set_xlabel("# representations (depth)")
    ax1.set_ylabel("convergence d*")
    ax1.set_title("d* vs depth (consistency, NOT pruning)")
    ax1.legend(fontsize=8)
    # right: MLP-persistence runtime vs depth (scaling)
    ax2.plot(depths, persist_s, "s-", color="tab:red")
    ax2.set_xlabel("# representations (depth)")
    ax2.set_ylabel("MLP-persistence median seconds")
    ax2.set_title("Runtime scaling vs depth")
    fig.tight_layout()
    fig.savefig(_out(path), dpi=150)
    plt.close(fig)
    print(f"\nsaved {path}")


def main(depths, seeds, prep="pca", scaling=True, layer_stride=1):
    results = [run_model(d, seeds, prep=prep, layer_stride=layer_stride)
               for d in depths]

    print("\n" + "=" * 70)
    print("RESNET FAMILY SUMMARY  (redundancy diagnostic, NOT a pruning claim)")
    print("=" * 70)
    print(f"  {'model':<10}{'reps':>6}{'R (mean+-std)':>16}{'collapsed':>16}"
          f"{'carrier':>16}{'probe@':>8}{'persist_s':>11}")
    for r in results:
        cf = ", ".join(f"{k}:{v}" for k, v in r.get("carrier_frac", {}).items())
        psat = r.get("probe", {}).get("saturates_at", "-")
        print(f"  ResNet-{r['depth']:<4}{r['n_reps']:>6}"
              f"{r.get('R_mean',0):>9.1f}+/-{r.get('R_std',0):<4.1f}"
              f"{r.get('collapsed_mean',0):>9.1f}+/-{r.get('collapsed_std',0):<4.1f}"
              f"{cf:>16}{str(psat):>8}"
              f"{r['stage_median_s'].get('mlp_persistence', float('nan')):>11.2f}")
    print("\n  Reading: R is a SCALE-DEPENDENT count (governed by the per-layer epsilon /")
    print("  alpha / bootstrap) and is noisy across input subsamples; the STABLE reading")
    print("  is where merging concentrates (late blocks), corroborated by the linear")
    print("  probe ('probe@' = layer where separability saturates). We do NOT claim")
    print("  pruning. R/collapsed are read at the chosen layer-stride.")
    if len(results) >= 2:
        plot_family(results)

    if scaling:
        # complexity + subsample-robustness on the smallest model (cheapest)
        runtime_scaling(min(depths), per_class_values=[5, 10, 20, 40], prep=prep)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="20/32/44/56 (default: 20 & 56)")
    ap.add_argument("--seeds", type=int, default=PARAMS["seeds"])
    ap.add_argument("--raw", action="store_true",
                    help="use RAW block features (no PCA) instead of Hiraoka-global PCA")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="override N_BOOT for the epsilon band (e.g. 30 for speed)")
    ap.add_argument("--alpha", type=float, default=None,
                    help=f"scale-axis significance level (dataset default {PARAMS['alpha']})")
    ap.add_argument("--no-scaling", action="store_true",
                    help="skip the runtime-vs-N sweep")
    ap.add_argument("--subsample-sweep", action="store_true",
                    help="W3 robustness: sweep per-class subsample size and report R / "
                         "collapsed depth mean+-std over --seeds input subsamples "
                         "(weights fixed). Runs instead of the family analysis.")
    ap.add_argument("--sweep-per-class", type=str, default="20,35,50,75",
                    help="comma-separated per-class sizes for --subsample-sweep "
                         "(default 20,35,50,75; N = 10x these).")
    ap.add_argument("--sweep-depth", type=int, default=None,
                    help="model depth for --subsample-sweep (default: deepest requested).")
    ap.add_argument("--fast-eps", action="store_true",
                    help="use the non-bootstrap epsilon in --subsample-sweep (faster, "
                         "but does not reproduce the reported bootstrap-band variance).")
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="Proposal 2 (functoriality): compose/skip layers, analysing the "
                         "tower on every k-th representation (k>1) for a ~k-fold speedup. "
                         "Endpoints are always kept; the linear probe stays on full layers.")
    ap.add_argument("--per-class", type=int, default=None,
                    help=f"analysis-cloud size per CIFAR class for the FAMILY run "
                         f"(default {PER_CLASS}; N = 10x this). Raise it (e.g. 50 -> N=500) "
                         f"to steady R now that the H0-edge band makes the band cheap.")
    ap.add_argument("--biband", action="store_true",
                    help="bi-persistence band on ONE model (--sweep-depth, seed 0): sweep "
                         "a scale multiplier around the chosen epsilon and report the beta0 "
                         "surface, R(t) curve and a scale-stability scalar. Single run.")
    ap.add_argument("--biband-trange", type=str, default="0.7,1.3,7",
                    help="lo,hi,n for the bi-persistence band scale multipliers "
                         "(default 0.7,1.3,7 -> 7 values; t=1 = the chosen epsilon).")
    args = ap.parse_args()
    pl.set_alpha(args.alpha if args.alpha is not None else PARAMS["alpha"])
    if args.per_class is not None:
        PER_CLASS = args.per_class       # module global read by run_model
    if args.n_boot is not None:
        pl.N_BOOT = args.n_boot          # run_model passes pl.N_BOOT explicitly
        pl.PARAMS["N_BOOT"] = args.n_boot
    depths = PARAMS["depths_all"] if args.all else PARAMS["depths_default"]
    prep = "raw" if args.raw else PARAMS["prep"]
    from ltep import output
    sweep_depth = args.sweep_depth if args.sweep_depth else max(depths)
    tag = (f"{prep}_d{'-'.join(map(str,depths))}_s{args.seeds}_a{pl.PARAMS['ALPHA']}"
           f"_pc{PER_CLASS}_k{args.layer_stride}")
    if args.subsample_sweep:
        tag = f"subsweep_d{sweep_depth}_s{args.seeds}_a{pl.PARAMS['ALPHA']}_k{args.layer_stride}"
    if args.biband:
        tag = f"biband_d{sweep_depth}_pc{PER_CLASS}_k{args.layer_stride}_a{pl.PARAMS['ALPHA']}"
    OUTDIR = output.run_dir("resnet", tag=tag)
    output.save_params(OUTDIR, dict(depths=depths, seeds=args.seeds, prep=prep,
                                    scaling=(not args.no_scaling),
                                    subsample_sweep=args.subsample_sweep,
                                    sweep_per_class=args.sweep_per_class,
                                    sweep_depth=sweep_depth, per_class=PER_CLASS,
                                    layer_stride=args.layer_stride, **dict(pl.PARAMS)))
    with output.capture(OUTDIR):
        if args.biband:
            lo, hi, n = args.biband_trange.replace(" ", "").split(",")
            t_grid = np.round(np.linspace(float(lo), float(hi), int(n)), 4).tolist()
            run_biband(sweep_depth, seed=0, prep=prep, t_grid=t_grid,
                       layer_stride=args.layer_stride, n_boot=args.n_boot)
        elif args.subsample_sweep:
            pcs = [int(x) for x in args.sweep_per_class.replace(" ", "").split(",") if x]
            subsample_sweep(sweep_depth, per_class_values=pcs,
                            seeds=tuple(range(args.seeds)), prep=prep,
                            use_bootstrap=(not args.fast_eps), n_boot=args.n_boot,
                            layer_stride=args.layer_stride)
        else:
            main(depths=depths, seeds=tuple(range(args.seeds)),
                 prep=prep, scaling=(not args.no_scaling),
                 layer_stride=args.layer_stride)