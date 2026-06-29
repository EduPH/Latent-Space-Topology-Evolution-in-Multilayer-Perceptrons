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


def run_model(depth, seeds, prep="pca", full=True, significance=False):
    model = load_model(depth)
    n_blocks = len(find_basic_blocks(model))
    print("\n" + "#" * 70)
    print(f"# ResNet-{depth}: {n_blocks} blocks -> {n_blocks + 1} representations")
    print("#" * 70)

    dstars, carriers, timings_acc, total_times = [], [], None, []
    rstars, collapsed_depths, blocks_per_seed = [], [], []
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

        timings = {"extract": m_ext}
        t0 = time.perf_counter()

        with rs.measure() as m:
            eps_res = pl.select_epsilon(latents, use_bootstrap=full,
                                        n_boot=pl.N_BOOT, rng=s)   # scale-axis tau band
        timings["epsilon"] = m
        eps = eps_res["epsilons_H0"]    # ResNet carrier is H0 (H1 is noise); read at eps_H0

        # NO resampling on the layer axis: significance is already in the epsilon
        # choice, so MLP-persistence just tracks how the surviving features span layers.
        with rs.measure() as m:
            conv = pl.convergence_depth(latents, eps, significance=significance,
                                        augment_output=False, rng=s)
        timings["mlp_persistence"] = m

        # CIFAR is the H0 exemplar: H1 is non-recurrent noise (it surfaces in scattered
        # layers but never threads consecutive ones), and for a classifier the carrier is
        # H0 cluster-merging. We therefore read the metric in H0 ONLY -- recompute the
        # collapsible blocks / d* from the H0 events alone, so noise-H1 never enters.
        red = pl.collapsible_blocks({0: conv["per_dim"][0]}, conv["n_layers"])
        d_star_h0 = max(red["active_transitions"], default=None)
        carrier = 0
        total = time.perf_counter() - t0
        last_conv, last_names = conv, names
        last_latents, last_eps = latents, eps_res

        dstars.append(d_star_h0)
        rstars.append(red["redundancy"])
        collapsed_depths.append(red["collapsed_depth"])
        blocks_per_seed.append(tuple(map(tuple, red["collapsible_blocks"])))
        carriers.append(carrier)
        total_times.append(total)
        # accumulate per-stage seconds across seeds (report the median later)
        if timings_acc is None:
            timings_acc = {k: [] for k in timings}
        for k, rec in timings.items():
            timings_acc[k].append((rec["seconds"], (rec.get("py_peak_bytes") or 0) / 1e6))

        # d* is deterministic per seed now (no resampling); seed-to-seed variation
        # below measures robustness to the data subsample, not a resampling gate.
        print(f"  seed {s}: d*={d_star_h0}  R={red['redundancy']} "
              f"(collapsed depth {red.get('collapsed_depth')})  carrier=H{carrier}  "
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

    print(f"\n  -> ResNet-{depth}: d* mode={d_mode} (across-seed agreement "
          f"{agreement:.2f}), mean={d_mean:.1f}+/-{d_std:.1f}, range=[{min(dvals)},"
          f"{max(dvals)}] of {n_blocks + 1} reps  "
          f"(stable-topology tail = {max(n_blocks + 1 - 1 - (d_mode or 0), 0)} layers "
          f"with no births/deaths)")
    print(f"     carrier stability: {carrier_frac}  "
          f"(dominant H{dom_carrier}: {dom_n}/{n_seeds})")
    rvals = [r for r in rstars if r is not None]
    cvals = [c for c in collapsed_depths if c is not None]
    modal_blocks = Counter(blocks_per_seed).most_common(1)[0][0] if blocks_per_seed else ()
    print(f"     redundancy R = {np.mean(rvals):.1f}+/-{np.std(rvals):.1f} removable "
          f"layer(s); collapsed depth = {np.mean(cvals):.1f} of {n_blocks + 1} reps; "
          f"modal collapsible blocks = {list(modal_blocks) or 'none'} "
          f"(consistency, NOT a pruning claim)")
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
    # trajectory flow across blocks
    plots.plot_trajectory_flow(last_latents, last_eps, layer_names=last_names,
                               d_star=last_conv.get("d_star"),
                               path=_out(f"resnet{depth}_trajectory_{prep}.png"))
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
                carriers=carriers, carrier_frac=carrier_frac,
                dstars=dvals, probe=probe,
                redundancy_mean=float(np.mean(rvals)) if rvals else float("nan"),
                collapsed_depth_mean=float(np.mean(cvals)) if cvals else float("nan"),
                modal_blocks=list(modal_blocks),
                median_total=float(np.median(total_times)),
                stage_median_s=stage_med)


# pretrained CIFAR-10 top-1 accuracies (chenyaofo/pytorch-cifar-models), used as the
# ground-truth cost of simplification: a shallower member is the trained realization
# of collapsing a deeper one, so its accuracy is what that collapse would cost.
TOP1 = {20: 92.6, 32: 93.5, 44: 94.0, 56: 94.4}


def family_simplification(results, path="resnet_family_simplification.png"):
    """Treat the family as MUTUAL SIMPLIFICATIONS (no retrain). For each model we read
    the collapsed depth (n_reps - R) from the redundancy diagnostic and map it to the
    nearest trained family member by representation count; that member is the
    ground-truth network at the collapsed depth, and the gap in pretrained top-1 is
    the accuracy cost of the simplification. Small R / collapsed ~ full depth => the
    model uses its depth; large R that lands on a shallower member at little cost =>
    the deeper blocks were redundant and the shallower member IS the simplification."""
    rows = sorted(results, key=lambda r: r["n_reps"])
    print("\n" + "=" * 78)
    print("FAMILY AS MUTUAL SIMPLIFICATION  (shallower members = ground-truth collapses)")
    print("=" * 78)
    print(f"  {'model':<10}{'reps':>5}{'d*':>5}{'R':>6}{'collapsed':>10}"
          f"{'top1':>7}   simplifies toward")
    for r in rows:
        cd = r.get("collapsed_depth_mean", float("nan"))
        R = r.get("redundancy_mean", float("nan"))
        t1 = TOP1.get(r["depth"])
        # nearest member whose rep count is <= this model's reps and closest to cd
        cands = [x for x in rows if x["n_reps"] <= r["n_reps"]]
        nearest = min(cands, key=lambda x: abs(x["n_reps"] - cd)) if cands else r
        if nearest["depth"] == r["depth"] or abs(nearest["n_reps"] - cd) > 1.5:
            note = "itself (uses its depth)"
        else:
            dacc = t1 - TOP1.get(nearest["depth"], t1)
            note = f"ResNet-{nearest['depth']}  (cost {dacc:+.1f} pp top1)"
        print(f"  ResNet-{r['depth']:<4}{r['n_reps']:>5}{str(r['d_star']):>5}"
              f"{R:>6.1f}{cd:>10.1f}{(t1 if t1 is not None else float('nan')):>7.1f}"
              f"   {note}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = [r["n_reps"] for r in rows]
    cds = [r.get("collapsed_depth_mean", np.nan) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(reps, reps, "--", color="0.6", label="uses full depth (collapsed = reps)")
    ax.plot(reps, cds, "o-", color="tab:blue", label="collapsed depth (reps $-$ R)")
    for x in reps:                       # member-depth gridlines = available simplifications
        ax.axhline(x, color="0.9", lw=1, zorder=0)
    for r, c in zip(rows, cds):
        ax.annotate(f"ResNet-{r['depth']}\n{TOP1.get(r['depth'])}%",
                    (r["n_reps"], c), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("# representations (architecture depth)")
    ax.set_ylabel("collapsed depth (topological stages)")
    ax.set_title("ResNet family as mutual simplification")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(_out(path), dpi=150)
    plt.close(fig)
    print(f"\n  saved {path}")


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


def main(depths, seeds, prep="pca", scaling=True, significance=False):
    results = [run_model(d, seeds, prep=prep, significance=significance) for d in depths]

    print("\n" + "=" * 70)
    print("RESNET FAMILY SUMMARY  (d* is a consistency probe, not a pruning claim)")
    print("=" * 70)
    print(f"  {'model':<10}{'reps':>6}{'R':>5}{'coll':>6}{'d*':>5}{'agree':>7}"
          f"{'carrier':>14}{'probe@':>8}{'persist_s':>10}")
    for r in results:
        cf = ", ".join(f"{k}:{v}" for k, v in r.get("carrier_frac", {}).items())
        psat = r.get("probe", {}).get("saturates_at", "-")
        print(f"  ResNet-{r['depth']:<4}{r['n_reps']:>6}"
              f"{r.get('redundancy_mean', float('nan')):>5.1f}"
              f"{r.get('collapsed_depth_mean', float('nan')):>6.1f}"
              f"{str(r['d_star']):>5}{r.get('d_agreement', 0):>7.2f}{cf:>14}{str(psat):>8}"
              f"{r['stage_median_s'].get('mlp_persistence', float('nan')):>10.2f}")
        if r.get("modal_blocks"):
            print(f"        collapsible blocks: {r['modal_blocks']}")
    print("\n  Reading: R = removable layers from the COLLAPSIBLE BLOCKS (inner OR tail,")
    print("  Def. collapsible); 'coll' = collapsed depth (topological stages). Small R")
    print("  / coll ~ reps => the network uses its depth; an inner block flags candidate-")
    print("  redundant layers. d* is just the tail special case (last active layer). The")
    print("  family-simplification table maps each collapsed depth to a trained member.")
    if len(results) >= 2:
        plot_family(results)
        family_simplification(results)

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
    ap.add_argument("--significance", action="store_true",
                    help="apply the layer-axis recurrence gate (filters non-recurrent "
                         "noise H1, matching the diagnostic) so the carrier/R/d* reflect "
                         "only recurrence-significant events. Recommended for ResNet, "
                         "where H1 is noise; slower (adds resamples).")
    args = ap.parse_args()
    pl.set_alpha(args.alpha if args.alpha is not None else PARAMS["alpha"])
    if args.n_boot is not None:
        pl.N_BOOT = args.n_boot          # run_model passes pl.N_BOOT explicitly
        pl.PARAMS["N_BOOT"] = args.n_boot
    depths = PARAMS["depths_all"] if args.all else PARAMS["depths_default"]
    prep = "raw" if args.raw else PARAMS["prep"]
    from ltep import output
    tag = f"{prep}_d{'-'.join(map(str,depths))}_s{args.seeds}_a{pl.PARAMS['ALPHA']}"
    OUTDIR = output.run_dir("resnet", tag=tag)
    output.save_params(OUTDIR, dict(depths=depths, seeds=args.seeds, prep=prep,
                                    scaling=(not args.no_scaling), **dict(pl.PARAMS)))
    with output.capture(OUTDIR):
        main(depths=depths, seeds=tuple(range(args.seeds)),
             prep=prep, scaling=(not args.no_scaling), significance=args.significance)