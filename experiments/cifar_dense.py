#!/usr/bin/env python
# coding: utf-8
"""
check_pipeline_cifar.py
=======================

Apply the unified pipeline to CIFAR-10, compare two Stage-0 preprocessings (RAW vs
Hiraoka-PCA) on the SAME trained net, report PER-STAGE RUNTIME (+ peak memory), and
-- with --sweep -- run the depth-sweep keystone that validates the topological d*
against the actual accuracy-vs-depth curve.

Reuses architecture / data / caching from Experiment_cifar_dense.py.

    python check_pipeline_cifar.py            # FAST: provisional, quick wiring + compare
    python check_pipeline_cifar.py --full     # FULL: bootstrap band + recurrence + plots + runtime
    python check_pipeline_cifar.py --sweep    # FULL + train depths 1..DENSE_DEPTH and plot acc vs d*

Outputs (FULL): diag_layer_{raw,pca}.png, diag_mlp_{raw,pca}.png
Outputs (sweep): cifar_depth_sweep.png

Defaults reflect the deep-net regime: DENSE_DEPTH=10, NORMALIZE='global' (whiten was
shown to manufacture H1 artifacts on CIFAR), EPOCHS=60, PER_CLASS=20.
"""

import sys
import time
import numpy as np

# --- make the repo root importable whether run as a script or via -m ---
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# all figures are written under this dir (set per-run in __main__ via ltep.output)
OUTDIR = "."
def _out(name):
    return os.path.join(OUTDIR, name)

# ============================================================================
# PARAMETERS FOR THIS DATASET (CIFAR dense MLP) -- edit here, nowhere else
# ============================================================================
PARAMS = dict(
    alpha=0.01,            # second H0 pruning exemplar (alongside cardio)
)

from ltep import pipeline as pl, plots
from ltep import runtime as rs
from ltep.datasets.cifar_dense import (
    load_cifar10, class_balanced_subsample, train_or_load,
    head_layer_names, head_representations, N_CLASSES,
)

SEED = 1234
DENSE_DEPTH = 10         # deep head so the dense stack can actually saturate
PER_CLASS = 20           # 20*10 = 200 analysed points (pipeline runs TWICE)
EPOCHS = 60
PCA_K = 10               # fixed signal-subspace dimension
NORMALIZE = "global"     # 'global' recommended for CIFAR (whiten amplifies noise)
PLATEAU_MARGIN = 0.02    # accuracy gain beyond d* above which the diagnosis is FALSIFIED


# ----------------------------------------------------------------------------
# Runtime reporting
# ----------------------------------------------------------------------------

def print_runtime(name, timings, total_s):
    print(f"\n-- runtime [{name}] --")
    print(f"  {'stage':<16}{'seconds':>10}{'peak_MB':>10}")
    for stage, m in timings.items():
        mb = (m.get("py_peak_bytes") or 0) / 1e6
        print(f"  {stage:<16}{m['seconds']:>10.2f}{mb:>10.1f}")
    print(f"  {'TOTAL':<16}{total_s:>10.2f}")


# ----------------------------------------------------------------------------
# One pipeline run (timed, stage by stage)
# ----------------------------------------------------------------------------

def run_one(name, latents, full, seed=SEED):
    print("\n" + "#" * 70)
    print(f"# {name} preprocessing  (dims per rep: {[X.shape[1] for X in latents]})")
    print("#" * 70)
    timings = {}
    t_start = time.perf_counter()

    with rs.measure() as m:
        eps_res = pl.select_epsilon(latents, use_bootstrap=full, rng=seed)
    timings["epsilon"] = m
    epsilons = eps_res["epsilons"]

    with rs.measure() as m:
        conv = pl.convergence_depth(latents, epsilons, significance=full,
                                    augment_output=False, rng=seed)
    timings["convergence"] = m

    carrier = pl.carrier_dimension(conv)
    with rs.measure() as m:
        xc = pl.cross_check_bottleneck(latents, conv["d_star"], homology_dim=carrier)
    timings["cross_check"] = m

    with rs.measure() as m:
        anom = pl.output_loop_anomaly(latents, rng=seed)
    timings["output_anomaly"] = m

    total_s = time.perf_counter() - t_start
    result = dict(params=dict(pl.PARAMS), epsilons=epsilons,
                  epsilon_audit=eps_res["per_layer"], convergence=conv,
                  carrier_dim=carrier, cross_check=xc, output_anomaly=anom)
    pl.pretty_print(result)
    print_runtime(name, timings, total_s)

    if full:
        tag = name.lower()
        plots.plot_layer_persistence(latents, dict(per_layer=result["epsilon_audit"]),
                                     path=_out(f"diag_layer_{tag}.png"))
        plots.plot_trajectory_flow(latents, dict(per_layer=result["epsilon_audit"]),
                                   d_star=conv.get("d_star"),
                                   path=_out(f"diag_trajectory_{tag}.png"))
        plots.plot_mlp_persistence(conv, path=_out(f"diag_mlp_{tag}.png"))
        print(f"   saved diag_layer_{tag}.png, diag_mlp_{tag}.png")
    return result, timings, total_s


def summarize(name, result):
    c = result["convergence"]
    h1, h0 = c["per_dim"][1], c["per_dim"][0]
    s = c.get("d_star_stability") or {}
    return dict(name=name, carrier=result["carrier_dim"], d_star=c["d_star"],
                ci=s.get("ci95"), agreement=s.get("agreement"),
                stable=s.get("stable"), inert=c["inert_layers"],
                h1_res=h1["resolved_by"], h0_res=h0["resolved_by"])


# ----------------------------------------------------------------------------
# Depth-sweep keystone: does test accuracy plateau at the topological d*?
# ----------------------------------------------------------------------------

def depth_sweep(Xtr, ytr, Xte, yte, depths, d_star, d_star_ci=None,
                epochs=EPOCHS, path="cifar_depth_sweep.png"):
    print("\n" + "=" * 70)
    print("DEPTH-SWEEP KEYSTONE: test accuracy vs dense depth, against d*")
    print("=" * 70)
    accs = []
    for d in depths:
        _, acc = train_or_load(Xtr, ytr, Xte, yte, dense_depth=d, epochs=epochs)
        accs.append(float(acc))
        print(f"  depth {d:2d}: test acc {acc:.4f}")

    verdict = "d* unavailable"
    if d_star is not None:
        acc_by = dict(zip(depths, accs))
        ref = acc_by.get(d_star, float(np.interp(d_star, depths, accs)))
        beyond = [a for dd, a in zip(depths, accs) if dd > d_star]
        gain = (max(beyond) - ref) if beyond else 0.0
        falsified = gain > PLATEAU_MARGIN
        verdict = (f"acc(d*={d_star})={ref:.3f}  best-beyond={max(beyond):.3f} "
                   f"gain={gain:+.3f}  -> " +
                   ("STILL RISING past d*: diagnosis FALSIFIED" if falsified
                    else "PLATEAU at/before d*: diagnosis SUPPORTED"))
        print("  " + verdict)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(depths, accs, "o-", color="tab:blue", label="test accuracy")
    if d_star is not None:
        ax.axvline(d_star, color="k", ls="--", label=f"topological d*={d_star}")
        if d_star_ci and None not in d_star_ci:
            ax.axvspan(d_star_ci[0], d_star_ci[1], color="k", alpha=0.08,
                       label="d* 95% CI")
    ax.set_xlabel("dense head depth")
    ax.set_ylabel("test accuracy")
    ax.set_title("Depth sweep vs topological d* (keystone)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_out(path), dpi=150)
    plt.close(fig)
    print(f"  saved {path}")
    return dict(depths=list(depths), accs=accs, verdict=verdict)


def main(full=False, sweep=False):
    full = full or sweep                          # the sweep needs a d* -> needs FULL
    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"CIFAR-10: train {Xtr.shape}, test {Xte.shape}")
    model, acc = train_or_load(Xtr, ytr, Xte, yte, dense_depth=DENSE_DEPTH, epochs=EPOCHS)
    print(f"dense_depth={DENSE_DEPTH}  test accuracy {acc:.4f}")

    Xsub, ysub, _ = class_balanced_subsample(Xte, yte, PER_CLASS, seed=SEED)
    names = head_layer_names(DENSE_DEPTH)
    reps = head_representations(model, Xsub, names)
    print(f"subsample: {len(Xsub)} pts; raw dims {[r.shape[1] for r in reps]}")

    lat_raw = pl.preprocess_latents(reps, "raw")
    lat_pca = pl.preprocess_latents(reps, "pca", n_components=PCA_K, normalize=NORMALIZE)

    res_raw, _, t_raw = run_one("RAW", lat_raw, full)
    res_pca, _, t_pca = run_one("PCA", lat_pca, full)

    a, b = summarize("RAW", res_raw), summarize("PCA", res_pca)
    print("\n" + "=" * 70)
    print("RAW vs PCA -- did Stage-0 change the conclusion?")
    print("=" * 70)
    print(f"  carrier dim   RAW=H{a['carrier']}   PCA=H{b['carrier']}")
    print(f"  d* (mode)     RAW={a['d_star']} (CI {a['ci']}, stable={a['stable']})"
          f"   PCA={b['d_star']} (CI {b['ci']}, stable={b['stable']})")
    print(f"  inert tail    RAW={a['inert']}   PCA={b['inert']}")
    print(f"  runtime (s)   RAW={t_raw:.1f}   PCA={t_pca:.1f}")
    changed = (a["carrier"] != b["carrier"]) or (a["d_star"] != b["d_star"])
    print("  => CONCLUSION " + ("CHANGES with PCA (report both)" if changed
                                else "is ROBUST to preprocessing"))

    if sweep:
        # use the more stable preprocessing's d* for the overlay
        chosen = a if (a["stable"] and not b["stable"]) else (b if (b["stable"] and not a["stable"]) else a)
        depth_sweep(Xtr, ytr, Xte, yte, depths=list(range(1, DENSE_DEPTH + 1)),
                    d_star=chosen["d_star"], d_star_ci=chosen["ci"])
        print(f"  (d* overlaid from {chosen['name']} preprocessing)")

    return res_raw, res_pca


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--alpha", type=float, default=None,
                    help=f"scale-axis significance level (dataset default {PARAMS['alpha']})")
    args = ap.parse_args()
    from ltep import output
    alpha = args.alpha if args.alpha is not None else PARAMS["alpha"]
    pl.set_alpha(alpha)
    OUTDIR = output.run_dir("cifar_dense", tag=f"alpha{alpha}")
    output.save_params(OUTDIR, dict(PARAMS, alpha=alpha, full=args.full, sweep=args.sweep))
    with output.capture(OUTDIR):
        main(full=args.full, sweep=args.sweep)
