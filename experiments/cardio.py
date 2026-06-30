#!/usr/bin/env python
# coding: utf-8
"""
check_pipeline_cardio.py
========================

Run the unified pipeline (pipeline.py) on cardiotocography under the NO-RESAMPLING
design (matches the COIL classifier and the ResNet family): significance lives ONCE,
in the per-layer epsilon choice (the scale-axis bootstrap tau band). The layer axis
(MLP-persistence) is then pure tracking -- every cycle/component present at the chosen
epsilon is a real event, drawn and counted; d* and the stable-topology tail are read
directly from the single full-data barcode's births/deaths. No recurrence gate.

Place next to pipeline.py, pipeline_plots.py, Experiment_cardiotocography.py:

    python check_pipeline_cardio.py          # fraction-threshold epsilon, no plots
    python check_pipeline_cardio.py --full   # bootstrap tau band for epsilon + plots

Outputs (--full):
    diag_layer_persistence.png   per-layer diagrams with tau band + chosen epsilon
    diag_mlp_persistence.png     layer-indexed barcode (all events solid) + d*
"""

import os
import sys
import numpy as np

# --- make the repo root importable whether run as a script or via -m ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltep import pipeline as pl, plots
from ltep import runtime as rs
from ltep.datasets.cardio import (
    load_cardio_dataset, build_mlp, train_model, get_all_latents, sparsify,
    MLP_SPARSE_SQDIST,
)

# ============================================================================
# PARAMETERS FOR THIS DATASET (cardiotocography) -- edit here, nowhere else
# ============================================================================
PARAMS = dict(
    hidden_widths=(32, 16, 8, 4),   # the intentionally deep MLP to diagnose
    epochs=1000,
    seed=1234,
    alpha=0.01,                     # scale-axis significance (cardio: low -> fewer noise H0)
    max_hom_dim=0,                  # H0-only by default (cardio is the H0 exemplar)
    analysis_points=300,            # target size of the delta-net topology cloud (was ~99
                                    # at sqdist=0.5); more points -> tighter deep-layer tau
                                    # bands. H0 is near-linear, so a few hundred is cheap.
)


def sparsify_to_target(X, n_target, lo=1e-4, hi=4.0, iters=30):
    """Greedy delta-net (gudhi) sized to ~n_target points by binary-searching the
    min-squared-distance: lower delta keeps MORE points (monotone), so we shrink the
    upper bound when we have too few and raise the lower bound when we have too many.
    Keeps the interleaving/stability guarantee of the delta-net (unlike a random draw).
    If n_target >= len(X), returns the full cloud. Returns (points, indices, sqdist, n)."""
    from ltep.datasets.cardio import sparsify
    X = np.asarray(X, float)
    if n_target >= len(X):
        idx = np.arange(len(X))
        return X, idx, 0.0, len(X)
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        sp, idx = sparsify(X, mid)
        n = len(sp)
        if best is None or abs(n - n_target) < abs(best[3] - n_target):
            best = (sp, idx, mid, n)
        if n < n_target:        # too few points -> need a smaller delta
            hi = mid
        elif n > n_target:      # too many points -> need a larger delta
            lo = mid
        else:
            break
    return best


def print_barcode_readout(conv):
    """Single-barcode readout (no resampling): the H0/H1 events the tower produces,
    d*, and the prunable stable-topology tail. Every bar present at the chosen epsilon
    is a genuine event -- significance was already applied in the epsilon choice."""
    print("\n" + "=" * 70)
    print("MLP-PERSISTENCE (single full-data barcode; significance is in the epsilon)")
    print("=" * 70)
    caps = pl._caps(conv["n_layers"], conv.get("exclude_output", True))
    for k, name in [(1, "H1 loops"), (0, "H0 components")]:
        bars = np.asarray(conv["ref_bars"][k], float).reshape(-1, 2)
        ess = int(np.sum(bars[:, 1] >= caps[k])) if bars.size else 0
        finite = int(np.sum(bars[:, 1] < caps[k])) if bars.size else 0
        print(f"\n  {name}  (cap={caps[k]}): {len(bars)} bars "
              f"({finite} resolved, {ess} essential/preserved)")
        for b, d in bars:
            b, d = int(round(b)), int(round(d))
            mark = "* essential" if d >= caps[k] else ""
            print(f"     {b} -> {d}  {mark}")
    print(f"\n  d* = {conv['d_star']}  of {conv['n_layers']-1} transitions")
    print(f"  prunable stable-topology tail (no births/deaths): {conv['inert_layers']}")
    red = conv.get("redundancy")
    if red is not None:
        print(f"\n  -- collapsible structure (reformulated d*) --")
        print(f"  active transitions (work happens): {red['active_transitions']}")
        print(f"  stages: {red['stages']}")
        print(f"  COLLAPSIBLE BLOCKS (>=2 layers, same topology): "
              f"{red['collapsible_blocks'] or 'none'}")
        print(f"  redundancy R = {red['redundancy']} removable layer(s); "
              f"collapsed depth = {red['collapsed_depth']} stage(s) "
              f"(of {conv['n_layers']} representations)")
    print(f"  signal      H{pl.signal_dimension(conv)} (a feature is PRESENT)")
    print(f"  simplification H{pl.simplification_dimension(conv)} "
          f"(a feature is RESOLVED -> drives convergence)")
    print(f"  carrier dim  H{pl.carrier_dimension(conv)}")


def validate_pruning(d_star, hidden_widths, X_train, y_train, X_test, y_test,
                     full_acc, epochs, seed):
    """Empirically check the prunability claim: retrain a network with the inert
    tail removed (keep the d* hidden layers that do topological work) and compare
    test accuracy to the full network. Also retrains one layer SHORTER as a control
    -- if d* is the right cut, removing the tail is ~free but cutting into the
    working layers should start to hurt."""
    from ltep.datasets.cardio import build_mlp, train_model

    def _train_acc(widths):
        if len(widths) == 0:                      # no hidden layers -> logistic regression
            m = build_mlp(X_train.shape[1], [], seed=seed)
        else:
            m = build_mlp(X_train.shape[1], list(widths), seed=seed)
        train_model(m, X_train, y_train, epochs=epochs)
        pred = (m.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
        return float((pred == np.asarray(y_test)).mean())

    n_full = len(hidden_widths)
    keep_full = tuple(hidden_widths[:d_star])               # prune inert tail -> d* layers
    keep_ctrl = tuple(hidden_widths[:max(d_star - 1, 0)])   # control: one layer shorter

    print("\n" + "=" * 70)
    print("PRUNING VALIDATION (retrain with the inert tail removed)")
    print("=" * 70)
    rows = [("full (no pruning)", tuple(hidden_widths), full_acc, "")]
    rows.append((f"pruned to d*={d_star}", keep_full, _train_acc(keep_full),
                 f"{n_full - d_star} tail layer(s) removed"))
    if d_star >= 1:
        rows.append((f"control d*-1={d_star-1}", keep_ctrl, _train_acc(keep_ctrl),
                     "cuts INTO the working layers"))
    print(f"  {'network':>20} | {'hidden widths':>18} | {'test acc':>8} | note")
    for name, widths, acc, note in rows:
        w = str(widths) if widths else "() linear"
        print(f"  {name:>20} | {w:>18} | {acc:8.4f} | {note}")
    pruned_acc = rows[1][2]
    print(f"\n  verdict: pruning the inert tail changed test accuracy by "
          f"{pruned_acc - full_acc:+.4f}  "
          f"({'kept' if pruned_acc >= full_acc - 0.01 else 'DROPPED'} accuracy).")
    return dict(full_acc=full_acc, d_star=d_star,
                pruned_widths=keep_full, pruned_acc=pruned_acc)


def validate_collapse(conv, hidden_widths, X_train, y_train, X_test, y_test,
                      full_acc, epochs, seed):
    """Empirically check the COLLAPSE claim (reformulated d*): retrain a network with
    every collapsible block reduced to its first layer -- dropping the redundant
    layers wherever they sit (inner OR tail), not just truncating a tail -- and
    compare test accuracy to the full network. A control drops ONE further layer
    (cutting into a stage that does real work) to bound the free reduction."""
    from ltep.datasets.cardio import build_mlp, train_model

    def _train_acc(widths):
        m = build_mlp(X_train.shape[1], list(widths), seed=seed)
        train_model(m, X_train, y_train, epochs=epochs)
        pred = (m.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
        return float((pred == np.asarray(y_test)).mean())

    blocks = conv["redundancy"]["collapsible_blocks"]
    collapsed = pl.collapsed_hidden_widths(hidden_widths, blocks)
    R = conv["redundancy"]["redundancy"]

    print("\n" + "=" * 70)
    print("COLLAPSE VALIDATION (retrain with collapsible blocks reduced to 1 layer)")
    print("=" * 70)
    rows = [("full (no collapse)", tuple(hidden_widths), full_acc,
             f"{len(hidden_widths)} hidden layers")]
    if R > 0:
        rows.append((f"collapsed (R={R})", collapsed, _train_acc(collapsed),
                     f"{len(hidden_widths)-len(collapsed)} redundant layer(s) removed"))
        ctrl = collapsed[:-1] if len(collapsed) >= 1 else collapsed
        if ctrl != collapsed:
            rows.append(("control (1 more)", ctrl, _train_acc(ctrl),
                         "cuts INTO a working stage"))
    else:
        print("  no collapsible block (R=0): every stage is a single layer -> nothing "
              "to collapse for this net.")
    print(f"  {'network':>20} | {'hidden widths':>22} | {'test acc':>8} | note")
    for name, widths, acc, note in rows:
        w = str(widths) if widths else "() linear"
        print(f"  {name:>20} | {w:>22} | {acc:8.4f} | {note}")
    if R > 0:
        collapsed_acc = rows[1][2]
        print(f"\n  verdict: collapsing {R} redundant layer(s) changed test accuracy by "
              f"{collapsed_acc - full_acc:+.4f}  "
              f"({'kept' if collapsed_acc >= full_acc - 0.01 else 'DROPPED'} accuracy).")
    return dict(full_acc=full_acc, redundancy=R, collapsed_widths=collapsed,
                collapsed_acc=(rows[1][2] if R > 0 else full_acc))


# ============================================================================
# Matched-depth baselines (reviewer R1.2: comparison vs pruning/representation
# methods). Remove the SAME number of hidden layers the topological collapse
# removes (R), but choose WHICH layers by a non-topological rule, retrain
# identically, and compare. Isolates the value of the topological LOCALISATION,
# not merely the depth reduction.
# ============================================================================

def _linear_cka(A, B):
    """Linear CKA between two representations on the SAME rows (feature dims may
    differ). 1 = identical up to rotation/scale; small = the layer changed the
    representation a lot. Mirrors the helper in ltep.datasets.cardio."""
    A = np.asarray(A, float) - np.asarray(A, float).mean(0)
    B = np.asarray(B, float) - np.asarray(B, float).mean(0)
    hsic = np.linalg.norm(B.T @ A, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(hsic / den) if den > 0 else 0.0


def _topological_drop(hidden_widths, blocks):
    """0-based hidden-layer indices removed by collapsing each block to its first
    layer (the same rule as pl.collapsed_hidden_widths). Representation i (1-based
    over latents) is hidden layer i-1."""
    drop_reps = set()
    for p, q in blocks:
        drop_reps.update(range(p + 1, q + 1))          # keep the block's first layer
    return sorted(j for j in range(len(hidden_widths)) if (j + 1) in drop_reps)


def validate_baselines(conv, hidden_widths, latents, X_train, y_train, X_test, y_test,
                       full_acc, epochs, seed, n_random=5):
    """Matched-depth ablation against two non-topological layer-removal rules:

      (1) RANDOM-position drop -- remove R random hidden layers, averaged over
          n_random distinct draws. The control for the 'redundancy is often
          interior' claim: if random removal of R layers does as well, the
          topological localisation adds nothing.
      (2) REPRESENTATION-SIMILARITY (CKA) drop -- remove the R hidden layers that
          change the representation least, scored by linear CKA with the preceding
          representation. The non-topological analogue of an 'inert transition'.

    The topological collapse removes the same R layers chosen by the collapsible
    blocks. All nets are retrained with the SAME seed and epochs, so only the layer
    SELECTION differs. Returns a record for table building."""
    from ltep.datasets.cardio import build_mlp, train_model

    _retrain_secs = []      # wall-clock of every baseline retrain (runtime accounting)

    def _train_acc(widths, s=seed):
        with rs.measure() as t:
            m = build_mlp(X_train.shape[1], list(widths), seed=s)
            train_model(m, X_train, y_train, epochs=epochs)
        _retrain_secs.append(t["seconds"])
        pred = (m.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
        return float((pred == np.asarray(y_test)).mean())

    blocks = conv["redundancy"]["collapsible_blocks"]
    R = conv["redundancy"]["redundancy"]
    n_full = len(hidden_widths)

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (matched-depth layer-removal rules)")
    print("=" * 70)
    if R == 0:
        print("  R=0: no layers removed -> no matched-depth baseline to compare.")
        return dict(redundancy=0)

    topo_drop = _topological_drop(hidden_widths, blocks)
    collapsed = pl.collapsed_hidden_widths(hidden_widths, blocks)
    topo_acc = _train_acc(collapsed)

    # (1) random-position drop, averaged over n_random distinct R-subsets
    rng = np.random.default_rng(seed)
    rand_accs, rand_drops, seen, tries = [], [], set(), 0
    while len(rand_accs) < n_random and tries < 50 * n_random:
        tries += 1
        drop = tuple(sorted(int(j) for j in rng.choice(n_full, size=R, replace=False)))
        if drop in seen:
            continue
        seen.add(drop)
        widths_r = tuple(w for j, w in enumerate(hidden_widths) if j not in drop)
        rand_accs.append(_train_acc(widths_r))
        rand_drops.append(drop)

    # (2) CKA-similarity drop: latent i (1..m) is hidden layer i-1; score = CKA(prev,
    # this); highest CKA = least change = dropped first.
    cka = [(_linear_cka(latents[i - 1], latents[i]), i - 1) for i in range(1, n_full + 1)]
    cka_drop = sorted(j for _, j in sorted(cka, key=lambda t: -t[0])[:R])
    widths_cka = tuple(w for j, w in enumerate(hidden_widths) if j not in cka_drop)
    cka_acc = _train_acc(widths_cka)

    rmean, rstd = float(np.mean(rand_accs)), float(np.std(rand_accs))
    print(f"  full network {tuple(hidden_widths)}  acc {full_acc:.4f}; "
          f"remove R={R} hidden layer(s)\n")
    print(f"  {'rule':>26} | {'layers removed':>18} | {'test acc':>20}")
    print(f"  {'topological (collapse)':>26} | {str(topo_drop):>18} | {topo_acc:20.4f}")
    print(f"  {'representation CKA':>26} | {str(cka_drop):>18} | {cka_acc:20.4f}")
    print(f"  {'random (mean+-std)':>26} | {'(varied)':>18} | "
          f"{rmean:.4f} +/- {rstd:.4f}")
    print(f"  {'random (worst..best)':>26} | {'':>18} | "
          f"{min(rand_accs):.4f} .. {max(rand_accs):.4f}")
    print(f"\n  verdict: topological - random(mean) = {topo_acc - rmean:+.4f}; "
          f"topological - CKA = {topo_acc - cka_acc:+.4f}  "
          f"(positive => topological localisation helps)")
    n_retr = len(_retrain_secs)
    retrain_total = float(np.sum(_retrain_secs))
    print(f"  runtime: {n_retr} baseline retrains, "
          f"{retrain_total:.1f}s total ({retrain_total/max(n_retr,1):.1f}s/retrain)")
    return dict(redundancy=R, topo=dict(drop=topo_drop, acc=topo_acc),
                cka=dict(drop=cka_drop, acc=cka_acc),
                random=dict(accs=rand_accs, drops=[list(d) for d in rand_drops],
                            mean=rmean, std=rstd),
                runtime=dict(n_retrains=n_retr, retrain_total_s=retrain_total,
                             retrain_mean_s=retrain_total / max(n_retr, 1)))


def run_baseline_trials(widths0, X, y, X_train, y_train, X_test, y_test, *, epochs,
                        seeds, analysis_points, use_bootstrap=True, n_random=5,
                        max_hom_dim=0, outdir="."):
    """Multi-seed matched-depth baseline comparison with PAIRED per-seed deltas.
    For each seed it runs the FULL analysis pipeline (analyse_net: train, epsilon band,
    MLP persistence -- all timed; the standard layer-persistence, MLP-persistence and
    trajectory plots are saved for the first seed), then removes the same R hidden
    layers by the topological / CKA / random rules and retrains (same seed and epochs).
    Prints a per-seed paired table, a paired-delta summary, and a per-stage RUNTIME
    table; saves baselines.json. The paired design controls for seed-to-seed variance:
    each rule is compared on the SAME trained net."""
    import json

    print("\n" + "#" * 70)
    print(f"# BASELINE TRIALS over {len(seeds)} seeds (full widths={tuple(widths0)})")
    print("#" * 70)
    records, analysis_timings = [], []
    for si, s in enumerate(seeds):
        print(f"\n  -- seed {s} --")
        # full pipeline analysis (timed; plots for the first seed only)
        arec, objs = analyse_net(
            widths0, X, y, X_train, y_train, X_test, y_test, epochs=epochs, seed=s,
            analysis_points=analysis_points, max_hom_dim=max_hom_dim,
            use_bootstrap=use_bootstrap, plot=(si == 0), outdir=outdir,
            tag=f"baseline_seed{s}", return_objs=True)
        analysis_timings.append(arec["timings"])
        rec = validate_baselines(objs["conv"], widths0, objs["latents"],
                                 X_train, y_train, X_test, y_test,
                                 full_acc=objs["full_acc"], epochs=epochs,
                                 seed=s, n_random=n_random)
        rec["seed"] = s
        rec["full_acc"] = objs["full_acc"]
        rec["analysis_timings"] = arec["timings"]
        records.append(rec)

    usable = [r for r in records if r.get("redundancy", 0) > 0]
    print("\n" + "=" * 70)
    print("BASELINE TRIALS -- per-seed paired accuracies (matched removal depth)")
    print("=" * 70)
    print(f"  {'seed':>4} | {'R':>2} | {'full':>7} | {'topo':>7} | {'CKA':>7} | "
          f"{'rand(mean)':>10} | {'topo-rand':>9} | {'topo-CKA':>8}")
    for r in usable:
        topo, cka, rm = r["topo"]["acc"], r["cka"]["acc"], r["random"]["mean"]
        print(f"  {r['seed']:>4} | {r['redundancy']:>2} | {r['full_acc']:7.4f} | "
              f"{topo:7.4f} | {cka:7.4f} | {rm:10.4f} | "
              f"{topo-rm:+9.4f} | {topo-cka:+8.4f}")
    skipped = [r["seed"] for r in records if r.get("redundancy", 0) == 0]
    if skipped:
        print(f"  (seeds with R=0, no removal, excluded: {skipped})")

    summary = dict(n_usable=len(usable))
    if usable:
        d_tr = [r["topo"]["acc"] - r["random"]["mean"] for r in usable]
        d_tc = [r["topo"]["acc"] - r["cka"]["acc"] for r in usable]
        summary.update(
            paired_topo_minus_random_mean=float(np.mean(d_tr)),
            paired_topo_minus_random_std=float(np.std(d_tr)),
            paired_topo_minus_cka_mean=float(np.mean(d_tc)),
            paired_topo_minus_cka_std=float(np.std(d_tc)),
            topo_ge_random=int(sum(d >= 0 for d in d_tr)),
            topo_ge_cka=int(sum(d >= 0 for d in d_tc)))
        print(f"\n  paired mean (topo - random) = {np.mean(d_tr):+.4f} +/- "
              f"{np.std(d_tr):.4f}   (n={len(usable)} seeds)")
        print(f"  paired mean (topo - CKA)    = {np.mean(d_tc):+.4f} +/- "
              f"{np.std(d_tc):.4f}")
        print(f"  topo >= random in {summary['topo_ge_random']}/{len(d_tr)} seeds; "
              f"topo >= CKA in {summary['topo_ge_cka']}/{len(d_tc)} seeds")
        print("  (positive paired deltas => topological LOCALISATION beats a matched-"
              "count removal by random position or CKA similarity)")

    # ---- per-stage RUNTIME table (mean over seeds) ----
    print("\n" + "=" * 70)
    print("BASELINE TRIALS -- runtime per stage (mean over seeds)")
    print("=" * 70)
    print(f"  {'stage':>22} | {'mean s':>8} | {'std s':>7}")
    runtime_summary = {}
    for stage in ("train", "epsilon", "persistence"):
        secs = [t[stage]["seconds"] for t in analysis_timings if stage in t]
        runtime_summary[stage] = float(np.mean(secs)) if secs else 0.0
        print(f"  {('analysis: ' + stage):>22} | {np.mean(secs):8.2f} | {np.std(secs):7.2f}")
    bsecs = [r["runtime"]["retrain_total_s"] for r in usable if "runtime" in r]
    bn = [r["runtime"]["n_retrains"] for r in usable if "runtime" in r]
    if bsecs:
        runtime_summary["baseline_retrains_total"] = float(np.mean(bsecs))
        print(f"  {'baseline retrains':>22} | {np.mean(bsecs):8.2f} | {np.std(bsecs):7.2f}"
              f"   ({int(np.mean(bn))} retrains/seed: topo+CKA+{n_random} random)")
    print(f"  N={analysis_points} analysis points, B={pl.PARAMS['N_BOOT']} bootstrap, "
          f"{len(seeds)} seeds")

    with open(os.path.join(outdir, "baselines.json"), "w") as f:
        json.dump(dict(seeds=list(seeds), start_widths=list(widths0),
                       n_random=n_random, summary=summary,
                       runtime_summary=runtime_summary, records=records),
                  f, indent=2)
    print(f"\n  saved {os.path.join(outdir, 'baselines.json')}")
    return records


# ============================================================================
# Multi-trial + iterative collapse: train -> analyse -> collapse -> retrain,
# repeated until no block remains; run over several seeds for mean+-std tables.
# ============================================================================

def _ms(xs, fmt="{:.4f}"):
    a = np.asarray(xs, float)
    return f"{fmt.format(a.mean())} +/- {fmt.format(a.std())}"


def analyse_net(widths, X, y, X_train, y_train, X_test, y_test, *, epochs, seed,
                analysis_points, max_hom_dim=0, use_bootstrap=True,
                plot=False, outdir=".", tag="", return_objs=False):
    """Train an MLP with the given hidden widths, run the H0 collapse analysis on the
    delta-net cloud, and return one record: test acc, the collapsible blocks, the
    redundancy R, the collapsed widths, and per-stage TIMINGS (train / epsilon band /
    MLP persistence, seconds + peak MB). When plot=True the standard diagnostics are
    saved (layer diagrams, MLP barcode, tower-consistent trajectory flow)."""
    with rs.measure() as t_train:
        model = build_mlp(X_train.shape[1], list(widths), seed=seed)
        train_model(model, X_train, y_train, epochs=epochs)
    pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = float((pred == np.asarray(y_test)).mean())

    X_sparse, sp_idx, _, n_pts = sparsify_to_target(X, analysis_points)
    latents = get_all_latents(model, X_sparse)
    with rs.measure() as t_eps:
        eps_res = pl.select_epsilon(latents, use_bootstrap=use_bootstrap,
                                    max_hom_dim=max_hom_dim, rng=seed)
    with rs.measure() as t_pers:
        conv = pl.convergence_depth(latents, eps_res["epsilons_H0"], significance=False,
                                    max_dim=max_hom_dim + 1)
    blocks = conv["redundancy"]["collapsible_blocks"]

    if plot:
        sfx = f"_{tag}" if tag else ""
        plots.plot_layer_persistence(
            latents, eps_res, path=os.path.join(outdir, f"diag_layer{sfx}.png"))
        plots.plot_mlp_persistence(
            conv, epsilons=eps_res["epsilons_H0"],
            path=os.path.join(outdir, f"diag_mlp_barcode{sfx}.png"),
            title=f"cardio MLP persistence ({tag or 'full'})")
        try:
            lab = np.asarray(y)[np.asarray(sp_idx)]
            lab = lab if len(lab) == len(latents[0]) else None
        except Exception:
            lab = None
        plots.plot_trajectory_flow(
            latents, eps_res, labels=lab, trees=conv["pullback_trees"],
            d_star=conv["d_star"],
            path=os.path.join(outdir, f"diag_trajectory{sfx}.png"))

    timings = {"train": t_train, "epsilon": t_eps, "persistence": t_pers}
    rec = dict(widths=tuple(widths), n_hidden=len(widths), acc=acc,
               R=conv["redundancy"]["redundancy"], blocks=blocks,
               collapsed=pl.collapsed_hidden_widths(widths, blocks),
               n_points=n_pts,
               timings={k: dict(seconds=v["seconds"],
                                mb=(v.get("py_peak_bytes") or 0) / 1e6)
                        for k, v in timings.items()})
    if return_objs:
        # for the baseline path: reuse the (timed, plotted) analysis without recomputing
        return rec, dict(conv=conv, latents=latents, eps_res=eps_res, full_acc=acc)
    return rec


def iterative_collapse(widths0, X, y, X_train, y_train, X_test, y_test, *, epochs, seed,
                       analysis_points, max_hom_dim=0, use_bootstrap=True, max_iter=6,
                       plot=False, outdir="."):
    """One seed's simplification chain: train -> analyse -> collapse redundant blocks
    -> retrain on the smaller net, until R=0 (no block) or max_iter. Each step is a
    fresh retrain of the collapsed architecture (not a fine-tune). When plot=True the
    diagnostics of every iteration of THIS seed are saved (no extra retrains)."""
    chain, widths = [], tuple(widths0)
    for it in range(max_iter):
        r = analyse_net(widths, X, y, X_train, y_train, X_test, y_test, epochs=epochs,
                        seed=seed, analysis_points=analysis_points,
                        max_hom_dim=max_hom_dim, use_bootstrap=use_bootstrap,
                        plot=plot, outdir=outdir, tag=f"seed{seed}_it{it}")
        r["iter"] = it
        chain.append(r)
        print(f"    seed {seed} it{it}: widths={r['widths']} acc={r['acc']:.4f} "
              f"R={r['R']} -> collapse to {r['collapsed']}  "
              f"[{r['timings']['train']['seconds']:.1f}s train, "
              f"{r['timings']['epsilon']['seconds']:.1f}s eps, "
              f"{r['timings']['persistence']['seconds']:.1f}s pers]")
        if r["R"] == 0 or r["collapsed"] == widths:
            break
        widths = r["collapsed"]
    return chain


def run_iterative_trials(widths0, X, y, X_train, y_train, X_test, y_test, *, epochs,
                         seeds, analysis_points, max_hom_dim=0, use_bootstrap=True,
                         max_iter=6, outdir="."):
    """Run the iterative collapse chain over several seeds and print three tables:
    (1) trials at the FULL depth, (2) the iterative chain aggregated by iteration,
    and (3) a per-stage TIME table. Saves the per-iteration diagnostics for the first
    seed plus a depth-and-accuracy-vs-iteration summary plot."""
    import json
    from collections import Counter

    print("\n" + "#" * 70)
    print(f"# ITERATIVE COLLAPSE over {len(seeds)} seeds (start widths={tuple(widths0)})")
    print("#" * 70)
    chains = []
    for si, s in enumerate(seeds):
        print(f"  -- seed {s} --")
        chains.append(iterative_collapse(
            widths0, X, y, X_train, y_train, X_test, y_test, epochs=epochs, seed=s,
            analysis_points=analysis_points, max_hom_dim=max_hom_dim,
            use_bootstrap=use_bootstrap, max_iter=max_iter,
            plot=(si == 0), outdir=outdir))      # diagnostics for the first seed

    # ---- Table 1: trials at the full (starting) depth ----
    full = [c[0] for c in chains]
    print("\n" + "=" * 70)
    print("TABLE 1 -- trials at full depth (per seed)")
    print("=" * 70)
    print(f"  {'seed':>5} | {'test acc':>8} | {'R':>3} | {'blocks':>16} | collapsed widths")
    for s, r in zip(seeds, full):
        print(f"  {s:>5} | {r['acc']:8.4f} | {r['R']:>3} | "
              f"{str(r['blocks']):>16} | {r['collapsed']}")
    modal_block = Counter(tuple(map(tuple, r["blocks"])) for r in full).most_common(1)[0][0]
    modal_coll = Counter(r["collapsed"] for r in full).most_common(1)[0][0]
    print(f"  ---- mean acc {_ms([r['acc'] for r in full])}, "
          f"mean R {_ms([r['R'] for r in full], '{:.1f}')}; "
          f"modal blocks {list(modal_block)}, modal collapsed {modal_coll}")

    # ---- Table 2: iterative chain aggregated by iteration index ----
    max_len = max(len(c) for c in chains)
    print("\n" + "=" * 70)
    print("TABLE 2 -- iterative simplification (aggregated by iteration)")
    print("=" * 70)
    print(f"  {'iter':>4} | {'#seeds':>6} | {'#hidden (modal)':>15} | "
          f"{'test acc (mean+-std)':>22} | {'R (mean)':>9} | modal widths")
    for it in range(max_len):
        recs = [c[it] for c in chains if len(c) > it]
        modal_w = Counter(r["widths"] for r in recs).most_common(1)[0][0]
        modal_nh = Counter(r["n_hidden"] for r in recs).most_common(1)[0][0]
        print(f"  {it:>4} | {len(recs):>6} | {modal_nh:>15} | "
              f"{_ms([r['acc'] for r in recs]):>22} | "
              f"{np.mean([r['R'] for r in recs]):>9.2f} | {modal_w}")
    final = [c[-1] for c in chains]
    print(f"\n  converged hidden depth: {_ms([r['n_hidden'] for r in final], '{:.1f}')} "
          f"(from {len(widths0)} layers); "
          f"accuracy full {_ms([r['acc'] for r in full])} -> final {_ms([r['acc'] for r in final])}")

    # ---- Table 3: per-stage TIME table (mean over all analyse passes) ----
    all_recs = [r for c in chains for r in c]
    print("\n" + "=" * 70)
    print("TABLE 3 -- runtime per stage (mean over all analyse passes)")
    print("=" * 70)
    print(f"  {'stage':>14} | {'mean s':>8} | {'std s':>7} | {'mean MB':>8}")
    stage_means = {}
    for stage in ("train", "epsilon", "persistence"):
        secs = [r["timings"][stage]["seconds"] for r in all_recs]
        mbs = [r["timings"][stage]["mb"] for r in all_recs]
        stage_means[stage] = float(np.mean(secs))
        print(f"  {stage:>14} | {np.mean(secs):8.2f} | {np.std(secs):7.2f} | "
              f"{np.mean(mbs):8.1f}")
    topo_total = stage_means["epsilon"] + stage_means["persistence"]
    print(f"  {'analysis only':>14} | {topo_total:8.2f} | {'':>7} |   "
          f"(epsilon + persistence, the topology cost; excludes training)")
    print(f"  passes: {len(all_recs)} total over {len(seeds)} seeds; "
          f"N={analysis_points} analysis points, B={pl.PARAMS['N_BOOT']} bootstrap")

    # ---- depth & accuracy vs iteration (summary plot) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axd, axa) = plt.subplots(1, 2, figsize=(11, 4.2))
        for c in chains:
            xs = [r["iter"] for r in c]
            axd.plot(xs, [r["n_hidden"] for r in c], "-o", color="0.7", alpha=0.6, ms=4)
            axa.plot(xs, [r["acc"] for r in c], "-o", color="0.7", alpha=0.6, ms=4)
        mx = max(len(c) for c in chains)
        md = [np.mean([c[it]["n_hidden"] for c in chains if len(c) > it]) for it in range(mx)]
        ma = [np.mean([c[it]["acc"] for c in chains if len(c) > it]) for it in range(mx)]
        axd.plot(range(mx), md, "-o", color="tab:blue", lw=2.5, label="mean")
        axa.plot(range(mx), ma, "-o", color="tab:red", lw=2.5, label="mean")
        axd.set_xlabel("iteration"); axd.set_ylabel("# hidden layers")
        axd.set_title("Depth vs iteration (collapse)"); axd.legend(fontsize=8)
        axa.set_xlabel("iteration"); axa.set_ylabel("test accuracy")
        axa.set_title("Accuracy vs iteration"); axa.legend(fontsize=8)
        for ax in (axd, axa):
            ax.set_xticks(range(mx))
        fig.tight_layout()
        p = os.path.join(outdir, "iterative_depth_accuracy.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"\n  saved {p}")
    except Exception as e:
        print(f"  (summary plot skipped: {e})")

    # save raw chains for LaTeX table building
    with open(os.path.join(outdir, "iterative_chains.json"), "w") as f:
        json.dump(dict(seeds=list(seeds), start_widths=list(widths0),
                       chains=[[{k: (list(v) if isinstance(v, tuple) else
                                     [list(b) for b in v] if k == "blocks" else v)
                                 for k, v in r.items()} for r in c]
                               for c in chains]), f, indent=2)
    print(f"\n  saved {os.path.join(outdir, 'iterative_chains.json')}")
    return chains


def main(hidden_widths=(32, 16, 8, 4), epochs=2000, seed=1234, full=False,
         max_hom_dim=0, manual_eps=None, outdir='.', validate=True,
         analysis_points=300, baselines=False, n_random=5,
         biband=False, biband_trange=None):
    # max_hom_dim=0 -> H0-only (cardio is the H0 exemplar). max_hom_dim=1 -> also loops,
    # producing TWO barcodes: one read at the H0 epsilon, one at the H1 epsilon.
    # manual_eps (str/list) -> heuristic OFF: one barcode at your hand-picked epsilons.
    # 1. data + an intentionally deep net (the regime to diagnose)
    X, y, X_train, y_train, X_test, y_test = load_cardio_dataset()
    model = build_mlp(X_train.shape[1], list(hidden_widths), seed=seed)
    train_model(model, X_train, y_train, epochs=epochs)
    pred = (model.predict(X_test, verbose=0).ravel() > 0.5).astype(int)
    acc = float((pred == np.asarray(y_test)).mean())
    print(f"\ntrained {len(hidden_widths)}-hidden-layer MLP, test accuracy {acc:.4f}")

    # 2. sparsify the input once, propagate the SAME points (shared row order).
    # Target a fixed analysis-cloud size via the delta-net (more points -> tighter
    # deep-layer tau bands and a less sub-threshold-driven d*).
    X_sparse, sp_keep, sqd_used, n_pts = sparsify_to_target(X, analysis_points)
    print(f"analysis cloud: {n_pts} points "
          f"(delta-net min_sqdist={sqd_used:.4f}, target {analysis_points}; full {len(X)})")
    latents = get_all_latents(model, X_sparse)           # [X0, h1, ..., output]
    print("representations (rows, dims):", [r.shape for r in latents])

    md = max_hom_dim + 1                                 # VR expansion dim

    # ---- heuristic OFF: one barcode at user-supplied epsilons ----
    if manual_eps is not None:
        eps = pl.parse_manual_epsilons(manual_eps, len(latents))
        print(f"MANUAL epsilon (heuristic off): {[round(e,3) for e in eps]}")
        conv = pl.convergence_depth(latents, eps, significance=False, max_dim=md)
        print("\n--- barcode @ MANUAL epsilon ---")
        print_barcode_readout(conv)
        if full:
            p = plots.plot_mlp_persistence(conv, path=os.path.join(outdir, "diag_mlp_persistence_manual.png"),
                                           epsilons=eps)
            print(f"saved {p}")
        out = dict(epsilons=eps, convergence=conv)
        if validate:                # same collapse-retrain check as the heuristic path
            out["collapse"] = validate_collapse(
                conv, hidden_widths, X_train, y_train, X_test, y_test,
                full_acc=acc, epochs=epochs, seed=seed)
            if baselines:
                out["baselines"] = validate_baselines(
                    conv, hidden_widths, latents, X_train, y_train, X_test, y_test,
                    full_acc=acc, epochs=epochs, seed=seed, n_random=n_random)
        return out

    # ---- heuristic ON: select both epsilon sequences ----
    print(f"homology: up to H{max_hom_dim}")
    eps_res = pl.select_epsilon(latents, use_bootstrap=full, max_hom_dim=max_hom_dim,
                                rng=seed)

    # one barcode per epsilon-choice scheme (each displays H0 AND H1 at its scale)
    schemes = [("H0", eps_res["epsilons_H0"])]
    if max_hom_dim >= 1:
        schemes.append(("H1", eps_res["epsilons_H1"]))

    results = {}
    for tag, eps in schemes:
        print(f"\n--- barcode @ epsilon chosen for {tag} ---")
        conv = pl.convergence_depth(latents, eps, significance=False, max_dim=md)
        print_barcode_readout(conv)
        carrier = pl.carrier_dimension(conv)
        results[tag] = dict(
            params=dict(pl.PARAMS), epsilons=eps, epsilon_scheme=tag,
            epsilon_audit=eps_res["per_layer"], convergence=conv, carrier_dim=carrier,
            cross_check=pl.cross_check_bottleneck(latents, conv["d_star"],
                                                  homology_dim=carrier))
        pl.pretty_print(results[tag])
        if full:
            plots.plot_mlp_persistence(
                conv, path=os.path.join(outdir, f"diag_mlp_persistence_{tag}.png"), epsilons=eps,
                title=f"cardio MLP persistence (epsilon chosen for {tag})")

    # 6. the visual epsilon check -- one layer-diagram grid, both epsilon lines drawn
    if full:
        p1 = plots.plot_layer_persistence(latents, eps_res,
                                          path=os.path.join(outdir, "diag_layer_persistence.png"))
        print(f"\nsaved {p1} and diag_mlp_persistence_<scheme>.png")
        # trajectory flow: per-point H0-community flow across layers, coloured by class
        try:
            tl = np.asarray(y)[np.asarray(sp_keep)]
            traj_labels = tl if len(tl) == len(latents[0]) else None
        except Exception:
            traj_labels = None
        h0_conv = results["H0"]["convergence"] if "H0" in results else None
        plots.plot_trajectory_flow(
            latents, eps_res, labels=traj_labels,
            trees=(h0_conv["pullback_trees"] if h0_conv else None),
            d_star=(h0_conv["d_star"] if h0_conv else None),
            path=os.path.join(outdir, "diag_trajectory_flow.png"))

    # 7. EMPIRICAL collapse check: retrain with collapsible blocks reduced to one
    # layer each (reformulated d*) and compare accuracy to the full net.
    if validate and "H0" in results:
        results["collapse"] = validate_collapse(
            results["H0"]["convergence"], hidden_widths,
            X_train, y_train, X_test, y_test,
            full_acc=acc, epochs=epochs, seed=seed)
        # matched-depth baselines: random-position and CKA-similarity layer removal
        if baselines:
            results["baselines"] = validate_baselines(
                results["H0"]["convergence"], hidden_widths, latents,
                X_train, y_train, X_test, y_test,
                full_acc=acc, epochs=epochs, seed=seed, n_random=n_random)
            import json
            with open(os.path.join(outdir, "baselines_singlerun.json"), "w") as f:
                json.dump(dict(seed=seed, full_acc=acc, n_random=n_random,
                               record=results["baselines"]), f, indent=2)
            print(f"  saved {os.path.join(outdir, 'baselines_singlerun.json')}")

    # 8. BI-PERSISTENCE BAND (single run): sweep a scale multiplier around the chosen H0
    # epsilon and report the beta0 surface, R(t) curve and a scale-stability scalar.
    if biband and "H0" in results:
        import json
        from ltep import bipersistence as bp
        names = [f"L{i}" for i in range(len(latents))]
        band = bp.bipersistence_band(latents, results["H0"]["epsilons"],
                                     t_grid=biband_trange)
        bp.plot_bipersistence_band(band, os.path.join(outdir, "diag_biband.png"),
                                   title="cardio bi-persistence band (H0)",
                                   layer_names=names)
        with open(os.path.join(outdir, "biband.json"), "w") as f:
            json.dump(band, f, indent=2)
        print(f"\n  bi-persistence band: R(t)={band['R']}, R@t1={band['R_at_t1']}, "
              f"stability={band['stability']:.2f}; saved diag_biband.png + biband.json")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="bootstrap tau band for epsilon + save plots")
    ap.add_argument("--with-h1", action="store_true",
                    help="also track loops (default: H0-only, cardio is the H0 exemplar)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="scale-axis significance level (default: pipeline ALPHA)")
    ap.add_argument("--epsilons", type=str, default=None,
                    help="heuristic OFF: comma-separated per-layer epsilon read off the "
                         "layer diagrams, e.g. '0.7,0.5,0.4,0.3,0.2,0.1'")
    ap.add_argument("--no-prune-check", action="store_true",
                    help="skip retraining the pruned network to validate prunability")
    ap.add_argument("--baselines", action="store_true",
                    help="matched-depth baseline comparison on a SINGLE net: remove the "
                         "same R hidden layers by random-position and CKA-similarity "
                         "rules, retrain, and compare against the topological collapse.")
    ap.add_argument("--baseline-trials", action="store_true",
                    help="multi-seed matched-depth baselines with PAIRED per-seed deltas "
                         "(uses --trials seeds, --n-random draws); saves baselines.json.")
    ap.add_argument("--n-random", type=int, default=5,
                    help="number of random-position draws per net for the baselines "
                         "(default 5).")
    ap.add_argument("--biband", action="store_true",
                    help="bi-persistence band (single net): sweep a scale multiplier around "
                         "the chosen H0 epsilon and report the beta0 surface, R(t) curve and "
                         "a scale-stability scalar (diag_biband.png + biband.json).")
    ap.add_argument("--biband-trange", type=str, default="0.7,1.3,7",
                    help="lo,hi,n for the bi-persistence band scale multipliers "
                         "(default 0.7,1.3,7; t=1 = the chosen epsilon).")
    ap.add_argument("--analysis-points", type=int, default=None,
                    help=f"target size of the delta-net topology cloud "
                         f"(default {PARAMS['analysis_points']}; was ~99 at sqdist=0.5). "
                         f"More points = tighter deep-layer tau bands.")
    ap.add_argument("--hidden-widths", type=str, default=None,
                    help="comma-separated hidden-layer widths (default "
                         f"{','.join(map(str, PARAMS['hidden_widths']))}). Use a deeper "
                         "stack with redundant layers, e.g. '64,32,16,8,8,8,8,8', to "
                         "make the inert tail (prunability) visible.")
    ap.add_argument("--epochs", type=int, default=None,
                    help=f"training epochs (default {PARAMS['epochs']}; a deeper net may "
                         "need more to converge before the tail reads inert).")
    ap.add_argument("--iterative", action="store_true",
                    help="run the multi-seed ITERATIVE collapse (train->analyse->collapse"
                         "->retrain until R=0) and print the trials + simplification tables.")
    ap.add_argument("--trials", type=int, default=5,
                    help="number of seeds for --iterative (default 5).")
    ap.add_argument("--max-iter", type=int, default=6,
                    help="max collapse iterations per seed (default 6).")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="override the epsilon-band bootstrap count (e.g. 30 to speed up "
                         "the many retrains in --iterative).")
    args = ap.parse_args()
    from ltep import output
    # alpha: CLI overrides the dataset PARAMS default
    alpha = args.alpha if args.alpha is not None else PARAMS["alpha"]
    n_pts = args.analysis_points if args.analysis_points is not None else PARAMS["analysis_points"]
    widths = (tuple(int(w) for w in args.hidden_widths.replace(" ", "").split(",") if w)
              if args.hidden_widths else PARAMS["hidden_widths"])
    epochs = args.epochs if args.epochs is not None else PARAMS["epochs"]
    pl.set_alpha(alpha)
    if args.n_boot is not None:
        pl.N_BOOT = args.n_boot
        pl.PARAMS["N_BOOT"] = args.n_boot
    mhd = 1 if args.with_h1 else PARAMS["max_hom_dim"]
    arch = "-".join(map(str, widths))
    tag = (f"alpha{alpha}_n{n_pts}_w{arch}" + ("_iter" if args.iterative else "")
           + ("_h1" if mhd >= 1 else "") + ("_manual" if args.epsilons else ""))
    rd = output.run_dir("cardio", tag=tag)
    output.save_params(rd, dict(PARAMS, alpha=alpha, max_hom_dim=mhd, analysis_points=n_pts,
                                hidden_widths=list(widths), epochs=epochs,
                                iterative=args.iterative, trials=args.trials,
                                manual_eps=args.epsilons, full=args.full))
    with output.capture(rd):
        if args.baseline_trials:
            X, y, X_train, y_train, X_test, y_test = load_cardio_dataset()
            run_baseline_trials(
                widths, X, y, X_train, y_train, X_test, y_test, epochs=epochs,
                seeds=list(range(args.trials)), analysis_points=n_pts,
                use_bootstrap=args.full, n_random=args.n_random, outdir=rd)
        elif args.iterative:
            X, y, X_train, y_train, X_test, y_test = load_cardio_dataset()
            run_iterative_trials(
                widths, X, y, X_train, y_train, X_test, y_test, epochs=epochs,
                seeds=list(range(args.trials)), analysis_points=n_pts,
                max_hom_dim=mhd, use_bootstrap=args.full, max_iter=args.max_iter,
                outdir=rd)
        else:
            biband_trange = None
            if args.biband:
                lo, hi, nn = args.biband_trange.replace(" ", "").split(",")
                biband_trange = list(np.round(np.linspace(float(lo), float(hi), int(nn)), 4))
            main(hidden_widths=widths, epochs=epochs,
                 seed=PARAMS["seed"], full=args.full, max_hom_dim=mhd,
                 manual_eps=args.epsilons, outdir=rd, validate=(not args.no_prune_check),
                 analysis_points=n_pts, baselines=args.baselines, n_random=args.n_random,
                 biband=args.biband, biband_trange=biband_trange)
