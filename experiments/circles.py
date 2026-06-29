#!/usr/bin/env python
# coding: utf-8
"""
circles.py
==========

Illustrative toy: two concentric classes (inner disk vs.\ outer ring,
``sklearn.make_circles``) through a tiny MLP with a single $3$-dimensional hidden
layer ($2\\to3\\to1$). Every representation is low-dimensional and can be drawn ---
the 2D input (two circles, each an H1 loop), the 3D hidden layer, and the 1D output
--- so it visualises the whole framework end to end:

  * layer persistence (H0/H1 diagrams per layer),
  * MLP persistence (the input loops are RESOLVED as the net separates the classes),
  * the pullback VR complexes drawn on the 2D input and 3D hidden clouds,
  * the tower-consistent trajectory flow.

Pipeline of record: ltep (same machinery as cardio / COIL / ResNet). Reuses the
generic sigmoid MLP from ltep.datasets.cardio (build_mlp(2,[3]) is exactly this net).

    python experiments/circles.py            # train (or load cache) + all figures
    python experiments/circles.py --retrain  # force retraining
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import datasets

from ltep import pipeline as pl, plots, output, vr
from ltep.datasets.cardio import build_mlp, train_model, get_all_latents

PARAMS = dict(n_points=300, factor=0.5, noise=0.05, epochs=2000, seed=1234, alpha=0.05)
CMAP = "coolwarm"


def make_data(n_points, factor, noise, seed):
    X, y = datasets.make_circles(n_samples=n_points, factor=factor, noise=noise,
                                 random_state=seed)
    return X.astype("float32"), y.astype(int)


# ----------------------------------------------------------------------------
# Geometry figures (the illustrative, visualisable part)
# ----------------------------------------------------------------------------

def plot_input_boundary(model, X, y, outdir):
    x0, x1 = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y0, y1 = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    Z = (model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0).ravel() > 0.5)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.22, cmap=CMAP)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP, edgecolor="k", s=18, linewidth=0.3)
    ax.set_title("Input $\\mathbb{R}^2$ + decision boundary")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout()
    p = os.path.join(outdir, "circles_input_boundary.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", os.path.basename(p))


def plot_latent_geometry(reps, y, outdir):
    Xh, Xo = reps[1], reps[-1]
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Xh[:, 0], Xh[:, 1], Xh[:, 2], c=y, cmap=CMAP, s=18,
               edgecolor="k", linewidth=0.3)
    ax.set_title("Hidden layer $\\mathbb{R}^3$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout()
    p = os.path.join(outdir, "circles_hidden_3d.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", os.path.basename(p))

    fig, ax = plt.subplots(figsize=(4.6, 1.5))
    ax.scatter(Xo.ravel(), np.zeros(len(Xo)), c=y, cmap=CMAP, s=18,
               edgecolor="k", linewidth=0.3)
    ax.axvline(0.5, ls="--", color="0.5", lw=1)
    ax.set_yticks([]); ax.set_xlabel("output logit")
    ax.set_title("Output $\\mathbb{R}$ (separable)")
    fig.tight_layout()
    p = os.path.join(outdir, "circles_output_1d.png")
    fig.savefig(p, dpi=150); plt.close(fig); print("saved", os.path.basename(p))


def make_pipeline_figure(reps, eps_res, conv, y, outdir):
    """Compact pipeline diagram for the MLP-persistence reading on the toy:
      (1) the input layer-persistence diagram with the bootstrap confidence band,
      (2) the significant H1 generators highlighted above the band, then
      (3) a clean MLP-persistence barcode --- the two loops born at the input and
          resolved at the output, the two class-components persisting.
    The barcode is drawn by hand so the H1 bars run to the output layer (where the
    loops die) and stay readable at small print size."""
    import gudhi as gd
    from scipy.spatial.distance import cdist
    from matplotlib.lines import Line2D
    from ltep.metrics import _finite_bars

    P = np.asarray(reps[0], float)
    dm = cdist(P, P)
    st = gd.SimplexTree.create_from_array(dm, max_filtration=2.5)
    st.expansion(2); st.compute_persistence(homology_coeff_field=2)
    d0 = _finite_bars(np.asarray(st.persistence_intervals_in_dimension(0)).reshape(-1, 2))
    d1 = _finite_bars(np.asarray(st.persistence_intervals_in_dimension(1)).reshape(-1, 2))
    tau = float(eps_res["per_layer"][0].get("tau_H1") or 0.0)

    fig, (axd, axb) = plt.subplots(1, 2, figsize=(9.2, 4.1),
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))

    # (1)+(2) layer persistence diagram + confidence band + significant H1
    hi = float(max(d0.max() if d0.size else 1.0, d1.max() if d1.size else 1.0)) * 1.05
    axd.plot([0, hi], [0, hi], color="0.6", lw=1, zorder=0)
    axd.fill_between([0, hi], [0, hi], [tau, hi + tau], color="0.88", zorder=0,
                    label="$\\tau$ confidence band")
    if d0.size:
        axd.scatter(d0[:, 0], d0[:, 1], s=16, c="tab:blue", alpha=0.6, label="$H_0$")
    if d1.size:
        axd.scatter(d1[:, 0], d1[:, 1], s=34, c="tab:red", marker="^", label="$H_1$")

    # (3) MLP-persistence barcode from the ACTUAL intervals: each loop runs from its
    # birth to its death layer (one resolves at the hidden layer, one at the output);
    # the H0 class-components persist.
    L = len(reps)
    h1bars = np.asarray(conv["ref_bars"][1], float).reshape(-1, 2)
    n_loops = len(h1bars)
    n_h0 = int(len(np.unique(y)))            # persisting class-components

    # highlight the n_loops most persistent H1 generators in the diagram (the circles)
    if d1.size and n_loops:
        order = np.argsort(d1[:, 1] - d1[:, 0])[::-1][:n_loops]
        sig = d1[order]
        axd.scatter(sig[:, 0], sig[:, 1], s=170, facecolors="none",
                    edgecolors="red", linewidths=1.6, zorder=3)
        axd.annotate(f"{n_loops} significant $H_1$\nloops (the circles)",
                     xy=(sig[0, 0], sig[0, 1]), xytext=(0.06 * hi, 0.76 * hi),
                     color="red", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="red", lw=1))
    axd.set_xlim(0, hi); axd.set_ylim(0, hi)
    axd.set_xlabel("birth"); axd.set_ylabel("death")
    axd.set_title("(1) layer persistence $+$ confidence set", fontsize=10)
    axd.legend(fontsize=7, loc="lower right")

    yk = 0
    for b, d in h1bars:                       # loops: birth-layer -> death-layer (x = resolved)
        bi, di = int(round(b)), int(round(min(d, L - 1)))
        axb.plot([bi, di], [yk, yk], color="tab:red", lw=7, solid_capstyle="butt")
        axb.scatter([di], [yk], marker="x", c="k", s=48, zorder=3)
        yk += 1
    for _ in range(n_h0):                      # classes: persist (essential)
        axb.plot([0, L - 1], [yk, yk], color="tab:blue", lw=7, solid_capstyle="butt")
        axb.annotate("", xy=(L - 0.55, yk), xytext=(L - 1, yk),
                     arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
        yk += 1
    axb.set_xticks(range(L)); axb.set_xticklabels(["input", "hidden", "output"])
    axb.set_yticks([]); axb.set_xlim(-0.15, L - 0.2); axb.set_ylim(-0.9, yk - 0.4)
    axb.set_title("(3) MLP persistence", fontsize=10)
    axb.legend([Line2D([0], [0], color="tab:red", lw=7, marker="x", markevery=[1],
                       mec="k", mfc="k"),
                Line2D([0], [0], color="tab:blue", lw=7)],
               [f"{n_loops} $H_1$ loops ($\\times$ = resolved)",
                f"{n_h0} $H_0$ classes (persist)"], fontsize=8,
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.suptitle("MLP-persistence pipeline on two circles", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    p = os.path.join(outdir, "circles_pipeline.png")
    fig.savefig(p, dpi=160); plt.close(fig); print("saved", os.path.basename(p))


def plot_pullback_complexes(reps, eps, outdir):
    """Draw the pullback VR complexes on the 2D input and 3D hidden clouds --- the
    geometric picture behind the MLP-persistence tower."""
    X0, Xh, Xo = reps[0], reps[1], reps[-1]
    out = np.c_[np.asarray(Xo, float).reshape(len(Xo), -1),
                (np.asarray(Xo).ravel() > 0.5).astype(float)]
    st_out = vr.compute_vietoris_rips_complex(out, eps[-1], max_dimension=1)
    ms = vr.get_maximal_simplices(st_out, eps[-1])

    k1 = vr.vr_pullback(Xh, eps[1], ms, max_dimension=2)
    vr.visualize_vr_complex_3d(Xh, k1, eps[1], show_labels=False)
    plt.title("Pullback VR complex at the hidden layer")
    plt.savefig(os.path.join(outdir, "circles_vr_hidden_3d.png"), dpi=150)
    plt.close()

    ms1 = vr.get_maximal_simplices(k1, eps[1])
    k0 = vr.vr_pullback(X0, eps[0], ms1, max_dimension=2)
    vr.visualize_vr_complex_2d(X0, k0, eps[0], show_labels=False)
    plt.title("Pullback VR complex at the input")
    plt.savefig(os.path.join(outdir, "circles_vr_input_2d.png"), dpi=150)
    plt.close()
    print("saved circles_vr_hidden_3d.png, circles_vr_input_2d.png")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main(retrain=False, outdir=".", seed=PARAMS["seed"], epochs=PARAMS["epochs"]):
    X, y = make_data(PARAMS["n_points"], PARAMS["factor"], PARAMS["noise"], seed)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"circles_2_3_1_s{seed}.keras")
    if os.path.exists(cache) and not retrain:
        from tensorflow.keras.models import load_model
        model = load_model(cache); print(f"loaded cached {os.path.basename(cache)}")
    else:
        model = build_mlp(2, [3], seed=seed)
        train_model(model, X, y, epochs=epochs)
        model.save(cache)
    acc = float(((model.predict(X, verbose=0).ravel() > 0.5).astype(int) == y).mean())
    print(f"2->3->1 MLP (seed {seed}, {epochs} epochs), train accuracy {acc:.4f}")

    reps = get_all_latents(model, X)             # [X0 (2D), hidden (3D), output (1D)]
    names = ["input", "hidden", "output"]
    print("representations:", [r.shape for r in reps])

    # --- geometry ---
    plot_input_boundary(model, X, y, outdir)
    plot_latent_geometry(reps, y, outdir)

    # --- framework: data-driven scales, layer + MLP persistence ---
    eps_res = pl.select_epsilon(reps, use_bootstrap=True, max_hom_dim=1, rng=seed)
    plots.plot_layer_persistence(reps, eps_res, layer_names=names,
                                 path=os.path.join(outdir, "circles_layer_persistence.png"))

    for scheme, eps_seq in (("H0", eps_res["epsilons_H0"]), ("H1", eps_res["epsilons_H1"])):
        # exclude_output=False for the H1 reading so the loop bars run to the OUTPUT,
        # where they actually die (the 1D output, augmented to two separated clusters,
        # fills them). This only reports the intervals through the last layer; the
        # pipeline is unchanged.
        conv = pl.convergence_depth(reps, eps_seq, significance=False, max_dim=2,
                                    augment_output=True,
                                    exclude_output=(scheme == "H0"))
        h1 = np.asarray(conv["ref_bars"][1], float).reshape(-1, 2)
        red = conv["redundancy"]
        print(f"  [eps@{scheme}] H1 bars (loops): {h1.tolist()}  "
              f"d*={conv['d_star']}  signal=H{pl.signal_dimension(conv)}  "
              f"simplification=H{pl.simplification_dimension(conv)}  "
              f"blocks={red['collapsible_blocks'] or 'none'}")
        plots.plot_mlp_persistence(
            conv, layer_names=names, epsilons=eps_seq,
            title=f"circles MLP persistence (eps@{scheme})",
            path=os.path.join(outdir, f"circles_mlp_barcode_{scheme}.png"))
        if scheme == "H1":
            plots.plot_trajectory_flow(
                reps, eps_res, labels=y, trees=conv["pullback_trees"],
                d_star=conv["d_star"], layer_names=names,
                path=os.path.join(outdir, "circles_trajectory_flow.png"))
            plot_pullback_complexes(reps, eps_seq, outdir)
            make_pipeline_figure(reps, eps_res, conv, y, outdir)

    print("\nDone. Figures in:", outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--seed", type=int, default=PARAMS["seed"],
                    help="weight-init seed; sweep to find a run where the inner loop "
                         "collapses at the hidden layer (the richer illustration).")
    ap.add_argument("--epochs", type=int, default=PARAMS["epochs"],
                    help="training epochs (more can sharpen the inner-loop collapse).")
    args = ap.parse_args()
    pl.set_alpha(args.alpha if args.alpha is not None else PARAMS["alpha"])
    rd = output.run_dir("circles", tag=f"alpha{pl.PARAMS['ALPHA']}_s{args.seed}")
    output.save_params(rd, dict(PARAMS, alpha=pl.PARAMS["ALPHA"], seed=args.seed,
                                epochs=args.epochs))
    with output.capture(rd):
        main(retrain=args.retrain, outdir=rd, seed=args.seed, epochs=args.epochs)
