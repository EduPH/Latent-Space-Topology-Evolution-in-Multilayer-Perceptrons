#!/usr/bin/env python
# coding: utf-8
"""
bipersistence.py
================

Bi-persistence band: restore the SCALE axis that the single-epsilon pipeline collapses.

The pipeline fixes one epsilon per layer from the bootstrap confidence set. That is a
single horizontal slice of the bi-persistence grid (Theorem bi_persistence): layer index
x scale. Here we sweep a common scale multiplier t over a grid around the chosen
epsilons (so the confidence-set choice sits at t=1) and read the layer-wise H0
decomposition at each t via the faithful composed tower (every layer's VR still
constrains the edges, per-layer eps_j*t). This probes how the redundancy reading depends
on scale -- turning "R is epsilon-noisy" into a measured stability statement.

2-parameter persistence has no complete discrete invariant, so we do NOT claim a full
2-parameter barcode. Instead we report three compact, faithful summaries:

  1. Betti surface  beta_0(layer, t)            -- one heatmap (the whole band).
  2. R(t)           redundancy vs scale          -- a curve; its PLATEAU is the trustworthy
                                                    reading.
  3. stability      fraction of the t-band over which the active-transition set equals the
                    one at t=1                    -- a single scale-robustness scalar.

H0 only (components): cardio and the ResNet family. The reading is union-find on the
AND'd edge graph; no simplex trees are built.
"""

import numpy as np


def composed_h0_reading(latents, epsilons, read_idx=None):
    """Faithful edge-level H0 MLP-persistence with per-layer eps_j and layer composition.
    An edge {r,s} survives the downward intersection at layer i iff dist_j <= eps_j at
    EVERY layer j>=i (all layers, kept or composed-over); its birth layer is
    1 + (last j with dist_j > eps_j). Reads beta0 only at read_idx (default all). Returns
    beta0 per read layer, active transitions, redundancy R and collapsed depth (read
    granularity). No simplex trees; H0 is union-find."""
    L = len(latents)
    N = len(latents[0])
    if read_idx is None:
        read_idx = list(range(L))
    read_idx = sorted(set(int(i) for i in read_idx))

    last_violation = -np.ones((N, N), dtype=int)
    for j in range(L):
        Xj = np.asarray(latents[j], float)
        d = np.sqrt(np.maximum(((Xj[:, None, :] - Xj[None, :, :]) ** 2).sum(-1), 0.0))
        last_violation = np.where(d > epsilons[j], j, last_violation)
    birth_full = last_violation + 1

    read_arr = np.asarray(read_idx)
    pos = np.searchsorted(read_arr, birth_full)
    appears = pos < len(read_arr)
    birth_read = np.where(appears, read_arr[np.clip(pos, 0, len(read_arr) - 1)], L + 1)

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
            ra, sb = find(int(r)), find(int(s))
            if ra != sb:
                parent[ra] = sb; comps -= 1
        beta0.append(comps)

    stages = 1
    for a in range(1, len(beta0)):
        if beta0[a] != beta0[a - 1]:
            stages += 1
    active = tuple(read_idx[a] for a in range(1, len(beta0)) if beta0[a] != beta0[a - 1])
    n_read = len(read_idx)
    return dict(read_idx=read_idx, beta0=beta0, active_transitions=active,
                collapsed_depth=stages, redundancy=n_read - stages, n_read_layers=n_read)


def bipersistence_band(latents, eps_chosen, t_grid=None, read_idx=None):
    """Sweep a common scale multiplier t over t_grid; at each t the per-layer scales are
    t*eps_chosen[j], so the confidence-set choice is t=1. Reads the composed H0
    decomposition at each t. Returns the bi-persistence band summaries:
      t_grid, beta0_surface (len(t) x len(read_idx)), R (per t), collapsed (per t),
      active (tuple per t), and stability = fraction of t with active(t)==active(t=1)."""
    if t_grid is None:
        t_grid = np.round(np.linspace(0.7, 1.3, 7), 4).tolist()
    t_grid = [float(t) for t in t_grid]
    eps_chosen = np.asarray(eps_chosen, float)
    if read_idx is None:
        read_idx = list(range(len(latents)))

    surface, Rs, colls, actives = [], [], [], []
    for t in t_grid:
        rd = composed_h0_reading(latents, (t * eps_chosen).tolist(), read_idx=read_idx)
        surface.append(rd["beta0"]); Rs.append(rd["redundancy"])
        colls.append(rd["collapsed_depth"]); actives.append(rd["active_transitions"])

    # reference at t closest to 1
    t1 = int(np.argmin([abs(t - 1.0) for t in t_grid]))
    ref = actives[t1]
    stability = float(np.mean([a == ref for a in actives]))
    return dict(t_grid=t_grid, read_idx=list(read_idx),
                beta0_surface=[list(map(int, row)) for row in surface],
                R=[int(r) for r in Rs], collapsed=[int(c) for c in colls],
                active_transitions=[list(a) for a in actives],
                R_at_t1=int(Rs[t1]), stability=stability,
                R_min=int(min(Rs)), R_max=int(max(Rs)))


def plot_bipersistence_band(band, path, title="", layer_names=None):
    """Two-panel figure: (left) beta0 surface heatmap over (read layer, t) with t=1
    marked; (right) R(t) curve with t=1 and the stability scalar annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_grid = band["t_grid"]; read_idx = band["read_idx"]
    surf = np.asarray(band["beta0_surface"], float)            # (n_t, n_read)
    fig, (axS, axR) = plt.subplots(1, 2, figsize=(12, 4.4),
                                   gridspec_kw=dict(width_ratios=[1.6, 1]))
    im = axS.imshow(surf, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, len(read_idx), min(t_grid), max(t_grid)])
    fig.colorbar(im, ax=axS, label=r"$\beta_0$")
    axS.axhline(1.0, color="w", ls="--", lw=1.2)
    axS.set_xlabel("read layer index"); axS.set_ylabel("scale multiplier $t$ (t=1 = chosen $\\varepsilon$)")
    axS.set_title("Bi-persistence $\\beta_0$ surface")
    if layer_names is not None and len(layer_names) == len(read_idx):
        axS.set_xticks(np.arange(len(read_idx)) + 0.5)
        axS.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=6)
    axR.plot(t_grid, band["R"], "o-", color="tab:red")
    axR.axvline(1.0, color="k", ls="--", lw=1.2, label="chosen $\\varepsilon$ (t=1)")
    axR.set_xlabel("scale multiplier $t$"); axR.set_ylabel("redundancy $R$")
    axR.set_title(f"R(t)  (stability {band['stability']:.2f}; "
                  f"R={band['R_at_t1']} at t=1, range [{band['R_min']},{band['R_max']}])")
    axR.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
