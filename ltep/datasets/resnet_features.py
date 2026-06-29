#!/usr/bin/env python
# coding: utf-8
"""
diagnostic_pretrained_cifar.py
==============================

OPTION-3 DIAGNOSTIC: does significant, stable H1 exist in the representations of a
*trusted, converged* CIFAR-10 model?  This removes our own training as a confound:
whatever the persistence diagrams show is then a fact about good CIFAR features, not
about our optimiser.

Model: ResNet-20 from chenyaofo/pytorch-cifar-models (smallest CIFAR-10 ResNet,
~0.27M params, ~92.6% top-1). Loaded via torch.hub (weights hosted on github.com).

Layer axis: the sequence of residual-BLOCK outputs, each GLOBAL-AVERAGE-POOLED to a
vector so every representation shares sample-row order (what the pullback tower
needs). Sequence = [stem, block_0_0, block_0_1, ..., pre-logit pooled, logits].

What it answers (per seed, then aggregated):
  * scale-axis significance (tau band) on EACH block -> does any layer have
    significant H1? does H0 settle toward ~N_CLASSES at the end?
  * if --full, the FULL pipeline (MLP-persistence d*, carrier dimension) on the
    block sequence -> is the carrier H1 or H0 in a good model?

Decision this informs:
  * significant, stable H1 appears  -> CIFAR CAN be an H1 experiment; attach a dense
    head to the frozen backbone (option 2) and run the depth story on structured input.
  * no significant H1 even here      -> CIFAR is an H0 exemplar; report it as
    scalability + cluster-merging, let COIL-100 carry H1.

Requirements: torch, torchvision (CPU is fine for ResNet-20 feature extraction).
Network: needs github.com reachable for the torch.hub download (allowed on your machine;
NOT reachable in the sandbox -- run locally).

    python diagnostic_pretrained_cifar.py            # scale-axis significance, 3 seeds
    python diagnostic_pretrained_cifar.py --full      # + full pipeline (d*, carrier)
    python diagnostic_pretrained_cifar.py --seeds 5   # more seeds
"""

import sys
import argparse
import numpy as np

from .. import pipeline as pl


# ----------------------------------------------------------------------------
# Model + feature extraction (PyTorch)
# ----------------------------------------------------------------------------

def load_resnet20():
    import torch
    model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                           "cifar10_resnet20", pretrained=True)
    model.eval()
    return model


def find_basic_blocks(model):
    """
    Discover the residual BasicBlock modules dynamically (robust to exact naming and
    reusable for resnet32/56). A BasicBlock is identified structurally: a module whose
    class name contains 'block' OR that has the conv1/bn1/conv2/bn2 signature.
    Returns an ORDERED list of (qualified_name, module).
    """
    blocks = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        is_block = ("block" in cls) or (
            hasattr(mod, "conv1") and hasattr(mod, "conv2")
            and hasattr(mod, "bn1") and hasattr(mod, "bn2"))
        if is_block:
            blocks.append((name, mod))
    # named_modules() is emission-ordered (depth-first, registration order) = forward
    # order for sequential resnets; keep as-is.
    return blocks


def extract_block_features(model, images):
    """
    Forward `images` once with hooks on every BasicBlock; return an ordered list of
    GLOBAL-AVERAGE-POOLED block outputs, plus a parallel list of layer names. Each
    array is (n_samples, n_channels) -- rows aligned to `images`.
    """
    import torch
    blocks = find_basic_blocks(model)
    feats, names, handles = [], [], []

    def make_hook(nm):
        def hook(_m, _inp, out):
            # out: (N, C, H, W) -> global average pool over H,W -> (N, C)
            t = out.detach()
            if t.dim() == 4:
                t = t.mean(dim=(2, 3))
            feats.append((nm, t.cpu().numpy()))
        return hook

    for nm, mod in blocks:
        handles.append(mod.register_forward_hook(make_hook(nm)))

    with torch.no_grad():
        logits = model(images)

    for h in handles:
        h.remove()

    # assemble in the (registration = forward) order we hooked
    ordered = sorted(range(len(feats)), key=lambda i: i)  # feats already in hook order
    names = [feats[i][0] for i in ordered]
    arrs = [feats[i][1] for i in ordered]
    # append the final logits as the output representation
    names.append("logits")
    arrs.append(logits.detach().cpu().numpy())
    return arrs, names


def cifar_subsample(per_class, seed, n_classes=10, train=False):
    """Class-balanced CIFAR-10 subsample, normalised the way the model expects.
    train=False -> test split (analysis default); train=True -> train split
    (used for the disjoint linear-probe training set)."""
    import torch
    import torchvision
    import torchvision.transforms as T
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = torchvision.datasets.CIFAR10(root="~/.torchvision/datasets/CIFAR10",
                                      train=train, download=True, transform=tf)
    targets = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_classes):
        cls_idx = np.where(targets == c)[0]
        idx.extend(rng.choice(cls_idx, size=per_class, replace=False).tolist())
    rng.shuffle(idx)
    imgs = torch.stack([ds[i][0] for i in idx])
    labels = targets[idx]
    return imgs, labels


# ----------------------------------------------------------------------------
# Diagnostic
# ----------------------------------------------------------------------------

def scale_axis_significance(latents, rng):
    """Per-layer significant-feature counts (scale axis), using the shared selector."""
    res = pl.select_epsilon(latents, use_bootstrap=True, rng=rng)
    return res["per_layer"]


def run_diagnostic(per_class=20, seeds=(0, 1, 2), full=False):
    model = load_resnet20()
    block_names = [nm for nm, _ in find_basic_blocks(model)]
    print(f"discovered {len(block_names)} residual blocks:")
    for nm in block_names:
        print(f"    {nm}")
    print(f"layer axis = [{len(block_names)} blocks] + logits  "
          f"= {len(block_names) + 1} representations\n")

    h1_any_per_seed, carrier_per_seed, dstar_per_seed = [], [], []
    for s in seeds:
        print("=" * 70)
        print(f"SEED {s}")
        print("=" * 70)
        imgs, labels = cifar_subsample(per_class, seed=s)
        reps, names = extract_block_features(model, imgs)
        # PCA-Hiraoka (global) -- block features are up to 64-d; mirror CIFAR runner
        latents = pl.preprocess_latents(reps, "pca", n_components=10, normalize="global")
        print(f"  {len(latents)} reps, dims {[r.shape[1] for r in reps]} (raw) -> "
              f"{[r.shape[1] for r in latents]} (pca)")

        audit = scale_axis_significance(latents, rng=s)
        print(f"  {'layer':<22}| {'epsH0':>7}{'tauH0':>7}{'nH0':>5}  | "
              f"{'epsH1':>7}{'tauH1':>7}{'nH1':>5}")
        h1_significant_layers = 0
        for a, nm in zip(audit, names):
            n_h1 = int(a.get("n_sig_H1", 0))
            if n_h1 > 0:
                h1_significant_layers += 1
            tau_h1 = a.get("tau_H1")
            print(f"  {nm:<22}| {a['eps_H0_used']:>7.3f}{a['tau_H0']:>7.3f}"
                  f"{a['n_sig_H0']:>5}  | {a['eps_H1_used']:>7.3f}"
                  f"{(tau_h1 if tau_h1 is not None else 0.0):>7.3f}{n_h1:>5}")
        h1_any_per_seed.append(h1_significant_layers)
        print(f"  -> layers with significant H1 (scale axis): {h1_significant_layers}")

        if full:
            result = pl.run_pipeline(latents, augment_output=False,
                                     compute_confidence=True, use_bootstrap=True, rng=s)
            c = result["convergence"]
            carrier_per_seed.append(result["carrier_dim"])
            dstar_per_seed.append(c["d_star"])
            st = c.get("d_star_stability") or {}
            print(f"  -> FULL pipeline: carrier=H{result['carrier_dim']}  "
                  f"d*={c['d_star']}  stable={st.get('stable')}  "
                  f"H1 resolved_by={c['per_dim'][1]['resolved_by']}")
        print()

    # ---- verdict ----
    print("#" * 70)
    print("# DIAGNOSTIC VERDICT")
    print("#" * 70)
    print(f"  layers with significant H1 per seed (scale axis): {h1_any_per_seed}")
    any_h1 = any(h > 0 for h in h1_any_per_seed)
    stable_h1 = all(h > 0 for h in h1_any_per_seed)
    if full:
        print(f"  carrier per seed: {['H%d'%c for c in carrier_per_seed]}")
        print(f"  d* per seed      : {dstar_per_seed}")
        carrier_h1 = all(c == 1 for c in carrier_per_seed)
    else:
        carrier_h1 = None

    print()
    if stable_h1 and (carrier_h1 in (True, None)):
        print("  => significant H1 present (and stable across seeds).")
        print("     CIFAR CAN be an H1 experiment. Next: freeze this backbone, attach")
        print("     a deep dense head (option 2), run the depth story on its features.")
    elif any_h1:
        print("  => H1 appears but is NOT stable across seeds. Borderline; treat as")
        print("     noise unless it concentrates in specific blocks consistently.")
    else:
        print("  => NO significant H1 even in a trusted, converged model.")
        print("     CIFAR is an H0 exemplar: report it as scalability + cluster-merging")
        print("     (H0 -> ~N_CLASSES), and let COIL-100 carry the H1 story.")
    print("#" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=20)
    args = ap.parse_args()
    run_diagnostic(per_class=args.per_class, seeds=tuple(range(args.seeds)),
                   full=args.full)
