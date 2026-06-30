# ltep — Latent-space Topology Evolution Pipeline

A topological-data-analysis pipeline that tracks how the topology of a neural network's
latent space evolves across depth. It builds a **simplicial tower** over the per-layer
representations and reads two complementary views:

* **layer persistence** — topological features *within* a layer, across scales;
* **MLP persistence** — how those features *transform through the network*.

From the MLP-persistence tower it reads **collapsible blocks** (maximal runs of layers
joined only by inert transitions), the **redundancy `R`** (number of removable layers)
and the **collapsed depth** (number of topological stages). Significance enters once, on
the scale axis, via a bootstrap confidence band (Fasy et al.); the layer axis is then
pure tracking.

This is the analysis code for the paper *Latent Space Topology Evolution in Multilayer
Perceptrons* ([arXiv:2506.01569](https://arxiv.org/abs/2506.01569)). See
[Citation](#citation).

---

## Installation

Requires Python >= 3.9.

```bash
git clone <your-repo-url>
cd package                       # the directory containing pyproject.toml
pip install -e .                 # installs the `ltep` package (editable)
```

The core install pulls `numpy`, `scipy`, `gudhi`, `scikit-learn`, `matplotlib`,
`networkx`. Backend-specific dependencies used only by some dataset loaders are optional:

```bash
pip install -e ".[tf]"           # TensorFlow          -- cardio / COIL MLPs
pip install -e ".[torch]"        # PyTorch/torchvision -- pretrained ResNet features
pip install -e ".[tf,torch]"     # everything
```

---

## Repository tree

```
package/
├── pyproject.toml
├── README.md
├── PIPELINE.md                     # formal, definition-level reference for the method
├── ltep/                           # SHARED CORE — common logic in one place
│   ├── pipeline.py                 #   select_epsilon (H0+H1 bands), convergence_depth,
│   │                               #   collapsible_blocks / redundancy R, mlp_persistence
│   ├── bipersistence.py            #   composed (functoriality) H0 tower + bi-persistence band
│   ├── metrics.py                  #   low-level TDA: persistence, Betti curves, tau band
│   ├── plots.py                    #   layer-persistence diagrams, MLP barcodes, trajectories
│   ├── vr.py                       #   Vietoris–Rips / pullback complex builders
│   ├── runtime.py                  #   bottleneck / sparsify / timing helpers
│   ├── output.py                   #   per-run output folders (log + params + figures)
│   └── datasets/                   #   per-dataset LOADERS (data + model building)
│       ├── cardio.py               #     load_cardio_dataset, build_mlp, get_all_latents …
│       ├── coil100.py              #     load_coil100, diameter_normalize, longest_h1_bar …
│       ├── resnet_features.py      #     find_basic_blocks, extract_block_features, cifar_subsample
│       ├── resnet.py               #     ResNet family RUNNER (see note below)
│       ├── cifar_dense.py
│       └── _mlp_persistence.py
└── experiments/                    # per-dataset RUNNERS — loader + PARAMS + CLI
    ├── cardio.py                   #   python -m experiments.cardio
    ├── coil100.py                  #   python -m experiments.coil100
    ├── circles.py                  #   python -m experiments.circles   (running toy example)
    └── cifar_dense.py              #   python -m experiments.cifar_dense
```

> **Note on the ResNet runner.** For historical reasons the ResNet family runner lives at
> `ltep/datasets/resnet.py` and is invoked as `python -m ltep.datasets.resnet` (not
> `experiments/resnet.py`, which is an older version). The commands below use the correct
> path.

---

## Use the package directly on any dataset

The method needs **only** a list of per-layer latent representations with a **shared row
order** — row `r` is the image of the same input `x_r` at every layer. Anything that
produces such a list (any framework, any architecture) works.

```python
import numpy as np
from ltep import pipeline as pl
from ltep import bipersistence as bp

# 1. Per-layer representations of YOUR data through YOUR network, input -> output.
#    Each is an (N, d_i) array; row r is the same sample x_r at every layer.
latents = [X0, H1, H2, ..., out]            # list of numpy arrays

# 2. One scale per layer from the bootstrap confidence band (significance, once).
pl.set_alpha(0.05)                          # this run's level (pre-commit it)
eps = pl.select_epsilon(latents, max_hom_dim=0)["epsilons_H0"]   # H0 (components)
#   use max_hom_dim=1 and ["epsilons_H1"] to also track loops

# 3. Read the MLP-persistence tower: collapsible blocks, redundancy R, collapsed depth.
conv = pl.convergence_depth(latents, eps, significance=False)
red  = conv["redundancy"]
print("R =", red["redundancy"], " collapsed depth =", red["collapsed_depth"])
print("collapsible blocks:", red["collapsible_blocks"])
print("active transitions:", red["active_transitions"])

# 4. (Optional) scale robustness — the bi-persistence band around the chosen epsilon.
#    Sweeps a multiplier t (t=1 == chosen eps); returns the beta0 surface, R(t) and a
#    scale-stability scalar. A flat R(t) near t=1 means the reading is scale-robust.
band = bp.bipersistence_band(latents, eps)
print("R(t) =", band["R"], " stability =", band["stability"])
bp.plot_bipersistence_band(band, "biband.png")

# 5. (Optional, deep H0 nets) faithful layer COMPOSITION (functoriality) for a cheaper /
#    coarser reading. Reads at a strided subset while every intermediate layer still
#    constrains the edges (per-layer eps_j); H0 via union-find, no simplex trees.
read = bp.composed_h0_reading(latents, eps, read_idx=list(range(0, len(latents), 2)))
print("composed R =", read["redundancy"], " beta0 =", read["beta0"])

# Figures
from ltep import plots
plots.plot_mlp_persistence(conv, epsilons=eps)
```

Helper loaders/extractors are available if you want to reuse ours, e.g.
`ltep.datasets.cardio.get_all_latents(model, X)` (Keras) or
`ltep.datasets.resnet_features.extract_block_features(model, imgs)` (pretrained ResNet).

---

## Reproduce our experiments

Run from the repo root (`package/`). Every run creates its own timestamped output folder
under `results/<dataset>/` (`log.txt`, `params.json`, figures, JSON tables).

### Cardiotocography — deep tabular MLP (H0; actionable compression)

```bash
# Iterated train–collapse–retrain over 5 seeds (8 -> ~5 hidden layers, accuracy retained)
python -m experiments.cardio --full --iterative --trials 5 \
    --hidden-widths 64,32,16,8,8,8,8,8 --epochs 2000 --alpha 0.05

# Matched-depth baselines (paired): topological vs random-position vs CKA layer removal
python -m experiments.cardio --full --baseline-trials --trials 5 --n-random 5 \
    --hidden-widths 64,32,16,8,8,8,8,8 --epochs 2000 --alpha 0.05

# Bi-persistence band (single net): beta0 surface + R(t) + scale-stability
python -m experiments.cardio --full --biband \
    --hidden-widths 64,32,16,8,8,8,8,8 --epochs 2000 --alpha 0.05
```

### COIL-100 — dense autoencoder (H1-preserving feature; encoder collapse)

```bash
# Encoder-depth collapse (5 -> 1) + non-topological depth-selection baselines
# (reconstruction-only and CKA), over 5 seeds
python -m experiments.coil100 --collapse-ae --baselines --select-by-loop \
    --objects 5 --enc-depth 5 --bottleneck 32 --ae-width 256 \
    --angles 24 --epochs 200 --seeds 5
```

### ResNet family — pretrained CIFAR-10 (H0; scaling + redundancy diagnostic)

```bash
# Family analysis (H0-edge band makes the bootstrap fast); reports R / collapsed depth
python -m ltep.datasets.resnet --all --seeds 5

# With functoriality layer-composition (stride 2) and a larger subsample (N = 500)
python -m ltep.datasets.resnet --all --seeds 5 --layer-stride 2 --per-class 50

# Subsample-robustness sweep (R / collapsed depth vs N), composed tower
python -m ltep.datasets.resnet --subsample-sweep --sweep-depth 56 \
    --seeds 5 --sweep-per-class 20,35,50 --layer-stride 2

# Bi-persistence band on ResNet-56 (single subsample)
python -m ltep.datasets.resnet --biband --sweep-depth 56 --per-class 50 --layer-stride 2
```

### Running toy example (every pipeline step, fully visualisable)

```bash
python -m experiments.circles
```

Useful shared flags: `--alpha` (scale-axis significance, pre-committed per dataset),
`--n-boot` (bootstrap replicates; lower = faster, noisier), `--seeds` / `--trials` (input
subsamples). Each runner opens with a `PARAMS` block documenting its dataset-specific
defaults.

---

## Outputs

```
results/<dataset>/<run-tag>_<timestamp>/
    log.txt        full stdout/stderr of the run (mirrored to console)
    params.json    the exact parameters used (including the active alpha, layer_stride, …)
    *.png          figures (layer diagrams, MLP barcodes, trajectories, bi-persistence band)
    *.json         tables (iterative_chains.json, baselines.json, *_subsample_sweep.json, biband.json)
```

The `<run-tag>` encodes the distinguishing parameters; a timestamp is appended so repeated
runs never overwrite each other. Change the root with the `LTEP_RESULTS` environment
variable (default `./results`).

---

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{paluzohidalgo2025latentspacetopologyevolution,
  title         = {Latent Space Topology Evolution in Multilayer Perceptrons},
  author        = {Eduardo Paluzo-Hidalgo},
  year          = {2025},
  eprint        = {2506.01569},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2506.01569},
}
```

Paluzo-Hidalgo, E. (2025). *Latent Space Topology Evolution in Multilayer Perceptrons.*
arXiv:2506.01569.
