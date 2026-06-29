# ltep — Latent-space Topology Evolution Pipeline

A topological-data-analysis pipeline for tracking how the topology of a neural
network's latent space evolves across depth. It builds a **simplicial tower** over
the per-layer representations, reads **layer persistence** (topological features
within a layer, across scales) and **MLP persistence** (how those features
transform through the network), and reports a **convergence depth `d*`** that
identifies a prunable, stable-topology tail of layers.

This is the analysis code for the paper *Latent Space Topology Evolution in
Multilayer Perceptrons* ([arXiv:2506.01569](https://arxiv.org/abs/2506.01569)).
See [Citation](#citation).

---

## Key ideas

* **Per-layer significance, once.** For each representation, a scale (`epsilon`) is
  chosen from a bootstrap confidence band (Fasy et al.): features whose persistence
  clears the noise floor `tau` are kept. Significance lives in this scale choice — the
  layer axis is then pure tracking (no resampling).
* **Two scales, never combined.** H0 (connected components) and H1 (loops) live at
  different scales, so `select_epsilon` returns **two** per-layer sequences,
  `epsilons_H0` and `epsilons_H1`, and builds **two** barcodes. They are never
  merged with `max(...)`, because that can only push the scale up — the one direction
  a loop dies.
* **`d*` and the prunable tail.** Reading the single full-data barcode, `d*` is the
  last layer transition that still carries a topological event (a birth/death); the
  trailing layers with no events are inert and can be pruned.

---

## Installation

Requires Python >= 3.9.

```bash
git clone <your-repo-url>
cd package                      # the directory containing pyproject.toml
pip install -e .                # installs the `ltep` package (editable)
```

The core install pulls `numpy`, `scipy`, `gudhi`, `scikit-learn`, `matplotlib`,
`networkx`. The heavy, backend-specific dependencies used only by some dataset
loaders are optional extras:

```bash
pip install -e ".[tf]"          # TensorFlow  -- cardio / COIL MLPs
pip install -e ".[torch]"       # PyTorch     -- pretrained ResNet features
pip install -e ".[tf,torch]"    # everything
```

### Layout

```
package/
|-- pyproject.toml
|-- README.md
|-- ltep/                        # SHARED CORE -- edit common logic in one place
|   |-- pipeline.py              #   epsilon selection (H0+H1), convergence_depth, pretty_print
|   |-- metrics.py               #   low-level TDA: persistence, Betti curves, tau band
|   |-- plots.py                 #   layer-persistence diagrams + MLP barcodes
|   |-- vr.py                    #   Vietoris-Rips / pullback complex builders
|   |-- runtime.py               #   bottleneck / sparsify / timing
|   |-- output.py                #   per-run output folders (log + params + figures)
|   `-- datasets/                #   PER-DATASET LOADERS (data + model building)
|       |-- cardio.py            #     load_cardio_dataset, build_mlp, ...
|       |-- coil100.py           #     load_coil100, diameter_normalize, ...
|       |-- cifar_dense.py
|       |-- resnet_features.py
|       `-- _mlp_persistence.py
`-- experiments/                 # PER-DATASET RUNNERS -- loader + PARAMS + run()
    |-- cardio.py                #   python -m experiments.cardio
    |-- coil100.py               #   python -m experiments.coil100
    |-- resnet.py                #   python -m experiments.resnet
    `-- cifar_dense.py           #   python -m experiments.cifar_dense
```

A file in `ltep/datasets/` is a **loader** (`load_...`, `build_...`); the same-named
file in `experiments/` is the **runner** (`main`, command-line flags). They are
different files.

---

## Usage

Run an experiment as a module from the repo root. Every run creates its own output
folder automatically (see [Outputs](#outputs)).

```bash
# cardiotocography -- H0 pruning exemplar
python -m experiments.cardio  --full --alpha 0.01

# COIL-100 autoencoder -- the rotation loop is preserved (H1 carrier)
python -m experiments.coil100 --tasks ae  --group-size 1 --alpha 0.05

# COIL-100 classifier -- within-class loops resolve early (H0 carrier)
python -m experiments.coil100 --tasks clf --group-size 2 --subsample 0.8

# pretrained ResNet family on CIFAR -- H0 carrier across depths
python -m experiments.resnet  --all --seeds 3 --alpha 0.01

# CIFAR dense MLP -- second H0 exemplar
python -m experiments.cifar_dense --full
```

The runners self-bootstrap the repo root onto `sys.path`, so
`python experiments/cardio.py --full` works too.

### Using the library directly

```python
from ltep import pipeline as pl, plots, output

pl.set_alpha(0.01)                       # this run's significance level

# latents: list of (n_points, dim) arrays, one per representation (input -> output)
eps = pl.select_epsilon(latents)         # -> {"epsilons_H0": [...], "epsilons_H1": [...], "per_layer": [...]}

# one barcode per scale (no max-combination)
conv_h0 = pl.convergence_depth(latents, eps["epsilons_H0"], significance=False)
conv_h1 = pl.convergence_depth(latents, eps["epsilons_H1"], significance=False)

print("d* (H0):", conv_h0["d_star"], " inert tail:", conv_h0["inert_layers"])

# manual epsilon override (heuristic off): scales read off the layer diagrams
eps_manual = pl.parse_manual_epsilons("0.74,0.49,1.07,0.84,0.64,0.19", n_layers=len(latents))
conv_manual = pl.convergence_depth(latents, eps_manual, significance=False)
```

---

## Hyperparameters

### Shared defaults (`ltep/pipeline.py`)

Change these once and every experiment inherits them. They govern the
significance machinery, not any single dataset.

| Name | Default | Meaning |
|---|---|---|
| `ALPHA` | `0.01` | Significance level for the scale-axis bootstrap band. The noise floor is the `(1-ALPHA)` quantile of bootstrap bottleneck distances. **Lower -> stricter** (higher `tau`, fewer significant features). A pre-committed choice, reported per run -- not swept for nicer numbers. Override per run with `--alpha`. |
| `N_BOOT` | `100` | Bootstrap replicates for the `tau` band. Fewer (e.g. 30) is faster and noisier. |
| `USE_BOOTSTRAP` | `True` | Use the principled bootstrap band. `False` falls back to a fraction-threshold proxy (fast wiring check only). |
| `MAX_DIMENSION` | `2` | Vietoris-Rips expansion dimension (enough for H0 and H1). |
| `CONV_N_RESAMPLE` | `50` | Row-resamples for the optional layer-axis significance / `d*` stability check (the default pipeline path uses `significance=False`). |
| `CONV_SUBSAMPLE_FRAC` | `0.8` | m-out-of-n subsample fraction for that stability check. |
| `AGREEMENT_MIN` | `0.8` | Stability gate: only assert pruning if the `d*` agreement across resamples >= this. |
| `TAU_FLOOR_FRAC` | `0.01` | If `tau` is below this fraction of the cloud diameter, the layer is flagged **degenerate** (near-collapsed; significance test unreliable). |

### Per-dataset parameters (`experiments/<dataset>.py`, the `PARAMS` block)

Each runner opens with a `PARAMS` dict -- that dataset's choices, the only place to
edit them. Anything also exposed as a CLI flag (e.g. `--alpha`) overrides the
`PARAMS` default for a single run.

**cardio** (`experiments/cardio.py`)

| Param | Default | Meaning |
|---|---|---|
| `hidden_widths` | `(32, 16, 8, 4)` | Intentionally deep MLP -- the regime to diagnose. |
| `epochs` | `1000` | Training epochs. |
| `seed` | `1234` | Reproducibility seed. |
| `alpha` | `0.01` | Low -> fewer noise H0 components (cardio is the H0 exemplar). |
| `max_hom_dim` | `0` | H0-only by default; `--with-h1` adds the (noise-diagnostic) loop barcode. |

**COIL-100** (`experiments/coil100.py`)

| Param | Default | Meaning |
|---|---|---|
| `tasks` | `["ae", "clf"]` | `ae`: loop preserved; `clf`: loop resolved early; `reg`: circle linearised. |
| `n_objects` | `10` | Number of COIL objects. |
| `epochs` | `150` | Training epochs. |
| `enc_depth` / `dense_depth` | `3` / `4` | Encoder depth (AE) / dense depth (classifier). |
| `bottleneck` | `32` | AE bottleneck width (room for the per-object cycles). |
| `ae_width` | `256` | AE encoder/decoder hidden width. |
| `group_size` | `1` | `1` = per object (AE); `>1` pools classes (the classifier's between-class structure needs this). |
| `subsample` | `1.0` | Fraction of points kept per object/class for the heavy figures. |
| `alpha` | `0.01` | COIL loops are weak -- **raise to `0.05`** to capture more of them. |

**ResNet family** (`experiments/resnet.py`)

| Param | Default | Meaning |
|---|---|---|
| `depths_default` | `[20, 56]` | Models run by default. |
| `depths_all` | `[20, 32, 44, 56]` | Full family (`--all`). |
| `seeds` | `1` | Number of seeds (`--seeds N`); seeds act as data-subsample robustness. |
| `prep` | `"pca"` | Hiraoka-global PCA features; `--raw` uses block features instead. |
| `alpha` | `0.01` | ResNet carrier is H0 (H1 is noise). |

**CIFAR dense** (`experiments/cifar_dense.py`)

| Param | Default | Meaning |
|---|---|---|
| `alpha` | `0.01` | Second H0 pruning exemplar (alongside cardio). |

---

## Outputs

Every run writes all of its artefacts into one folder, created automatically:

```
results/<dataset>/<run-tag>_<timestamp>/
    log.txt        full stdout+stderr of the run (mirrored -- console still shows it)
    params.json    the exact parameters the run used (including the active ALPHA)
    *.png          every figure (layer diagrams, both barcodes, ...)
```

The `<run-tag>` encodes the distinguishing parameters (e.g. cardio `alpha0.01`,
COIL `gs2_ss0.80_bn32_w256_ep150`), and a timestamp is appended so repeated runs
never overwrite each other -- no manual `tee` needed. Change the root with the
`LTEP_RESULTS` environment variable (default `./results`).

> Note: the log captures Python-level output. A few low-level TensorFlow/absl
> banners are emitted below Python's `sys.stdout` and may appear only on the
> console; the pipeline's own reports (audit tables, barcodes, `d*`) are all captured.

---

## Reading the output

* **epsilon audit** -- per layer: `eps_H0`, `eps_H1` (or `-` when no loop is
  significant there), `nH0`, `nH1`, and flags (`H0fallback`, `capped`, `degenerate`).
* **MLP-persistence barcode** -- one per scale; H0 bars (components) and H1 bars
  (loops) drawn at that scale, with the chosen `epsilon` printed under each layer.
* **`d*` and inert/prunable tail** -- the convergence depth and the trailing layers
  that carry no topological event. `d*` is robust to the scale scheme even when the
  feature counts differ -- that stability is itself a reported result.

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

Paluzo-Hidalgo, E. (2025). *Latent Space Topology Evolution in Multilayer
Perceptrons.* arXiv:2506.01569.
