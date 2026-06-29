"""
ltep -- Latent-space Topology Evolution Pipeline
================================================

Shared, dataset-agnostic core for the topological analysis of MLP/CNN latent
spaces (simplicial towers, MLP-persistence, convergence depth d*). All common
auxiliary functions live here so they can be edited in ONE place; each dataset's
loader, parameter choices, and run logic live in `experiments/<dataset>.py`.

Submodules
----------
ltep.metrics   low-level TDA: persistence diagrams, Betti curves, bootstrap
               significance (tau band), per-layer epsilon plateau selection.
ltep.pipeline  the pipeline: select_epsilon (two scales: H0 and H1, no max),
               convergence_depth (single barcode, no resampling), the
               signal / simplification / carrier dimension split, manual-epsilon
               override, per-run alpha (set_alpha), and pretty_print.
ltep.plots     layer-persistence diagrams (both epsilon lines) and MLP barcodes.

Stable infrastructure modules `VR_trajectories` and `runtime_sensitivity` are
imported flat from the repository root (they are not edited as part of this
work, so they are left in place rather than vendored).

Typical use in an experiment file
---------------------------------
    from ltep import pipeline as pl, plots
    pl.set_alpha(0.01)                       # this dataset's significance level
    eps = pl.select_epsilon(latents, ...)    # -> epsilons_H0, epsilons_H1
    conv = pl.convergence_depth(latents, eps["epsilons_H0"], significance=False)
"""

from . import metrics, pipeline, plots, output

# the names experiment files reach for most often, re-exported for convenience
from .pipeline import (
    select_epsilon,
    convergence_depth,
    signal_dimension,
    simplification_dimension,
    carrier_dimension,
    relevant_dimension,
    parse_manual_epsilons,
    cross_check_bottleneck,
    output_loop_anomaly,
    pretty_print,
    set_alpha,
    PARAMS,
)

__all__ = [
    "metrics", "pipeline", "plots",
    "select_epsilon", "convergence_depth",
    "signal_dimension", "simplification_dimension", "carrier_dimension",
    "relevant_dimension", "parse_manual_epsilons", "cross_check_bottleneck",
    "output_loop_anomaly", "pretty_print", "set_alpha", "PARAMS",
]

__version__ = "0.1.0"
