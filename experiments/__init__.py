"""Per-dataset experiments. Each module = data loading + that dataset's PARAMS
block + run logic, built on the shared `ltep` core. Run from the repo root:

    python -m experiments.cardio --full
    python -m experiments.coil100 --tasks ae --group-size 1
    python -m experiments.resnet  --all --seeds 3
    python -m experiments.cifar_dense --full
"""
