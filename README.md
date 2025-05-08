# Latent-Space-Topology-Evolution-in-Multilayer-Perceptrons

This repository provides tools for analyzing the topological structure of neural network latent spaces using techniques from computational topology. The primary focus is on visualizing and understanding how data representations evolve through the layers of a neural network by constructing and analyzing simplicial complexes.

## Overview

Neural networks transform data through a series of layers, gradually reshaping the representation space to make classification or regression tasks easier. This repository implements a novel approach to understanding these transformations by:

1. Constructing Vietoris-Rips complexes at each layer of the network
2. Analyzing the topological features of these complexes
3. Tracking how data points "flow" through the network using community detection
4. Visualizing trajectories to reveal how the network separates different classes

## Features

- Construction of Vietoris-Rips complexes for data points at each network layer
- Pullback operations to relate topological structures across layers
- Community detection to identify clusters in latent spaces
- Trajectory analysis to track how data flows through the network
- Visualization tools for 1D, 2D, and 3D data representations
- Analysis of class separation based on topological features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-topology.git
cd neural-network-topology

# Install dependencies
pip install numpy matplotlib tensorflow gudhi networkx scikit-learn scipy
```

For community detection using the Louvain method, you'll also need:

```bash
pip install python-louvain
```

## Usage

### Basic Example

```python
from VR_trajectories import *
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets

# Generate sample data
X, y = datasets.make_circles(n_samples=300, factor=0.5, noise=0.05)

# Create and train a simple model
model = Sequential([
    Dense(3, activation='sigmoid', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, batch_size=16, verbose=0)

# Extract layer outputs
hidden_layer_model = Model(inputs=model.input, outputs=model.layers[0].output)
X1 = hidden_layer_model.predict(X)  # Hidden layer output
X2 = model.predict(X)               # Output layer

# Construct VR complexes
epsilon_values = [0.3, 0.2, 0.1]  # For input, hidden, output layers
st2 = compute_vietoris_rips_complex(X2, epsilon_values[2], max_dimension=1)
ms2 = get_maximal_simplices(st2, epsilon_values[2])

# Perform pullback operations
k1 = vr_pullback(X1, epsilon_values[1], ms2, max_dimension=2)
ms1 = get_maximal_simplices(k1, epsilon_values[1])
k0 = vr_pullback(X, epsilon_values[0], ms1, max_dimension=2)

# Organize data and complexes by layer
data_by_layer = [X, X1, X2]
simplex_trees_by_layer = [k0, k1, st2]

# Compute and visualize trajectories
trajectories, communities_by_layer, graphs_by_layer = compute_vr_trajectories(
    data_by_layer, simplex_trees_by_layer, epsilon_values, 
    community_method='louvain'
)

# Visualize trajectories
visualize_trajectory_flow(trajectories, len(data_by_layer), class_labels=y)
plt.show()

# Analyze trajectories
analysis = analyze_trajectories(trajectories, y)
```

For a complete example, see `Experiment.py`.

## Key Concepts

### Vietoris-Rips Complex

A Vietoris-Rips complex is a simplicial complex constructed from a point cloud where:
- Vertices are the data points
- An edge connects two vertices if their distance is less than epsilon
- Higher-dimensional simplices are added whenever all of their vertices form edges

### Pullback Operation

The pullback operation establishes relationships between topological structures across network layers. Given a cover or clustering of layer i+1, it computes a corresponding cover for layer i that respects the mapping defined by the neural network.

### Trajectories

A trajectory tracks how a data point moves through the community structures of each layer. By analyzing trajectories, we can understand how the network separates different classes and identify the key transformations that enable classification.

## Documentation

For detailed documentation of functions and classes, see the docstrings in the source code or refer to the [API Documentation](docs/API.md).

## References

This work is related to concepts from computational topology for data analysis, particularly:
- Vietoris-Rips complexes
- Nerve complexes and the Nerve theorem
- Persistent homology
- Mapper algorithm

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{neural_network_topology,
  author = {Eduardo Paluzo-Hidalgo},
  title = {Latent Space Topology Evolution in Multilayer Perceptrons},
  year = {2025},
  url = {https://github.com/yourusername/neural-network-topology}
}
```
