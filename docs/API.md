# API Documentation

This document provides detailed information about the modules and functions in the Neural Network Topology Analysis package.

## Table of Contents
- [VR_trajectories Module](#vr_trajectories-module)
  - [Vietoris-Rips Complex Functions](#vietoris-rips-complex-functions)
  - [Visualization Functions](#visualization-functions)
  - [Pullback Operations](#pullback-operations)
  - [Community Detection](#community-detection)
  - [Trajectory Analysis](#trajectory-analysis)

## VR_trajectories Module

The `VR_trajectories.py` module contains the core functionality for constructing Vietoris-Rips complexes, performing pullback operations, detecting communities, and analyzing trajectories.

### Vietoris-Rips Complex Functions

#### `compute_vietoris_rips_complex(points, epsilon, max_dimension=2)`

Computes the Vietoris-Rips complex for a given point cloud at a specified epsilon value.

**Parameters:**
- `points` (numpy.ndarray): Array of shape (n_points, n_dimensions) representing the point cloud
- `epsilon` (float): Distance threshold for the Vietoris-Rips complex
- `max_dimension` (int, optional): Maximum dimension of the simplices in the complex. Default is 2.

**Returns:**
- `simplex_tree` (gudhi.SimplexTree): A simplex tree representation of the Vietoris-Rips complex

**Example:**
```python
import numpy as np
from VR_trajectories import compute_vietoris_rips_complex

# Create a simple point cloud
points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Compute VR complex with epsilon = 1.5
st = compute_vietoris_rips_complex(points, 1.5)
```

#### `get_maximal_simplices(simplex_tree, epsilon)`

Extracts all maximal simplices from a simplex tree at or below the given threshold.

**Parameters:**
- `simplex_tree` (gudhi.SimplexTree): Simplex tree from which to extract maximal simplices
- `epsilon` (float): Threshold value for filtration

**Returns:**
- List of maximal simplices, where each simplex is a list of vertex indices

#### `is_face_of_any(simplex, maximal_simplices)`

Checks if a simplex is a face of any simplex in a list of maximal simplices.

**Parameters:**
- `simplex` (list or tuple): The simplex to check
- `maximal_simplices` (list): List of maximal simplices to check against

**Returns:**
- `bool`: True if simplex is a face of any maximal simplex, False otherwise

### Visualization Functions

#### `visualize_vr_complex_2d(points, simplex_tree, epsilon, show_labels=True)`

Visualizes the Vietoris-Rips complex in 2D.

**Parameters:**
- `points` (numpy.ndarray): Array of shape (n_points, 2) representing the 2D point cloud
- `simplex_tree` (gudhi.SimplexTree): Simplex tree representation of the Vietoris-Rips complex
- `epsilon` (float): Distance threshold used to compute the complex
- `show_labels` (bool, optional): Whether to show vertex labels. Default is True.

**Returns:**
- Matplotlib axes object

#### `visualize_vr_complex_3d(points, simplex_tree, epsilon, show_labels=True)`

Visualizes the Vietoris-Rips complex in 3D.

**Parameters:**
- `points` (numpy.ndarray): Array of shape (n_points, 3) representing the 3D point cloud
- `simplex_tree` (gudhi.SimplexTree): Simplex tree representation of the Vietoris-Rips complex
- `epsilon` (float): Distance threshold used to compute the complex
- `show_labels` (bool, optional): Whether to show vertex labels. Default is True.

**Returns:**
- Matplotlib axes object

#### `visualize_communities(points, graph, communities, simplex_tree=None, epsilon=None, title=None)`

Visualizes communities of a VR complex, selecting the appropriate visualization based on the dimensionality of the data.

**Parameters:**
- `points` (numpy.ndarray or None): Array representing the point cloud. If None, just shows the graph.
- `graph` (networkx.Graph): The 1-skeleton graph
- `communities` (dict): Dictionary mapping node indices to community indices
- `simplex_tree` (gudhi.SimplexTree, optional): The simplex tree for visualizing higher-dimensional simplices
- `epsilon` (float, optional): Epsilon value for the VR complex
- `title` (str, optional): Plot title

**Returns:**
- Matplotlib axes object

#### `visualize_trajectory_flow(trajectories, n_layers, class_labels=None, figsize=(12, 8))`

Visualizes trajectories using a flow diagram based on matplotlib.

**Parameters:**
- `trajectories` (dict): Dictionary mapping point indices to trajectories
- `n_layers` (int): Number of layers in the network
- `class_labels` (array-like, optional): Class labels for each point for coloring
- `figsize` (tuple, optional): Figure size. Default is (12, 8).

**Returns:**
- Matplotlib axes object

### Pullback Operations

#### `vr_pullback(points_previous, epsilon_previous, maximal_simplices_current, max_dimension=2)`

Computes the pullback of a VR complex from one layer to the previous layer.

**Parameters:**
- `points_previous` (numpy.ndarray): Point cloud in the previous layer
- `epsilon_previous` (float): Epsilon value for previous layer
- `maximal_simplices_current` (list): List of maximal simplices in current layer
- `max_dimension` (int): Maximum dimension to consider. Default is 2.

**Returns:**
- `gudhi.SimplexTree`: Simplex tree representation of the pullback complex

#### `filter_vr_complex_by_maximal_simplices(vr_complex_previous, maximal_simplices_current)`

Filters a VR complex from a previous layer based on maximal simplices in the current layer.

**Parameters:**
- `vr_complex_previous` (gudhi.SimplexTree): VR complex from the previous layer
- `maximal_simplices_current` (list): List of maximal simplices in current layer

**Returns:**
- `filtered_complex` (gudhi.SimplexTree): Filtered VR complex for the previous layer

### Community Detection

#### `extract_1_skeleton_graph(simplex_tree, epsilon)`

Extracts the 1-skeleton (graph) from a simplex tree at a given epsilon value.

**Parameters:**
- `simplex_tree` (gudhi.SimplexTree): The simplex tree representation of the VR complex
- `epsilon` (float): Threshold value for filtration

**Returns:**
- `G` (networkx.Graph): The 1-skeleton as a NetworkX graph

#### `identify_communities(G, n_clusters=None, method='louvain')`

Identifies communities in the 1-skeleton graph, handling disconnected components.

**Parameters:**
- `G` (networkx.Graph): The 1-skeleton graph
- `n_clusters` (int, optional): Target number of clusters (used as guidance, not strict)
- `method` (str, optional): Community detection method: 'connected_components', 'louvain', 'label_propagation', or 'greedy_modularity'

**Returns:**
- `communities` (dict): Dictionary mapping node indices to community indices

### Trajectory Analysis

#### `compute_vr_trajectories(data_by_layer, simplex_trees_by_layer, epsilons, n_clusters=None, community_method='louvain')`

Computes trajectories using VR complexes and community detection.

**Parameters:**
- `data_by_layer` (list): List of data arrays for each layer [X, X1, X2, ...]
- `simplex_trees_by_layer` (list): List of simplex trees for each layer
- `epsilons` (list): List of epsilon values for each layer
- `n_clusters` (list or int, optional): Number of clusters to use for each layer. If None, estimated automatically.
- `community_method` (str, optional): Community detection method to use. Default is 'louvain'.

**Returns:**
- `trajectories` (dict): Dictionary mapping each point index to its trajectory
- `communities_by_layer` (list): List of community assignments for each layer
- `graphs_by_layer` (list): List of 1-skeleton graphs for each layer

#### `analyze_trajectories(trajectories, y=None)`

Analyzes properties of the computed trajectories.

**Parameters:**
- `trajectories` (dict): Dictionary mapping point indices to trajectories
- `y` (array-like, optional): Class labels for each point

**Returns:**
- `analysis` (dict): Dictionary containing analysis results, including:
  - `n_unique_trajectories`: Number of unique trajectories
  - `trajectory_sizes`: Dictionary mapping trajectories to their sizes
  - `class_trajectories`: Dictionary mapping trajectories to class distributions (if `y` is provided)
  - `trajectory_purity`: Dictionary mapping trajectories to purity values (if `y` is provided)
  - `avg_trajectory_purity`: Average purity of trajectories (if `y` is provided)
  - `perfect_class_separation`: Whether trajectories perfectly separate classes (if `y` is provided)

## Experiment Module

The `Experiment.ipynb` jupyter notebook provides a complete example of using the `VR_trajectories` module to analyze a simple neural network.

Key steps in the example:
1. Generate a synthetic dataset (circles)
2. Train a simple neural network with one hidden layer
3. Extract activations from each layer
4. Construct VR complexes and perform pullback operations
5. Identify communities and compute trajectories
6. Visualize and analyze the results

This file serves as a practical demonstration of how to apply the topology analysis tools to a neural network.
