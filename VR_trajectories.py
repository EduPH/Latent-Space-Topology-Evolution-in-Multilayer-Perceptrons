import numpy as np
import gudhi as gd
from gudhi import CoverComplex
from gudhi import SimplexTree
import matplotlib.pyplot as plt
from sklearn import datasets
import networkx as nx
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from scipy.spatial.distance import cdist
from itertools import combinations
from mpl_toolkits.mplot3d import art3d


#%% Step 1: VR-clustering  compute_vietoris_rips_complex

def compute_vietoris_rips_complex(points, epsilon, max_dimension=2):
    """
    Compute the Vietoris-Rips complex for a given point cloud at a fixed epsilon value.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (n_points, n_dimensions) representing the point cloud
    epsilon : float
        Distance threshold for the Vietoris-Rips complex
    max_dimension : int, optional (default=2)
        Maximum dimension of the simplices in the complex
        
    Returns:
    --------
    simplex_tree : gudhi.SimplexTree
        A simplex tree representation of the Vietoris-Rips complex
    """
    # Initialize an empty simplex tree
    simplex_tree = SimplexTree()
    
    # Add vertices (0-simplices)
    for i in range(len(points)):
        simplex_tree.insert([i], filtration=0.0)
    

    dm = cdist(points, points)
    # Compute pairwise distances
    n_points = len(points)
    for i in range(n_points):
        for j in range(i+1, n_points):
            # Compute Euclidean distance between points i and j
            distance = dm[i,j]
            
            # If distance is less than or equal to epsilon, add the edge (1-simplex)
            if distance <= epsilon:
                simplex_tree.insert([i, j], filtration=distance)
    
    # Expand the simplex tree to include higher-dimensional simplices
    simplex_tree.expansion(max_dimension)
    
    return simplex_tree

#%% Plot VR 2D

def visualize_vr_complex_2d(points, simplex_tree, epsilon, show_labels=True):
    """
    Visualize the Vietoris-Rips complex in 2D.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (n_points, 2) representing the 2D point cloud
    simplex_tree : gudhi.SimplexTree
        Simplex tree representation of the Vietoris-Rips complex
    epsilon : float
        Distance threshold used to compute the complex
    show_labels : bool, optional (default=True)
        Whether to show vertex labels
    """
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50, zorder=3)
    
    # Add vertex labels
    if show_labels:
        for i, point in enumerate(points):
            plt.text(point[0], point[1], str(i), fontsize=12, 
                     ha='right', va='bottom', zorder=4)
    
    # Plot edges (1-simplices)
    for simplex, filtration in simplex_tree.get_simplices():
        if len(simplex) == 2 and filtration <= epsilon:
            i, j = simplex
            plt.plot([points[i, 0], points[j, 0]], 
                     [points[i, 1], points[j, 1]], 'k-', alpha=0.6, zorder=1)
    
    # Plot triangles (2-simplices)
    for simplex, filtration in simplex_tree.get_simplices():
        if len(simplex) == 3 and filtration <= epsilon:
            i, j, k = simplex
            triangle = plt.Polygon([points[i], points[j], points[k]], 
                                   alpha=0.3, color='gray', zorder=0)
            plt.gca().add_patch(triangle)
    
    plt.title(f'Vietoris-Rips Complex (ε = {epsilon:.2f})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gca()

#%% Plot VR 3D
def visualize_vr_complex_3d(points, simplex_tree, epsilon, show_labels=True):
    """
    Visualize the Vietoris-Rips complex in 3D.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (n_points, 3) representing the 3D point cloud
    simplex_tree : gudhi.SimplexTree
        Simplex tree representation of the Vietoris-Rips complex
    epsilon : float
        Distance threshold used to compute the complex
    show_labels : bool, optional (default=True)
        Whether to show vertex labels
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=50, zorder=3)
    
    # Add vertex labels
    if show_labels:
        for i, point in enumerate(points):
            ax.text(point[0], point[1], point[2], str(i), fontsize=12)
    
    # Plot edges (1-simplices)
    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 2 and filtration <= epsilon:
            i, j = simplex
            ax.plot([points[i, 0], points[j, 0]], 
                   [points[i, 1], points[j, 1]],
                   [points[i, 2], points[j, 2]], 'k-', alpha=0.6)
    
    # Plot triangles (2-simplices)
    for simplex, filtration in simplex_tree.get_filtration():
        if len(simplex) == 3 and filtration <= epsilon:
            i, j, k = simplex
            verts = [points[i], points[j], points[k]]
            # Create a Polygon3D - properly import art3d
            tri = art3d.Poly3DCollection([verts], alpha=0.3, color='gray')
            ax.add_collection3d(tri)
    
    ax.set_title(f'Vietoris-Rips Complex in 3D (ε = {epsilon:.2f})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax
#%% Print simplex tree info
def print_simplex_tree_info(simplex_tree, max_dim=2):
    """
    Print information about the simplices in the simplex tree.
    
    Parameters:
    -----------
    simplex_tree : gudhi.SimplexTree
        Simplex tree to analyze
    max_dim : int, optional (default=2)
        Maximum dimension to display
    """
    print(f"Simplex Tree Summary:")
    print(f"Number of simplices: {simplex_tree.num_simplices()}")
    print(f"Number of vertices: {simplex_tree.num_vertices()}")
    
    # Count simplices by dimension
    for dim in range(max_dim + 1):
        count = sum(1 for s in simplex_tree.get_simplices() if len(s[0]) == dim + 1)
        print(f"Number of {dim}-simplices: {count}")
    
    print("\nSimplices by dimension:")
    for dim in range(max_dim + 1):
        simplices = [s[0] for s in simplex_tree.get_simplices() if len(s[0]) == dim + 1]
        print(f"{dim}-simplices: {simplices}")

#%% Maximal simplices

def get_maximal_simplices(simplex_tree, epsilon):
    """
    Extract all maximal simplices from a simplex tree at or below the given threshold.
    
    A simplex is maximal if it is not a face of any other simplex in the complex.
    
    Parameters:
    -----------
    simplex_tree : gudhi.SimplexTree
        Simplex tree from which to extract maximal simplices
    epsilon : float, optional (default=infinity)
        Threshold value for filtration
        
    Returns:
    --------
    list : List of maximal simplices, where each simplex is a list of vertex indices
    """
    # Get all simplices with filtration value <= epsilon
    all_simplices = []
    for simplex, filtration_value in simplex_tree.get_filtration():
        if filtration_value <= epsilon:
            all_simplices.append(simplex)
    
    # Sort simplices by dimension (decreasing)
    all_simplices.sort(key=len, reverse=True)
    
    # Keep track of maximal simplices
    maximal_simplices = []
    covered_simplices = set()
    
    # Process simplices in order of decreasing dimension
    for simplex in all_simplices:
        simplex_tuple = tuple(sorted(simplex))
        
        # If the simplex or any of its faces is already covered, skip it
        if simplex_tuple in covered_simplices:
            continue
        
        # Mark this simplex as maximal
        maximal_simplices.append(simplex)
        
        # Mark all faces of this simplex as covered
        for d in range(1, len(simplex) + 1):
            for face in combinations(simplex, d):
                covered_simplices.add(tuple(sorted(face)))
    
    return maximal_simplices
#%% Is face

def is_face_of_any(simplex, maximal_simplices):
    """
    Check if a simplex is a face of any simplex in a list of maximal simplices.
    
    Parameters:
    -----------
    simplex : list or tuple
        The simplex to check
    maximal_simplices : list
        List of maximal simplices to check against
        
    Returns:
    --------
    bool : True if simplex is a face of any maximal simplex, False otherwise
    """
    simplex_set = set(simplex)
    
    for max_simplex in maximal_simplices:
        if simplex_set.issubset(set(max_simplex)):
            return True
    
    return False

#%% Filtered VR

def filter_vr_complex_by_maximal_simplices(vr_complex_previous, maximal_simplices_current):
    """
    Filter a VR complex from a previous layer based on maximal simplices in the current layer.
    Since indices are preserved across layers, we can do this directly.
    
    Parameters:
    -----------
    vr_complex_previous : gudhi.SimplexTree
        VR complex from the previous layer
    maximal_simplices_current : list
        List of (simplex, filtration) tuples for maximal simplices in current layer
    
    Returns:
    --------
    filtered_complex : gudhi.SimplexTree
        Filtered VR complex for the previous layer
    """
    # Extract just the simplices without filtration values
    #max_simplices_current = [simplex for simplex, _ in maximal_simplices_current]
    
    # Create a new simplex tree for the filtered complex
    filtered_complex = SimplexTree()
    
    # For each simplex in the previous layer complex
    for simplex, filtration in vr_complex_previous.get_filtration():
        # Skip empty simplex
        if len(simplex) == 0:
            continue
        
        # Check if simplex is a face of any maximal simplex in current layer
        for max_simplex in maximal_simplices_current:
            # Check if the simplex is a subset (face) of the maximal simplex
            keep_simplex = is_face_of_any(simplex, maximal_simplices_current)
            # Add to filtered complex if it's a face of some maximal simplex
            if keep_simplex:
                filtered_complex.insert(simplex, filtration)
    return filtered_complex
#%% Constructive VR pullback
def vr_pullback(points_previous, epsilon_previous, 
                            maximal_simplices_current, max_dimension=2):
    """
    Final optimal implementation for VR complex pullback that uses the insight
    that expansion automatically generates only valid simplices.
    
    Parameters:
    -----------
    points_previous : numpy.ndarray
        Point cloud in the previous layer
    epsilon_previous : float
        Epsilon value for previous layer
    maximal_simplices_current : list
        List of (simplex, filtration) tuples for maximal simplices in current layer
    max_dimension : int
        Maximum dimension to consider
        
    Returns:
    --------
    gudhi.SimplexTree : Simplex tree representation of the pullback complex
    """
    # # Extract just the simplices without filtration values
    # max_simplices_list = [simplex for simplex, _ in maximal_simplices_current]
    
    # Create a new simplex tree for the pullback complex
    pullback_complex = SimplexTree()
    
    # Step 1: Extract all unique vertices from maximal simplices
    vertices = set()
    for simplex in maximal_simplices_current:
        vertices.update(simplex)
    
    # Step 2: Precompute which maximal simplices each vertex belongs to
    vertex_to_max_simplices = {}
    for i, max_simplex in enumerate(maximal_simplices_current):
        for v in max_simplex:
            if v not in vertex_to_max_simplices:
                vertex_to_max_simplices[v] = set()
            vertex_to_max_simplices[v].add(i)
    
    # Step 3: Precompute distance matrix
    distances = cdist(points_previous,points_previous)
    
    # Step 4: Add all vertices to the complex
    for v in vertices:
        pullback_complex.insert([v], 0.0)
    
    # Step 5: Add edges that satisfy both conditions:
    # 1. Both vertices belong to at least one common maximal simplex
    # 2. The distance between them is <= epsilon_previous
    for i, v1 in enumerate(vertices):
        for v2 in vertices:
            if v1 < v2:  # Only process each pair once
                # Check if they share at least one maximal simplex
                if vertex_to_max_simplices[v1] & vertex_to_max_simplices[v2]:
                    # Check distance condition
                    dist = distances[v1, v2]
                    if dist <= epsilon_previous:
                        pullback_complex.insert([v1, v2], dist)
    
    # Step 6: Use expansion to compute higher-dimensional simplices
    # Since we've already filtered the edges, expansion will only
    # generate valid simplices that are faces of maximal simplices
    if max_dimension > 1:
        pullback_complex.expansion(max_dimension)
    
    return pullback_complex

#%% Skeleton to networkx for trajectories
def extract_1_skeleton_graph(simplex_tree, epsilon):
    """
    Extract the 1-skeleton (graph) from a simplex tree at a given epsilon value.
    
    Parameters:
    -----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree representation of the VR complex
    epsilon : float
        Threshold value for filtration
        
    Returns:
    --------
    G : networkx.Graph
        The 1-skeleton as a NetworkX graph
    """
    G = nx.Graph()
    
    # Add vertices (0-simplices)
    for simplex, filtration in simplex_tree.get_skeleton(0):
        if filtration <= epsilon:
            G.add_node(simplex[0])
    
    # Add edges (1-simplices)
    for simplex, filtration in simplex_tree.get_skeleton(1):
        if len(simplex) == 2 and filtration <= epsilon:
            G.add_edge(simplex[0], simplex[1], weight=1.0-filtration/epsilon)
    
    return G
#%% Communities / clustering
from sklearn.cluster import spectral_clustering
from scipy.sparse import csr_matrix
def identify_communities(G, n_clusters=None, method='louvain'):
    """
    Identify communities in the 1-skeleton graph, handling disconnected components.
    
    Parameters:
    -----------
    G : networkx.Graph
        The 1-skeleton graph
    n_clusters : int, optional
        Target number of clusters (used as guidance, not strict)
    method : str, optional
        Community detection method: 'connected_components', 'louvain', 
        'label_propagation', or 'greedy_modularity'
        
    Returns:
    --------
    communities : dict
        Dictionary mapping node indices to community indices
    """
    if len(G.nodes()) == 0:
        return {}
    
    # Always start by identifying connected components
    components = list(nx.connected_components(G))
    
    if method == 'connected_components' or len(components) >= (n_clusters or 2):
        # If there are already enough components or we explicitly want components
        communities = {}
        for i, component in enumerate(components):
            for node in component:
                communities[node] = i
        return communities
    
    # For other methods, we'll detect communities within each connected component
    communities = {}
    community_counter = 0
    
    for component in components:
        # Extract the subgraph for this component
        subgraph = G.subgraph(component)
        
        if len(subgraph) == 1:
            # Single node component gets its own community
            node = list(subgraph.nodes())[0]
            communities[node] = community_counter
            community_counter += 1
            continue
            
        # Determine approx. number of communities for this component
        if n_clusters is not None:
            # Proportionally allocate n_clusters across components
            component_size = len(subgraph)
            graph_size = len(G)
            comp_clusters = max(1, int(n_clusters * component_size / graph_size))
        else:
            comp_clusters = None  # Let the algorithm decide
        
        # Apply the selected community detection method
        if method == 'louvain':
            try:
                import community.community_louvain as community_louvain
                partition = community_louvain.best_partition(subgraph)
                
                # Renumber communities to be consecutive with other components
                for node, comm_id in partition.items():
                    communities[node] = comm_id + community_counter
                
                # Update counter for next component
                if partition:
                    community_counter += max(partition.values()) + 1
                
            except ImportError:
                print("python-louvain package not found. Using label propagation instead.")
                # Fall back to label propagation
                method = 'label_propagation'
        
        if method == 'label_propagation':
            local_communities = {}
            result = nx.algorithms.community.label_propagation.label_propagation_communities(subgraph)
            for i, community in enumerate(result):
                for node in community:
                    local_communities[node] = i
            
            # Add to global communities
            for node, comm_id in local_communities.items():
                communities[node] = comm_id + community_counter
            
            if local_communities:
                community_counter += max(local_communities.values()) + 1
            
        elif method == 'greedy_modularity':
            local_communities = {}
            result = nx.algorithms.community.modularity_max.greedy_modularity_communities(subgraph)
            for i, community in enumerate(result):
                for node in community:
                    local_communities[node] = i
            
            # Add to global communities
            for node, comm_id in local_communities.items():
                communities[node] = comm_id + community_counter
            
            if local_communities:
                community_counter += max(local_communities.values()) + 1
    
    return communities

#%% Compute trajectories
def compute_vr_trajectories(data_by_layer, simplex_trees_by_layer, epsilons, 
                           n_clusters=None, community_method='louvain'):
    """
    Compute trajectories using VR complexes and community detection.
    
    Parameters:
    -----------
    data_by_layer : list
        List of data arrays for each layer [X, X1, X2, ...]
    simplex_trees_by_layer : list
        List of simplex trees for each layer
    epsilons : list
        List of epsilon values for each layer
    n_clusters : list, optional
        Number of clusters to use for each layer. If None, estimated automatically.
    community_method : str, optional
        Community detection method to use
        
    Returns:
    --------
    trajectories : dict
        Dictionary mapping each point index to its trajectory
    communities_by_layer : list
        List of community assignments for each layer
    graphs_by_layer : list
        List of 1-skeleton graphs for each layer
    """
    n_layers = len(simplex_trees_by_layer)
    graphs_by_layer = []
    communities_by_layer = []
    
    # If n_clusters is a single number, repeat it for each layer
    if n_clusters is not None and not isinstance(n_clusters, list):
        n_clusters = [n_clusters] * n_layers
    
    # Extract 1-skeleton and identify communities for each layer
    for i in range(n_layers):
        # Extract 1-skeleton graph
        G = extract_1_skeleton_graph(simplex_trees_by_layer[i], epsilons[i])
        graphs_by_layer.append(G)
        
        # Identify communities
        cluster_count = None if n_clusters is None else n_clusters[i]
        communities = identify_communities(G, n_clusters=cluster_count, 
                                          method=community_method)
        communities_by_layer.append(communities)
    
    # Compute trajectories for each point
    trajectories = {}
    for point_idx in range(len(data_by_layer[0])):
        trajectory = []
        
        # Get community assignment for this point in each layer
        for i in range(n_layers):
            if point_idx in communities_by_layer[i]:
                trajectory.append(communities_by_layer[i][point_idx])
            else:
                # Point might not be in the graph (isolated)
                trajectory.append(-1)
        
        trajectories[point_idx] = trajectory
    
    return trajectories, communities_by_layer, graphs_by_layer
#%% Visualize trajectories
def visualize_trajectory_flow(trajectories, n_layers, class_labels=None, figsize=(12, 8)):
    """
    Visualize trajectories using a flow diagram based on matplotlib.
    
    Parameters:
    -----------
    trajectories : dict
        Dictionary mapping point indices to trajectories
    n_layers : int
        Number of layers in the network
    class_labels : array-like, optional
        Class labels for each point for coloring
    figsize : tuple, optional
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Count transitions between communities in consecutive layers
    transitions = []
    for layer in range(n_layers - 1):
        transition_counts = {}
        for point_idx, traj in trajectories.items():
            source_community = traj[layer]
            target_community = traj[layer + 1]
            
            # Skip trajectories involving -1 communities (no community assigned)
            if source_community == -1 or target_community == -1:
                continue
                
            key = (source_community, target_community)
            if key not in transition_counts:
                transition_counts[key] = 0
            transition_counts[key] += 1
        transitions.append(transition_counts)
    
    # Get all unique community IDs across all layers
    community_ids = set()
    for traj in trajectories.values():
        community_ids.update([c for c in traj if c != -1])
    
    # Determine positions for community nodes
    community_positions = {}
    max_communities_per_layer = 0
    
    # Count communities per layer
    communities_per_layer = [set() for _ in range(n_layers)]
    for point_idx, traj in trajectories.items():
        for layer, community in enumerate(traj):
            if community != -1:
                communities_per_layer[layer].add(community)
    
    # Determine positions
    for layer in range(n_layers):
        layer_communities = sorted(communities_per_layer[layer])
        max_communities_per_layer = max(max_communities_per_layer, len(layer_communities))
        
        for i, community in enumerate(layer_communities):
            position = i - len(layer_communities) / 2 + 0.5  # Center the communities
            community_positions[(layer, community)] = (layer, position)
    
    # Set up colormap for classes
    if class_labels is not None:
        unique_classes = np.unique(class_labels)
        cmap = plt.cm.get_cmap('tab10', max(len(unique_classes), 10))
        
        # Determine dominant class for each community in each layer
        community_classes = {}
        for layer in range(n_layers):
            for community in communities_per_layer[layer]:
                community_classes[(layer, community)] = {}
        
        for point_idx, traj in trajectories.items():
            if point_idx < len(class_labels):
                class_label = class_labels[point_idx]
                for layer, community in enumerate(traj):
                    if community != -1:
                        if class_label not in community_classes[(layer, community)]:
                            community_classes[(layer, community)][class_label] = 0
                        community_classes[(layer, community)][class_label] += 1
        
        # Assign color based on dominant class
        community_colors = {}
        for (layer, community), class_counts in community_classes.items():
            if class_counts:
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                # Find position of dominant_class in unique_classes
                class_idx = np.where(unique_classes == dominant_class)[0][0]
                community_colors[(layer, community)] = cmap(class_idx)
            else:
                community_colors[(layer, community)] = 'gray'
    
    # Draw transitions as curved lines with width proportional to count
    max_count = max([max(counts.values()) for counts in transitions]) if transitions else 1
    
    for layer, transition_counts in enumerate(transitions):
        for (source_community, target_community), count in transition_counts.items():
            # Get positions
            if (layer, source_community) not in community_positions or \
               (layer + 1, target_community) not in community_positions:
                continue
                
            x0, y0 = community_positions[(layer, source_community)]
            x1, y1 = community_positions[(layer + 1, target_community)]
            
            # Line width proportional to count
            line_width = 0.5 + 5.0 * count / max_count
            
            # Line color based on source community if class_labels provided
            if class_labels is not None and (layer, source_community) in community_colors:
                color = community_colors[(layer, source_community)]
            else:
                color = 'gray'
            
            # Draw a curved line
            plt.plot(
                [x0, x1], 
                [y0, y1],
                '-', 
                color=color, 
                alpha=0.6, 
                linewidth=line_width,
                zorder=1
            )
    
    # Draw community nodes
    for (layer, community), (x, y) in community_positions.items():
        # Node size proportional to number of points
        point_count = sum(1 for point_idx, traj in trajectories.items() 
                         if len(traj) > layer and traj[layer] == community)
        
        node_size = 100 + 200 * point_count / len(trajectories)
        
        # Node color based on dominant class if class_labels provided
        if class_labels is not None and (layer, community) in community_colors:
            color = community_colors[(layer, community)]
        else:
            color = 'lightblue'
        
        plt.scatter(x, y, s=node_size, c=[color], edgecolor='black', zorder=2)
        plt.text(x, y, f"{community}", ha='center', va='center', fontsize=8, zorder=3)
    
    # Set up axes
    plt.xlim(-0.5, n_layers-0.5)
    plt.ylim(-max_communities_per_layer/2 - 0.5, max_communities_per_layer/2 + 0.5)
    plt.xticks(range(n_layers), [f"Layer {i}" for i in range(n_layers)])
    plt.yticks([])
    plt.grid(axis='x', alpha=0.3)
    plt.title("Neural Network Trajectory Flow")
    
    plt.tight_layout()
    return plt.gca()

#%% Visualization communities (new)

def visualize_vr_communities_graph(graph, communities, title=None):
    """
    Visualize communities in a VR complex as a spring layout graph,
    especially useful for high-dimensional data.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The 1-skeleton graph
    communities : dict
        Dictionary mapping node indices to community indices
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Set up colormap for communities
    unique_communities = set(communities.values())
    n_communities = len(unique_communities)
    cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
    
    # Assign colors to nodes based on their community
    node_colors = []
    for node in graph.nodes():
        if node in communities:
            community = communities[node]
            color = cmap(community % 10)
            node_colors.append(color)
        else:
            node_colors.append('gray')
    
    # Position the graph using a spring layout
    pos = nx.spring_layout(graph, seed=42)  # Fixed seed for reproducibility
    
    # Draw the graph
    nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_nodes(graph, pos, 
                           node_color=node_colors, 
                           node_size=100,
                           edgecolors='black',
                           linewidths=0.5)
    
    # Add community labels at the centroid of each community
    for community_id in unique_communities:
        # Find nodes in this community
        community_nodes = [n for n, c in communities.items() if c == community_id]
        if not community_nodes:
            continue
        
        # Calculate centroid position
        centroid_x = sum(pos[n][0] for n in community_nodes) / len(community_nodes)
        centroid_y = sum(pos[n][1] for n in community_nodes) / len(community_nodes)
        
        # Add label
        plt.text(centroid_x, centroid_y, f"C{community_id}", 
                 size=16, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round'))
    
    if title:
        plt.title(title, size=16)
    else:
        plt.title(f"Community Structure ({n_communities} communities)", size=16)
    
    plt.axis('off')
    plt.tight_layout()
    return plt.gca()

def visualize_communities(points, graph, communities, simplex_tree=None, epsilon=None, title=None):
    """
    Visualize communities of a VR complex, selecting the appropriate
    visualization based on the dimensionality of the data.
    
    Parameters:
    -----------
    points : numpy.ndarray or None
        Array representing the point cloud. If None, just shows the graph.
    graph : networkx.Graph
        The 1-skeleton graph
    communities : dict
        Dictionary mapping node indices to community indices
    simplex_tree : gudhi.SimplexTree, optional
        The simplex tree for visualizing higher-dimensional simplices
    epsilon : float, optional
        Epsilon value for the VR complex
    title : str, optional
        Plot title
    """
    # If no points are provided, visualize just the graph
    if points is None:
        return visualize_vr_communities_graph(graph, communities, title)
    
    # Determine dimensionality
    dim = points.shape[1]
    
    if dim == 1:
        # For 1D data, create a 2D visualization with offsets
        plt.figure(figsize=(12, 8))
        
        # Set up colormap for communities
        unique_communities = set(communities.values())
        n_communities = len(unique_communities)
        cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
        
        # Sort points by their value
        sorted_indices = np.argsort(points[:, 0])
        x_values = points[sorted_indices, 0]
        
        # Assign y values to separate points with similar x values
        y_values = np.zeros(len(points))
        
        # Group points by their x values (within epsilon/10 of each other)
        group_epsilon = epsilon / 10 if epsilon else 0.01
        current_group = 0
        current_x = x_values[0]
        group_sizes = {}
        
        for i, idx in enumerate(sorted_indices):
            if x_values[i] > current_x + group_epsilon:
                # Start a new group
                current_group += 1
                current_x = x_values[i]
            
            # Assign this point to the current group
            if current_group not in group_sizes:
                group_sizes[current_group] = 0
            
            # Assign y value based on position within group
            # Alternate positive and negative to spread points
            sign = 1 if group_sizes[current_group] % 2 == 0 else -1
            y_values[idx] = sign * (group_sizes[current_group] // 2 + 1) * 0.1
            
            group_sizes[current_group] += 1
        
        # Create 2D points for visualization
        points_viz = np.column_stack((points[:, 0], y_values))
        
        # Draw edges
        for u, v in graph.edges():
            plt.plot([points_viz[u, 0], points_viz[v, 0]], 
                    [points_viz[u, 1], points_viz[v, 1]], 'k-', alpha=0.3, zorder=1)
        
        # Draw points colored by community
        for i, point in enumerate(points_viz):
            if i in communities:
                community = communities[i]
                color = cmap(community % 10)
                plt.scatter(point[0], point[1], c=[color], s=80, edgecolor='black', linewidth=0.5, zorder=2)
            else:
                # Isolated points (not in any community)
                plt.scatter(point[0], point[1], c='gray', s=80, edgecolor='black', linewidth=0.5, zorder=2)
        
        # Add labels for communities
        for community_id in unique_communities:
            community_points = [i for i, c in communities.items() if c == community_id]
            if community_points:
                centroid = np.mean(points_viz[community_points], axis=0)
                plt.text(centroid[0], centroid[1], f"C{community_id}", 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                        zorder=3)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"1D Data Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
        plt.xlabel("Value")
        plt.ylabel("Offset (for visualization)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        return plt.gca()
        
    elif dim == 2:
        # For 2D data, visualize points directly
        plt.figure(figsize=(12, 10))
        
        # Set up colormap for communities
        unique_communities = set(communities.values())
        n_communities = len(unique_communities)
        cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
        
        # Draw triangles if simplex_tree is provided
        if simplex_tree is not None and epsilon is not None:
            for simplex, filtration in simplex_tree.get_filtration():
                if len(simplex) == 3 and filtration <= epsilon:
                    i, j, k = simplex
                    plt.fill([points[i, 0], points[j, 0], points[k, 0]],
                            [points[i, 1], points[j, 1], points[k, 1]],
                            color='lightgray', alpha=0.15, zorder=0)
        
        # Draw edges
        for u, v in graph.edges():
            plt.plot([points[u, 0], points[v, 0]], 
                    [points[u, 1], points[v, 1]], 'k-', alpha=0.3, zorder=1)
        
        # Draw points colored by community
        for i, point in enumerate(points):
            if i in communities:
                community = communities[i]
                color = cmap(community % 10)
                plt.scatter(point[0], point[1], c=[color], s=80, edgecolor='black', linewidth=0.5, zorder=2)
            else:
                # Isolated points (not in any community)
                plt.scatter(point[0], point[1], c='gray', s=80, edgecolor='black', linewidth=0.5, zorder=2)
        
        # Add labels for communities
        for community_id in unique_communities:
            community_points = [i for i, c in communities.items() if c == community_id]
            if community_points:
                centroid = np.mean(points[community_points], axis=0)
                plt.text(centroid[0], centroid[1], f"C{community_id}", 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                        zorder=3)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"2D Data Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
        plt.axis('equal')
        plt.tight_layout()
        return plt.gca()
        
    elif dim == 3:
        # For 3D data, use a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up colormap for communities
        unique_communities = set(communities.values())
        n_communities = len(unique_communities)
        cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
        
        # Draw edges
        for u, v in graph.edges():
            ax.plot([points[u, 0], points[v, 0]], 
                   [points[u, 1], points[v, 1]],
                   [points[u, 2], points[v, 2]], 'k-', alpha=0.3)
        
        # Draw points colored by community
        for i, point in enumerate(points):
            if i in communities:
                community = communities[i]
                color = cmap(community % 10)
                ax.scatter(point[0], point[1], point[2], c=[color], s=80, edgecolor='black', linewidth=0.5)
            else:
                # Isolated points (not in any community)
                ax.scatter(point[0], point[1], point[2], c='gray', s=80, edgecolor='black', linewidth=0.5)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"3D Data Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
        return ax
        
    else:
        # For higher-dimensional data, just show the graph structure
        return visualize_vr_communities_graph(graph, communities, 
                                             title=title or f"{dim}D Data Communities Graph")
def visualize_vr_communities_detailed(points, graph, communities, simplex_tree=None, epsilon=None, title=None):
    """
    Visualize communities in a VR complex with detailed information.
    Handles 1D, 2D, and 3D data appropriately.
    
    Parameters:
    -----------
    points : numpy.ndarray
        Array of shape (n_points, n_dimensions) representing the point cloud
    graph : networkx.Graph
        The 1-skeleton graph
    communities : dict
        Dictionary mapping node indices to community indices
    simplex_tree : gudhi.SimplexTree, optional
        The simplex tree for visualizing higher-dimensional simplices
    epsilon : float, optional
        Epsilon value for the VR complex
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Set up colormap for communities
    unique_communities = set(communities.values())
    n_communities = len(unique_communities)
    cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
    
    # Handle different dimensionality of input data
    if points.shape[1] == 1:
        # For 1D data, create a 2D visualization where:
        # - x-axis is the original 1D value
        # - y-axis separates points within the same x value
        
        # First, sort points by their value
        sorted_indices = np.argsort(points[:, 0])
        x_values = points[sorted_indices, 0]
        
        # Assign y values to separate points with similar x values
        y_values = np.zeros(len(points))
        
        # Group points by their x values (within epsilon/10 of each other)
        group_epsilon = epsilon / 10 if epsilon else 0.01
        current_group = 0
        current_x = x_values[0]
        group_sizes = {}
        
        for i, idx in enumerate(sorted_indices):
            if x_values[i] > current_x + group_epsilon:
                # Start a new group
                current_group += 1
                current_x = x_values[i]
            
            # Assign this point to the current group
            if current_group not in group_sizes:
                group_sizes[current_group] = 0
            
            # Assign y value based on position within group
            # Alternate positive and negative to spread points
            sign = 1 if group_sizes[current_group] % 2 == 0 else -1
            y_values[idx] = sign * (group_sizes[current_group] // 2 + 1) * 0.1
            
            group_sizes[current_group] += 1
        
        # Create 2D points for visualization
        points_viz = np.column_stack((points[:, 0], y_values))
        
    elif points.shape[1] == 2:
        # For 2D data, use as is
        points_viz = points
    elif points.shape[1] == 3:
        # For 3D data, use a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw edges
        for u, v in graph.edges():
            ax.plot([points[u, 0], points[v, 0]], 
                   [points[u, 1], points[v, 1]],
                   [points[u, 2], points[v, 2]], 'k-', alpha=0.3)
        
        # Draw points colored by community
        for i, point in enumerate(points):
            if i in communities:
                community = communities[i]
                color = cmap(community % 10)
                ax.scatter(point[0], point[1], point[2], c=[color], s=80, edgecolor='black', linewidth=0.5)
            else:
                # Isolated points (not in any community)
                ax.scatter(point[0], point[1], point[2], c='gray', s=80, edgecolor='black', linewidth=0.5)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"VR Complex Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
        return ax
    else:
        # For higher-dimensional data, use t-SNE or PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        points_viz = pca.fit_transform(points)
    
    # For 1D and 2D visualization:
    
    # Draw edges
    for u, v in graph.edges():
        plt.plot([points_viz[u, 0], points_viz[v, 0]], 
                 [points_viz[u, 1], points_viz[v, 1]], 'k-', alpha=0.3, zorder=1)
    
    # Draw triangles if simplex_tree is provided
    if simplex_tree is not None and epsilon is not None:
        for simplex, filtration in simplex_tree.get_filtration():
            if len(simplex) == 3 and filtration <= epsilon:
                i, j, k = simplex
                plt.fill([points_viz[i, 0], points_viz[j, 0], points_viz[k, 0]],
                         [points_viz[i, 1], points_viz[j, 1], points_viz[k, 1]],
                         color='lightgray', alpha=0.15, zorder=0)
    
    # Draw points colored by community
    for i, point in enumerate(points_viz):
        if i in communities:
            community = communities[i]
            color = cmap(community % 10)
            plt.scatter(point[0], point[1], c=[color], s=80, edgecolor='black', linewidth=0.5, zorder=2)
        else:
            # Isolated points (not in any community)
            plt.scatter(point[0], point[1], c='gray', s=80, edgecolor='black', linewidth=0.5, zorder=2)
    
    # Add labels for communities
    for community_id in unique_communities:
        # Find all points in this community
        community_points = [i for i, c in communities.items() if c == community_id]
        if not community_points:
            continue
        
        # Calculate centroid
        centroid = np.mean(points_viz[community_points], axis=0)
        
        # Add label
        plt.text(centroid[0], centroid[1], f"C{community_id}", 
                 ha='center', va='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                 zorder=3)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"VR Complex Communities (ε = {epsilon:.2f}, {n_communities} communities)")
    
    if points.shape[1] == 1:
        plt.xlabel("Value")
        plt.ylabel("Offset (for visualization only)")
    
    plt.grid(alpha=0.2)
    plt.tight_layout()
    return plt.gca()
#%% Analysis trajectories
# Analyze trajectories to verify theoretical predictions
def analyze_trajectories(trajectories, y=None):
    """Analyze properties of the computed trajectories."""
    analysis = {}
    
    # Count unique trajectories
    unique_trajectories = {}
    for point_idx, traj in trajectories.items():
        traj_tuple = tuple(traj)
        if traj_tuple not in unique_trajectories:
            unique_trajectories[traj_tuple] = []
        unique_trajectories[traj_tuple].append(point_idx)
    
    analysis['n_unique_trajectories'] = len(unique_trajectories)
    analysis['trajectory_sizes'] = {traj: len(points) for traj, points in unique_trajectories.items()}
    
    # Sort trajectories by size (number of points)
    sorted_trajectories = sorted(
        unique_trajectories.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    print("\nTop trajectories by size:")
    for i, (traj, points) in enumerate(sorted_trajectories[:5]):  # Show top 5
        print(f"Trajectory {i+1}: {traj} - {len(points)} points")
    
    # If class labels are provided, analyze class separation
    if y is not None:
        # Check if different classes follow different trajectories
        class_trajectories = {}
        for traj, points in unique_trajectories.items():
            class_counts = {}
            for point_idx in points:
                if point_idx < len(y):
                    class_label = y[point_idx]
                    if class_label not in class_counts:
                        class_counts[class_label] = 0
                    class_counts[class_label] += 1
            
            class_trajectories[traj] = class_counts
        
        analysis['class_trajectories'] = class_trajectories
        
        # Calculate purity of each trajectory (how well it separates classes)
        trajectory_purity = {}
        for traj, class_counts in class_trajectories.items():
            total = sum(class_counts.values())
            if total > 0:
                max_class_count = max(class_counts.values())
                purity = max_class_count / total
            else:
                purity = 0
            trajectory_purity[traj] = purity
        
        analysis['trajectory_purity'] = trajectory_purity
        analysis['avg_trajectory_purity'] = sum(trajectory_purity.values()) / len(trajectory_purity)
        
        print(f"\nAverage trajectory purity: {analysis['avg_trajectory_purity']:.2f}")
        
        # Show purity for top trajectories
        print("\nClass distribution for top trajectories:")
        for i, (traj, points) in enumerate(sorted_trajectories[:5]):
            if traj in class_trajectories:
                class_counts = class_trajectories[traj]
                purity = trajectory_purity[traj]
                print(f"Trajectory {i+1}: {traj} - Purity: {purity:.2f}")
                for class_label, count in class_counts.items():
                    print(f"  Class {class_label}: {count} points ({count/len(points)*100:.1f}%)")
        
        # Check if the trajectories of different classes are disconnected
        class_separation = True
        for traj, class_counts in class_trajectories.items():
            if len(class_counts) > 1:
                class_separation = False
                break
        
        analysis['perfect_class_separation'] = class_separation
        print(f"\nPerfect class separation: {class_separation}")
    
    return analysis

#%% (old)

# def visualize_vr_communities_detailed(points, graph, communities, simplex_tree=None, epsilon=None, title=None):
#     """
#     Visualize communities in a VR complex with detailed information.
    
#     Parameters:
#     -----------
#     points : numpy.ndarray
#         Array of shape (n_points, n_dimensions) representing the point cloud
#     graph : networkx.Graph
#         The 1-skeleton graph
#     communities : dict
#         Dictionary mapping node indices to community indices
#     simplex_tree : gudhi.SimplexTree, optional
#         The simplex tree for visualizing higher-dimensional simplices
#     epsilon : float, optional
#         Epsilon value for the VR complex
#     title : str, optional
#         Plot title
#     """
#     if points.shape[1] > 3:
#         # For high-dimensional data, perform dimensionality reduction
#         from sklearn.manifold import TSNE
#         points_viz = TSNE(n_components=2).fit_transform(points)
#     elif points.shape[1] == 3:
#         # For 3D data, create a 3D plot
#         fig = plt.figure(figsize=(12, 10))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Set up colormap for communities
#         unique_communities = set(communities.values())
#         n_communities = len(unique_communities)
#         cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
        
#         # Draw edges
#         for u, v in graph.edges():
#             ax.plot([points[u, 0], points[v, 0]], 
#                    [points[u, 1], points[v, 1]],
#                    [points[u, 2], points[v, 2]], 'k-', alpha=0.3, zorder=1)
        
#         # Draw points colored by community
#         for i, point in enumerate(points):
#             if i in communities:
#                 community = communities[i]
#                 color = cmap(community % 10)
#                 ax.scatter(point[0], point[1], point[2], c=[color], s=50, zorder=2)
#             else:
#                 # Isolated points (not in any community)
#                 ax.scatter(point[0], point[1], point[2], c='gray', s=50, zorder=2)
        
#         # Draw triangles if simplex_tree is provided
#         if simplex_tree is not None and epsilon is not None:
#             from mpl_toolkits.mplot3d import art3d
#             for simplex, filtration in simplex_tree.get_filtration():
#                 if len(simplex) == 3 and filtration <= epsilon:
#                     i, j, k = simplex
#                     verts = [points[i], points[j], points[k]]
#                     # Create a Poly3DCollection
#                     tri = art3d.Poly3DCollection([verts], alpha=0.15, color='lightgray')
#                     ax.add_collection3d(tri)
        
#         if title:
#             ax.set_title(title)
#         else:
#             ax.set_title(f"VR Complex Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
#         return ax
#     else:
#         # For 2D data, create a standard plot
#         points_viz = points
        
#         plt.figure(figsize=(12, 10))
        
#         # Set up colormap for communities
#         unique_communities = set(communities.values())
#         n_communities = len(unique_communities)
#         cmap = plt.cm.get_cmap('tab10', max(n_communities, 10))
        
#         # Draw triangles if simplex_tree is provided
#         if simplex_tree is not None and epsilon is not None:
#             for simplex, filtration in simplex_tree.get_filtration():
#                 if len(simplex) == 3 and filtration <= epsilon:
#                     i, j, k = simplex
#                     plt.fill([points_viz[i, 0], points_viz[j, 0], points_viz[k, 0]],
#                              [points_viz[i, 1], points_viz[j, 1], points_viz[k, 1]],
#                              color='lightgray', alpha=0.15, zorder=1)
        
#         # Draw edges
#         for u, v in graph.edges():
#             plt.plot([points_viz[u, 0], points_viz[v, 0]], 
#                      [points_viz[u, 1], points_viz[v, 1]], 'k-', alpha=0.3, zorder=2)
        
#         # Draw points colored by community
#         for i, point in enumerate(points_viz):
#             if i in communities:
#                 community = communities[i]
#                 color = cmap(community % 10)
#                 plt.scatter(point[0], point[1], c=[color], s=80, edgecolor='black', linewidth=0.5, zorder=3)
#             else:
#                 # Isolated points (not in any community)
#                 plt.scatter(point[0], point[1], c='gray', s=80, edgecolor='black', linewidth=0.5, zorder=3)
        
#         # Add labels for communities
#         for community_id in unique_communities:
#             # Find all points in this community
#             community_points = [i for i, c in communities.items() if c == community_id]
#             if not community_points:
#                 continue
            
#             # Calculate centroid
#             centroid = np.mean(points_viz[community_points], axis=0)
            
#             # Add label
#             plt.text(centroid[0], centroid[1], f"C{community_id}", 
#                      ha='center', va='center', fontsize=12, 
#                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
#         if title:
#             plt.title(title)
#         else:
#             plt.title(f"VR Complex Communities (ε = {epsilon:.2f}, {n_communities} communities)")
        
#         plt.axis('equal')
#         plt.tight_layout()
#         return plt.gca()


#%% MLP persistence
def create_combined_filtration(simplex_trees_by_layer, max_dimension=2):
    """
    Create a single combined filtration from a sequence of simplex trees
    by assigning filtration values corresponding to the layer index.
    
    Parameters:
    -----------
    simplex_trees_by_layer : list
        List of Gudhi SimplexTree objects representing each layer
    max_dimension : int, optional (default=2)
        Maximum dimension of simplices to include
        
    Returns:
    --------
    combined_st : gudhi.SimplexTree
        A simplex tree containing the combined filtration
    """
    import gudhi as gd
    
    # Create a new empty simplex tree for the combined filtration
    combined_st = gd.SimplexTree()
    
    # Keep track of simplices we've already added
    added_simplices = set()
    
    # Process each layer
    for layer_idx, st in enumerate(simplex_trees_by_layer):
        # Get all simplices up to max_dimension
        for dim in range(max_dimension + 1):
            for simplex, _ in st.get_skeleton(dim):
                # Create a frozen set for the simplex (for hashing)
                simplex_set = frozenset(simplex)
                
                # If this simplex hasn't been added yet, add it with filtration = layer_idx
                if simplex_set not in added_simplices:
                    combined_st.insert(simplex, layer_idx)
                    added_simplices.add(simplex_set)
    
    return combined_st
