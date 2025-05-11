from VR_trajectories import *
from st_to_sophia import *

#%% Experiment 1
#%% Generate a dataset for demonstration
def generate_sample_data():
    """Generate a simple dataset for demonstration"""
    n_points = 300
    X, y = datasets.make_circles(n_samples=n_points, factor=0.5, noise=0.05)
    return X, y

X, y = generate_sample_data()
points = X
plt.plot(points[:,0],points[:,1],'.')
plt.show()    
    
    
#%% Model training
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model



model = Sequential([
    layers.Input(shape=(2,),name='input_layer'),
    layers.Dense(3, activation='sigmoid' ,name='hidden_layer'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])


# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

#model.fit(X, y, epochs=2000,batch_size = 16)

#model.save('experiment.keras')
model = keras.models.load_model('experiment.keras')


#%% Latent representations
X2 = model.predict(X)

plt.scatter(X[:,0], X[:,1], c=X2>0.5)
plt.show()

hidden_layer_model = Model(
    inputs= model.get_layer(index=0).input,
    outputs=model.get_layer('hidden_layer').output
)

X1 = hidden_layer_model.predict(X)

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(X1[:,0],X1[:,1],X1[:,2],c = X2>0.5)
plt.show()
#%% VR-complexes and pullback
epsilon_values = [0.3,0.2,0.1]
st2 = compute_vietoris_rips_complex(X2,epsilon_values[2],max_dimension=1)

ms2 = get_maximal_simplices(st2, epsilon_values[2])

k1 = vr_pullback(X1, epsilon_values[1], ms2, max_dimension=2)
visualize_vr_complex_3d(X1, k1, epsilon_values[1], show_labels=False)

ms1 = get_maximal_simplices(k1, epsilon_values[1])

k0 = vr_pullback(X, epsilon_values[0], ms1, max_dimension=2)
visualize_vr_complex_2d(X, k0, epsilon_values[0], show_labels=False)
#%% Communities and trajectory analysis
# After computing the VR complexes for each layer
# Define parameters
#epsilon_values = [0.3, 0.2, 0.1]  # For input, hidden, output layers
community_method = 'louvain'      # Community detection method
n_clusters = None                # Auto-detect number of clusters

# Organize data and complexes by layer
data_by_layer = [X, X1, X2]  # Input, hidden, output
simplex_trees_by_layer = [
    k0,
    k1,  # Your pullback complex for hidden layer
    st2  # Your VR complex for output layer
]

# Extract 1-skeleton graphs and identify communities
graphs_by_layer = []
communities_by_layer = []

for i, (points, st) in enumerate(zip(data_by_layer, simplex_trees_by_layer)):
    # Extract 1-skeleton graph
    G = extract_1_skeleton_graph(st, epsilon_values[i])
    graphs_by_layer.append(G)
    
    # Identify communities (connected components or using community detection)
    communities = identify_communities(G, method=community_method)
    communities_by_layer.append(communities)
    
    # Print community statistics
    unique_communities = set(communities.values())
    print(f"Layer {i}: {len(unique_communities)} communities detected")
    
    # Visualize communities
    if points.shape[1] <= 3:
        visualize_communities(
            points, G, communities, 
            simplex_tree=st, epsilon=epsilon_values[i],
            title=f"Layer {i} Communities"
        )
        plt.show()

# Compute trajectories
trajectories = {}
for point_idx in range(len(data_by_layer[0])):
    trajectory = []
    
    # Get community assignment for this point in each layer
    for i in range(len(data_by_layer)):
        communities = communities_by_layer[i]
        if point_idx in communities:
            trajectory.append(communities[point_idx])
        else:
            # Point might not be in the graph (isolated)
            trajectory.append(-1)
    
    trajectories[point_idx] = trajectory

# Visualize trajectory flow
visualize_trajectory_flow(trajectories, len(data_by_layer), class_labels=y)
plt.show()



# Run trajectory analysis
analysis = analyze_trajectories(trajectories, y)

# Layer persistence diagrams
def compute_plot_pd(X,max_dim = 2):
    dm = cdist(X, X)
    simplex_tree = gd.SimplexTree.create_from_array(dm, max_filtration=3)
    simplex_tree.expansion(max_dim)
    persistence = simplex_tree.persistence(homology_coeff_field = 2)
    gd.plot_persistence_diagram(persistence)
    plt.show()
    return
compute_plot_pd(X,max_dim=2)
compute_plot_pd(X1,max_dim=2)
compute_plot_pd(X2,max_dim=0)



# ML Persistence
k0.expansion(1)
k1.expansion(2)

# Diagrams H0
simplex_trees_by_layer=[k0,k1,st2]
k=create_combined_filtration(simplex_trees_by_layer)
k.compute_persistence()
intervals = k.persistence_intervals_in_dimension(0)
gd.plot_persistence_diagram(intervals)
gd.plot_persistence_barcode(intervals)
plt.show()


# Diagrams H1
simplex_trees_by_layer=[k0,k1]
k=create_combined_filtration(simplex_trees_by_layer)
k.compute_persistence()
intervals = k.persistence_intervals_in_dimension(1)
gd.plot_persistence_diagram(intervals)
gd.plot_persistence_barcode(intervals)
plt.show()

# Diagrams H2
simplex_trees_by_layer=[k1]
k=create_combined_filtration(simplex_trees_by_layer)
k.compute_persistence(persistence_dim_max=True)
intervals = k.persistence_intervals_in_dimension(2)
gd.plot_persistence_diagram(intervals)
gd.plot_persistence_barcode(intervals)
plt.show()

