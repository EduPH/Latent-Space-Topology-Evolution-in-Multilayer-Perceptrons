from VR_trajectories import *
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import numpy as np

#%% Experiment with Cardiotocography dataset
#%% Load and prepare dataset
def load_cardio_dataset():
    """Load the Cardiotocography dataset and prepare it for binary classification"""
    # Fetch the UCI Cardiotocography dataset
    cardio = fetch_ucirepo(id=193)
    
    # Get features and target
    X = cardio.data.features.values
    y = cardio.data.targets
    
    # For binary classification, convert NSP (Fetal state class code) to binary
    # <=4 is one class, >4 is another class
    y_binary = np.array(y)[:,0]>4
    
    # Standardize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for training and visualization
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
    
    print(f"Dataset shape: {X_scaled.shape}")
    print(f"Binary class distribution: {np.bincount(y_binary)}")
    
    return X, y, X_scaled, y_binary, X_train, y_train, X_test, y_test

# Load and prepare the dataset
X_original, y_original, X, y, X_train, y_train, X_test, y_test = load_cardio_dataset()



## Auxiliary function for layer persistence

def compute_plot_pd(X,filename,max_dim = 2):
    X=gd.subsampling.sparsify_point_set(points=X, min_squared_dist=0.05)
    dm = cdist(X, X)
    simplex_tree = gd.SimplexTree.create_from_array(dm, max_filtration=5)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(max_dim)
    persistence = simplex_tree.persistence(homology_coeff_field = 2)
    simplex_tree.write_persistence_diagram(filename)
    gd.plot_persistence_diagram(persistence)
    plt.show()
    return simplex_tree

## Model training

# Define the model
model = Sequential([
    layers.Input(shape=(X.shape[1],), name='input_layer'),
    layers.Dense(128, activation='sigmoid', name='hidden_layer1'),
    layers.Dense(1, activation='sigmoid', name='output_layer')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Train the model
history = model.fit(X_train, y_train, epochs=5000, batch_size=32,validation_split=0.2, verbose=1)       

#%% Latent representations
# Get predictions
X2 = model.predict(X)


# Extract hidden layer representations
hidden_layer_model = Model(
    inputs=model.get_layer(index=0).input,
    outputs=model.get_layer('hidden_layer1').output
)

X1 = hidden_layer_model.predict(X)


#%% Layer persistence diagrams

# Compute persistence diagrams for each layer
print("Input Layer Persistence Diagram:")
st0 = compute_plot_pd(X,"Ex2_Layer_pers_0_2.txt",max_dim=2)

print("Hidden Layer 1 Persistence Diagram:")
st1 = compute_plot_pd(X1,"Ex2_Layer_pers_1_2.txt", max_dim=2)

print("Output layer Persistence Diagram:")
st2 = compute_plot_pd(X2,"Ex2_Layer_pers_2_2.txt" ,max_dim=2)


#%% VR-complexes and pullback
epsilon_values = [1,2.5,0.2]
#epsilon_values = [2,2,0.02]
X0_sparse = gd.subsampling.sparsify_point_set(points=X, min_squared_dist=0.5)
X2_sparse = model.predict(X0_sparse)


# Extract hidden layer repreentations
hidden_layer_model = Model(
    inputs=model.get_layer(index=0).input,
    outputs=model.get_layer('hidden_layer1').output
)

X1_sparse = hidden_layer_model.predict(X0_sparse)


# MLP Persistence

st2 = compute_vietoris_rips_complex(X2_sparse, epsilon_values[2], max_dimension=1)

ms2 = get_maximal_simplices(st2, epsilon_values[2])

k1 = vr_pullback(X1_sparse, epsilon_values[1], ms2, max_dimension=2)

ms1 = get_maximal_simplices(k1, epsilon_values[1])

k0 = vr_pullback(X0_sparse, epsilon_values[0], ms1, max_dimension=2)



k0.expansion(2)
k1.expansion(2)

# Diagrams H0
simplex_trees_by_layer=[k0,k1,st2]
k=create_combined_filtration(simplex_trees_by_layer)
k.compute_persistence()
intervals0 = k.persistence_intervals_in_dimension(0)

# Diagrams H1
simplex_trees_by_layer=[k0,k1]
k=create_combined_filtration(simplex_trees_by_layer)
k.compute_persistence()
intervals1 = k.persistence_intervals_in_dimension(1)

pds = {}
pds[0] = intervals0
pds[1] = intervals1

l = []
for i in [0,1]:
    for [j,k] in pds[i]:
        if i==1:
            if k == np.inf:
                l.append((i,(j,2)))
            else:
                l.append((i,(j,k)))
        else:
            l.append((i,(j,k)))
gd.plot_persistence_barcode(l)

# Organize data and complexes by layer
data_by_layer = [X0_sparse, X1_sparse, X2_sparse]  # Input, hidden, output
simplex_trees_by_layer = [
    k0,
    k1,
    st2  # Your VR complex for output layer
]


## Trajectories

# Extract 1-skeleton graphs and identify communities
graphs_by_layer = []
communities_by_layer = []

for i, (points, st) in enumerate(zip(data_by_layer, simplex_trees_by_layer)):
    # Extract 1-skeleton graph
    G = extract_1_skeleton_graph(st, epsilon_values[i])
    graphs_by_layer.append(G)
    
    # Identify communities (connected components or using community detection)
    communities = nx.connected_components(G)#identify_communities(G, method=community_method)
    communities = {}
    ids = 0
    for c in list(nx.connected_components(G)):
        for v in c:
            communities[v]=ids
        ids+=1
    communities_by_layer.append(communities)
    
    # Print community statistics
    unique_communities = set(communities.values())
    print(f"Layer {i}: {len(unique_communities)} communities detected")
    
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

indices = find_row_indices(np.array(X),np.array(X0_sparse))

# Visualize trajectory flow
visualize_trajectory_flow(trajectories, len(data_by_layer), class_labels=y[indices])
plt.show()



# Run trajectory analysis
analysis = analyze_trajectories(trajectories, y[indices])


