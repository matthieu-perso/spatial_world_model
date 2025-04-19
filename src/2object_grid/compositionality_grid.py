import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm

from prob_grid import load_saved_probe

# -------------------------------
# Spatial vector extraction and analysis
# -------------------------------
def extract_spatial_vectors(probe):
    """Extract spatial vectors from the probe's weight matrix."""
    # Get the weights from the linear layer
    weights = probe.linear.weight.data
    
    # Reshape to get per-class, per-voxel weights
    d_model = weights.shape[1]
    grid_size = probe.grid_size
    num_classes = probe.num_classes
    
    # Reshape to [num_classes, z, y, x, d_model]
    weights_reshaped = weights.view(num_classes, grid_size[2], grid_size[1], grid_size[0], d_model)
    
    return weights_reshaped

def compute_spatial_relation_vectors(grid_vectors):
    """Compute spatial relation vectors by analyzing the probe's weight patterns."""
    # Extract objects' vectors at different positions
    # Grid vectors shape: [num_classes, z, y, x, d_model]
    num_classes, z_dim, y_dim, x_dim, d_model = grid_vectors.shape
    
    # Center of the grid
    center_z, center_y, center_x = z_dim // 2, y_dim // 2, x_dim // 2
    
    # Get reference vectors for each spatial relation
    from data_grid import SPATIAL_MAP
    
    relation_vectors = {}
    
    # For each spatial relationship
    for relation_name, (dx, dy, dz) in SPATIAL_MAP.items():
        # Calculate position relative to center
        pos_x = center_x + dx
        pos_y = center_y + dy
        pos_z = center_z + dz
        
        # Check if position is within grid boundaries
        if (0 <= pos_x < x_dim and 0 <= pos_y < y_dim and 0 <= pos_z < z_dim):
            # Get difference between this position and center for all classes
            # We use class 1 (corresponding to "chair") for reference
            chair_class = 1
            pos_vector = grid_vectors[chair_class, pos_z, pos_y, pos_x]
            center_vector = grid_vectors[chair_class, center_z, center_y, center_x]
            
            # Compute relation vector as the difference
            relation_vector = pos_vector - center_vector
            relation_vectors[relation_name] = relation_vector
    
    return relation_vectors

# -------------------------------
# PCA analysis
# -------------------------------
def analyze_in_pca_space(relation_vectors, n_components=3):
    """Analyze spatial relations in PCA-reduced space"""
    # Stack vectors for PCA
    rel_names = list(relation_vectors.keys())
    vecs = torch.stack([relation_vectors[rel] for rel in rel_names]).cpu().numpy()

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_vecs = pca.fit_transform(vecs)

    # Create mapping from relations to PCA vectors
    pca_relation_vectors = {rel: pca_vecs[i] for i, rel in enumerate(rel_names)}

    # Explain variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by {n_components} components: {explained_variance}")
    print(f"Total explained variance: {sum(explained_variance):.4f}")

    # Check compositional relations in PCA space
    check_compositional_relations(pca_relation_vectors)

    # Visualize in 3D PCA space
    if n_components >= 3:
        visualize_3d_pca(pca_relation_vectors)

    # Visualize in 2D PCA space
    visualize_2d_pca(pca_relation_vectors)

    return pca_relation_vectors, pca

def check_compositional_relations(pca_relation_vectors):
    """Check compositional properties in PCA space"""
    # Check compositional relations
    compositional_pairs = [
        ("diagonally_front_left", ["left", "in_front"]),
        ("diagonally_front_right", ["right", "in_front"]),
        ("diagonally_back_left", ["left", "behind"]),
        ("diagonally_back_right", ["right", "behind"]),
        ("above_and_left", ["above", "left"]),
        ("above_and_right", ["above", "right"]),
        ("below_and_left", ["below", "left"]),
        ("below_and_right", ["below", "right"]),
        ("above_and_in_front", ["above", "in_front"]),
        ("above_and_behind", ["above", "behind"]),
        ("below_and_in_front", ["below", "in_front"]),
        ("below_and_behind", ["below", "behind"]),
    ]

    print("\n=== COMPOSITIONAL RELATIONS IN PCA SPACE ===")
    for complex_rel, component_rels in compositional_pairs:
        if complex_rel in pca_relation_vectors and all(r in pca_relation_vectors for r in component_rels):
            complex_vec = pca_relation_vectors[complex_rel]

            # Sum the component vectors
            component_sum = sum(pca_relation_vectors[r] for r in component_rels)

            # Compute cosine similarity
            cos_sim = cosine_similarity([complex_vec], [component_sum])[0][0]

            # Compute angle
            angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Compute Euclidean distance
            euclidean_dist = np.linalg.norm(complex_vec - component_sum)

            print(f"Testing composition: {complex_rel} vs {'+'.join(component_rels)}")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  Angle: {angle_deg:.2f}°")
            print(f"  Euclidean distance: {euclidean_dist:.4f}")
            print()

def visualize_2d_pca(pca_relation_vectors):
    """Visualize relations in 2D PCA space"""
    plt.figure(figsize=(14, 12))

    # All relation types
    relation_types = {
        'Basic': ['above', 'below', 'left', 'right', 'in_front', 'behind'],
        'Diagonal': ['diagonally_front_left', 'diagonally_front_right',
                     'diagonally_back_left', 'diagonally_back_right'],
        'Compound': ['above_and_left', 'above_and_right', 'below_and_left', 'below_and_right',
                    'above_and_in_front', 'above_and_behind', 'below_and_in_front', 'below_and_behind']
    }

    # Colors for each type
    colors = {'Basic': 'blue', 'Diagonal': 'green', 'Compound': 'red'}

    # Plot origin
    plt.scatter(0, 0, color='black', s=100, marker='x', label='Origin')

    # Plot each relation vector
    for rel_type, rels in relation_types.items():
        for rel in rels:
            if rel in pca_relation_vectors:
                vec = pca_relation_vectors[rel]
                plt.arrow(0, 0, vec[0], vec[1],
                         head_width=0.05, head_length=0.1,
                         fc=colors[rel_type], ec=colors[rel_type],
                         alpha=0.7, label=rel if rel_type == 'Basic' else None)
                plt.text(vec[0]*1.1, vec[1]*1.1, rel, fontsize=9)

    # Draw lines connecting compositional pairs
    compositional_pairs = [
        ("above_and_left", ["above", "left"]),
        ("diagonally_front_right", ["right", "in_front"]),
        ("below_and_behind", ["below", "behind"])
    ]

    for complex_rel, component_rels in compositional_pairs:
        if complex_rel in pca_relation_vectors and all(r in pca_relation_vectors for r in component_rels):
            # Draw the composed vector
            complex_vec = pca_relation_vectors[complex_rel]

            # Draw component vectors sum
            comp1 = pca_relation_vectors[component_rels[0]]
            comp2 = pca_relation_vectors[component_rels[1]]
            sum_vec = comp1 + comp2

            # Draw a line connecting them
            plt.plot([complex_vec[0], sum_vec[0]], [complex_vec[1], sum_vec[1]],
                    'k--', alpha=0.5)

            # Label the sum
            plt.text(sum_vec[0]*1.05, sum_vec[1]*1.05,
                    f"{component_rels[0]}+{component_rels[1]}",
                    fontsize=8, color='purple')

    # Add legend, grid, axes
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title("2D PCA Projection of Spatial Relation Vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_3d_pca(pca_relation_vectors):
    """Visualize relations in 3D PCA space"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot origin
    ax.scatter(0, 0, 0, color='black', s=100, marker='x', label='Origin')

    # Plot each relation vector
    for rel, vec in pca_relation_vectors.items():
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                 length=1.0, normalize=True,
                 color='b' if 'and' not in rel else 'r',
                 alpha=0.7)
        ax.text(vec[0]*1.1, vec[1]*1.1, vec[2]*1.1, rel, fontsize=9)

    # Set labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Projection of Spatial Relation Vectors')

    plt.tight_layout()
    plt.show()

# -------------------------------
# Dimensional plane analysis
# -------------------------------
def analyze_by_planes(relation_vectors, pca_relation_vectors):
    """Analyze spatial relations in each 2D plane separately"""
    # Collect vectors for each plane
    planes = {
        "xy_plane": {},  # Horizontal-Depth plane
        "yz_plane": {},  # Depth-Vertical plane
        "xz_plane": {}   # Horizontal-Vertical plane
    }

    # Assign relation vectors to appropriate planes
    for rel, vec in pca_relation_vectors.items():
        # XY plane (horizontal and depth)
        planes["xy_plane"][rel] = vec[:2]  # First two components

        # YZ plane (depth and vertical)
        if len(vec) >= 3:
            planes["yz_plane"][rel] = vec[1:3]  # Second and third components
        else:
            planes["yz_plane"][rel] = np.array([vec[1], 0])  # Only the second component

        # XZ plane (horizontal and vertical)
        if len(vec) >= 3:
            planes["xz_plane"][rel] = np.array([vec[0], vec[2]])  # First and third components
        else:
            planes["xz_plane"][rel] = np.array([vec[0], 0])  # Only the first component

    # Define relevant compositions for each plane
    plane_compositions = {
        "xy_plane": [  # Horizontal-Depth plane
            ("diagonally_front_left", ["left", "in_front"]),
            ("diagonally_front_right", ["right", "in_front"]),
            ("diagonally_back_left", ["left", "behind"]),
            ("diagonally_back_right", ["right", "behind"])
        ],
        "yz_plane": [  # Depth-Vertical plane
            ("above_and_in_front", ["above", "in_front"]),
            ("above_and_behind", ["above", "behind"]),
            ("below_and_in_front", ["below", "in_front"]),
            ("below_and_behind", ["below", "behind"])
        ],
        "xz_plane": [  # Horizontal-Vertical plane
            ("above_and_left", ["above", "left"]),
            ("above_and_right", ["above", "right"]),
            ("below_and_left", ["below", "left"]),
            ("below_and_right", ["below", "right"])
        ]
    }

    # Analyze compositional properties in each plane
    results = {}
    for plane_name, plane_vectors in planes.items():
        print(f"\n=== COMPOSITIONAL ANALYSIS IN {plane_name.upper()} ===")

        compositions = plane_compositions[plane_name]
        plane_results = analyze_compositions_in_plane(plane_vectors, compositions)
        results[plane_name] = plane_results

        # Visualize this plane
        visualize_plane(plane_vectors, compositions, plane_name)

    return results

def analyze_compositions_in_plane(plane_vectors, compositions):
    """Analyze compositional properties within a 2D plane"""
    plane_results = []

    for complex_rel, component_rels in compositions:
        if complex_rel in plane_vectors and all(r in plane_vectors for r in component_rels):
            complex_vec = plane_vectors[complex_rel]

            # Sum the component vectors
            component_sum = sum(plane_vectors[r] for r in component_rels)

            # Compute cosine similarity
            cos_sim = cosine_similarity([complex_vec], [component_sum])[0][0]

            # Compute angle
            angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Compute Euclidean distance
            euclidean_dist = np.linalg.norm(complex_vec - component_sum)

            plane_results.append({
                "Complex Relation": complex_rel,
                "Component Relations": component_rels,
                "Cosine Similarity": cos_sim,
                "Angle (degrees)": angle_deg,
                "Euclidean Distance": euclidean_dist
            })

            print(f"Composition: {complex_rel} vs {'+'.join(component_rels)}")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  Angle: {angle_deg:.2f}°")
            print(f"  Euclidean distance: {euclidean_dist:.4f}")

    return plane_results

def visualize_plane(plane_vectors, compositions, plane_name):
    """Visualize spatial relations in a specific 2D plane"""
    plt.figure(figsize=(12, 10))

    # Plot origin
    plt.scatter(0, 0, color='black', s=100, marker='x', label='Origin')

    # Plot each vector in this plane
    for rel, vec in plane_vectors.items():
        # Color coding based on relation type
        if rel in ["above", "below", "left", "right", "in_front", "behind"]:
            color = 'blue'  # Basic relations
        elif rel.startswith("diagonally"):
            color = 'green'  # Diagonal relations
        elif "and" in rel:
            color = 'red'  # Compound relations
        else:
            color = 'gray'  # Other relations

        # Plot vector from origin
        plt.arrow(0, 0, vec[0], vec[1],
                 head_width=0.05, head_length=0.1,
                 fc=color, ec=color, alpha=0.7)
        plt.text(vec[0]*1.1, vec[1]*1.1, rel, fontsize=9)

    # Highlight compositions
    for complex_rel, component_rels in compositions:
        if complex_rel in plane_vectors and all(r in plane_vectors for r in component_rels):
            # Complex vector
            complex_vec = plane_vectors[complex_rel]

            # Sum of component vectors
            sum_vec = sum(plane_vectors[r] for r in component_rels)

            # Draw line connecting them
            plt.plot([complex_vec[0], sum_vec[0]], [complex_vec[1], sum_vec[1]],
                    'k--', alpha=0.5)

            # Mark the sum vector
            plt.scatter(sum_vec[0], sum_vec[1], color='purple', s=80, alpha=0.7)
            plt.text(sum_vec[0]*1.05, sum_vec[1]*1.05,
                    f"{'+'.join(component_rels)}", fontsize=8, color='purple')

    # Set labels based on plane
    if plane_name == "xy_plane":
        xlabel, ylabel = "X (Horizontal)", "Y (Depth)"
    elif plane_name == "yz_plane":
        xlabel, ylabel = "Y (Depth)", "Z (Vertical)"
    elif plane_name == "xz_plane":
        xlabel, ylabel = "X (Horizontal)", "Z (Vertical)"

    # Add grid, labels, title
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f"Spatial Relations in {plane_name}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main PCA Analysis
# -------------------------------
def run_pca_analysis(model_path="spatial_3d_probe.pt"):
    """Main function to run PCA analysis on the spatial relation vectors"""
    # Load linear probe model
    linear_probe, _ = load_saved_probe(model_path)

    # Extract grid vectors
    grid_vectors = extract_spatial_vectors(linear_probe)

    # Compute relation vectors
    relation_vectors = compute_spatial_relation_vectors(grid_vectors)

    print("=" * 50)
    print("PCA ANALYSIS OF SPATIAL RELATION VECTORS")
    print("=" * 50)

    # Analyze in PCA space with 3 components
    pca_vectors, pca_model = analyze_in_pca_space(relation_vectors, n_components=3)

    # Analyze in PCA space with 2 components (for clearer 2D visualization)
    pca_vectors_2d, pca_model_2d = analyze_in_pca_space(relation_vectors, n_components=2)

    # Analyze by dimensional planes
    plane_analysis_results = analyze_by_planes(relation_vectors, pca_vectors)

    return {
        "pca_vectors": pca_vectors,
        "pca_model": pca_model,
        "pca_vectors_2d": pca_vectors_2d,
        "pca_model_2d": pca_model_2d,
        "original_vectors": relation_vectors,
        "plane_analysis": plane_analysis_results
    }

if __name__ == "__main__":
    # If run directly, perform PCA analysis
    results = run_pca_analysis()
