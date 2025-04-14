import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import time
import torch

from dataclasses import dataclass
from itertools import permutations
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import Dict, List, Tuple





# =======================================
# function: check compositional relations
# =======================================
def check_compositional_relations(directions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Check compositional relations among spatial relation probes in both the original high-dimensional space
    and in a 2D PCA-reduced space.

    We assume that certain compositional relations should be captured by the sum of two atomic relations.
    For example, we expect:
        "diagonally above and to the right of" ≈ "above" + "to the right of"

    For each compositional relation defined in composition_pairs, this function computes:
      - In the original space:
          * Cosine similarity between the direct compositional vector and the sum of the atomic vectors,
          * Euclidean difference,
          * Angle (in degrees) between these vectors.
      - In the PCA space:
          * The same metrics computed on the 2D projections.

    Additionally, the function plots the PCA projections and draws arrows for the atomic vectors,
    their sum (the composed vector), and the direct compositional relation.

    Parameters:
        directions (Dict[str, np.ndarray]): A dictionary mapping relation names to their d_model-dimensional vectors.

    Returns:
        pd.DataFrame: DataFrame containing the computed metrics for each compositional relation pair.
    """
    # Define expected compositional pairs.
    # Format: "compositional_relation": (atomic_relation1, atomic_relation2)

    composition_pairs = {
        "diagonally above and to the right of": ("above", "to the right of"),
        "diagonally above and to the left of": ("above", "to the left of"),
        "diagonally below and to the right of": ("below", "to the right of"),
        "diagonally below and to the left of": ("below", "to the left of"),
    }

    results = []

    # ---- Original High-Dimensional Space Analysis ----
    for comp_rel, (atomic1, atomic2) in composition_pairs.items():
        if comp_rel in directions and atomic1 in directions and atomic2 in directions:
            direct_vec = directions[comp_rel]
            composed_vec = directions[atomic1] + directions[atomic2]

            vec1 = directions[atomic1]
            vec2 = directions[atomic2]

            # compute cosine similarity between direct_vc and composed_vec
            cosine_sim = cosine_similarity([direct_vec], [composed_vec])[0][0]

            # Compute Euclidean distance between vec1 and vec2.
            euclidean_dist = np.linalg.norm(direct_vec - composed_vec)

            # Compute angle in degrees between vec1 and the negative of vec2.
            angle_rad = np.arccos(np.clip(cosine_sim, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            results.append({
                'Compositional Relation': comp_rel,
                'Atomic Relation 1': atomic1,
                'Atomic Relation 2': atomic2,
                'Cosine Similarity (Original)': cosine_sim,
                'Euclidean Diff (Original)': euclidean_dist,
                'Angle (Original, °)': angle_deg
            })

            print(f"Original Space for '{comp_rel}':")
            print(f"  Expected from '{atomic1}' + '{atomic2}'")
            print(f"  Cosine Similarity: {cosine_sim:.4f}")
            print(f"  Euclidean Difference: {euclidean_dist:.4f}")
            print(f"  Angle: {angle_deg:.2f}°\n")

    # ---- PCA Space Analysis ----
    relation_names = list(directions.keys())
    W = np.stack([directions[rel] for rel in relation_names], axis=0)

    # Perform PCA on all relation vectors
    pca = PCA(n_components=2)
    W_2d = pca.fit_transform(W)

    # Create a mapping from relation name to its 2D PCA projection.
    directions_2d = {rel: vec for rel, vec in zip(relation_names, W_2d)}

    # Plot PCA projections for context.
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, rel in enumerate(relation_names):
        x, y = directions_2d[rel]
        plt.scatter(x, y, color=colors[i % len(colors)], s=100, label=rel)
        plt.text(x + 0.02, y + 0.02, rel, fontsize=12)
    plt.title("PCA Projection of Spatial Relation Probes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Now compute compositional metrics in PCA space.
    for comp_rel, (atomic1, atomic2) in composition_pairs.items():
        if comp_rel in directions_2d and atomic1 in directions_2d and atomic2 in directions_2d:
            direct_vec_2d = directions_2d[comp_rel]
            composed_vec_2d = directions_2d[atomic1] + directions_2d[atomic2]

            cosine_sim_pca = cosine_similarity([direct_vec_2d], [composed_vec_2d])[0][0]
            euclidean_diff_pca = np.linalg.norm(direct_vec_2d - composed_vec_2d)
            angle_rad_pca = np.arccos(np.clip(cosine_sim_pca, -1.0, 1.0))
            angle_deg_pca = np.degrees(angle_rad_pca)

            # Append PCA metrics to our results.
            results[-1].update({
                'Cosine Similarity (PCA)': cosine_sim_pca,
                'Euclidean Diff (PCA)': euclidean_diff_pca,
                'Angle (PCA, °)': angle_deg_pca
            })
            print(f"PCA Space for '{comp_rel}':")
            print(f"  Expected from '{atomic1}' + '{atomic2}'")
            print(f"  PCA Cosine Similarity: {cosine_sim_pca:.4f}")
            print(f"  PCA Euclidean Difference: {euclidean_diff_pca:.4f}")
            print(f"  PCA Angle: {angle_deg_pca:.2f}°\n")

            # Plot arrows to visualize composition in PCA space.
            plt.figure(figsize=(8, 6))
            # Plot atomic relation points.
            plt.scatter(directions_2d[atomic1][0], directions_2d[atomic1][1], color='blue', s=100, label=atomic1)


            plt.scatter(directions_2d[atomic2][0], directions_2d[atomic2][1], color='green', s=100, label=atomic2)
            # Plot direct compositional relation.
            plt.scatter(direct_vec_2d[0], direct_vec_2d[1], color='red', s=100, label=comp_rel)
            # Plot composed vector (sum of atomic vectors).
            plt.scatter(composed_vec_2d[0], composed_vec_2d[1], color='purple', s=100, label=f"Composed ({atomic1}+{atomic2})")
            # Draw arrows from origin.
            origin = np.array([0, 0])
            plt.arrow(origin[0], origin[1], directions_2d[atomic1][0], directions_2d[atomic1][1],
                      head_width=0.1, color='blue', alpha=0.5)
            plt.arrow(origin[0], origin[1], directions_2d[atomic2][0], directions_2d[atomic2][1],
                      head_width=0.1, color='green', alpha=0.5)
            plt.arrow(origin[0], origin[1], composed_vec_2d[0], composed_vec_2d[1],
                      head_width=0.1, color='purple', alpha=0.7)
            plt.arrow(origin[0], origin[1], direct_vec_2d[0], direct_vec_2d[1],
                      head_width=0.1, color='red', alpha=0.7)
            plt.title(f"PCA Composition: {comp_rel} vs. {atomic1}+{atomic2}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            plt.grid(True)
            plt.show()

    df = pd.DataFrame(results)
    return df

# =======================================
# Grid-based Compositionality
# =======================================
# Add Grid-based Compositionality Analysis functions
def check_compositional_relations_grid(grid_vectors, composition_relations=None, spatial_map=None, center=None):
    """Check compositional relationships using grid vectors.
    
    Args:
        grid_vectors: 5D tensor of shape [classes, z, y, x, features]
        composition_relations: Dictionary mapping complex relations to component relation pairs
        spatial_map: Dictionary mapping relation names to (dx, dy, dz) offsets
        center: Optional tuple specifying the grid center coordinates
    
    Returns:
        pd.DataFrame: Compositional analysis results
    """
    # Extract relation vectors from grid
    relation_vectors = compute_relation_vectors_from_grid(grid_vectors, spatial_map, center)
    
    # Default composition relations if not provided
    if composition_relations is None:
        composition_relations = {
            "diagonally_front_left": ("left", "in_front"),
            "diagonally_front_right": ("right", "in_front"),
            "diagonally_back_left": ("left", "behind"),
            "diagonally_back_right": ("right", "behind"),
            "above_and_left": ("above", "left"),
            "above_and_right": ("above", "right"),
            "below_and_left": ("below", "left"),
            "below_and_right": ("below", "right"),
            "above_and_in_front": ("above", "in_front"),
            "above_and_behind": ("above", "behind"),
            "below_and_in_front": ("below", "in_front"),
            "below_and_behind": ("below", "behind"),
        }
    
    # Analyze compositions
    results = []
    
    for complex_rel, (rel1, rel2) in composition_relations.items():
        if complex_rel in relation_vectors and rel1 in relation_vectors and rel2 in relation_vectors:
            complex_vec = relation_vectors[complex_rel]
            component_sum = relation_vectors[rel1] + relation_vectors[rel2]
            
            # Calculate metrics similar to original function
            # Cosine similarity
            norm_complex = complex_vec / torch.norm(complex_vec)
            norm_sum = component_sum / torch.norm(component_sum)
            cosine_sim = torch.dot(norm_complex, norm_sum).item()
            
            # Euclidean distance
            euclidean_diff = torch.norm(complex_vec - component_sum).item()
            
            # Angle
            angle_rad = torch.acos(torch.clamp(torch.tensor(cosine_sim), -1.0, 1.0))
            angle_deg = angle_rad.item() * 180 / np.pi
            
            results.append({
                "Direct Relation": complex_rel,
                "Atomic Relation 1": rel1,
                "Atomic Relation 2": rel2,
                "Cosine Similarity (Original)": cosine_sim,
                "Euclidean Diff (Original)": euclidean_diff,
                "Angle (Original, °)": angle_deg
            })
    
    # Create DataFrame
    df_metrics = pd.DataFrame(results)
    return df_metrics

def compute_relation_vectors_from_grid(grid_vectors, spatial_map, center=None):
    """Extract relation vectors from grid representation.
    
    Args:
        grid_vectors: 5D tensor [classes, z, y, x, features]
        spatial_map: Dictionary mapping relation names to (dx, dy, dz) offsets
        center: Optional center coordinates (default is middle of grid)
        
    Returns:
        Dictionary mapping relation names to their vector representations
    """
    if center is None:
        # Use the center of the grid
        center = (grid_vectors.shape[3]//2, grid_vectors.shape[2]//2, grid_vectors.shape[1]//2)

    # Extract the center vector (reference point)
    center_x, center_y, center_z = center
    center_vector = grid_vectors[0, center_z, center_y, center_x, :]  # Using class 0 (background)

    # Compute relation vectors for each spatial relation
    relation_vectors = {}
    for relation, (dx, dy, dz) in spatial_map.items():
        # Calculate target position
        target_x = center_x + dx
        target_y = center_y + dy
        target_z = center_z + dz

        # Check if position is within grid bounds
        if (0 <= target_x < grid_vectors.shape[3] and
            0 <= target_y < grid_vectors.shape[2] and
            0 <= target_z < grid_vectors.shape[1]):

            # Extract vector for target position
            target_vector = grid_vectors[0, target_z, target_y, target_x, :]

            # Compute relation vector (target - center)
            relation_vectors[relation] = target_vector - center_vector

    return relation_vectors