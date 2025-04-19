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
