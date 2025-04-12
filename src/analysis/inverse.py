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


# =========================================================================
# function: check inverse relation between linear probe's binary directions
# =========================================================================
def check_inverse_relations(directions: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Check inverse relations between binary spatial relations.

    For each pair of relations (e.g., "above" and "below") in the directions dictionary,
    this function computes:
      - Cosine similarity between the first relation and the negative of the second,
      - Euclidean distance between their vectors,
      - Angle (in degrees) between the first vector and the negative of the second.

    It then prints out the cosine similarity and angle for each inverse pair.
    Additionally, the function performs PCA to reduce the directions to 2 dimensions,
    plots the PCA projections, computes the inverse metrics in PCA space, and plots the
    decision boundaries (assuming a simple linear classifier with zero bias) for each inverse pair.

    Parameters:
        directions (Dict[str, np.ndarray]): A dictionary mapping relation names to their direction vectors.

    Returns:
        pd.DataFrame: DataFrame containing computed metrics for each inverse pair.
    """

    # Define pairs of opposite spatial relations
    binary_spatial_relations = [
        ("above", "below"),
        ("to the left of", "to the right of"),
    ]

    results = []
    for rel1, rel2 in binary_spatial_relations:
        if rel1 in directions and rel2 in directions:
            vec1 = directions[rel1]
            vec2 = directions[rel2]

            # Compute cosine similarity between vec1 and the negative of vec2.
            cosine_sim = cosine_similarity([vec1], [-vec2])[0][0]

            # Compute Euclidean distance between vec1 and vec2.
            euclidean_dist = np.linalg.norm(vec1 - vec2)

            # Compute angle in degrees between vec1 and the negative of vec2.
            angle_rad = np.arccos(np.clip(cosine_sim, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            results.append({
                'Relation 1': rel1,
                'Relation 2 (Inverse)': rel2,
                'Cosine Similarity': cosine_sim,
                'Euclidean Distance': euclidean_dist,
                'Angle (°)': angle_deg
            })
            # Prompt (print) the metrics for each relation pair individually:
            print(f"Inverse Pair: '{rel1}' and '{rel2}'")
            print(f"  Cosine Similarity (between {rel1} and -{rel2}): {cosine_sim:.4f}")
            print(f"  Angle: {angle_deg:.2f}°\n")

    # Create DataFrame for easy analysis
    df = pd.DataFrame(results)

    if not df.empty:
        # Also print summary statistics (if desired)
        print("=== Inverse Spatial Relations Summary ===")
        print(f"Average Cosine Similarity: {df['Cosine Similarity'].mean():.4f}")
        print(f"Average Euclidean Distance: {df['Euclidean Distance'].mean():.4f}")
        print(f"Average Angle (°): {df['Angle (°)'].mean():.2f}°\n")

        # Sort by angle (higher angle means less aligned when compared to the negative)
        df_sorted = df.sort_values('Angle (°)', ascending=False)
        print("Inverse relation pairs sorted by angle:")
        print(df_sorted[['Relation 1', 'Relation 2 (Inverse)', 'Angle (°)']].to_string(index=False))

    # === PCA Visualization Section ===
    relation_names = list(directions.keys())
    W = np.stack([directions[rel] for rel in relation_names], axis=0)

    pca = PCA(n_components=2)
    W_2d = pca.fit_transform(W)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, rel in enumerate(relation_names):
        x, y = W_2d[i]
        plt.scatter(x, y, color=colors[i % len(colors)], s=100, label=rel)
        plt.text(x + 0.02, y + 0.02, rel, fontsize=12)

    plt.title("PCA Projection of Spatial Relation Probes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === PCA Inverse Metrics ===
    print("\n=== PCA Space Inverse Metrics ===")
    for rel1, rel2 in binary_spatial_relations:
        if rel1 in directions and rel2 in directions:
            i1 = relation_names.index(rel1)
            i2 = relation_names.index(rel2)
            vec1_2d = W_2d[i1]
            vec2_2d = W_2d[i2]
            # Compute cosine similarity between vec1_2d and the negative of vec2_2d.
            cosine_sim_2d = cosine_similarity([vec1_2d], [-vec2_2d])[0][0]
            angle_rad_2d = np.arccos(np.clip(cosine_sim_2d, -1.0, 1.0))
            angle_deg_2d = np.degrees(angle_rad_2d)
            print(f"PCA Inverse Pair: '{rel1}' and '{rel2}'")
            print(f"  PCA Cosine Similarity (between {rel1} and -{rel2}): {cosine_sim_2d:.4f}")
            print(f"  PCA Angle: {angle_deg_2d:.2f}°\n")

            # Plot decision boundaries for this inverse pair in PCA space.
            # We assume a simple linear decision boundary: w^T x = 0, with zero bias.
            # For relation rel1, the decision function is f(x) = w_pca^T x.
            # Its boundary is the line where f(x)=0.
            # For illustration, we plot the decision boundary for rel1 and for -w_pca of rel2.

            # Get the PCA weight vectors:
            w_rel1 = vec1_2d  # for rel1
            w_rel2 = vec2_2d  # for rel2
            # For the inverse, we consider -w_rel2.

            # Create a grid in PCA space:
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]

            # Compute decision values: f(x) = w^T x for rel1, and for -w_rel2.
            decision_rel1 = grid.dot(w_rel1)
            decision_inv_rel2 = grid.dot(-w_rel2)

            decision_rel1 = decision_rel1.reshape(xx.shape)
            decision_inv_rel2 = decision_inv_rel2.reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, decision_rel1 > 0, alpha=0.3, cmap=plt.cm.Paired)
            plt.contour(xx, yy, decision_rel1, levels=[0], colors='k', linewidths=2)
            plt.title(f"Decision Boundary for '{rel1}' in PCA Space")
            for i, rel in enumerate(relation_names):
                x, y = W_2d[i]
                plt.scatter(x, y, color=colors[i % len(colors)], s=100, label=rel)
                plt.text(x + 0.02, y + 0.02, rel, fontsize=12)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, decision_inv_rel2 > 0, alpha=0.3, cmap=plt.cm.Paired)
            plt.contour(xx, yy, decision_inv_rel2, levels=[0], colors='k', linewidths=2)
            plt.title(f"Decision Boundary for Inverse of '{rel2}' (i.e. -w) in PCA Space")
            for i, rel in enumerate(relation_names):
                x, y = W_2d[i]
                plt.scatter(x, y, color=colors[i % len(colors)], s=100, label=rel)
                plt.text(x + 0.02, y + 0.02, rel, fontsize=12)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            plt.show()


    return df