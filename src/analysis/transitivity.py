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





def check_transitivity_relations(
    composition_relations: Dict[str, Tuple[str, str]],
    directions_rel_1: Dict[str, np.ndarray],
    directions_rel_2: Dict[str, np.ndarray],
    directions_direct: Dict[str, np.ndarray],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Check the transitivity (composition) assumption using separate dictionaries for the two atomic relations
    and the direct (compositional) relation.

    For each direct relation in the composition_relations dict, we assume:

         v_direct ≈ v_atomic1 + v_atomic2

    where:
      - v_atomic1 is obtained from directions_rel_1 using the key corresponding to atomic relation 1,
      - v_atomic2 is obtained from directions_rel_2 using the key corresponding to atomic relation 2,
      - v_direct is obtained from directions_direct using the direct relation key.

    The function computes the following metrics in both the original high-dimensional space and in a 2D PCA-reduced space:
      - Cosine similarity,
      - Euclidean difference,
      - Angle (in degrees)

    between the direct vector and the composed vector (sum of the atomic vectors).

    It also produces a PCA plot with arrows for:
       - atomic relation 1 vector,
       - atomic relation 2 vector,
       - the composed vector (atomic1 + atomic2),
       - and the direct relation vector.

    Parameters:
       composition_relations (Dict[str, Tuple[str, str]]): A dictionary mapping a direct relation (e.g.,
             "diagonally above and to the right of") to a tuple of atomic relations (e.g., ("above", "to the right of")).
       directions_rel_1 (Dict[str, np.ndarray]): Dictionary mapping atomic relation names (e.g., "above", "below") to their learned vectors.
       directions_rel_2 (Dict[str, np.ndarray]): Dictionary mapping atomic relation names (e.g., "to the right of", "to the left of") to their learned vectors.
       directions_direct (Dict[str, np.ndarray]): Dictionary mapping direct relation names (composed, e.g.,
             "diagonally above and to the right of") to their learned vectors.
       verbose (bool): If True, prints detailed metrics.

    Returns:
       pd.DataFrame: A DataFrame with computed metrics for each direct (compositional) relation.
    """
    results = []

    for direct_rel, (atomic1, atomic2) in composition_relations.items():
        # Check that the required keys exist in the appropriate dictionaries.
        if atomic1 not in directions_rel_1:
            print(f"Skipping '{direct_rel}': '{atomic1}' not found in directions_rel_1.")
            continue
        if atomic2 not in directions_rel_2:
            print(f"Skipping '{direct_rel}': '{atomic2}' not found in directions_rel_2.")
            continue
        if direct_rel not in directions_direct:
            print(f"Skipping '{direct_rel}': not found in directions_direct.")
            continue

        vec_atomic1 = directions_rel_1[atomic1]
        vec_atomic2 = directions_rel_2[atomic2]
        vec_direct = directions_direct[direct_rel]

        # Compose the atomic vectors.
        vec_composed = vec_atomic1 + vec_atomic2

        # --- High-Dimensional Metrics ---
        norm_direct = vec_direct / np.linalg.norm(vec_direct)
        norm_composed = vec_composed / np.linalg.norm(vec_composed)
        cosine_sim_orig = np.dot(norm_direct, norm_composed)
        euclidean_diff_orig = np.linalg.norm(vec_direct - vec_composed)
        angle_rad_orig = np.arccos(np.clip(cosine_sim_orig, -1.0, 1.0))
        angle_deg_orig = np.degrees(angle_rad_orig)

        # --- PCA Space Analysis ---
        # Project the three atomic vectors, the direct vector, and the composed vector.
        all_vectors = np.stack([vec_atomic1, vec_atomic2, vec_direct, vec_composed], axis=0)
        pca = PCA(n_components=2)
        all_vectors_2d = pca.fit_transform(all_vectors)
        proj_atomic1 = all_vectors_2d[0]
        proj_atomic2 = all_vectors_2d[1]
        proj_direct = all_vectors_2d[2]
        proj_composed = all_vectors_2d[3]

        norm_proj_direct = proj_direct / np.linalg.norm(proj_direct)
        norm_proj_composed = proj_composed / np.linalg.norm(proj_composed)
        cosine_sim_pca = np.dot(norm_proj_direct, norm_proj_composed)
        euclidean_diff_pca = np.linalg.norm(proj_direct - proj_composed)
        angle_rad_pca = np.arccos(np.clip(cosine_sim_pca, -1.0, 1.0))
        angle_deg_pca = np.degrees(angle_rad_pca)

        # Append metrics for this direct relation.
        results.append({
            "Direct Relation": direct_rel,
            "Atomic Relation 1": atomic1,
            "Atomic Relation 2": atomic2,
            "Cosine Similarity (Original)": cosine_sim_orig,
            "Euclidean Diff (Original)": euclidean_diff_orig,
            "Angle (Original, °)": angle_deg_orig,
            "Cosine Similarity (PCA)": cosine_sim_pca,
            "Euclidean Diff (PCA)": euclidean_diff_pca,
            "Angle (PCA, °)": angle_deg_pca
        })

        if verbose:
            print(f"--- Transitivity for '{direct_rel}' ---")
            print(f"Atomic relations: '{atomic1}' and '{atomic2}'")
            print(f"High-Dimensional: Cosine Sim = {cosine_sim_orig:.4f}, Euclidean Diff = {euclidean_diff_orig:.4f}, Angle = {angle_deg_orig:.2f}°")
            print(f"PCA Space: Cosine Sim = {cosine_sim_pca:.4f}, Euclidean Diff = {euclidean_diff_pca:.4f}, Angle = {angle_deg_pca:.2f}°\n")

            # --- Plotting in PCA Space ---
            plt.figure(figsize=(8, 6))
            labels = [atomic1, atomic2, "direct ("+direct_rel+")", "composed ("+atomic1+"+"+atomic2+")"]
            colors = ["blue", "green", "red", "purple"]
            for j, lab in enumerate(labels):
                x, y = all_vectors_2d[j]
                plt.scatter(x, y, color=colors[j], s=100, label=lab)
                plt.text(x+0.05, y+0.05, lab, fontsize=12)

            # Draw arrows from the origin.
            origin = np.array([0,0])
            for j, col in enumerate(colors):
                plt.arrow(origin[0], origin[1], all_vectors_2d[j, 0], all_vectors_2d[j, 1],
                          head_width=0.1, color=col, alpha=0.8)

            plt.title(f"PCA Projection for '{direct_rel}'")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.legend()
            plt.grid(True)
            plt.show()

    df_metrics = pd.DataFrame(results)
    return df_metrics
