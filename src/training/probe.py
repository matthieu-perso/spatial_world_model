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



# ============================
# class: probe train arguments
# ============================
@dataclass
class ProbeTrainingArgs:
    verbose: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # layer information
    layer_name: str = "layer_8"

    # spatial relation state
    options: int = 4
    d_model: int = 3072

    # standard training hyperparams
    epochs: int = 10

    # hyperparams for optimizer
    lr: float = 1e-3

    # saving & logging
    use_wandb: bool = False
    wandb_project: str | None = "spatial-relation-probe"
    wandb_name: str | None = None

    # cpde to get randomly initialized probe
    def setup_probe(self) -> torch.Tensor:
        linear_probe = torch.randn(self.d_model, self.options, device=self.device, dtype=torch.float32) / np.sqrt(self.d_model)
        linear_probe.requires_grad = True
        print(f"shape of linear probe is: d_model = {linear_probe.shape[0]} and options = {linear_probe.shape[-1]}" if self.verbose else "")
        return linear_probe


# ===========================
# class: Linear probe trainer
# ===========================
class LinearProbeTrainer:
    def __init__(self, args: ProbeTrainingArgs, dataloader: torch.utils.data.DataLoader):
        self.args = args
        self.linear_probe = args.setup_probe()
        self.dataloader = dataloader

    def train(self):
        if self.args.verbose:
            print(f"\ntraining a linear probe for spatial relations of layer {self.args.layer_name} for {self.args.epochs} epochs ...\n")
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

        # define optimizer
        optimizer = torch.optim.Adam(
            [self.linear_probe], lr=self.args.lr
        )

        # define loss criterion
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}"):

                # move data to device
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

                # get probe output
                probe_preds = einops.einsum(
                    inputs,
                    self.linear_probe,
                    "batch d_model, d_model options -> batch options",
                    )

                # compute loss
                loss = criterion(probe_preds, targets)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # update loss and accuracy
                total_loss += loss.item()
                _, predicted = torch.max(probe_preds.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # print epoch results
            avg_loss = total_loss / len(self.dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{self.args.epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")


        if self.args.use_wandb:
            wandb.finish()


# ========================
# function: evaluate probe
# ========================
def evaluate_probe(linear_probe: torch.Tensor, dataloader: DataLoader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    evaluate a linear probe model
    """
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for inputs, targets in dataloader:

            # move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # get linear probe preds
            probe_preds = einops.einsum(
                inputs,
                linear_probe,
                "batch d_model, d_model options -> batch options",
                )
            _, predicted = torch.max(probe_preds, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=1)

    return accuracy, report

# ========================
# function: Grid Linear Probe
# ========================

# Add Grid Linear Probe class
class GridLinearProbe(nn.Module):
    """Linear probe for 3D spatial grid analysis."""
    
    def __init__(self, d_model, grid_size, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        # Linear layer mapping from embedding to 3D grid
        self.linear = nn.Linear(
            d_model,
            grid_size[0] * grid_size[1] * grid_size[2] * num_classes
        )
        
        # Initialize weights properly
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.linear(x)
        # Reshape to 3D grid: [batch, classes, z, y, x]
        out = x.view(batch_size, self.num_classes,
                    self.grid_size[2], self.grid_size[1], self.grid_size[0])
        return out
    
    def extract_spatial_vectors(self):
        """Extract spatial vectors from the model weights."""
        return self.linear.weight.data.view(
            self.num_classes, self.grid_size[2], self.grid_size[1], self.grid_size[0], -1
        )

# Add training function for grid probe
def train_grid_probe(grid_probe, dataloader, device, epochs=30, lr=1e-3):
    """Train a grid-based linear probe.
    
    Args:
        grid_probe: GridLinearProbe instance
        dataloader: DataLoader with (embeddings, grid_labels) pairs
        device: Torch device for training
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained grid probe
    """
    grid_probe = grid_probe.to(device)
    optimizer = torch.optim.Adam(grid_probe.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, grid_labels in dataloader:
            inputs, grid_labels = inputs.to(device), grid_labels.to(device)
            
            # Forward pass
            outputs = grid_probe(inputs)
            
            # Reshape for loss calculation
            B, C, Z, Y, X = outputs.shape
            outputs_flat = outputs.view(B, C, -1)
            targets_flat = grid_labels.view(B, -1)
            
            # Calculate loss
            loss = criterion(outputs_flat, targets_flat)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.detach(), dim=1)
            correct += (predicted == grid_labels).sum().item()
            total += grid_labels.numel()
        
        # Print epoch results
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return grid_probe

# Add evaluation function
def evaluate_grid_probe(grid_probe, dataloader, device):
    """Evaluate a trained grid probe.
    
    Args:
        grid_probe: Trained GridLinearProbe instance
        dataloader: DataLoader with (embeddings, grid_labels) pairs
        device: Torch device for evaluation
        
    Returns:
        Tuple of (accuracy, loss)
    """
    grid_probe.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, grid_labels in dataloader:
            inputs, grid_labels = inputs.to(device), grid_labels.to(device)
            
            # Forward pass
            outputs = grid_probe(inputs)
            
            # Reshape for loss calculation
            B, C, Z, Y, X = outputs.shape
            outputs_flat = outputs.view(B, C, -1)
            targets_flat = grid_labels.view(B, -1)
            
            # Calculate loss
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.detach(), dim=1)
            correct += (predicted == grid_labels).sum().item()
            total += grid_labels.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return accuracy, avg_loss