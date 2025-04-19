import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.auto import tqdm
import time
import os

from data_grid import get_last_token_activation, parse_sentence_to_grid, OBJECTS

# -------------------------------
# Probe model implementation
# -------------------------------
class LinearSpatialProbe(nn.Module):
    def __init__(self, d_model, grid_size, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Simple linear layer directly mapping from embedding to 3D grid
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
        # Reshape to 3D grid
        out = x.view(batch_size, self.num_classes,
                     self.grid_size[2], self.grid_size[1], self.grid_size[0])
        return out

# -------------------------------
# Training and evaluation
# -------------------------------
def train_model(model, dataloader, num_epochs=50):
    """Training with cosine learning rate schedule and gradient clipping."""
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    # Cosine learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Use focal loss to help with class imbalance
    def focal_loss(preds, targets, gamma=2.0):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_activations, batch_grids, _ in tqdm(dataloader):
            batch_activations = batch_activations.cuda()
            batch_grids = batch_grids.cuda()

            # Forward pass
            predictions = model(batch_activations)

            # Reshape for loss calculation
            B, C, Z, Y, X = predictions.shape
            predictions_flat = predictions.view(B, C, -1)
            targets_flat = batch_grids.view(B, -1)

            # Calculate loss with object weighting
            loss = focal_loss(predictions_flat, targets_flat)

            # Backprop with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return model

def save_probe_model(model, d_model, grid_size, objects_map, save_path="spatial_3d_probe.pt"):
    """Save the probe model with metadata."""
    # Save model with metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'grid_size': grid_size,
        'num_objects': len(objects_map) + 1,
        'objects_map': objects_map,
        'timestamp': timestamp,
        'd_model': d_model,
        'hyperparams': {
            'epochs': 50,
            'batch_size': 32,
            'model_type': 'LinearSpatialProbe'
        }
    }

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def load_saved_probe(model_path):
    """Load a previously saved probe model with all its metadata."""
    saved_data = torch.load(model_path)

    # Initialize the model with saved parameters
    d_model = saved_data.get('d_model')
    grid_size = saved_data.get('grid_size')
    num_objects = saved_data.get('num_objects')

    # Create model instance
    probe = LinearSpatialProbe(d_model, grid_size, num_objects).cuda()

    # Load weights
    probe.load_state_dict(saved_data['model_state_dict'])

    print(f"Loaded model trained at {saved_data.get('timestamp')}")
    print(f"Model hyperparameters: {saved_data.get('hyperparams')}")

    return probe, saved_data

# -------------------------------
# Evaluation and Visualization
# -------------------------------
def evaluate_3d_probe(model, sentence, tokenizer, llm_model):
    """Use the trained probe to predict a 3D spatial grid from a sentence."""
    # Extract activation
    activation = get_last_token_activation(sentence, llm_model, tokenizer).unsqueeze(0).cuda()

    # Forward pass through the probe
    predicted_grid = model(activation)

    # Process prediction result
    predicted_grid = predicted_grid.cpu().detach().numpy()
    predicted_grid = np.argmax(predicted_grid, axis=1)

    return predicted_grid.squeeze(0)  # Remove batch dimension

def visualize_3d_objects(grid, title="3D Spatial relationship"):
    """Create enhanced 3D visualization of objects in the grid."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Find objects in the grid
    object_positions = {}
    for z in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[2]):
                obj_id = grid[z, y, x]
                if obj_id > 0:
                    obj_name = list(OBJECTS.keys())[list(OBJECTS.values()).index(obj_id)]
                    object_positions[obj_name] = (x, y, z)

    # Define colors for different objects
    colors = {
        'chair': 'red', 'table': 'blue', 'car': 'green', 'lamp': 'purple',
        'box': 'orange', 'book': 'brown', 'vase': 'pink',
        'plant': 'lime', 'computer': 'cyan', 'phone': 'magenta'
    }

    # Draw clear coordinate axes
    center = (grid.shape[2]//2, grid.shape[1]//2, grid.shape[0]//2)
    ax.plot([0, grid.shape[2]-1], [center[1], center[1]], [center[2], center[2]],
            'r-', linewidth=2, label="X axis (left-right)")
    ax.plot([center[0], center[0]], [0, grid.shape[1]-1], [center[2], center[2]],
            'g-', linewidth=2, label="Y axis (back-front)")
    ax.plot([center[0], center[0]], [center[1], center[1]], [0, grid.shape[0]-1],
            'b-', linewidth=2, label="Z axis (down-up)")

    # Draw objects with labels
    for obj_name, (x, y, z) in object_positions.items():
        ax.scatter(x, y, z, color=colors.get(obj_name, 'gray'),
                  s=500, label=obj_name, edgecolors='black', alpha=0.8)
        ax.text(x+0.3, y+0.3, z+0.3, obj_name, fontsize=12, weight='bold')

    # If there are exactly two objects, draw a line between them
    if len(object_positions) == 2:
        obj_names = list(object_positions.keys())
        pos1 = object_positions[obj_names[0]]
        pos2 = object_positions[obj_names[1]]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]],
                'k--', alpha=0.6, linewidth=2)

    # Add direction labels
    ax.text(grid.shape[2]-1, center[1], center[2], "right", color='red', fontsize=12)
    ax.text(0, center[1], center[2], "left", color='red', fontsize=12)
    ax.text(center[0], grid.shape[1]-1, center[2], "front", color='green', fontsize=12)
    ax.text(center[0], 0, center[2], "back", color='green', fontsize=12)
    ax.text(center[0], center[1], grid.shape[0]-1, "up", color='blue', fontsize=12)
    ax.text(center[0], center[1], 0, "down", color='blue', fontsize=12)

    # Emphasize grid center
    ax.scatter(*center, color='black', s=100, alpha=0.3, label="center")

    # Set axis labels and limits
    ax.set_xlabel('X axis (left → right)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y axis (back → front)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z axis (down → up)', fontsize=12, labelpad=10)

    # Set axis limits to show the entire grid
    ax.set_xlim(0, grid.shape[2]-1)
    ax.set_ylim(0, grid.shape[1]-1)
    ax.set_zlim(0, grid.shape[0]-1)

    # Add grid lines and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title(title, fontsize=14, fontweight='bold')

    # Show the figure
    plt.tight_layout()
    plt.show()

    # Print object positions for reference
    print("Object positions:", object_positions)

# -------------------------------
# Testing functions
# -------------------------------
def test_opposite_relationships(model, tokenizer, llm_model, save_plots=False):
    """Test counter spatial relationships to evaluate the model's understanding."""
    test_pairs = [
        # Left-right pair
        ("The chair is to the left of the table.", "The chair is to the right of the table."),
        # Above-below pair
        ("The lamp is above the box.", "The lamp is below the box."),
        # Front-back pair
        ("The car is in front of the plant.", "The car is behind the plant."),
        # Diagonal relationships
        ("The book is diagonally front-left of the vase.", "The book is diagonally back-right of the vase."),
        # Compound relationship tests
        ("The lamp is above and behind the box.", "The lamp is below and in front of the box.")
    ]

    for sentence1, sentence2 in test_pairs:
        print(f"Testing counter spatial relationship:")
        print(f"  1. {sentence1}")
        print(f"  2. {sentence2}")

        # Get predictions for both sentences
        grid1 = evaluate_3d_probe(model, sentence1, tokenizer, llm_model)
        grid2 = evaluate_3d_probe(model, sentence2, tokenizer, llm_model)

        # Extract object positions
        positions1 = {}
        positions2 = {}

        for z in range(grid1.shape[0]):
            for y in range(grid1.shape[1]):
                for x in range(grid1.shape[2]):
                    if grid1[z, y, x] > 0:
                        obj_id = grid1[z, y, x]
                        # Ensure index is within valid range
                        if obj_id <= len(list(OBJECTS.keys())):
                            obj_name = list(OBJECTS.keys())[list(OBJECTS.values()).index(obj_id)]
                            positions1[obj_name] = (x, y, z)

                    if grid2[z, y, x] > 0:
                        obj_id = grid2[z, y, x]
                        # Ensure index is within valid range
                        if obj_id <= len(list(OBJECTS.keys())):
                            obj_name = list(OBJECTS.keys())[list(OBJECTS.values()).index(obj_id)]
                            positions2[obj_name] = (x, y, z)

        # Analyze relative positions
        print(f"  Position in first sentence: {positions1}")
        print(f"  Position in second sentence: {positions2}")

        # Check if both objects exist in both grids
        if len(positions1) == 2 and len(positions2) == 2:
            print("  Both objects exist ✓")
        else:
            print("  Missing object ✗")

            # Display object IDs for debugging
            print("  Object 1 IDs:")
            unique_ids1 = set()
            for z in range(grid1.shape[0]):
                for y in range(grid1.shape[1]):
                    for x in range(grid1.shape[2]):
                        if grid1[z, y, x] > 0:
                            unique_ids1.add(grid1[z, y, x])
            print(f"  {unique_ids1}")

            print("  Object 2 IDs:")
            unique_ids2 = set()
            for z in range(grid2.shape[0]):
                for y in range(grid2.shape[1]):
                    for x in range(grid2.shape[2]):
                        if grid2[z, y, x] > 0:
                            unique_ids2.add(grid2[z, y, x])
            print(f"  {unique_ids2}")

        # Display 3D visualizations
        if save_plots:
            plt.figure(figsize=(12, 10))
            plt.savefig(f"test_1_{sentence1[:20]}.png")
            plt.close()

            plt.figure(figsize=(12, 10))
            plt.savefig(f"test_2_{sentence2[:20]}.png")
            plt.close()
        else:
            visualize_3d_objects(grid1, title=f"3D visualization: {sentence1}")
            visualize_3d_objects(grid2, title=f"3D visualization: {sentence2}")

        print("-" * 80)

def run_comprehensive_tests(model, tokenizer, llm_model):
    """Enhanced test suite with better validation and systematic organization"""

    # Organize tests by category for better coverage
    test_categories = {
        "Basic Orthogonal": [
            "The chair is above the table.",
            "The lamp is below the box.",
            "The car is to the right of the chair.",
            "The book is to the left of the vase."
        ],
        "Depth Relations": [
            "The plant is behind the computer.",
            "The phone is in front of the lamp."
        ],
        "Diagonal Relations": [
            "The book is diagonally front-left of the vase.",
            "The computer is diagonally back-right of the plant.",
            "The lamp is diagonally front-right of the box.",
            "The chair is diagonally back-left of the table.",
            # Add variations to test robustness
            "The vase is to the left diagonally in front of the plant.",
            "The car is diagonally to the right and behind the chair."
        ],
        "Compound Relations": [
            "The chair is above and to the right of the table.",
            "The lamp is below and behind the box.",
            "The phone is above and in front of the box.",
            "The vase is below and to the left of the table."
        ],
    }

    # Track success metrics
    results = {category: {"total": 0, "successful": 0} for category in test_categories}
    detailed_failures = []

    for category, tests in test_categories.items():
        print(f"\n==== TESTING CATEGORY: {category} ====\n")

        for test in tests:
            print(f"Test: {test}")
            grid = evaluate_3d_probe(model, test, tokenizer, llm_model)

            # Extract objects from the sentence
            sentence_objects = [obj for obj in OBJECTS.keys() if obj in test.lower()]

            # Check if we found the expected objects in the grid
            grid_objects = {}
            for z in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    for x in range(grid.shape[2]):
                        obj_id = grid[z, y, x]
                        if obj_id > 0 and obj_id <= len(OBJECTS):
                            obj_name = list(OBJECTS.keys())[list(OBJECTS.values()).index(obj_id)]
                            grid_objects[obj_name] = (x, y, z)

            # Validate results
            results[category]["total"] += 1

            # Success criteria: all mentioned objects appear and aren't at center
            center = (grid.shape[2]//2, grid.shape[1]//2, grid.shape[0]//2)
            center_pos = (center[0], center[1], center[2])

            success = True
            # Check that we found all expected objects
            for obj in sentence_objects:
                if obj not in grid_objects:
                    print(f"  ❌ Object '{obj}' not found in grid")
                    success = False

            # Check that objects aren't all at center position
            positions = list(grid_objects.values())
            if len(positions) > 1 and positions.count(center_pos) > 1:
                print(f"  ❌ Multiple objects at center position")
                success = False

            # Check if relation was applied (basic check - objects have different positions)
            if len(grid_objects) > 1 and len(set(positions)) < len(positions):
                print(f"  ❌ Some objects share the same position")
                success = False

            if success:
                print(f"  ✅ Test passed")
                results[category]["successful"] += 1
            else:
                detailed_failures.append({
                    "category": category,
                    "test": test,
                    "objects_found": grid_objects,
                    "expected_objects": sentence_objects
                })

            # Report object positions
            print(f"  Object positions: {grid_objects}")

            # Visualize (optional - can comment out to run tests faster)
            visualize_3d_objects(grid, title=f"3D Visualization: {test}")
            print("-" * 50)

    # Print summary report
    print("\n==== TEST SUMMARY ====")
    for category, stats in results.items():
        success_rate = (stats["successful"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{category}: {stats['successful']}/{stats['total']} passed ({success_rate:.1f}%)")

    print(f"\nOverall: {sum(r['successful'] for r in results.values())}/{sum(r['total'] for r in results.values())} " +
          f"({sum(r['successful'] for r in results.values())/sum(r['total'] for r in results.values())*100:.1f}%)")

    # Print detailed failure report if needed
    if detailed_failures:
        print("\n==== DETAILED FAILURES ====")
        for failure in detailed_failures:
            print(f"Category: {failure['category']}")
            print(f"Test: {failure['test']}")
            print(f"Found: {failure['objects_found']}")
            print(f"Expected objects: {failure['expected_objects']}")
            print("-" * 30)

    return results
