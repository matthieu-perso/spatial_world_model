# ==================
# function: set seed
# ==================

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)

    # ensures deterministic operations on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark mode


# ==================
# function: Grid Utility
# ==================
# Add Grid Utility functions
def visualize_3d_grid(grid, objects=None, title="3D Spatial Grid"):
    """Visualize a 3D grid with objects.
    
    Args:
        grid: 3D numpy array with object class IDs
        objects: Dictionary mapping object names to their class IDs
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Default objects if not provided
    if objects is None:
        objects = {
            "chair": 1, "table": 2, "car": 3, "lamp": 4, "box": 5,
            "book": 6, "vase": 7, "plant": 8, "computer": 9, "phone": 10
        }
    
    # Reverse mapping from ID to name
    id_to_name = {v: k for k, v in objects.items()}
    
    # Find objects in the grid
    object_positions = {}
    for z in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[2]):
                obj_id = grid[z, y, x]
                if obj_id > 0 and obj_id in id_to_name:
                    obj_name = id_to_name[obj_id]
                    object_positions[obj_name] = (x, y, z)
    
    # Define colors for different objects
    colors = {
        'chair': 'red', 'table': 'blue', 'car': 'green', 'lamp': 'purple',
        'box': 'orange', 'book': 'brown', 'vase': 'pink',
        'plant': 'lime', 'computer': 'cyan', 'phone': 'magenta'
    }
    
    # Get grid center
    center = (grid.shape[2]//2, grid.shape[1]//2, grid.shape[0]//2)
    
    # Draw coordinate axes
    ax.plot([0, grid.shape[2]-1], [center[1], center[1]], [center[2], center[2]],
            'r-', linewidth=2, label="X axis (left-right)")
    ax.plot([center[0], center[0]], [0, grid.shape[1]-1], [center[2], center[2]],
            'g-', linewidth=2, label="Y axis (back-front)")
    ax.plot([center[0], center[0]], [center[1], center[1]], [0, grid.shape[0]-1],
            'b-', linewidth=2, label="Z axis (down-up)")
    
    # Plot objects
    for obj_name, (x, y, z) in object_positions.items():
        ax.scatter(x, y, z, color=colors.get(obj_name, 'gray'),
                 s=500, label=obj_name, edgecolors='black', alpha=0.8)
        ax.text(x+0.3, y+0.3, z+0.3, obj_name, fontsize=12, weight='bold')
    
    # Draw a line between objects if there are exactly two
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
    
    # Highlight grid center
    ax.scatter(*center, color='black', s=100, alpha=0.3, label="center")
    
    # Set labels and limits
    ax.set_xlabel('X axis (left → right)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y axis (back → front)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z axis (down → up)', fontsize=12, labelpad=10)
    
    ax.set_xlim(0, grid.shape[2]-1)
    ax.set_ylim(0, grid.shape[1]-1)
    ax.set_zlim(0, grid.shape[0]-1)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def convert_grid_directions_to_vectors(relation_vectors_grid, relation_map):
    """Convert grid-based relation vectors to format expected by project analysis.
    
    Args:
        relation_vectors_grid: Dictionary from grid-based approach
        relation_map: Mapping from grid relation names to project relation names
        
    Returns:
        Dictionary with converted relation vectors
    """
    converted = {}
    
    for grid_rel, vec in relation_vectors_grid.items():
        # Convert relation name if it's in the mapping
        if grid_rel in relation_map:
            project_rel = relation_map[grid_rel]
            converted[project_rel] = vec
        else:
            # Keep original name if not in mapping
            converted[grid_rel] = vec
    
    return converted