import numpy as np
import re
import random
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Objects and their IDs
# -------------------------------
OBJECTS = {
    "chair": 1, "table": 2, "car": 3, "lamp": 4, "box": 5,
    "book": 6, "vase": 7, "plant": 8, "computer": 9, "phone": 10
}

# -------------------------------
# Spatial Mapping
# -------------------------------
SPATIAL_MAP = {
    # horizontal (x)
    "left": (-2, 0, 0),
    "to the left of": (-2, 0, 0),
    "right": (2, 0, 0),
    "to the right of": (2, 0, 0),
    
    # depth (y)
    "in front": (0, 2, 0),
    "in front of": (0, 2, 0),
    "front": (0, 2, 0),
    "behind": (0, -2, 0),
    "back": (0, -2, 0),
    
    # vertical (z)
    "above": (0, 0, 2),
    "below": (0, 0, -2),
    
    # diagonal horizontal-depth
    "diagonally front-left": (-2, 2, 0),
    "diagonally front-right": (2, 2, 0),
    "diagonally back-left": (-2, -2, 0),
    "diagonally back-right": (2, -2, 0),
    
    # diagonal vertical-horizontal
    "diagonally above and to the left of": (-2, 0, 2),
    "diagonally above and to the right of": (2, 0, 2),
    "diagonally below and to the left of": (-2, 0, -2),
    "diagonally below and to the right of": (2, 0, -2),
    
    # diagonal vertical-depth
    "diagonally above and in front of": (0, 2, 2),
    "diagonally above and behind": (0, -2, 2),
    "diagonally below and in front of": (0, 2, -2),
    "diagonally below and behind": (0, -2, -2),
    
    # 3D compound combinations
    "above and to the left of": (-2, 0, 2),
    "above and to the right of": (2, 0, 2),
    "below and to the left of": (-2, 0, -2),
    "below and to the right of": (2, 0, -2),
    "above and in front of": (0, 2, 2),
    "above and behind": (0, -2, 2),
    "below and in front of": (0, 2, -2),
    "below and behind": (0, -2, -2),
}

# ============================================
# function: normalize relation
# ============================================
def normalize_relation(relation: str) -> str:
    """
    Enhanced relation normalization with better handling of variations.
    
    Args:
        relation: The spatial relation string to normalize
        
    Returns:
        Normalized relation string that matches keys in SPATIAL_MAP
    """
    relation = relation.lower().strip()
    relation = relation.replace(" of", "").strip()
    
    # Handle "to the X" variants
    if relation.startswith("to the "):
        base_rel = relation.replace("to the ", "").strip()
        if base_rel in ["left", "right"]:
            return f"to the {base_rel} of"
    
    # Handle diagonal combinations
    if "diagonally" in relation:
        if "left" in relation and "front" in relation:
            return "diagonally front-left"
        elif "right" in relation and "front" in relation:
            return "diagonally front-right"
        elif "left" in relation and "back" in relation:
            return "diagonally back-left"
        elif "right" in relation and "back" in relation:
            return "diagonally back-right"
        elif "above" in relation and "left" in relation:
            return "diagonally above and to the left of"
        elif "above" in relation and "right" in relation:
            return "diagonally above and to the right of"
        elif "below" in relation and "left" in relation:
            return "diagonally below and to the left of"
        elif "below" in relation and "right" in relation:
            return "diagonally below and to the right of"
        elif "above" in relation and "front" in relation:
            return "diagonally above and in front of"
        elif "above" in relation and "behind" in relation:
            return "diagonally above and behind"
        elif "below" in relation and "front" in relation:
            return "diagonally below and in front of"
        elif "below" in relation and "behind" in relation:
            return "diagonally below and behind"
    
    # Handle compound relations with "and"
    if " and " in relation:
        parts = relation.split(" and ")
        parts = [p.strip() for p in parts]
        
        # Normalize individual parts
        norm_parts = []
        for part in parts:
            if part in ["left", "right"]:
                norm_parts.append(f"to the {part} of")
            else:
                norm_parts.append(part)
        
        # Build normalized compound relation
        if len(norm_parts) == 2:
            return f"{norm_parts[0]} and {norm_parts[1]}"
    
    # If no special cases apply, return original or map to closest known relation
    if relation in SPATIAL_MAP:
        return relation
    
    # Try common substitutions
    common_subs = {
        "in front": "in front of",
        "front of": "in front of",
        "to left": "to the left of",
        "to right": "to the right of"
    }
    
    for pattern, replacement in common_subs.items():
        if pattern in relation:
            return replacement
    
    return relation


# ============================================
# function: parse sentence to grid
# ============================================
def parse_sentence_to_grid(sentence: str, objects: Dict[str, int], spatial_map: Dict[str, Tuple[int, int, int]], 
                           grid_size: Tuple[int, int, int] = (9, 9, 9)) -> np.ndarray:
    """
    Enhanced parser that converts a sentence describing spatial relations into a 3D grid.
    
    Args:
        sentence: A sentence containing spatial relation description
        objects: Dictionary mapping object names to their class IDs
        spatial_map: Dictionary mapping relation names to (dx, dy, dz) offsets
        grid_size: Tuple of (x, y, z) dimensions for the grid
        
    Returns:
        3D numpy array representing object positions in the grid
    """
    grid = np.zeros(grid_size, dtype=int)
    center = (grid_size[0]//2, grid_size[1]//2, grid_size[2]//2)
    center_x, center_y, center_z = center
    
    # Regex patterns to match various sentence formats
    patterns = [
        r"The (\w+) is (.*?) the (\w+)",  # The X is Y the Z
        r"The (\w+) is (.*?) and (.*?) the (\w+)",  # The X is Y and Z the W (for compound relations)
        r"There is a (\w+) (.*?) the (\w+)",  # There is a X Y the Z
        r"A (\w+) is (.*?) the (\w+)",  # A X is Y the Z
        r"The (\w+) has a (\w+) (.*?) it",  # The X has a Y Z it
        r"The (\w+) with a (\w+) (.*?) it"  # The X with a Y Z it
    ]
    
    # First try to match compound relations (they have a more complex pattern)
    compound_match = re.search(r"The (\w+) is (.*?) and (.*?) the (\w+)", sentence, re.IGNORECASE)
    if compound_match:
        obj1 = compound_match.group(1)
        rel1 = compound_match.group(2)
        rel2 = compound_match.group(3)
        obj2 = compound_match.group(4)
        
        # Combine the two relations
        combined_relation = f"{rel1} and {rel2}"
        norm_relation = normalize_relation(combined_relation)
        
        # Place reference object (obj2) at center
        if obj2 in objects:
            grid[center_z, center_y, center_x] = objects[obj2]
        
        # Find displacement for combined relation
        if norm_relation in spatial_map:
            dx, dy, dz = spatial_map[norm_relation]
        else:
            # Try to combine individual relations
            norm_rel1 = normalize_relation(rel1)
            norm_rel2 = normalize_relation(rel2)
            
            if norm_rel1 in spatial_map and norm_rel2 in spatial_map:
                dx1, dy1, dz1 = spatial_map[norm_rel1]
                dx2, dy2, dz2 = spatial_map[norm_rel2]
                dx, dy, dz = dx1 + dx2, dy1 + dy2, dz1 + dz2
            else:
                # Use default displacement if we can't determine
                dx, dy, dz = 0, 0, 0
        
        # Calculate target position for obj1
        target_x = center_x + dx
        target_y = center_y + dy
        target_z = center_z + dz
        
        # Ensure coordinates are within grid bounds
        target_x = max(0, min(target_x, grid_size[0]-1))
        target_y = max(0, min(target_y, grid_size[1]-1))
        target_z = max(0, min(target_z, grid_size[2]-1))
        
        # Place first object
        if obj1 in objects:
            grid[target_z, target_y, target_x] = objects[obj1]
        
        return grid
    
    # Try standard patterns if no compound relation found
    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            # Handle different patterns
            if "with" in pattern and len(match.groups()) == 3:
                # For "The box with a lamp below it", reverse the objects
                obj2, obj1, relation = match.groups()  # box, lamp, below
            elif "has a" in pattern and len(match.groups()) == 3:
                # For "has a" pattern, the objects are also reversed
                obj2, obj1, relation = match.groups()
            elif len(match.groups()) == 3:
                # Standard pattern
                obj1, relation, obj2 = match.groups()
            else:
                # Unrecognized pattern
                continue
            
            # Place reference object at center
            if obj2 in objects:
                grid[center_z, center_y, center_x] = objects[obj2]
            
            # Normalize relation and get displacement
            norm_relation = normalize_relation(relation)
            if norm_relation in spatial_map:
                dx, dy, dz = spatial_map[norm_relation]
            else:
                # Default to no displacement if relation unknown
                dx, dy, dz = 0, 0, 0
            
            # Calculate target position for obj1
            target_x = center_x + dx
            target_y = center_y + dy
            target_z = center_z + dz
            
            # Ensure coordinates are within grid bounds
            target_x = max(0, min(target_x, grid_size[0]-1))
            target_y = max(0, min(target_y, grid_size[1]-1))
            target_z = max(0, min(target_z, grid_size[2]-1))
            
            # Place first object
            if obj1 in objects:
                grid[target_z, target_y, target_x] = objects[obj1]
            
            return grid
    
    # Return empty grid if no patterns match
    return grid


# ============================================
# function: generate 3D grid data
# ============================================
def generate_3d_grid_data(sentences: List[str], objects: Dict[str, int], spatial_map: Dict[str, Tuple[int, int, int]], 
                         grid_size: Tuple[int, int, int] = (9, 9, 9), verbose: bool = False) -> List[np.ndarray]:
    """
    Generate 3D grid representations from spatial relation sentences.
    
    Args:
        sentences: List of sentences containing spatial relations
        objects: Dictionary mapping object names to their class IDs
        spatial_map: Dictionary mapping relation names to (dx, dy, dz) offsets
        grid_size: Tuple of (x, y, z) dimensions for the grid
        verbose: Whether to print progress information
        
    Returns:
        List of 3D grid representations as numpy arrays
    """
    if verbose:
        print(f"Generating 3D grid data for {len(sentences)} sentences...")
    
    grids = []
    for sentence in tqdm(sentences, disable=not verbose):
        grid = parse_sentence_to_grid(sentence, objects, spatial_map, grid_size)
        grids.append(grid)
    
    return grids


# ============================================
# function: visualize 3D grid
# ============================================
def visualize_3d_grid(grid: np.ndarray, objects: Dict[str, int] = None, title: str = "3D Spatial Grid",
                     save_path: str = None) -> None:
    """
    Enhanced visualization of a 3D grid with objects.
    
    Args:
        grid: 3D numpy array with object class IDs
        objects: Dictionary mapping object names to their class IDs
        title: Plot title
        save_path: Optional path to save the figure
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    # Print object positions for reference
    print("Object positions:", object_positions)


# ============================================
# function: generate balanced grid dataset
# ============================================
def generate_balanced_grid_dataset(objects: Dict[str, int], spatial_map: Dict[str, Tuple[int, int, int]],
                                 num_examples: int = 1000, grid_size: Tuple[int, int, int] = (9, 9, 9),
                                 verbose: bool = False) -> List[Tuple[np.ndarray, Tuple[str, str, str]]]:
    """
    Generate a balanced dataset of 3D grids with varied spatial relationships.
    
    Args:
        objects: Dictionary mapping object names to their class IDs
        spatial_map: Dictionary mapping relation names to (dx, dy, dz) offsets
        num_examples: Number of grid examples to generate
        grid_size: Tuple of (x, y, z) dimensions for the grid
        verbose: Whether to print progress information
        
    Returns:
        List of tuples containing (grid, (obj1, relation, obj2))
    """
    # Group relations by category for balanced sampling
    relation_categories = {
        "vertical": [],
        "horizontal": [],
        "depth": [],
        "diagonal": [],
        "compound": []
    }
    
    # Categorize relations
    for relation in spatial_map.keys():
        if relation in ["above", "below"]:
            relation_categories["vertical"].append(relation)
        elif relation in ["left", "right", "to the left of", "to the right of"]:
            relation_categories["horizontal"].append(relation)
        elif relation in ["in front", "in front of", "front", "behind", "back"]:
            relation_categories["depth"].append(relation)
        elif "diagonally" in relation:
            relation_categories["diagonal"].append(relation)
        elif "and" in relation:
            relation_categories["compound"].append(relation)
    
    # Set category proportions
    category_proportions = {
        "vertical": 0.2,    # 20%
        "horizontal": 0.2,  # 20%
        "depth": 0.2,       # 20%
        "diagonal": 0.2,    # 20%
        "compound": 0.2     # 20%
    }
    
    # Calculate examples per category
    examples_per_category = {
        cat: int(num_examples * prop) for cat, prop in category_proportions.items()
    }
    
    # Adjust for rounding errors
    total_allocated = sum(examples_per_category.values())
    if total_allocated < num_examples:
        # Add remaining to compound (usually the most complex)
        examples_per_category["compound"] += (num_examples - total_allocated)
    
    if verbose:
        print(f"Generating {num_examples} balanced grid examples:")
        for cat, count in examples_per_category.items():
            print(f"  {cat}: {count} examples ({count/num_examples*100:.1f}%)")
    
    # Generate grids
    dataset = []
    
    for category, count in examples_per_category.items():
        relations = relation_categories[category]
        if not relations:
            continue
            
        # Balance examples across relations in this category
        examples_per_relation = max(1, count // len(relations))
        
        for relation in relations:
            for _ in range(examples_per_relation):
                # Sample two different objects
                obj1_name, obj2_name = random.sample(list(objects.keys()), 2)
                obj1_id, obj2_id = objects[obj1_name], objects[obj2_name]
                
                # Create an empty grid
                grid = np.zeros(grid_size, dtype=int)
                center = (grid_size[0]//2, grid_size[1]//2, grid_size[2]//2)
                
                # Place reference object (obj2) at center
                grid[center[2], center[1], center[0]] = obj2_id
                
                # Get displacement vector for relation
                if relation in spatial_map:
                    dx, dy, dz = spatial_map[relation]
                    
                    # Calculate target position for obj1
                    target_x = center[0] + dx
                    target_y = center[1] + dy
                    target_z = center[2] + dz
                    
                    # Ensure coordinates are within grid bounds
                    target_x = max(0, min(target_x, grid_size[0]-1))
                    target_y = max(0, min(target_y, grid_size[1]-1))
                    target_z = max(0, min(target_z, grid_size[2]-1))
                    
                    # Place first object
                    grid[target_z, target_y, target_x] = obj1_id
                    
                    # Add to dataset
                    dataset.append((grid, (obj1_name, relation, obj2_name)))
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Trim to exactly num_examples
    dataset = dataset[:num_examples]
    
    return dataset


# ============================================
# function: compute relation vectors from grid
# ============================================
def compute_relation_vectors_from_grid(grid_vectors, center=None):
    """Extract relation vectors from the grid vectors.
    
    Args:
        grid_vectors: 5D tensor [classes, z, y, x, features]
        center: Optional center coordinates (default is middle of grid)
        
    Returns:
        Dictionary mapping relation names to their vector representations
    """
    import torch
    
    if center is None:
        # Use the center of the grid
        center = (grid_vectors.shape[3]//2, grid_vectors.shape[2]//2, grid_vectors.shape[1]//2)

    # Extract the center vector (reference point)
    center_x, center_y, center_z = center
    center_vector = grid_vectors[0, center_z, center_y, center_x, :]  # Using class 0 (background)

    # Compute relation vectors for each spatial relation
    relation_vectors = {}
    for relation, (dx, dy, dz) in SPATIAL_MAP.items():
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


def extract_relation_from_sentence(sentence):
    """Enhanced extraction of spatial relations from sentences."""
    # First try to match compound relations
    compound_pattern = r"is ((?:to the )?\w+(?:[ -]\w+)*) and ((?:to the )?\w+(?:[ -]\w+)*)(?: of)? the"
    compound_match = re.search(compound_pattern, sentence, re.IGNORECASE)

    if compound_match:
        rel1 = compound_match.group(1)
        rel2 = compound_match.group(2)
        return f"{rel1} and {rel2}"

    # Try more specific patterns for different formulations
    patterns = [
        r"is ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? the",
        r"is ((?:diagonally )?\w+(?:[- ]\w+)*)(?: of)? the",
        r"is ((?:in front )?\w+(?:[- ]\w+)*)(?: of)? the",
        r"is ((?:behind )?\w+(?:[- ]\w+)*)(?: of)? the",
        r"a (\w+) ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? the"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                relation = matches[0][1] if len(matches[0]) > 1 else matches[0][0]
            else:
                relation = matches[0]
            return relation

    # Last resort - check for basic spatial terms
    spatial_terms = ["left", "right", "above", "below", "in front", "behind",
                     "front", "back", "diagonally"]

    for term in spatial_terms:
        if term in sentence.lower():
            return term

    return "unknown"

def generate_diverse_spatial_sentences(num_sentences=15000):
    """Generate a more balanced set of spatial relationship sentences."""
    sentences = []

    # Allocate specific proportions to each category
    category_counts = {
        "horizontal": num_sentences * 0.2,  # 20%
        "vertical": num_sentences * 0.2,    # 20%
        "depth": num_sentences * 0.2,       # 20% (increased)
        "diagonal": num_sentences * 0.2,    # 20%
        "compound": num_sentences * 0.2     # 20%
    }

    # Generate horizontal relations
    horizontal_relations = ["left", "right"]
    for _ in range(int(category_counts["horizontal"] // len(horizontal_relations))):
        for relation in horizontal_relations:
            obj1, obj2 = random.sample(list(OBJECTS.keys()), 2)
            formatted_relation = f"to the {relation} of"
            sentences.append(f"The {obj1} is {formatted_relation} the {obj2}.")

    # Generate vertical relations
    vertical_relations = ["above", "below"]
    for _ in range(int(category_counts["vertical"] // len(vertical_relations))):
        for relation in vertical_relations:
            obj1, obj2 = random.sample(list(OBJECTS.keys()), 2)
            sentences.append(f"The {obj1} is {relation} the {obj2}.")

    # Generate depth relations (increased proportion)
    depth_relations = ["in front of", "behind"]
    for _ in range(int(category_counts["depth"] // len(depth_relations))):
        for relation in depth_relations:
            obj1, obj2 = random.sample(list(OBJECTS.keys()), 2)
            sentences.append(f"The {obj1} is {relation} the {obj2}.")

    # Generate diagonal relations
    diagonal_relations = [
        "diagonally front-left of", "diagonally front-right of",
        "diagonally back-left of", "diagonally back-right of"
    ]
    for _ in range(int(category_counts["diagonal"] // len(diagonal_relations))):
        for relation in diagonal_relations:
            obj1, obj2 = random.sample(list(OBJECTS.keys()), 2)
            sentences.append(f"The {obj1} is {relation} the {obj2}.")

    # Generate compound relations
    compound_relations = [
        ("above", "left"), ("above", "right"),
        ("below", "left"), ("below", "right"),
        ("above", "in front of"), ("above", "behind"),
        ("below", "in front of"), ("below", "behind")
    ]
    for _ in range(int(category_counts["compound"] // len(compound_relations))):
        for rel1, rel2 in compound_relations:
            obj1, obj2 = random.sample(list(OBJECTS.keys()), 2)

            # Format relations properly
            formatted_rel1 = rel1
            formatted_rel2 = rel2

            if rel1 in ["left", "right"]:
                formatted_rel1 = f"to the {rel1}"
            if rel2 in ["left", "right"]:
                formatted_rel2 = f"to the {rel2}"

            sentences.append(f"The {obj1} is {formatted_rel1} and {formatted_rel2} of the {obj2}.")

    # Shuffle the data
    random.shuffle(sentences)
    return sentences[:num_sentences]
