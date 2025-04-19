import torch
import numpy as np
import re
import random
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Grid representation
# -------------------------------
GRID_SIZE = (9, 9, 9)
OBJECTS = {
    "chair": 1, "table": 2, "car": 3, "lamp": 4, "box": 5,
    "book": 6, "vase": 7, "plant": 8, "computer": 9, "phone": 10
}

SPATIAL_MAP = {
    # horizontal (x)
    "left": (-2, 0, 0),
    "right": (2, 0, 0),

    # depth (y)
    "in_front": (0, 2, 0),
    "front": (0, 2, 0),
    "behind": (0, -2, 0),
    "back": (0, -2, 0),

    # vertical (z)
    "above": (0, 0, 2),
    "below": (0, 0, -2),

    # diagonal
    "diagonally_front_left": (-2, 2, 0),
    "diagonally_front_right": (2, 2, 0),
    "diagonally_back_left": (-2, -2, 0),
    "diagonally_back_right": (2, -2, 0),

    # 3D combination
    "above_and_left": (-2, 0, 2),
    "above_and_right": (2, 0, 2),
    "below_and_left": (-2, 0, -2),
    "below_and_right": (2, 0, -2),
    "above_and_in_front": (0, 2, 2),
    "above_and_behind": (0, -2, 2),
    "below_and_in_front": (0, 2, -2),
    "below_and_behind": (0, -2, -2),
    "to_the_left_diagonally": (-2, 1, 0),
    "to_the_right_diagonally": (2, 1, 0),
    "left_front": (-2, 2, 0),
    "right_front": (2, 2, 0),
    "left_back": (-2, -2, 0),
    "right_back": (2, -2, 0)
}

# RELATIONSHIP_CATEGORIES
RELATIONSHIP_CATEGORIES = {
    "vertical": ["above", "below", "above_and_in_front", "above_and_behind",
                "below_and_in_front", "below_and_behind"],
    "horizontal": ["left", "right", "above_and_left", "above_and_right",
                  "below_and_left", "below_and_right"],
    "depth": ["in_front", "front", "behind", "back"],
    "diagonal": ["diagonally_front_left", "diagonally_front_right",
                "diagonally_back_left", "diagonally_back_right",
                "to_the_left_diagonally", "to_the_right_diagonally",
                "left_front", "right_front", "left_back", "right_back",
                "diagonally_front_right", "diagonally_front_left",
                "diagonally_back_right", "diagonally_back_left",
                "left_diagonally", "right_diagonally"]
}

# Create reverse mapping for category lookup
RELATIONSHIP_TO_CATEGORY = {}
for category, relations in RELATIONSHIP_CATEGORIES.items():
    for relation in relations:
        RELATIONSHIP_TO_CATEGORY[relation] = category

# -------------------------------
# Language Model setup functions
# -------------------------------
def setup_language_model(model_name, token=None):
    """Initialize the language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                               output_hidden_states=True,
                                               token=token).cuda()
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_last_token_activation(text, model, tokenizer):
    """Extract activation from the last hidden state of the last token."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)["input_ids"].cuda()

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)

    # Get last token
    last_token_idx = tokens.shape[1] - 1
    last_token_activation = outputs.hidden_states[-1][:, last_token_idx, :]

    return last_token_activation.squeeze(0)

# -------------------------------
# Spatial relationship parsing
# -------------------------------
def normalize_relation(relation):
    """Enhanced relation normalization with better variant handling."""
    relation = relation.lower().strip()
    relation = relation.replace(" of", "").replace("to the ", "")

    # Standardize spacing and dashes
    relation = relation.replace("-", "_")
    relation = relation.replace(" ", "_")

    # Handle more diagonal relation variations
    if ("left" in relation and "diagonally" in relation and "front" in relation) or \
       ("left" in relation and "diagonal" in relation and "front" in relation):
        return "diagonally_front_left"
    elif ("right" in relation and "diagonally" in relation and "front" in relation) or \
         ("right" in relation and "diagonal" in relation and "front" in relation):
        return "diagonally_front_right"
    elif ("left" in relation and "diagonally" in relation and "back" in relation) or \
         ("left" in relation and "diagonal" in relation and "back" in relation):
        return "diagonally_back_left"
    elif ("right" in relation and "diagonally" in relation and "back" in relation) or \
         ("right" in relation and "diagonal" in relation and "back" in relation):
        return "diagonally_back_right"

    # Handle existing normalizations
    if relation in ["in_front_of", "front_of"]:
        relation = "in_front"
    elif relation in ["to_the_left", "to_left"]:
        relation = "left"
    elif relation in ["to_the_right", "to_right"]:
        relation = "right"

    # Handle compound relations
    if "and" in relation:
        parts = relation.split("and")
        parts = [p.strip('_') for p in parts]
        relation = f"{parts[0]}_and_{parts[1]}"

    return relation

def determine_relationship_category(relation):
    """Map relation to its category index (0=vertical, 1=horizontal, 2=depth, 3=diagonal, 4=compound)."""
    relation = normalize_relation(relation)

    # First check if it's a compound relation
    if "_and_" in relation:
        return 4  # Compound relation

    if relation in RELATIONSHIP_TO_CATEGORY:
        category = RELATIONSHIP_TO_CATEGORY[relation]
        if category == "vertical":
            return 0
        elif category == "horizontal":
            return 1
        elif category == "depth":
            return 2
        else:  # diagonal
            return 3

    # Default to horizontal relation
    return 1

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

def parse_sentence_to_grid(sentence, grid_size=GRID_SIZE):
    """Enhanced parser to ensure both objects are placed in the grid with better handling of compound relations."""
    grid = np.zeros(grid_size, dtype=int)
    center = (grid_size[0]//2, grid_size[1]//2, grid_size[2]//2)

    patterns = [
        r"The (\w+) is ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? the (\w+)",
        r"The (\w+) is ((?:to the )?\w+(?:[ -]\w+)*) and ((?:to the )?\w+(?:[ -]\w+)*)(?: of)? the (\w+)",  # Handle compound relations
        r"There is a (\w+) ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? the (\w+)",
        r"A (\w+) is ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? the (\w+)",
        r"The (\w+) has a (\w+) ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? it",
        r"The (\w+) with a (\w+) ((?:to the )?\w+(?:[- ]\w+)*)(?: of)? it"
    ]

    # First try to match compound relations
    compound_match = re.search(r"The (\w+) is ((?:to the )?\w+(?:[ -]\w+)*) and ((?:to the )?\w+(?:[ -]\w+)*)(?: of)? the (\w+)", sentence, re.IGNORECASE)
    if compound_match:
        obj1 = compound_match.group(1)
        relation1 = compound_match.group(2)
        relation2 = compound_match.group(3)
        obj2 = compound_match.group(4)

        # Combine the two relations
        combined_relation = f"{relation1}_and_{relation2}"
        norm_relation = normalize_relation(combined_relation)

        # Place reference object (obj2) at center
        x2, y2, z2 = center
        if obj2 in OBJECTS:
            grid[z2, y2, x2] = OBJECTS[obj2]

        # Calculate compound displacement vector
        if norm_relation in SPATIAL_MAP:
            dx, dy, dz = SPATIAL_MAP[norm_relation]
        # Try to handle compound relations by combining parts
        elif "and" in norm_relation:
            parts = norm_relation.split("_and_")
            dx, dy, dz = 0, 0, 0
            for part in parts:
                if part in SPATIAL_MAP:
                    pdx, pdy, pdz = SPATIAL_MAP[part]
                    dx += pdx
                    dy += pdy
                    dz += pdz
        else:
            # Try to process the two relations separately and combine displacements
            norm_rel1 = normalize_relation(relation1)
            norm_rel2 = normalize_relation(relation2)

            if norm_rel1 in SPATIAL_MAP and norm_rel2 in SPATIAL_MAP:
                dx1, dy1, dz1 = SPATIAL_MAP[norm_rel1]
                dx2, dy2, dz2 = SPATIAL_MAP[norm_rel2]
                dx, dy, dz = dx1 + dx2, dy1 + dy2, dz1 + dz2
            else:
                # Default displacement
                dx, dy, dz = (0, 0, 0)

        # Calculate obj1's position relative to obj2
        x1, y1, z1 = x2 + dx, y2 + dy, z2 + dz

        # Ensure coordinates are within grid boundaries
        x1 = max(0, min(x1, grid_size[0]-1))
        y1 = max(0, min(y1, grid_size[1]-1))
        z1 = max(0, min(z1, grid_size[2]-1))

        # Place first object
        if obj1 in OBJECTS:
            grid[z1, y1, x1] = OBJECTS[obj1]

        return grid

    # Regular relation matching
    for pattern in patterns:
        matches = re.findall(pattern, sentence, re.IGNORECASE)
        if matches:
            # Handle different pattern formats
            if "with" in pattern and len(matches[0]) == 3:
                # For "The box with a lamp below it", reverse the objects
                obj2, obj1, relation = matches[0]  # box, lamp, below
            elif "has a" in pattern and len(matches[0]) == 3:
                # For "has a" pattern, the objects are also reversed
                obj2, obj1, relation = matches[0]
            elif len(matches[0]) == 3:
                # Standard pattern
                obj1, relation, obj2 = matches[0]
            else:
                # Compound relation
                obj1, rel1, rel2, obj2 = matches[0]
                relation = f"{rel1} and {rel2}"

            # Place reference object at center
            x2, y2, z2 = center
            if obj2 in OBJECTS:
                grid[z2, y2, x2] = OBJECTS[obj2]

            # Normalize and look up relation
            norm_relation = normalize_relation(relation)

            # Calculate displacement vector
            if norm_relation in SPATIAL_MAP:
                dx, dy, dz = SPATIAL_MAP[norm_relation]
            else:
                # Default displacement
                dx, dy, dz = (0, 0, 0)

            # Calculate obj1's position relative to obj2
            x1, y1, z1 = x2 + dx, y2 + dy, z2 + dz

            # Ensure coordinates are within grid boundaries
            x1 = max(0, min(x1, grid_size[0]-1))
            y1 = max(0, min(y1, grid_size[1]-1))
            z1 = max(0, min(z1, grid_size[2]-1))

            # Place first object
            if obj1 in OBJECTS:
                grid[z1, y1, x1] = OBJECTS[obj1]

            return grid

    # If no match, return grid
    return grid

# -------------------------------
# Sentence generation
# -------------------------------
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

# -------------------------------
# Data processing
# -------------------------------
def extract_and_save_activations(sentences, model, tokenizer, save_path="spatial_activations.pt"):
    """Extract activations from sentences and save them with labels for later use."""
    print(f"Extracting activations from {len(sentences)} sentences...")

    # Initialize storage
    activations = []
    grid_labels = []
    relation_categories = []

    # Process in batches to save memory
    batch_size = 100
    num_batches = (len(sentences) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sentences))
        batch_sentences = sentences[start_idx:end_idx]

        print(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_sentences)} sentences)...")

        for sentence in tqdm(batch_sentences):
            # Extract activation
            activation = get_last_token_activation(sentence, model, tokenizer)
            activations.append(activation.cpu())  # Move to CPU to save memory

            # Parse grid
            grid = parse_sentence_to_grid(sentence)
            grid_labels.append(grid)

            # Determine relation category
            relation = extract_relation_from_sentence(sentence)
            category = determine_relationship_category(relation)
            relation_categories.append(category)

    # Convert to appropriate tensor formats
    activations_tensor = torch.stack(activations)
    grid_labels_tensor = torch.tensor(np.array(grid_labels), dtype=torch.long)
    relation_categories_tensor = torch.tensor(relation_categories, dtype=torch.long)

    # Create dictionary with all data
    data_dict = {
        'activations': activations_tensor,
        'grid_labels': grid_labels_tensor,
        'relation_categories': relation_categories_tensor,
        'grid_size': GRID_SIZE,
        'num_objects': len(OBJECTS),
        'objects_map': OBJECTS,
        'sentences': sentences  # Also save original sentences for reference
    }

    # Save to disk
    torch.save(data_dict, save_path)
    print(f"Saved {len(sentences)} activations to {save_path}")

    # Analyze data distribution
    analyze_saved_data(data_dict)

    return data_dict

def analyze_saved_data(data_dict):
    """Analyze the distribution of relations in the saved data."""
    print("Analyzing distribution of spatial relations...")

    sentences = data_dict['sentences']
    relation_categories = data_dict['relation_categories'].tolist()

    # Extract relations from sentences
    relations = []
    for sentence in tqdm(sentences, desc="Analyzing sentences for spatial relations"):
        relation = extract_relation_from_sentence(sentence)
        relations.append(relation)

    # Analyze distribution
    relation_counts = {}
    category_counts = {
        "VERTICAL": 0,
        "HORIZONTAL": 0,
        "DEPTH": 0,
        "DIAGONAL": 0,
        "UNKNOWN": 0
    }

    category_map = {
        0: "VERTICAL",
        1: "HORIZONTAL",
        2: "DEPTH",
        3: "DIAGONAL",
        4: "COMPOUND",  # This is treated as UNKNOWN in the counts
        -1: "UNKNOWN"
    }

    for relation, category in zip(relations, relation_categories):
        # Count relations
        if relation not in relation_counts:
            relation_counts[relation] = 0
        relation_counts[relation] += 1

        # Count categories
        cat_name = category_map.get(category, "UNKNOWN")
        if cat_name == "COMPOUND":
            cat_name = "UNKNOWN"  # Simplify for analysis
        category_counts[cat_name] += 1

    # Print summary statistics
    total_valid = sum(category_counts.values())
    print(f"TOTAL ANALYSIS:")
    print(f"Total sentences with valid relations: {total_valid}")
    print(f"Unique relations found: {len(relation_counts)}")

    print("\nRELATION DISTRIBUTION BY CATEGORY:")
    for category, count in category_counts.items():
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0
        print(f"{category}: {count} ({percentage:.1f}%)")

    # Print top relations
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTOP 15 MOST COMMON RELATIONS:")
    for i, (relation, count) in enumerate(sorted_relations[:15]):
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0

        # Determine relation category
        relation_norm = normalize_relation(relation)
        if relation_norm in RELATIONSHIP_TO_CATEGORY:
            cat = RELATIONSHIP_TO_CATEGORY[relation_norm]
        else:
            cat = "unknown"

        print(f"{relation}: {count} ({percentage:.1f}%) - {cat}")

def prepare_data_from_saved(data_dict, batch_size=32):
    """Prepare dataloader from saved activations."""
    activations = data_dict['activations']
    grid_labels = data_dict['grid_labels']
    relation_categories = data_dict['relation_categories']

    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(
        activations, grid_labels, relation_categories
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader

def load_activations(load_path="spatial_activations.pt"):
    """Load previously saved activations and related data."""
    print(f"Loading data from {load_path}...")
    
    import os
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Activation file {load_path} not found!")
        
    data_dict = torch.load(load_path)
    
    print(f"Found {len(data_dict['sentences'])} sentences in saved data")
    
    # Analyze data distribution
    analyze_saved_data(data_dict)
    
    return data_dict
