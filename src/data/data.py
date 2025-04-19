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


# ===========================
# function: encode quintuples
# ===========================
def encode_quintuples(
    quintuples: List[Tuple[str, str, str, str, str]],
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    """
    Encode quintuples for model training.

    Each quintuple is expected to be in the form:
       (object_1, relation_1, object_2, relation_2, object_3)

    This function encodes each component separately using a LabelEncoder.

    Parameters:
        quintuples (List[Tuple[str, str, str, str, str]]): List of quintuples.
        verbose (bool): If True, prints out summary information.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
            - labels: A dictionary mapping each component key to its encoded numpy array.
            - mappings: A dictionary mapping each component key to a dictionary of the form {class: index}.
    """
    start_time = time.time()
    print("Encoding quintuples..." if verbose else "")

    # Unpack quintuples into five separate lists.
    objects1, relations1, objects2, relations2, objects3 = zip(*quintuples)

    # Replace empty strings with "UNKNOWN"
    objects1 = [obj if obj else "UNKNOWN" for obj in objects1]
    relations1 = [rel if rel else "UNKNOWN" for rel in relations1]
    objects2 = [obj if obj else "UNKNOWN" for obj in objects2]
    relations2 = [rel if rel else "UNKNOWN" for rel in relations2]
    objects3 = [obj if obj else "UNKNOWN" for obj in objects3]

    # Create a label encoder for each component.
    obj1_encoder = LabelEncoder()
    rel1_encoder = LabelEncoder()
    obj2_encoder = LabelEncoder()
    rel2_encoder = LabelEncoder()
    obj3_encoder = LabelEncoder()

    obj1_labels = obj1_encoder.fit_transform(objects1)
    rel1_labels = rel1_encoder.fit_transform(relations1)
    obj2_labels = obj2_encoder.fit_transform(objects2)
    rel2_labels = rel2_encoder.fit_transform(relations2)
    obj3_labels = obj3_encoder.fit_transform(objects3)

    labels = {
        "object_1": obj1_labels,
        "relation_1": rel1_labels,
        "object_2": obj2_labels,
        "relation_2": rel2_labels,
        "object_3": obj3_labels
    }

    # Create mapping dictionaries for later analysis.
    obj1_mapping = dict(zip(obj1_encoder.classes_, range(len(obj1_encoder.classes_))))
    rel1_mapping = dict(zip(rel1_encoder.classes_, range(len(rel1_encoder.classes_))))
    obj2_mapping = dict(zip(obj2_encoder.classes_, range(len(obj2_encoder.classes_))))
    rel2_mapping = dict(zip(rel2_encoder.classes_, range(len(rel2_encoder.classes_))))
    obj3_mapping = dict(zip(obj3_encoder.classes_, range(len(obj3_encoder.classes_))))

    mappings = {
        "object_1": obj1_mapping,
        "relation_1": rel1_mapping,
        "object_2": obj2_mapping,
        "relation_2": rel2_mapping,
        "object_3": obj3_mapping
    }

    if verbose:
        print(f"Found {len(obj1_mapping)} unique objects as subject (object_1)")
        print(f"Found {len(rel1_mapping)} unique relations (relation_1)")
        print(f"Found {len(obj2_mapping)} unique intermediate objects (object_2)")
        print(f"Found {len(rel2_mapping)} unique relations (relation_2)")
        print(f"Found {len(obj3_mapping)} unique objects as final object (object_3)")
        print("\nUnique relation_1 values:")
        for i, rel in enumerate(list(rel1_mapping.keys())):
            print(f"{i}: {rel}")
        print("\nUnique relation_2 values:")
        for i, rel in enumerate(list(rel2_mapping.keys())):
            print(f"{i}: {rel}")
        print(f"\nEncoding completed in {time.time() - start_time:.2f} seconds")

    return labels, mappings




# ==========================================
# function: generate single sentence dataset
# ==========================================
def generate_single_sentence_dataset(objects: List[str], relations: List[str], verbose: bool = False, num_print_sentences: int = 10) -> List[str]:
    """
    generate a list of single-sentence examples using all ordered pairs of the given objects and the specified spataial relations.

    each sentence has the form:
        "The {object1} is {relation} the {object2}"
    """
    sentences = []

    # use ordered pairs (A, B) where A != B
    for obj1, obj2 in permutations(objects, 2):
        for rel in relations:
            sentence = f"The {obj1} is {rel} the {obj2}."
            sentences.append(sentence)

    if verbose:
        print(f"generated {len(sentences)} sentences.\n")
        print("first few sentences:\n")
        for sentence in sentences[:num_print_sentences]:
            print(sentence)


    return sentences

# ============================
# function: extract embeddings
# ============================
def extract_embeddings(
    sentences: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers_list: List[str] = [8, 16, 24],
    verbose: bool=False,
    idx_break: int=None
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    extract embeddings for a list of sentences using the given Transformer model.
    """

    # allocate memory
    sentences = []
    embeddings = {}

    for k in layers_list:
        layer_name = f"layer_{k}"
        embeddings[layer_name] = []

    # extract embeddings
    print("extracting embeddings from dataset of sentences...\n" if verbose else "")
    for idx, sentence_text in enumerate(tqdm(train_sentences, desc="Rows")):
        # print(f"\nsentence : {sentence_text}\n" if verbose else "")

        # tokenize
        inputs = tokenizer(sentence_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # inference
        with torch.no_grad():
            # forward pass of the model
            outputs = model(**inputs, output_hidden_states=True)

            # compute hidden states
            hidden_states = outputs.hidden_states # tupple of len = (num_layers + 1). For each layer, shape is (1, seq, d_model)

            # last token's embedding: hidden_states[layer][:, -1, :]
            for k in layers_list:
                layer_name = f"layer_{k}"
                embeddings[layer_name].append(hidden_states[k][:, -1, :].squeeze(0).cpu().numpy())

            # append sentence
            sentences.append(sentence_text)

        if idx_break is not None and idx >= idx_break:
            print(f"\nbreak at {idx_break}")
            break

    # transform to numpy arrays
    for k in layers_list:
        layer_name = f"layer_{k}"
        embeddings[layer_name] = np.stack(embeddings[layer_name], axis=0)
        print(f"{layer_name} shape: {embeddings[layer_name].shape}" if verbose else "")

    return sentences, embeddings



# =========================
# function: parse sentences
# =========================
def parse_sentences(
    sentences: List[str],
    verbose: bool = True,
    num_sample_triplets: int = 100
) -> Dict[str, List]:
    """
    Parses sentences into structured triplets with precise relation extraction.
    """
    start_time = time.time()

    triplets = []
    valid_indices = []

    # Updated regex pattern to capture the full multi-word relation
    pattern = r"The\s+(.*?)\s+is\s+(.*)\s+the\s+(.*?)\."

    for i, sentence in enumerate(sentences):
        match = re.match(pattern, sentence, re.IGNORECASE)
        if match:
            obj1 = match.group(1).strip()
            relation = match.group(2).strip()
            obj2 = match.group(3).strip()
            triplets.append((obj1, relation, obj2))
            valid_indices.append(i)

    if verbose:
        print("\n=== triplets ===")
        for i in range(min(num_sample_triplets, len(triplets))):
            print(f"{i}: {triplets[i]}")

    return {
        "triplets": triplets,
        "valid_indices": valid_indices
    }

# =========================
# function: encode triplets
# =========================
def encode_triplets(
    triplets: List[Tuple[str, str, str]],
    verbose: bool = True
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    """
    encode triplets for model training
    """
    start_time = time.time()
    print("encoding triplets..." if verbose else "")

    # extract separate components
    objects1, relations, objects2 = zip(*triplets)

    # filter out empty strings
    objects1 = [obj if obj else "UNKNOWN" for obj in objects1]
    relations = [rel if rel else "UNKNOWN" for rel in relations]
    objects2 = [obj if obj else "UNKNOWN" for obj in objects2]

    # encode each component
    obj1_encoder = LabelEncoder()
    rel_encoder = LabelEncoder()
    obj2_encoder = LabelEncoder()

    obj1_labels = obj1_encoder.fit_transform(objects1)
    rel_labels = rel_encoder.fit_transform(relations)
    obj2_labels = obj2_encoder.fit_transform(objects2)

    labels = {"object_1": obj1_labels, "relation": rel_labels, "object_2": obj2_labels}

    # create mapping dictionaries for later analysis
    obj1_mapping = dict(zip(obj1_encoder.classes_, range(len(obj1_encoder.classes_))))
    rel_mapping = dict(zip(rel_encoder.classes_, range(len(rel_encoder.classes_))))
    obj2_mapping = dict(zip(obj2_encoder.classes_, range(len(obj2_encoder.classes_))))

    mappings = {"object_1": obj1_mapping, "relation": rel_mapping, "object_2": obj2_mapping}

    if verbose:
        print(f"found {len(obj1_mapping)} unique objects as subject")
        print(f"found {len(rel_mapping)} unique relations")
        print(f"found {len(obj2_mapping)} unique objects as object")

        # Print some of the unique relations
        print("\nrelations used:")
        sample_relations = list(rel_mapping.keys())
        for i, rel in enumerate(sample_relations):
            print(f"{i}: {rel}")

        print(f"encoding completed in {time.time() - start_time:.2f} seconds")

    return labels, mappings


# ============================
# function: prepare data split
# ============================
def prepare_data_split(layer_data: torch.Tensor, valid_indices: List[int], labels: Dict[str, np.ndarray], test_size: float=0.2) -> Dict[str, torch.Tensor]:
    """
    prepare single train/test split for all labels
    """
    # filter embeddings to keep only valid indices
    X = layer_data[valid_indices]

    # create a single train/test split for all labels
    X_train, X_test, y_obj1_train, y_obj1_test, y_rel_train, y_rel_test, y_obj2_train, y_obj2_test = train_test_split(
        X,
        labels["object_1"],
        labels["relation"],
        labels["object_2"],
        test_size=test_size,
        random_state=42
        )

    # convert to PyTorch tensors
    X_train_tensor = X_train.clone().detach().float()
    X_test_tensor = X_test.clone().detach().float()

    y_obj1_train_tensor = torch.tensor(y_obj1_train, dtype=torch.long)
    y_obj1_test_tensor = torch.tensor(y_obj1_test, dtype=torch.long)

    y_rel_train_tensor = torch.tensor(y_rel_train, dtype=torch.long)
    y_rel_test_tensor = torch.tensor(y_rel_test, dtype=torch.long)

    y_obj2_train_tensor = torch.tensor(y_obj2_train, dtype=torch.long)
    y_obj2_test_tensor = torch.tensor(y_obj2_test, dtype=torch.long)

    return {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor,
        'y_obj1_train': y_obj1_train_tensor,
        'y_obj1_test': y_obj1_test_tensor,
        'y_rel_train': y_rel_train_tensor,
        'y_rel_test': y_rel_test_tensor,
        'y_obj2_train': y_obj2_train_tensor,
        'y_obj2_test': y_obj2_test_tensor
        }



# ========================================
# function: generate transitivity datasets
# ========================================
def generate_transitivity_datasets(
    objects: List[str],
    composition_relations: Dict,
    verbose: bool = False,
    num_print_sentences: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Generate two datasets of transitivity examples using ordered triples of objects and a dictionary of
    compositional relations.

    The input composition_relations is a dictionary where each key is a direct (composed) relation and its
    value is a tuple containing the two atomic relations that compose it.

    For each ordered triple (object1, object2, object3) where all objects are distinct,
    and for each compositional relation in composition_relations, the function generates:

    1. A chain sentence:
       "The {object1} is {atomic1} the {object2}, and the {object2} is {atomic2} the {object3}."

    2. A direct sentence:
       "The {object1} is {composed_relation} the {object3}."

    """
    chain_sentences = []
    direct_sentences = []

    # For each compositional relation and its atomic pair, generate sentences for all ordered triples.
    for composed_rel, (atomic1, atomic2) in composition_relations.items():
        for obj1, obj2, obj3 in permutations(objects, 3):
            chain_sentence = f"The {obj1} is {atomic1} the {obj2}, and the {obj2} is {atomic2} the {obj3}."
            direct_sentence = f"The {obj1} is {composed_rel} the {obj3}."
            chain_sentences.append(chain_sentence)
            direct_sentences.append(direct_sentence)

    if verbose:
        print(f"Generated {len(chain_sentences)} chain sentences and {len(direct_sentences)} direct sentences.\n")
        print("First few examples:\n")
        for i in range(min(num_print_sentences, len(chain_sentences))):
            print("Chain:", chain_sentences[i])
            print("Direct:", direct_sentences[i])
            print()

    return chain_sentences, direct_sentences

# ============================
# function: extract embeddings
# ============================
def extract_embeddings(
    train_sentences: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers_list: List[str] = [8, 16, 24],
    verbose: bool=False,
    idx_break: int=None
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    extract embeddings for a list of sentences using the given Transformer model.
    """

    # allocate memory
    sentences = []
    embeddings = {}

    for k in layers_list:
        layer_name = f"layer_{k}"
        embeddings[layer_name] = []

    # extract embeddings
    print("extracting embeddings from dataset of sentences...\n" if verbose else "")
    for idx, sentence_text in enumerate(tqdm(train_sentences, desc="Rows")):
        # print(f"\nsentence : {sentence_text}\n" if verbose else "")

        # tokenize
        inputs = tokenizer(sentence_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # inference
        with torch.no_grad():
            # forward pass of the model
            outputs = model(**inputs, output_hidden_states=True)

            # compute hidden states
            hidden_states = outputs.hidden_states # tupple of len = (num_layers + 1). For each layer, shape is (1, seq, d_model)

            # last token's embedding: hidden_states[layer][:, -1, :]
            for k in layers_list:
                layer_name = f"layer_{k}"
                embeddings[layer_name].append(hidden_states[k][:, -1, :].squeeze(0).cpu().numpy())

            # append sentence
            sentences.append(sentence_text)

        if idx_break is not None and idx >= idx_break:
            print(f"\nbreak at {idx_break}")
            break

    # transform to numpy arrays
    for k in layers_list:
        layer_name = f"layer_{k}"
        embeddings[layer_name] = np.stack(embeddings[layer_name], axis=0)
        print(f"{layer_name} shape: {embeddings[layer_name].shape}" if verbose else "")

    return sentences, embeddings


# ================================
# function: parse chained senteces
# ================================
def parse_chained_sentences(
    sentences: List[str],
    verbose: bool = True,
    num_sample_quintuples: int = 10
) -> Dict[str, List]:
    """
    Parses chained sentences (of the form:
       "The {obj1} is {rel1} the {obj2}, and the {obj2} is {rel2} the {obj3}.")
    by splitting them into two direct sentences, parsing each into triplets, and then
    combining the results into a quintuple:
       (obj1, rel1, obj2, rel2, obj3)

    Returns a dictionary with:
       "quintuples": List of quintuples,
       "valid_indices": List of indices for which parsing was successful.
    """
    start_time = time.time()
    quintuples = []
    valid_indices = []

    for i, sentence in enumerate(sentences):
        # Split the sentence at ", and"
        parts = re.split(r",\s+and\s+", sentence)
        if len(parts) != 2:
            if verbose:
                print(f"Sentence index {i} does not split into two parts: {sentence}")
            continue
        # Ensure each part ends with a period.
        if not parts[0].strip().endswith('.'):
            parts[0] = parts[0].strip() + "."
        if not parts[1].strip().endswith('.'):
            parts[1] = parts[1].strip() + "."

        # Parse each part using the parse_sentences function.
        parsed_first = parse_sentences([parts[0]], verbose=False)
        parsed_second = parse_sentences([parts[1]], verbose=False)
        if not parsed_first["triplets"] or not parsed_second["triplets"]:
            if verbose:
                print(f"Parsing failed for sentence index {i}: {sentence}")
            continue
        triplet1 = parsed_first["triplets"][0]  # (obj1, rel1, obj2)
        triplet2 = parsed_second["triplets"][0] # (obj2, rel2, obj3)
        # Check that the middle object is consistent.
        if triplet1[2].lower() != triplet2[0].lower():
            if verbose:
                print(f"Warning: Mismatch in middle objects at index {i}: {triplet1[2]} vs {triplet2[0]}")
            continue
        quintuple = (triplet1[0], triplet1[1], triplet1[2], triplet2[1], triplet2[2])
        quintuples.append(quintuple)
        valid_indices.append(i)

    if verbose:
        print("\n=== Extracted Quintuples (obj1, rel1, obj2, rel2, obj3) ===")
        for j in range(min(num_sample_quintuples, len(quintuples))):
            print(f"{j}: {quintuples[j]}")
        elapsed = time.time() - start_time
        print(f"\nParsing completed in {elapsed:.2f} seconds.")

    return {
        "quintuples": quintuples,
        "valid_indices": valid_indices
    }



# ============================
# function: prepare data split
# ============================
def prepare_data_split_chained(layer_data: torch.Tensor, valid_indices: List[int], labels: Dict[str, np.ndarray], test_size: float=0.2) -> Dict[str, torch.Tensor]:
    """
    prepare single train/test split for all labels
    """
    # filter embeddings to keep only valid indices
    X = layer_data[valid_indices]

    # create a single train/test split for all labels
    X_train, X_test, y_obj1_train, y_obj1_test, y_rel_1_train, y_rel_1_test, y_obj2_train, y_obj2_test, y_rel_2_train, y_rel_2_test, y_obj3_train, y_obj3_test = train_test_split(
        X,
        labels["object_1"],
        labels["relation_1"],
        labels["object_2"],
        labels["relation_2"],
        labels["object_3"],
        test_size=test_size,
        random_state=42
        )

    # convert to PyTorch tensors
    X_train_tensor = X_train.clone().detach().float()
    X_test_tensor = X_test.clone().detach().float()

    y_obj1_train_tensor = torch.tensor(y_obj1_train, dtype=torch.long)
    y_obj1_test_tensor = torch.tensor(y_obj1_test, dtype=torch.long)

    y_rel_1_train_tensor = torch.tensor(y_rel_1_train, dtype=torch.long)
    y_rel_1_test_tensor = torch.tensor(y_rel_1_test, dtype=torch.long)

    y_obj2_train_tensor = torch.tensor(y_obj2_train, dtype=torch.long)
    y_obj2_test_tensor = torch.tensor(y_obj2_test, dtype=torch.long)

    y_rel_2_train_tensor = torch.tensor(y_rel_2_train, dtype=torch.long)
    y_rel_2_test_tensor = torch.tensor(y_rel_2_test, dtype=torch.long)

    return {
        'X_train': X_train_tensor,
        'X_test': X_test_tensor,
        'y_obj1_train': y_obj1_train_tensor,
        'y_obj1_test': y_obj1_test_tensor,
        'y_rel_1_train': y_rel_1_train_tensor,
        'y_rel_1_test': y_rel_1_test_tensor,
        'y_obj2_train': y_obj2_train_tensor,
        'y_obj2_test': y_obj2_test_tensor,
        'y_rel_2_train': y_rel_2_train_tensor,
        'y_rel_2_test': y_rel_2_test_tensor,
        }

# =============================
# function: create data loaders
# =============================
def create_data_loaders_chained(split_data: Dict[str, torch.Tensor], batch_size: int=256) -> Dict[str, Dict[str, TensorDataset]]:
    """
    create data loaders for training and testing
    """
    # create training datasets
    train_obj1_dataset = TensorDataset(split_data['X_train'], split_data['y_obj1_train'])
    train_rel_1_dataset = TensorDataset(split_data['X_train'], split_data['y_rel_1_train'])
    train_obj2_dataset = TensorDataset(split_data['X_train'], split_data['y_obj2_train'])
    train_rel_2_dataset = TensorDataset(split_data['X_train'], split_data['y_rel_2_train'])

    # create test datasets
    test_obj1_dataset = TensorDataset(split_data['X_test'], split_data['y_obj1_test'])
    test_rel_1_dataset = TensorDataset(split_data['X_test'], split_data['y_rel_1_test'])
    test_obj2_dataset = TensorDataset(split_data['X_test'], split_data['y_obj2_test'])
    test_rel_2_dataset = TensorDataset(split_data['X_test'], split_data['y_rel_2_test'])

    # create data loaders
    train_obj1_loader = DataLoader(train_obj1_dataset, batch_size=batch_size, shuffle=True)
    train_rel_1_loader = DataLoader(train_rel_1_dataset, batch_size=batch_size, shuffle=True)
    train_obj2_loader = DataLoader(train_obj2_dataset, batch_size=batch_size, shuffle=True)
    train_rel_2_loader = DataLoader(train_rel_2_dataset, batch_size=batch_size, shuffle=True)

    test_obj1_loader = DataLoader(test_obj1_dataset, batch_size=batch_size)
    test_rel_1_loader = DataLoader(test_rel_1_dataset, batch_size=batch_size)
    test_obj2_loader = DataLoader(test_obj2_dataset, batch_size=batch_size)
    test_rel_2_loader = DataLoader(test_rel_2_dataset, batch_size=batch_size)

    return {
        'train': {
            'object_1': train_obj1_loader,
            'relation_1': train_rel_1_loader,
            'object_2': train_obj2_loader,
            'relation_2': train_rel_2_loader,
            },
        'test': {
            'object_1': test_obj1_loader,
            'relation_1': test_rel_1_loader,
            'object_2': test_obj2_loader,
            'relation_2': test_rel_2_loader,
            }
        }



# =============================
# function: create data loaders
# =============================
def create_data_loaders(split_data: Dict[str, torch.Tensor], batch_size: int=256) -> Dict[str, Dict[str, TensorDataset]]:
    """
    create data loaders for training and testing
    """
    # create training datasets
    train_obj1_dataset = TensorDataset(split_data['X_train'], split_data['y_obj1_train'])
    train_rel_dataset = TensorDataset(split_data['X_train'], split_data['y_rel_train'])
    train_obj2_dataset = TensorDataset(split_data['X_train'], split_data['y_obj2_train'])

    # create test datasets
    test_obj1_dataset = TensorDataset(split_data['X_test'], split_data['y_obj1_test'])
    test_rel_dataset = TensorDataset(split_data['X_test'], split_data['y_rel_test'])
    test_obj2_dataset = TensorDataset(split_data['X_test'], split_data['y_obj2_test'])

    # create data loaders
    train_obj1_loader = DataLoader(train_obj1_dataset, batch_size=batch_size, shuffle=True)
    train_rel_loader = DataLoader(train_rel_dataset, batch_size=batch_size, shuffle=True)
    train_obj2_loader = DataLoader(train_obj2_dataset, batch_size=batch_size, shuffle=True)

    test_obj1_loader = DataLoader(test_obj1_dataset, batch_size=batch_size)
    test_rel_loader = DataLoader(test_rel_dataset, batch_size=batch_size)
    test_obj2_loader = DataLoader(test_obj2_dataset, batch_size=batch_size)

    return {
        'train': {
            'object_1': train_obj1_loader,
            'relation': train_rel_loader,
            'object_2': train_obj2_loader
            },
        'test': {
            'object_1': test_obj1_loader,
            'relation': test_rel_loader,
            'object_2': test_obj2_loader
            }
        }


# ============================
# function: check transitivity
# ============================
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
