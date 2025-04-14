import torch
import random
import numpy as np

from data.data import generate_single_sentence_dataset, extract_embeddings, parse_sentences, encode_triplets, prepare_data_split, create_data_loaders, generate_3d_grid_data, parse_sentence_to_grid
from analysis.inverse import check_inverse_relations
from analysis.compositionality import check_compositional_relations, check_compositional_relations_grid, compute_relation_vectors_from_grid
from analysis.transitivity import check_transitivity_relations, check_transitivity_relations_grid
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import set_seed, visualize_3d_grid
from training.probe import ProbeTrainingArgs, LinearProbeTrainer, evaluate_probe, get_directions, GridLinearProbe, train_grid_probe, evaluate_grid_probe
from torch import Tensor

VERBOSE = False
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# define objects
all_objects = [
    "book", "mug", "lamp", "phone", "remote", "cushion", "plate", "plate", "notebook", "pen", "cup",
    "clock", "chair", "table", "keyboard", "mouse", "bottle", "plant", "vase", "wallet", "bag", "shoe",
    "hat", "pencil", "eraser", "folder", "speaker", "picture", "mirror", "pillow", "blanket", "carpet",
    "painting", "flower", "stapler", "calculator", "projector", "monitor", "printer", "scanner",
    "microphone", "camera", "laptop", "tablet", "mousepad", "desk", "couch", "sofa", "bed", "dresser",
    "wardrobe", "bookshelf", "stool", "bench", "armchair",
    "recliner", "footstool", "rug", "curtain", "chandelier", "lamp", "candle",
]

# split train and test objects
num_train = int(1.0 * len(all_objects))
num_test = len(all_objects) - num_train

train_objects = all_objects[:num_train]
test_objects = all_objects[num_train:]

# define relations
basic_relations = ["above", "below", "to the left of", "to the right of"]
diagonal_relations = [
    "diagonally above and to the right of",
    "diagonally above and to the left of",
    "diagonally below and to the right of",
    "diagonally below and to the left of"
]
all_relations = basic_relations + diagonal_relations

# define train sentences
train_sentences = generate_single_sentence_dataset(train_objects, all_relations, verbose=VERBOSE)

# define test sentences
# test_sentences = generate_single_sentence_dataset(test_objects, all_relations, verbose=VERBOSE)   # to test on unseen objects

# extract hidden states for training sentences
layers_list = [8, 16, 24]

# define the model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
    output_hidden_states=True,
    return_dict_in_generate=True,
    device_map="auto"
)
model.eval()

idx_break = None   # int for debugging (None if not)

# get embeddings for training linear-probe
random.shuffle(train_sentences)

sentences, train_embeddings = extract_embeddings(
    train_sentences,
    model,
    tokenizer,
    layers_list=layers_list,
    verbose=VERBOSE,
    idx_break=idx_break
    )

# summary of statistics
print("\n" + "=" * 40)
print("📊 dataset and embedding statistics")
print("=" * 40)

# dataset statistics
print(f"\n🔢 dataset overview:")
print(f"• total training rows: {len(train_sentences):,}")
print(f"• total sentences: {len(sentences):,}")

# layer details
print(f"\n🧬 embedding layers extracted:")
for k in layers_list:
    layer_name = f"layer_{k}"
    layer_shape = train_embeddings[layer_name].shape
    print(f"• layer {k}: shape: {layer_shape}")

print("\n" + "=" * 40)

"""## training linear probe"""

# define layer
selected_layer: int = 24

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"we are using {device}" if VERBOSE else "")

# seet rng seed
set_seed(1234)

# transform data to pytorch
layer_name = f"layer_{selected_layer}"
layer_tensor: Tensor = torch.tensor(train_embeddings[layer_name][:])
print(f"layer {selected_layer} embeddings shape (# sentences, d_model): {layer_tensor.shape}")

# parse senteces
pars_sentences = parse_sentences(sentences, verbose=VERBOSE, num_sample_triplets=20)

# encode triplets
labels, mappings = encode_triplets(pars_sentences["triplets"], verbose=VERBOSE)

# prepare layer data for split
batch_size, d_model = layer_tensor.shape
print(f"layer data shape (num, d_model): {layer_tensor.shape}" if VERBOSE else "")

# get number of relations
options = len(mappings["relation"])
print(f"number of unique relations: {options}" if VERBOSE else "")

# create data loaders
test_size = 0.2
split_data = prepare_data_split(layer_tensor, pars_sentences["valid_indices"], labels, test_size=test_size)
dataloaders = create_data_loaders(split_data, batch_size=1024)

# train the linear probe
torch.set_grad_enabled(True)

args = ProbeTrainingArgs(
    d_model=d_model,
    options=options,
    layer_name=layer_name,
    epochs=100,
    lr=1e-3,
    verbose=VERBOSE,
    )

trainer = LinearProbeTrainer(args, dataloader=dataloaders["train"]["relation"])      # we focus only on the spatial relation
trainer.train()

"""## evaluate probe"""

# report accuracy
accuracy, report = evaluate_probe(trainer.linear_probe, dataloaders["test"]["relation"], device=device)
print(f"Linear probe accuracy: {accuracy:.4f}" if VERBOSE else "")

# extract the directions of each spatial relation as a Dict
directions = get_directions(trainer.linear_probe, options, mappings, verbose=VERBOSE)  # each direction has shape (d_model,)

"""## Analyze Spatial Relations"""

# 1. Check inverse relations (for basic relations)
print("\n" + "=" * 40)
print("🔄 Analyzing Inverse Relations")
print("=" * 40)
basic_directions = {rel: directions[rel] for rel in basic_relations}
inverse_df = check_inverse_relations(basic_directions)

# 2. Check compositional relations (for diagonal relations)
print("\n" + "=" * 40)
print("🧩 Analyzing Compositional Relations")
print("=" * 40)
composition_df = check_compositional_relations(directions)

# 3. Check transitivity relations
print("\n" + "=" * 40)
print("🔄 Analyzing Transitivity Relations")
print("=" * 40)
composition_relations = {
    "diagonally above and to the right of": ("above", "to the right of"),
    "diagonally above and to the left of": ("above", "to the left of"),
    "diagonally below and to the right of": ("below", "to the right of"),
    "diagonally below and to the left of": ("below", "to the left of"),
}

# Split directions into basic and diagonal for transitivity analysis
basic_directions = {rel: directions[rel] for rel in basic_relations}
diagonal_directions = {rel: directions[rel] for rel in diagonal_relations}

transitivity_df = check_transitivity_relations(
    composition_relations,
    basic_directions,
    basic_directions,
    diagonal_directions,
    verbose=VERBOSE
) 

# ========================
# Grid Model
# ========================
"""## Analyze Spatial Relations with Grid-based Approach"""

# Define 3D grid parameters
grid_size = (9, 9, 9)
objects = {
    "chair": 1, "table": 2, "car": 3, "lamp": 4, "box": 5,
    "book": 6, "vase": 7, "plant": 8, "computer": 9, "phone": 10
}

# Define spatial map for grid-based approach
spatial_map = {
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
    # 3D combinations
    "above_and_left": (-2, 0, 2),
    "above_and_right": (2, 0, 2),
    "below_and_left": (-2, 0, -2),
    "below_and_right": (2, 0, -2),
    "above_and_in_front": (0, 2, 2),
    "above_and_behind": (0, -2, 2),
    "below_and_in_front": (0, 2, -2),
    "below_and_behind": (0, -2, -2),
}

# Generate 3D grid representations from sentences
print("\n" + "=" * 40)
print("📊 3D Grid Analysis")
print("=" * 40)

# Parse sentences to 3D grids
grid_labels = generate_3d_grid_data(sentences, objects, spatial_map, grid_size)
grid_labels_tensor = torch.tensor(np.array(grid_labels), dtype=torch.long)

# Create grid probe
grid_probe = GridLinearProbe(d_model, grid_size, len(objects) + 1)
    
# Create dataset and dataloader for grid probe
grid_dataset = torch.utils.data.TensorDataset(layer_tensor, grid_labels_tensor)
grid_dataloader = torch.utils.data.DataLoader(grid_dataset, batch_size=64, shuffle=True)

# Train the grid probe
print("\nTraining grid-based linear probe...")
grid_probe = train_grid_probe(grid_probe, grid_dataloader, device, epochs=50)

# Evaluate grid probe
grid_accuracy, grid_loss = evaluate_grid_probe(grid_probe, grid_dataloader, device)
print(f"Grid probe accuracy: {grid_accuracy:.4f}, loss: {grid_loss:.4f}")

# Extract spatial vectors from grid probe
grid_vectors = grid_probe.extract_spatial_vectors()

# Analyze compositional relations with grid approach
print("\n" + "=" * 40)
print("🧩 Analyzing Compositional Relations (Grid-based)")
print("=" * 40)
grid_composition_df = check_compositional_relations_grid(grid_vectors, spatial_map=spatial_map)

# Analyze transitivity with grid approach
print("\n" + "=" * 40)
print("🔄 Analyzing Transitivity Relations (Grid-based)")
print("=" * 40)
composition_relations = {
    "diagonally_front_right": ("right", "in_front"),
    "diagonally_front_left": ("left", "in_front"),
    "diagonally_back_right": ("right", "behind"),
    "diagonally_back_left": ("left", "behind"),
    "above_and_right": ("above", "right"),
    "above_and_left": ("above", "left"),
    "below_and_right": ("below", "right"),
    "below_and_left": ("below", "left"),
    "above_and_in_front": ("above", "in_front"),
    "above_and_behind": ("above", "behind"),
    "below_and_in_front": ("below", "in_front"),
    "below_and_behind": ("below", "behind"),
}
grid_transitivity_df = check_transitivity_relations_grid(grid_vectors, composition_relations, spatial_map)

# Compare approaches
print("\n" + "=" * 40)
print("🔍 Comparing Approaches")
print("=" * 40)

print("\nCompositional Analysis:")
print(f"Original approach average similarity: {composition_df['Cosine Similarity (Original)'].mean():.4f}")
print(f"Grid approach average similarity: {grid_composition_df['Cosine Similarity (Original)'].mean():.4f}")

print("\nTransitivity Analysis:")
print(f"Original approach average similarity: {transitivity_df['Cosine Similarity (Original)'].mean():.4f}")
print(f"Grid approach average similarity: {grid_transitivity_df['Cosine Similarity (Original)'].mean():.4f}")