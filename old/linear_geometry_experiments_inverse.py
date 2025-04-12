import torch
import random

from src.data.data import generate_single_sentence_dataset, extract_embeddings, parse_sentences, encode_triplets, prepare_data_split, create_data_loaders
from src.analysis.inverse import check_inverse_relations
from src.analysis.compositionality import check_compositional_relations
from src.analysis.transitivity import check_transitivity_relations
from transformers import AutoTokenizer, AutoModelForCausalLM

VERBOSE = True
hf_token = userdata.get('HF_TOKEN')

# define objects
all_objects = [
    "book", "mug", "lamp", "phone", "remote", "cushion", "plate", "plate", "notebook", "pen", "cup",
    "clock", "chair", "table", "keyboard", "mouse", "bottle", "plant", "vase", "wallet", "bag", "shoe",
    "hat", "pencil", "eraser", "folder", "speaker", "picture", "mirror", "pillow", "blanket", "carpet",
    "painting", "flower", "stapler", "calculator", "projector", "monitor", "printer", "scanner",
    "microphone", "camera", "laptop", "tablet", "mousepad", "desk", "couch", "sofa", "bed", "dresser",
    "wardrobe", "bookshelf", "stool", "bench", "armchair",
    # "recliner",  "footstool", "rug", "curtain", "chandelier", "lamp", "candle",
]

# split train and test objects
num_train = int(1.0 * len(all_objects))
num_test = len(all_objects) - num_train

train_objects = all_objects[:num_train]
test_objects = all_objects[num_train:]

# define relations
relations = ["above", "below", "to the left of", "to the right of"]

# define train sentences
train_sentences = generate_single_sentence_dataset(train_objects, relations, verbose=VERBOSE)

# define test sentences
test_sentences = generate_single_sentence_dataset(test_objects, relations, verbose=VERBOSE)

# extract hidden states for training sentences
layers_list = [8, 16, 24]
idx_break = None

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

# get training embeddings for linear-probe
random.shuffle(train_sentences)

sentences, train_embeddings = extract_embeddings(
    train_sentences,
    model,
    tokenizer,
    layers_list=layers_list,
    verbose=False,
    idx_break=idx_break
    )

# summary of statistics
print("\n" + "=" * 40)
print("📊 dataset and Embedding Statistics")
print("=" * 40)

# dataset statistics
print(f"\n🔢 dataset Overview:")
print(f"• total training rows: {len(train_sentences):,}")
print(f"• total sentences: {len(sentences):,}")

# layer details
print(f"\n🧬 embedding layers extracted:")
for k in layers_list:
    layer_name = f"layer_{k}"
    layer_shape = train_embeddings[layer_name].shape
    print(f"• layer {k}: shape: {layer_shape}")

print("\n" + "=" * 40)

"""## training linear probes"""

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

# create data loadersthe
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

"""## evaluate probe

"""

# report accuracy
accuracy, report = evaluate_probe(trainer.linear_probe, dataloaders["test"]["relation"], device=device)
print(f"Linear probe accuracy: {accuracy:.4f}" if VERBOSE else "")

# extract the directions of each spatial relation as a Dict
directions = get_directions(trainer.linear_probe, options, mappings, verbose=VERBOSE)  # each direction has shape (d_model,)

# check directions' properties
directions_df = check_inverse_relations(directions)
directions_df = check_compositional_relations(directions)


"""So, we have

$$
\text{PCA}(w_\text{above}) \approx - \text{PCA}(w_\text{below})
$$
and
$$
\text{PCA}(w_\text{left}) \approx - \text{PCA}(w_\text{right})
$$
"""