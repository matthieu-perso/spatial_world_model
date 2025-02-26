import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Load the dataset
dataset = pd.read_csv('dataset.csv')
from huggingface_hub import login

# Log in to Hugging Face
login(token="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")

# Load the Llama model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get the representation of the last token at specified layers
def get_last_token_representation(text, layers):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    embeddings = {}
    for layer in layers:
        layer_representation = hidden_states[layer]
        last_token_representation = layer_representation[:, -1, :]  # Get the representation of the last token
        embeddings[layer] = last_token_representation.detach().numpy()
    return embeddings

# Define the layers at 25%, 50%, and 75% of the total layers
total_layers = len(model.config.hidden_layers)
layers = [total_layers // 4, total_layers // 2, (3 * total_layers) // 4]
print(f'We are using the layers: {layers}')

# Get embeddings for each text in the dataset
all_embeddings = []
for text in dataset['text']:
    embeddings = get_last_token_representation(text, layers)
    all_embeddings.append(embeddings)

# Save the embeddings
embeddings_df = pd.DataFrame(all_embeddings)
embeddings_df.to_csv('embeddings.csv', index=False)

