# 3D Spatial Relationship Analysis in Language Models

This repository contains code for analyzing how language models encode 3D spatial relationships between objects. The project creates a grid-world representation of spatial relationships and probes a language model to extract spatial information.

## Overview

The project demonstrates that language models encode rich spatial information that can be extracted using linear probes. The code:

1. Generates diverse sentences describing spatial relationships between objects
2. Extracts activations from a language model (Llama 3.2) for these sentences
3. Trains a linear probe to map from model activations to 3D grid representations
4. Visualizes and evaluates the model's understanding of spatial relationships
5. Analyzes the compositionality of spatial relationships using PCA

## Project Structure

- `data_grid.py`: Contains grid representation, data generation, and processing functions
- `prob_grid.py`: Contains the probe model, training, and evaluation code
- `compositionality_grid.py`: Contains PCA analysis for spatial relationship compositionality
- `main_grid.py`: Main execution script to tie everything together

## Requirements

- PyTorch
- Transformers
- NumPy
- Matplotlib
- scikit-learn
- tqdm

## Usage

### Basic Usage

```bash
# Load a pretrained model and run comprehensive tests
python main_grid.py --model_file spatial_3d_probe.pt --run_tests --test_comprehensive
```

### Generate New Data and Train a New Model

```bash
# Extract new activations and train a new probe
python main_grid.py --extract_new --train_new --num_sentences 15000 --epochs 50 --model_name "meta-llama/Llama-3.2-3B" --hf_token YOUR_HF_TOKEN
```

### Run PCA Analysis

```bash
# Run PCA analysis on spatial relationship vectors
python main_grid.py --model_file spatial_3d_probe.pt --run_pca
```

## Examples of Spatial Relationships

The model can represent various spatial relationships, including:

- Basic relationships: "above", "below", "left", "right", "in front of", "behind"
- Diagonal relationships: "diagonally front-left", "diagonally back-right", etc.
- Compound relationships: "above and to the left", "below and in front of", etc.

## Visualization Examples

The code provides 3D visualizations of spatial relationships, showing how the model places objects in a 3D grid based on textual descriptions.

## Compositionality Analysis

The project analyzes whether compound spatial relationships (e.g., "diagonally front-left") are represented as compositions of basic relationships (e.g., "left" + "in front"). This is done through PCA analysis and by examining the relationships between vectors in the spatial representation space.

## License

This project is open source and available under the MIT License.
