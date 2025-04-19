# Grid-Based Spatial World Model Analysis

This document provides instructions for running the standalone grid-based spatial reasoning analysis using `main_grid.py`.

## Overview

The grid-based approach represents spatial relationships between objects in a 3D grid, providing a more explicit spatial understanding of how language models position objects when given spatial descriptions. This approach complements the vector-based linear probe method by offering a more intuitive, geometrically grounded analysis.

## Key Features

- Maps object relationships onto a 3D grid representation
- Analyzes compositionality of spatial relationships in grid space
- Tests transitivity of spatial reasoning in grid space
- Provides visualization of object positions in 3D space

## Requirements

- PyTorch
- Transformers
- Matplotlib
- NumPy
- Pandas
- Hugging Face token (set as environment variable `HF_TOKEN`)

## Running the Grid Analysis

### Basic Usage

```bash
python src/main_grid.py
```

This will run the analysis with default parameters:
- Model: meta-llama/Llama-3.2-3B-Instruct
- Layer: 24
- Grid size: 9×9×9
- Output directory: ../results/grid_analysis

### Command-Line Options

You can customize the analysis using these options:

```bash
python src/main_grid.py --model <model_name> --layer <layer_number> --grid_size <x> <y> <z> --epochs <num_epochs> --seed <random_seed> --output_dir <output_directory> --visualize
```

Parameters:
- `--model`: Model to use for analysis (default: "meta-llama/Llama-3.2-3B-Instruct")
- `--layer`: Layer to extract embeddings from (default: 24)
- `--grid_size`: 3D grid dimensions as x y z (default: 9 9 9)
- `--epochs`: Number of epochs for training grid probe (default: 50)
- `--seed`: Random seed (default: 1234)
- `--output_dir`: Directory to save results (default: "../results/grid_analysis")
- `--visualize`: Include this flag to visualize 3D grids

### Example

```bash
python src/main_grid.py --model meta-llama/Llama-3.2-1B-Instruct --layer 16 --grid_size 7 7 7 --epochs 30 --visualize
```

## Output

The script generates the following outputs in the specified directory:
- `grid_probe.pt`: Trained grid probe model
- `grid_composition_results.csv`: Results of composition analysis
- `grid_transitivity_results.csv`: Results of transitivity analysis
- Grid visualization images (if `--visualize` is used)

## Analysis Description

### Compositional Analysis

This examines whether complex spatial relationships (like "diagonally above and to the right of") can be accurately represented as the composition of simpler relationships ("above" + "to the right of").

### Transitivity Analysis

This tests whether the model maintains consistency in chained spatial relationships. For example, if A is above B and B is to the right of C, is A correctly positioned relative to C (diagonally above and to the right of C)?

## Visualizing Results

The grid visualizations show the 3D positions of objects with coordinate axes:
- X-axis (red): left-right dimension
- Y-axis (green): back-front dimension 
- Z-axis (blue): down-up dimension

Objects are shown as colored dots with labels, and the spatial relationships between them are indicated by their relative positions in the grid.
