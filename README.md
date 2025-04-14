# Spatial World Models in LLMs

In this project, we investigate how large language models represent and reason about spatial relationships between objects, with a particular focus on transitivity in spatial relations. We test whether these models maintain consistent spatial relationships through direct and chained spatial relationships.

## Approaches

The project uses two complementary approaches to analyze spatial reasoning in LLMs:

1. **Vector-based approach**: Uses linear probes to extract spatial relationship vectors from model embeddings.
2. **Grid-based approach**: Maps object relationships onto a 3D grid representation to analyze spatial understanding.


## Data

The project uses a set of common objects and their spatial relationships:
- Direct relationships (e.g., "A is above B")
- Chained relationships (e.g., "A is above B and B is to the right of C")
- Composed relationships (e.g., "A is diagonally above and to the right of C")

The spatial relationships are defined in the notebooks and include:
- Basic relationships: "above", "below", "to the right of", "to the left of"
- Composed relationships: "diagonally above and to the right of", "diagonally above and to the left of", etc.

## Experiments

The experiments are conducted in the following notebooks:

1. **`1_inverse.ipynb`**: Creates training data with spatial relationships between objects
2. **`2_compositionality.ipynb`**: Compositionality of spatial relationships
3. **`3_transitivity.ipynb`**: Transitivity of spatial relationships
4. **`4_grid_object_transitivity.ipynb`**: Transitivity with explicit object positioning in 3D grids

## Results

The project investigates:
- How well language models represent spatial relationships
- Whether models maintain transitivity in spatial reasoning
- The effectiveness of linear probes in extracting spatial relationship information
- Comparison between direct and chained spatial relationship representations
- The consistency of object positioning when applying chained spatial relationships

## Requirements

Also, you need GPUs to implement the code efficiently, as it uses the Llama-3.2-3B-Instruct model.


## How to Cite

<!-- ```bibtex
@article{spatialworldmodel2024,
  title={Spatial World Models in LLMs: Investigating Transitivity in Spatial Reasoning},
  author={[]},
  journal={[Journal Name]},
  year={2024}
}
``` -->