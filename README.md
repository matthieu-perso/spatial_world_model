# Linear Spatial World Models in Large Language Models

This repository investigates how large language models (LLMs) represent and reason about spatial relationships between objects. We focus on whether LLMs encode linear spatial world models and maintain transitivity and compositionality in spatial reasoning.


---

## Experiments

- **Inverse Analysis:** Are "above" and "below" (or "left" and "right") represented as geometric inverses?
- **Compositionality:** Are diagonal or compound relations (e.g., "diagonally above and to the right") vector sums of basic relations?
- **Steering:** Steering the model based on the subspace

All experiments can be run via scripts in `src/` or interactively in the provided notebooks.

---

## Code Structure 

- `src/`: Source code for the experiments.
- `notebooks/`: Jupyter notebooks for the experiments.
- `data/`: Data for the experiments.

Main analysis can be found in the notebooks:
- `notebooks/1_inverse.ipynb`: Inverse analysis
- `notebooks/2_compositionality.ipynb`: Compositionality analysis
- `notebooks/3_steering.ipynb`: Steering analysis



---

## Results

- LLMs encode spatial relationships as linear structures in their embeddings.
- Linear probes and grid probes can extract and visualize these relationships.
- The models maintain transitivity and compositionality in spatial reasoning.

---

## Citation

If you use this work, please cite:

```bibtex
@article{tehenan2025linear,
  title={Linear Spatial World Models Emerge in Large Language Models},
  author={Tehenan, Matthieu and Moya, Christian Bolivar and Long, Tenghai and Lin, Guang},
  journal={arXiv preprint arXiv:2506.02996},
  year={2025}
}
```

---

## License

MIT License
