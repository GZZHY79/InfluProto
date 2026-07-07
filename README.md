# InfluProto: Prototype Learning for Influenza A Virus Host Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Predicting cross-species transmission of influenza A virus via **prototype learning** — modeling host adaptation as a **continuous evolutionary process** rather than a discrete classification problem.

## Overview

Traditional classification models force discrete host assignments (Human / Avian / Swine), but viral host adaptation is inherently **continuous** — intermediate strains can infect multiple hosts simultaneously.

InfluProto learns three **learnable prototype vectors** (one per host species) and computes cosine distance from the viral genome embedding to each prototype. The prototype distance serves as a **quantifiable metric** for cross-species transmission potential.

```
8 viral segments → MEGATransformer → Attention Pooling → 128-d embedding → cos(z, p_k) → Host Prediction
```

### Key Features

- **Prototype-based prediction**: Cosine distance to learnable host prototypes replaces discrete classification
- **Continuous risk metric**: Prototype distance quantifies cross-species transmission potential
- **Interpretable**: Attention weights reveal host-specific genomic patterns
- **Lightweight inference**: Single command-line interface for predictions

## Installation

```bash
git clone https://github.com/GZZHY79/FluProto.git
cd FluProto
pip install -r requirements.txt
```

### Requirements

- Python = 3.10.0
- PyTorch = 2.10.0
- Transformers = 4.57.1
- NumPy, Pandas, Matplotlib, scikit-learn

## Quick Start

### Command-Line Inference

```bash
python -m InfluProto.inference \
    --checkpoint_path /path/to/model.bin \
    --input example.fasta \
    --output_dir ./results \
    --device cuda
```

**Input format**: A FASTA file with exactly 8 influenza A genomic segments in order:
`NA → HA → NP → PA → NS → MP → PB1 → PB2`

**Output**: Predicted host (Human / Avian / Swine), distance to each prototype, and a distance visualization plot.

### Python API

```python
from InfluProto import load_model, predict, parse_fasta_file

# Load model
model, tokenizer = load_model("path/to/checkpoint.bin", device="cpu")

# Parse FASTA and predict
sequences = parse_fasta_file("example.fasta")
texts, distances = predict(model, tokenizer, sequences)

for t in texts:
    print(t)
```

## Model Architecture

| Component | Description |
|-----------|-------------|
| **Encoder** | MEGATransformer (pretrained on influenza genomes) |
| **Pooling** | Attention-based pooling → 128-dimensional embedding |
| **Prototype Head** | 3 learnable prototype vectors (Human, Avian, Swine) with cosine distance |
| **Capability Head** | MLP for multi-label host infectivity scoring |

### Prototype Learning

Instead of a softmax classifier, the model learns prototype vectors $p_h, p_c, p_s$ and computes:

$$d_k = 1 - \cos(z, p_k)$$

where $z$ is the viral genome embedding. The predicted host is $\arg\min_k d_k$.

This formulation naturally captures intermediate strains: zoonotic viruses fall **between** two host prototypes in the distance space.

## Repository Structure

```
FluProto/
├── README.md
├── LICENSE
├── requirements.txt
├── example.fasta                   # Example input (EPI_ISL_277234)
└── InfluProto/                     # Core Python package
    ├── __init__.py
    ├── model.py                    # HostPredictionModel, prototype heads
    ├── tokenizer.py                # Custom BioTokenizer
    ├── inference.py                # Inference utilities + CLI
    └── vocab.txt                   # Tokenizer vocabulary (11 tokens)
```

## Checkpoint

Model checkpoints are available at:
<!-- TODO: Add download link (Zenodo, Figshare, or Google Drive) -->

The checkpoint is a standard PyTorch `.bin` file containing the full model state dict, including:
- MEGATransformer encoder weights
- Attention pooling layer
- Host prototype vectors
- Capability head MLP

## Citation

If you use InfluProto in your research, please cite:

```bibtex
@article{geng2025influproto,
  title   = {Predicting Cross-Species Transmission of Influenza A Virus via Prototype Learning},
  author  = {Geng, Zongyi},
  journal = {TBD},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Zongyi Geng (耿宗依)
- **Institution**: CNCB-NGDC
- **Advisor**: Prof. Shuhui Song, Dr. Lun Li
