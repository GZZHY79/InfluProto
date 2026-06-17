"""
Inference utilities for ProtoInflu.

Provides model loading, sequence prediction, FASTA parsing,
and visualization. Usable as both a Python module and CLI script.

CLI usage:
    python -m protoinflu.inference \
        --checkpoint_path model.bin \
        --input sample.fasta \
        --output_dir ./results
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HOST_NAMES = {0: "Human", 1: "Avian", 2: "Swine"}


def load_model(checkpoint_path: str, device: str = "cpu",
               num_hidden_layers: int = 2, vocab_file: str = ""):
    """
    Load the full HostPredictionModel from a checkpoint.

    Args:
        checkpoint_path: Path to .bin checkpoint file.
        device: "cpu" or "cuda".
        num_hidden_layers: Number of MEGATransformer hidden layers.
        vocab_file: Path to vocab.txt. Auto-detected if empty.

    Returns:
        (model, tokenizer) tuple.
    """
    from transformers import MegaConfig, MegaModel
    from protoinflu.model import HostPredictionModel
    from protoinflu.tokenizer import BioTokenizer

    # Auto-detect vocab file
    if not vocab_file:
        vocab_file = os.path.join(os.path.dirname(__file__), "vocab.txt")

    tokenizer = BioTokenizer(vocab_file=vocab_file)
    vocab_size = tokenizer.vocab_size

    config = MegaConfig(vocab_size=vocab_size, num_hidden_layers=num_hidden_layers)
    encoder = MegaModel(config=config)

    model = HostPredictionModel(
        encoder=encoder,
        hidden_dim=128,
        num_hosts=3,
        prototype_metric="cosine",
        capability_use_distance=True,
        capability_hidden_dim=256,
        pooling="attention",
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, sequences: List[str],
            device: str = "cpu") -> Tuple[List[str], np.ndarray]:
    """
    Run inference on a list of viral sequences.

    Each sequence string should be 8 segments joined by <sep>,
    e.g. "PB2_seq<sep>PB1_seq<sep>...<sep>NS_seq"

    Args:
        model: HostPredictionModel.
        tokenizer: BioTokenizer instance.
        sequences: List of <sep>-joined segment strings.
        device: "cpu" or "cuda".

    Returns:
        texts: List of human-readable prediction strings.
        distances: [B, K] numpy array of cosine distances.
    """
    if not sequences:
        return [], np.array([])

    enc = tokenizer(sequences, padding="longest", return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    distances = out["distances"].cpu().numpy()

    texts = []
    for i in range(len(sequences)):
        d = distances[i]
        pred_host = int(np.argmin(d))

        dist_lines = "\n".join(
            f"  Distance to {HOST_NAMES[j]}: {d[j]:.4f}" for j in range(len(d))
        )
        texts.append(
            f"Predicted host: {HOST_NAMES[pred_host]}\n"
            f"Distance to prototypes:\n{dist_lines}"
        )
    return texts, distances


def draw_distance_plot(distances: np.ndarray, output_path: str):
    """
    Draw a dot-plot visualization of distances to three prototypes.

    Args:
        distances: [3] array [d_Human, d_Avian, d_Swine].
        output_path: Path to save the PNG.
    """
    hosts = ["Human", "Avian", "Swine"]
    colors = {"Human": "#e74c3c", "Avian": "#27ae60", "Swine": "#2980b9"}
    dist_dict = {h: float(distances[i]) for i, h in enumerate(hosts)}

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(-0.05, 2.15)
    ax.set_ylim(-0.5, 3.2)
    ax.axis("off")

    ax.text(1.0, 3.05, "Distance to Three Prototypes", fontsize=13, fontweight="bold",
            color="#7f8c8d", ha="center", va="top", family="sans-serif")

    ax.axhline(y=2.3, xmin=0.07, xmax=0.93, color="#bdc3c7", linewidth=1.5, clip_on=False)

    for v in np.arange(0, 2.01, 0.25):
        x = v / 2.0
        is_major = v in (0.0, 0.5, 1.0, 1.5, 2.0)
        lw = 1.2 if is_major else 0.5
        color = "#95a5a6" if is_major else "#dce1e8"
        ax.plot([x, x], [2.3, 2.3 - (0.06 if is_major else 0.04)],
                color=color, linewidth=lw, clip_on=False)
        if is_major:
            ax.text(x, 2.12, f"{v:.1f}", fontsize=9, ha="center", va="top",
                    color="#7f8c8d", family="sans-serif")

    for v in np.arange(0, 2.01, 0.25):
        x = v / 2.0
        ax.axvline(x=x, ymin=0.04, ymax=0.82, color="#f0f3f5", linewidth=0.5)

    row_y = [1.6, 0.9, 0.2]
    for i, h in enumerate(hosts):
        d = dist_dict[h]
        x = d / 2.0
        y = row_y[i]

        ax.text(-0.04, y, h, fontsize=13, fontweight="bold", color=colors[h],
                ha="right", va="center", family="sans-serif")
        ax.scatter(x, y, s=180, c=colors[h], zorder=5, edgecolors="white", linewidth=2)
        ax.text(x + 0.04, y, f"{d:.4f}", fontsize=12, fontweight="bold",
                color=colors[h], ha="left", va="center", family="sans-serif")

    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=150, facecolor="white", edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)


def parse_fasta_file(filepath: str) -> List[str]:
    """
    Parse a FASTA file with 8 segments, return single <sep>-joined string.

    Args:
        filepath: Path to FASTA file.

    Returns:
        List with one string: 8 segments joined by <sep>.

    Raises:
        ValueError: If the file does not contain exactly 8 segments.
    """
    with open(filepath) as f:
        raw = f.read()

    sequences = []
    in_seq = False
    for line in raw.strip().split('\n'):
        s = line.strip()
        if not s:
            continue
        if s.startswith('>'):
            in_seq = True
            sequences.append('')
        elif in_seq:
            sequences[-1] += s.upper()

    sequences = [s for s in sequences if s]
    if len(sequences) != 8:
        raise ValueError(
            f'Expected exactly 8 segments, got {len(sequences)}. '
            'Order: NA, HA, NP, PA, NS, MP, PB1, PB2'
        )
    return ['<sep>'.join(sequences)]


# ── CLI ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ProtoInflu: Predict influenza A virus host from genomic sequences"
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint (.bin)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input FASTA file (8 segments)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results (default: ./results)")
    parser.add_argument("--num_hidden_layers", type=int, default=2,
                        help="Number of MEGATransformer hidden layers (default: 2)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run on: 'cpu' or 'cuda' (default: cpu)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model(
        args.checkpoint_path, args.device,
        num_hidden_layers=args.num_hidden_layers,
    )

    seq_input = parse_fasta_file(args.input)
    texts, dists = predict(model, tokenizer, seq_input, args.device)

    basename = os.path.splitext(os.path.basename(args.input))[0]

    # Save text prediction
    txt_path = os.path.join(args.output_dir, f"{basename}_prediction.txt")
    with open(txt_path, 'w') as f:
        for r in texts:
            f.write(r + "\n")

    # Save distance plot
    if dists.shape[0] > 0:
        png_path = os.path.join(args.output_dir, f"{basename}_prediction.png")
        draw_distance_plot(dists[0], png_path)

    # Print to stdout
    for r in texts:
        print(r)
    print(f"\nResults saved to: {txt_path}")
    if dists.shape[0] > 0:
        print(f"Plot saved to: {png_path}")


if __name__ == "__main__":
    main()
