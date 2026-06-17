"""
ProtoInflu — Prototype Learning for Influenza A Virus Host Prediction.

Predicts host tropism (Human, Avian, Swine) using learnable prototype vectors
and cosine distance, capturing the continuous nature of viral host adaptation.
"""

from .model import HostPredictionModel, AttentionPooling, HostPrototypeHead, HostCapabilityHead
from .tokenizer import BioTokenizer
from .inference import load_model, predict, parse_fasta_file, draw_distance_plot

__version__ = "0.1.0"
