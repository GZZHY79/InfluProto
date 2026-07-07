"""
Model definition for InfluProto — prototype-based host prediction.

Architecture:
    8 viral segments (<sep>-joined) → MEGATransformer → Attention Pooling
    → 128-d embedding → cosine distance to 3 learnable prototypes → Host Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AttentionPooling(nn.Module):
    """Attention-based pooling to obtain genome-level embedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: [B, L, D]
            attention_mask: [B, L]
        Returns:
            pooled: [B, D]
            weights: [B, L]
        """
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class HostPrototypeHead(nn.Module):
    """
    Learnable prototype vectors with L2 or cosine distance.

    Each prototype represents a host species. Distance to prototypes
    captures the continuous nature of viral host adaptation.
    """

    def __init__(self, num_hosts: int, hidden_dim: int, metric: str = "cosine"):
        super().__init__()
        self.num_hosts = num_hosts
        self.hidden_dim = hidden_dim
        self.metric = metric

        self.prototypes = nn.Parameter(torch.randn(num_hosts, hidden_dim))
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, h):
        """
        Args:
            h: [B, D] sequence embeddings
        Returns:
            distances: [B, K] distance to each prototype
        """
        if self.metric == "l2":
            h = F.normalize(h, dim=-1)
            prototypes = F.normalize(self.prototypes, dim=-1)
            h2 = (h ** 2).sum(dim=1, keepdim=True)
            c2 = (prototypes ** 2).sum(dim=1)
            hc = h @ prototypes.t()
            distances = h2 - 2 * hc + c2
        elif self.metric == "cosine":
            h_norm = F.normalize(h, dim=-1)
            c_norm = F.normalize(self.prototypes, dim=-1)
            distances = 1.0 - torch.matmul(h_norm, c_norm.t())
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return distances


class HostCapabilityHead(nn.Module):
    """MLP head for multi-label host infectivity prediction."""

    def __init__(self, input_dim: int, num_hosts: int,
                 hidden_dim: int = 256, use_distance: bool = True):
        super().__init__()
        self.use_distance = use_distance
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_hosts),
        )

    def forward(self, x):
        return self.mlp(x)


class HostPredictionModel(nn.Module):
    """
    Full InfluProto model.

    encoder → pooling → prototype head (distances) → capability head (logits)
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_hosts: int,
                 prototype_metric: str = "cosine",
                 capability_use_distance: bool = True,
                 capability_hidden_dim: int = 256,
                 pooling: str = "attention"):
        super().__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.num_hosts = num_hosts

        if pooling == "attention":
            self.pooler = AttentionPooling(hidden_dim)
        elif pooling == "mean":
            self.pooler = None
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.prototype_head = HostPrototypeHead(
            num_hosts=num_hosts, hidden_dim=hidden_dim, metric=prototype_metric)

        cap_input_dim = num_hosts if capability_use_distance else hidden_dim
        self.capability_head = HostCapabilityHead(
            input_dim=cap_input_dim, num_hosts=num_hosts,
            hidden_dim=capability_hidden_dim)
        self.capability_use_distance = capability_use_distance

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooler is not None:
            h, attn_weights = self.pooler(hidden_states, attention_mask)
        else:
            mask = attention_mask.unsqueeze(-1)
            h = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            attn_weights = None

        distances = self.prototype_head(h)

        cap_input = distances if self.capability_use_distance else h
        capability_logits = self.capability_head(cap_input)

        out = {
            "embedding": h,
            "distances": distances,
            "capability_logits": capability_logits,
        }
        if return_attention:
            out["attention_weights"] = attn_weights
        return out
