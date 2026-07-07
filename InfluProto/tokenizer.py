"""
Custom tokenizer for influenza A virus genomic sequences.

Vocabulary (11 tokens): <s>, <pad>, <unk>, <mask>, <sep>, </s>, A, C, G, T, N

Input: 8 segments joined by <sep>, with nucleotide characters already
separated by spaces (pre-tokenized). The tokenizer splits on whitespace
and maps each token to its vocabulary id.

Example input:
    "NA_seq<sep>HA_seq<sep>NP_seq<sep>PA_seq<sep>NS_seq<sep>MP_seq<sep>PB1_seq<sep>PB2_seq"
"""

import os
from typing import List, Optional

from transformers import PreTrainedTokenizer


def _load_vocab(vocab_file: str) -> List[str]:
    with open(vocab_file) as f:
        return [line.strip() for line in f]


class BioTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        cls_token: str = "<s>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        eos_token: str = "</s>",
        sep_token: str = "<sep>",
        **kwargs,
    ):
        if vocab_file is None:
            vocab_file = os.path.join(os.path.dirname(__file__), "vocab.txt")

        self._all_tokens = _load_vocab(vocab_file)
        self._id_to_token = dict(enumerate(self._all_tokens))
        self._token_to_id = {tok: i for i, tok in enumerate(self._all_tokens)}

        super().__init__(
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            sep_token=sep_token,
            **kwargs,
        )

        # Register all tokens as no-split so they are matched atomically
        self.unique_no_split_tokens = self._all_tokens
        self._update_trie(self.unique_no_split_tokens)

    # ── core abstract methods ──────────────────────────────────────────

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> str:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return text.split()

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    # ── special-token wiring ───────────────────────────────────────────

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Prepend <s> and append </s> (or <sep> for single sequence)."""
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [1 if t in self.all_special_ids else 0 for t in token_ids_0]
        mask = [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask
