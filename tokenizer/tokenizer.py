"""
GPT‑2 style byte‑level tokenizer implementing encode/decode using a trained
vocabulary and merge rules. This mirrors how modern LLM tokenizers operate
internally: text → bytes → unicode mapping → BPE merges → token ids.
"""
import re
from typing import Dict, List, Iterable, Iterator
import json

from .regex import GPT2_PATTERN


# Creates a reversible byte↔unicode mapping so that every possible byte
# value can be represented as a valid unicode character for BPE processing.
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges,  # unused in GPT-2 encode
        special_tokens: List[str] | None = None,
    ):
        self.pattern = GPT2_PATTERN

        # Byte ↔ unicode maps used to transform raw UTF‑8 bytes into symbols BPE can operate on
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Convert stored byte tokens into unicode symbols so they can participate in BPE merges
        self.id_to_symbol = {}
        self.symbol_to_id = {}

        for idx, token_bytes in vocab.items():
            symbol = ''.join(self.byte_encoder[b] for b in token_bytes)
            self.id_to_symbol[idx] = symbol
            self.symbol_to_id[symbol] = idx

        # Build BPE priority ranks from merge rules so the tokenizer merges pairs in the
        # exact order they were learned during BPE training
        self.bpe_ranks = {}
        for rank, (a, b) in enumerate(merges):
            sa = ''.join(self.byte_encoder[x] for x in a)
            sb = ''.join(self.byte_encoder[x] for x in b)
            self.bpe_ranks[(sa, sb)] = rank

        # Register special tokens (e.g. <|endoftext|>) that must be matched before BPE processing
        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

        for tok in self.special_tokens:
            if tok not in self.symbol_to_id:
                idx = len(self.id_to_symbol)
                self.id_to_symbol[idx] = tok
                self.symbol_to_id[tok] = idx


    # -----------------------------------------------------------------------------

    def _bpe(self, token: str) -> List[str]:
        """
        Apply learned BPE merges to a unicode‑mapped token by repeatedly merging
        adjacent symbol pairs according to their learned priority ranks.
        """
        word = tuple(token)
        if len(word) == 0:
            return []

        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            if not pairs:
                break

            min_rank = float("inf")
            best_pair = None
            for pair in pairs:
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

        return list(word)

    # -----------------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(text)

        while i < n:
            # Special tokens are matched before any regex or BPE logic
            matched = False
            for tok in self.special_tokens_sorted:
                if text.startswith(tok, i):
                    ids.append(self.symbol_to_id[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue

            # Process raw text until the next special token boundary
            j = i
            while j < n and not any(text.startswith(tok, j) for tok in self.special_tokens_sorted):
                j += 1

            span = text[i:j]

            # GPT‑2 regex pretokenization over the raw text span
            for piece in self.pattern.findall(span):
                # Convert text piece to raw UTF‑8 bytes
                piece_bytes = piece.encode("utf-8")

                # Map bytes into unicode symbols so BPE can operate at byte level
                mapped = ''.join(self.byte_encoder[b] for b in piece_bytes)

                # Apply BPE merges, then convert resulting symbols into token ids
                tokens = self._bpe(mapped)
                for t in tokens:
                    if t not in self.symbol_to_id:
                        raise KeyError(f"token from BPE '{t}' not found in vocabulary; check your vocab/merges")
                    ids.append(self.symbol_to_id[t])

            i = j

        return ids

    # -----------------------------------------------------------------------------

    # Stream encoding support for large datasets without loading full text into memory
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id

    # -----------------------------------------------------------------------------

    # Reconstruct original text by reversing symbol→byte mapping and decoding UTF‑8
    def decode(self, ids: List[int]) -> str:
        text = ''.join(self.id_to_symbol[i] for i in ids)
        byte_arr = bytearray(self.byte_decoder[c] for c in text)
        return byte_arr.decode("utf-8", errors="replace")
    
    # Utility to load vocabulary and merges from disk and construct a Tokenizer instance
    def load_tokenizer(vocab_path, merges_path, special_tokens=None):
      with open(vocab_path) as f:
        vocab_json = json.load(f)

      with open(merges_path) as f:
        merges_json = json.load(f)

    # latin-1 back to bytes
      vocab = {int(k): v.encode("latin-1") for k, v in vocab_json.items()}
      merges = [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges_json]

      return Tokenizer(vocab, merges, special_tokens)