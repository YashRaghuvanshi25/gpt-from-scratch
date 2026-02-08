"""
Implements Byte-Pair Encoding (BPE) training used to learn vocabulary and
merge rules for the tokenizer. This is used once during tokenizer training
and is not required during model inference.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable
import regex as re


# GPT-2 pretokenization pattern (exact)
# GPT-2 style pretokenization: splits text into byte-level pieces while
# preserving whitespace and punctuation as separate tokens.
GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    """
    Build word frequency table:
        Dict[word_tuple_of_bytes, frequency]
    """
    # ---- split on special tokens (CRITICAL) ----
    if special_tokens:
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    word_freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    # ---- GPT-2 regex pretokenization using finditer ----
    for chunk in chunks:
        for m in GPT2_PATTERN.finditer(chunk):
            piece = m.group(0)
            b = piece.encode("utf-8")
            word = tuple(bytes([x]) for x in b)
            word_freq[word] += 1

    return word_freq


def get_pairs(word: Tuple[bytes, ...]) -> Counter:
    pairs = Counter()
    for i in range(len(word) - 1):
        pairs[(word[i], word[i + 1])] += 1
    return pairs


def train_bpe(
    corpus: str,
    vocab_size: int,
    special_tokens: Iterable[str] | None = None,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:

    special_tokens = list(special_tokens or [])

    # ---- Initialize vocab with 256 bytes + specials ----
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    # ---- Build word frequency table ----
    word_freq = pretokenize(corpus, special_tokens)

    # ---- Build initial pair stats ----
    pair_counts: Counter = Counter()
    pair_to_words: Dict[Tuple[bytes, bytes], set] = defaultdict(set)

    for word, freq in word_freq.items():
        pairs = get_pairs(word)
        for p, c in pairs.items():
            pair_counts[p] += c * freq
            pair_to_words[p].add(word)

    merges: List[Tuple[bytes, bytes]] = []

    # Main BPE merge loop: repeatedly merge the most frequent adjacent byte pairs
    # to build a larger vocabulary.
    while next_id < vocab_size and pair_counts:
        # choose most frequent, lexicographically greater tie-break
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        a, b = best

        merges.append((a, b))
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        affected_words = list(pair_to_words[best])
        pair_counts.pop(best)
        pair_to_words.pop(best)

        updates = defaultdict(int)

        # ---- Replace in all affected words ----
        for word in affected_words:
            freq = word_freq.pop(word)

            # remove old pair contributions
            old_pairs = get_pairs(word)
            for p, c in old_pairs.items():
                pair_counts[p] -= c * freq
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_to_words.pop(p, None)
                else:
                    pair_to_words[p].discard(word)

            # apply merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            updates[new_word] += freq

        # ---- Add updated words back ----
        for new_word, freq in updates.items():
            existed = new_word in word_freq
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            new_pairs = get_pairs(new_word)
            for p, c in new_pairs.items():
                pair_counts[p] += c * freq
                if not existed:
                    pair_to_words[p].add(new_word)

    return vocab, merges