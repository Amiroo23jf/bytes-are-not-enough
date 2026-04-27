"""BPE tokenizer over a binary bit stream (Theorem 3 setup).

Wraps HuggingFace `tokenizers.models.BPE` for *training* the merge table on
a long bit sequence, and exposes two interchangeable encoders for inference:

  - "greedy"   -- pure-Python left-to-right longest-match over a trie built
                  from the trained vocabulary. Matches the analytical setting
                  of Theorem 3.
  - "priority" -- canonical BPE: delegate `encode` to the trained HF
                  `Tokenizer` instance (priority-based merge application).

Both encoders share the same vocabulary, `decode`, and `token_bit_lengths`;
only segmentation of the input differs.

Bit-stream glue: bits are passed to/from the HF tokenizer as the literal
characters `"0"` / `"1"`. `encode(np.ndarray[uint8])` joins bits into one
string, feeds it to the active encoder, and returns int64 ids.
`decode(np.ndarray[int64])` returns a uint8 bit array via per-id
`id_to_bits` lookup (built once at construction time).

Determinism: `os.environ["TOKENIZERS_PARALLELISM"] = "false"` is set before
training so the HF trainer is single-threaded. `seed` controls (a) chunk
start positions when `train_bits` is given, and (b) HF BPE tie-breaks (the
latter empirically a non-issue on bit streams). Given identical
`(sequence, vocab_size, encoder, seed, train_bits, chunk_size)` the
resulting tokenizer -- and every subsequent `encode(...)` output -- is
bit-for-bit reproducible.

Subsampled training: pass `train_bits=N` to `BPETokenizer.train(...)` to
fit the merge table on `~N` randomly placed contiguous chunks of
`chunk_size` bits drawn from `sequence`. Pair frequencies in a stationary
Markov source converge quickly, so `N` much smaller than `len(sequence)`
typically yields the same merge decisions at a fraction of the cost.
Encoding still operates on the full sequence the caller passes to
`encode(...)`, independently of `train_bits`.
"""

import json
import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def _bits_to_string(bits):
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.size == 0:
        return ""
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Input bits must be 0 or 1")
    return (arr + ord("0")).astype(np.uint8).tobytes().decode("ascii")


def _string_to_bits(s):
    if not s:
        return np.empty(0, dtype=np.uint8)
    out = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")
    if not np.all((out == 0) | (out == 1)):
        raise ValueError(f"Token string contains non-0/1 chars: {s!r}")
    return out.astype(np.uint8)


def _build_trie(id_to_bits):
    """Build a 0/1 trie. Each node is a list `[child0, child1, token_id]`.

    `token_id` is None unless the path from the root spells a vocabulary
    string.
    """
    root = [None, None, None]
    for tid, bits in enumerate(id_to_bits):
        node = root
        for b in bits.tolist():
            child = node[b]
            if child is None:
                child = [None, None, None]
                node[b] = child
            node = child
        node[2] = tid
    return root


def _greedy_encode_bits(bits, trie):
    """Left-to-right longest-match BPE-vocabulary encoding via a trie.

    The base alphabet `{0, 1}` is always in the trained vocabulary, so each
    step is guaranteed to advance by at least one bit; raises if not (which
    would indicate a corrupt vocabulary).
    """
    n = int(bits.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    bits_list = bits.astype(np.uint8).tolist()
    out = []
    pos = 0
    while pos < n:
        node = trie
        last_id = None
        last_pos = pos
        i = pos
        while i < n:
            b = bits_list[i]
            child = node[b]
            if child is None:
                break
            node = child
            i += 1
            if node[2] is not None:
                last_id = node[2]
                last_pos = i
        if last_id is None:
            raise ValueError(
                f"No matching token at bit position {pos}; vocabulary likely "
                f"missing the base alphabet {{0, 1}}."
            )
        out.append(last_id)
        pos = last_pos
    return np.asarray(out, dtype=np.int64)


class BPETokenizer:
    """Bit-level BPE with selectable greedy or priority encoder."""

    name = "bpe"

    def __init__(
        self,
        *,
        id_to_bits,
        vocab_size,
        encoder="greedy",
        seed=None,
        hf_tokenizer=None,
        hf_json=None,
    ):
        if encoder not in ("greedy", "priority"):
            raise ValueError(
                f"encoder must be 'greedy' or 'priority', got {encoder!r}"
            )
        if encoder == "priority" and hf_tokenizer is None:
            raise ValueError("priority encoder requires an hf_tokenizer instance")

        self.vocab_size = int(vocab_size)
        self.encoder = encoder
        self.seed = seed
        self.id_to_bits = [np.asarray(b, dtype=np.uint8) for b in id_to_bits]
        if len(self.id_to_bits) != self.vocab_size:
            raise ValueError(
                f"len(id_to_bits)={len(self.id_to_bits)} != vocab_size={self.vocab_size}"
            )
        if any(b.size == 0 for b in self.id_to_bits):
            raise ValueError("Empty token strings are not allowed")
        for b in self.id_to_bits:
            if not np.all((b == 0) | (b == 1)):
                raise ValueError("Token bits must be 0 or 1")

        self._token_bit_lengths = np.asarray(
            [b.size for b in self.id_to_bits], dtype=np.int64
        )
        self._hf_tokenizer = hf_tokenizer
        self._hf_json = hf_json
        self._trie = _build_trie(self.id_to_bits) if encoder == "greedy" else None

    @classmethod
    def train(
        cls,
        sequence,
        vocab_size,
        encoder="greedy",
        seed=None,
        train_bits=None,
        chunk_size=8192,
    ):
        """Train a BPE tokenizer on the bit `sequence` and return an instance.

        Parameters
        ----------
        sequence : np.ndarray of uint8 / int (entries in {0, 1})
        vocab_size : int >= 2
        encoder : {"greedy", "priority"}
        seed : Optional[int]
            Controls chunk start positions (when `train_bits` is given) and
            HF BPE tie-breaks. Stored on the returned instance.
        train_bits : Optional[int]
            If given and < `len(sequence)`, fit the merge table on
            ~`train_bits` bits drawn as randomly placed contiguous chunks
            of `chunk_size` bits each (using a `default_rng(seed)`); each
            chunk is passed to HF as a separate document so merges never
            cross chunk seams. Default `None` -> train on the full
            sequence (legacy behaviour).
        chunk_size : int
            Length of each contiguous training chunk in bits. Should be
            comfortably larger than the source's Markov order so each chunk
            sees stationary statistics.
        """
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        if encoder not in ("greedy", "priority"):
            raise ValueError(
                f"encoder must be 'greedy' or 'priority', got {encoder!r}"
            )
        if train_bits is not None and int(train_bits) <= 0:
            raise ValueError(f"train_bits must be > 0 if given, got {train_bits}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        seq = np.asarray(sequence, dtype=np.uint8)
        if seq.size == 0:
            raise ValueError("Cannot train on empty sequence")
        if not np.all((seq == 0) | (seq == 1)):
            raise ValueError("Sequence must contain only 0s and 1s")

        if train_bits is None or int(train_bits) >= seq.size:
            docs = [_bits_to_string(seq)]
        else:
            cs = min(int(chunk_size), seq.size)
            n_chunks = max(1, (int(train_bits) + cs - 1) // cs)
            rng = np.random.default_rng(seed if seed is not None else 0)
            max_start = seq.size - cs
            starts = rng.integers(0, max_start + 1, size=n_chunks)
            docs = [_bits_to_string(seq[s : s + cs]) for s in starts.tolist()]

        hf_tok = Tokenizer(BPE())
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            initial_alphabet=["0", "1"],
            special_tokens=[],
            show_progress=False,
            min_frequency=0,
        )
        hf_tok.train_from_iterator(docs, trainer=trainer)

        vocab = hf_tok.get_vocab()
        n = len(vocab)
        id_to_bits = [None] * n
        for s, i in vocab.items():
            if not (0 <= i < n):
                raise RuntimeError(f"Token id {i} out of range [0, {n})")
            id_to_bits[i] = _string_to_bits(s)
        for i, b in enumerate(id_to_bits):
            if b is None:
                raise RuntimeError(f"Vocabulary missing token id {i}")

        return cls(
            id_to_bits=id_to_bits,
            vocab_size=n,
            encoder=encoder,
            seed=seed,
            hf_tokenizer=hf_tok,
            hf_json=hf_tok.to_str(),
        )

    def token_bit_lengths(self):
        """Length-V int64 array; entry `i` = number of bits in token `i`."""
        return self._token_bit_lengths.copy()

    def encode(self, sequence):
        """Encode a bit array (uint8) to int64 token ids."""
        seq = np.asarray(sequence, dtype=np.uint8)
        if seq.size == 0:
            return np.empty(0, dtype=np.int64)
        if not np.all((seq == 0) | (seq == 1)):
            raise ValueError("Sequence must contain only 0s and 1s")
        if self.encoder == "greedy":
            return _greedy_encode_bits(seq, self._trie)
        text = _bits_to_string(seq)
        enc = self._hf_tokenizer.encode(text)
        return np.asarray(enc.ids, dtype=np.int64)

    def decode(self, tokens):
        """Decode int64 token ids back to a uint8 bit array (lossless)."""
        toks = np.asarray(tokens, dtype=np.int64)
        if toks.size == 0:
            return np.empty(0, dtype=np.uint8)
        if toks.min() < 0 or toks.max() >= self.vocab_size:
            raise ValueError(
                f"token id out of range [0, {self.vocab_size}): "
                f"min={int(toks.min())}, max={int(toks.max())}"
            )
        return np.concatenate(
            [self.id_to_bits[int(t)] for t in toks]
        ).astype(np.uint8)

    @staticmethod
    def _hf_companion_path(path):
        base, _ = os.path.splitext(path)
        return base + "_hf.json"

    def save(self, path):
        """Persist tokenizer state.

        Writes a JSON sidecar at `path` plus, if available, the native HF
        tokenizer JSON at `<stem>_hf.json` so the priority encoder can be
        reconstituted on load without retraining.
        """
        sidecar = {
            "name": self.name,
            "vocab_size": int(self.vocab_size),
            "encoder": self.encoder,
            "seed": self.seed,
            "id_to_bits": [b.tolist() for b in self.id_to_bits],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(sidecar, f, indent=2)
        if self._hf_json is not None:
            hf_path = self._hf_companion_path(path)
            with open(hf_path, "w") as f:
                f.write(self._hf_json)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            sidecar = json.load(f)
        id_to_bits = [np.asarray(b, dtype=np.uint8) for b in sidecar["id_to_bits"]]
        hf_path = cls._hf_companion_path(path)
        hf_tok = None
        hf_json = None
        if os.path.exists(hf_path):
            with open(hf_path, "r") as f:
                hf_json = f.read()
            hf_tok = Tokenizer.from_str(hf_json)
        return cls(
            id_to_bits=id_to_bits,
            vocab_size=int(sidecar["vocab_size"]),
            encoder=sidecar["encoder"],
            seed=sidecar.get("seed"),
            hf_tokenizer=hf_tok,
            hf_json=hf_json,
        )
