"""Tokenizer wrappers for the WikiText-103 span-distribution experiment.

All wrappers expose:
    .name        str
    .vocab_size  int
    .encode_with_char_lengths(text) -> (ids: int32 ndarray, char_lens: int32 ndarray)

char_lens[i] is the number of Unicode characters in the original text that
token i accounts for. For the offset-based wrappers (HFTokenizer,
TransformersTokenizer) the lengths partition the text exactly, so
char_lens.sum() == len(text): characters that no token's offset span covers
(e.g. the whitespace that BertPreTokenizer drops) are assigned to the next
token. See `_char_lens_from_offsets`.

Factory
-------
    build_tokenizer(tok_cfg, train_text, artifacts_dir) -> TokenizerWrapper
"""

import os
import tempfile

import numpy as np


def _char_lens_from_offsets(offsets, n_chars):
    """Per-token character lengths that partition a chunk of `n_chars` chars.

    Token i is assigned the characters after the previous token's end up to
    its own end, i.e. (end_{i-1}, end_i]. Any characters skipped between
    tokens (whitespace dropped by WordPiece/BERT pre-tokenization) therefore
    count toward the following token, as GPT-2-style tokenizers do with their
    leading-space tokens, and the last token also absorbs any trailing
    characters. Ends are made non-decreasing first, so tokens sharing one
    character (byte-level pieces of a multi-byte character) get length 0
    after the first. Sum of the result is exactly `n_chars`.

    Raw `end - start` would instead silently drop those skipped characters,
    understating both the span of a token window and the character count used
    to normalize loss to bits/char.
    """
    ends = np.fromiter((e for _, e in offsets), dtype=np.int64, count=len(offsets))
    np.maximum.accumulate(ends, out=ends)
    lens = np.diff(ends, prepend=0)
    lens[-1] += n_chars - ends[-1]
    return lens.astype(np.int32)


def _encode_chunked(text, chunk_chars, encode_chunk):
    """Encode `text` chunk by chunk with `encode_chunk(chunk) -> (ids, offsets)`.

    Returns (ids, char_lens) with char_lens.sum() == len(text). A chunk that
    produces no tokens (whitespace only) carries its characters over to the
    next token.
    """
    ids_list, clens_list = [], []
    carry = 0
    for start in range(0, len(text), chunk_chars):
        chunk = text[start : start + chunk_chars]
        ids, offsets = encode_chunk(chunk)
        if not len(ids):
            carry += len(chunk)
            continue
        clens = _char_lens_from_offsets(offsets, len(chunk))
        clens[0] += carry
        carry = 0
        ids_list.append(np.asarray(ids, dtype=np.int32))
        clens_list.append(clens)
    if carry and clens_list:
        clens_list[-1][-1] += carry
    return np.concatenate(ids_list), np.concatenate(clens_list)

# ---------------------------------------------------------------------------
# Character tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """One token per Unicode code point; char_len is always 1."""

    def __init__(self, vocab):
        self.name = "char"
        self._c2i = {c: i for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def train(cls, text):
        return cls(sorted(set(text)))

    def encode_with_char_lengths(self, text):
        # Build a code-point → vocab-id lookup table (one per call, negligible size).
        max_cp = max(ord(c) for c in self._c2i)
        lut = np.zeros(max_cp + 1, dtype=np.int32)
        for c, idx in self._c2i.items():
            lut[ord(c)] = idx
        # text.encode("utf-32-le") gives 4 bytes per Unicode code point; view as uint32.
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        ids = lut[np.minimum(codes, max_cp)]
        return ids, np.ones(len(ids), dtype=np.int32)


# ---------------------------------------------------------------------------
# HuggingFace `tokenizers` library  (BPE, WordPiece)
# ---------------------------------------------------------------------------

class HFTokenizer:
    """Wraps a tokenizers.Tokenizer; uses encoding offsets for char lengths."""

    def __init__(self, name, tokenizer):
        self.name = name
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def encode_with_char_lengths(self, text, chunk_chars=100_000):
        # Encode in chunks so enc.ids never becomes a multi-GB Python list.
        def encode_chunk(chunk):
            enc = self._tok.encode(chunk)
            return enc.ids, enc.offsets

        return _encode_chunked(text, chunk_chars, encode_chunk)

    @classmethod
    def train_bpe(cls, name, lines, vocab_size):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>"],
            show_progress=True,
        )
        tok.train_from_iterator(lines, trainer)
        return cls(name, tok)

    @classmethod
    def train_wordpiece(cls, name, lines, vocab_size):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.normalizer = normalizers.BertNormalizer(lowercase=False)
        tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            show_progress=True,
        )
        tok.train_from_iterator(lines, trainer)
        return cls(name, tok)


# ---------------------------------------------------------------------------
# SentencePiece
# ---------------------------------------------------------------------------

class SPTokenizer:
    """Wraps sentencepiece; derives char lengths from decoded piece strings."""

    # U+2581 LOWER ONE EIGHTH BLOCK is SentencePiece's whitespace placeholder.
    _SP_SPACE = "▁"

    def __init__(self, name, sp, model_path):
        self.name = name
        self._sp = sp
        self._model_path = model_path
        self.vocab_size = sp.GetPieceSize()

    def encode_with_char_lengths(self, text, chunk_chars=100_000):
        # Encode in chunks so piece lists never become multi-GB Python objects.
        ids_list, clens_list = [], []
        for start in range(0, len(text), chunk_chars):
            pieces = self._sp.encode_as_pieces(text[start : start + chunk_chars])
            if pieces:
                ids_list.append(
                    np.array([self._sp.piece_to_id(p) for p in pieces], dtype=np.int32)
                )
                clens_list.append(
                    np.array(
                        [len(p.replace(self._SP_SPACE, " ")) for p in pieces],
                        dtype=np.int32,
                    )
                )
        return np.concatenate(ids_list), np.concatenate(clens_list)

    @classmethod
    def train(cls, name, text, vocab_size, model_dir):
        import sentencepiece as spm
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            tmpfile = f.name
        model_prefix = os.path.join(model_dir, name)
        spm.SentencePieceTrainer.train(
            input=tmpfile,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=-1,
            eos_id=-1,
            hard_vocab_limit=False,
            minloglevel=2,
        )
        os.unlink(tmpfile)
        sp = spm.SentencePieceProcessor()
        sp.Load(model_prefix + ".model")
        return cls(name, sp, model_prefix + ".model")


# ---------------------------------------------------------------------------
# HuggingFace Transformers  (pre-trained BERT, GPT-2)
# ---------------------------------------------------------------------------

# Process text in chunks to stay well within any model_max_length limit.
_CHUNK_CHARS = 50_000


class TransformersTokenizer:
    """Wraps a transformers AutoTokenizer; uses offset_mapping for char lengths."""

    def __init__(self, name, tokenizer):
        self.name = name
        self._tok = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def _encode_chunk(self, text):
        enc = self._tok(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        return enc["input_ids"], enc["offset_mapping"]

    def encode_with_char_lengths(self, text):
        return _encode_chunked(text, _CHUNK_CHARS, self._encode_chunk)

    @classmethod
    def from_pretrained(cls, name, model_name):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        return cls(name, tok)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tokenizer(tok_cfg, train_text, artifacts_dir):
    """Instantiate (and train if needed) a tokenizer from a config dict.

    Parameters
    ----------
    tok_cfg : dict
        Must contain 'name' and 'type'. Type-specific keys:
          bpe / wordpiece / sentencepiece : vocab_size (int)
          pretrained                      : model (str, HF model id)
    train_text : str
        Full training corpus text (used for trained tokenizers).
    artifacts_dir : str
        Directory for saving trained tokenizer artefacts (SentencePiece models).
    """
    name = tok_cfg["name"]
    typ = tok_cfg["type"]
    os.makedirs(artifacts_dir, exist_ok=True)

    if typ == "character":
        print(f"  [{name}] building character tokenizer ...", flush=True)
        return CharTokenizer.train(train_text)

    elif typ == "bpe":
        V = int(tok_cfg["vocab_size"])
        print(f"  [{name}] training BPE tokenizer (vocab_size={V}) ...", flush=True)
        lines = [l for l in train_text.split("\n") if l.strip()]
        return HFTokenizer.train_bpe(name, lines, V)

    elif typ == "wordpiece":
        V = int(tok_cfg["vocab_size"])
        print(f"  [{name}] training WordPiece tokenizer (vocab_size={V}) ...", flush=True)
        lines = [l for l in train_text.split("\n") if l.strip()]
        return HFTokenizer.train_wordpiece(name, lines, V)

    elif typ == "sentencepiece":
        V = int(tok_cfg["vocab_size"])
        print(f"  [{name}] training SentencePiece tokenizer (vocab_size={V}) ...", flush=True)
        return SPTokenizer.train(name, train_text, V, artifacts_dir)

    elif typ == "pretrained":
        model = tok_cfg["model"]
        print(f"  [{name}] loading pre-trained tokenizer '{model}' ...", flush=True)
        return TransformersTokenizer.from_pretrained(name, model)

    else:
        raise ValueError(f"Unknown tokenizer type: {typ!r}")
