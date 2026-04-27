"""Decoder stubs: raw bytes -> source-symbol stream.

Each decoder is responsible for the *first* stage of representation: turning
a raw byte string into a 1-D integer sequence of source symbols, with a
known `vocab_size`. Tokenization (BPE / LZW / ...) happens *after* this stage
inside `Pipeline`.

Skeleton implementations only -- actual modality logic will be filled in
when the compression_benchmarks experiment is implemented.
"""

import numpy as np


class Decoder:
    """Abstract decoder: bytes -> (1-D int array, vocab_size)."""

    name = "abstract"
    vocab_size = None

    def decode(self, raw_bytes):  # pragma: no cover - abstract
        raise NotImplementedError


class TextUTF8Decoder(Decoder):
    """UTF-8 bytes -> sequence of Unicode codepoints (or bytes if `byte_level`).

    TODO: decide between byte-level (vocab_size=256) and codepoint-level
    decoding once the text-corpus choice for compression_benchmarks is
    finalized.
    """

    name = "text_utf8"

    def __init__(self, byte_level=True):
        self.byte_level = bool(byte_level)
        self.vocab_size = 256 if self.byte_level else None

    def decode(self, raw_bytes):  # pragma: no cover - stub
        if self.byte_level:
            return np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int64)
        raise NotImplementedError("Codepoint-level UTF-8 decoder not yet implemented.")


class AudioPCM16Decoder(Decoder):
    """Raw bytes -> 16-bit PCM samples (signed int16, little-endian).

    Output is in {-2**15, ..., 2**15 - 1}. The compression_benchmarks
    experiment will typically shift these to unsigned 0..65535 before passing
    to a tokenizer / model. TODO when the audio dataset is wired up.
    """

    name = "audio_pcm16"
    vocab_size = 65536

    def decode(self, raw_bytes):  # pragma: no cover - stub
        return np.frombuffer(raw_bytes, dtype="<i2").astype(np.int64)


class ImageRGBDecoder(Decoder):
    """Raw image bytes -> sequence of channel-interleaved RGB byte values.

    TODO: decide on raster order and whether to keep channels separate or
    interleave; depends on the image dataset chosen for benchmarks.
    """

    name = "image_rgb"
    vocab_size = 256

    def decode(self, raw_bytes):  # pragma: no cover - stub
        raise NotImplementedError("Image decoder not yet implemented.")


_REGISTRY = {
    "text_utf8": TextUTF8Decoder,
    "audio_pcm16": AudioPCM16Decoder,
    "image_rgb": ImageRGBDecoder,
}


def get_decoder(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown decoder {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
