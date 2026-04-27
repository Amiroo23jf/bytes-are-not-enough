"""Representation add-on.

Front-end of every modality-aware experiment in the repo. A `Pipeline`
converts raw bytes from a file (text, audio, image, ...) into a token id
sequence by composing:

    raw bytes
        -> Decoder            # bytes -> source symbols (e.g. PCM samples,
                              #          UTF-8 codepoints, RGB pixel ints)
        -> Tokenizer (opt.)   # source symbols -> tokens; only applied if
                              #                   the source vocabulary
                              #                   exceeds `max_vocab`

The output stream is what the model (transformer LM, EDPC, TRACE, ...) sees.
This module is the *structural* place where new modalities and new tokenizers
plug in for the compression benchmarks experiment.
"""

from .decoders import (
    AudioPCM16Decoder,
    Decoder,
    ImageRGBDecoder,
    TextUTF8Decoder,
    get_decoder,
)
from .pipeline import Pipeline

__all__ = [
    "Decoder",
    "TextUTF8Decoder",
    "AudioPCM16Decoder",
    "ImageRGBDecoder",
    "get_decoder",
    "Pipeline",
]
