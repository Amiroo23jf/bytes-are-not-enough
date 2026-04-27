"""Decoder + (optional) tokenizer pipeline.

Wraps a `representations.decoders.Decoder` and an optional
`tokenizers.<...>` instance into a single callable. Used by
`compression_benchmarks` (and any future modality-aware experiment) to take a
raw byte file and produce the token id stream the model consumes.
"""


class Pipeline:
    """Chain a decoder with an optional tokenizer.

    Parameters
    ----------
    decoder : representations.decoders.Decoder
        First stage: raw bytes -> source-symbol sequence.
    tokenizer : object with `.encode` / `.decode` / `.vocab_size`, or None
        Optional second stage. Skipped when `tokenizer is None` *or* when
        `decoder.vocab_size <= max_vocab`.
    max_vocab : int or None
        If set, the tokenizer is bypassed when the decoder's vocab is already
        small enough. Lets the same pipeline definition cover both
        "byte-level text (no tokenizer needed, vocab=256)" and "PCM audio
        (vocab=65536, must tokenize)".

    Output stream attributes:
        .vocab_size : effective vocabulary the model sees.
    """

    def __init__(self, decoder, tokenizer=None, max_vocab=None):
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab

    @property
    def uses_tokenizer(self):
        if self.tokenizer is None:
            return False
        if self.max_vocab is None:
            return True
        return self.decoder.vocab_size is not None and (
            self.decoder.vocab_size > self.max_vocab
        )

    @property
    def vocab_size(self):
        if self.uses_tokenizer:
            return self.tokenizer.vocab_size
        return self.decoder.vocab_size

    def __call__(self, raw_bytes):
        seq = self.decoder.decode(raw_bytes)
        if self.uses_tokenizer:
            seq = self.tokenizer.encode(seq)
        return seq
