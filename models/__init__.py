"""Model definitions, modality-agnostic.

Members:
    transformer_lm.TransformerLM -- vanilla causal transformer LM, used by the
        fragmentation_transformer and tokenization_transformer experiments and
        wrapped by compressors.transformer.
"""

from .transformer_lm import TransformerLM

__all__ = ["TransformerLM"]
