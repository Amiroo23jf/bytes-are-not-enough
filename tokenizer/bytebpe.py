"""ByteLevel BPE over file bytes (latin1 round-trip), with frequency-ordered IDs.

This module trains a tokenizer, encodes data, and remaps token IDs by corpus
frequency. It does **not** read or write files except those opened by callers
via *input_filename* for training/encoding; it does not persist models or
arrays — callers (e.g. ``datasets.text_pipeline``) save artifacts.
"""

import json
import logging
import os
import random

import numpy as np
from tokenizers import ByteLevelBPETokenizer, Tokenizer


def decode_bytes(tokenizer: Tokenizer, ids: np.ndarray) -> bytes:
    """Decode frequency-ordered token IDs back to raw file bytes.

    Training and encoding use latin1 to map bytes ↔ strings; this inverts that.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {ids.shape}")
    return tokenizer.decode(ids.tolist()).encode("latin1")


def tokenize_data(
    input_filename,
    vocab_size: int = 16000,
    train_sample_size: int | None = None,
    train_num_segments: int = 40,
    encode_chunk_bytes: int = 5_000_000,
    encode_batch_segments: int = 4,
    base_dir: str | None = None,
):
    """
    Train a ByteLevel BPE tokenizer on sampled segments of the input file,
    then encode the *entire* file in a single streaming pass.

    After encoding, token IDs are **remapped** so that the most frequent
    token in the data gets ID 0, the second-most-frequent gets ID 1, etc.
    This frequency-ordered labelling is required by adaptive-softmax layers.

    Parameters
    ----------
    input_filename : str
        Path to the input file, or a name relative to ``base_dir`` when set.
    base_dir : str or None
        If set, relative ``input_filename`` is resolved under this directory.
        If ``None``, ``input_filename`` is used as-is (must be absolute or
        cwd-relative).

    Returns
    -------
    data_arr : np.ndarray (int64, 1D)
        All token IDs for the full file (frequency-ordered).
    tokenizer : tokenizers.Tokenizer
        The tokenizer with its vocabulary remapped to frequency order.
    unigram_probs : np.ndarray (float64, 1D, length = vocab_size_actual)
        ``unigram_probs[i]`` is the probability of token with (new) ID ``i``.
        Sorted in descending order (most frequent first).
    old_to_new : np.ndarray (int64, 1D)
        ``old_to_new[old_id] = new_id`` for the frequency remap.
    sorted_old_ids : np.ndarray (int64, 1D)
        ``sorted_old_ids[new_id] = old_id`` (descending frequency of old IDs).
    """

    tokenizer = ByteLevelBPETokenizer()
    if os.path.isabs(input_filename):
        file_path = input_filename
    elif base_dir is None:
        file_path = input_filename
    else:
        file_path = os.path.join(base_dir, input_filename)

    file_path = os.path.abspath(file_path)
    file_size = os.path.getsize(file_path)
    encoding = "latin1"

    logging.info(f"File size: {file_size/1024/1024:.2f} MB")

    # ----- Training: sample segments across the file -----
    if train_sample_size is None:
        # Use the whole file but still spread out in segments
        total_train_bytes = file_size
    else:
        total_train_bytes = min(train_sample_size, file_size)

    # Ensure at least 1 byte per segment
    train_num_segments = max(1, train_num_segments)
    per_seg_bytes = max(1, total_train_bytes // train_num_segments)
    # Adjust total bytes to be an integer multiple of per_seg_bytes
    total_train_bytes = per_seg_bytes * train_num_segments

    logging.info(
        f"Training using ~{total_train_bytes/1024/1024:.2f} MB "
        f"from {train_num_segments} segments "
        f"({per_seg_bytes/1024/1024:.4f} MB each)."
    )

    # Seed RNG for reproducible segment sampling (and thus reproducible tokenizer)
    random.seed(1337)

    segments_bytes = []
    with open(file_path, "rb") as f:
        for i in range(train_num_segments):
            # Define a window for this segment
            window_start = (i * file_size) // train_num_segments
            # Make sure we have space for per_seg_bytes
            max_start = max(0, file_size - per_seg_bytes)
            window_end = min(max_start, ((i + 1) * file_size) // train_num_segments)

            if window_end <= window_start:
                start = window_start
            else:
                start = random.randint(window_start, window_end)

            f.seek(start)
            seg_bytes = f.read(per_seg_bytes)
            segments_bytes.append(seg_bytes)
            logging.debug(
                f"Segment {i+1}/{train_num_segments}: start={start}, "
                f"len={len(seg_bytes)} bytes"
            )

    # Decode segments and train
    text_segments = [seg.decode(encoding) for seg in segments_bytes]
    total_chars = sum(len(s) for s in text_segments)
    logging.info(
        f"Training tokenizer on {total_chars} characters "
        f"from {train_num_segments} segments..."
    )
    tokenizer.train_from_iterator(text_segments, vocab_size=vocab_size, min_frequency=2)
    logging.info("Successfully trained tokenizer")

    # Free training data — no longer needed
    del segments_bytes, text_segments

    # ----- Encoding: stream full file, collect IDs in memory -----
    encode_chunk_bytes = max(1, encode_chunk_bytes)
    encode_batch_segments = max(1, encode_batch_segments)

    logging.info(
        f"Encoding full file in chunks of {encode_chunk_bytes/1024/1024:.2f} MB, "
        f"batch size {encode_batch_segments}."
    )

    id_chunks: list[np.ndarray] = []
    total_tokens = 0
    texts_batch: list[str] = []
    chunks_seen = 0

    with open(file_path, "rb") as f:
        while True:
            chunk_bytes = f.read(encode_chunk_bytes)
            if not chunk_bytes:
                break

            text_chunk = chunk_bytes.decode(encoding)
            texts_batch.append(text_chunk)
            chunks_seen += 1

            if len(texts_batch) >= encode_batch_segments:
                encodings = tokenizer.encode_batch(texts_batch)
                for e in encodings:
                    arr = np.array(e.ids, dtype=np.int64)
                    total_tokens += len(arr)
                    id_chunks.append(arr)
                del encodings, texts_batch
                texts_batch = []
                logging.debug(
                    f"Encoded {chunks_seen} chunks so far, "
                    f"total tokens: {total_tokens}"
                )

    # Encode any remaining chunks
    if texts_batch:
        encodings = tokenizer.encode_batch(texts_batch)
        for e in encodings:
            arr = np.array(e.ids, dtype=np.int64)
            total_tokens += len(arr)
            id_chunks.append(arr)
        del encodings, texts_batch
        logging.debug(f"Encoded final chunks, total tokens: {total_tokens}")

    data_arr = np.concatenate(id_chunks)
    del id_chunks
    logging.info(
        f"Tokenization complete. Total tokens: {data_arr.shape[0]}, "
        f"dtype: {data_arr.dtype}"
    )

    # ----- Remap token IDs to frequency order -----
    data_arr, tokenizer, unigram_probs, old_to_new, sorted_old_ids = (
        _remap_ids_by_frequency(data_arr, tokenizer)
    )

    return data_arr, tokenizer, unigram_probs, old_to_new, sorted_old_ids


def _remap_ids_by_frequency(
    data_arr: np.ndarray,
    tokenizer: ByteLevelBPETokenizer,
) -> tuple[np.ndarray, Tokenizer, np.ndarray, np.ndarray, np.ndarray]:
    """Remap token IDs so that ID 0 is the most frequent token, etc.

    Parameters
    ----------
    data_arr : np.ndarray (int64, 1D)
        Token IDs produced by *tokenizer*.
    tokenizer : ByteLevelBPETokenizer
        The trained (original merge-order) tokenizer.

    Returns
    -------
    data_arr : np.ndarray
        Remapped IDs (frequency order).
    new_tokenizer : tokenizers.Tokenizer
        Vocab relabelled by frequency.
    unigram_probs : np.ndarray (float64, 1D)
        Probs in new ID order.
    old_to_new : np.ndarray (int64, 1D)
        ``old_to_new[old_id] = new_id``.
    sorted_old_ids : np.ndarray (int64, 1D)
        ``sorted_old_ids[new_id] = old_id``.
    """
    actual_vocab_size = tokenizer.get_vocab_size()

    # 1. Count occurrences of each old token ID
    counts = np.bincount(data_arr, minlength=actual_vocab_size).astype(np.float64)

    # 2. Rank by frequency (descending).
    #    sorted_old_ids[new_id] = old_id
    sorted_old_ids = np.argsort(-counts).astype(np.int64)

    # 3. Build the inverse map: old_to_new[old_id] = new_id
    old_to_new = np.empty(actual_vocab_size, dtype=np.int64)
    old_to_new[sorted_old_ids] = np.arange(actual_vocab_size, dtype=np.int64)

    # 4. Remap data_arr
    data_arr = old_to_new[data_arr]
    logging.info("Remapped token IDs to frequency order.")

    # 5. Rebuild the tokenizer JSON with the new vocab IDs.
    #    Merges are stored as string pairs, so only the vocab dict and
    #    the added_tokens list need relabelling.
    tok_json = json.loads(tokenizer.to_str())

    # Remap model vocab
    old_vocab: dict[str, int] = tok_json["model"]["vocab"]
    tok_json["model"]["vocab"] = {
        token: int(old_to_new[old_id]) for token, old_id in old_vocab.items()
    }

    # Remap added_tokens (special tokens, etc.)
    if "added_tokens" in tok_json:
        for entry in tok_json["added_tokens"]:
            entry["id"] = int(old_to_new[entry["id"]])

    new_tokenizer = Tokenizer.from_str(json.dumps(tok_json))

    # 6. Unigram probabilities in new ID order
    new_counts = counts[sorted_old_ids]
    total = new_counts.sum()
    unigram_probs = new_counts / total if total > 0 else new_counts

    logging.info(
        f"Frequency-ordered vocab: top-5 probs = "
        f"{unigram_probs[:5]}"
    )

    return data_arr, new_tokenizer, unigram_probs, old_to_new, sorted_old_ids
