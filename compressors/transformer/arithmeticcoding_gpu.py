import math

import torch
from numba import cuda, int64, float32
import numpy as np

# -- Packed-bit helpers (inline in kernels) --
# bits buffer is uint8, shape (num_streams, max_bytes).
# Bit k lives in byte k>>3, at position 7-(k&7)  (MSB-first).
# Write: bits[s, k>>3] |= (val << (7-(k&7)))   (buffer zeroed → only set 1s)
# Read:  (bits[s, k>>3] >> (7-(k&7))) & 1

@cuda.jit
def parallel_encode_full_sequence_kernel(
    cumul_full,      # [num_streams, seq_len, vocab_size+1]
    symbols_full,    # [num_streams, seq_len]
    low,             # [num_streams] - current low bound (int)
    high,            # [num_streams] - current high bound (int)
    num_underflow,   # [num_streams] - underflow counter
    bits,            # [num_streams, max_bytes] - packed uint8
    bit_counts,      # [num_streams]
    num_streams,
    seq_len,
    vocab_size,
    STATE_SIZE,
    MASK,
    TOP_MASK,
    SECOND_MASK,
    MAX_BITS
):
    """
    Process entire sequence of symbols per stream in a single kernel launch.
    Each thread handles one complete stream from start to finish.
    """
    stream_idx = cuda.grid(1)
    if stream_idx >= num_streams:
        return
    
    low_val = low[stream_idx]
    high_val = high[stream_idx]
    underflow = num_underflow[stream_idx]
    bc = bit_counts[stream_idx]
    
    # Process entire sequence for this stream
    for pos in range(seq_len):
        symbol = symbols_full[stream_idx, pos]
        
        # Get cumulative frequencies
        total = cumul_full[stream_idx, pos, vocab_size]
        symlow = cumul_full[stream_idx, pos, symbol]
        symhigh = cumul_full[stream_idx, pos, symbol + 1]
        
        # Update range (standard arithmetic coding)
        range_val = high_val - low_val + 1
        newlow = low_val + (symlow * range_val) // total
        newhigh = low_val + (symhigh * range_val) // total - 1
        
        low_val = newlow
        high_val = newhigh
        
        # Renormalization loop - continue until fully normalized
        while True:
            xor_val = low_val ^ high_val
            top_bits_same = (xor_val & TOP_MASK) == 0
            
            if top_bits_same:
                # Top bits match - emit bit
                bit = (low_val >> (STATE_SIZE - 1)) & 1
                if bc < MAX_BITS:
                    if bit == 1:
                        bits[stream_idx, bc >> 3] = bits[stream_idx, bc >> 3] | (1 << (7 - (bc & 7)))
                    bc += 1
                
                # Emit pending underflow bits (opposite of bit)
                obit = bit ^ 1
                for i in range(underflow):
                    if bc >= MAX_BITS:
                        break
                    if obit == 1:
                        bits[stream_idx, bc >> 3] = bits[stream_idx, bc >> 3] | (1 << (7 - (bc & 7)))
                    bc += 1
                underflow = 0
                
                # Shift
                low_val = (low_val << 1) & MASK
                high_val = ((high_val << 1) & MASK) | 1
                continue
            
            # Check for underflow
            underflow_cond = (low_val & ~high_val & SECOND_MASK) != 0
            if underflow_cond:
                underflow += 1
                low_val = (low_val << 1) & (MASK >> 1)
                high_val = ((high_val << 1) & (MASK >> 1)) | TOP_MASK | 1
                continue
            
            # Both conditions false - exit renormalization
            break
    
    # Save state back
    low[stream_idx] = low_val
    high[stream_idx] = high_val
    num_underflow[stream_idx] = underflow
    bit_counts[stream_idx] = bc

@cuda.jit
def finalize_encoding_kernel(
    low, high, num_underflow, bits, bit_counts,
    num_streams, STATE_SIZE, MAX_BITS
):
    """Emit final bit AND pad to byte boundary"""
    stream_idx = cuda.grid(1)
    if stream_idx >= num_streams:
        return
    
    bc = bit_counts[stream_idx]
    
    # Emit bit 1 to finish encoding (packed)
    if bc < MAX_BITS:
        bits[stream_idx, bc >> 3] = bits[stream_idx, bc >> 3] | (1 << (7 - (bc & 7)))
        bc += 1
    
    # PAD TO BYTE BOUNDARY (zeros — already zeroed, just advance counter)
    while (bc % 8) != 0 and bc < MAX_BITS:
        bc += 1
    
    bit_counts[stream_idx] = bc


# -- Fused log-prob → cumulative frequency kernel -----------------

@cuda.jit
def _logprob_to_freq_kernel(log_probs_flat, freq_out_flat, n):
    """Per-element: log_prob -> exp -> x1e7+1 -> int64.

    One thread per element = full GPU saturation + perfect coalescing.
    The sequential cumsum is left to PyTorch's optimised parallel scan.

    Args:
        log_probs_flat: (n,) float32
        freq_out_flat:  (n,) int64
    """
    idx = cuda.grid(1)
    if idx >= n:
        return
    prob = math.exp(log_probs_flat[idx])
    freq_out_flat[idx] = int64(prob * float32(1e7)) + int64(1)


def fused_logprob_to_cumul(log_probs_3d, cumul_out_3d):
    """Fused conversion from log probabilities to cumulative frequency table.

    Replaces the multi-op chain:
        torch.cumsum(torch.floor(lp.float().exp() * 1e7 + 1).to(int64), dim=2, out=...)

    Uses a per-element Numba kernel (32M threads, full GPU saturation) for
    the fused exp+scale+int64 step, then PyTorch's optimised cumsum.

    Args:
        log_probs_3d: (B, T, V) log-probabilities on GPU  (float16/float32)
        cumul_out_3d: (B, T, V+1) output cumulative freq  (int64, GPU)
                      cumul_out[:, :, 0] will be set to 0.
    """
    B, T, V = log_probs_3d.shape
    n = B * T * V

    # Flatten to 1-D for the per-element kernel
    lp_flat = log_probs_3d.detach().float().reshape(-1).contiguous()
    freq_flat = torch.empty(n, dtype=torch.int64, device=log_probs_3d.device)

    d_lp = cuda.as_cuda_array(lp_flat)
    d_fr = cuda.as_cuda_array(freq_flat)

    threads = 256
    blocks = (n + threads - 1) // threads
    _logprob_to_freq_kernel[blocks, threads](d_lp, d_fr, n)

    # PyTorch parallel-scan cumsum (well-optimised, coalesced)
    cumul_out_3d[:, :, 0] = 0
    torch.cumsum(freq_flat.reshape(B, T, V), dim=2, out=cumul_out_3d[:, :, 1:])


# -- Sparse log-prob → cumulative frequency (3 values only) --------
# Computes only cumul[symbol], cumul[symbol+1], and cumul[V] (= total)
# per (batch, time) position.  Avoids the full freq tensor and cumsum.

_SPARSE_TPB = 256  # threads per block — must match shared array sizes


@cuda.jit
def _sparse_cumul_kernel(log_probs, symbols, cumul_out, B_T, V):
    """Block-parallel sparse cumul.

    Grid:  one block per (b,t) position  (= B*T blocks)
    Block: _SPARSE_TPB threads

    Each thread handles a strided chunk of V entries, computing 3 partial
    sums (low, sym_freq, total).  Then a shared-memory tree reduction
    combines them, and thread-0 writes the 3 values the encoder reads.
    """
    pos = cuda.blockIdx.x
    if pos >= B_T:
        return
    tid = cuda.threadIdx.x
    bsz = cuda.blockDim.x

    sym = symbols[pos]

    # -- Phase 1: each thread accumulates its chunk ----------------------
    partial_low   = int64(0)
    partial_sfr   = int64(0)
    partial_total = int64(0)

    v = tid
    while v < V:
        prob = math.exp(log_probs[pos, v])
        freq = int64(prob * float32(1e7)) + int64(1)
        partial_total += freq
        if v < sym:
            partial_low += freq
        elif v == sym:
            partial_sfr = freq
        v += bsz

    # -- Phase 2: shared-memory tree reduction ---------------------------
    s_low   = cuda.shared.array(_SPARSE_TPB, dtype=int64)
    s_sfr   = cuda.shared.array(_SPARSE_TPB, dtype=int64)
    s_total = cuda.shared.array(_SPARSE_TPB, dtype=int64)

    s_low[tid]   = partial_low
    s_sfr[tid]   = partial_sfr
    s_total[tid] = partial_total
    cuda.syncthreads()

    stride = bsz >> 1
    while stride > 0:
        if tid < stride:
            s_low[tid]   += s_low[tid + stride]
            s_sfr[tid]   += s_sfr[tid + stride]
            s_total[tid] += s_total[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    # -- Phase 3: thread 0 writes the 3 entries the encoder reads --------
    if tid == 0:
        cumul_out[pos, sym]     = s_low[0]
        cumul_out[pos, sym + 1] = s_low[0] + s_sfr[0]
        cumul_out[pos, V]       = s_total[0]


def sparse_logprob_to_cumul(log_probs_3d, symbols, cumul_out_3d):
    """Populate only the 3 cumul entries the encoder needs per position.

    Drop-in alternative to ``fused_logprob_to_cumul`` when the target
    symbols are known ahead of time (encoding path).  Skips the full
    cumsum and intermediate freq tensor entirely.

    Args:
        log_probs_3d: (B, T, V)   log-probabilities on GPU  (float16/float32)
        symbols:      (B, T)      target symbols             (int64, GPU)
        cumul_out_3d: (B, T, V+1) output cumulative freq     (int64, GPU)
                      Only entries [sym], [sym+1], and [V] are written per row.
    """
    B, T, V = log_probs_3d.shape
    n = B * T

    lp_flat = log_probs_3d.detach().float().reshape(n, V).contiguous()
    sym_flat = symbols.detach().reshape(n).contiguous().to(torch.int64)
    cumul_flat = cumul_out_3d.view(n, V + 1)

    d_lp = cuda.as_cuda_array(lp_flat)
    d_sym = cuda.as_cuda_array(sym_flat)
    d_cumul = cuda.as_cuda_array(cumul_flat)

    _sparse_cumul_kernel[n, _SPARSE_TPB](d_lp, d_sym, d_cumul, n, V)


class ParallelArithmeticEncoderGPU:
    """
    Optimized parallel arithmetic encoder on GPU
    """
    
    def __init__(self, statesize, num_streams, device='cuda', max_bits_per_stream=65536):
        self.device = device
        self.num_streams = num_streams
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << self.STATE_SIZE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1
        self.max_bits_per_stream = max_bits_per_stream
        self.max_bytes_per_stream = (max_bits_per_stream + 7) // 8

        # State arrays on GPU
        self.low = torch.zeros(num_streams, dtype=torch.int64, device=device)
        self.high = torch.full((num_streams,), self.MASK, dtype=torch.int64, device=device)
        self.num_underflow = torch.zeros(num_streams, dtype=torch.int64, device=device)
        
        # Packed bit output buffer (8× smaller than int8-per-bit)
        self.bits = torch.zeros((num_streams, self.max_bytes_per_stream), 
                                dtype=torch.uint8, device=device)
        self.bit_counts = torch.zeros(num_streams, dtype=torch.int64, device=device)
        
        # Cache CUDA array wrappers to avoid repeated conversions
        self._cuda_arrays_cached = False
        self.d_low = None
        self.d_high = None
        self.d_underflow = None
        self.d_bits = None
        self.d_bit_counts = None
        
        # Auto-tune threads per block
        self.threads_per_block = self._auto_tune_threads(num_streams)
        self.blocks = (num_streams + self.threads_per_block - 1) // self.threads_per_block

    def _auto_tune_threads(self, num_streams):
        """
        Automatically determine optimal threads_per_block based on GPU properties.
        """
        try:
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            max_threads_per_block = props.max_threads_per_block
            multiprocessor_count = props.multi_processor_count
        except:
            max_threads_per_block = 1024
            multiprocessor_count = 80
        
        # Target: 2-4 blocks per SM for good occupancy
        target_blocks = multiprocessor_count * 3
        ideal_threads = max(32, num_streams // target_blocks)
        
        # Round to nearest multiple of 32 (warp size)
        threads = min(max_threads_per_block, ((ideal_threads + 31) // 32) * 32)
        threads = max(64, threads)
        
        print(f"[GPU Tuning] num_streams={num_streams}, SM count={multiprocessor_count}")
        print(f"[GPU Tuning] Using {threads} threads/block → {(num_streams + threads - 1) // threads} blocks")
        
        return threads

    def _cache_cuda_arrays(self):
        """Cache CUDA array wrappers to avoid repeated conversions."""
        if not self._cuda_arrays_cached:
            self.d_low = cuda.as_cuda_array(self.low.detach())
            self.d_high = cuda.as_cuda_array(self.high.detach())
            self.d_underflow = cuda.as_cuda_array(self.num_underflow.detach())
            self.d_bits = cuda.as_cuda_array(self.bits.detach())
            self.d_bit_counts = cuda.as_cuda_array(self.bit_counts.detach())
            self._cuda_arrays_cached = True

    def write(self, cumul_batch, symbols_batch):
        """
        Incrementally encode a batch and append to existing bit stream.
        maintains state across calls.
        
        Args:
            cumul_batch: [num_streams, seq_len, vocab_size+1] for this iteration
            symbols_batch: [num_streams, seq_len] for this iteration
        """
        # Cache CUDA arrays if first call
        self._cache_cuda_arrays()
        
        # Convert inputs
        d_cumul = cuda.as_cuda_array(cumul_batch)
        d_symbols = cuda.as_cuda_array(symbols_batch)
        
        num_streams, seq_len, vocab_p1 = cumul_batch.shape
        vocab_size = vocab_p1 - 1
        
        # Encode and APPEND to existing bits
        parallel_encode_full_sequence_kernel[self.blocks, self.threads_per_block](
            d_cumul,
            d_symbols,
            self.d_low,
            self.d_high,
            self.d_underflow,
            self.d_bits,
            self.d_bit_counts,
            num_streams,
            seq_len,
            vocab_size,
            self.STATE_SIZE,
            self.MASK,
            self.TOP_MASK,
            self.SECOND_MASK,
            self.max_bits_per_stream
        )

    def get_encoded_bytes(self, stream_idx):
        """Return compressed bytes for a stream (bits already packed in uint8)."""
        bit_count = int(self.bit_counts[stream_idx].cpu().item())
        byte_count = (bit_count + 7) // 8
        return bytes(self.bits[stream_idx, :byte_count].cpu().numpy())
    
    def reset_state(self):
        """Reset encoder state for reuse."""
        self.low.zero_()
        self.high.fill_(self.MASK)
        self.num_underflow.zero_()
        self.bits.zero_()
        self.bit_counts.zero_()

    def finish(self):
        """Finalize encoding for all streams."""
        finalize_encoding_kernel[self.blocks, self.threads_per_block](
            self.d_low,
            self.d_high,
            self.d_underflow,
            self.d_bits,
            self.d_bit_counts,
            self.num_streams,
            self.STATE_SIZE,
            self.max_bits_per_stream
        )

################################################################
## Parallel Arithmetic Decoder 
################################################################


@cuda.jit
def parallel_decode_full_sequence_kernel(
    cumul_full,      # [num_streams, seq_len, vocab_size+1]
    symbols_out,     # [num_streams, seq_len] - output decoded symbols
    code,            # [num_streams] - current code value
    low,             # [num_streams] - current low bound
    high,            # [num_streams] - current high bound
    bits,            # [num_streams, max_bytes] - packed uint8
    bit_positions,   # [num_streams] - current read position in bit stream
    num_streams,
    seq_len,
    vocab_size,
    STATE_SIZE,
    MASK,
    TOP_MASK,
    SECOND_MASK,
    MAX_BITS
):
    """
    Decode entire sequence of symbols per stream in a single kernel launch.
    Each thread handles one complete stream from start to finish.
    """
    stream_idx = cuda.grid(1)
    if stream_idx >= num_streams:
        return
    
    code_val = code[stream_idx]
    low_val = low[stream_idx]
    high_val = high[stream_idx]
    bit_pos = bit_positions[stream_idx]
    
    # Process entire sequence for this stream
    for pos in range(seq_len):
        # Get cumulative frequencies for this position
        total = cumul_full[stream_idx, pos, vocab_size]
        
        # Translate from coding range to frequency table scale
        range_val = high_val - low_val + 1
        offset = code_val - low_val
        value = ((offset + 1) * total - 1) // range_val
        
        # Binary search to find symbol
        start = 0
        end = vocab_size
        while end - start > 1:
            middle = (start + end) >> 1
            if cumul_full[stream_idx, pos, middle] > value:
                end = middle
            else:
                start = middle
        
        symbol = start
        symbols_out[stream_idx, pos] = symbol
        
        # Update range using the decoded symbol
        symlow = cumul_full[stream_idx, pos, symbol]
        symhigh = cumul_full[stream_idx, pos, symbol + 1]
        
        newlow = low_val + (symlow * range_val) // total
        newhigh = low_val + (symhigh * range_val) // total - 1
        
        low_val = newlow
        high_val = newhigh
        
        # Renormalization loop (read packed bits to maintain precision)
        for _ in range(STATE_SIZE * 2):
            xor_val = low_val ^ high_val
            top_bits_same = (xor_val & TOP_MASK) == 0
            
            if top_bits_same:
                # Shift and read a packed bit
                low_val = (low_val << 1) & MASK
                high_val = ((high_val << 1) & MASK) | 1
                
                if bit_pos < MAX_BITS:
                    bit = (bits[stream_idx, bit_pos >> 3] >> (7 - (bit_pos & 7))) & 1
                else:
                    bit = 0
                bit_pos += 1
                code_val = ((code_val << 1) & MASK) | bit
                continue
            
            # Check for underflow
            underflow_cond = (low_val & ~high_val & SECOND_MASK) != 0
            if underflow_cond:
                low_val = (low_val << 1) & (MASK >> 1)
                high_val = ((high_val << 1) & (MASK >> 1)) | TOP_MASK | 1
                
                if bit_pos < MAX_BITS:
                    bit = (bits[stream_idx, bit_pos >> 3] >> (7 - (bit_pos & 7))) & 1
                else:
                    bit = 0
                bit_pos += 1
                code_val = (code_val & TOP_MASK) | ((code_val << 1) & (MASK >> 1)) | bit
                continue
            
            break
    
    # Save state back
    code[stream_idx] = code_val
    low[stream_idx] = low_val
    high[stream_idx] = high_val
    bit_positions[stream_idx] = bit_pos


class ParallelArithmeticDecoderGPU:
    """
    Optimized parallel arithmetic decoder - processes entire sequences in single kernel launches.
    Mirrors the encoder architecture for consistency.
    """
    
    def __init__(self, statesize, num_streams, device='cuda', max_bits_per_stream=65536):
        self.device = device
        self.num_streams = num_streams
        self.STATE_SIZE = statesize
        self.MAX_RANGE = 1 << self.STATE_SIZE
        self.MASK = self.MAX_RANGE - 1
        self.TOP_MASK = self.MAX_RANGE >> 1
        self.SECOND_MASK = self.TOP_MASK >> 1
        self.max_bits_per_stream = max_bits_per_stream
        self.max_bytes_per_stream = (max_bits_per_stream + 7) // 8

        # State arrays on GPU
        self.code = torch.zeros(num_streams, dtype=torch.int64, device=device)
        self.low = torch.zeros(num_streams, dtype=torch.int64, device=device)
        self.high = torch.full((num_streams,), self.MASK, dtype=torch.int64, device=device)
        
        # Packed bit input buffer (8× smaller than int8-per-bit)
        self.bits = torch.zeros((num_streams, self.max_bytes_per_stream), 
                                dtype=torch.uint8, device=device)
        self.bit_positions = torch.zeros(num_streams, dtype=torch.int64, device=device)
        
        # Cache CUDA array wrappers
        self._cuda_arrays_cached = False
        self.d_code = None
        self.d_low = None
        self.d_high = None
        self.d_bits = None
        self.d_bit_positions = None
        
        # Auto-tune threads per block
        self.threads_per_block = self._auto_tune_threads(num_streams)
        self.blocks = (num_streams + self.threads_per_block - 1) // self.threads_per_block

    def _auto_tune_threads(self, num_streams):
        """Automatically determine optimal threads_per_block."""
        try:
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            max_threads_per_block = props.max_threads_per_block
            multiprocessor_count = props.multi_processor_count
        except:
            max_threads_per_block = 1024
            multiprocessor_count = 80
        
        target_blocks = multiprocessor_count * 3
        ideal_threads = max(32, num_streams // target_blocks)
        threads = min(max_threads_per_block, ((ideal_threads + 31) // 32) * 32)
        threads = max(64, threads)
        
        print(f"[GPU Decoder Tuning] num_streams={num_streams}, SM count={multiprocessor_count}")
        print(f"[GPU Decoder Tuning] Using {threads} threads/block → {(num_streams + threads - 1) // threads} blocks")
        
        return threads

    def _cache_cuda_arrays(self):
        """Cache CUDA array wrappers to avoid repeated conversions."""
        if not self._cuda_arrays_cached:
            self.d_code = cuda.as_cuda_array(self.code.detach())
            self.d_low = cuda.as_cuda_array(self.low.detach())
            self.d_high = cuda.as_cuda_array(self.high.detach())
            self.d_bits = cuda.as_cuda_array(self.bits.detach())
            self.d_bit_positions = cuda.as_cuda_array(self.bit_positions.detach())
            self._cuda_arrays_cached = True

    def load_compressed_bytes(self, stream_idx, byte_data):
        """Load compressed bytes for a specific stream and initialize code value."""
        byte_arr = np.frombuffer(byte_data, dtype=np.uint8)
        load_len = min(len(byte_arr), self.max_bytes_per_stream)
        self.bits[stream_idx, :load_len] = torch.from_numpy(
            byte_arr[:load_len].copy()).to(self.device)
        if load_len < self.max_bytes_per_stream:
            self.bits[stream_idx, load_len:] = 0
        
        # Initialize code value by reading STATE_SIZE bits from packed bytes
        code_val = 0
        for i in range(self.STATE_SIZE):
            byte_idx = i >> 3
            shift = 7 - (i & 7)
            if byte_idx < len(byte_arr):
                bit = (int(byte_arr[byte_idx]) >> shift) & 1
            else:
                bit = 0
            code_val = (code_val << 1) | bit
        
        self.code[stream_idx] = code_val
        self.bit_positions[stream_idx] = self.STATE_SIZE

    def read_batch_parallel(self, cumul_gpu, seq_len, debug=False):
        """
        Decode entire batch in parallel with a single kernel launch.
        
        Args:
            cumul_gpu: [num_streams, seq_len, vocab_size+1] cumulative frequency table
            seq_len: number of symbols to decode
            debug: if True, print timing information
            
        Returns:
            symbols_batch: [num_streams, seq_len] decoded symbols
        """
        num_streams, _, vocab_p1 = cumul_gpu.shape
        vocab_size = vocab_p1 - 1
        
        if debug:
            import time
            start = time.perf_counter()
        
        # Output buffer for decoded symbols
        symbols_out = torch.zeros((num_streams, seq_len), dtype=torch.int64, device=self.device)
        
        # Cache state array wrappers (only done once)
        self._cache_cuda_arrays()
        
        # Convert input arrays
        d_cumul = cuda.as_cuda_array(cumul_gpu.detach())
        d_symbols = cuda.as_cuda_array(symbols_out.detach())
        
        if debug:
            torch.cuda.synchronize()
            convert_time = time.perf_counter() - start
            kernel_start = time.perf_counter()
        
        # Single kernel launch for entire sequence!
        parallel_decode_full_sequence_kernel[self.blocks, self.threads_per_block](
            d_cumul,
            d_symbols,
            self.d_code,
            self.d_low,
            self.d_high,
            self.d_bits,
            self.d_bit_positions,
            num_streams,
            seq_len,
            vocab_size,
            self.STATE_SIZE,
            self.MASK,
            self.TOP_MASK,
            self.SECOND_MASK,
            self.max_bits_per_stream
        )
        
        if debug:
            torch.cuda.synchronize()
            kernel_time = time.perf_counter() - kernel_start
            total_time = time.perf_counter() - start
            print(f"[GPU Decoding] Convert: {convert_time:.4f}s, "
                  f"Kernel: {kernel_time:.4f}s, "
                  f"Total: {total_time:.4f}s")
        
        return symbols_out
    
    def reset_state(self):
        """Reset decoder state for reuse."""
        self.code.zero_()
        self.low.zero_()
        self.high.fill_(self.MASK)
        self.bits.zero_()
        self.bit_positions.zero_()
        
        # Invalidate cached CUDA arrays
        self._cuda_arrays_cached = False
        self.d_code = None
        self.d_low = None
        self.d_high = None
        self.d_bits = None
        self.d_bit_positions = None