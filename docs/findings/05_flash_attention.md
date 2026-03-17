# Flash Attention — Tiling SRAM to Escape the O(N²) Wall

This document explains why standard attention breaks down at long sequences,
how Flash Attention restructures the same computation to avoid materializing
the full attention matrix in HBM, and what the T4 benchmark results show
about where the benefit becomes visible in practice.

---

## 1. Why Standard Attention Has a Memory Problem

Every transformer layer computes attention the same way. Each token produces three vectors — Query, Key, and Value — and attention measures how much each token should attend to every other token.
```
Q = input × W_q     ← what this token is looking for
K = input × W_k     ← what this token is offering
V = input × W_v     ← the actual content to pass forward

scores = softmax(Q × K^T / √d_k)
output = scores × V
```

The problem lives in Q × K^T. If sequence length is N, this produces an N×N matrix. Every token attends to every other token — and the matrix has to exist somewhere in memory before softmax can run.
```
Standard attention memory requirement:

    Sequence 128 tokens:
        scores matrix = 128 × 128 = 16,384 values
        At float16: 16,384 × 2 bytes = 32KB

    Sequence 512 tokens:
        scores matrix = 512 × 512 = 262,144 values
        At float16: 262,144 × 2 bytes = 512KB

    Sequence 1024 tokens:
        scores matrix = 1024 × 1024 = 1,048,576 values
        At float16: 1,048,576 × 2 bytes = 2MB

    This is per layer, per attention head.
    TinyLlama has 22 layers and 32 heads.

    At sequence 1024:
        2MB × 22 layers × 32 heads = 1.4GB
        just for attention scores — before weights, KV cache, activations
```

And since the SRAM per SM on the T4 is only 48KB, there's no way to store this score matrix in SRAM. It has to go to HBM.
```
    Standard attention HBM round trips per layer:

    Step 1: Compute Q × K^T        → write scores to HBM
    Step 2: Read scores from HBM   → run softmax
    Step 3: Write softmax result   → to HBM
    Step 4: Read from HBM          → multiply with V
    Step 5: Write output           → to HBM

    5 HBM operations.
    Scores matrix size = O(N²).
    Sequence doubles → matrix quadruples → HBM traffic quadruples.
```

This is what causes ITL to increase as the sequence grows. It's not because there's more computation involved — but because the data that has to go back and forth to HBM grows quadratically.

## 2. Flash Attention — The Tiling Insight

Flash Attention doesn't change the computational results. The output is identical to standard attention. What changes is how the computation is executed—specifically, which level of memory the intermediate results are stored in.
**The core insight is one sentence:** we don't need to materialize the full N×N matrix if we can compute the softmax incrementally.
```
The obstacle to tiling:

    Matrix multiply can be tiled trivially.
    Calculate partial, store, continue, merge.

    Softmax can't — it needs all values ​​at once:
        softmax(x_i) = exp(x_i) / Σ exp(x_j)
                                   ↑
                                sum over ALL j
                                cannot be partial   

    If you only look at 256 of the 1024 scores,
    you can't calculate the correct softmax
    because the denominator needs all 1024 values.
```

Flash Attention solves this with online softmax — a mathematically equivalent softmax calculation method that can be updated incrementally without needing all the values ​​at once.
```
Online softmax key idea:

    Store two running statistics:
        m = running maximum scores seen
        l = running sum of exp(score - m)

    For each new block:
        Update m with the new maximum
        Rescale the old l because m has changed
        Add the new block's contribution to l
        Update the output accumulator with the same rescaling

    After all blocks have been processed:
        Output = output_accumulator / l_final
        This is identical to a full softmax over all scores.
        No approximation. No information loss.
```

With online softmax, we can process K and V in small blocks that fit in SRAM. The full N×N matrix never needs to exist in HBM.
```
Flash Attention HBM traffic:

    Read Q, K, V from HBM       ← O(N), only once
    All intermediate results    → stay in SRAM
    Write output to HBM         ← O(N), just once

    HBM traffic: O(N)
    Standard attention: O(N²)

    At sequence 1024, 32 heads, 22 layers:
        Standard: ~1.4GB roundtrip HBM
        Flash: ~0.05GB roundtrip HBM
        Reduction: ~28x less HBM traffic
```

## 3. Block Size — Derived from Hardware, Not Chosen Arbitrarily

Block size in Flash Attention isn't a hyperparameter that needs to be tuned. It's derived from the SRAM per SM constraint.
```
One Flash Attention block needs to be in SRAM at once:
    Q block: block_size × d_head × 2 bytes
    K block: block_size × d_head × 2 bytes
    V block: block_size × d_head × 2 bytes
    scores block: block_size × block_size × 2 bytes
    output: block_size × d_head × 2 bytes

TinyLlama: d_head = 64, float16

    With block_size = 64:
        Q: 64 × 64 × 2 = 8KB
        K: 64 × 64 × 2 = 8KB
        V: 64 × 64 × 2 = 8KB
        scores: 64 × 64 × 2 = 8KB
        output: 64 × 64 × 2 = 8KB
        Total: 40KB ← fits in 48KB of T4 SRAM

    Full N×N for N=1024:
        1024 × 1024 × 2 = 2MB ← doesn't fit, must switch to HBM
```

The Flash Attention implementation in PyTorch and HuggingFace auto-detects the block size based on the GPU detected at runtime. On the T4 with 48KB of SRAM, the block size is automatically set to 64.

## 4. Why Float16 is Required

Flash Attention only supports float16 and bfloat16. It doesn't work on float32.
```
There are two reasons:

    Memory constraints:
        SRAM per SM = 48KB on T4
        With float32 (4 bytes per value):
            block_size = floor(48,000 / (5 × 64 × 4)) = 37
            Too small — tiling overhead > benefit

        With float16 (2 bytes per value):
            block_size = floor(48,000 / (5 × 64 × 2)) = 75 → rounded to 64
            Large enough for efficient tiling

    Tensor Core support:
        T4 Tensor Cores have a native float16 matrix multiply
        Flash Attention kernel designed to exploit Tensor Cores
        Float32 doesn't enter the Tensor Core fast path on T4
```

This is also why in run_flash_attention.py we always load the model with torch_dtype=torch.float16 — not because float16 is the baseline we're comparing against, but because Flash Attention physically can't run without it.

## 5. The Crossover Point — When Flash Attention Actually Helps

Flash Attention isn't always faster. In very short sequences, the overhead of tiling logic can outweigh the benefits.
```
    Sequence 32 tokens:
        N×N matrix = 32 × 32 = 1,024 values ​​= 2KB
        2KB fits in SRAM — standard attention can stay in SRAM too
        Flash Attention tiling overhead > HBM savings
        Standard attention can be faster or the same

    Sequence 128-256 tokens:
        N×N matrix = ~64KB-128KB
        Starting to not fit in SRAM
        Flash Attention benefits start to be visible
        The crossover point is in this range

    Sequence 512+ tokens:
        N×N matrix = 512KB+
        Far beyond SRAM
        Standard attention must go full roundtrip to HBM
        Flash Attention benefit significant and growing

    Sequence 1024 tokens:
        Flash Attention dominant
        The ITL gap between standard and FA is largest here
        This is what Q3 measures
```

The crossover point isn't fixed — it depends on the GPU's SRAM size. On the A100 with 192KB of SRAM, crossover occurs earlier because standard attention can fit a larger matrix before resorting to HBM.

## 6. Connection to KV Cache Experiment

The Flash Attention and KV cache experiment (Q4) measures two sides of the same problem.
```
Flash Attention (Q3) asks:
    "If we reorganize attention computation,
    will the ITL remain flat even as the sequence grows?"
    → The solution: tiling to avoid the need for a full N×N in HBM.

KV Cache (Q4) asks:
    "At what token position does the KV cache pressure
    start to become visible in the ITL?"
    → The problem: with each token generated, the KV cache grows,
    more data in the HBM needs to be attended.

The connection:
    The KV cache is the source of growing HBM pressure during generation.
    Flash Attention is a technique that reduces HBM pressure
    from the attention computation itself.

    Without Flash Attention:
        KV cache grows + attention matrix grows → ITL increases rapidly

    With Flash Attention:
        KV cache grows → ITL increases slowly
        Attention matrix does not go to HBM → component is flat

    These two experiments together answer:
        Where is the bottleneck, and is Flash Attention enough
        to keep ITL flat until the 1000th token?
```

## 7. Production Benchmark Results
