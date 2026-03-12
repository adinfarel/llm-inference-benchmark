# Benchmark Core — Intuition Behind Every Measurement

This document explains the intuition behind every function
in `src/benchmark_core.py`. Every design decision has a
hardware reason behind it.

---

## 1. Why We Measure TTFT and TPOT Separately

TTFT and TPOT are not just "first token" and "other tokens."
They represent two fundamentally different hardware phases
with different bottlenecks.
```
TTFT (Prefill phase):
    All input tokens are processed at once in 1 forward pass.
    The GPU reads the entire prompt from HBM simultaneously.
    Bottleneck = memory bandwidth — how fast can HBM
    deliver all those tokens to the compute units.

    Longer prompt = more data from HBM = higher TTFT.
    This is memory bandwidth bound.

TPOT / ITL (Decode phase):
    One token generated per forward pass.
    KV cache holds previous token representations.
    GPU only needs to compute for the new token.
    Bottleneck = compute speed — how fast can the GPU
    run the matrix multiplications for 1 token.

    This is compute bound — or at least less IO-bound
    than prefill, because KV cache reduces redundant reads.
```

They are measured separately because optimizing one does not
necessarily optimize the other. A technique that speeds up
prefill might not affect decode at all, and vice versa.

## 2. Why torch.cuda.synchronize() Is Non-Negotiable

GPU is asynchronous. When Python calls `model.generate()`,
Python does not wait — it dispatches the command and
immediately moves to the next line.
```
Without synchronize():

    t_start = perf_counter()
    model.generate(...)       ← dispatched, Python moves on
    t_end = perf_counter()    ← GPU still running in background

    measured time = Python dispatch time = ~0.1ms
    actual GPU time = ~420ms
    Result: completely wrong

With synchronize():

    t_start = perf_counter()
    model.generate(...)       ← dispatched
    torch.cuda.synchronize()  ← Python waits here until GPU done
    t_end = perf_counter()    ← now GPU is actually done

    measured time = actual GPU execution time = ~420ms
    Result: valid
```

Every timing measurement in this project calls
`synchronize()` before recording `t_end`. Without this,
every number in the results would be meaningless.

## 3. Why Warmup Run Exists

The first time a CUDA kernel runs, the GPU compiles and
caches it. This one-time compilation overhead does not
represent steady-state inference performance.
```
Run 1 (cold):   compile + execute = 800ms  ← biased, too high
Run 2 (warm):   execute only      = 420ms  ← real performance
Run 3 (warm):   execute only      = 421ms  ← real performance
```

The warmup run is never measured. It exists only to bring
the GPU into steady state before measurement begins.

In `generate_and_measure()`, warmup happens once outside
the measurement loop — not inside. If warmup were inside
the loop, every run would re-warm unnecessarily, adding
overhead that does not reflect real inference behavior.

## 4. Why force_max_tokens Exists

Without forcing all models to generate the same number of
tokens, latency comparison becomes unfair.
```
Scenario without force_max_tokens:

    float32: generates "Neural networks are systems."
             EOS at token 35
             ITL list = 35 values, no KV pressure visible

    int4:    generates longer response
             EOS at token 67
             ITL list = 67 values, KV pressure starting to show

    Conclusion: "int4 is slower"
    Reality: int4 generated more tokens — not a fair comparison
```

With `force_max_tokens=True`, every model generates exactly
`max_new_tokens` regardless of EOS. All ITL lists have the
same length. The comparison is controlled.

`force_max_tokens=False` exists for quality evaluation —
when we want the model to finish naturally for perplexity
or output inspection.

## 5. Why We Measure ITL Per Position, Not Just Mean

Mean ITL (TPOT) hides what is actually happening across
token positions. Per-position ITL reveals KV cache pressure.
```
Token position 1:   ITL = 8ms   (KV cache small)
Token position 50:  ITL = 10ms  (KV cache growing)
Token position 100: ITL = 13ms  (KV cache pressure)

Mean ITL = 9.1ms  ← hides the slope entirely

If we only reported mean:
    "Model generates at 9.1ms per token"

With per-position:
    "Model starts at 8ms but degrades to 13ms
     as KV cache grows — 62% slowdown at position 100"
```

This per-position data is what proves or disproves the
Flash Attention hypothesis — FA should flatten this slope
because it accesses KV cache more efficiently from HBM.

## 6. Why Perplexity Uses Sliding Window, Not Naive Chunking

WikiText-103 is too long for one forward pass. Naive chunking
cuts it into independent chunks — but the first token of each
chunk has no context from the previous chunk, producing
artificially high perplexity.
```
Naive chunking:
    Chunk 1: token 0    → 2048  (all tokens have context)
    Chunk 2: token 2049 → 4096  (token 2049 has NO context)
    Chunk 3: token 4097 → 6144  (token 4097 has NO context)

    Token at chunk boundary = model is blind
    Perplexity at boundaries = inflated, biased high

Sliding window (stride=512):
    Window 1: token 0    → 2048
    Window 2: token 512  → 2560  (overlap 1536 tokens)
    Window 3: token 1024 → 3072  (overlap 1536 tokens)

    Every token gets context from previous tokens
    Only the new 512 tokens per window are evaluated
    Overlap tokens are masked with -100 (ignored in loss)
```

The `-100` mask is critical — it tells PyTorch's
CrossEntropyLoss to ignore those positions completely.
This prevents double-counting tokens that appear in
multiple windows.

## 7. Why ONNX Path Has No KV Cache in ITL Measurement

In the PyTorch path, `past_key_values` is passed between
decode steps — the KV cache grows step by step and stays
in HBM.

In the ONNX path, `past_key_values` is reset to `None`
every step. ORT standard models do not support KV cache
pass-through — every step is a full recomputation from scratch.
```
PyTorch ITL curve:    ────────/
                              slope rises = KV pressure visible

ONNX ITL curve:       ────────
                              flat = no KV cache growth
                              but baseline lower = no dispatch overhead
```

This means ONNX and PyTorch ITL curves tell different stories:
- PyTorch curve shows KV cache behavior
- ONNX curve shows dispatch overhead elimination

Both are valid measurements — they just measure different
architectural properties. This limitation is documented
as a known constraint of the runtime experiment.

## 8. Why p50 / p90 / p99, Not Just Mean

Mean hides outliers. In production systems, SLAs are defined
at percentiles — not averages.
```
Example ITL values across 3 runs (300 values):
    Most values: 8-10ms
    A few spikes: 25ms, 30ms (OS interrupts, cache misses)

Mean = 9.1ms   ← looks fine
p99  = 28ms    ← 1% of tokens take 28ms
               ← user experiences this as a stutter

If you only report mean:
    "Model is fast at 9.1ms per token"
    
If you report p99:
    "Model stutters at 28ms for 1% of tokens
     — unacceptable for real-time chat applications"
```

p50 = median, typical experience
p90 = 90% of tokens finish within this time
p99 = worst-case experience, what SLAs are built around