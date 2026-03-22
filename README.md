# LLM Inference Optimization Benchmark

> Benchmarking 10+ inference optimization techniques on TinyLlama 1.1B
> quantization, pruning, distillation, flash attention, onnx, and serving strategies
> on a GPU environment. Every result is backed by profiler traces and explained
> from hardware level, not just reported.

![Architecture Overview](docs/images/inference-flow-bg.png)

**Author:** Adin Ramdan Farelino  
**Hardware:** GCP T4 GPU (16GB VRAM)    
**Model:** TinyLLama/TinyLlama-1.1B-Chat-v1.0  
**Status:** In Progress - actively running experiments  

## Motivation

Most LLM inference optimization tutorials stop at showing numbers.
This project started differently - from a question that kept coming back:

> *"Why is quantized inference sometimes slower than the baseline"*

The answer turned out to be a hardware story. Running an initial exploration on CPU revealed that optimization
techniques behave very differently depending on the hardware they run on. int4 quantization was **7x slowe** than float32
on CPU - not because of a bug, but because CPU has no native integer arithmetic. The model had to dequantize weights back to float32 before every computation, making the overhead larger than the memory saving

That exploration raised 9 questions that could not be answered on CPU alone. This project exists to answer all of them - on a GPU where these techniques were actually designed to run.

## Research Questions

This project is built around 9 questions that emerged from the CPU exploration and could not be answered without a proper GPU environment.

**Q1 - Does int4 quantization actually speed up inference on GPU?**
On CPU, int4 was 7x slower than float32 due to dequantization overhead. On GPU with Tensor Core native integer arithmetic, the expectation flips. This project measures whether that theoretical advantage holds in practice

**Q2 - Between AWQ vs NF4 - which int4 technique preserves perplexity best?**
Both compress weights to 4-bit but with fundamentally different strategies.
NF4 optimizes quantization points for normal weight distributions.
AWQ protects salient weights based on activation magnitude before quantizing.
This project measures which approach preserves perplexity better on TinyLlama.

**Q3 - Does Flash Attention actually flatten ITL at long sequences on T4?** 
Standard attention has O(n*n) memory complexity - every token attends to every other token, and intermediate results must pass through HBM repeatedly. Flash Attention tiles the computation to stay in SRAM. This project measures whether ITL stays flat from token 1 to token 1000, or starts degrading.

**Q4 - At which token position does KV cache pressure become visible?** 
KV cache grows every generated token. At same point, the cache size starts pressuring GPU memory and ITL begins to rise. This project maps the inflection point by measuring ITL per token position across 1000 tokens.

**Q5 - What is the optimal batch size for throughput on a T4?**  
Batch size 1 underutilizes GPU parallelism. Too large a batch causes
memory pressure. There is a sweet spot where throughput peaks before
latency per request becomes unacceptable. This project finds that point
across batch sizes 1, 2, 4, 8, 16, and 32.

**Q6 - Does prompt length affect TTFT in a way consistent with O(n²) complexity?**  
Prefill — processing the input prompt before generating the first token —
is quadratic in sequence length for standard attention. This project measures
TTFT across prompt lengths 32 to 1024 tokens and tests whether Flash Attention
changes that scaling behavior.

**Q7 - What is the production cost of distillation — how much speed does a student gain, and what quality does it sacrifice?**    
from a larger teacher. The CPU exploration demonstrated the KD mechanics: hard loss, soft loss via KL divergence, and temperature scaling. This project measures the production trade-off on GPU — benchmarking OpenLLaMA 3B (teacher) against TinyLlama 1.1B (student) on TTFT, throughput, memory, and perplexity to answer whether the speed advantage justifies the quality gap in a real serving environment.

**Q8 - Does torch.compile reduce dispatch overhead and improve throughput?**
PyTorch dynamic graph dispatches ~220 kernel launches per token, each with
fixed overhead regardless of computation size. torch.compile fuses operations
and reduces kernel launches at compile time. This project measures whether
that translates to measurable throughput improvement over float16 baseline.

**Q9 - Does static graph execution (ONNX Runtime) reduce dispatch overhead compared to PyTorch dynamic graph?** 
PyTorch eager mode dispatches ~220 kernel launches per token, each with fixed
CPU overhead before GPU execution begins. ONNX Runtime resolves the full
execution plan at load time — inference runs with zero Python dispatch per token.
This project measures the empirical difference in ITL between PyTorch dynamic
graph and ONNX static graph execution, and whether dispatch overhead elimination
translates to measurable throughput improvement on T4.

## Environment

| Component | Details |
|-----------|---------|
| Cloud | Google Cloud Platform |
| VRAM | 16GB GDDR6 |
| CUDA | 12.8 |
| Python | 3.10 |
| PyTorch | 2.1.0+cu121 |
| Transformers | 4.38.0 |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Parameters | 1.1 Billion |
| Evaluation Dataset | WikiText-103 |

## Experiment Overview

Each experiment measures the same core metrics to ensure fair comparison.
Latency metrics are reported with p50, p90, and p99 percentiles -
not just averages — because production SLAs are defined at p99, not mean.

- **TTFT** — Time to First Token, how fast the model starts responding
- **TPOT** — Time Per Output Token, reported as mean, p50, p90, p99
- **ITL** — Inter-Token Latency per token position, up to 1000 tokens
- **Throughput** — tokens per second under sustained load
- **Perplexity** — quality on WikiText-103, not a single reference sentence

| Section | Techniques | Key Question |
|---------|------------|--------------|
| Quantization | float32, float16, int8, int4 NF4, compiled, AWQ | Does int4 beat float16 on GPU? |
| Pruning | Sparsity 10%, 30%, 50%, 70% | Where does quality collapse? |
| Distillation | Teacher (OpenLLaMA 3B) vs Student (TinyLlama 1.1B) | Does the speed advantage justify the quality gap? |
| Runtime | ONNX and torch.compile() | Which runtime wins on T4? |
| Flash Attention | FA on vs off × sequence length | Does ITL stay flat at 1000 tokens? |
| Serving | Batch size 1, 2, 4, 8, 16, 32 | What is the throughput sweet spot? |
| Context Length | Prompt 32 to 1024 tokens | Does TTFT scale quadratically? |
| KV Cache | Generate up to 1000 tokens | When does memory pressure hit? |

## How to Reproduce

All experiments are designed to run end-to-end with a single command after environment setup. Full reproducibility is a core
requirement of this project - not an afterthought

### 1. Clone the repository

git clone https://github.com/adinfarel/llm-inference-benchmark.git  
cd llm-inference-benchmark

### 2. Set up the environment

bash setup.sh

This script installs all dependencies, confirms CUDA availability,
and validates the model can be loaded before any experiment runs.

### 3. Run all experiments

python experiments/run_all.py

Or run individual sections:

python experiments/run_quantization.py  
python experiments/run_flash_attention.py  
python experiments/run_kv_cache.py  

### 4. View results

Results are saved to results/metrics/ as CSV files.  
Analysis notebooks are in analysis/ - open them in order,  
starting from 01_quantization.ipynb.  

> Note: All experiments require a CUDA-capable GPU with minimum 16GB VRAM
> Tested on GCP n1-standard-4 with NVIDIA t4

## Results  

Results are updated as each experiment completes.
Raw metrics are in `results/metrics/all_results.csv`.
Full drill-down per experiment is in `docs/`.

### Quantization

| label | ttft_p50_ms | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb | perplexity |
|-----------|-------------|------------|------------|----------------|----------------|------------|
| float32 | 70.5 | 26.7 | 58.2 | 32.7 | 5912 | 7.817 |
| float16 | 44.5 | 35.2 | 63.4 | 25.6 | 3589 | 7.817 |
| int8 | 209.8 | 123.1 | 196.6 | 7.6 | 2680 | 7.855 |
| int4 | 96.9 | 67.5 | 114.3 | 13.4 | 2229 | 8.111 |
| compiled | 59.0 | 20.8 | 302.7 | 2.5 | 2453 | 7.817 |
| awq | — | — | — | — | — | — |
| gptq | — | — | — | — | — | — |

*AWQ and GPTQ pending GCP environment — requires CUDA 12.1 for auto-gptq build.*  
Full analysis → `docs/01_quantization.md`

### Pruning

| label | ttft_p50_ms | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb | perplexity |
|-------------|-------------|------------|------------|----------------|----------------|------------|
| float16 (base) | 44.5 | 35.2 | 63.4 | 25.6 | 3589 | 7.817 |
| pruned_10pct | 34.3 | 28.7 | 48.2 | 31.3 | 3589 | 7.830 |
| pruned_30pct | 44.0 | 28.2 | 40.6 | 26.0 | 3588 | 8.168 |
| pruned_50pct | 33.1 | 28.1 | 111.1 | 30.3 | 3588 | 11.820 |
| pruned_70pct | 33.8 | 28.6 | 48.3 | 31.3 | 3588 | 246.604 |

Full analysis → `docs/02_pruning.md`

### Distillation

| label | ttft_p50_ms | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb | perplexity |
|------------|-------------|------------|------------|----------------|----------------|------------|
| teacher_3B | 79.1 | 36.6 | 238.5 | 21.2 | 9295 | 21.4 |
| student_1B | 35.4 | 30.3 | 53.0 | 29.7 | 3589 | 7.817 |

*Teacher perplexity not directly comparable — domain mismatch. See `docs/04_distillation.md`.*  
Full analysis → `docs/04_distillation.md`

### Flash Attention

| label | prompt_length | ttft_p50_ms | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb |
|----------------------|---------------|-------------|------------|------------|----------------|----------------|
| standard_len(128) | 128 | 45.5 | 28.0 | 49.1 | 34.2 | 2118 |
| standard_len(256) | 256 | 69.2 | 34.0 | 60.6 | 26.1 | 2136 |
| standard_len(512) | 512 | 110.7 | 34.3 | 68.4 | 27.1 | 2196 |
| standard_len(1024) | 1024 | 251.0 | 34.5 | 64.1 | 24.7 | 2411 |
| flash_attn_len(128) | 128 | 49.5 | 34.3 | 61.1 | 26.6 | 2121 |
| flash_attn_len(256) | 256 | 68.3 | 34.8 | 64.0 | 25.8 | 2146 |
| flash_attn_len(512) | 512 | 138.2 | 35.0 | 66.2 | 25.2 | 2219 |
| flash_attn_len(1024) | 1024 | 350.7 | 34.4 | 73.7 | 24.6 | 2471 |

Full analysis → `docs/05_flash_attention.md`

### KV Cache

| label | max_new_tokens | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb |
|------------------|----------------|------------|------------|----------------|----------------|
| kv_cache_100tok  | 100  | 30.94 | 55.60 | 29.1 | 2113 |
| kv_cache_300tok  | 300  | 30.15 | 51.43 | 30.2 | 2122 |
| kv_cache_500tok  | 500  | 29.87 | 51.61 | 30.6 | 2132 |
| kv_cache_750tok  | 750  | 29.81 | 53.04 | 30.4 | 2143 |
| kv_cache_1000tok | 1000 | 30.01 | 51.31 | 30.4 | 2155 |

Full analysis → `docs/06_kv_cache.md`

### Context Length

| label | prompt_length | ttft_p50_ms |
|----------------------|---------------|-------------|
| context_len(32) | 34 | 61.5 |
| context_len(64) | 67 | 59.9 |
| context_len(128) | 133 | 62.0 |
| context_len(256) | 265 | 60.2 |
| context_len(512) | 518 | 114.9 |
| context_len(1024) | 1024 | 288.6 |

*p90/p99 not reported — inflated by warmup artifact at first run per length.
p50 is valid as representative steady-state measurement.*  
Full analysis → `docs/07_context_length.md`

### Batching

| batch_size | throughput_p50_tps | latency_p50_ms | latency_p99_ms |
|------------|-------------------|----------------|----------------|
| 1  | 32.2  | 1555.1 | — |
| 2  | 51.1  | 979.0  | — |
| 4  | 125.0 | 400.0  | — |
| 8  | 246.5 | 202.8  | — |
| 16 | 432.6 | 115.6  | — |
| 32 | 897.9 | 55.7   | 58.2 |

*p99 only available for batch=32 — other batch sizes pending re-run with save=True.*  
Full analysis → `docs/08_batching.md`

### ONNX Runtime

| label | ttft_p50_ms | itl_p50_ms | itl_p99_ms | throughput_tps | peak_memory_mb | perplexity |
|--------------|-------------|------------|------------|----------------|----------------|------------|
| float16 (baseline) | 44.5 | 35.2 | 63.4 | 25.6 | 3589 | 7.817 |
| onnx_runtime | 68.0 | 33.6 | 96.1 | 22.7 | 2774 | 7.816 |

*ONNX Runtime via ORTModelForCausalLM (Optimum). ITL p50 4.5% better,
TTFT 53% worse. Full dispatch elimination not achieved due to
HuggingFace wrapper overhead and dynamic KV cache shapes.*  
Full analysis → `docs/09_onnx.md`

## Findings

Key findings are updated as research questions are answered.
Full hardware-level explanation for each finding is in the linked docs.

**Q1 — Int4 quantization speed-up inference on GPU?** — *pending*   
**Q2 — Between nf4 vs AWQ, which the best?** — *pending*    
**Q3 — Flash Attention ITL flattening on T4**   
ITL flattening not observed. itl_p50 identical between standard and
flash_attn at 1024 tokens (34.5ms vs 34.4ms). Flash Attention via SDPA
is 40% slower at TTFT for 1024 token prompts (350.7ms vs 251.0ms).
Root cause: T4 Turing lacks asynchronous memory copy units that Flash
Attention tiling relies on. SDPA dispatcher overhead outweighs HBM
traffic reduction on this architecture. Expected to differ on Ampere.
Full breakdown → `docs/05_flash_attention.md`

**Q4 — KV cache pressure inflection point**     
Pressure not visible within TinyLlama 1.1B context window (1000 tokens).
KV cache at token 1000 = 180MB, contributing ~0.6ms to ITL against
30ms baseline from weight loading — below measurement noise floor.
Inflection point estimated at ~10,000–12,000 tokens, beyond TinyLlama
max context of 2048. Pressure would be visible at this scale with a
larger model (7B+) where KV cache per token is significantly larger.
Full breakdown → `docs/06_kv_cache.md`
      
**Q5 — Optimal batch size for throughput on T4**    
No throughput saturation observed up to batch=32 for TinyLlama 1.1B.
Throughput grows from 32.2 tps at batch=1 to 897.9 tps at batch=32 —
a 27.9x improvement from amortizing 2.2GB weight loading across more requests.
Root cause of no plateau: TinyLlama only uses ~2.7GB of T4's 16GB VRAM
at batch=32, leaving significant memory and compute headroom.
Scaling efficiency drops from 100% to ~87% at batch=32 due to
padding overhead and scheduling cost in naive batching.
Production recommendation: batch=16 for real-time chat (SLA <100ms,
latency=115ms borderline), batch=32 for async pipelines.
True optimal batch size for TinyLlama on T4 estimated at batch=64–128
where compute saturation would begin.
Full breakdown → `docs/08_batching.md`

**Q6 — TTFT scaling with prompt length**     
TTFT stays flat between 34 and 265 tokens (60–62ms) because fixed
overhead — weight loading 2.2GB, 220 kernel launches, MLP and layer norm —
dominates and masks the quadratic attention component entirely.
Above 265 tokens, attention matrix grows large enough to add measurable
HBM traffic. At 518 tokens TTFT reaches 114.9ms (1.91x baseline),
at 1024 tokens 288.6ms (4.79x baseline) — super-linear, consistent with
mixed O(N²) attention and O(N) linear operations.
Crossover point: between 265 and 518 tokens on T4.
Production implication: benchmarking with short prompts gives false
confidence — quadratic cost only manifests at production-realistic
context lengths (RAG, long conversation, multi-modal).
Full breakdown → `docs/07_context_length.md`    

**Q7 — Production cost of distillation**  
Student (1.1B) is 2.23x faster at TTFT and uses 2.6x less VRAM than teacher (3B).
ITL p99 is 4.5x lower — student never breaches 100ms across 300 token positions,
teacher breaches it 23 times. Quality gap cannot be measured by perplexity alone
due to training domain mismatch between OpenLLaMA and TinyLlama.
Full breakdown → `docs/04_distillation.md`

**Q8 — torch.compile dispatch overhead**  
Kernel fusion works — itl_p50 improves 2x over float16 (17.8ms vs 35.6ms).
Overall throughput degrades because dynamic KV cache shapes trigger recompilation
at every new sequence length. itl_std of 962ms confirms recompilation spikes.
Fix planned: `torch.compile(model, dynamic=True)` in GCP re-run.
Full breakdown → `docs/01_quantization.md`

**Q9 — ONNX Runtime dispatch overhead elimination**
Partial dispatch elimination confirmed — ITL p50 improved 4.5%
(33.6ms vs 35.2ms) but TTFT degraded 53% (68.0ms vs 44.5ms).
Throughput 11% lower than float16 baseline (22.7 vs 25.6 tps).
Memory footprint 23% smaller (2,774MB vs 3,589MB).
Perplexity identical (7.816 vs 7.817) — quality preserved.
Full elimination not achieved: ORTModelForCausalLM wrapper retains
Python generate() loop overhead, dynamic KV cache shapes prevent
full static graph optimization. Raw OnnxRuntime InferenceSession
or TensorRT expected to show larger benefit — planned for GCP Phase 2.
Full breakdown → `docs/09_onnx.md`

## Limitations

**AWQ and GPTQ excluded from quantization experiment**  
auto-gptq requires CUDA kernel compilation from source.
Colab ships CUDA 12.8 — no pre-built wheels available for this version.
Both techniques are planned for the dedicated GCP environment with CUDA 12.1.

**torch.compile recompilation on dynamic KV cache shapes**  
mode="default" recompiles when KV cache tensor shape changes each decode step.
Planned fix: dynamic=True flag in GCP re-run.

**Distillation perplexity comparison is not valid across models**  
OpenLLaMA 3B (RedPajama) and TinyLlama 1.1B (SlimPajama) have different
training domains. Perplexity on WikiText-103 is not comparable across models
with different training distributions. Task-specific evaluation planned for Phase 2.

**All experiments run at batch size 1**  
Batch size 1 is IO-bound and does not reflect GPU compute utilization
at production scale. Serving experiment (Q5) will address this directly.

**Unstructured pruning on Turing architecture (T4)**  
T4 has no Sparse Tensor Core support. Unstructured zeros are computed
identically to non-zero weights — no speed or memory benefit observed.
Structured 2:4 sparsity requires Ampere architecture (A100, A10G) or newer.

**Flash Attention benefit not observable on T4 Turing architecture**
T4 lacks asynchronous memory copy units that Flash Attention tiling
relies on for maximum benefit. SDPA dispatcher overhead on PyTorch 2.10
outweighs HBM traffic reduction at all tested sequence lengths.
Flash Attention ITL flattening is expected on Ampere (A100, A10G) or newer.
Planned for GCP Phase 2.

**SDPA used instead of flash_attention_2 package**
flash_attention_2 requires source compilation — no pre-built wheels for
CUDA 12.8. SDPA implements the same algorithm but adds dispatcher overhead
on Turing architecture. Results may differ with flash_attention_2 on
compatible CUDA version.

**Periodic recompilation spikes every ~100 tokens**
Both standard and SDPA show ITL spikes at token position ~100 and ~200.
Root cause: KV cache tensor shape change triggers PyTorch dynamic graph
recompilation. Not memory pressure — recompilation overhead.
Fix: torch.compile with dynamic=True. Planned for GCP re-run.

**ONNX Runtime via wrapper, not raw InferenceSession**
ORTModelForCausalLM from Optimum retains Python generate() loop —
dispatch overhead partially eliminated, not fully.
Raw OnnxRuntime InferenceSession would show cleaner benefit
but requires manual token generation loop without HuggingFace utilities.
Planned for GCP Phase 2 alongside TensorRT comparison.

**TensorRT not tested — requires GCP Cloud Console access**
TensorRT is NVIDIA's production inference optimizer with T4-specific
kernel compilation. Expected to outperform both PyTorch eager and
ONNX Runtime by compiling model to hardware-optimized execution plan.
Requires GCP environment with proper CUDA toolkit — planned for Phase 2.

**ONNX perplexity uses manual logit extraction**
ORTModelForCausalLM does not support labels parameter in forward pass.
Perplexity computed via manual logit extraction and CrossEntropyLoss.
Result verified identical to float16 baseline (7.816 vs 7.817) —
methodology documented in src/benchmark_core.py measure_perplexity().