# LLM Inference Optimization Benchmark

> Benchmarking 10+ inference optimization techniques on TinyLlama 1.1B
> quantization, pruning, distillation, flash attention, onnx, and serving strategies
> on a GPU environment. Every result is backed by profiler traces and explained
> from hardware level, not just reported

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

That exploration raised 7 questions that could not be answered on CPU alone. This project exists to answer all of them - on a GPU where these techniques were actually designed to run.
