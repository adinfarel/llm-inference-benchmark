# Hardware intuition - Foundation for All Experiments

This documents is the foundation that explains *why* every optimization technique behaves
the way it does. Before reading any experiment findings, make sure you understand everything here.

---

## 1. Memory Hierarchy

Modern GPUs have three levels of memory. Each level is a trade-off between speed and size -
faster the memory, the smaller it is.
```
Level       Speed           Size            Scope
-------     -----------     -----------     -------------------
Register    ~1 cycle        ~256KB per SM   Private per thread
SRAM        ~20 cycles      ~48-192KB per SM  Shared within 1 SM
HBM         ~200 cycles     16-80GB         Global, all SMs
```

### Why three levels exist

Imagine if everything lived in registers - registers are the fastest memory, so why not put
everything there?

Because registers are tiny. One SM has ~256KB of register space. TinyLlama weights alone are 4.4GB.
There is no world where 4.4GB fits in 256KB.

So the hierarchy exists because of physics - faster memory costs more to manufacture and generates more heat. You cannot have both fast and large at the same time.

### How data move during inference

When a model runs a forward pass, this is what happens at the memory level:
```
1. Weights loaded from HBM → SRAM
   (slow, ~200 cycles, but only once per layer)

2. Active computation happens in registers
   (fast, ~1 cycle, threads do math here)

3. Results written back to HBM
   (slow again, but necessary for next layer to read)
```

Every time data moves between levels, cycles are spent waiting.
This waiting is the bottleneck. Not the math - the waiting

### Why HBM is meeting point

Registers are private per thread. One matrix multiply operation
splits across thousands of threads on multiple SMs. After each
thread finishes its piece of the computation, the results need
to meet somewhere all threads can access.

That meeting point is HBM — the only memory visible to all SMs
at the same time.

SRAM cannot serve this role because SRAM is local to one SM only.
Thread on SM-1 cannot read SRAM of SM-2.
```
SM-1 threads compute → results go to HBM
SM-2 threads compute → results go to HBM
SM-3 threads compute → results go to HBM

Next operation reads from HBM → distributes to all SMs again
```

This round trip to HBM happens between every layer in the model.
TinyLlama has 22 layers - that is 22 round trips minimum per forward pass

### The number that matters
```
HBM Bandwidth on T4 = ~300GB/s

TinyLlama float32 weights = 4.4GB
Loading weights for one forward pass = 4.4GB / 300 GB/s
                                     = ~14.7ms just for data movement

And this happens every single token generated.
100 tokens = 100 × 14.7ms = 1470ms just moving data.
The actual math takes much less time than the data movement.

This is why LLM inference is memory bandwidth bound,
not compute bound.
```

## 2. GPU Execution Model

### Threads, Warps, and SMs

When a GPU runs an operation like matrix multiply, it does not
run it in one big sequential process. It splits the work into
thousands of small units called **threads**, and runs them
all at the same time.
```
1 matrix multiply operation
    → split into thousands of threads
    → threads grouped into warps (32 threads each)
    → warps assigned to SMs
    → all SMs run in parallel
```

A thread is the smallest unit of work. Each thread handles
one small piece of the computation — for example, one element
of the output matrix.

A warp is a group of 32 threads that always run together
as one unit. The GPU scheduler cannot split a warp — either
all 32 threads run, or none of them do.

An SM (Streaming Multiprocessor) is the physical processor
that runs warps. The T4 GPU has 40 SMs. Each SM can run
multiple warps at the same time.
```
T4 GPU:
    40 SMs
    Each SM: up to 32 warps simultaneously
    Each warp: 32 threads
    Total: 40 × 32 × 32 = 40,960 threads running at once
```

### Kernel Launch

A **kernel** is a function that runs on the GPU. Every time
PyTorch executes one operation — one matrix multiply, one
softmax, one layer norm — it launches a kernel.

Launching a kernel has a fixed overhead cost of ~5-20 microseconds,
regardless of how small or large the operation is.
```
Kernel launch sequence:
    CPU decides what to run
    CPU sends launch command to GPU driver
    GPU driver schedules the kernel
    GPU starts executing
    
    Fixed cost: ~5-20 microseconds per launch
    This happens before any actual computation begins
```

TinyLlama has 22 layers. Each layer has roughly 10 operations
(Q, K, V projections, attention, MLP, layer norm, etc).
That is approximately 220 kernel launches per token generated.
```
220 launches × 10 microseconds = 2.2ms overhead per token
purely from kernel launch, before any math happens
```

### Why GPU is Asynchronous

When Python calls a GPU operation, it does not wait for the GPU
to finish. Python sends the command and immediately moves to the
next line of code. The GPU runs the command in the background.
```
Python:                     GPU:
model.generate(...)  →      starts computing...
next line of code    →      still computing...
next line of code    →      still computing...
                            done.
```

This is why `torch.cuda.synchronize()` is critical for accurate
timing. Without it, you measure how fast Python *dispatches*
the command, not how fast the GPU *executes* it.
```

Without synchronize():
    t_start
    model.generate()    ← Python dispatches, returns immediately
    t_end               ← measured ~0.1ms
    GPU still running in background
    Result: completely wrong measurement

With synchronize():
    t_start
    model.generate()    ← Python dispatches
    synchronize()       ← Python waits here until GPU is done
    t_end               ← measured ~420ms
    Result: actual GPU execution time
```

### Why Warmup Matters

The first time a kernel runs, the GPU needs to compile and cache
the kernel code. This one-time compilation adds extra time that
does not represent steady-state performance.
```
Run 1 (no cache):   kernel compile + execute = 800ms  ← not real
Run 2 (cached):     execute only             = 420ms  ← real
Run 3 (cached):     execute only             = 421ms  ← real
```

This is why every benchmark in this project runs one warmup pass
before measuring. We want to measure the model, not the compiler.

## 3. I0-Bound vs Compute-Bound

Every computation has a bottleneck — something that limits
how fast it can go. For GPU workloads, the bottleneck is
always one of two things:
```
IO-Bound:
    Bottleneck = data movement speed
    GPU is fast enough, but it spends most of its time
    waiting for data to arrive from HBM
    ALUs (math units) are idle while waiting

Compute-Bound:
    Bottleneck = math speed
    Data arrives fast enough, but the GPU needs more time
    to finish the actual computation
    HBM is idle while GPU computes
```

Think of it like a factory:
```
IO-Bound   = workers are fast, but raw materials
             arrive too slowly from the warehouse
             Workers sit idle waiting for materials

Compute-Bound = materials arrive fast, but workers
                need more time to process them
                Warehouse is full, workers are the limit
```

### Where LLM Inference Lives

LLM inference at batch size 1 is almost always IO-Bound.

Here is why:
```
TinyLlama float32:
    Model weights = 4.4GB
    Every forward pass = load 4.4GB from HBM
    
    T4 HBM bandwidth = ~300 GB/s
    Time just moving data = 4.4GB / 300GB/s = ~14.7ms
    
    Actual matrix multiply time for 1 token = ~2-3ms
    
    Data movement (14.7ms) >> Computation (2-3ms)
    = IO-Bound
```

The GPU math units (Tensor Cores) are sitting idle most of
the time, waiting for the weights to arrive from HBM.
This is wasted hardware potential.

### Why Batch Size Changes Everything

At batch size 1, the GPU loads 4.4GB of weights to process
just 1 token. At batch size 32, the GPU loads the same 4.4GB
but processes 32 tokens at once.
```
Batch size 1:
    Load 4.4GB → process 1 token
    Compute utilization = very low
    IO-Bound

Batch size 32:
    Load 4.4GB → process 32 tokens simultaneously
    Same data movement cost, 32x more useful work done
    Compute utilization = much higher
    Starts moving toward Compute-Bound
```

This is why throughput improves dramatically with larger
batch sizes — we amortize the IO cost across more tokens.
But latency per request gets worse because each request
has to wait for others in the batch to finish.

This exact trade-off is what the batching experiment will
measure and quantify.

### Why All Optimizations Attack IO, Not Compute

Since LLM inference is IO-Bound, every major optimization
technique is fundamentally about reducing data movement — not
making the math faster.
```
Quantization (int4):
    float32 weight = 4 bytes
    int4 weight    = 0.5 bytes
    
    Same number of weights, 8x smaller
    4.4GB → 0.55GB to load from HBM
    Less data movement = faster

Pruning:
    Zero out weights that contribute little
    Ideally skip loading zeros entirely
    Less data to move from HBM

Distillation:
    Replace large model with small model
    Small model = fewer weights total
    Less data to load from HBM per forward pass

Flash Attention:
    Reorganize attention computation
    Keep intermediate results in SRAM instead of HBM
    Avoid round trips to HBM for attention scores

ONNX / Static Graph:
    Fuse multiple kernels into one
    Intermediate results stay in SRAM
    Fewer HBM round trips between operations
```

Every single technique, at the hardware level, is trying
to answer the same question:

> *How do we move less data between HBM and the compute units?*

## 4. Static vs Dynamic Graph

### How PyTorch Executes Operations

PyTorch uses a **dynamic graph** — also called define-by-run.
This means the computation graph is built and executed at the
same time, line by line, every single forward pass.
```
Every time model(input) is called:

    Python reads line 1  → dispatch kernel 1 to GPU
    Python reads line 2  → dispatch kernel 2 to GPU
    Python reads line 3  → dispatch kernel 3 to GPU
    ...
    Python reads line 220 → dispatch kernel 220 to GPU

This happens every forward pass, every token generated.
Python interpreter is involved in every single step.
```

This is flexible — you can change the graph dynamically,
add conditionals, debug easily. But flexibility has a cost.

### The Dispatch Overhead Problem

Each kernel dispatch from Python to GPU has a fixed cost.
This is not the kernel execution time — this is the overhead
before execution even begins.
```
PyTorch forward pass for 1 token:

    Kernel 1  dispatch: ~10µs  + execute: 2ms
    Kernel 2  dispatch: ~10µs  + execute: 1ms
    Kernel 3  dispatch: ~10µs  + execute: 0.5ms
    ...
    Kernel 220 dispatch: ~10µs + execute: ...

    Total dispatch overhead = 220 × 10µs = 2.2ms
    This overhead exists even if the model weights
    are already loaded and ready in HBM.
```

For a single token that takes ~8ms to generate, 2.2ms
is ~27% pure overhead. The GPU is doing nothing useful
during those 2.2ms — just waiting for Python to tell
it what to do next.

### How Static Graph Solves This

A **static graph** resolves the entire execution plan once,
at load time — before any inference runs. After that,
inference is pure execution with no Python involvement.
```
ONNX static graph:

    Load time (once):
        Read the model graph
        Resolve all operations and their order
        Build optimized execution plan
        Fuse compatible operations together
    
    Inference time (every token):
        Data in → execution plan runs → data out
        No Python interpreter
        No dispatch decisions
        No per-op overhead
```

The execution plan is fixed. ONNX Runtime knows exactly
what to run and in what order before the first token
is ever generated.

### The Real-World Difference
```
PyTorch dynamic (per token):
    220 kernel dispatches from Python
    Python overhead every forward pass
    GPU waits between dispatches
    
    Timeline:
    [dispatch][kernel][dispatch][kernel][dispatch][kernel]...
     ^10µs     ^exec   ^10µs    ^exec   ^10µs     ^exec

ONNX static (per token):
    0 kernel dispatches from Python during inference
    Execution plan runs continuously
    GPU never waits for Python
    
    Timeline:
    [kernel][kernel][kernel][kernel][kernel][kernel]...
     ^exec   ^exec   ^exec   ^exec   ^exec   ^exec
     
    Gaps between kernels = near zero
```

This is visible in the Chrome profiler trace — PyTorch shows
gaps between CUDA kernels where Python dispatch is happening.
ONNX shows kernels packed tightly with minimal gaps.

### Trade-off — Why Not Always Use Static Graph

Static graph is faster but less flexible.
```
Dynamic graph (PyTorch):
    Can change behavior based on input
    Easy to debug — standard Python tools work
    Can use if/else, loops based on runtime values
    Great for research and experimentation
    
Static graph (ONNX):
    Execution plan is fixed at export time
    Cannot change behavior at runtime
    Harder to debug
    Not all PyTorch operations export cleanly to ONNX
    Great for production deployment
```

This project uses PyTorch for all experiments except the
runtime experiment, where we explicitly compare PyTorch
vs ONNX to measure the dispatch overhead elimination.

## 5. Fused Kernel

### The Problem With Separate Kernels

In a standard PyTorch forward pass, every operation is its
own kernel. Each kernel reads input from HBM, computes,
then writes output back to HBM — even if the next operation
immediately needs that output.
```
Without fusion — 3 separate operations:

Operation 1 (Q projection):
    Read weights from HBM    ← slow
    Compute matmul
    Write result to HBM      ← slow

Operation 2 (K projection):
    Read weights from HBM    ← slow
    Compute matmul
    Write result to HBM      ← slow

Operation 3 (Softmax):
    Read QK result from HBM  ← slow
    Compute softmax
    Write result to HBM      ← slow

Total HBM round trips = 6 (3 reads + 3 writes)
Each round trip = ~200 cycles of waiting
```

The intermediate results — the output of operation 1 that
feeds into operation 2 — exist in HBM for just a moment
before being read and discarded. That write + read is pure
waste if we know operation 2 needs it immediately.

### What Kernel Fusion Does

A fused kernel combines multiple operations into one single
kernel. The intermediate results never leave SRAM — they
stay alive in fast memory until all fused operations are done.
```
With fusion — 3 operations fused into 1 kernel:

Fused Kernel (Q + K + Softmax):
    Read weights from HBM        ← once
    Compute Q projection         → result stays in SRAM
    Compute K projection         → result stays in SRAM
    Compute Softmax(QK)          → result stays in SRAM
    Write final result to HBM    ← once

Total HBM round trips = 2 (1 read + 1 write)
Down from 6 — same math, 3x less data movement
```

The key insight: SRAM access is ~10x faster than HBM access.
Keeping intermediate results in SRAM instead of bouncing
them through HBM is a massive win.

### Where Fusion Appears In This Project
```
ONNX Runtime:
    At export time, ORT identifies sequences of operations
    that can be fused together.
    LayerNorm + activation, attention projections, etc.
    This is one reason ONNX is faster than PyTorch —
    not just less dispatch overhead, but also less
    HBM traffic from fused operations.

Flash Attention:
    The most famous example of kernel fusion in LLMs.
    Standard attention = 5-6 separate kernels:
        QK matmul → write to HBM
        Scale     → read + write HBM
        Softmax   → read + write HBM
        AV matmul → read + write HBM

    Flash Attention = 1 fused kernel:
        All of the above stays in SRAM
        HBM accessed only for Q, K, V input
        and final output
        
    At long sequences (1024+ tokens), the attention
    matrix is huge. Standard attention must write this
    entire matrix to HBM. Flash Attention never
    materializes the full matrix — it tiles the
    computation so everything fits in SRAM.
    
    This is why Flash Attention flattens the ITL curve
    at long sequences — less HBM pressure per step.

torch.compile:
    PyTorch's built-in fusion mechanism.
    Analyzes the computation graph at runtime and
    automatically fuses compatible operations.
    Middle ground between dynamic PyTorch and
    fully static ONNX.
```

### The Mental Model To Remember
```
Every time data goes to HBM and comes back = wasted cycles.

Fusion = keep data in SRAM as long as possible.
         Only touch HBM at the very beginning (load input)
         and very end (save output).

The more operations you can fuse,
the fewer HBM round trips,
the faster the inference.

All roads lead back to the same root cause:
HBM is the bottleneck.
Fusion is one way to fight it.
```