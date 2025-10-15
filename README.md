# GPU-Mode Reference Kernels

This is a personal fork of the [GPU-Mode reference-kernels repository](https://github.com/gpu-mode/reference-kernels) for writing and testing GPU kernels locally. The original repository contains reference implementations and competition problems for GPU kernel optimization challenges hosted by the [GPU-Mode community](https://www.gpumode.com/).

**Purpose of this fork:**
- ✍️ Write and test custom CUDA/Triton kernels locally on WSL2
- 🧪 Experiment with kernel optimizations before submission
- 📚 Learn GPU programming through hands-on practice
- 🏃 Run benchmarks on local hardware (RTX 2050, etc.)

## Table of Contents
- [Available Problems](#available-problems)
- [Getting Started](#getting-started)
- [Local Development & Testing](#local-development--testing)
- [Problem Structure](#problem-structure)
- [Implementation Examples](#implementation-examples)
- [Example Workflows](#example-workflows)

---

## Available Problems

### PMPP Practice Problems (`problems/pmpp/`)

Fundamental GPU programming problems based on the "Programming Massively Parallel Processors" (PMPP) curriculum. Perfect for learning CUDA optimization techniques.

**Problems:**
- `vectoradd_py` - Vector addition (great starting point!)
- `vectorsum_py` - Vector summation/reduction
- `matmul_py` - Matrix multiplication
- `conv2d_py` - 2D convolution
- `histogram_py` - Histogram computation
- `prefixsum_py` - Parallel prefix sum (scan)
- `grayscale_py` - Image grayscale conversion
- `sort_py` - Parallel sorting

### Other Problem Sets

- `problems/amd/` - AMD kernel optimization challenges
- `problems/bioml/` - Bioinformatics and ML kernels
- `problems/amd_distributed/` - Multi-GPU distributed kernels

---

## Getting Started

### Prerequisites
- **NVIDIA GPU** (for CUDA kernels) with compute capability 7.0+ recommended
- **Python 3.8+**
- **CUDA Toolkit 12.1+** (for compiling inline CUDA kernels)
- **PyTorch** with CUDA support

### Quick Start (Linux/WSL2)

1. **Clone this repository:**
```bash
git clone https://github.com/thefirehacker/reference-kernels.git
cd reference-kernels
```

2. **Create a Python virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install PyTorch with CUDA support:**
```bash
# For CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# For CUDA 12.3
pip install --index-url https://download.pytorch.org/whl/cu123 torch
```

4. **Verify CUDA is working:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Local Development & Testing

### WSL2 on Windows (Recommended for Windows Users)

If you're running **Windows with WSL2**, you can test kernels locally on your NVIDIA GPU with full CUDA support!

📖 **[Complete WSL2 + CUDA + PyTorch Setup Guide](problems/docs/WSL2_VectorAdd_Quickstart.md)**

This comprehensive guide provides a **known-good, reproducible path** from a fresh WSL2 Ubuntu installation to running compiled CUDA kernels:

**What's covered:**
- ✅ Installing CUDA Toolkit 12.1 on WSL2 Ubuntu 22.04
- ✅ Configuring environment variables (`CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`)
- ✅ Installing PyTorch with matching CUDA version
- ✅ Setting up the vectoradd problem with inline CUDA
- ✅ Running and benchmarking your kernels locally
- ✅ Troubleshooting common issues (OOM, compilation errors, etc.)

**Tested Hardware:**
- NVIDIA RTX 2050 (4GB VRAM, SM 8.6)
- NVIDIA RTX 3060 and newer
- Most modern NVIDIA GPUs with CUDA support

**Why WSL2?**
- Native CUDA support via GPU passthrough (no virtualization overhead)
- Full `nvcc` compiler access for inline CUDA kernels
- Linux development environment on Windows
- Seamless integration with Windows filesystem

### Running Benchmarks Locally

Once you have CUDA and PyTorch set up, each problem includes a `run_local.py` script for local testing:

```bash
cd problems/pmpp/vectoradd_py
python run_local.py
```

**What `run_local.py` does:**
1. **Correctness validation** - Checks your kernel output against the reference implementation
2. **Automatic sizing** - Detects available VRAM and picks appropriate test sizes
3. **Warmup runs** - Ensures GPU is ready and caches are warm
4. **Precision timing** - Uses CUDA events for accurate millisecond measurements
5. **Statistical reporting** - Reports mean, best, and worst case timings

**Example output:**
```
Detected dims=2, free≈3.4 GB, using sizes=[4096, 8192, 12288, 16384] (s_max≈16384)
size=4096:  mean=0.3421 ms, best=0.3401 ms, worst=0.3502 ms
size=8192:  mean=1.2134 ms, best=1.2089 ms, worst=1.2301 ms
size=12288: mean=2.7892 ms, best=2.7801 ms, worst=2.8123 ms
size=16384: mean=4.9123 ms, best=4.9034 ms, worst=4.9456 ms
```

### Alternative: Using the Official Evaluator

To test exactly as the competition evaluator does:

```bash
cd problems/pmpp/vectoradd_py

# Create test cases
printf "size: 4096; seed: 42\nsize: 8192; seed: 43\n" > tests.txt

# Run correctness tests
exec 3>&1
POPCORN_FD=3 python ../eval.py test tests.txt

# Run benchmarks
POPCORN_FD=3 python ../eval.py benchmark tests.txt
```

---

## Problem Structure

Each problem follows a standardized structure for consistency and ease of use:

```
problems/pmpp/vectoradd_py/
├── reference.py          # Reference PyTorch implementation (ground truth)
├── task.py              # Type definitions for inputs/outputs
├── task.yml             # Problem specification and test configurations
├── submission.py        # YOUR kernel implementation (what you edit!)
├── template.py          # Empty template to start from
├── utils.py             # Helper functions (correctness checking, seeding)
├── eval.py              # Official competition evaluator
├── run_local.py         # Local benchmark script (auto-sizing, timing)
└── solutions/           # Example reference solutions
    ├── correct/
    │   ├── submission_pytorch.py      # Pure PyTorch solution
    │   ├── submission_triton.py       # Triton kernel solution
    │   └── submission_cuda_inline.py  # Inline CUDA solution
    └── incorrect/       # Examples of common mistakes (for learning)
```

### Key Files Explained

**`reference.py`**
- Contains the reference PyTorch implementation
- Defines `generate_input()` for creating test cases
- Defines `check_implementation()` for correctness validation
- Your kernel must match this output (within tolerance)

**`task.py`**
- Defines type hints: `input_t` and `output_t`
- Documents expected tensor shapes, dtypes, and devices
- Ensures type safety across submissions

**`task.yml`**
- Specifies problem metadata (name, description, difficulty)
- Defines test case configurations
- Sets timeout limits and resource constraints

**`submission.py`**
- **This is where you write your kernel!**
- Must implement `custom_kernel(data: input_t) -> output_t`
- Can use PyTorch, Triton, inline CUDA, or CuPy
- Will be evaluated for both correctness and performance

**`utils.py`**
- `set_seed()` - Ensures reproducible random inputs
- `verbose_allclose()` - Detailed tensor comparison with error reporting
- `match_reference()` - Convenience function for correctness checking

---

## Implementation Examples

### Different Approaches

You can implement kernels using any of these approaches:

#### 1. **Pure PyTorch** (easiest, baseline performance)
```python
def custom_kernel(data: input_t) -> output_t:
    A, B = data
    return A + B
```

#### 2. **Triton** (high-level, good performance)
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    c = a + b
    tl.store(c_ptr + offset, c, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    A, B = data
    C = torch.empty_like(A)
    N = A.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](A, B, C, N, BLOCK_SIZE=1024)
    return C
```

#### 3. **Inline CUDA** (maximum control, best performance potential)
```python
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty_like(A);
    int N = A.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        N
    );
    return C;
}
"""

add_module = load_inline(
    name='add_cuda',
    cpp_sources='torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B);',
    cuda_sources=cuda_source,
    functions=['add_cuda']
)

def custom_kernel(data: input_t) -> output_t:
    return add_module.add_cuda(data[0], data[1])
```

### Testing Your Implementation

When testing your kernel locally, ensure:

✅ **Correctness** - Output matches reference implementation  
✅ **Type safety** - Follow `input_t` and `output_t` definitions  
✅ **Device compatibility** - Tensors must stay on CUDA device  
✅ **Numerical precision** - Match reference within tolerance (typically `rtol=1e-5, atol=1e-8`)  
✅ **Performance** - Track improvements with `run_local.py` benchmarks

---

## Example Workflows

### Workflow 1: Starting from Scratch

```bash
# 1. Pick a problem
cd problems/pmpp/vectoradd_py

# 2. Start with the template
cp template.py submission.py

# 3. Implement your kernel (edit submission.py)
nano submission.py  # or use your favorite editor

# 4. Test locally
python run_local.py

# 5. If it works, optimize and iterate!
```

### Workflow 2: Learning from Examples

```bash
cd problems/pmpp/vectoradd_py

# Try the PyTorch solution first
cp solutions/correct/submission_pytorch.py submission.py
python run_local.py

# Compare with Triton
cp solutions/correct/submission_triton.py submission.py
python run_local.py

# Try inline CUDA for maximum performance
cp solutions/correct/submission_cuda_inline.py submission.py
python run_local.py
```

### Workflow 3: Optimization Iteration

```bash
# Test baseline
python run_local.py > baseline.txt

# Make optimization changes to submission.py
# ... edit code ...

# Compare performance
python run_local.py > optimized.txt
diff baseline.txt optimized.txt
```

---

## Resources

- 📖 [WSL2 Setup Guide](problems/docs/WSL2_VectorAdd_Quickstart.md) - Complete setup for Windows/WSL2
- 🌐 [GPU-Mode Website](https://www.gpumode.com/) - Original community and resources
- 💬 [GPU-Mode Discord](https://discord.gg/gpumode) - Join the community
- 📚 [Original Repository](https://github.com/gpu-mode/reference-kernels) - Upstream source

---

## Tips for Learning & Optimization

💡 **Start simple** - Get a correct PyTorch implementation first, then optimize  
💡 **Compare approaches** - Try PyTorch, Triton, and CUDA versions to understand trade-offs  
💡 **Profile before optimizing** - Use `nvprof` or `nsys` to find actual bottlenecks  
💡 **Check occupancy** - Higher occupancy often (but not always) means better performance  
💡 **Memory coalescing** - Ensure adjacent threads access adjacent memory for bandwidth  
💡 **Shared memory** - Use it strategically to reduce expensive global memory accesses  
💡 **Test on various sizes** - Verify your kernel works across different input scales  
💡 **Benchmark locally** - Use `run_local.py` to measure real performance on your GPU  
💡 **Learn from solutions** - Study the `solutions/` folder for different implementation techniques  
💡 **Iterate quickly** - WSL2 setup enables fast compile-test-optimize cycles




