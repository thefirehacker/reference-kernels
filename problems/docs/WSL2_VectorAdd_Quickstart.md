
# WSL2 + CUDA + PyTorch Inline Kernel: End‑to‑End Quickstart (vectoradd_py)

This single page takes you **from a fresh WSL2 Ubuntu shell** to a **compiled CUDA inline kernel** running the `problems/pmpp/vectoradd_py/run_local.py` benchmark.

> **Assumptions**
> - Windows has the latest NVIDIA driver installed.
> - You’re using **WSL2 Ubuntu 22.04**.
> - GPU example: **RTX 2050 (SM 8.6)** — works for others too.
> - You want a **known‑good path** that compiles CUDA (`nvcc`) and runs PyTorch’s JIT extension.

---

## 0) Verify GPU passthrough (quick sanity)
```bash
# Inside WSL Ubuntu
nvidia-smi
```

You should see your GPU and a CUDA version in the header (e.g., 12.x).

---

## 1) Install CUDA Toolkit 12.1 (compiler + headers)

> We use NVIDIA’s official repo; this installs **nvcc** into `/usr/local/cuda-12.1`.

```bash
# Stay out of your repo folder to avoid stray files
cd ~

# Add NVIDIA CUDA keyring/repo and update
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 12.1
sudo apt-get install -y cuda-toolkit-12-1

# Export environment for this shell AND persist in ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# Load it now
source ~/.bashrc

# Verify nvcc is available
nvcc --version
```

> If your Windows driver advertises CUDA 12.3 in `nvidia-smi`, you can use `cuda-toolkit-12-3` instead. The PyTorch wheel selection below should then be `cu123` accordingly.

---

## 2) Clone your repo and create a Python venv

```bash
# Clone YOUR fork (or clone upstream if you prefer)
cd ~
git clone https://github.com/thefirehacker/reference-kernels
cd reference-kernels

# Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install PyTorch that matches CUDA 12.1
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# (Optional but recommended) extra utilities
pip install pyyaml ninja
```

> If you installed `cuda-toolkit-12-3` above, use this instead:
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cu123 torch
> ```

Quick torch‑CUDA check:
```bash
python - << 'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## 3) Prepare the PMPP `vectoradd_py` problem

```bash
cd problems/pmpp/vectoradd_py

# (Optional) Use the official correct inline CUDA solution as your submission
#   Skip these two lines if you already wrote your own submission.py
cp solutions/correct/submission_cuda_inline.py submission.py
git status  # just to see what's there; not required
```

---

## 4) Create the local benchmark runner (`run_local.py`)

The `run_local.py` below:
- auto‑detects whether the tensors are 1D or 2D,
- picks sizes that fit available VRAM (~80% headroom),
- verifies correctness vs `reference.py`,
- times the kernel with CUDA events and prints mean/best/worst (ms).

```bash
cat > run_local.py << 'PY'
import math, inspect, torch
from submission import custom_kernel
from reference import check_implementation, generate_input

def bench(case, warmup=10, iters=100):
    # correctness once
    out = custom_kernel(case)
    err = check_implementation(case, out)
    assert not err, err

    # warmup
    for _ in range(warmup):
        custom_kernel(case)
    torch.cuda.synchronize()

    # timing with CUDA events (ms)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        custom_kernel(case)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return sum(times)/len(times), min(times), max(times)

# detect arg name (size vs N) and tensor dimensionality
param = next(iter(inspect.signature(generate_input).parameters))
probe = generate_input(**{param: 512, "seed": 42})
A = probe[0]
dims = A.dim()
bytes_per_elem = A.element_size()

# pick sizes that fit current free VRAM (~80% headroom), accounting for A,B,C
free_bytes, _ = torch.cuda.mem_get_info()
budget = int(free_bytes * 0.8)

if dims == 2:
    # 3 tensors of size s*s, e.g., fp16 => 2 bytes
    s_max = int(math.sqrt(budget / (3 * bytes_per_elem)))
    candidates = [4096, 8192, 12288, 16384]
    sizes = [s for s in candidates if s <= s_max]
    if not sizes: sizes = [max(1024, s_max // 2)]
else:
    # 3 tensors of size s (1D case)
    s_max = int(budget / (3 * bytes_per_elem))
    sizes = [s for s in [1<<18, 1<<20, 1<<22] if s <= s_max]
    if not sizes: sizes = [max(1<<16, s_max // 2)]

print(f"Detected dims={dims}, free≈{free_bytes/1e9:.2f} GB, using sizes={sizes} (s_max≈{s_max})")

for s in sizes:
    try:
        case = generate_input(**{param: s, "seed": 42})
        mean, best, worst = bench(case)
        print(f"{param}={s}: mean={mean:.4f} ms, best={best:.4f} ms, worst={worst:.4f} ms")
    except torch.cuda.OutOfMemoryError:
        print(f"Skipped {s} (OOM)")
        torch.cuda.empty_cache()
PY
```

> **Optional for RTX 2050** (SM 8.6) — faster first compile:
> ```bash
> export TORCH_CUDA_ARCH_LIST="8.6"
> ```

---

## 5) Run the local benchmark

```bash
# You should still be in problems/pmpp/vectoradd_py and your venv should be active
python run_local.py
```

Expected shape of output:
```
Using /home/<you>/.cache/torch_extensions/py310_cu121 as PyTorch extensions root...
Loading extension module add_cuda...
Detected dims=2, free≈3.4 GB, using sizes=[4096, 8192, 12288, 16384] (s_max≈...)
size=4096:  mean=... ms, best=... ms, worst=... ms
size=8192:  mean=... ms, best=... ms, worst=... ms
...
```

> If the 16384 case is slow, consider switching your kernel launch to a **grid‑stride loop** and **cap blocks ≈ SM*32** to avoid launching ~1M tiny blocks. (You can do this later for performance tuning.)

---

## 6) (Optional) Use the PMPP harness instead of `run_local.py`

If you want to run the official evaluator (`../eval.py`) exactly like GPU‑Mode’s bot does, create a small tests file and use its special file‑descriptor logging:

```bash
printf "size: 4096; seed: 42\nsize: 8192; seed: 43\nsize: 16384; seed: 44\n" > tests.txt
exec 3>&1
POPCORN_FD=3 python ../eval.py test tests.txt
POPCORN_FD=3 python ../eval.py benchmark tests.txt
```

---

## Troubleshooting

- **`nvcc: command not found`** → Ensure you installed `cuda-toolkit-12-1` and exported PATH/LD_LIBRARY_PATH. Re‑open terminal or `source ~/.bashrc`.
- **`CUDA_HOME not set`** → `export CUDA_HOME=/usr/local/cuda-12.1` (and add to `~/.bashrc`).
- **`torch.cuda.is_available() == False`** → install the correct PyTorch wheel (`cu121` or `cu123`) and verify `nvidia-smi` works in WSL.
- **OOM on big sizes** → That’s expected on 4 GB GPUs for 2D `size×size` tensors. Use the auto‑sizing script above or reduce sizes.
- **Slow 16k case** → Use grid‑stride kernel + cap blocks to avoid ~1,048,576 block launches.

---

### Done ✅
You now have a complete, reproducible path from environment setup to a compiled inline CUDA kernel with local benchmarking.
