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
    # 3 tensors of size s*s, fp16 → 2 bytes
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
