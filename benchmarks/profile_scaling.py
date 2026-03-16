import time
import torch
import argparse

def profile_attention(length: int, d_model: int = 4096):
    print(f"Profiling context length: {length}...")
    q = torch.randn(1, 32, length, d_model // 32).cuda().half()
    k = torch.randn(1, 32, length, d_model // 32).cuda().half()
    v = torch.randn(1, 32, length, d_model // 32).cuda().half()
    
    torch.cuda.synchronize()
    start = time.time()
    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Latency: {(end - start) * 1000:.2f} ms")
    print(f"VRAM Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 16384])
    args = parser.parse_args()
    
    # Simple check for GPU
    if torch.cuda.is_available():
        for l in args.lengths:
            profile_attention(l)
    else:
        print("CUDA not available. Benchmarking on CPU (not recommended for HPC research).")
