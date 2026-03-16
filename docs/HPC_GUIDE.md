# High-Performance Computing (HPC) Setup Guide

## 1. Slurm Integration
To run long-context experiments on a multi-node cluster, use the provided sbatch template:
```bash
sbatch infrastructure/slurm_job.sh
```

## 2. Distributed Environment
Ensure `nccl` backend is properly initialized for inter-GPU communication when scaling context via sequence parallelism.

## 3. Triton Kernels
For custom block attention (MoBA), kernels are compiled JIT. Ensure your environment has the correct LLVM and CUDA toolkit versions.
