# Mathematical Complexity of Long-Context Attention

## 1. Quadratic Bottleneck
Standard Self-Attention complexity is $O(N^2 \cdot d)$, where $N$ is sequence length.

## 2. Block Attention Optimization
By segmenting the sequence into blocks of size $B$, the complexity reduces to $O(N \cdot B \cdot d)$.

For $N = 128,000$ and $B = 1,000$:
- Standard: $\sim 16.3 	imes 10^9$ operations.
- Block: $\sim 128 	imes 10^6$ operations.

Moba (Mixture of Block Attention) further optimizes this by introducing sparsity patterns $S$.
