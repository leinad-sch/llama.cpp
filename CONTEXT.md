# Context

## Goal
Adopt V100 (sm70) performance optimizations from the llama.cpp-v100 fork
(https://github.com/1115714829/llama.cpp-v100) into this branch. Target hardware,
in priority order:
1. Dense models on a single V100
2. V100 + RTX 4060 laptop (heterogeneous 2-GPU)
3. MoE
4. other

The fork branched from upstream `08618ff8e735141d8e4e5be28e6d6af170e4757b`.
Adopted commit messages and comments are translated to English. Third-party
kernel license headers (1Cat-vLLM / v100-skinny / sglang-V100,
ai-bond/flash-attention-v100, Strata/AC922) are preserved.

## Adoption clusters
- Cluster 1 (dense, single V100): skinny Q4_K/Q8_0 repack kernels + sm70 flash
  attention. Biggest decode + prefill wins.
- Cluster 2 (V100 + 4060): allreduce-push, partner-fetch, numa-plan, seqpar,
  3-tier placement.
- Cluster 3 (MoE): ggml-cuda-expert, moe-grouped, fused-expert, qsa.
- Cluster 4 (other): fp8-sm70, gdn-chunk-sm70, diagmask.

## Commit conventions
- Do NOT add `Assisted-by:` to commit messages. Explicit user preference.
- Keep commit messages concise.
