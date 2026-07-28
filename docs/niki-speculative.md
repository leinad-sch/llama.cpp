# Niki's Speculative Decoding Enhancements

This document describes changes available in this fork.

- Features 1-3 are authored by Niki
- Feature 4 is from Michael Wand's branch (see below)

- [ngram-mod-v2: Persistent Keyed Hash Table](#ngram-mod-v2-persistent-keyed-hash-table)
- [Skip MTP: Avoid Redundant Draft Model Processing](#skip-mtp-avoid-redundant-draft-model-processing)
- [Draft Tokens Below p_min Are No Longer Dropped](#draft-tokens-below-p_min-are-no-longer-dropped)
- [NVFP4 Repack + MXFP6 (Experimental CUDA)](#nvfp4-repack--mxfp6-experimental-cuda)

---

## ngram-mod-v2: Persistent Keyed Hash Table

ngram-mod-v2 is an improved ngram-based speculative decoding implementation
that uses a **keyed hash table** to store observed token n-gram sequences.

### Key Improvements Over v1

| Feature | v1 (original) | v2 |
|---|---|---|
| False positives | Possible (hash collisions) | Eliminated (keyed hash with full entry verification) |
| Periodic reset | 25% of entries cleared every N steps | No reset -- grows until full |
| Persistence | Lost on restart | Save/load to file (`--spec-ngram-mod-file`) |
| Draft acceptor | Fixed-length acceptor | Adaptive length (tunes to recent hit rate) |

### Performance

- **~1.5x average speedup** on typical agentic coding workloads
- **Up to 5x speedup** during repetitive edit patterns (code refactoring, bug
  fixing, applying similar changes across files)
- Speedup improves over time as the n-gram cache accumulates patterns across
  sessions (when a persistent cache file is used)

### Recommended Parameters for Agentic Coding

```sh
--spec-type ngram-mod \
--spec-ngram-mod-v2 \
--spec-ngram-mod-n-match 12 \
--spec-ngram-mod-n-min 3 \
--spec-ngram-mod-n-max 108 \
--spec-ngram-mod-file /path/to/ngram-mod-cache.bin
```

Parameter explanation:

- `--spec-type ngram-mod` -- select the ngram-mod implementation
- `--spec-ngram-mod-v2` -- enable v2 (keyed hash table). Without this flag, v1
  (legacy) is used
- `--spec-ngram-mod-n-match 12` -- n-gram length used for lookup (shorter =
  more matches but lower quality, longer = fewer matches but higher quality)
- `--spec-ngram-mod-n-min 3` -- minimum draft length
- `--spec-ngram-mod-n-max 108` -- maximum draft length
- `--spec-ngram-mod-file <path>` -- persist the cache to disk so it survives
  restarts. On startup, the file is loaded; on shutdown, it is saved. Over time
  the cache accumulates patterns that match your editing style and project

### Combining with a Draft Model (MTP)

ngram-mod-v2 can be combined with MTP (Multi-Token Prediction) for even better
results. When both are specified, ngram-mod runs first as a higher-priority
implementation. See [Skip MTP](#skip-mtp-avoid-redundant-draft-model-processing)
below for how the two interact efficiently.

---

## Skip MTP: Avoid Redundant Draft Model Processing

### The Problem

When ngram-mod (or another high-priority draft implementation) handles all
drafting for a batch, MTP would still run its `process()` function to keep its
KV cache in sync with the target model. This consumed nearly all the time saved
by ngram-mod, negating the speedup.

### The Solution

A `skip_process` mechanism was added to the MTP implementation:

1. Before each draft round, MTP checks whether all sequences already have drafts
   from a higher-priority implementation.
2. If yes (`n_need_draft == 0`), MTP sets `skip_process = true` and returns
   without doing any work. Its `need_process()` returns `false`, so the main
   speculative loop skips `process()` entirely -- no KV cache sync, no decode.
3. On the next round where ngram-mod cannot provide drafts for all sequences,
   MTP detects `skip_process -> active` transition and sets `cold_start = true`.
   In `cold_start` mode, it produces at most 1 draft token per sequence to avoid
   verifying a long garbage sequence from stale KV cache state.
4. After the first successful draft, `cold_start` is cleared and normal MTP
   operation resumes.

This is not the most elegant solution, but it is effective: it allows the
combo of ngram-mod + MTP to deliver real speedups, whereas running both without
this change would waste most of the gain on redundant MTP cache computation.

---

## Draft Tokens Below p_min Are No Longer Dropped

### The Problem

When MTP samples draft tokens, it applies a `p_min` threshold
(`--spec-draft-p-min`). Tokens with confidence below this
threshold were previously **silently dropped** -- not added to the draft output
sequence -- even though the computation to produce them had already been spent.

### The Fix

The order of operations was changed so that the token is:

1. Accepted into the sampler (`common_sampler_accept`)
2. Pushed into the draft result (`result.push_back`)
3. Then checked against `p_min` to decide whether to continue drafting

This means every token that the draft model produced is submitted for
verification by the target model. If the target rejects it, the draft sequence
is truncated at that point -- no harm done. If the target accepts it, we got a
free token that would have been discarded.

Tested with `--spec-draft-p-min 0.8`; the fix is strictly faster than the
original behavior with no downside in output quality (the target model is the
final arbiter).

---

## NVFP4 Repack + MXFP6 (Experimental CUDA) -- Michael Wand

This feature is **not** authored by Niki. It was cherry-picked from
[Michael Wand's `nvfp4repack_mxfp6_cuda` branch](https://github.com/michaelw9999/llama.cpp/tree/nvfp4repack_mxfp6_cuda).

### What's Included

- **Improved NVFP4 repack**: tiles the repacked data for faster prompt
  processing and token generation; hoists per-channel post-scales for MMQ;
  reuses staging buffers for reduced memory allocation overhead
- **Experimental MXFP6 CUDA support**: CUDA kernels for MXFP6 (E2M3 format)
  on top of existing CPU support, including full-layer scale linking, direct
  FP8 usage for MMVQ paths

### Relevance

Useful for NVIDIA Blackwell GPUs (RTX 50 series) where NVFP4 is the native
format. The repack tiles increase throughput for both prefill and generation.
MXFP6 support is experimental and may be useful for quantization formats that
use 6-bit floating-point storage.

---

## Building

All features are compiled as part of the standard build. No special flags are
needed beyond those normally used for CUDA support:

```sh
cmake -B build -DGGML_CUDA=ON
cmake --build build --parallel
```

For the NVFP4/MXFP6 features specifically, ensure CUDA is enabled (they are
automatically included when `GGML_CUDA=ON`).

## Usage in llama-server

All `--spec-*` flags work with `llama-server` as well as `llama-speculative`
and `llama-cli`. Example server command with ngram-mod-v2 + MTP:

```sh
llama-server \
    -m /path/to/target-model-with-mtp-head.gguf \
    --spec-type ngram-mod,draft-mtp \
    --spec-draft-n-max 5 \
    --spec-draft-n-min 1 \
    --spec-draft-p-min 0.8 \
    --spec-ngram-mod-v2 \
    --spec-ngram-mod-n-match 12 \
    --spec-ngram-mod-n-min 3 \
    --spec-ngram-mod-n-max 108 \
    --spec-ngram-mod-file /path/to/cache.bin \
    -ngl 99
```
