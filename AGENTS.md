# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Test Commands

- **CPU build**: `cmake -B build`, `cmake --build build --config Release`
- **Debug build (single-config)**: `cmake -B build -DCMAKE_BUILD_TYPE=Debug`, `cmake --build build`
- **Debug build (multi-config)**: `cmake -B build -G "Xcode"`, `cmake --build build --config Debug`
- **Static build**: add `-DBUILD_SHARED_LIBS=OFF` to cmake command
- **Backend ops test** (verify different backend implementations produce consistent results): `./build/bin/test-backend-ops -b <backend> -o <operator>` or `./build/bin/test-backend-ops -b <backend>`
- **Perplexity verification**: `llama-perplexity -m model.gguf -f file.txt`
- **Performance benchmark**: `llama-bench -m model.gguf`
- **Local CI execution**: `bash ./ci/run.sh ./tmp/results ./tmp/mnt` (use `GG_BUILD_CUDA=1`, `GG_BUILD_SYCL=1`, `GG_BUILD_MUSA=1`, etc. for specific backends)

## Code Style Guidelines

### C/C++ Naming Conventions
- Use `snake_case` for function, variable and type names.
- Naming optimizes for longest common prefix: `int number_small; int number_big;` (not `small_number`, `big_number`).
- Enum values are always in upper case and prefixed with the enum name: `LLAMA_VOCAB_TYPE_NONE = 0`.
- General naming pattern: `<class>_<method>`, with `<method>` being `<action>_<noun>` (e.g., `llama_model_init`, `llama_sampler_chain_remove`, `llama_get_seed`).
- Use the `_t` suffix when a type is supposed to be opaque to the user: `llama_context_t`.
- In C++ code omit optional `struct` and `enum` keywords: `llama_context * ctx;` (not `struct llama_context * ctx;`).
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`.

### C/C++ Formatting
- Use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`.
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates.
- Preprocessor directives format:
  ```cpp
  #ifdef FOO
  #endif // FOO
  ```

### Python
- Filenames are all lowercase with underscores.
- `.flake8` config: `max-line-length = 125`, `ignore = E203,E211,E221,E225,E231,E241,E251,E261,E266,E501,E701,E704,W503`.
- `pyrightconfig.json`: `extraPaths` include `["gguf-py", "examples/model-conversion/scripts", "examples/model-conversion/scripts/utils"]`, `pythonVersion: "3.9"`.

## Critical Gotchas

- **Tensor dimensions**: Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices.
- **Matrix multiplication**: `C = ggml_mul_mat(ctx, A, B)` means $C^T = A B^T \Leftrightarrow C = B A^T.$
- **Backend ops testing**: If you modified a `ggml` operator or added a new one, add the corresponding test cases to `test-backend-ops`. This tool requires access to at least two different `ggml` backends to verify consistent results.
