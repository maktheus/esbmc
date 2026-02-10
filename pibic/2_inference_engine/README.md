# Case 2: Verifying Inference Engine Kernels

This example demonstrates how to use ESBMC to verify C++ kernels used in AI inference engines (like `llama.cpp` or custom CUDA implementations). We verify memory safety properties such as buffer overflows and valid pointer usage.

## Files

- `attention_kernel.cpp`: A simplified C++ implementation of a dot-product attention calculation using dynamic memory allocation.

## How to Run

To verify the kernel for memory leaks and bounds checking:

```bash
esbmc attention_kernel.cpp --multi-property --memory-leak-check --overflow-check
```

## Expected Result

ESBMC should report `VERIFICATION SUCCESSFUL` if the constraints (`seq_len <= 128`) match the allocation.

Try modifying `__ESBMC_assume(seq_len <= 128);` to `__ESBMC_assume(seq_len <= 200);` while keeping allocation at 128 to see it detect a buffer overflow!
