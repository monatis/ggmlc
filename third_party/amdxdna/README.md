# Bundled AMD XDNA2 GEMM dependencies

This is the dependency subset used by `runtime/amdxdna`. Builds and inference
do not require a FastFlowLM checkout, installation, executable, or model
format. System XRT development files/runtime, the XDNA userspace plugin, and
the installed amdxdna driver are still required.

The files were copied unchanged from
[ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM/tree/6c9ed874908e4b793014a87bce4b43af644ef0db),
commit `6c9ed874908e4b793014a87bce4b43af644ef0db`:

| Files | Purpose |
| --- | --- |
| `include/modules/gemm.hpp` and its transitive headers | GEMM API, instruction generation, XRT buffers and context management |
| `lib/libgemm.so` | Precompiled Linux x86_64 GEMM sequence generator |
| `lib/libaiebu.a` | Matching precompiled transaction-to-ELF assembler |
| `xclbins/mm.xclbin` | XDNA2 BF16 GEMM device program, from `src/xclbins/Qwen3-0.6B-NPU2/mm.xclbin` |

`PROVENANCE.json` records each upstream file path and SHA-256 hash. There are
67 headers and approximately 7.4 MB of bundled files. The bundle deliberately
excludes the FastFlowLM executable, model runners, model data, HRX runtime,
other device kernels, and kernel-driver implementation.

CMake copies the shared GEMM library and device program into the build tree.
The aiebu archive is linked statically. The shared GEMM library and device
program must be kept together as a compatible set when updating the bundle;
updates require rerunning native NPU tests and Laya CPU/NPU comparisons.

## Notices

These files retain their original licenses; ggmlc's project license does not
relicense them. Original notices inside the headers are preserved.

- FastFlowLM utility sources: [MIT](licenses/FastFlowLM-MIT.txt), copyright
  2026 Advanced Micro Devices, Inc.
- Precompiled artifacts from FastFlowLM: original
  [terms for proprietary binaries](licenses/FastFlowLM-TERMS.md), copied from
  that same commit. The upstream GEMM/kernel source is not included in this
  bundle; they are not built from source by ggmlc.
- `include/aiebu/aiebu.h`: MIT, copyright 2024–2025 Advanced Micro Devices,
  Inc. See [third-party MIT notices](licenses/ThirdParty-MIT.txt).
- `include/nlohmann/**`: MIT, copyright 2013–2025 Niels Lohmann and, for its
  Hedley headers, 2016–2021 Evan Nemerson. See
  [third-party MIT notices](licenses/ThirdParty-MIT.txt).
- `include/biovault_bfloat16.h`: [Apache-2.0](licenses/Apache-2.0.txt),
  copyright 2020 LKEB, Leiden University Medical Center; adapted from oneDNN.
  Its attribution and original source URL remain in the header.
- `include/npu_utils/amdxdna_accel.h`: GPL-2.0 WITH Linux-syscall-note,
  copyright 2022–2024 Advanced Micro Devices, Inc. This is a userspace UAPI
  header, not a copy of the driver implementation. See
  [GPL-2.0](licenses/GPL-2.txt) and the
  [Linux syscall exception](licenses/Linux-syscall-note.txt), copied from the
  [Linux license directory](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/LICENSES/exceptions/Linux-syscall-note).
