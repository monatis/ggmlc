# AMD XDNA2 NPU (experimental)

`GGMLC_ENABLE_AMDXDNA=ON` adds `--device amdxdna` to the ggmlc runtime and
Laya CLI. It runs immutable model-weight matrix multiplies on the Ryzen AI
NPU using the bundled BF16 GEMM kernel originally supplied by FastFlowLM.
Other operators, including attention,
normalization, embedding lookup, and small decision projections, run on CPU.
An English Laya Q8_0 forward on Ryzen AI 9 HX370 offloads 122 matrix multiplies.

This is experimental: FastFlowLM's BF16 arithmetic changes model probabilities
and can change decisions near their thresholds. Validate your own inputs
against `--device cpu` before choosing this backend for a decision workload.
Kernel integration tests verify layout and execution; they do not establish
model accuracy or calibration. NPU use alone does not guarantee a speedup.

## Requirements and build

- Linux x86_64 and XDNA2: PCI `1022:17f0`, revision `10` (Strix Point, tested on
  Ryzen AI 9 HX370) or `11` (Strix Halo, not tested here).
- A loaded `amdxdna` driver and working firmware, exposing `/dev/accel/accel0`.
- XRT NPU runtime with the XDNA userspace plugin. `xrt-smi examine` must list
  device 0. The DKMS module by itself does not provide userspace compute.
- XRT development headers (`libxrt-dev`), `uuid-dev`, and `libdrm-dev`.
- The GEMM headers, `libgemm.so`, `libaiebu.a`, and matching XDNA2 `mm.xclbin`
  are included in `third_party/amdxdna`. A FastFlowLM program, installation,
  source checkout, or environment variable is not required for build or run.
- Sufficient memlock allowance for NPU buffers. Check `ulimit -l`; follow
  FastFlowLM's Linux setup instructions if XRT cannot pin the model buffers.

From the ggmlc checkout:

```bash
cmake -S . -B build/amdxdna \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGMLC_ENABLE_AMDXDNA=ON \
  -DGGMLC_BUILD_TESTS=ON \
  -DGGMLC_BUILD_EXAMPLE_TIMESFM=OFF \
  -DGGMLC_BUILD_EXAMPLE_TAB_COMPLETION=OFF
cmake --build build/amdxdna --target laya test-amdxdna test-executor-cpu -j8
ctest --test-dir build/amdxdna --output-on-failure
```

If XRT development files are extracted into a local prefix rather than
installed, pass `-DGGMLC_XRT_ROOT=/path/to/prefix` (containing `include/xrt`).
On the HX370 development machine the headers were downloaded and extracted
under `build/deps/xrt/usr`, without changing system packages. The existing
build uses `-DGGMLC_XRT_ROOT=/home/banpot/ggmlc/build/deps/xrt/usr` and the system
shared runtime `/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2`.

The option is off by default. CPU builds do not use the bundled NPU files or XRT.
Missing development files cause an actionable configure error. Explicit
`--device amdxdna` fails if support was not compiled or initialization fails;
it does not silently select CPU. `auto` prefers CUDA/Metal when present, then
an available XDNA2 device, then CPU. Laya's existing `auto` initialization
fallback still applies. Python uses the same native executor with
`ggmlc.load(path, device="amdxdna")` when its optional native extension is built.

## Run Laya GGUF

The existing GGUF is read directly. No checkpoint re-export or change to the
GGUF file is required.

```bash
build/amdxdna/examples/laya/laya decide laya_english_q8_0.gguf \
  --preset email --device amdxdna --threads 8 --json

build/amdxdna/examples/laya/laya serve laya_english_q8_0.gguf \
  --device amdxdna --threads 8 --port 8080

build/amdxdna/examples/laya/laya bench laya_english_q8_0.gguf \
  --preset email --device amdxdna --threads 8 --warmup 1 --runs 3
```

The first successful forward prints an `[amdxdna]` JSON summary to stderr.
`npu_matmuls` counts actual NPU operations in that forward; `cpu_nodes` counts
CPU graph nodes including metadata views; `cached_weight_bytes` reports the
BF16 NPU weight cache. `npu_ms` includes host conversion, packing, buffer
transfers and dispatch, and is not a hardware-only kernel timer. Set
`GGMLC_AMDXDNA_TRACE=1` to print it on every forward. JSON responses on stdout
remain parseable. Native/Python callers can inspect `amdxdna_summary()`.

## Install Laya with the NPU runtime assets

Use the `Laya` install component to install the executable, shared GEMM
library, matching NPU kernel, and original license notices together:

```bash
cmake -S . -B build/amdxdna -DGGMLC_ENABLE_AMDXDNA=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/laya
cmake --build build/amdxdna --target laya -j8
sudo cmake --install build/amdxdna --prefix /opt/laya --component Laya

/opt/laya/bin/laya help
ldd /opt/laya/bin/laya
/opt/laya/bin/laya decide /path/to/laya_english_q8_0.gguf \
  --preset email --device amdxdna --threads 8 --json
```

With the default install directories the installed files are:

```text
/opt/laya/bin/laya
/opt/laya/lib/libgemm.so
/opt/laya/share/laya/amdxdna/mm.xclbin
/opt/laya/share/laya/amdxdna/licenses/
```

The executable's install RPATH finds `libgemm.so` relative to its location.
The backend finds the installed kernel relative to `/proc/self/exe`, so the
working directory does not matter. This also supports a different
`cmake --install --prefix` or moving the complete installed directory tree.
Copying only `laya` omits required runtime assets; use the installer or copy
the complete tree. No `LD_LIBRARY_PATH`, `ldconfig`, or kernel-path environment
variable is needed for this layout. An explicit `GGMLC_AMDXDNA_XCLBIN` still
overrides kernel lookup. GGUF files are supplied separately.

The installer includes the application-specific GEMM library. XRT, its XDNA
plugin, and amdxdna/firmware remain system dependencies, as described above.
The system XRT library directory is retained in the installed RPATH for
installations using an XRT prefix outside the system loader's default paths.

## Create a local release archive

The `laya-package` target builds Laya, stages its install component, and
creates a `.tar.gz` with `bin`, `lib`, `share`, a generated README,
`BUILD-INFO.txt`, and per-file SHA-256 checksums. It also writes an archive
checksum beside the `.tar.gz`. It performs no GitHub operations or commits.

```bash
cmake -S . -B build/laya-release-amdxdna \
  -DCMAKE_BUILD_TYPE=Release -DGGMLC_ENABLE_AMDXDNA=ON -DGGML_NATIVE=OFF \
  -DGGMLC_BUILD_EXAMPLE_TIMESFM=OFF -DGGMLC_BUILD_EXAMPLE_TAB_COMPLETION=OFF \
  -DGGMLC_LAYA_PACKAGE_NAME=laya-linux-x86_64-amdxdna-ubuntu26.04 \
  -DGGMLC_LAYA_PACKAGE_OUTPUT_DIR="$PWD/dist"
cmake --build build/laya-release-amdxdna --target laya-package -j8
cd dist
sha256sum -c laya-linux-x86_64-amdxdna-ubuntu26.04.tar.gz.sha256
```

Without overrides, the archive name is `laya-linux-x86_64-amdxdna.tar.gz`
and the output directory is `<build>/dist`. Package names must be filenames,
and install directories must be relative to the prefix. There is no sudo
step when packaging locally. The entire extracted directory can be moved.

The package includes application runtime assets and licenses. The target
machine supplies XRT, the XDNA plugin, amdxdna/firmware, glibc and libstdc++.
The package records minimum GLIBC/GLIBCXX/CXXABI symbol versions from its ELF
files, so a binary built on a newer Ubuntu is not labeled compatible with
older releases. GGUF model files are downloaded separately.

F32/F16/BF16 and supported GGML quantized weights (including Q8_0 and Q4_K)
are decoded once and cached as packed BF16. For English Q8_0 the offloaded
weight cache is 765,198,336 bytes, in addition to the original host weights,
compute buffers and XRT allocations. Loading a smaller GGUF therefore does
not imply an equally small NPU cache. First-run latency includes packing and
instruction/module creation; benchmark subsequent runs separately.

## Validate numerical drift

This script requires only Python's standard library and runs real CPU/NPU
inference, asserting that the NPU actually received work:

```bash
python3 examples/laya/validate_amdxdna.py \
  --binary build/amdxdna/examples/laya/laya \
  --model laya_english_q8_0.gguf \
  --report build/amdxdna/laya-validation.json
```

It saves both sets of answers, probability/confidence differences and changed
choice/yes-no decisions. A successful script run confirms execution and
produces a report; it does not certify the reported differences as acceptable.
Four initial English Q8_0 examples on HX370 included one changed yes/no answer
(`sensitive_data`, 0.4724 on CPU versus 0.5523 on NPU). Keeping CPU execution
available is necessary for workloads that require the original calibration.

On 2026-10-03, the 7-question email preset (B=7, S=124, eight CPU threads,
one warmup and three timed runs) averaged 1000.0 ms on CPU and 762.6 ms with
amdxdna (about 1.31x throughput). These are local measurements, not an accuracy
equivalence claim; first-run costs were excluded. The four-example precision
report observed maximum probability differences from 0.0517 to 0.2492.
Both native CTest suites passed on the host. The optional Python extension
was not built in this environment because nanobind was unavailable.
The referenced source checkouts were FastFlowLM
`6c9ed874908e4b793014a87bce4b43af644ef0db` and amdxdna-dkms
`bc8437aa9eb3dbf71cfe16de99e6a6aa6e9e7960`.

`test-amdxdna` covers padding in M/K/N, multi-plane batches, changing shapes,
weight-cache reuse, CPU graph segments, and F32/F16/BF16/Q8_0/Q4_K weights.
It uses dyadic operands to isolate layout from the vendor kernel's rounding;
quantized CPU dot products also introduce activation quantization. Without an
accessible NPU the test returns CTest's skip code 77. Run it on the host with
device access; a sandbox that hides `/dev/accel` will skip it.

## How it reaches amdxdna

The implementation follows FastFlowLM's `modules/gemm.hpp`,
`npu_utils/npu_utils_xrt.hpp`, and `buffer.hpp`:

1. Open `xrt::device(0)`, register the matching `mm.xclbin`, and create a shared
   hardware context with `npu_xclbin_manager`.
2. Generate BF16 GEMM control sequences with `Gemm::generate_seq`. FastFlowLM
   uses aiebu to assemble the transaction into an ELF module and XRT kernel.
3. Allocate XRT BOs, synchronize packed input/weights to the device, dispatch
   the kernel, require `ERT_CMD_STATE_COMPLETED`, and synchronize output back.
4. Execute consecutive CPU graph ranges between offloaded GEMMs. All GGML
   tensors remain in host buffers; output is expanded from BF16 to F32 before
   dependent CPU operators execute.

The driver UAPI in `amdxdna-dkms/src/include/uapi/drm/amdxdna_accel.h` provides
hardware contexts, BO allocation/synchronization and command submission
through `DRM_IOCTL_AMDXDNA_CREATE_HWCTX`, `CREATE_BO`, `SYNC_BO` and `EXEC_CMD`.
XRT's XDNA plugin handles those ioctls and firmware interactions. This backend
does not link or modify the DKMS module, copy kernel-driver code, install a
driver, or construct firmware commands directly.

GEMM dimensions are padded to M=256, K=512, N=128 boundaries. Weight tiles use
`[N/128, K/512, N_tile/8, K_inner(8), K_tile/8, N_inner(8)]`; activations and
results use row-major layout. The layout was verified with impulse matrices
on the NPU and multi-tile numerical comparisons. Only immutable, host-backed,
row-contiguous 2D weights with K>=512 and N>=128 are offloaded. Mutable,
batched-weight, unsupported, or small GEMMs remain on CPU.
An operator that explicitly requests `GGML_PREC_F32` also remains on CPU.

Control sequences and packed weights are cached per executor. A kernel error
throws instead of substituting an unnoticed CPU result. The integration is
synchronous and uses XRT device 0; CUDA graphs and paged CUDA KV cache do not
apply to it. This implementation uses the XRT variant of FastFlowLM, not HRX
or ryzenai-corelib.

## Bundled files and runtime assets

`third_party/amdxdna` contains the 67 transitive headers needed by the adapter
and three matching binary artifacts. Upstream paths, the pinned commit, and
SHA-256 hashes are recorded in `PROVENANCE.json`; original headers are copied
unchanged. See [the bundle's README](../../third_party/amdxdna/README.md) for
the file inventory and license notices. No FastFlowLM CLI or model runner is
built or invoked.

CMake copies `libgemm.so` into `<build>/runtime/amdxdna/lib` and `mm.xclbin`
into `<build>/runtime/amdxdna/assets`. Build-time headers and the static aiebu
archive are resolved within ggmlc. Executables run directly from the build
tree use these build assets. Installed Laya uses the relative library search
path and kernel lookup described above, independently of the build tree.

`GGMLC_AMDXDNA_XCLBIN` can also select a different kernel at configure time;
it must match the bundled GEMM library. Existing build trees using the former
FastFlowLM default migrate to the bundled kernel automatically. The former
`GGMLC_FASTFLOWLM_ROOT` and `GGMLC_FLM_*` cache entries are retired.

A fresh `build/amdxdna-standalone` build on HX370 was verified on 2026-10-03
with no FastFlowLM paths in its compiler dependencies or runtime search path.
Both native tests passed with real device access and without `FASTFLOWLM_ROOT`
or `LD_LIBRARY_PATH`; English Q8_0 inference again executed 122 NPU GEMMs.
The existing `build/amdxdna` cache and binaries were also migrated to the bundle.

Including these files does not change their licenses. FastFlowLM's utility
source is MIT, with separate notices for included third-party headers. Its
precompiled NPU components retain the original
[binary terms included in the bundle](../../third_party/amdxdna/licenses/FastFlowLM-TERMS.md).
The backend is independent of an external FastFlowLM checkout, but its GEMM
implementation and kernel are still the bundled upstream binaries.
