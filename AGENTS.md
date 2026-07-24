# AGENTS.md

A map for working in this repo. Read this first; for model-port and PR
conventions see [CONTRIBUTING.md](CONTRIBUTING.md) (it has the detail).

## Project overview

`audio.cpp` is a high-performance C++ audio inference framework built on top of
`ggml`. It runs many model families — TTS, voice cloning/conversion, ASR,
diarization, VAD, source separation, alignment, codecs, and music generation —
through one shared native runtime instead of per-model Python stacks. Primary
languages: **C/C++** (engine, models, apps) and **Python** (tooling, parity
references, the model-manager WebUI). JavaScript appears only minimally. The
core compute backends are `ggml`'s CPU, CUDA, Vulkan, and Metal.

## Directory map

| Path | Purpose |
|---|---|
| `src/framework/` | The engine library `engine_runtime`: `core/`, `runtime/`, `audio/`, `modules/`, `codecs/`, `decoders/`, `sampling/`, `tokenizers/`, `text/`, `io/`, `assets/`, `debug/` |
| `src/models/` | Per-family model implementations (~33 families: TTS, ASR, codec, music, VAD, diarization, VC…) |
| `src/community_models/` | Community-contributed model families (`outetts`, `vietneu_tts`, shared `common`) |
| `app/cli/` | `audiocpp_cli` — args, batch, request handling |
| `app/server/` | `audiocpp_server` — HTTP server, multipart uploads, config, runtime, `busy_guard` |
| `app/gguf/` | `audiocpp_gguf` — platform-neutral GGUF load/convert/run binary |
| `app/streaming/`, `app/workflow/` | Streaming session + experimental JSON pipeline/workflow engine (linked into the CLI) |
| `capi/` | C ABI shared library `libaudiocpp` (`include/audiocpp.h`, `src/audiocpp_capi.cpp`, `README.md`) — gated by `AUDIOCPP_BUILD_CAPI`. Exposes 37 functions: model lifecycle, TTS/ASR/VAD/Diar/Align/SEP/VC, streaming (chunk-push), device enumeration, model inspection, WAV I/O, artifact types. CI builds 6 platform/backend variants and uploads to OSS. |
| `include/engine/` | Public headers mirrored by subsystem (`framework/`, `models/`, `community_models/`) |
| `tests/` | Per-family test dirs + `tests/unittests/` (registered via `add_engine_unittest`), `tests/core/`, `tests/perf/`, parity/conversion Python scripts |
| `tools/` | Build/packaging + model tooling, incl. `check_loader_catalog_sync.py` |
| `webui/` | Python model-manager WebUI (`model_manager_webui.py`), realtime server/pipeline |
| `external/` | Vendored deps: `ggml`, `llama_tokenizer`, `sentencepiece`, `libyaml`, `cJSON` |
| `model_specs/` | Model spec files consumed by package/tooling paths |
| `docs/` | Per-task docs (`tts.md`, `asr.md`, `gguf.md`, …), `maintainers/`, `models/`, `reports/`, `proposals/` |
| `scripts/` | Platform build helpers (`build_windows.ps1`, `build_linux.sh`, `build_metal.sh`, `build_xcframework.sh`, packaging) |
| `.devops/`, `.github/workflows/` | Devops assets and CI (Windows, Linux cpu/vulkan, macOS, Docker, C-API build) |

## Common commands

All builds are CMake-driven; platform helpers wrap it. Key CMake options:
`ENGINE_ENABLE_{CUDA,VULKAN,METAL,LLAMAFILE,CUDA_GRAPHS,NATIVE_CPU,OPENMP}`,
`ENGINE_BUILD_{EXAMPLES,TESTS,WARMBENCH}`, `AUDIOCPP_BUILD_CAPI`,
`AUDIOCPP_DEPLOYMENT_BUILD`.

```bash
# Linux (CPU / Vulkan). CI uses gcc-13. Adjust ENGINE_ENABLE_VULKAN.
cmake -S . -B build/ci-linux -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_VULKAN=OFF
cmake --build build/ci-linux --target audiocpp_cli -j
# Server: --target audiocpp_server   GGUF tool: --target audiocpp_gguf

# Windows (PowerShell, presets)
.\scripts\build_windows.ps1 -Preset windows-cpu-release -Target audiocpp_cli

# macOS (Metal)
sh scripts/build_metal.sh

# Build + run the framework unit tests (ENGINE_BUILD_TESTS=ON)
cmake -S . -B build -DENGINE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
# Individual unit tests live in tests/unittests/ (test_*.cpp), run directly or via ctest.

# Loader/package-catalog sync check (run before model/registry changes; enforced in CI)
python3 tools/check_loader_catalog_sync.py --self-test
python3 tools/check_loader_catalog_sync.py

# Nix dev shell (optional)
nix develop
```

There is **no project-level lint/format config** (no `.clang-format`,
`.clang-tidy`, ruff, etc.). Match the style of surrounding files.

## Architecture notes

- **Entry points** — `audiocpp_cli` (`app/cli/main.cpp`), `audiocpp_server`
  (`app/server/main.cpp`), `audiocpp_gguf` (`app/gguf/main.cpp`). All link the
  static `engine_runtime` + `ggml`.
- **`engine_runtime` (lib)** is the heart — `src/framework/`. `runtime/` owns
  the registry, model/session, graph executor + optimizer, KV cache, options;
  `core/` has backend, execution context, module base; `modules/` holds the
  reusable neural building blocks (attention, conv, conformer, vocoders,
  encoders, norms…).
- **Loader registry** — `ModelRegistry::register_loader` in
  `src/framework/runtime/registry.cpp`. Each model family registers a loader;
  installable `ModelPackage` entries must stay in sync with
  `audiocpp_cli --list-loaders` (enforced by the sync check above).
- **Compute backends** come from vendored `external/ggml` (CPU/CUDA/Vulkan/
  Metal). Backend selection is a CMake option, not a runtime flag.
- **Server** — OpenAI-compatible HTTP surface in `app/server/` with a
  `busy_guard` that serializes requests so a stuck model can't hang later ones,
  multipart upload support, and `SIGPIPE` ignoring for client disconnects.
- **Cross-cutting**: tensor sources (`assets/` — safetensors, GGUF, torch
  bins), JSON/YAML/config IO (`io/`), tokenizers (`tokenizers/`, sentencepiece +
  llama.cpp tokenizer), audio DSP utilities (`audio/` — STFT/ISTFT, FFT,
  resampling, denoise, enhancement). Parity tooling exists per family to
  validate against Python reference paths.

## Conventions

Follow the team standards in the `miaosoft-zcode-plugin` skills — don't restate
them here:

- **`dev-workflow`** — code style, naming, error handling.
- **`testing-standards`** — test structure and coverage expectations.
- **`pr-review`** — PR template and review checklist.

Repo-specific addenda: keep loader registrations and the package catalog in sync
(run `check_loader_catalog_sync.py`); prefer additive experimental modules
(e.g. `xxxExp`) over rewriting shared framework modules existing models depend
on (see CONTRIBUTING.md "Framework Modules (High Risk)"); for new model PRs
include exact build/run commands, model paths/ids, generated outputs, and
parity/path-test results.

## Code exploration

symgraph is wired for this repo (`.zcode/config.json`). After big changes,
refresh the graph:

```bash
symgraph index   # rebuilds .git/symgraph/index.db (not tracked)
```

Prefer symgraph MCP tools over grepping for structural queries:
`mcp__symgraph__callers`, `callees`, `references`, `module-graph`, `impact`,
`unused`. The graph lives at `.git/symgraph/index.db`.

> If `mcp__symgraph__*` tools aren't loaded this session, restart ZCode (or
> reload MCP in Settings → MCP) so the server connects.
