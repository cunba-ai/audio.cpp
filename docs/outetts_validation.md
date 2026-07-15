# OuteTTS validation

This procedure exercises OuteTTS 1.0 1B in one long-lived audio.cpp session.
It covers normal TTS, framework long-form text chunking, voice cloning, repeated
reference-profile cache hits, cached-step graph reuse, stage timings, and
default-versus-`mem_saver` memory behavior.

## Model setup

Install the safetensors model, IBM DAC, and Qwen3 Forced Aligner resources:

```bash
python tools/model_manager.py install outetts_1_0_1b --models-dir models
```

The standalone GGUF command is documented in [TTS](tts.md#outetts). The packed
file used below contains the OuteTTS language model, IBM DAC, Qwen aligner,
tokenizers, configuration, and package specification.

## Python reference setup

The maintainer can install the official OuteTTS reference without using any
audio.cpp conversion code:

```bash
python -m venv .venv-outetts
. .venv-outetts/bin/activate
python -m pip install --upgrade pip outetts
```

On Windows PowerShell with the optional CUDA llama.cpp backend:

```powershell
python -m venv .venv-outetts
.\.venv-outetts\Scripts\Activate.ps1
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
python -m pip install --upgrade pip outetts
```

Select `outetts.Backend.HF` for a Transformers comparison or
`outetts.Backend.LLAMACPP` for the official llama.cpp-backed route. Use
`outetts.Models.VERSION_1_0_SIZE_1B`, temperature `0.4`, repetition penalty
`1.1` over the latest 64 tokens, top-k `40`, top-p `0.9`, and min-p
`0.05`.

## Build

Windows CUDA:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target outetts_warm_bench -Jobs 4
```

Linux CPU:

```bash
cmake -S . -B build/cpu-release -DCMAKE_BUILD_TYPE=Release -DENGINE_BUILD_WARMBENCH=ON
cmake --build build/cpu-release --target outetts_warm_bench -j
```

## Long-lived session

```powershell
build\windows-cuda-release\bin\outetts_warm_bench.exe `
  --model models\Llama-OuteTTS-1.0-1B-Q8_0\model.gguf `
  --backend cuda --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --hold-seconds 5 `
  --audio-out-dir build\logs\warmbench\outetts-default `
  --log-file build\logs\warmbench\outetts-default.log
```

Expected trace evidence:

- `outetts.text_chunk_count` is greater than one for `tts_longform`.
- `outetts.llama.step.graph_reused=1` appears after the first compatible
  generation request or chunk.
- the second identical reference reports `outetts.reference_cache.hit=1`.
- `tts_cold` and `tts_repeat` are byte-identical, as are `clone_cold`
  and `clone_repeat`; this verifies that warm graph/profile reuse does not
  change deterministic output.
- `outetts.aligner.runtime_reused=1` is observable for uncached references
  while the default session retains the aligner.
- only one active OuteTTS Llama runtime is retained; switching between native
  TTS weights and the CUDA F32 cloning fallback replaces the previous runtime.

Run the same request sequence in a fresh process with memory saver enabled:

```powershell
build\windows-cuda-release\bin\outetts_warm_bench.exe `
  --model models\Llama-OuteTTS-1.0-1B-Q8_0\model.gguf `
  --backend cuda --threads 8 `
  --request-file tests\outetts\warm_bench_requests.json `
  --session-option outetts.mem_saver=true `
  --hold-seconds 5 `
  --audio-out-dir build\logs\warmbench\outetts-mem-saver `
  --log-file build\logs\warmbench\outetts-mem-saver.log
```

The memory-saver trace reports a positive
`outetts.llama.step.released_cache_capacity` after generation and
`outetts.aligner.runtime_released=1` after an uncached reference.

Sample per-process VRAM once per second while each fresh benchmark runs:

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv -l 1
```

Compare peak and final resident VRAM, request wall time, output duration, and RTF
between the two runs. Keep model, backend, device, seed, requests, and
quantization identical.

## Measured validation

The committed five-request sequence was measured with the packed Q8 GGUF on an
NVIDIA GeForce RTX 3090 (CUDA 12.4). VRAM was sampled every 250 ms from total
device usage with no other CUDA workload. Resident VRAM was sampled during the
five-second hold after all requests completed.

| Mode | Sequence wall | Audio | RTF | Peak VRAM | Resident VRAM |
|---|---:|---:|---:|---:|---:|
| default | 33342.24 ms | 11.424s | 2.918 | 17653 MiB | 294 MiB |
| `outetts.mem_saver=true` | 30990.93 ms | 11.424s | 2.712 | 5780 MiB | 294 MiB |

Both modes produced the same SHA-256 for the cold and repeated TTS pair and for
the cold and repeated clone pair. All ten generated WAV files passed an ffmpeg
decode check. The traces confirmed four framework chunks for the long-form
request, compatible step-graph reuse in default mode, explicit graph release in
memory-saver mode, and a reference-profile cache hit on the repeated clone.

Generation is the dominant measured hot path. In the default CUDA run, the cold
reference path took about 890 ms (109 ms alignment, 139 ms DAC encoding, and
214 ms profile construction), while the repeated cached reference took 0.29 ms.
The memory-saver result is within normal run-to-run timing variation; its
purpose here is the lower peak, not a speedup guarantee.
