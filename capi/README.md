# audio.cpp C ABI (`libaudiocpp`)

A minimal C interface exposing audio.cpp's model loading, inference, and
streaming capabilities to foreign-language runtimes (Rust FFI, Python
ctypes, Go cgo, etc.).

## Quick Start

### Build

The C ABI shared library is built when `AUDIOCPP_BUILD_CAPI=ON` (default).
CI produces prebuilt binaries for 6 platform/backend combinations:

| Artifact | Platform | Backend |
|---|---|---|
| `audiocpp-windows-cuda-cpu.zip` | Windows | CUDA + CPU fallback |
| `audiocpp-windows-rocm-cpu.zip` | Windows | ROCm (HIP) |
| `audiocpp-windows-sycl-cpu.zip` | Windows | Intel SYCL |
| `audiocpp-linux-cuda-cpu.zip` | Linux | CUDA + CPU fallback |
| `audiocpp-linux-rocm-cpu.zip` | Linux | ROCm (HIP) |
| `audiocpp-linux-sycl-cpu.zip` | Linux | Intel SYCL |

Each zip contains:
- `audiocpp.dll` / `libaudiocpp.so` — the shared library
- `audiocpp.lib` — Windows import library (not needed for runtime loading)
- `audiocpp.h` — the C header

### Minimal Usage (Rust via libloading)

```rust
use libloading::Library;

let lib = Library::new("audiocpp.dll")?;

// Enumerate devices → pick GPU
let device_count: Symbol<unsafe extern "C" fn() -> i32> =
    unsafe { lib.get(b"audiocpp_device_count") }?;

// Load model
let model = load_model(path, "qwen3_asr", TASK_ASR, BACKEND_CUDA, 0, 4, &mut err);

// Run ASR
let text = asr(model, pcm, n_samples, 16000, "en", &mut err);
println!("{}", text.text);
```

---

## API Reference

The library exports **37 functions** across these categories:

### 1. Backend & Device Selection

#### Enums

```c
// Backend type
AUDIOCPP_BACKEND_CPU    = 0
AUDIOCPP_BACKEND_CUDA   = 1   // also covers AMD ROCm (HIP)
AUDIOCPP_BACKEND_VULKAN = 2
AUDIOCPP_BACKEND_METAL  = 3
AUDIOCPP_BACKEND_SYCL   = 4   // Intel oneAPI
AUDIOCPP_BACKEND_BEST   = 5   // auto-select

// Device type
AUDIOCPP_DEVICE_CPU  = 0
AUDIOCPP_DEVICE_GPU  = 1
AUDIOCPP_DEVICE_IGPU = 2
```

#### Functions

| Function | Description |
|---|---|
| `audiocpp_device_count()` | Count all compute devices across compiled backends |
| `audiocpp_device_info(index, *out)` | Get device name, backend, memory, per-backend device_id |
| `audiocpp_list_devices()` | Print all devices to stdout (convenience) |

**Device selection flow**: enumerate → find GPU → pass `backend` + `device_id`
to `audiocpp_load_model`. The `device_id` from `audiocpp_device_info` is
per-backend (e.g. `device_id=0` for the first CUDA GPU, independent from
the first SYCL GPU).

---

### 2. Model Lifecycle

```c
audiocpp_model_t *audiocpp_load_model(
    const char *model_path,       // model directory or GGUF file
    const char *family_hint,      // e.g. "qwen3_asr", NULL = auto-detect
    int task,                     // AUDIOCPP_TASK_*
    int backend,                  // AUDIOCPP_BACKEND_*
    int device_id,                // GPU index (from device enumeration)
    int n_threads,                // CPU threads (0 = auto)
    audiocpp_error_t *err         // optional error output
);
void audiocpp_free_model(audiocpp_model_t *model);
```

**Important**: Always pass `family_hint` when known. Without it, the loader
iterates all 35 loaders' `can_load` probes, which is slower and may hit
side effects from incompatible models probing the same directory.

---

### 3. Task Types

All 13 engine task kinds are exposed:

```c
AUDIOCPP_TASK_VAD   = 0    // Voice Activity Detection
AUDIOCPP_TASK_ASR   = 1    // Speech-to-Text
AUDIOCPP_TASK_DIAR  = 2    // Speaker Diarization
AUDIOCPP_TASK_SEP   = 3    // Source Separation
AUDIOCPP_TASK_GEN   = 4    // Audio/Music Generation
AUDIOCPP_TASK_TTS   = 5    // Text-to-Speech
AUDIOCPP_TASK_ALIGN = 6    // Forced Alignment
AUDIOCPP_TASK_VC    = 7    // Voice Conversion
AUDIOCPP_TASK_CLON  = 8    // Voice Cloning (TTS + speaker reference)
AUDIOCPP_TASK_S2S   = 9    // Speech-to-Speech (codec-based)
AUDIOCPP_TASK_VDES  = 10   // Voice Design (prompt-based)
AUDIOCPP_TASK_SPK   = 11   // Speaker Recognition
AUDIOCPP_TASK_SVC   = 12   // Singing Voice Conversion
```

---

### 4. Inference Functions (Offline)

Each returns a result handle (or NULL on error). The caller owns the result
and must free it with the matching `audiocpp_free_*` function.

| Function | Task | Input | Output | Free with |
|---|---|---|---|---|
| `audiocpp_tts` | TTS | text | audio | `free_audio` |
| `audiocpp_tts_with_voice_ref` | TTS/Clone | text + inline PCM voice ref | audio | `free_audio` |
| `audiocpp_asr` | ASR | audio PCM | text + language | `free_text` |
| `audiocpp_diar` | Diar | audio PCM | speaker turns | `free_diar` |
| `audiocpp_vad` | VAD | audio PCM | speech segments | `free_vad` |
| `audiocpp_align` | Align | audio PCM + text + language | word timestamps | `free_align` |
| `audiocpp_audio_transform` | SEP/VC | audio PCM | single audio output | `free_audio` |
| `audiocpp_audio_transform_with_voice_ref` | VC | audio PCM + inline voice ref | audio | `free_audio` |
| `audiocpp_transform_stems` | SEP/GEN | audio PCM (+ voice ref) | **all** named stems (vocals/drums/bass/...) | `free_stems` |

**`audiocpp_transform_stems`** returns all named audio outputs (unlike
`audiocpp_audio_transform` which only returns the first). Use this for
source separation models (demucs, roformer) that emit multiple stems.

---

### 5. Streaming (Chunk-Push Model)

For real-time / low-latency processing (streaming VAD, streaming ASR):

```c
// 1. Start stream (creates a new streaming session)
audiocpp_stream_t *stream = audiocpp_stream_start(
    model, TASK_VAD, NULL, 512, &err);

// 2. Push audio chunks → get events synchronously
while (have_audio) {
    audiocpp_stream_event_t *ev = audiocpp_stream_push(
        stream, pcm_chunk, chunk_len, 16000, &err);
    // ev->va_events[]      → VAD events (speech start/end)
    // ev->partial_text     → ASR partial transcript
    // ev->audio_samples    → streaming TTS output
    audiocpp_free_stream_event(ev);
}

// 3. Finish → get final result
audiocpp_text_t final;
audiocpp_stream_finish(stream, &final, &err);
audiocpp_free_text(&final);
audiocpp_stream_free(stream);
```

| Function | Description |
|---|---|
| `audiocpp_stream_start` | Create streaming session (RunMode::Streaming) |
| `audiocpp_stream_push` | Push audio chunk, get event (VAD/ASR/TTS) |
| `audiocpp_stream_finish` | End stream, get final accumulated result |
| `audiocpp_free_stream_event` | Free event returned by push |
| `audiocpp_stream_free` | Free stream handle |

**Two streaming patterns supported**:
- **VAD (silero)**: each chunk returns voice-activity events immediately
- **ASR (nemotron)**: chunks accumulate; final decode happens in `finish`

---

### 6. Model Inspection

Query a loaded model's metadata and capabilities before running:

```c
audiocpp_model_info_t info;
audiocpp_model_info(model, &info);
// info.family, info.variant, info.description

audiocpp_model_capabilities_t caps;
audiocpp_model_capabilities(model, &caps);
// caps.supported_tasks[], caps.languages[]
// caps.supports_speaker_reference, supports_style_condition, supports_timestamps
```

| Function | Description |
|---|---|
| `audiocpp_model_info` | Get family/variant/description |
| `audiocpp_model_capabilities` | Get supported tasks, languages, feature flags |
| `audiocpp_free_model_info` | Free info struct |
| `audiocpp_free_capabilities` | Free capabilities struct |

---

### 7. WAV I/O Utilities

```c
// Read WAV → mono f32 PCM
float *samples; int64_t n; int rate;
audiocpp_read_wav("input.wav", &samples, &n, &rate);
// ... use samples ...
free(samples);

// Write mono f32 PCM → 16-bit WAV
audiocpp_write_wav("output.wav", samples, n, rate);
```

---

### 8. Artifacts (Reserved)

VoiceArtifact types for passing opaque data (embeddings, tokens) between
models. **Currently no shipping model produces or consumes artifacts.**
These types exist for forward compatibility.

```c
audiocpp_artifact_t *art = audiocpp_artifact_create(
    AUDIOCPP_ARTIFACT_SPEAKER_EMBEDDING, "spk_001",
    embedding_bytes, embedding_size);
audiocpp_artifact_set_meta(art, "dim", "256");
audiocpp_artifact_free(art);
```

| Function | Description |
|---|---|
| `audiocpp_artifact_create` | Create artifact with kind/id/payload |
| `audiocpp_artifact_set_meta` | Set metadata key-value pair |
| `audiocpp_artifact_free` | Free artifact |

---

### 9. Memory Management

All result handles are owned by the caller. Free with the matching function:

| Result type | Free function |
|---|---|
| `audiocpp_model_t` | `audiocpp_free_model` |
| `audiocpp_audio_t` | `audiocpp_free_audio` |
| `audiocpp_text_t` | `audiocpp_free_text` |
| `audiocpp_diar_t` | `audiocpp_free_diar` |
| `audiocpp_vad_t` | `audiocpp_free_vad` |
| `audiocpp_align_t` | `audiocpp_free_align` |
| `audiocpp_stems_t` | `audiocpp_free_stems` |
| `audiocpp_stream_t` | `audiocpp_stream_free` |
| `audiocpp_stream_event_t` | `audiocpp_free_stream_event` |
| `audiocpp_model_info_t` | `audiocpp_free_model_info` |
| `audiocpp_model_capabilities_t` | `audiocpp_free_capabilities` |
| `audiocpp_artifact_t` | `audiocpp_artifact_free` |
| `char *` (from error.message) | `audiocpp_free_string` |

### Error Handling

All functions that take `audiocpp_error_t *err` catch C++ exceptions and
convert them to error codes:
- `code = 0`: success
- `code = -1`: `std::exception` (message in `err.message`)
- `code = -2`: unknown exception

Pass `NULL` for `err` to ignore errors (not recommended).

---

## Symbol Visibility

The shared library exports **only** `audiocpp_*` symbols. All internal
symbols (ggml, CUDA, sentencepiece, cJSON, libyaml) are hidden via:
- `__declspec(dllexport)` on exported functions (AUDIOCPP_API macro)
- `CXX_VISIBILITY_PRESET hidden` on the target + all static dependencies
- `.def` file (Windows) / version script (Linux) as secondary filter

Verified by CI: a post-build symbol check step fails if any non-`audiocpp_*`
symbol appears in the export table.

---

## Platform Notes

### Windows
- The DLL links the MSVC C++ runtime dynamically (`MSVCP140.dll`,
  `VCRUNTIME140.dll`, `VCOMP140.dll`). Deploy the VCRedist or ship these
  alongside.
- CUDA builds require CUDA runtime DLLs (`cudart64_12.dll`, `cublas64_12.dll`,
  `cufft64_11.dll`) in the same directory or on PATH.
- ROCm builds require the AMD HIP SDK runtime DLLs.
- SYCL builds require oneAPI runtime DLLs.

### Linux
- The `.so` is self-contained for CPU/SYCL builds.
- CUDA builds require CUDA runtime libraries on the system or `LD_LIBRARY_PATH`.

### GPU Architecture Support

| Backend | Compiled architectures |
|---|---|
| CUDA | sm_61, sm_70, sm_75, sm_80, sm_86, sm_89, sm_120 (Pascal–Blackwell) |
| ROCm (HIP) | gfx1100, gfx1101, gfx1102, gfx1150, gfx1151, gfx1200, gfx1201 (RDNA3+4) |
| SYCL | All Intel GPU architectures (runtime JIT from SPIR-V) |
