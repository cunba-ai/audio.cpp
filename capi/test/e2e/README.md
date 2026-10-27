# CAPI end-to-end suite (sound-rs calling style)

`audiocpp_e2e.c` drives `audiocpp.dll` exactly the way the sound-rs backend
does (`sound-server/src/backend/audiocpp.rs`): the library is loaded at
runtime (`LoadLibraryA` + `GetProcAddress`, the C equivalent of libloading),
results follow the null-return + `audiocpp_error_t` protocol, errors are
released with `audiocpp_clear_error`, and per-run progress callbacks are
installed/cleared around runs. This catches ABI drift, export regressions and
behavioral breakage that static unit tests (which compile `audiocpp_capi.cpp`
directly) cannot see.

## Scenario → API coverage

| Scenario | Model (from 4080) | C API exercised |
|---|---|---|
| info | — | version, build_id, backend_available, device_count/info |
| tts-moss-en | moss-tts-nano-100m q8 (seed pinned) | tts, write_wav_ex; makes `input_en.wav` |
| clon-pocket | pocket-tts-english q8 | tts_with_voice_ref (clone; GGUF ships no preset voice) |
| tts-batch-moss | moss-tts-nano-100m q8 | tts_batch (CONCAT) + chapter ranges |
| tts-qwen3 | qwen3-tts customvoice q8 | load_model_ex (session opts), tts, model_info/capabilities |
| clon-qwen3 | qwen3-tts base 0.6B (converted) | tts_with_voice_ref + reference_text + progress callback |
| vdes-qwen3 | qwen3-tts voicedesign q8 | tts with `instruct` option |
| asr-qwen3-zh | qwen3-asr-0.6b q8 | asr, free_text (zh transcript round-trip) |
| asr-parakeet-en | parakeet-tdt q8 | asr with `return_timestamps` (keyword check) |
| denoise-gtcrn | *(embedded)* | denoise (no registry); output is the 16k mono feed below |
| asr-stream-nemotron | nemotron-3.5 streaming q8 | stream_start (audio contract) + push + finish |
| tts-stream-supertonic | supertonic-3 | stream_start (TTS) + stream_pull loop until end-of-stream |
| vad-embedded | *(embedded silero)* | load_model(NULL,"silero_vad"), vad, vad_energy |
| concat-2spk | — | read_wav + write_wav_ex (stitch two voices for diar) |
| diar-input-16k | *(embedded gtcrn)* | denoise as 16k resampler (sortformer wants mono 16 kHz) |
| diar-sortformer | sortformer-diar q8 | diar, free_diar (turns >= 1 on the 2-speaker mix) |
| align-qwen3 | qwen3-forced-aligner q8 | align, free_align (word timestamps) |
| gen-stable-audio | stable-audio-3-small-sfx q8 | generate + gen_result; output feeds sep/midi |
| sep-htdemucs | htdemucs q8 | transform_stems, free_stems (4 named stems) |
| vc-seed-vc | seed-vc-mlx q8 | audio_transform_with_voice_ref |
| superres-flashsr | *(embedded)* | super_resolve (16k -> 48k) |
| midi-muscriptor | muscriptor-small f32 | midi_from_wav + artifacts (MIDI payload) |
| errpath | — | bad path / malformed options / Metal probe |

Audio inputs chain through real outputs: `tts-pocket` makes `input_en.wav`,
`tts-qwen3` makes `input_zh.wav`, `gen-stable-audio` makes the `sep`/`midi`
input — the same "generate then analyze" shape sound-server uses.

## Running

```powershell
# 1. one-time: copy the model subset from the 4080 machine (~17 GB)
powershell -File capi\test\e2e\fetch_models.ps1            # -> C:\audiocpp-e2e\models

# 2. build the DLL (any preset with AUDIOCPP_BUILD_CAPI; cpu default) and run all
powershell -File capi\test\e2e\run_e2e.ps1

# options
powershell -File capi\test\e2e\run_e2e.ps1 -Filter tts        # subset by name
powershell -File capi\test\e2e\run_e2e.ps1 -SkipSlow          # skip gen/vc/midi on CPU
powershell -File capi\test\e2e\run_e2e.ps1 -BuildDir .\build\windows-cuda-release -Backend 1
```

Every scenario prints one machine-readable line
`E2E\t<sub>\tstatus=ok|fail\tms=...\tk=v...`; the runner applies expectations
(duration/RMS floors, keyword match, stem/artifact counts) and exits non-zero
on any failure. Human detail goes to stderr and is echoed by the runner.

Models are intentionally kept OUTSIDE the repo (`C:\audiocpp-e2e\models`);
`fetch_models.ps1` lists the exact subset. The local machine runs the CPU
backend; point `-BuildDir`/`-Backend` at a CUDA build on a GPU host to reuse
the same suite.

## Model notes & known findings (2026-10-28, CPU build, 23/23 pass)

- **qwen3 base GGUF**: the `qwen3-tts-12hz-1.7b-base-q8_0.gguf` on the 4080
  model server predates the current converter and fails to load
  (`GGUF tensor data range is out of bounds`), while customvoice/voicedesign
  from the same era load fine. The suite instead uses a freshly converted
  `qwen3-tts-12hz-0.6b-base-q8_0.gguf` produced from the HF original
  (`Qwen3-TTS-12Hz-0.6B-Base`, copied by `fetch_models.ps1`):

  ```powershell
  build\windows-cpu-release\bin\audiocpp_gguf.exe `
    --input model_weights=C:\audiocpp-e2e\models\Qwen3-TTS-12Hz-0.6B-Base\model.safetensors `
    --input speech_tokenizer_weights=C:\audiocpp-e2e\models\Qwen3-TTS-12Hz-0.6B-Base\speech_tokenizer\model.safetensors `
    --output C:\audiocpp-e2e\models\qwen3-tts-12hz-0.6b-base-q8_0.gguf `
    --type q8_0 --family qwen3_tts --root C:\audiocpp-e2e\models\Qwen3-TTS-12Hz-0.6B-Base
  ```

- **Nemotron streaming decodes to empty text** through the whole stack —
  reproduced with the engine's own `audiocpp_cli --mode streaming` on the same
  checkpoint (clean finish, zero partials, empty final text). The e2e scenario
  asserts the streaming *protocol* (start/push/finish) and reports `chars`;
  the empty decode is an engine-side issue worth reporting upstream.
- **`audiocpp_tts` does not forward progress callbacks** on this build while
  `audiocpp_tts_with_voice_ref` does (both engine-side emit sites exist).
  Progress is therefore asserted in `clon-*` scenarios and only reported for
  plain `tts`.
- **Determinism**: `tts-moss-en` pins `seed` so the chained English input (and
  everything downstream: ASR/VAD/diar/align) is byte-stable across runs;
  `clon-qwen3` pins its seed for the same reason.
- **Spec discovery**: `run_e2e.ps1` deploys the repo's `model_specs/*.json`
  into `<ModelsRoot>\model_specs\` — supertonic (HF directory layout) and the
  legacy-spec GGUFs (sortformer_diar, htdemucs) resolve their contracts from
  there.
