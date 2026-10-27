# fetch_models.ps1 — copy the e2e model subset from the 4080 model server.
#
# Models live on the remote at D:\iStation\Models\sound\audiocpp (flat layout:
# single .gguf files or model directories). We copy into a local root OUTSIDE
# the repo (models are gitignored by policy); run_e2e.ps1 points --model at
# <root>\<name>.
#
# usage: powershell -File fetch_models.ps1 [-Dest C:\audiocpp-e2e\models] [-HostAlias 4080]
param(
    [string]$Dest = "C:\audiocpp-e2e\models",
    [string]$HostAlias = "4080"
)

$ErrorActionPreference = "Stop"
$remoteRoot = "D:/iStation/Models/sound/audiocpp"

# The e2e subset: one model per scenario (see run_e2e.ps1 for the mapping).
# Directory entries need scp -r; marked with a trailing '/'.
$items = @(
    "pocket-tts-english-q8_0.gguf",             # tts (en) + reference/align transcript source
    "moss-tts-nano-100m-q8_0.gguf",             # tts-batch
    "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf",# tts (zh, voice_id)
    "Qwen3-TTS-12Hz-0.6B-Base/",                # clon source: HF original; convert with audiocpp_gguf (README)
    "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf",# vdes (instruct)
    "qwen3-asr-0.6b-q8_0.gguf",                 # asr (zh)
    "parakeet-tdt-0.6b-v3-q8_0.gguf",           # asr words/timestamps (en)
    "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",# asr-stream
    "supertonic-3/",                            # tts-stream (pull events)  [directory]
    "sortformer-diar-4spk-v1-q8_0.gguf",        # diar
    "qwen3-forced-aligner-0.6b-q8_0.gguf",      # align
    "stable-audio-3-small-sfx-q8_0.gguf",       # gen (also feeds sep)
    "htdemucs-q8_0.gguf",                       # sep (transform_stems)
    "seed-vc-mlx-q8_0.gguf",                    # vc
    "muscriptor-small-f32.gguf"                 # midi
)

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$failed = @()
foreach ($name in $items) {
    $isDir = $name.EndsWith("/")
    $leaf = if ($isDir) { $name.TrimEnd("/") } else { $name }
    $local = Join-Path $Dest $leaf
    if (Test-Path $local) {
        Write-Host "[skip] $leaf (already present)"
        continue
    }
    Write-Host "[get ] $leaf"
    if ($isDir) {
        & scp -q -r "${HostAlias}:${remoteRoot}/${leaf}" $local
    } else {
        & scp -q "${HostAlias}:${remoteRoot}/${leaf}" $local
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $local)) {
        Remove-Item -Recurse -Force $local -ErrorAction SilentlyContinue  # never leave partial GB files to be "skipped" next run
        $failed += $leaf
    }
}
if ($failed.Count -gt 0) {
    Write-Warning ("failed: " + ($failed -join ", "))
    exit 1
}
Write-Host "done -> $Dest"
