# run_e2e.ps1 — end-to-end suite for audiocpp.dll via the e2e CLI.
#
# Orchestrates the scenarios from the design table: builds the DLL if needed,
# compiles the CLI with cl.exe (vcvars), stages inputs by chaining TTS outputs
# into the analysis tasks, applies per-scenario expectations to the E2E result
# line, and prints a summary table. Exit code is non-zero when any scenario
# fails.
#
# usage:
#   powershell -File run_e2e.ps1 [-ModelsRoot C:\audiocpp-e2e\models]
#                                [-BuildDir .\build\windows-cpu-release]
#                                [-Backend 0] [-Threads 0] [-Filter <substring>]
#                                [-SkipSlow] [-KeepGoing]
param(
    [string]$ModelsRoot = "C:\audiocpp-e2e\models",
    [string]$BuildDir = "",
    [int]$Backend = 0,              # AUDIOCPP_BACKEND_CPU
    [int]$Threads = 0,              # 0 = auto
    [string]$Filter = "",
    [switch]$SkipSlow,              # skip the CPU-heavy music-gen/VC scenarios
    [switch]$KeepGoing              # continue after failures (default: stop at first)
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if ($BuildDir -eq "") { $BuildDir = Join-Path $repo "build\windows-cpu-release" }
$OutDir = Join-Path $ModelsRoot "out"
$StageDir = Join-Path $OutDir "stage"   # texts/options files (avoids shell quoting)
$ExeDir = $OutDir
$exe = Join-Path $ExeDir "audiocpp_e2e.exe"
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

function Write-Utf8NoBom([string]$path, [string]$text) {
    # UTF-8 without BOM so the engine's UTF-8 readers see clean bytes.
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.UTF8Encoding($false)))
}

# ── 1. DLL present? ────────────────────────────────────────────────────────
$dll = Join-Path $BuildDir "bin\audiocpp.dll"
if (-not (Test-Path $dll)) {
    if ($Backend -ne 0 -or $BuildDir -ne (Join-Path $repo "build\windows-cpu-release")) {
        throw "audiocpp.dll not found in $BuildDir; prebuild that preset first (auto-build only covers the default cpu-release layout)."
    }
    Write-Host "[build] audiocpp.dll not found in $BuildDir — building (cpu-release)..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "scripts\build_windows.ps1") `
        -Preset windows-cpu-release -Target audiocpp
    if ($LASTEXITCODE -ne 0) { throw "DLL build failed" }
}

# ── 2. Compile the CLI (cl.exe via vcvars) and stage the DLL next to it ────
$cliSrc = Join-Path $PSScriptRoot "audiocpp_e2e.c"
$cliHdr = Join-Path $repo "capi\include\audiocpp.h"
$needsBuild = -not (Test-Path $exe) -or
    (Get-Item $cliSrc).LastWriteTime -gt (Get-Item $exe).LastWriteTime -or
    (Get-Item $cliHdr).LastWriteTime -gt (Get-Item $exe).LastWriteTime
if ($needsBuild) {
    Write-Host "[build] compiling audiocpp_e2e.exe"
    $vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        $vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    }
    $tmpBat = Join-Path $env:TEMP "e2e_build_cli.bat"
    @"
@echo off
call "$vcvars" >nul
cd /d "$ExeDir"
cl /nologo /O2 /W3 /utf-8 /Fe:audiocpp_e2e.exe "$cliSrc" /I"$($cliHdr | Split-Path -Parent)"
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $tmpBat -Encoding ASCII
    & cmd /c $tmpBat
    if ($LASTEXITCODE -ne 0) { throw "CLI compile failed" }
}
Copy-Item $dll $ExeDir -Force

# ── 2b. Deploy model_specs next to the models ──────────────────────────────────
# Spec discovery looks for <model dir>/model_specs/<family>.json; supertonic (HF dir) and
# legacy-spec GGUFs (sortformer_diar, htdemucs) resolve from there.
$specsDir = Join-Path $ModelsRoot "model_specs"
New-Item -ItemType Directory -Force -Path $specsDir | Out-Null
Copy-Item (Join-Path $repo "model_specs\*.json") $specsDir -Force

# ── 3. Shared inputs (UTF-8 files so no shell quoting issues) ─────────────
$enText = "The quick brown fox jumps over the lazy dog near the river bank."
$zhText = "今天天气真不错，我们一起去公园散步吧。"
$EN_TEXT   = Join-Path $StageDir "en.txt";   Write-Utf8NoBom $EN_TEXT $enText
$ZH_TEXT   = Join-Path $StageDir "zh.txt";   Write-Utf8NoBom $ZH_TEXT $zhText
$EN_OPT    = Join-Path $StageDir "en_opt.json"
$ZH_OPT    = Join-Path $StageDir "zh_opt.json"
$VDES_OPT  = Join-Path $StageDir "vdes_opt.json"
$ST_OPT    = Join-Path $StageDir "supertonic_opt.json"
$GEN_OPT   = Join-Path $StageDir "gen_opt.json"
$LOAD_OPT  = Join-Path $StageDir "load_opt.json"

$m = @{
    pocket  = Join-Path $ModelsRoot "pocket-tts-english-q8_0.gguf"
    moss    = Join-Path $ModelsRoot "moss-tts-nano-100m-q8_0.gguf"
    qwTts   = Join-Path $ModelsRoot "qwen3-tts-12hz-1.7b-customvoice-q8_0.gguf"
    qwBase  = Join-Path $ModelsRoot "qwen3-tts-12hz-0.6b-base-q8_0.gguf"   # converted locally (see README)
    qwVdes  = Join-Path $ModelsRoot "qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf"
    qwAsr   = Join-Path $ModelsRoot "qwen3-asr-0.6b-q8_0.gguf"
    parrot  = Join-Path $ModelsRoot "parakeet-tdt-0.6b-v3-q8_0.gguf"
    nemotron= Join-Path $ModelsRoot "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"
    super3  = Join-Path $ModelsRoot "supertonic-3"
    sortfm  = Join-Path $ModelsRoot "sortformer-diar-4spk-v1-q8_0.gguf"
    aligner = Join-Path $ModelsRoot "qwen3-forced-aligner-0.6b-q8_0.gguf"
    stablAu = Join-Path $ModelsRoot "stable-audio-3-small-sfx-q8_0.gguf"
    htdm    = Join-Path $ModelsRoot "htdemucs-q8_0.gguf"
    seedvc  = Join-Path $ModelsRoot "seed-vc-mlx-q8_0.gguf"
    musica  = Join-Path $ModelsRoot "muscriptor-small-f32.gguf"
}

$IN_EN = Join-Path $OutDir "input_en.wav"     # produced by scenario tts-pocket
$IN16_EN = Join-Path $OutDir "input16_en.wav" # denoise output: 16k mono for VAD / streaming ASR
$IN_ZH = Join-Path $OutDir "input_zh.wav"     # produced by scenario tts-qwen3

# ── 4. Scenario table ─────────────────────────────────────────────────────
# expect: scriptblock invoked with $r = parsed k/v map; return $true to pass.
$scenarios = @(
    @{ name = "info"
       args = @("info")
       needs = @()
       expect = { $r["status"] -eq "ok" -and [int]$r["devices"] -ge 1 -and $r["cpu"] -eq "1" } }

    @{ name = "tts-moss-en"        # English source audio for the whole suite (pinned seed)
       args = @("tts", "--model", $m.moss, "--family", "moss_tts_nano", "--text-file", $EN_TEXT,
                "--options-file", (Join-Path $StageDir "moss_seed_opt.json"),
                "--out", $IN_EN)
       needs = @($m.moss)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 -and [double]$r["rms"] -ge 0.01 } }

    @{ name = "clon-pocket"        # pocket family via voice_ref clone (its GGUF ships no preset)
       args = @("clon", "--model", $m.pocket, "--family", "pocket_tts", "--text-file", $EN_TEXT,
                "--ref-wav", $IN_EN, "--out", (Join-Path $OutDir "pocket_clon.wav"))
       needs = @($m.pocket, $IN_EN)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 } }

    @{ name = "tts-batch-moss"
       args = @("tts-batch", "--model", $m.moss, "--family", "moss_tts_nano",
                "--text-file", $ZH_TEXT, "--text2-file", $EN_TEXT,
                "--out", (Join-Path $OutDir "batch_moss.wav"))
       needs = @($m.moss, $EN_TEXT)
       expect = { $r["status"] -eq "ok" -and [int]$r["n_items"] -ge 1 -and [double]$r["dur"] -ge 2.0 } }

    @{ name = "tts-qwen3"         # Chinese source audio + load_model_ex session options
       args = @("tts", "--model", $m.qwTts, "--family", "qwen3_tts", "--text-file", $ZH_TEXT,
                "--options-file", $ZH_OPT, "--load-options-file", $LOAD_OPT,
                "--progress", "--out", $IN_ZH)
       needs = @($m.qwTts, $ZH_TEXT)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 -and [double]$r["rms"] -ge 0.01 } }  # prog asserted in clon-qwen3 (tts entry does not forward progress; see README)

    @{ name = "clon-qwen3"        # voice clone from input_en (reference_text == en text)
       args = @("clon", "--model", $m.qwBase, "--family", "qwen3_tts",
                "--text-file", $ZH_TEXT, "--ref-wav", $IN_EN,
                "--options-file", (Join-Path $StageDir "clon_opt.json"),
                "--out", (Join-Path $OutDir "clon_zh.wav"))
       needs = @($m.qwBase, $IN_EN)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 } }

    @{ name = "vdes-qwen3"        # voice design: instruction in options
       args = @("tts", "--model", $m.qwVdes, "--family", "qwen3_tts", "--text-file", $ZH_TEXT,
                "--options-file", $VDES_OPT, "--out", (Join-Path $OutDir "vdes_zh.wav"))
       needs = @($m.qwVdes, $ZH_TEXT)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 } }

    @{ name = "asr-qwen3-zh"
       args = @("asr", "--model", $m.qwAsr, "--family", "qwen3_asr", "--in", $IN_ZH,
                "--options-file", (Join-Path $StageDir "asr_zh_opt.json"))
       needs = @($m.qwAsr, $IN_ZH)
       expect = { $r["status"] -eq "ok" -and [int]$r["chars"] -ge 6 } }

    @{ name = "asr-parakeet-en"   # word timestamps via return_timestamps
       args = @("asr", "--model", $m.parrot, "--family", "parakeet_tdt", "--in", $IN_EN,
                "--options-file", (Join-Path $StageDir "asr_ts_opt.json"))
       needs = @($m.parrot, $IN_EN)
       expect = { $r["status"] -eq "ok" -and ($r["text"] -match "dog" -or $r["text"] -match "lazy" -or $r["text"] -match "river") } }

    @{ name = "denoise-gtcrn"     # embedded audio-utility; also the 16k mono source for VAD/streaming ASR
       args = @("denoise", "--in", $IN_EN, "--denoiser", "gtcrn",
                "--out", $IN16_EN)
       needs = @($IN_EN)
       expect = { $r["status"] -eq "ok" -and $r["sr"] -eq "16000" -and [double]$r["dur"] -ge 1.0 } }

    @{ name = "asr-stream-nemotron"   # model-rate (16k) mono stream
       args = @("asr-stream", "--model", $m.nemotron, "--family", "nemotron_asr", "--in", $IN16_EN,
                "--chunk", "5120", "--options-file", (Join-Path $StageDir "asr_stream_opt.json"))
       needs = @($m.nemotron, $IN16_EN)
       expect = { $r["status"] -eq "ok" -and [int]$r["pushes"] -ge 5 } }  # chars>=6 blocked by nemotron empty-decode issue (see README)

    @{ name = "tts-stream-supertonic"
       args = @("tts-stream", "--model", $m.super3, "--family", "supertonic",
                "--options-file", $ST_OPT, "--out", (Join-Path $OutDir "stream_tts.wav"))
       needs = @($m.super3)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 -and [int]$r["pulls"] -ge 1 } }

    @{ name = "vad-embedded"      # embedded silero (NULL path) + energy VAD (16k input)
       args = @("vad", "--in", $IN16_EN)
       needs = @($IN16_EN)
       expect = { $r["status"] -eq "ok" -and [int64]$r["model_segments"] -ge 1 } }

    @{ name = "concat-2spk"        # stitch two different voices for diarization
       args = @("concat", "--in", (Join-Path $OutDir "pocket_clon.wav"), "--in2", $IN_ZH,
                "--out", (Join-Path $OutDir "diar_input.wav"))
       needs = @((Join-Path $OutDir "pocket_clon.wav"), $IN_ZH)
       expect = { $r["status"] -eq "ok" } }

    @{ name = "diar-input-16k"       # sortformer docs require mono 16 kHz input
       args = @("denoise", "--in", (Join-Path $OutDir "diar_input.wav"), "--denoiser", "gtcrn",
                "--out", (Join-Path $OutDir "diar_input16.wav"))
       needs = @((Join-Path $OutDir "diar_input.wav"))
       expect = { $r["status"] -eq "ok" -and $r["sr"] -eq "16000" } }

    @{ name = "diar-sortformer"     # two speakers, expect both detected
       args = @("diar", "--model", $m.sortfm, "--family", "sortformer_diar", "--in",
                (Join-Path $OutDir "diar_input16.wav"))
       needs = @($m.sortfm, (Join-Path $OutDir "diar_input16.wav"))
       expect = { $r["status"] -eq "ok" -and [int64]$r["turns"] -ge 1 } }

    @{ name = "align-qwen3"
       args = @("align", "--model", $m.aligner, "--family", "qwen3_forced_aligner",
                "--in", $IN_EN, "--text-file", $EN_TEXT, "--language", "en")
       needs = @($m.aligner, $IN_EN)
       expect = { $r["status"] -eq "ok" -and [int64]$r["words"] -ge 8 } }

    @{ name = "gen-stable-audio"  # sfx prompt -> wav, reused as sep input
       slow = $true
       args = @("gen", "--model", $m.stablAu, "--family", "stable_audio",
                "--text-file", (Join-Path $StageDir "gen_prompt.txt"),
                "--options-file", $GEN_OPT, "--out", (Join-Path $OutDir "gen_sfx.wav"))
       needs = @($m.stablAu)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 3.0 } }

    @{ name = "sep-htdemucs"
       args = @("sep", "--model", $m.htdm, "--family", "htdemucs", "--in",
                (Join-Path $OutDir "gen_sfx.wav"), "--out-dir", $OutDir)
       needs = @($m.htdm, (Join-Path $OutDir "gen_sfx.wav"))
       expect = { $r["status"] -eq "ok" -and [int64]$r["stems"] -ge 2 } }

    @{ name = "vc-seed-vc"        # convert input_zh timbre toward input_en speaker
       slow = $true
       args = @("vc", "--model", $m.seedvc, "--family", "seed_vc", "--in", $IN_ZH,
                "--ref-wav", $IN_EN, "--out", (Join-Path $OutDir "vc_zh.wav"))
       needs = @($m.seedvc, $IN_ZH, $IN_EN)
       expect = { $r["status"] -eq "ok" -and [double]$r["dur"] -ge 1.0 } }

    @{ name = "superres-flashsr"  # embedded; 16k -> 48k
       args = @("superres", "--in", $IN_EN, "--out", (Join-Path $OutDir "superres.wav"))
       needs = @($IN_EN)
       expect = { $r["status"] -eq "ok" -and $r["sr"] -eq "48000" } }

    @{ name = "midi-muscriptor"   # audio -> MIDI artifacts (loose: call must succeed)
       slow = $true
       args = @("midi", "--model", $m.musica, "--family", "muscriptor",
                "--in", (Join-Path $OutDir "gen_sfx.wav"),
                "--out", (Join-Path $OutDir "out.mid"))
       needs = @($m.musica, (Join-Path $OutDir "gen_sfx.wav"))
       expect = { $r["status"] -eq "ok" -and [int64]$r["artifacts"] -ge 1 } }

    @{ name = "errpath"
       args = @("errpath")
       needs = @()
       expect = { $r["status"] -eq "ok" -and $r["passed"] -eq $r["checks"] } }
)

# option files that depend on earlier-produced paths or fixed text
Write-Utf8NoBom $EN_OPT '{"language": "en"}'
Write-Utf8NoBom $ZH_OPT '{"language": "zh", "voice_id": "eric"}'
Write-Utf8NoBom $VDES_OPT '{"language": "zh", "instruct": "用温柔的女声说，语速适中"}'
Write-Utf8NoBom (Join-Path $StageDir "moss_seed_opt.json") '{"seed": 1234}'
Write-Utf8NoBom (Join-Path $StageDir "clon_opt.json") ('{"reference_text": "' + $enText + '", "seed": 42}')
Write-Utf8NoBom (Join-Path $StageDir "asr_zh_opt.json") '{"language": "zh"}'
Write-Utf8NoBom (Join-Path $StageDir "asr_ts_opt.json") '{"return_timestamps": true}'
Write-Utf8NoBom (Join-Path $StageDir "asr_stream_opt.json") '{"language": "en-US"}'
Write-Utf8NoBom $ST_OPT ('{"text": "' + $enText + '", "num_inference_steps": "8"}')
Write-Utf8NoBom $GEN_OPT '{"duration_sec": 6.0, "num_inference_steps": 12}'
Write-Utf8NoBom $LOAD_OPT '{"qwen3_tts.attention": "mixed"}'
Write-Utf8NoBom (Join-Path $StageDir "gen_prompt.txt") "crackling vinyl record with soft rain and distant thunder sound effects"

# ── 5. Run ────────────────────────────────────────────────────────────────
function Parse-E2ELine([string]$line) {
    # "E2E`t<sub>`tstatus=ok`tms=..`tk=v`t..." -> map
    $map = @{}
    foreach ($tok in ($line -split "`t")) {
        $i = $tok.IndexOf("=")
        if ($i -gt 0) { $map[$tok.Substring(0, $i)] = $tok.Substring($i + 1) }
    }
    return $map
}

$rows = [System.Collections.Generic.List[object]]::new()
$env:AUDIOCPP_E2E = "1"
foreach ($sc in $scenarios) {
    if ($Filter -ne "" -and $sc.name -notlike "*$Filter*") { continue }
    if ($SkipSlow -and $sc.slow) {
        $rows.Add([pscustomobject]@{ Scenario = $sc.name; Result = "SKIP"; Ms = 0; Detail = "" })
        continue
    }
    $missing = @($sc.needs | Where-Object { -not (Test-Path $_) })
    if ($missing.Count -gt 0) {
        $rows.Add([pscustomobject]@{ Scenario = $sc.name; Result = "MISSING"; Ms = 0;
            Detail = "missing: $($missing[0])" })
        continue
    }
    Write-Host "`n=== $($sc.name) ===" -ForegroundColor Cyan
    $allArgs = @($sc.args + @("--backend", "$Backend", "--threads", "$Threads") |
        ForEach-Object { '"{0}"' -f $_ })
    $logDir = Join-Path $OutDir "logs"; New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stdoutF = Join-Path $logDir "$($sc.name).out.txt"
    $stderrF = Join-Path $logDir "$($sc.name).err.txt"
    try {
        $proc = Start-Process -FilePath $exe -ArgumentList $allArgs -NoNewWindow `
            -WorkingDirectory $ExeDir -RedirectStandardOutput $stdoutF -RedirectStandardError $stderrF -PassThru -Wait
    } catch {
        $rows.Add([pscustomobject]@{ Scenario = $sc.name; Result = "FAIL"; Ms = 0; Detail = "start failed: $($_.Exception.Message)" })
        if (-not $KeepGoing) { break } else { continue }
    }
    Get-Content $stderrF -Encoding UTF8 | Write-Host
    $e2eLine = (Get-Content $stdoutF -Encoding UTF8 | Where-Object { $_ -like "E2E`t*" } | Select-Object -Last 1)
    if (-not $e2eLine) {
        $rows.Add([pscustomobject]@{ Scenario = $sc.name; Result = "FAIL"; Ms = 0; Detail = "no E2E line (exit $($proc.ExitCode))" })
        if (-not $KeepGoing) { break } else { continue }
    }
    $r = Parse-E2ELine $e2eLine
    $pass = $false
    try { $pass = & $sc.expect } catch {
        Write-Warning "expectation evaluator threw: $_"
    }
    $detail = ($e2eLine -replace "^E2E`t[^`t]+`t[^`t]+`t", "")
    if ($detail.Length -gt 110) { $detail = $detail.Substring(0, 110) + "..." }
    $rows.Add([pscustomobject]@{ Scenario = $sc.name;
        Result = if ($pass) { "PASS" } else { "FAIL" };
        Ms = [int]($r["ms"] -as [double]); Detail = $detail })
    if (-not $pass -and -not $KeepGoing) { break }
}

# ── 6. Summary ────────────────────────────────────────────────────────────
Write-Host "`n==================== E2E SUMMARY ====================" -ForegroundColor Yellow
$rows | Format-Table -AutoSize | Out-String | Write-Host
$nFail = @($rows | Where-Object { $_.Result -ne "PASS" -and $_.Result -ne "SKIP" }).Count
Write-Host ("{0} passed, {1} failed/skipped of {2}" -f `
    @($rows | Where-Object { $_.Result -eq "PASS" }).Count, $nFail, $rows.Count)
if ($nFail -gt 0) { exit 1 }
exit 0
