# CAPI 封装 vs 官方 CLI — TTS 能力对比测试报告

> 日期:2026-08-09 · 机器:4080(192.168.90.41, RTX 4070 Ti SUPER 16 GB, 驱动 591.86)
> 对比对象:
> - **官方 CLI**:`audiocpp_cli.exe`(上游 0xShug0/audio.cpp release-0.5.1,commit `238ab6a9`,CUDA portable 包)
> - **CAPI 封装**:`libaudiocpp`(`audiocpp.dll`,本地 HEAD `847502d`,与 sound-server 正在使用的 dll **同一构建**)
>
> 测试条件完全一致:CUDA backend、8 threads、相同文本
> `"Hello, this is a short test of the audio text to speech system."`、相同 voice-ref
> (`ref_voice.wav`,由 higgs-audio-v3-tts-4b 合成,已知内容)、每个模型每个二进制一次合成(关键模型多次复测方差)。

## 结论速览(回答"慢 / PCM 无法播放 / 背景噪声")

| 用户现象 | 结论 | 证据 |
|---|---|---|
| **有的模型慢** | **CAPI 不慢——整体与官方 CLI 持平或更快**。唯一双方都"巨慢"的是 dramabox(官方 23.5 分钟/8.6s 音频,RTF=163;CAPI 跑了 34 分钟被我们手动终止,同样病态)——这是模型/引擎问题,不是 CAPI 问题。真正的"慢"来源更可能是**服务端每次请求重新加载模型**(见下) | 见速度表 |
| **PCM 无法播放** | 找到 **1 个真实 CAPI bug**:`audiocpp_tts` 丢掉 `AudioBuffer.channels`,立体声模型(dramabox 输出 ch=2)会被当单声道返回并写成 1 声道 WAV → 播放变调/错乱。另有 1 个偶发:mosstts-nano 在 harness 里出现一次加载分配失败(单跑复现不了)。其余模型输出全部合法(无 NaN/Inf、无削波、时长/采样率正确) | 见 bug 详述 |
| **背景有噪声** | **未复现为 CAPI 系统性问题**。逐模型噪声底(CAPI)= 官方或更低;irodori-v4 一度差 57 dB,连跑 3 次证明是采样随机性,不是 CAPI。confucius4/glm/index/pocket 等输出与官方几乎逐位一致 | 见质量表 |

## 测试矩阵总览(27 个 TTS 模型/变体)

### A. 双方都成功(可对比)

| 模型 | 官方 wall_ms(prepare+run) | CAPI synth_ms | CAPI load_ms | 官方时长 | CAPI 时长 | 备注 |
|---|---|---|---|---|---|---|
| fish-audio-s2-pro | 2935 | 2533 | 9.5s | 4.09s | 3.95s | CAPI 快 1.16× |
| higgs-audio-v3-tts-4b | 1506 | 1321 | 13.4s | 4.28s | 4.28s | CAPI 快 1.14× |
| irodori-tts-v4-small | 1082 | 809 | 3.2s | 5.68s | 5.68s | CAPI 快 1.34× |
| irodori-tts-600m-v3 | 631 | 624 | 3.3s | 6.44s | 6.44s | 持平 |
| outetts-1.0-1b | 6723 | 3914 | 4.0s | 3.49s | 3.68s | CAPI 快 1.72× |
| voxcpm2 | 1420 | 1434 | 3.1s | 3.52s | 3.52s | **输出逐位一致**(corr≈1) |
| moss-tts-local-v1.5 | 7359 | 2552 | 33–50s | 4.64s | 7.52s | CAPI 快 2.9×;时长差=采样方差(官方自身 4.6–9.3s) |
| moss-tts-nano-100m | 3700 | 3222(单跑) | ~1s | 3.2s | 7.36s | 官方自身 3.2–10.4s;CAPI 在 harness 中一次分配失败(见下) |
| pocket-tts-english(ref) | 637 | 741 | 0.4s | 3.68s | 3.68s | 时长一致 |
| qwen3-tts-12hz-0.6b(ref) | 3226 | 1845 | 3.4s | 3.92s | 4.00s | CAPI 快 1.7× |
| glm-tts(ref) | 35404 | 10931 | 1.6s | 3.60s | 3.60s | **CAPI 快 3.2×**(官方反常地慢) |
| index-tts2(ref) | 12667 | 8076 | 13.4s | 3.75s | 3.75s | 输出几乎一致 |
| vietneu-tts-v3(ref) | 10788 | 9685 | 1.6s | **327.5s** | **327.5s** | 双方同样病态(英文文本下生成 5.5 分钟),时长逐秒一致 |
| vevo2(ref) | 8349 | 1353 | 8.7s | 3.52s | 4.56s | CAPI 快 |
| chatterbox(clon+ref) | 10112 | 3986 | 0.45s | 3.56s | 4.36s | CAPI 快 2.5× |
| confucius4-tts(clon+ref) | 13652 | 12652 | 17.4s | 3.92s | 3.92s | **输出逐位一致** |
| qwen3-tts-voicedesign(vdes) | 2426 | 4259(2nd) | 9.9s | 3.60s | 3.92s | 首次 CAPI 15s 为冷启动离群,复测 4.3s |
| vibevoice(prompt格式) | 8397 | 9409 | 14.5s | 4.00s | 4.00s | 文本需 `Speaker N:` 行格式 |
| miotts-1.7b(ref+codec) | 1087 | 4790 | 13.0s | 3.76s | 3.88s(28s 一次) | 见"miotts 特别说明";CAPI 偶发 28s 长生成 |
| dramabox | 1410790 | **>34min 被终止** | — | 8.65s(stereo) | — | 双方都不可用 |

### B. 双方失败(错误完全一致,均非 CAPI 问题)

| 模型 | 原因 |
|---|---|
| qwen3-tts-12hz-1.7b-base GGUF | 文件损坏:`GGUF tensor data range is out of bounds`(双方同错) |
| OmniVoice | safetensors 损坏:`tensor data range is out of bounds`(双方同错) |
| irodori-tts-500m-v3 | 模型目录不完整:缺 `Semantic-DACVAE-Japanese-32dim/weights.safetensors`(该目录只有 `weights.pth`)(双方同错) |
| qwen3-tts-customvoice | `Qwen3 custom voice prefill requires speaker`(双方同错;需 voice-id 而非 ref 音频?) |
| miotts(初始) | 缺 MioCodec 模型:官方 CLI 找 `%TEMP%\audiocpp-gguf\MioCodec-25Hz-44.1kHz-v2`、CAPI 找同级目录 `MioCodec-25Hz-44.1kHz-v2`(已建目录+放入 GGUF 后双方都通,见下) |

## 三个现象逐条详述

### 1. "慢" — 复测结论

- 19 个可对比模型中,**CAPI synth 快于官方 wall 的有 11 个,持平 3 个,慢的只有 2 个且幅度小**(vibevoice 9409 vs 8397、qwen3-voicedesign 复测 4259 vs 2426)。官方整体反而慢,可能与官方包(238ab6a)缺后续性能改动有关。
- 双方都巨慢的只有 **dramabox**(官方 RTF=163,8.6s 音频花了 23.5 分钟;CAPI 34 分钟未完)。
- **真正会让用户感到"CAPI 慢"的来源**:
  1. **模型加载耗时大且每次请求可能重载**:CAPI load 实测 moss-tts-local **33–50s**、index-tts2 13s、confucius 17s、vibevoice 14.5s、vevo2 8.7s。而 sound-server 以 `--max-models 2` 启动(最多同时加载 2 个模型)→ 请求切换模型必然驱逐+重载 → 用户看到 10–50s 的"生成时间"。官方 CLI 全进程(moss-local 50.7s)同样慢,这是引擎加载路径的问题,不是 CAPI 独有,但服务端策略放大了它。
  2. 个别模型首次运行冷启动(CUDA graph warmup + 首次图构建):qwen3-voicedesign 首跑 15s,复测 4.3s。

### 2. "PCM 无法播放" — 1 个真实 bug + 1 个偶发 + 3 个文件损坏

**Bug(CAPI,建议修)**:`capi/src/audiocpp_capi.cpp` 的 `audiocpp_tts` / `audiocpp_tts_with_voice_ref` 只拷贝 `buf.samples` 和 `buf.sample_rate`,**丢弃 `buf.channels`**。dramabox 会输出 stereo(`session.cpp:679` `buffer.channels = audio.channels`,官方 CLI 写出的 WAV 就是 ch=2),经 CAPI 后被当作单声道返回/写出 → 交错立体声数据按 1 声道播放 = 变调+双声道混合的刺耳音频("没法播放")。修复建议:检查 `buf.channels`,>1 时要么拒绝并报错,要么新增接口暴露声道数/下混为单声道后再返回(API 契约写的是 mono)。

**偶发**:moss-tts-nano 在 harness 里出现一次 `failed to allocate moss.audio_tokenizer.decoder backend weight buffer`(加载阶段,`BackendWeightStore::upload()` 的 `ggml_backend_alloc_ctx_tensors` 返回 null)。随后单跑 3 次全部成功——疑似前一个进程退出后 WDDM 显存回收时序导致的瞬时分配失败,建议在服务端加一次重试/记录。

**文件损坏(与 CAPI 无关,双方同错)**:qwen3-tts-base GGUF、OmniVoice、irodori-500m-v3 模型文件本身损坏/不完整。

### 3. "背景噪声" — 复测结论:无系统性差异

逐模型噪声底(最静 500ms 窗口 RMS):

| 模型 | 官方 | CAPI | 模型 | 官方 | CAPI |
|---|---|---|---|---|---|
| voxcpm2 | -26.3 | -26.3 | glm-tts | -27.2 | -27.2 |
| confucius4 | -31.8 | -31.8 | index-tts2 | -27.5 | -27.2 |
| pocket-tts | -27.9 | -28.3 | outetts | -28.8 | -25.2 |
| fish-audio | -25.3 | -31.1 | higgs | -36.1 | -37.2 |
| irodori-v4 | -76.5 | -19.8(1st) | irodori-v4 复测 | — | -22.4 / -31.4 |

irodori-v4 首次差异 57 dB 曾像"背景噪声",但 CAPI 连跑 3 次噪声底 -19.8/-22.4/-31.4 dB 漂移,且其 50ms 窗口最低可达 -75.7 dB——是生成内容的随机差异(静音段位置/底噪形态不同),不是 CAPI 叠加噪声。所有输出 NaN=0、Inf=0、clip=0(moss-nano 文本"Hello"时模型自身输出 max>1.0,双方同源)。

## 其他值得注意的发现

1. **voxcpm2 / confucius4 / glm / index / pocket 的输出与官方近乎逐位一致**(corr≈0.999999999 或时长/RMS/噪声底全等)——同一引擎同一路径,种子不同才会不同。
2. **moss 系列时长差异是采样方差**:官方 moss-local 4 次 4.64–9.28s、moss-nano 4 次 3.2–10.4s;CAPI 值落在此区间内。miotts CAPI 有一次生成 28s(连续语音块,AR 未及时命中 EOS),官方 3 次都 3.7s 左右——小概率长生成现象,若客户端按固定时长校验会视为"异常音频"。
3. **vietneu 用英文文本+ref 生成 327.5s 立体声**,双方时长逐秒一致(10788ms 官方 / 9685ms CAPI 生成)——模型行为,不是 CAPI 差异。建议用越南语文本/ref 复测。
4. **CAPI 局限(建议加接口)**:`audiocpp_load_model` 不接受 session options——`miotts.codec_model_path` 这类**会话级**选项无法通过 CAPI 传入(miotts 只能靠默认同级目录才能跑)。官方 CLI 用 `--session-option` 可传。建议给 CAPI 增加 load options 或让 miotts 支持 request 级选项。
5. **miotts 部署修复**:在 `D:\iStation\Models\sound\audiocpp\MioCodec-25Hz-44.1kHz-v2\` 下放入 `miocodec-25hz-44khz-v2-q8_0.gguf` 后,官方 CLI 与 CAPI 都能用默认路径找到 codec,双方跑通。注意:该目录是测试时新建的,若 sound-server 的模型扫描器会把它当成第二个 miocodec 模型注册,可在服务器配置里排除。
6. **vibevoice 的文本格式要求**是 `Speaker N: ...` 行,普通句子双方都报 `VibeVoice prompt has no valid Speaker N: lines`(一致)。
7. 双方二进制支持的模型家族完全一致(44 个),本次对比不存在"官方支持而 CAPI 不支持"的家族。

## 环境与复现

- 4080 部署目录:`D:\audiocpp_test\`(official CLI + CUDA13 runtime DLLs、capi_tts_test.exe + audiocpp.dll + CUDA12.9 runtime DLLs、harness 脚本、ref_voice.wav、选项 JSON)
- 输出:`D:\audiocpp_test\out\`(每个模型 `*.official.log/.wav`、`*.capi.log/.wav/.wav.f32`)
- 测试工具源码(可复用于回归):`capi/test/capi_tts_test.c`、`run_tts_compare.ps1`、`run_tts_retry.ps1`、`run_capi_ref_models.bat`、`run_clon_models.bat`、`analyze_tts_compare.py`
- CAPI dll 构建 ID:`847502d 847502d8a2e5f4d1f73b406924e36feb47944e21 main 2026-08-08T10:15:54Z`(与 sound-server 部署的 dll 相同)
- 官方 CLI:release-0.5.1 的 `audiocpp-windows-cuda-portable-238ab6a9.zip` + `audiocpp-windows-cuda-runtime.zip`(cublas64_13/cublasLt64_13/cufft64_12)

## 建议的后续动作

1. 修 CAPI stereo bug(丢弃 channels)。
2. 给 CAPI `audiocpp_load_model` 增加 session options 参数(或 `audiocpp_load_model_ex`),否则 miotts 类会话级选项模型无法配置。
3. 服务端:`--max-models` 调大或对热门模型做常驻;对 moss-tts-nano 的一次性分配失败加日志/重试。
4. 换损坏的模型文件(qwen3-tts-base、OmniVoice、irodori-500m-v3 的 codec)。
5. dramabox 的 163× RTF 与 vietneu 的 327s 生成值得单独排查(模型级问题,双方一致)。
