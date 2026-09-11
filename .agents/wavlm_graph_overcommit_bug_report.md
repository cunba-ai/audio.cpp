# MioCodec VC OOM — WavLM 图 buffer 过度分配(根因已定位并修复)

> **状态:已修复(2026-08-07)。**根因是 `WavlmRunner::ensure_graph()` 用
> `ggml_backend_alloc_ctx_tensors` 分配图张量,该 API 给 context 里**每个** tensor
> 分配独立 buffer、完全不分析生命周期、零重用。WavLM base 的 12 层 transformer 每层
> 物化一份 `[B, H, tokens, tokens]` 的 attention scores,在 35 s 音频(tokens≈1786)
> 下光 WavLM 图 buffer 就 ~15 GiB,CUDA 上合成成单个 ~46 GiB 的 cudaMalloc 直接 OOM。
> 所有同类 encoder(HuBERT / wav2vec2-bert / whisper / campplus)和 MioCodec 自身的图
> 都早已用 `ggml_gallocr_alloc_graph`,WavLM 是唯一漏网的。改用 gallocr 后峰值内存
> 17.61 GiB → 3.14 GiB,输出数值完全一致。

## TL;DR

sound-server 调 miocodec-25hz-44khz-v2 做 voice conversion,35.7 s 音频(2.17 MB WAV)、
VRAM 空闲 11 GiB,却报 OOM:`ggml_backend_cuda_buffer_type_alloc_buffer: allocating
46868.30 MiB ... cudaMalloc failed: out of memory`。模型本身极小(Q8 GGUF),加载只用
~1 GiB。

- **崩溃点**:WavLM SSL 特征提取器的图 buffer 分配。
- **根因**:`wavlm_encoder.cpp::WavlmRunner::ensure_graph()` 用
  `ggml_backend_alloc_ctx_tensors(ggml_, backend)` 分配整个 context 的张量。该 API 不做
  生命周期分析,给 context 里每个 tensor(含 12 层各自独立的 attention scores
  `[1, 12, 1786, 1786]` ≈ 0.14 GiB/层 + softmax 输出同尺寸)各分一块独立 buffer。
- **对比**:HuBERT / wav2vec2-bert / whisper / campplus / MioCodec 三图全部用
  `ggml_gallocr_alloc_graph`,gallocr 分析图拓扑后重用 lifetime 不重叠的张量 buffer,
  所以它们对同样的 1786 tokens 只有 0.05–0.19 GiB。WavLM 是唯一漏网。
- **修复**:把 WavLM 的分配换成 gallocr 三件套(`ggml_gallocr_new` + `reserve` +
  `alloc_graph`),与 HuBERT 范式一致。
- **效果**(35 s recorded.wav,CPU backend,VC with voice_ref=same clip):
  - WavLM 图 buffer:**15.05 GiB → 0.82 GiB(18×)**
  - 进程峰值 WS:**17.61 GiB → 3.14 GiB(5.6×)**
  - 输出音频数值完全一致(`max_abs=0.4880`,非零 1573140/1574370,99.9%)。

## 现象

- **触发**:`POST /api/transform_file` 用 `miocodec-25hz-44khz-v2` + `voice_ref_*` 做语音
  转换,输入 `recorded.wav`(stereo 16 kHz 16-bit,2.17 MB,35.70 s)。
- **报错**(两次请求,数字完全相同):
  ```
  ggml_backend_cuda_buffer_type_alloc_buffer: allocating 46868.30 MiB on device 0: cudaMalloc failed: out of memory
  alloc_tensor_range: failed to allocate CUDA0 buffer of size 49144978048
  ```
  → HTTP 500,`elapsed_ms=9068 / 4388`(够跑完 WavLM,在建 wave 相关图时炸)。
- **反常点**:46868.30 MiB(45.77 GiB)对一个 35 s 音频完全不合理;数字固定不变
  (`49144978048` = 2⁷ × 19 × 23 × 878593)说明不是数据相关,是固定/结构性的过度分配。

## 环境

- **模型**:`miocodec-25hz-44khz-v2-q8_0.gguf`(Q8,加载后 ~1 GiB VRAM)
- **音频**:`recorded.wav`,stereo 16 kHz 16-bit,571264 frames/声道,35.70 s
- **sound-server 路径**:`run_transform` 先把 16k→44.1k 重采样(mono 1574546 samples),
  再经 `audio_transform_with_voice_ref` 传 sr=44100 + voice_ref(16k)。
- **本机调试**:无 CUDA,用 CPU backend 复现(CPU backend 用同一个图 buffer,只是
  分配走系统内存/页面文件,够大就不会 OOM——但峰值内存照样暴露过度分配)。

## 调查方法(systematic debugging)

### 阶段 1:证伪"推理路径正常大小"假设

先精确算 35.7 s 音频在 MioCodec 各阶段的 token 帧数:

```
mono_44k     = 1574546 samples
WavLM 16k 输入 = 571601(44.1k→16k 重采样 + padding)
WavLM tokens  = 1786(conv [10,3,3,3,3,2,2] / stride [5,2,2,2,2,2,2])
content frames= 893(conv_downsample ÷2)
stft frames   = 1785(= samples / hop(98) / upsample_factor(3×3=9))
wave upsampler输出 = 1785×9 = 16065
```

按此,三个 MioCodec 图(content_encoder / global_encoder / wave_decoder)的 attention
mask/scores 都很小(单张 `[1,1,1786,1786] f32 ≈ 12 MiB`),整个 wave_decoder 图峰值
~200 MiB,**完全无法解释 46 GiB**。所以先排除了 MioCodec 本身的图构造。

### 阶段 2:复现 + 加诊断定位"哪一个图"

用 Python ctypes 直接调 `libaudiocpp.dll`(绕开 sound-server,排除 Rust 侧解析/重采样
差异),带 voice_ref 跑 VC,CPU backend:

- 不带 voice_ref → 抛 `MioCodec run() requires voice speaker audio`(符合 session.cpp:166)。
- 带 voice_ref → `code=0`,但进程峰值 WS **17.61 GiB**(模型才 0.62 GiB)。

在 `wavlm_encoder.cpp::ensure_graph()` 和三个 miocodec 图构造函数里加临时 fprintf
诊断(用 `ggml_gallocr_get_buffer_size` / `ggml_backend_buffer_get_size` 打印每个图的
buffer 字节数),重新编译跑同一音频:

```
[DIAG] wavlm graph: samples=571601 tokens=1786 n_nodes=1080 buffer=16159632416 bytes (15.05 GiB)  ← local SSL
[DIAG] content_encoder graph: ssl=1786 content=893 n_nodes=341 buffer0=198788960 bytes (0.19 GiB)
[DIAG] wavlm graph: samples=571601 tokens=1786 n_nodes=485  buffer=8211260544 bytes (7.65 GiB)   ← global SSL
[DIAG] global_encoder graph: ssl=1786 n_nodes=171 buffer0=49379328 bytes (0.05 GiB)
[DIAG] wave_decoder graph: content=893 stft=1785 n_nodes=1496 buffer0=138916160 bytes (0.13 GiB)
PEAK WS = 17.61 GiB
```

**罪魁是 WavLM**,不是 MioCodec 的任何一个图。两次 WavLM(local SSL 提取 source
content、global SSL 提取 voice reference)分别 15.05 GiB 和 7.65 GiB,加起来撑出
17.61 GiB 峰值(第一次跑完释放了才没到 22.7 GiB)。

### 阶段 3:定位 WavLM 为什么这么大

`build_wavlm_graph_layers` 对 12 层 transformer 循环,每层调
`build_wavlm_self_attention`,其中:

```cpp
auto scores = MatMulModule{}.build(ctx, q, k_t);           // [1, 12, 1786, 1786] f32 = 0.14 GiB
scores = mul_scalar(...);
scores = add_same(ctx, scores, rel_bias);
auto attn = ggml_soft_max_ext(..., scores.tensor, attention_mask.tensor, ...);  // 同尺寸 = 0.14 GiB
auto context = MatMulModule{}.build(ctx, attn, v);
```

理论上每层 scores 在被 softmax 消费后就死了,下一层的 scores 应能复用同一块 buffer。
但 `ensure_graph` 末尾用:

```cpp
buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, backend);  // ← 问题在这
```

`ggml_backend_alloc_ctx_tensors` 给 context 里**每个** tensor 各分一块独立 buffer,
**不做任何生命周期分析、零重用**。12 层 × (scores 0.14 + attn 0.14) ≈ 3.5 GiB,加上
`position_bias` `[1,12,1786,1786]` = 0.14 GiB(跨层共享但仍独立分配)、各层 FFN/hidden
激活,合计 ~15 GiB。global 那次输出层数少、n_nodes 少(485 vs 1080),所以 7.65 GiB。

### 阶段 4:确认是"唯一漏网"

grep 所有 speech encoder + miocodec 图的分配方式:

| encoder | 分配 API | 行为 |
|---------|---------|------|
| HuBERT | `ggml_gallocr_alloc_graph` | lifetime 重用 ✓ |
| wav2vec2-bert | `ggml_gallocr_alloc_graph`(input 走单独 ctx) | lifetime 重用 ✓ |
| whisper | `ggml_gallocr_alloc_graph` | lifetime 重用 ✓ |
| campplus | `ggml_gallocr_alloc_graph` | lifetime 重用 ✓ |
| MioCodec 三图 | `ggml_gallocr_alloc_graph` | lifetime 重用 ✓ |
| **WavLM** | **`ggml_backend_alloc_ctx_tensors`** | **零重用 ✗** |

WavLM 是唯一用 `alloc_ctx_tensors` 的。改为 gallocr 即与全部同类对齐。

## 修复

`src/framework/modules/speech_encoders/wavlm_encoder.cpp`:

- `WavlmRunner` 加成员 `ggml_gallocr_t gallocr_ = nullptr;`
- `release_graph()` 先 `ggml_gallocr_free(gallocr_)`(gallocr 持有并释放其内部 backend
  buffer,取代原来 `buffer_` 的角色;`buffer_` 保留为 nullptr,其 free 分支不再触发)。
- `ensure_graph()` 把 `ggml_backend_alloc_ctx_tensors(ggml_, backend)` 换成:

```cpp
gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
if (gallocr_ == nullptr ||
    !ggml_gallocr_reserve(gallocr_, graph_) ||
    !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
    release_graph();
    throw std::runtime_error("failed to allocate WavLM graph tensors");
}
```

与 HuBERT `ensure_graph`(hubert_encoder.cpp:553-559)逐字对齐。input tensor
(`input_`/`attention_mask_`/`position_bias_`/`token_mask_`/`first_conv_*`)都标了
`ggml_set_input`,gallocr 会给它们分配持久 buffer 且不重用,运行时 `write_tensor_f32`
照常工作(已验证)。

## 验证

CPU backend,35.7 s recorded.wav,VC with voice_ref = 同一文件:

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| WavLM graph buffer | 15.05 GiB | **0.82 GiB** |
| content_encoder graph | 0.19 GiB | 0.19 GiB(不变) |
| wave_decoder graph | 0.13 GiB | 0.13 GiB(不变) |
| 进程峰值 WS | 17.61 GiB | **3.14 GiB** |
| 输出 n_samples | 1574370 | 1574370 |
| 输出 max_abs | 0.4880 | 0.4880 |
| 输出 非零比例 | 1573140/1574370 (99.9%) | 1573140/1574370 (99.9%) |

输出音频数值完全一致(gallocr 只改 backing-buffer 管理,不动任何计算 op 或权重)。
`audiocpp_write_wav` 落盘 35.70 s @44.1k 有效 WAV。

## 与 indextts2 over-commit 的关系

用户最初怀疑"和之前 indextts2 的 OOM 同因"。

- **相同**:都是结构性的过度分配(over-commit),都不是模型本身大、都不是数据 bug。
- **不同**:
  - indextts2(commit `5d728bc`)是 `BackendWeightStore` 把整个 `context_bytes`
    当 no_alloc 元数据池整块 commit,7 个 runtime × 4096 MB = 28 GB 私有 commit 打爆
    **系统提交上限**(主机内存),修法是 cap metadata pool 到 16 MB。
  - 本次是 WavLM 图用 `alloc_ctx_tensors` 给每个中间张量独立 buffer,12 层 attention
    scores 物化不重用,打爆 **VRAM**(GPU 显存),修法是改用 gallocr 做 lifetime 重用。

两者都是"某层结构按 N² 或 ×runtimes 无界放大 + 分配策略不重用"导致的 over-commit,
方向一致,但具体机制和修复点不同。

## 后续建议(未实施,仅记录)

1. **本修复已足够让 miocodec VC 在 11 GiB 空闲 VRAM 上跑 35 s 音频**(峰值 ~3 GiB)。
   但 WavLM 的 attention 仍是 O(N²) 的物化 scores;若要支持数分钟级超长音频,可考虑:
   - 给 WavLM 图容量加 tiered 上限(复用 `GraphCapacityController`)。
   - 对超长音频在 MioCodec session 的 source/voice_ref 两侧做窗口分块(需处理 VC 跨块
     音色拼接边界)。
2. `MioCodecConfig` 里的 `*_max_seq_len` 字段(`assets.h:31-51`)是死字段——只在
   `assets.cpp` 解析,运行时从不引用、不做任何封顶。若加超长保护,可让它在
   `ensure_graph` 前对 `stft_frames`/`ssl_frames` 做上限校验。

## 复现脚本(留档)

诊断版 dll 在 `build/diag-cpu/bin/libaudiocpp.dll`(mingw toolchain)。复现:

```python
# 见本机 /tmp/repro_vc_tasklist.py(内存采样)和 /tmp/repro_verify.py(输出校验)
# 关键:lib = ctypes.WinDLL(libaudiocpp.dll),backend=CPU(0)
# model=miocodec, task=VC(7)
# audio_transform_with_voice_ref(mono, mono_n, 16000, b"{}", mono, mono_n, 16000)
```
