# 修复 Qwen3-TTS Base 变体采样失控(EOS 不收敛)

## 根因(已实测 + 上游确认)
Qwen3-TTS **Base** 变体(0.6B/1.7B)在官方默认采样参数(`top_p=1.0, temp=0.9, top_k=50`)下,~40-50% 的生成轨迹无法产生 EOS token,跑到 `max_new_tokens=8192` 上限,产出超长音频(655s)+ 内存暴涨到 OOM。这是**上游已知缺陷**([QwenLM/Qwen3-TTS#118](https://github.com/QwenLM/Qwen3-TTS/issues/118),官方 Closed as not planned),audio.cpp 忠实复现了官方默认参数故忠实复现了该问题。CustomVoice/VoiceDesign 变体 EOS 稳定,不受影响。

实测关键证据:
- 失控后端无关(CPU 13/30、CUDA 12/30 失控率相近),非 CUDA 精度/RNG bug
- `do_sample=false`(argmax)×3 100% 正常
- `top_p=0.8` 对所有测到的失控 seed 均修复;加大 top_k 无效
- 采样参数与 HF 官方 `generation_config.json` 完全一致

## 修复策略:Base 变体双层缓解(保留采样多样性 + 硬上限兜底)

**只针对 `Qwen3TTSVariant::Base`,不影响 CustomVoice/VoiceDesign。** 改动集中在 qwen3_tts 家族内部,无跨模型影响(已确认 Qwen3TTSGenerationOptions 不被其他家族共享)。

### 改动 1:收紧 Base 变体默认 top_p(治本,主防线)
**文件**:`src/models/qwen3_tts/session.cpp` — 函数 `generation_options_from_request`(line 29-80)

在函数末尾、`return options;` 前,加入 Base 变体的默认值收敛逻辑:
- 仅当 `config.variant == Qwen3TTSVariant::Base` 时生效
- 仅当**调用方未显式设置** `top_p`(即请求里没有该 option)时,把默认从结构体的 `1.0` 收敛到 `0.8`(实测 0.8 对所有失控 seed 有效)
- 同理把默认 `temperature` 从 `0.9` 降到 `0.8`(配合 top_p 收紧,进一步稳定分布;仍保留采样)
- 用户通过 request option 显式传 `top_p`/`temperature` 时**完全覆盖**,不强制

判定"用户是否设置":现有代码用 `parse_float_option` 返回 optional。引入局部标记(如先记录"是否已 parse 到 top_p"),Base 分支下仅对未设置的项收敛默认。这要求小幅重构该函数,把 top_p/temperature 的解析结果与"是否被请求设置"分开跟踪。

### 改动 2:Base 变体 max_new_tokens 硬上限(兜底,防 OOM)
**文件**:`src/models/qwen3_tts/session.cpp` — `generation_options_from_request`(同一函数,return 前)

- 仅当 `config.variant == Base`:对 `options.max_new_tokens` 取 `min(options.max_new_tokens, kBaseMaxNewTokensCap)`
- `kBaseMaxNewTokensCap` 定义为常量(建议 1024,在匿名 namespace 内)——理由:正常 TTS 短句 < 100 步(12Hz codec,10s 语音≈120 步),1024 是宽松上限;即便采样仍偶发失控,最坏只跑 1024 步(≈85s 音频),内存/CPU 远不到 OOM,且 120s curl 超时前能返回
- 同样:用户显式传更小的 max_tokens 时尊重之;只做"封顶",不做"放大"

> 注意:`generation_capacity_`(KV cache 容量)仍按 `assets_->config.max_new_tokens`(8192)在 session.cpp:282 预分配,本改动不缩减预分配内存(改动 1 是主防线,正常情况下不会跑到上限;预分配大一点不影响正确性,只是峰值内存上限未变——但实际峰值由"跑了多少步"决定,loop 上限降到 1024 后实际峰值内存大幅下降)。若后续要进一步降峰值,可单独再裁 generation_capacity_,本次不做以免触及 step_runtime 容量边界。

### 改动 3:测试(回归保障)
**新文件**:`tests/test_capi_tts_clone.c`(已存在,本次扩展)

- 已有的 CAPI 测试工具支持 `AUDIOCPP_SEED`/`AUDIOCPP_TOP_P`/`AUDIOCPP_MAX_TOKENS` 环境变量。在 V100 上用它跑固定 seed=1(修复前稳定失控)验证:修复后默认参数下不再失控
- 由于这是依赖模型文件 + GPU 的集成行为(非纯单元测试),不强行塞入 ctest 自动化(无模型文件会失败)。改为:在 qwen3_tts 目录加一个说明性 `RUNAWAY_REPRO.md`(或并入现有 docs),记录复现命令 + 期望(修复后 seed 1-30 失控率应降至 ≤5%,理想 0)

### 验证步骤(修复后)
1. 本机重编 audiocpp_cli + audiocpp.dll(已有 build/windows-cuda-release)
2. scp 到 V100,用 seed sweep(`sweep.py`)跑 1-30 × {cpu,cuda}
3. 验收:失控率从 ~40% 降到 ≤5%(top_p=0.8 收敛 + max_tokens 兜底)
4. 对比 argmax 输出确认采样路径音质未被破坏(随机抽听几个 out wav)

## 不做什么(及原因)
- **不改采样算法/cuda_fast_path**:已用 Node 数值验证其数学正确(exponential-race 等价于 Gumbel-max),非 bug
- **不加盲目"失控熔断截断"**:生成循环无差别累积音频帧(talker.cpp:1779),无法判定"在哪截"才不破坏有效长句;治本应防失控于未然(top_p),而非事后截断
- **不改 types.h 结构体默认值**:那会影响所有变体;本次只针对 Base,在 session 层按 variant 收敛
- **不动 1.7B/CustomVoice**:它们 EOS 稳定,实测正常,改了反而可能影响其音质

## 涉及文件
- `src/models/qwen3_tts/session.cpp`(改动 1+2,核心)
- `tests/test_capi_tts_clone.c`(已就绪,验证用)
- 新增文档记录复现/验收(改动 3)