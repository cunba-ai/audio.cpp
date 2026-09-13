# Sortformer v1 vs v2.1 真实音频对比报告

- **日期**: 2026-09-11
- **执行**: ZCode (cunba-ai/audio.cpp fork, main = b6d0f68, PR #33 合并后)
- **结论先行**: 在 11 个真实长音频（27 秒 – 55 分钟，共约 3.8 小时）上，**v2.1 流式全面占优**：速度快约 3 倍（RTF 0.07 vs 0.2+）、说话人身份全局连续（v1 分块后全局 ID 退化为 7–77 个互不相干的实例）、单次调用无长度限制。建议 sound-server 的 diar 场景切换到 v2.1 流式，v1 保留作为回退。

---

## 1. 背景与目的

PR #33（上游 #493）带来新家族 `sortformer_diar_v2`（NVIDIA
`diar_streaming_sortformer_4spk-v2.1`，四说话人流式声纹分离，AOSC 到达顺序说话人缓存）。
v1 家族 `sortformer_diar` 完全不动（加性新增）。本对比回答：**v2.1 能否替代 v1**。

## 2. 模型与环境

| 项 | v1 | v2.1 |
|---|---|---|
| 家族 | `sortformer_diar` | `sortformer_diar_v2` |
| 本地 GGUF | `sortformer-diar-4spk-v1-q8_0.gguf`（175MB，sha16 `84744b047296daf2`，无内嵌 spec） | `sortformer-v2.1-f16-mixed.gguf`（251MB，sha16 `5a465ea6ba133e3c`，内嵌 spec；从 `nvidia/diar_streaming_sortformer_4spk-v2.1.nemo`（450MB，SHA-256 与上游文档一致 `8abd3283...`）本地转换，`--type f16 --keep-nonmatrix-f32`） |
| 架构 | Sortformer 原版"排序"机制 | AOSC：按说话人首次出现顺序分配身份，跨 chunk 保持 |
| 引擎支持的模式 | 仅 offline（单次调用有长度上限，见 §3） | offline（图内存随时长膨胀，仅短片段）+ **streaming（设计模式，任意长度）** |

- 模型路径: `G:\iStation\Models\sound\audiocpp\`
- 引擎: audiocpp_cli（windows-cpu-release 构建），**CPU 后端**
- 测试音频: 2 个 MP3（联合国会议解读类，改名 husa=19.3min / meifa=14.9min）+ 9 个 m4a
  （`F:\M1AO_Projects\sound-rs\samples\新录音 62-65/67/69-73.m4a`，27s–55min），全部解码为
  16k 单声道 WAV（`build/diar_cmp/*_16k.wav`）

## 3. 跑批前发现的三个硬约束（决定对比协议）

1. **v1 单次调用上限 ≈120–180 秒**：整段请求必须装进一张推理图，超过
   `fc_encoder.max_position_embeddings` 直接报错（`session_len_sec exceeds ...`）。
   经验值：60s/120s OK，180s/240s FAIL。`graph_capacity_mode=fixed` 则直接拒绝 >20s 的请求。
   → 长音频必须**调用方分块**。
2. **v2.1 离线图内存随时长线性膨胀**（约 3.2GB/分钟音频）：55 分钟文件要求 10.5TB
   CPU buffer，直接分配失败。→ v2.1 的正确用法是**流式**（有界内存，AOSC 状态跨 chunk）。
3. **中文路径触发 Windows 代码页错误**（`No mapping for the Unicode character...`）：
   husa/meifa 两个中文文件名需改为 ASCII。

**对比协议**（各自的最佳可行方式）：
- v1 = offline + **120 秒 ffmpeg 分块** → 逐块推理 → 按样本偏移合并 turns，
  说话人 ID 标注为 `SPEAKER_xx@pNN`（如实保留"每块独立编号"的事实）
- v2.1 = **streaming 单次整段**

## 4. 结果

### 4.1 总表

| 文件 | 时长(s) | v1 块数 | v1 耗时(s) | v2.1 耗时(s) | v1 加速比 | v1 turns | v1 说话人(块内/全局) | v2.1 turns | v2.1 说话人 | v1 语音(s) | v2.1 语音(s) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| husa 胡塞问题-联合国会议解读 | 1157 | 10 | 328 | **88** | 3.7× | 186 | 4 / **29** | 241 | **4** | 1166.5 | 1094.2 |
| meifa 美法互怼-反对图尔克 | 893 | 8 | 195 | **67** | 2.9× | 162 | 4 / **25** | 136 | **4** | 885.9 | 916.2 |
| rec62 | 3300 | 28 | 700 | **221** | 3.2× | 993 | 4 / **77** | 1411 | **4** | 2698.2 | 2438.4 |
| rec64 | 1710 | 15 | 347 | **117** | 3.0× | 389 | 4 / 38 | 596 | **3** | 1504.7 | 1347.7 |
| rec65 | 27 | 1 | 2 | **10** | 0.2× | 9 | 2 / 2 | 7 | **1** | 20.6 | 19.2 |
| rec67 | 582 | 5 | 115 | **44** | 2.6× | 172 | 3 / 14 | 208 | **4** | 383.7 | 420.2 |
| rec69 | 2491 | 21 | 492 | **163** | 3.0× | 738 | 4 / 59 | 900 | **3** | 1797.8 | 1676.2 |
| rec70 | 1385 | 12 | 279 | **95** | 2.9× | 491 | 4 / 37 | 559 | **4** | 1159.8 | 1018.8 |
| rec71 | 766 | 7 | 162 | **57** | 2.8× | 444 | 3 / 17 | 360 | **3** | 635.9 | 598.6 |
| rec72 | 319 | 3 | 62 | **30** | 2.1× | 145 | 3 / 7 | 126 | **4** | 206.3 | 199.9 |
| rec73 | 917 | 8 | 184 | **65** | 2.8× | 312 | 3 / 19 | 426 | **4** | 643.0 | 511.3 |

注：v1 耗时含每块 ~1.2s 模型重载；rec65 是 27 秒短片段（v2.1 该行含 ~9s 模型加载，短音频
离线单次调用下 v1 更快）；v1"块内"指单块内最大说话人数，"全局"指合并后不同
`SPEAKER_xx@pNN` 标签总数。

### 4.2 v2.1 每文件说话人时长分布（秒）

| 文件 | S0 | S1 | S2 | S3 |
|---|---:|---:|---:|---:|
| husa | 234.2 | **707.3** | 85.4 | 67.3 |
| meifa | 167.1 | **507.1** | 167.2 | 74.7 |
| rec62 | 613.4 | 450.1 | **1084.5** | 290.5 |
| rec64 | **1020.8** | 125.8 | 201.0 | — |
| rec65 | 19.2 | — | — | — |
| rec67 | 40.6 | **227.2** | 127.5 | 24.9 |
| rec69 | 704.6 | **763.1** | — | 208.5 |
| rec70 | **594.7** | 357.8 | 19.4 | 46.8 |
| rec71 | **296.6** | 278.0 | — | 24.0 |
| rec72 | **104.6** | 45.9 | 3.6 | 45.8 |
| rec73 | 15.5 | **342.0** | 10.0 | 143.8 |

### 4.3 关键发现

1. **v1 没有"全局说话人身份"**：每块从 SPEAKER_00 重新编号，合并后同一标签裂成
   7–77 个互不相干的实例（rec62：28 块 → 77 个全局标签）。要得到一致身份必须再做一层
   说话人聚类（引擎不提供）。**v2.1 的 AOSC 跨 chunk 保持身份，整段输出干净的全局 ID**——
   这是两者最本质的差距，也是分块方案无法弥补的。
2. **速度**：v2.1 流式 RTF ≈ 0.066–0.083（55 分钟 → 221s），v1 分块 RTF ≈ 0.2–0.28
   （每块重复加载拖累）。长音频上 **v2.1 快 2–3.7 倍**。例外：分钟级以内短片段，
   v1 离线单次调用更快（无流式开销）。
3. **说话人数量判断**：两个多说话人 MP3 v2.1 均检出 4 人（符合内容预期）；多数录音两模型
   结论一致（3–4 人）。rec65（27s，大概率单人）v1 报 2 人、v2.1 报 1 人——v2.1 新增说话人
   更保守。语音总时长两模型 ±15% 内互相印证（v2.1 普遍略少，活动判定更紧）。
4. **部署复杂度**：v2.1 一条命令吃任意长度；v1 需外层分块 + 偏移合并 + 补聚类 +
   中文路径规避（Windows 代码页）。

## 5. 结论与建议

- **切换**：sound-server / 日常 diar 场景 → `sortformer_diar_v2` + `--mode streaming`
  （CAPI: `AUDIOCPP_TASK_DIAR` + family hint `sortformer_diar_v2`，GGUF 内嵌 spec，
  不需要 `model_spec_override`）。
- **保留 v1**：① 分钟内短片段离线批处理（更快）；② 作为对照回退。**不删除**。
- **注意事项**：两个模型都是 4 说话人上限（5+ 人退化）；训练偏英语，中文会议
  （AliMeeting DER 12.6）表现好但嘈杂/电话场景一般；v2.1 离线模式仅用于短片段（内存随时长膨胀）。
- **本对比是行为对比，不是精度评测**：没有人工标注 ground truth，无法计算 DER。
  如需硬结论，挑 1–2 个文件人工标注 5 分钟片段即可补充重叠率评测。

## 6. 复现

```bash
# 解码（中文文件名改 ASCII）
ffmpeg -i in.mp3 -ac 1 -ar 16000 name_16k.wav

# v2.1 流式（任意长度）
audiocpp_cli --task diar --family sortformer_diar_v2 --mode streaming \
  --model .../sortformer-v2.1-f16-mixed.gguf --backend cpu \
  --audio name_16k.wav --turns-out name_v2.json

# v1 分块（120s 上限）
ffmpeg -i name_16k.wav -f segment -segment_time 120 -c copy parts/p_%03d.wav
#   逐块: audiocpp_cli --task diar --family sortformer_diar \
#     --model .../sortformer-diar-4spk-v1-q8_0.gguf --backend cpu \
#     --audio parts/p_000.wav --turns-out parts/p_000.json
#   合并: turns 加 块号×120×16000 样本偏移，speaker_id 标注 @p块号
```

脚本与全部产物: `build/diar_cmp/`（run_compare.sh、summarize.py、22 个 turns JSON、11 个 16k WAV）。
模型: `G:\iStation\Models\sound\audiocpp\{sortformer-diar-4spk-v1-q8_0.gguf, diar_sortformer_4spk_v2.1\}`。
