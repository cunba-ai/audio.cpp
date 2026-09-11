## 目标
让 CAPI 能用上 PR #16 带来的两个"非音频/带 artifact"模型:**MuScriptor(文本→MIDI)** 和 **MiniMax-H3(文本→音频+可选视频)**,并顺手修掉合并引入的 artifact 枚举错位。框架头改动对 CAPI 都是加性的,编译兼容性已确认无需改动。

## 已确认的设计决策
- `audiocpp_generate` **只收文本 prompt**,不加音频入参(用户已确认)。
- MuScriptor 用 `AUDIOCPP_TASK_MIDI` 加载;MiniMax-H3 用 `AUDIOCPP_TASK_GEN` 加载。
- 一个 `audiocpp_generate` 入口返回 `{audio, artifacts}` 组合 —— MiniMax 一次 `run()` 同时产出 PCM + 可选视频,绝不能跑两次生成;MuScriptor 同入口产出 artifact、audio 为空。

## 关键设计点

1. **新增 `AUDIOCPP_TASK_MIDI = 13`**(接 SVC=12 后),`map_task` 映射到 `VoiceTaskKind::Midi`(MuScriptor 构造硬性要求)。

2. **artifact 枚举纠偏**(`audiocpp.h:755-764`):插入 `AUDIOCPP_ARTIFACT_MIDI = 4`,把 `ALIGNMENT→5 / DIAR→6 / VAD→7 / CUSTOM→8`,恢复 header 自己声称的"mirrors engine::runtime::ArtifactKind"契约。
   - 安全性:这组常量当前是死代码(无任何 CAPI 路径按数值读写 kind;`audiocpp_artifact_create` 文档标注 reserved/未用;已知 Rust 消费方 sound-server 不碰 artifact)。纯契约修正,会在 commit 里标注。

3. **复用 `audiocpp_artifact_t` 作输出元素**:其字段(kind/id/payload/payload_size/n_meta/meta_keys/meta_values)与 runtime `VoiceArtifact`(kind/id/payload[vector<byte>]/meta[map])1:1,直接当输出,不另造结构。

## 改动清单

### A. `capi/include/audiocpp.h`
- task 枚举加 `AUDIOCPP_TASK_MIDI = 13`(注释:音乐生成/MIDI,MuScriptor)。
- artifact 枚举:插入 `AUDIOCPP_ARTIFACT_MIDI = 4` 并顺延后继值;更新该 section 注释(改为说明 MuScriptor/MiniMax-H3 会产出 artifact)。
- 更新 `audiocpp_artifact_t` 文档注释(现也作为 `audiocpp_generate` 的输出元素)。
- 新增 typedef:
  ```c
  typedef struct { audiocpp_artifact_t *artifacts; int64_t n_artifacts; } audiocpp_artifacts_t;
  typedef struct { audiocpp_audio_t *audio; audiocpp_artifacts_t *artifacts; } audiocpp_gen_result_t;
  ```
- 声明三个新函数(均 `AUDIOCPP_API`,自动导出,无需改 .def):
  - `audiocpp_gen_result_t *audiocpp_generate(model, prompt, options_json, err)`
  - `void audiocpp_free_gen_result(audiocpp_gen_result_t*)`
  - `void audiocpp_free_artifacts(audiocpp_artifacts_t*)`

### B. `capi/src/audiocpp_capi.cpp`
- `map_task`(L205-222)加 `case AUDIOCPP_TASK_MIDI: return VoiceTaskKind::Midi;`
- 在 `audiocpp::detail`(pack_audio_output 旁,~L345)加两个 helper:
  - `pack_artifact_into(const VoiceArtifact&, audiocpp_artifact_t&)`:填 kind、`dup_cstr(id)`、`malloc+memcpy` payload、meta 的 key/value 各自 `dup_cstr` 成平行数组。
  - `collect_artifacts(const TaskResult&) -> audiocpp_artifacts_t*`:合并 `[artifact_output(若有)] + output_artifacts`(与 app/server/runtime.cpp:614-615 顺序一致)。
- 抽出 `free_artifact_members(audiocpp_artifact_t*)` 静态 helper,共享给 `audiocpp_artifact_free` 与新 `audiocpp_free_artifacts`;`audiocpp_artifact_free` 改为先调 helper 再 `delete`。
- 实现 `audiocpp_generate`:仿 `audiocpp_tts` 骨架(result 声明在 AUDIOCPP_CATCH 外 → 校验 model → 建 TaskRequest、text_input=prompt → apply_options → prepare → run → `pack_audio_output` 取 audio_output、`collect_artifacts` 取 artifacts → 装进 gen_result 返回)。
- 实现 `audiocpp_free_gen_result`(free audio + artifacts + wrapper,均 NULL-safe)和 `audiocpp_free_artifacts`(逐元素 free_artifact_members → free 数组 → delete wrapper)。

### C. 文档/元数据
- `AGENTS.md`:把"Exposes 37 functions"更新为新数(40)。
- `capi/README.md`:加一节 `audiocpp_generate`,给 MuScriptor(MIDI)和 MiniMax-H3(音频+`return_video` 视频)的 options 示例。

## 验证
1. **CMake configure + 编译 CAPI**:`-DAUDIOCPP_BUILD_CAPI=ON`,编通 `audiocpp_capi.cpp` 这个 TU 并链接 `libaudiocpp`(本机 `-j2`,只盯 capi 目标)。
2. **导出符号检查**:确认三个新函数出现在导出表(沿用现有 dumpbin/nm 方式)。
3. (可选,需模型权重)最小 C 客户端跑 MuScriptor→MIDI 落盘 `.mid`,字节与 `app/workflow/file_sink.cpp` 落盘路径一致;权重不可用则跳过。

## 不做
- 不改 server 端(原生 WebUI、模型卸载在 `app/server`,与 CAPI 共享库无关)。
- 不碰框架高风险内部模块。
- 不为 `audiocpp_generate` 加音频输入变体(用户已确认)。
- 不改 `map_task` default 回退 Tts 的旧行为。