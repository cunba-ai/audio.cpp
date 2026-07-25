# 离线进度回调(全 37 模型)+ 修两个 C ABI bug

## 一、修两个 C ABI bug(独立小改动)

### Bug 1:`audiocpp_stream_pull` 的 `timeout_ms` 未实现
- 位置:`capi/src/audiocpp_capi.cpp:1157` —— `(void)timeout_ms;` 直接忽略,但头文件声称支持 `-1/0/>0`
- **修复**:底层 `next_stream_event()` 是同步内联调用,无法真做超时。务实做法:**修正头文件文档**明确当前为同步阻塞,`timeout_ms` 保留为预留参数(未来加后台线程再启用),实现里校验值(`-1` 或 `>=0` 通过,其他报错),消除"声称支持但没实现"的歧义。不删参数(避免 ABI break)。

### Bug 2:Nemotron ASR partial-text 事件泄漏
- 位置:`audiocpp_stream_finish`(`capi/src/audiocpp_capi.cpp:1116`)调 `finish_stream()` 后,`set_stream_event_sink(nullptr)` 并丢弃 `stream->sink_events`,Nemotron 在 `finalize()` 内往 sink 推的 partial-text 调用方拿不到
- **修复**:清 sink 前先把 `sink_events` 里尚未消费的事件合并进最终结果(`out_text` 为空则用最后一条 partial_text 填充),然后清空 `sink_events` 避免重复消费

---

## 二、离线 run() 进度回调(全 37 模型覆盖)

### 关键架构发现:统一基类 `RuntimeSessionBase`
全部 37 个 offline session 都继承 `engine::runtime::RuntimeSessionBase`(`include/engine/framework/runtime/session_base.h`)。回调成员和发射辅助**放在基类一次定义,所有模型自动获得能力**,不用改 37 个头文件。

### 设计:基类持有回调 + 辅助方法,各模型 run() 循环插桩调用

**粒度:块级 + 支持取消**
- 有块循环的 21 个模型(TTS 文本块 / ASR 音频块 / 分离音频块):每块边界回调,`进度 = 已完成块 / 总块`,总块数循环前已知(全是 sized vector)
- 单次推理的 16 个模型:run() 开头发 `0.0`、结束前发 `1.0`(简单 start/end 信号)
- 回调返回 `bool`,返回 `false` 则 `run()` 抛 `ProgressCanceled` 中止

### 改动清单

**1. `include/engine/framework/runtime/session.h` —— 接口层**
- 新增进度载体和回调类型:
  ```cpp
  struct ProgressInfo {
      float progress;           // [0.0, 1.0]
      std::string stage;        // "qwen3_tts"/"qwen3_asr" 等家族名
      int64_t completed_units;  // 已完成块数
      int64_t total_units;      // 总块数
  };
  using ProgressCallback = std::function<bool(const ProgressInfo &)>;
  ```
- 新增取消异常:
  ```cpp
  struct ProgressCanceled : std::runtime_error {
      using std::runtime_error::runtime_error;
  };
  ```
- 在 `IVoiceTaskSession`(共享基类)加默认 no-op 虚函数:
  ```cpp
  virtual void set_progress_callback(ProgressCallback cb) { (void)cb; }
  ```

**2. `include/engine/framework/runtime/session_base.h` + `src/framework/runtime/session_base.cpp` —— 基类持有回调**
- `RuntimeSessionBase` protected 区加:
  ```cpp
  ProgressCallback progress_callback_;              // 成员
  void emit_progress(const char *stage, int64_t completed, int64_t total);
    // 空 callback 直接 return;否则构造 ProgressInfo 调用,返回 false 时抛 ProgressCanceled
  ```
- override `set_progress_callback` 存成员(基类提供实现,所有子类自动有)

**3. 21 个块循环模型 —— run() 循环插桩(全部覆盖,不只 5 个)**

每类改动模式相同(循环改索引式 + 进循环前/每块后调 `emit_progress`):

| 分组 | 模型(循环位置) |
|---|---|
| TTS 文本块(16) | qwen3_tts(3 个变体循环)、pocket_tts(generate()内)、chatterbox、fish_audio、index_tts2、irodori_tts、higgs_audio_tts、moss_tts_local、moss_tts_nano、omnivoice、voxcpm2(run_offline_request内)、supertonic、vevo2、miotts(两段循环)、vietneu_tts、outetts(pending deque,用 initial_count) |
| ASR 音频块(4) | qwen3_asr、higgs_audio_stt、vibevoice_asr、hviske_asr |
| 分离音频块(2) | demucs、roformer |
| 条件循环(1) | heartmula(infinite_mode 路径) |

**4. 16 个单次推理模型 —— start/end 信号**
run() 开头 `emit_progress(family, 0, 1)`、return 前 `emit_progress(family, 1, 1)`:
glm_tts、ace_step、stable_audio(两个)、seed_vc、vibevoice、miocodec、nemotron_asr、citrinet_asr、voxtral_realtime、qwen3_forced_aligner、silero_vad、marblenet_vad、sortformer_diar、heartmula(非 infinite 路径)

**5. C ABI 暴露(`capi/include/audiocpp.h` + `capi/src/audiocpp_capi.cpp`)**
- 新增 C 回调签名:
  ```c
  typedef int (*audiocpp_progress_fn)(float progress, const char *stage,
                                       int64_t completed, int64_t total, void *user);
  // 返回 0=继续,非 0=取消
  ```
- 新增 setter(不改现有 run 函数签名,零 ABI break):
  ```c
  AUDIOCPP_API void audiocpp_set_progress_callback(
      audiocpp_model_t *model, audiocpp_progress_fn fn, void *user_data);
  ```
- `audiocpp_model` 内部结构加 `progress_fn` + `progress_user` 两成员(此结构未暴露头文件,无 C ABI break)
- 每个 `audiocpp_tts`/`asr`/`vad`/`diar`/`align`/`audio_transform`/`transform_stems` 在 `offline->run()` 前,若有 callback 则包装成 `std::function` 调 `set_progress_callback`,run 后清空(try/catch 保证异常路径也清)
- catch `ProgressCanceled` → 设 `err->message = "canceled by progress callback"`,返回 null(不视为 fatal)
- `.def`/version-script 加 `audiocpp_set_progress_callback`

**6. CLI 验证入口(可选)**
`app/cli/main.cpp` 加 `--show-progress` 标志,注册一个打印百分比的 callback,用于手动验证

---

## 三、验证
1. 编译:`engine_runtime` + `audiocpp` DLL 通过(MSVC/MinGW),-Wall -Wextra -Wpedantic 无警告
2. 新增 `tests/unittests/test_progress_callback.cpp`:注册取消 callback,验证 run() 抛 `ProgressCanceled` + 已完成块数正确
3. 扩展 `tests/test_capi_load.c`:注册计数 callback,跑短文本 TTS,验证回调次数 = 块数、progress 单调递增
4. 手动:本地跑长文本 qwen3_tts + 长音频 qwen3_asr,确认进度实时回调

## 四、改动文件汇总(~45 个,但核心逻辑集中)
- `include/engine/framework/runtime/session.h`(ProgressInfo + 虚函数 + 异常)
- `include/engine/framework/runtime/session_base.h` + `src/framework/runtime/session_base.cpp`(成员 + emit_progress 辅助)
- 21 个块循环模型的 `session.cpp`(每个循环插桩,头文件不改——成员在基类)
- 16 个单次推理模型的 `session.cpp`(start/end 各一行)
- `capi/include/audiocpp.h` + `capi/src/audiocpp_capi.cpp`(setter + 包装 + 修两个 bug)
- `CMakeLists.txt`(.def 加导出)
- `tests/unittests/test_progress_callback.cpp`(新)+ `tests/test_capi_load.c`(扩展)
- `app/cli/main.cpp`(可选 --show-progress)

样板代码集中在基类的 `emit_progress`,各模型 run() 每个循环只加 2-3 行调用。所有模型默认继承能力,不 override 的也兼容。