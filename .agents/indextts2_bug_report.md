# IndexTTS2 推理初始化崩溃 — 根因已定位并修复

> **状态:已修复(2026-07-28)。**本文档记录调查过程。原始猜测("_aligned_malloc
> 卡 2GB 边界")**被证伪**;真正根因是 `BackendWeightStore` 把整个 `context_bytes`
> 当 no_alloc 元数据池提前 commit,7 个 runtime × 4096MB = 28GB 私有 commit 打爆
> 系统提交上限。详见下方「真正根因」与「修复」。

## TL;DR

IndexTTS2 在 audio.cpp 上无法推理。所有 GGUF 版本(q8_0 / f16 / orig)、所有
backend(CPU / CUDA)一致崩溃,其他模型(voxcpm2 / htdemucs / kokoro 等)正常。

- **崩溃点**:`ggml_aligned_malloc(2147483648)` 返回 NULL →
  `GGML_ASSERT(ctx->mem_buffer != NULL)` abort。请求的是 2GB 的 GPT graph arena。
- **原始猜测**:`_aligned_malloc` 在 INT32_MAX+1(2GB)边界有 bug。
- **实际根因(已用证据证伪原始猜测)**:崩溃前进程已 **commit 28GB 私有内存**
  (`privateUsage`),而机器系统提交上限(phys + pagefile)只有 ~51GB,基线已用 ~25GB,
  余量不足以再 commit 2GB。28GB 来自 7 个 runtime 各自构造的 `BackendWeightStore`,
  每个 store 都把 `context_bytes=4096MB` 当 no_alloc 池**整块 commit**,但实际只存了
  ~几十 KB 的张量元数据(数据在另一块 backend buffer 里)。
- **修复**:`BackendWeightStore` 构造时不再 commit 整个 `context_bytes`,改为
  `min(context_bytes, 16MB)` 的元数据预算。IndexTTS2 的 private commit 从 **26.31GB
  降到 4.88GB**,2GB GPT arena 顺利分配,音频正常生成(4.25s/4.64s 有效 wav)。

---

## 现象

IndexTTS2 在 audio.cpp 上**无法推理**,无论量化档位(q8_0 / f16 / orig 全崩)、无论
backend(CPU / CUDA)、无论 GGUF 来源(官方 `audio-cpp/audio.cpp-gguf` repo)。

- **`audiocpp_cli`**:`bad allocation`(顶层捕获)或带诊断版直接 abort
- **经 sound-rs / sound-server(audiocpp.dll)**:
  `GGML_ASSERT(ctx->mem_buffer != NULL) failed at ggml.c:1611` → 整个进程 abort

其他模型(voxcpm2 / htdemucs / kokoro 等)在同一环境、同一 CLI、同一 backend 下
**完全正常**。问题仅出现在 IndexTTS2。

## 环境

- **GPU**:Tesla V100-SXM2-32GB(compute capability 7.0,32GB VRAM 全空闲)
- **RAM**:47.15GB 总量,36.3GB 空闲
- **OS**:Windows 11,64 位
- **系统提交上限(关键)**:`TotalPageFile = 51.20 GB`(phys 47GB + ~4GB pagefile)
- **audio.cpp 构建**:`windows-cpu-release`(MSVC /O2,x64,portable profile)
- **GGUF**:全部来自官方 repo
  - `index-tts2-q8_0.gguf`(3465 MB) — 本次验证主用
  - `index-tts2-f16.gguf`(4432 MB)
  - `index-tts2-orig.gguf`(7710 MB)

## 调查方法(systematic debugging,四阶段)

### 阶段 1:证伪原始假设

原始 bug 报告的核心断言是"`_aligned_malloc(2147483648)` 在 2GB 边界返回 NULL"。
**先用独立探针在同一台机器上验证这个断言**(而不是直接相信):

在 5080 机器上跑独立 ctypes 探针,逐个 size 测 `_aligned_malloc` 和 `VirtualAlloc`:

```
== _aligned_malloc (CRT heap) — in fresh python process ==
  _aligned_malloc(   20971520 B =  0.02 GB) -> OK
  _aligned_malloc( 268435456 B =  0.25 GB) -> OK
  _aligned_malloc(2147483648 B =  2.00 GB) -> OK      ← 原报告说这里该 NULL,实际 OK
  _aligned_malloc(2269357184 B =  2.11 GB) -> OK
  _aligned_malloc(4294967296 B =  4.00 GB) -> OK
== VirtualAlloc 同样全部 OK ==
== 系统状态:availPhys=36.43GB availVirtual=131067GB ==
```

**结论:原始根因假设错误。**`_aligned_malloc` 在干净进程里能正常分配 2GB / 2.11GB /
4GB。崩溃是**进程状态相关**的 —— audio.cpp 进程在崩溃前做了某些事,把 commit
额度耗光了。报告里"加 `_aligned_malloc` fallback 到 VirtualAlloc"的修复方向因此
**不会生效**(两者底层走同一个 commit 机制)。

### 阶段 2:在 audio.cpp 内部定位 commit 去向

给 audio.cpp 三层加 stderr 诊断并重新编译,逐层定位:

**第 1 层 — 模型加载(`assets.cpp::load_index_tts2_assets`)**:9 个 tensor source
全部加载 OK。

**第 2 层 — session 构造(`session.cpp::IndexTTS2Session`)**:7 个 runtime
(semantic_encoder / semantic_codec / style_encoder / gpt / s2mel / vocoder /
qwen_emotion)全部构造 OK。

**第 3 层 — 在每个 runtime 构造边界打印进程 private commit**(关键):commit 随
runtime 构造**线性飙升**,远超 working set:

| 阶段 | privateCommit | workingSet | sys availCommit(pagefile) |
|------|--------------|-----------|---------------------------|
| before-runtime-build | 0.02 GB | 0.01 GB | 26.43 GB |
| after semantic_encoder (Wav2Vec2Bert) | **4.80 GB** | 0.80 GB | 21.62 GB |
| after semantic_codec | **8.86 GB** | 0.85 GB | 17.56 GB |
| after style_encoder | 9.42 GB | 0.91 GB | 17.00 GB |
| after gpt | **15.91 GB** | 3.38 GB | 10.53 GB |
| after s2mel | **20.11 GB** | 3.57 GB | 6.31 GB |
| after vocoder (BigVGAN) | 21.53 GB | 3.99 GB | 4.90 GB |
| after qwen_emotion | **26.31 GB** | 4.77 GB | **1.11 GB** |
| **失败(2GB GPT arena)** | 28.36 GB | 4.77 GB | **耗尽** |

`privateCommit`(28GB)是 `workingSet`(4.77GB)的 **6 倍** —— 23GB 被 commit 但
从未真正写入。系统提交余量从 26GB 跌到 1.11GB,再请求 2GB 就超限了。

### 阶段 3:为什么 commit 这么高 —— 锁定 BackendWeightStore

读代码定位 commit 来源:

1. 每个 runtime 在 `load_*_weights`(如 `gpt.cpp:916`)里 `weights->store =
   std::make_shared<BackendWeightStore>(backend, ..., weight_context_bytes)`,
   `weight_context_bytes` 默认 **4096MB**。
2. `BackendWeightStore` 构造函数(`backend_weight_store.h:34`)直接:
   ```cpp
   ggml_init_params params{context_bytes, nullptr, true};  // no_alloc=true
   ctx_.reset(ggml_init(params));
   ```
3. `ggml_init` → `ggml_aligned_malloc(mem_size)` 把整块 `context_bytes`
   **commit 出来**(`_aligned_malloc` 在 Windows 走 CRT heap,会 commit 整块)。
4. 但 `no_alloc=true` 意味着这个池**只存张量元数据**(`ggml_tensor` 结构体,
   每个约 120 字节);张量**数据**在 `upload()` 里由
   `ggml_backend_alloc_ctx_tensors` 分配到**另一块** backend buffer。

**所以 7 个 store × 4096MB = 28GB commit,实际只用来存 ~几十 KB 元数据 ——
over-commit 约 5 万倍。**这与观测到的 `privateCommit=26.31GB` 吻合。

对照:voxcpm2 只有 1 个 store(且 arena 小),不会触发;flashsr 直接传 32MB
(`flashsr.cpp:428`)。

### 阶段 4:修复 + 验证

**假设**:`BackendWeightStore` 不应在 no_alloc 模式下 commit 整个 `context_bytes`;
改为 commit 一个足够大的元数据预算即可消除 over-commit,2GB GPT arena 重新可分配。

**验证(failing test 优先)**:新增 `test_backend_weight_store_commit.cpp`,构造一个
`context_bytes=512MB` 的 store,断言 private commit delta < 64MB(即不能 over-commit)。
修复前测试**失败**(commit delta = 513MB);修复后**通过**(delta = 16MB)。

**端到端验证(5080 机器,q8_0 GGUF)**:修复后重新跑 IndexTTS2,日志干净(无诊断噪音),
输出 `audio_out=...fix_q8.out.wav`,wav 有效:mono/16bit/22050Hz/**4.25s**(另一句
4.64s)。private commit 从 26.31GB 降到 **4.88GB**。

## 真正根因(一句话)

`BackendWeightStore` 在 `no_alloc=true` 下把整个 `context_bytes`(默认 4096MB)当
元数据池**整块 commit**,但只用来存 ~几十 KB 的 `ggml_tensor` 元数据;IndexTTS2 有
7 个 runtime 各自一个 store,7 × 4096MB ≈ 28GB 私有 commit 打爆 Windows 系统提交
上限(~51GB),导致后续 2GB GPT graph arena 的 `_aligned_malloc` 因无 commit 额度
而返回 NULL,触发 `GGML_ASSERT`。

## 修复

**改动文件(3 个,scoped)**:

1. `include/engine/framework/core/backend_weight_store.h` — 构造函数里把 commit 的
   池大小改为 `min(context_bytes, kMetadataPoolBudget)`,其中
   `kMetadataPoolBudget = 16MB`(足够装 ~13 万个 `ggml_tensor`,远超任何模型)。
   `context_bytes` 参数仍保留以兼容;`make_backend_tensor` 的失败消息也改成指向
   `kMetadataPoolBudget`。
2. `tests/unittests/test_backend_weight_store_commit.cpp` — 新增回归测试。
3. `CMakeLists.txt` — 注册新测试。

**不改 `external/ggml`**(vendored,保持与上游一致;`ggml.c` 本次零功能改动)。
**不改 IndexTTS2 默认 arena**(原始报告的"方向 C"治标方案不需要了 —— 真正的浪费在
框架层,不在 IndexTTS2 配置)。

### 为什么 16MB 预算安全

`no_alloc=true` 池只存 `ggml_tensor` 元数据(`sizeof(ggml_tensor)` ≈ 120 字节 +
object header)。IndexTTS2 GPT 24 层 × 每层若干权重 + 各 encoder/codec,全模型
~500–800 个权重张量 → 元数据 ~60–100KB。16MB 预算是实际需求的 ~150–250 倍,即便
未来出现超深模型也绰绰有余。若真有极端模型把元数据池撑爆,`make_backend_tensor`
会抛清晰异常(指向 `kMetadataPoolBudget`),不会静默崩溃。

## 验证结果汇总

| 验证项 | 结果 |
|--------|------|
| `backend_weight_store_commit_test`(新增回归) | ✅ PASS(commit delta 513MB→16MB) |
| `asr_standalone_gguf_test`(直接用 BackendWeightStore) | ✅ PASS |
| `flashsr_utility_test`(BackendWeightStore,32MB 传入) | ✅ PASS |
| `conv_lowering_matrix_test` | ✅ PASS |
| IndexTTS2 q8_0 端到端(5080) | ✅ 生成 4.25s / 4.64s 有效 wav,private commit 26→5GB |
| voxcpm2 回归(5080) | ✅ 生成 1.12s wav(48kHz) |
| 干净构建(无诊断)重跑 IndexTTS2 | ✅ 日志干净,4.64s wav |

## 复现步骤(修复前)

```powershell
cd D:\Sound
.\audio.cpp\audiocpp_cli.exe --backend cpu --threads 8 --task tts --family index_tts2 `
  --model D:\Sound\models\audiocpp\index-tts2-q8_0.gguf `
  --text "Hello world." --voice-ref D:\Sound\samples\jfk.wav `
  --out D:\Sound\out.wav
# 修复前:GGML_ASSERT(ctx->mem_buffer != NULL) failed / bad allocation
# 修复后:audio_out=...out.wav(有效 wav)
```

## 经验教训

- **不要相信 bug 报告里的根因结论,要用最小探针在同一环境复现验证。**本例中原始
  报告说"`_aligned_malloc` 卡 2GB 边界",一个 10 行的 ctypes 探针就证伪了。
- **`_aligned_malloc` 失败时先查进程 private commit 和系统提交上限,别先怀疑 CRT。**
  Windows 上 `VirtualAlloc`/`_aligned_malloc` 都走 commit;commit 额度
  (phys + pagefile)才是硬约束,addr space(128TB)几乎不会是瓶颈。
- **`workingSet` ≠ `privateCommit`。**committed-but-unused 内存不出现在 working set
  里,但照样占 commit 额度。诊断时要同时看两者。
