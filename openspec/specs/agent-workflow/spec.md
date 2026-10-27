# agent-workflow Specification

<!--
  本文件是 openspec/specs/agent-workflow/spec.md 的团队基线模板，
  由 miaosoft-zcode-plugin 的 agent-workflow 技能维护。
  项目可在 ## Requirements 末尾追加自己的条目（追加约束可以，放宽不行）。
  格式遵循 OpenSpec 约定：## Purpose + ## Requirements，每条 Requirement 带 SHALL 语句，
  且至少一个 #### Scenario:（WHEN/THEN 列表）。
-->

## Purpose

规定 AI 代理在本仓库实施变更的工作流：分支策略、提交粒度、实施方式、
前置调研、代码审核、验收标准与决策记录。任何 AI 代理开工前必须读本规范；
本规范是项目内工作流问题的最终裁决依据。

## Requirements

### Requirement: 开工前读取规范
任何 AI 代理在开始本项目的开发工作前 SHALL 读取本规范。

#### Scenario: 会话开始首个任务
- **WHEN** AI 代理在本项目开始第一个开发任务
- **THEN** 先读取 `openspec/specs/agent-workflow/spec.md` 再动手实施

### Requirement: 批次分支策略
每一个批次 SHALL 从主干（`master`；若仓库默认主干为 `main`，则以 `main` 为准）创建一个特性分支，批次内的所有提交都落在这个分支上，不得直接在主干上开发。

#### Scenario: 开始一个新批次
- **WHEN** 一个批次的开发工作开始
- **THEN** 从主干最新状态切出 `feat/<批次名>` 特性分支，并切换到该分支工作

### Requirement: 会话工作区隔离（worktree）
每个会话 SHALL 在本会话专属的 git worktree 中工作：开工即 `git worktree add` 从主干挂载，在其中切特性分支并完成全部实施与提交；不得在可能被其他会话（AI / 人工）共用的主工作区中切分支或提交。特性分支 push（或合入主干）后 SHALL 立即 `git worktree remove` 清理本会话 worktree，不留残留。

#### Scenario: 多会话并行开发
- **WHEN** 同一仓库可能存在多个并行会话（其他 AI 代理或人工）
- **THEN** 本会话开工即建立独立 worktree 并在其中工作，主工作区的分支位置与未提交改动保持原样

#### Scenario: 共享工作区出现他人 WIP
- **WHEN** 当前工作区中存在不属于本任务的改动或未跟踪文件
- **THEN** 停止任何整仓暂存（`git add -A` / `git add .`），只以显式文件清单暂存本任务文件，并向用户提示存在并行会话

#### Scenario: 分支推送之后
- **WHEN** 本会话的特性分支已 push（或已合入主干）
- **THEN** 立即 `git worktree remove` 清理本会话 worktree，不残留废弃目录

### Requirement: 单任务单提交
每个任务 SHALL 对应恰好一个 commit，一个 commit 不得混合多个不相关任务的逻辑；提交信息由 `/commit` 按 Conventional Commits 生成。

#### Scenario: 完成一个任务
- **WHEN** 一个任务的改动通过验收线
- **THEN** 用 `/commit` 生成提交信息并提交，commit 只包含该任务的改动

#### Scenario: 改动混合了多个任务
- **WHEN** 待提交的改动包含多个不相关任务的逻辑
- **THEN** 先拆分为多个 commit，再逐个提交

#### Scenario: 暂存本任务文件
- **WHEN** 为提交暂存改动
- **THEN** 使用显式文件清单（`git add <files>`），不得 `git add -A` / `git add .` 整仓暂存——未跟踪与已修改文件可能属于并行会话的 WIP

### Requirement: 专家代理实施
涉及多文件或复杂逻辑的实施 SHALL 由专家子代理携带完整任务说明整体完成，不得逐文件手工修补。

#### Scenario: 多文件功能实现
- **WHEN** 一个任务需要修改三个以上文件，或涉及不熟悉的实现模式
- **THEN** 派出一个子代理一次性完成整个任务的实施，而不是主代理逐文件编辑

### Requirement: 子专家代码完整单测覆盖
专家子代理产出的实施代码 SHALL 附带完整单元测试，覆盖正常路径、边界条件与异常分支；单元测试 SHALL 随该任务的 commit 一同提交。缺失对应单测的子专家产出视为任务未完成，不得通过验收。

#### Scenario: 派出专家子代理实施
- **WHEN** 一个任务由专家子代理完成实施
- **THEN** 产出包含覆盖正常/边界/异常路径的完整单元测试，随任务 commit 一同提交并计入验收

#### Scenario: 子专家产出缺少单测
- **WHEN** 子专家完成的代码没有完整单元测试覆盖
- **THEN** 视为任务未完成，要求补齐测试并重新走验收线

### Requirement: 未知领域先调研
对不熟悉的技术领域，代理 SHALL 先启动只读调研代理收集事实，再决定实施方案。

#### Scenario: 遇到不熟悉的依赖或平台
- **WHEN** 任务涉及代理不熟悉的库、平台或协议
- **THEN** 先派只读调研代理产出事实摘要，基于事实设计实施方案

### Requirement: 缺陷根修止损
同一缺陷的第 2 次修复仍未收敛（症状未消除或验收未通过）时，代理 SHALL 停止继续打补丁，先派只读调研代理完成根因分析并产出事实摘要，再基于根因实施修复。

#### Scenario: 第二次修复仍未收敛
- **WHEN** 同一缺陷的第二次修复后问题仍然存在或验收不通过
- **THEN** 停止直接修补，派只读调研代理产出根因事实摘要，基于根因重新设计修复

### Requirement: 提交前简化审核
每个任务的代码改动 SHALL 在提交前完成 `/simplify` 审核并应用修复；未完成 `/simplify` 的任务不得提交 commit。纯文档改动豁免。

#### Scenario: 任务验收通过
- **WHEN** 任务的 check / test / lint 全部通过
- **THEN** 运行 `/simplify` 审核本任务改动并应用修复，完成后再提交

#### Scenario: 未审核就想提交
- **WHEN** 准备 `/commit` 时本任务改动尚未经过 `/simplify`
- **THEN** 先停下执行 `/simplify` 并应用修复，再回到提交流程

### Requirement: 验收线
一个任务 SHALL 仅在 check 零错误、测试全绿、lint 通过三项全部满足时视为完成；项目未配置的检查项视为跳过。

#### Scenario: 运行验收检查
- **WHEN** 实施完成，运行项目的 check / test / lint 命令
- **THEN** 已配置的检查项全部通过才允许提交；任一失败则继续修复

### Requirement: 定案实时入册
实施过程中产生的每一个定案 SHALL 在定案当时写入本规范，并随当前任务的 commit 一同提交，不得留到批次收尾补记。

#### Scenario: 产生技术定案
- **WHEN** 实施中确定了技术选型、接口约定或取舍
- **THEN** 立即在本规范中记录该定案并随当前任务提交

<!--
  项目自有条目请追加在 ## Requirements 末尾，格式同上：
  ### Requirement: <名称> + SHALL 语句 + #### Scenario: + WHEN/THEN 列表
-->

### Requirement: 上游合并按业务功能分批
合并上游（0xShug0/audio.cpp PR）时 SHALL 将上游 commits 按业务功能划分为时序连续的批次，逐批 `git merge` 并生成独立 merge commit（描述该批业务内容与冲突解法），不得以单个 merge commit 一次性合入。每批边界 SHALL 保持单个功能系列的 commits 完整（不拆散同一功能的提交序列）。合并全程 SHALL 遵守 AGENTS.md 的 fork-policy：上游 C ABI（`include/audiocpp.h`、`src/capi/`、`tests/capi/`、`docs/c_api.md`、`AUDIOCPP_BUILD_C_API`）一律保持删除；fork 私有修复（CAPI、HIP/gfx1151、ggml 内核修复、未验证后端门禁）必须在冲突解决中保留。合并完成后 SHALL 运行 fork 回归套件（`check_loader_catalog_sync.py` + `build_windows.ps1 -BuildTests ON` + `capi/test/build_fork_regression.bat`）验证。

#### Scenario: 合并一个上游 PR
- **WHEN** 上游 PR 携带大量 commits 需要合入
- **THEN** 先按业务功能划分时序连续批次，逐批 merge、逐批记录冲突解决，最后统一跑 fork 回归套件

#### Scenario: 冲突涉及 fork 私有修复
- **WHEN** 冲突文件含 fork 私有修复（如 ggml-cuda teardown、ssm-conv、reuse HIP 门禁、qwen35 KV 零初始化）
- **THEN** 以「上游新功能 + fork 修复并存」为解决目标，不得用任一侧整体覆盖另一侧

### Requirement: CAPI E2E 套件按 sound-rs 调用方式驱动 DLL
`capi/test/e2e/` SHALL 提供 `audiocpp_e2e.c`（运行时 `LoadLibraryA`+`GetProcAddress` 加载 `audiocpp.dll`，镜像 sound-rs `backend/audiocpp.rs` 的 dlopen 调用姿态：符号表、err/clear_error 协议、进度回调装卸、`n_samples` 为交错总数约定、输出 mono 混缩）与 `run_e2e.ps1` 编排器。场景覆盖 SHALL 包含：设备/版本探测、TTS（含 load_model_ex 会话选项与进度回调）、CLON（tts_with_voice_ref + reference_text）、VDES、ASR（离线+词级）、流式 ASR（start/push/finish）、流式 TTS（pull）、VAD（NULL 路径内嵌 silero + energy）、Diar、Align、SEP（transform_stems）、VC（audio_transform）、GEN（generate）、MIDI、嵌入式 denoise/super_resolve、错误路径（坏路径不崩溃）。模型资产 SHALL 从 4080（`D:\iStation\Models\sound\audiocpp`）拷贝到仓库外模型根（默认 `C:\audiocpp-e2e\models`），不入库。

#### Scenario: 跑一次 E2E
- **WHEN** 构建出 audiocpp.dll 并执行 `run_e2e.ps1`
- **THEN** 每个场景输出机器可读结果行，编排器校验产出阈值（时长/RMS/文本关键词/stem 数等）并汇总，任一失败退出码非零
