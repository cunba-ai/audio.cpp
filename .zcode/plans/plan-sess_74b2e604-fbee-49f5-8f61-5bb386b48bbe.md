# 给 C-API JSON 数字→选项字符串解析补完整单元测试

## 背景
C-API `apply_options()`（`capi/src/audiocpp_capi.cpp:310`）把 `{"min_tokens": 2}` 渲染成 `"2.000000"`、被后端 `stoll` 拒绝的 bug 已修复,但没有测试。而且修复用的边界是 `±9.2e18`(int64 范围),与 CLI 成熟实现 `app/cli/request.cpp:72` 的 `2^53` 边界**不一致**:介于 2^53 和 9.2e18 之间的整数(如 `9007199254740993`)会被 C-API 静默四舍五入 —— 这正是"完整覆盖各种 case"会暴露的真 bug。

## 决策(已与你确认)
1. **对齐 CLI 的 2^53 边界** —— 修掉静默舍入。
2. **C-API 内部 helper + 新测试目标** —— 不动 framework/CLI,改动聚焦。

---

## 改动清单

### 1. 新建内部头文件 `capi/src/audiocpp_internal.h`
抽出可测 helper,逻辑**完全对齐** CLI 的 `json_option_string`(`app/cli/request.cpp:72`):
```cpp
#pragma once
#include <string>
namespace audiocpp::detail {
// 把 cJSON/JSON 数字渲染成选项 map 里的字符串。整数(< 2^53)渲染成
// 整数字符串,以便后端 stoll 解析;超出 double 精确整数表示范围
// (>= 2^53)的值走科学计数法,让 stoll 拒绝而非静默舍入。
std::string option_number_to_string(double value);
}
```

### 2. `capi/src/audiocpp_capi.cpp`
- **include** 新头:`#include "audiocpp_internal.h"`(在 `#include "cJSON.h"` 附近)。
- **实现** `audiocpp::detail::option_number_to_string`:逻辑复制自 CLI(`std::isfinite` + `std::trunc(v)==v` + `fabs(v) < 9007199254740992.0` → `ostringstream << fixed << setprecision(0)`;否则 `engine::io::json::stringify_number`)。需新增 `#include <sstream>` 和 `#include <iomanip>`(文件已有 `<cmath>`)。
- **替换** `apply_options` 内 310-318 行的 inline 分支为单行调用:
  ```cpp
  else if (cJSON_IsNumber(item)) {
      req.options[key] = audiocpp::detail::option_number_to_string(item->valuedouble);
  }
  ```
  删掉那条解释 ±9.2e18 的注释,改为引用 helper。
- `engine::io::json::stringify_number` 已在 `engine_runtime` 库里(`include/engine/framework/io/json.h:76`),`capi_test` 链了 `engine_runtime`,可直接用。

### 3. 新建测试 `tests/unittests/test_capi_option_number.cpp`
纯 C++ 单测,直接调 `audiocpp::detail::option_number_to_string`,复用现有 `test_assert.h`(`engine::test::require_eq` / `require`)。保持**全 ASCII**(C-API 源含 CJK 注释,`capi_test` 目标没设 `/utf-8`)。

**完整覆盖矩阵(AAA 结构,表驱动):**

| case | 输入 | 期望输出 | 验证点 |
|---|---|---|---|
| 正整数 | `2.0` | `"2"` | 核心 bug:`"2"` 而非 `"2.000000"` |
| 零 | `0.0` | `"0"` | 边界 |
| 负整数 | `-7.0` | `"-7"` | 负数 |
| 大安全整数 | `1000000000000000`(1e15) | `"1000000000000000"` | <2^53 内大整数精确 |
| 1e15+1 | `1000000000000001` | `"1000000000000001"` | <2^53 内连续整数可区分 |
| 负大安全整数 | `-1e15` | `"-1000000000000000"` | 负数大整数 |
| 普通小数 | `0.7` | `"0.7"` | 浮点不丢小数 |
| 分数 | `1.25` | `"1.25"` | 多位小数 |
| 负小数 | `-0.5` | `"-0.5"` | 负浮点 |
| **2^53 边界** | `9007199254740992.0` | 含 `e`/`E` | 超 2^53 → 科学计数法 |
| **2^53+1** | `9007199254740993` | 含 `e`/`E` | **拒绝静默舍入**(原 9.2e18 实现会错) |

**回归断言(防 bug 复发):** 额外加一个 round-trip 检查 —— `option_number_to_string(2.0)` 出来的 `"2"` 喂给 `engine::runtime::parse_int_option`(options.cpp)必须返回 `2` 不抛异常;`option_number_to_string(9007199254740993)` 的结果喂进去必须抛 `"must be an integer"`。这直接锁死根因(stoll 拒绝 `2.000000` / 拒绝被舍入的大整数)。

### 4. CMake 注册(`CMakeLists.txt`,在 `capi_test` 目标 @1565 附近)
镜像现有 `capi_test` 模式新建目标 `capi_option_number_test`:
```cmake
add_executable(capi_option_number_test
    tests/unittests/test_capi_option_number.cpp
    capi/src/audiocpp_capi.cpp)
set_target_properties(capi_option_number_test PROPERTIES CXX_VISIBILITY_PRESET default)
target_compile_definitions(capi_option_number_test PRIVATE AUDIOCPP_EXPORTS)
if (MSVC)
    target_compile_options(capi_option_number_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()
target_include_directories(capi_option_number_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/capi/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/capi/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests")
target_link_libraries(capi_option_number_test PRIVATE engine_runtime ggml cjson_vendor)
add_test(NAME capi_option_number_test COMMAND capi_option_number_test)
```
**关键点**:① 静态编译 `audiocpp_capi.cpp`(与 capi_test 一致),内部链接 helper 可被同 TU 测试调用;② 加 `/utf-8` 防 CJK 注释在非英文 codepage 触发 C4819;③ **加 `add_test(...)`**(现有 capi_test 没注册进 ctest,我会确保新测试能被 ctest 跑到)。

---

## 验证步骤
1. `cmake -S . -B build/verify-test -DENGINE_BUILD_TESTS=ON -DAUDIOCPP_BUILD_CAPI=ON`
2. `cmake --build build/verify-test --target capi_option_number_test -j`
3. `ctest --test-dir build/verify-test -R capi_option_number_test --output-on-failure`
4. 全部 case pass,包括 2^53+1 拒绝静默舍入那条。

## 不做的事
- 不改 `app/cli/`(CLI 已对且已测),不去重到 framework 层(尊重"改动聚焦"决策)。
- 不动 public ABI / 不加 inspect 函数。
- 不碰 streaming / 其他 inference 入口(它们都走同一个 `apply_options`)。
