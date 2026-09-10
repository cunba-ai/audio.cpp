#pragma once

// Shared eSpeak-ng lifetime management for the community front ends that
// phonemize through it (sanoTTS, Inflect v2).
//
// Why this exists: eSpeak-ng keeps its phoneme tables and voice list in
// PROCESS-GLOBAL state. espeak_Initialize() builds that state once for the
// whole process and espeak_Terminate() frees it for the whole process — it is
// not per-handle. A front end that owns one eSpeak handle per loaded model
// therefore cannot tear the state down in its own destructor while other
// instances are still alive: doing so unloads the shared library and frees the
// tables the survivors still call into.
//
// Observed failure (2026-09-10, sound-rs): with an LRU cache holding more than
// one sanoTTS voice, evicting one of them ran ~EspeakApi → espeak_Terminate()
// → FreeLibrary("espeak-ng.dll") while a second voice was still cached; the
// process died with SIGSEGV inside audiocpp_free_model. Evicting the only live
// eSpeak user was clean, and evicting a non-eSpeak model never crashed —
// reproducing on a build predating the sanoTTS voice additions.
//
// Reference-count the teardown instead: the first user initializes, the last
// one terminates and releases the library, everything in between is a no-op.

#include <mutex>

namespace engine::community {

/// 计数器对所有翻译单元唯一(inline 函数内的函数级 static,C++ 保证合并)。
inline int & espeak_user_count() {
    static int count = 0;
    return count;
}

inline std::mutex & espeak_user_mutex() {
    static std::mutex mutex;
    return mutex;
}

/// 登记一个 eSpeak-ng 使用方。构造成功、全局状态可用之后调用一次。
inline void acquire_espeak() {
    const std::lock_guard<std::mutex> guard(espeak_user_mutex());
    ++espeak_user_count();
}

/// 注销一个使用方。返回 true 表示调用者是最后一个用户,可以安全地
/// espeak_Terminate() + 卸载共享库;返回 false 表示还有实例在用,
/// 调用者必须原样放行,不得触碰进程级状态。
[[nodiscard]] inline bool release_espeak() {
    const std::lock_guard<std::mutex> guard(espeak_user_mutex());
    if (espeak_user_count() == 0) {
        return false;  // 未登记就析构(构造中途抛异常),不要越权拆除
    }
    return --espeak_user_count() == 0;
}

}  // namespace engine::community
