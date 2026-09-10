// Regression guard for the eSpeak-ng process-lifetime refcount.
//
// eSpeak-ng's phoneme tables are process-global: espeak_Terminate() and
// unloading the shared library must happen only when the LAST front-end
// instance goes away. Before this was refcounted, sanoTTS/Inflect v2 each tore
// the global state down in their own destructor, so evicting one cached voice
// while another was still loaded killed the process (SIGSEGV inside
// audiocpp_free_model once an LRU cache held two sanoTTS voices).
#include "engine/community_models/espeak_lifetime.h"

#include "test_assert.h"

#include <iostream>

int main() {
    using engine::community::acquire_espeak;
    using engine::community::release_espeak;

    // 未登记就析构(构造中途抛异常):不许越权拆除全局状态。
    engine::test::require(!release_espeak(), "release without acquire must not tear down");

    // 单实例:登记一次 → 析构即最后一个用户。
    acquire_espeak();
    engine::test::require(release_espeak(), "sole user may tear down");

    // 计数器必须回到 0(否则下一个用例会看到脏状态,且真实进程里会漏拆)。
    engine::test::require(!release_espeak(), "counter returned to zero");

    // 两个实例:第一个析构时必须放行,最后一个才拆 —— 这正是崩溃场景。
    acquire_espeak();  // 实例 A(被驱逐)
    acquire_espeak();  // 实例 B(还在缓存里)
    engine::test::require(
        !release_espeak(), "first of two users must NOT terminate the shared eSpeak state");
    engine::test::require(release_espeak(), "last user tears down");

    // 三个实例交错释放,同样是最后一个才拆。
    acquire_espeak();
    acquire_espeak();
    acquire_espeak();
    engine::test::require(!release_espeak(), "1/3 still in use");
    engine::test::require(!release_espeak(), "2/3 still in use");
    engine::test::require(release_espeak(), "3/3 tears down");

    std::cout << "espeak lifetime tests passed\n";
    return 0;
}
