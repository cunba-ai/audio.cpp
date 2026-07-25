// Unit tests for the offline run() progress callback mechanism.
//
// Exercises RuntimeSessionBase::emit_progress() and ProgressCanceled directly
// via a mock session whose run() loops over a known number of units, mirroring
// how the real chunked models are instrumented. Verifies:
//   1. With no callback installed, run() completes normally (no-op).
//   2. With a callback, progress events fire in order with correct counts and
//      a monotonic fraction in [0,1].
//   3. Returning false from the callback aborts run() via ProgressCanceled,
//      and the number of completed units matches the cancellation point.

#include "engine/framework/runtime/session.h"
#include "engine/framework/runtime/session_base.h"

#include "test_assert.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Minimal session that loops `total_units` times, emitting progress at each
// boundary exactly like the real chunked models do.
class LoopingMockSession final
    : public engine::runtime::RuntimeSessionBase
    , public engine::runtime::IOfflineVoiceTaskSession {
public:
    explicit LoopingMockSession(int64_t total_units)
        : engine::runtime::RuntimeSessionBase(engine::runtime::SessionOptions{})
        , total_units_(total_units) {}

    std::string family() const override { return "looping_mock"; }
    engine::runtime::VoiceTaskKind task_kind() const override {
        return engine::runtime::VoiceTaskKind::Tts;
    }
    engine::runtime::RunMode run_mode() const override {
        return engine::runtime::RunMode::Offline;
    }
    void prepare(const engine::runtime::SessionPreparationRequest &) override {
        mark_prepared();
    }

    engine::runtime::TaskResult run(const engine::runtime::TaskRequest &) override {
        require_prepared("LoopingMockSession run()");
        emit_progress("looping_mock", 0, total_units_);
        for (int64_t i = 0; i < total_units_; ++i) {
            ++completed_so_far_;
            emit_progress("looping_mock", completed_so_far_, total_units_);
        }
        engine::runtime::TaskResult result;
        return result;
    }

    int64_t completed_so_far() const { return completed_so_far_; }

private:
    int64_t total_units_;
    int64_t completed_so_far_ = 0;
};

// Records every progress event seen by a callback.
struct ProgressRecorder {
    std::vector<engine::runtime::ProgressInfo> events;
    bool cancel_at = false;   // when true, request cancellation
    int64_t cancel_after = 0; // cancel once completed_units reaches this value

    bool operator()(const engine::runtime::ProgressInfo & info) {
        events.push_back(info);
        if (cancel_at && info.completed_units >= cancel_after) {
            return false;  // request cancellation
        }
        return true;
    }
};

void test_no_callback_is_noop() {
    LoopingMockSession session(5);
    session.prepare(engine::runtime::SessionPreparationRequest{});
    // No set_progress_callback call — run() must complete without throwing.
    engine::runtime::TaskRequest req;
    auto result = session.run(req);
    (void)result;
    engine::test::require_eq<int64_t, int64_t>(
        session.completed_so_far(), 5, "no-callback run should complete all units");
}

void test_progress_events_fire_in_order() {
    LoopingMockSession session(4);
    session.prepare(engine::runtime::SessionPreparationRequest{});
    ProgressRecorder rec;
    session.set_progress_callback(
        std::ref(rec));  // std::ref avoids copying the recorder into the std::function

    engine::runtime::TaskRequest req;
    session.run(req);

    // Expect 5 events: (0,4), (1,4), (2,4), (3,4), (4,4)
    engine::test::require_eq<size_t, size_t>(rec.events.size(), 5, "event count");
    engine::test::require_eq<int64_t, int64_t>(
        rec.events.front().completed_units, 0, "first event completed_units");
    engine::test::require_eq<int64_t, int64_t>(
        rec.events.front().total_units, 4, "first event total_units");
    engine::test::require_eq<int64_t, int64_t>(
        rec.events.back().completed_units, 4, "last event completed_units");
    engine::test::require_close(rec.events.back().progress, 1.0f, 1e-6f, "final progress");
    engine::test::require_eq<std::string, std::string>(
        rec.events.front().stage, "looping_mock", "stage name");

    // Monotonic non-decreasing progress within [0,1].
    for (const auto & ev : rec.events) {
        engine::test::require(ev.progress >= 0.0f - 1e-6f && ev.progress <= 1.0f + 1e-6f,
                              "progress out of [0,1] range");
    }
    for (size_t i = 1; i < rec.events.size(); ++i) {
        engine::test::require(rec.events[i].progress + 1e-6f >= rec.events[i - 1].progress,
                              "progress not monotonic");
    }
}

void test_single_shot_emits_start_and_end() {
    LoopingMockSession session(1);  // total_units == 1 mirrors single-shot models
    session.prepare(engine::runtime::SessionPreparationRequest{});
    ProgressRecorder rec;
    session.set_progress_callback(std::ref(rec));

    engine::runtime::TaskRequest req;
    session.run(req);

    engine::test::require_eq<size_t, size_t>(rec.events.size(), 2, "single-shot event count");
    engine::test::require_close(rec.events[0].progress, 0.0f, 1e-6f, "single-shot start");
    engine::test::require_close(rec.events[1].progress, 1.0f, 1e-6f, "single-shot end");
}

void test_cancellation_throws_and_stops_early() {
    LoopingMockSession session(10);
    session.prepare(engine::runtime::SessionPreparationRequest{});
    ProgressRecorder rec;
    rec.cancel_at = true;
    rec.cancel_after = 3;  // cancel once 3 units are done
    session.set_progress_callback(std::ref(rec));

    engine::runtime::TaskRequest req;
    bool caught = false;
    try {
        session.run(req);
    } catch (const engine::runtime::ProgressCanceled & e) {
        caught = true;
        engine::test::require(
            std::string(e.what()).find("canceled") != std::string::npos,
            "ProgressCanceled message should mention cancellation");
    }
    engine::test::require(caught, "run() should have thrown ProgressCanceled");
    // The session stopped at unit 3 (the cancel point), not all 10.
    engine::test::require_eq<int64_t, int64_t>(
        session.completed_so_far(), 3, "session should have stopped at cancel point");
}

void test_clearing_callback_resumes_noop() {
    LoopingMockSession session(3);
    session.prepare(engine::runtime::SessionPreparationRequest{});
    ProgressRecorder rec;
    session.set_progress_callback(std::ref(rec));
    session.set_progress_callback(nullptr);  // clear

    engine::runtime::TaskRequest req;
    session.run(req);
    engine::test::require_eq<size_t, size_t>(
        rec.events.size(), 0, "cleared callback must not fire");
}

}  // namespace

int main() {
    try {
        test_no_callback_is_noop();
        test_progress_events_fire_in_order();
        test_single_shot_emits_start_and_end();
        test_cancellation_throws_and_stops_early();
        test_clearing_callback_resumes_noop();
        std::cout << "progress_callback_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "progress_callback_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
