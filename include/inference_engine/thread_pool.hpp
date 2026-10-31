#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace inference_engine {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 1);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t num_threads() const noexcept { return num_threads_; }

    // Executes `task_fn(task_index, thread_index)` across `num_tasks` items [0, num_tasks).
    void parallel_for(size_t num_tasks,
                      const std::function<void(size_t task_idx, size_t thread_idx)>& task_fn);

private:
    size_t num_threads_ = 1;
    std::vector<std::thread> workers_{};
    std::mutex mutex_{};
    std::condition_variable cv_start_{};
    std::condition_variable cv_done_{};
    bool stop_ = false;
    uint64_t generation_ = 0;
    size_t total_tasks_ = 0;
    size_t next_task_ = 0;
    size_t active_workers_ = 0;
    std::function<void(size_t, size_t)> current_fn_{};

    void worker_loop(size_t worker_id);
};

}  // namespace inference_engine
