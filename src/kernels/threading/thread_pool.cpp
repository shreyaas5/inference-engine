#include "inference_engine/thread_pool.hpp"

#include <algorithm>

namespace inference_engine {

ThreadPool::ThreadPool(size_t num_threads)
    : num_threads_(std::max<size_t>(1, num_threads)) {
    if (num_threads_ > 1) {
        workers_.reserve(num_threads_ - 1);
        for (size_t i = 1; i < num_threads_; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        ++generation_;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
}

void ThreadPool::parallel_for(size_t num_tasks,
                              const std::function<void(size_t task_idx, size_t thread_idx)>& task_fn) {
    if (num_tasks == 0 || !task_fn) {
        return;
    }
    if (num_threads_ <= 1 || num_tasks == 1) {
        for (size_t i = 0; i < num_tasks; ++i) {
            task_fn(i, 0);
        }
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_fn_ = task_fn;
        total_tasks_ = num_tasks;
        next_task_ = 0;
        active_workers_ = workers_.size();
        ++generation_;
    }
    cv_start_.notify_all();

    // Dynamically chunk task acquisitions to eliminate lock contention on large workloads
    const size_t chunk_size = std::max<size_t>(1, num_tasks / (num_threads_ * 4));

    // Thread 0 participates in executing tasks.
    while (true) {
        size_t start_idx = 0;
        size_t end_idx = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (next_task_ >= total_tasks_) {
                break;
            }
            start_idx = next_task_;
            next_task_ = std::min(total_tasks_, start_idx + chunk_size);
            end_idx = next_task_;
        }
        for (size_t t = start_idx; t < end_idx; ++t) {
            task_fn(t, 0);
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this]() { return active_workers_ == 0 && next_task_ >= total_tasks_; });
    current_fn_ = nullptr;
}

void ThreadPool::worker_loop(size_t worker_id) {
    uint64_t seen_gen = 0;
    while (true) {
        std::function<void(size_t, size_t)> fn;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_start_.wait(lock, [this, &seen_gen]() { return stop_ || generation_ != seen_gen; });
            if (stop_) {
                return;
            }
            seen_gen = generation_;
            fn = current_fn_;
        }

        const size_t chunk_size = std::max<size_t>(1, total_tasks_ / (num_threads_ * 4));

        while (true) {
            size_t start_idx = 0;
            size_t end_idx = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (next_task_ >= total_tasks_) {
                    if (--active_workers_ == 0) {
                        cv_done_.notify_all();
                    }
                    break;
                }
                start_idx = next_task_;
                next_task_ = std::min(total_tasks_, start_idx + chunk_size);
                end_idx = next_task_;
            }
            for (size_t t = start_idx; t < end_idx; ++t) {
                fn(t, worker_id);
            }
        }
    }
}

}  // namespace inference_engine
