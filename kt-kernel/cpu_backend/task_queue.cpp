/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"

#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define KT_CPU_PAUSE() _mm_pause()
#else
#define KT_CPU_PAUSE() ((void)0)
#endif

namespace {
int64_t spin_us_from_env(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  const long long parsed = std::atoll(value);
  return parsed < 0 ? fallback : parsed;
}
// How long the worker spins for a follow-up task before going to sleep
// (KT_TASKQ_SPIN_US). Decode's gap between layers is the GPU's per-layer work
// (~300 us on DSv4.1, more on larger models); sleeping there puts a futex wake on
// every layer's critical path.
const int64_t kWorkerSpinUs = spin_us_from_env("KT_TASKQ_SPIN_US", 2000);
// How long sync() spins for the tail of the task it just submitted
// (KT_SYNC_SPIN_US). Decode tasks take 70-1000 us; past the spin, completion costs
// a futex wake. A prefill layer's task outlasts it and then sleeps as before.
const int64_t kSyncSpinUs = spin_us_from_env("KT_SYNC_SPIN_US", 1000);
}  // namespace

TaskQueue::TaskQueue(std::function<bool()> idle_poll) : done(false), pending(0), idle_poll_(std::move(idle_poll)) {
  Node* dummy = new Node();
  head.store(dummy, std::memory_order_relaxed);
  tail.store(dummy, std::memory_order_relaxed);
  workerThread = std::thread(&TaskQueue::worker, this);
}

TaskQueue::~TaskQueue() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    done.store(true, std::memory_order_release);
  }
  worker_cv.notify_all();
  sync_cv.notify_all();
  if (workerThread.joinable()) workerThread.join();

  Node* node = head.load(std::memory_order_relaxed);
  while (node) {
    Node* next = node->next.load(std::memory_order_relaxed);
    delete node;
    node = next;
  }
}

void TaskQueue::enqueue(std::function<void()> task) {
  pending.fetch_add(1, std::memory_order_acq_rel);
  Node* node = new Node(task);
  Node* prev = tail.exchange(node, std::memory_order_acq_rel);
  prev->next.store(node, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mtx);
  }
  worker_cv.notify_one();
}

void TaskQueue::hold() { pending.fetch_add(1, std::memory_order_acq_rel); }

void TaskQueue::release_hold() {
  {
    std::lock_guard<std::mutex> lock(mtx);
    pending.fetch_sub(1, std::memory_order_acq_rel);
  }
  sync_cv.notify_all();
}

void TaskQueue::sync(size_t allow_n_pending) {
  // A doorbell the worker has not claimed yet is claimed here, so the wait below
  // always covers the task it stands for.
  if (idle_poll_) idle_poll_();
  // The caller is usually waiting on the tail of a task it has just submitted
  // (one forward per layer during decode), so spin for a short while before
  // going to sleep: the wake-up would otherwise cost a futex round trip on the
  // critical path. The spin is unconditional; only the duration is bounded.
  if (pending.load(std::memory_order_acquire) > allow_n_pending) {
    auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(kSyncSpinUs);
    while (pending.load(std::memory_order_acquire) > allow_n_pending
           && !done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < spin_until) {
      KT_CPU_PAUSE();
    }
  }
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    sync_cv.wait(lock, [&] {
      return pending.load(std::memory_order_acquire) <= allow_n_pending
          || done.load(std::memory_order_acquire);
    });
    task_exception = first_exception;
    first_exception = nullptr;
  }
  if (task_exception) std::rethrow_exception(task_exception);
}

void TaskQueue::worker() {
  Node* curr = head.load(std::memory_order_relaxed);
  while (!done.load(std::memory_order_acquire)) {
    Node* next = curr->next.load(std::memory_order_acquire);
    if (next) {
      std::exception_ptr task_exception;
      if (next->task) {
        try {
          next->task();
        } catch (...) {
          task_exception = std::current_exception();
        }
        // This node stays the queue head until the next task arrives; release
        // the closure and its captures before sync() can observe completion.
        next->task = nullptr;
      }
      delete curr;
      curr = next;
      head.store(curr, std::memory_order_release);
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (task_exception && !first_exception) {
          first_exception = task_exception;
        }
        pending.fetch_sub(1, std::memory_order_acq_rel);
      }
      sync_cv.notify_all();
    } else {
      // Decode submits one task per layer, so blocking here makes every task
      // pay a futex wake and a context switch before the worker starts.  Spin
      // for a short while first, like the worker pools do, so back-to-back
      // submissions hand the task over without sleeping.
      auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(kWorkerSpinUs);
      while (curr->next.load(std::memory_order_acquire) == nullptr) {
        if (idle_poll_ && idle_poll_()) break;
        if (std::chrono::steady_clock::now() >= spin_until) {
          break;
        }
        KT_CPU_PAUSE();
      }
      if (curr->next.load(std::memory_order_acquire) != nullptr) {
        continue;
      }
      std::unique_lock<std::mutex> lock(mtx);
      worker_cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}
