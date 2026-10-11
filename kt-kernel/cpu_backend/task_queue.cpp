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
#include <iostream>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define KT_CPU_PAUSE() _mm_pause()
#else
#define KT_CPU_PAUSE() ((void)0)
#endif

namespace {
// How long the worker spins for a follow-up task before going to sleep.
constexpr int64_t kSpinUs = 100;
}  // namespace

TaskQueue::TaskQueue() : done(false), pending(0) {
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
  cv.notify_all();
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
  cv.notify_one();
}

void TaskQueue::sync(size_t allow_n_pending) {
  // The caller is usually waiting on the tail of a task it has just submitted
  // (one forward per layer during decode), so spin for a short while before
  // going to sleep: the wake-up would otherwise cost a futex round trip on the
  // critical path.
  if (pending.load(std::memory_order_acquire) > allow_n_pending) {
    auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(kSpinUs);
    while (pending.load(std::memory_order_acquire) > allow_n_pending
           && !done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < spin_until) {
      KT_CPU_PAUSE();
    }
  }
  std::exception_ptr task_exception;
  {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] {
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
      cv.notify_all();
    } else {
      // Decode submits one task per layer, so blocking here makes every task
      // pay a futex wake and a context switch before the worker starts.  Spin
      // for a short while first, like the worker pools do, so back-to-back
      // submissions hand the task over without sleeping.
      auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(kSpinUs);
      while (curr->next.load(std::memory_order_acquire) == nullptr) {
        if (std::chrono::steady_clock::now() >= spin_until) {
          break;
        }
        KT_CPU_PAUSE();
      }
      if (curr->next.load(std::memory_order_acquire) != nullptr) {
        continue;
      }
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return curr->next.load(std::memory_order_acquire) != nullptr
            || done.load(std::memory_order_acquire);
      });
    }
  }
}
