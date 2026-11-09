/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-16 10:43:18
 * @Version   : 1.0.0
 * @LastEditors : chenht
 * @LastEditTime : 2024-10-09 11:08:07
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_TASKQUEUE_H
#define CPUINFER_TASKQUEUE_H

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class TaskQueue {
 public:
  // `idle_poll` runs while the worker spins for work and once at the start of
  // sync(); it may enqueue a task (CPUInfer's doorbell) and returns whether it did.
  explicit TaskQueue(std::function<bool()> idle_poll = nullptr);
  ~TaskQueue();

  void enqueue(std::function<void()>);

  void sync(size_t allow_n_pending);

  // Count one more pending task without enqueuing, so sync() cannot see the queue
  // drained while a caller is about to enqueue; release_hold() undoes it.
  void hold();
  void release_hold();

 private:
  struct Node {
    std::function<void()> task;
    std::atomic<Node*> next;
    Node() : task(nullptr), next(nullptr) {}
    Node(const std::function<void()>& t) : task(t), next(nullptr) {}
  };

  std::atomic<Node*> head;
  std::atomic<Node*> tail;
  std::atomic<bool> done;
  std::atomic<size_t> pending;
  std::thread workerThread;
  std::mutex mtx;
  // Separate waiters: enqueue() only ever wakes the worker and the worker
  // only ever wakes sync(). Sharing one condition variable let a sync()
  // waiter consume the worker's notification (or vice versa), leaving the
  // other side asleep until the next enqueue.
  std::condition_variable worker_cv;
  std::condition_variable sync_cv;
  std::exception_ptr first_exception;
  std::function<bool()> idle_poll_;

  void worker();
};

#endif
