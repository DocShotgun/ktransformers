/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#elif KTRANSFORMERS_USE_ASCEND_NPU
#include "vendors/ascend_npu.h"
#endif

#include "./vendors/vendor.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

class CPUInfer;

// Common header for every host-callback argument block.
//
// The block is allocated by the *_task() binding on the Python thread and
// handed to CPUInfer::submit()/submit_with_cuda_stream().  `captured` is set
// by CPUInfer when the stream is being captured: a CUDA graph re-invokes the
// host callback with the same block on every replay, so the block must stay
// alive for as long as the graph does.  In that case CPUInfer owns the block
// and `release` frees it at teardown.  Otherwise the callback owns it and
// releases it once the task has been enqueued.
struct HostCallbackArgs {
  CPUInfer* cpuinfer;
  bool captured;
  void (*release)(void*);
};

class CPUInfer {
 public:
  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    delete backend_;
    delete task_queue_;
    // Argument blocks for captured graphs are owned by us: the graph can
    // replay the callback at any time, so they are only safe to free once the
    // graph is gone.  That is not observable here, but by the time the
    // CPUInfer is destroyed the graph is dead too.
    std::vector<std::function<void()>> captured;
    {
      std::lock_guard<std::mutex> lock(captured_args_mutex_);
      captured.swap(captured_args_);
    }
    captured.clear();
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  // Adopt a fully-formed task. The forward bindings use this to keep their
  // argument block alive until the worker has consumed it.
  void enqueue(std::function<void()> task) { task_queue_->enqueue(std::move(task)); }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    void (*func)(void*) = (void (*)(void*))params.first;
    auto* args = static_cast<HostCallbackArgs*>((void*)params.second);
    args->cpuinfer = this;
    // A captured stream will replay the callback with this same block.  Keep
    // it alive for the lifetime of the graph instead of freeing it in the
    // callback, or the second replay is a double free.
    args->captured = stream_is_capturing(user_cuda_stream);
    if (args->captured) {
      keep_alive([args]() { args->release(args); });
    }
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
    bool captured;
  };

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
  }

  static void sync_and_release_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    // Captured blocks are owned by CPUInfer; eager blocks are freed here even
    // if sync_() rethrows a worker exception.
    std::unique_ptr<SyncArgs> owned(args->captured ? nullptr : args);
    sync_(args);
  }

  void sync(size_t allow_n_pending = 0) {
    // Synchronous: the argument lives on the stack.
    SyncArgs args{this, allow_n_pending, false};
    sync_(&args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    auto* args = new SyncArgs{this, allow_n_pending, false};
    args->captured = stream_is_capturing(user_cuda_stream);
    if (args->captured) {
      keep_alive([args]() { delete args; });
    }
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_and_release_, (void*)args);
#endif
  }
#endif

 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;

 private:
  // True when `user_cuda_stream` is being captured into a CUDA graph.  The
  // host callback of a captured stream is re-invoked on every replay, so its
  // argument block must outlive the callback.
  static bool stream_is_capturing(intptr_t user_stream) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    return cudaStreamIsCapturing((cudaStream_t)user_stream, &status) == cudaSuccess &&
           status != cudaStreamCaptureStatusNone;
#elif defined(KTRANSFORMERS_USE_ROCM)
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing((hipStream_t)user_stream, &status) == hipSuccess &&
           status != hipStreamCaptureStatusNone;
#elif defined(KTRANSFORMERS_USE_MUSA)
    musaStreamCaptureStatus status = musaStreamCaptureStatusNone;
    return musaStreamIsCapturing((musaStream_t)user_stream, &status) == musaSuccess &&
           status != musaStreamCaptureStatusNone;
#elif defined(KTRANSFORMERS_USE_MACA)
    mcStreamCaptureStatus status = mcStreamCaptureStatusNone;
    return mcStreamIsCapturing((mcStream_t)user_stream, &status) == mcSuccess &&
           status != mcStreamCaptureStatusNone;
#else
    (void)user_stream;
    return false;  // Ascend ACL has no stream capture.
#endif
  }

  void keep_alive(std::function<void()> release) {
    std::lock_guard<std::mutex> lock(captured_args_mutex_);
    captured_args_.push_back(std::move(release));
  }

  std::vector<std::function<void()>> captured_args_;
  std::mutex captured_args_mutex_;
};

#endif
