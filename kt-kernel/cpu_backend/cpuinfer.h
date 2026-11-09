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
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
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
    task_queue_ = new TaskQueue([this] { return poll_doorbell(); });
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue([this] { return poll_doorbell(); });
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue([this] { return poll_doorbell(); });
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

  // submit_with_cuda_stream without the host node on graph replay. While the stream
  // is being captured, the task is registered in a doorbell slot and the graph gets
  // a stream memory write of slot+1 into a mapped word that the queue worker polls,
  // so the CPU starts as soon as the GPU reaches it instead of after the driver's
  // host-callback dispatch (~27 us on SM120), and the stream does not stall there.
  // Requires that this task's sync_with_cuda_stream comes before the next doorbell
  // submit (one KT forward in flight, no deferred task); otherwise, outside capture,
  // or when the memory write is unavailable, it is submit_with_cuda_stream.
  void submit_with_cuda_stream_doorbell(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
    if (stream_is_capturing(user_cuda_stream) && ensure_doorbell()) {
      const uint32_t slot = doorbell_slot_count_.load(std::memory_order_relaxed);
      if (slot < kDoorbellSlots) {
        auto* args = static_cast<HostCallbackArgs*>((void*)params.second);
        args->cpuinfer = this;
        args->captured = true;
        doorbell_slots_[slot] = {(void (*)(void*))params.first, args};
        doorbell_slot_count_.store(slot + 1, std::memory_order_release);
        if (write_value_((CUstream)user_cuda_stream, doorbell_device_, slot + 1, CU_STREAM_WRITE_VALUE_DEFAULT) ==
            CUDA_SUCCESS) {
          keep_alive([args]() { args->release(args); });
          return;
        }
        // The slot stays registered but is never rung; fall back to the host node.
        doorbell_disabled_ = true;
      }
    }
#endif
    submit_with_cuda_stream(user_cuda_stream, params);
  }

  // The doorbell of submit_with_cuda_stream_doorbell without its stream memory write:
  // while the stream is being captured, registers the task in a slot and returns the
  // mapped word (a device pointer) and the value that a kernel captured next on this
  // stream must store once its writes are visible (stage_moe_inputs does). (0, 0)
  // means the caller must submit the task itself. Same one-in-flight rule.
  std::pair<intptr_t, uint32_t> register_doorbell(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
    if (stream_is_capturing(user_cuda_stream) && ensure_doorbell()) {
      const uint32_t slot = doorbell_slot_count_.load(std::memory_order_relaxed);
      if (slot < kDoorbellSlots) {
        auto* args = static_cast<HostCallbackArgs*>((void*)params.second);
        args->cpuinfer = this;
        args->captured = true;
        doorbell_slots_[slot] = {(void (*)(void*))params.first, args};
        doorbell_slot_count_.store(slot + 1, std::memory_order_release);
        keep_alive([args]() { args->release(args); });
        return {static_cast<intptr_t>(doorbell_device_), slot + 1};
      }
    }
#endif
    (void)user_cuda_stream;
    (void)params;
    return {0, 0};
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

  // Captured forwards that ring a doorbell instead of a host node (for tests).
  uint32_t doorbell_slot_count() const { return doorbell_slot_count_.load(std::memory_order_acquire); }

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

  // Claims a rung doorbell and enqueues its task. Runs on the queue worker while it
  // spins and at the start of every sync(); the CAS makes exactly one of them win.
  // The hold keeps sync() from seeing nothing pending between the claim and the
  // enqueue, which would let the stream copy the output back before the task ran.
  bool poll_doorbell() {
    std::atomic<uint32_t>* bell = doorbell_.load(std::memory_order_acquire);
    if (bell == nullptr) return false;
    uint32_t value = bell->load(std::memory_order_acquire);
    if (value == 0) return false;
    task_queue_->hold();
    if (!bell->compare_exchange_strong(value, 0, std::memory_order_acq_rel)) {
      task_queue_->release_hold();
      return false;
    }
    // Slots are registered at capture, long before a replay can ring them.
    if (value > doorbell_slot_count_.load(std::memory_order_acquire)) {
      task_queue_->release_hold();
      return false;
    }
    const DoorbellSlot& slot = doorbell_slots_[value - 1];
    slot.func(slot.args);
    task_queue_->release_hold();
    return true;
  }

#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
  // Lazily maps the doorbell word and resolves cuStreamWriteValue32 through the
  // runtime (no libcuda link). KT_DOORBELL=0 keeps every submit on the host node.
  bool ensure_doorbell() {
    if (doorbell_disabled_) return false;
    if (doorbell_.load(std::memory_order_acquire) != nullptr) return true;
    const char* env = std::getenv("KT_DOORBELL");
    if (env != nullptr && env[0] == '0') {
      doorbell_disabled_ = true;
      return false;
    }
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult status;
    void* host = nullptr;
    void* device = nullptr;
    // Allocating pinned memory is not allowed inside a global-mode capture; relax
    // this thread's mode around it.
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    const bool ok = cudaGetDriverEntryPointByVersion("cuStreamWriteValue32", &fn, 12000, cudaEnableDefault,
                                                     &status) == cudaSuccess &&
                    status == cudaDriverEntryPointSuccess && fn != nullptr &&
                    cudaHostAlloc(&host, 64, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess &&
                    cudaHostGetDevicePointer(&device, host, 0) == cudaSuccess;
    cudaThreadExchangeStreamCaptureMode(&mode);
    if (!ok) {
      if (host != nullptr) cudaFreeHost(host);
      doorbell_disabled_ = true;
      return false;
    }
    write_value_ = reinterpret_cast<WriteValueFn>(fn);
    doorbell_device_ = reinterpret_cast<CUdeviceptr>(device);
    auto* bell = new (host) std::atomic<uint32_t>(0);
    doorbell_.store(bell, std::memory_order_release);
    return true;
  }

  using WriteValueFn = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  WriteValueFn write_value_ = nullptr;
  CUdeviceptr doorbell_device_ = 0;
#endif

  struct DoorbellSlot {
    void (*func)(void*);
    void* args;
  };
  // Fixed capacity so the worker never reads a reallocating table; one slot per
  // captured KT forward (graphs x MoE layers).
  static constexpr uint32_t kDoorbellSlots = 8192;
  std::unique_ptr<DoorbellSlot[]> doorbell_slots_{new DoorbellSlot[kDoorbellSlots]()};
  std::atomic<uint32_t> doorbell_slot_count_{0};
  std::atomic<std::atomic<uint32_t>*> doorbell_{nullptr};
  bool doorbell_disabled_ = false;

  std::vector<std::function<void()>> captured_args_;
  std::mutex captured_args_mutex_;
};

#endif
