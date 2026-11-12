/**
 * @Description  : Common AMX MoE base class extracted from K2 implementation.
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 0.1.0
 * @LastEditors  : oql, Codex and Claude
 * @LastEditTime : 2025-12-09
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_BASE_H
#define CPUINFER_OPERATOR_AMX_MOE_BASE_H

// #define FORWARD_TIME_PROFILE

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "la/amx.hpp"
#include "llama.cpp/ggml.h"

template <class T, class Derived>
class AMX_MOE_BASE {
 public:
  int tp_part_idx = 0;

  ggml_bf16_t* m_local_input_ = nullptr;
  ggml_bf16_t* m_local_gate_output_ = nullptr;
  ggml_bf16_t* m_local_up_output_ = nullptr;
  ggml_bf16_t* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t*> m_local_input_ptr_;
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  GeneralMOEConfig config_;
  using input_t = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  // Fused-decode completion counters, one cache line each.
  struct alignas(64) PaddedCounter {
    std::atomic<int> v{0};
  };
  std::unique_ptr<PaddedCounter[]> decode_expert_done_;   // [num_experts_per_tok]
  std::unique_ptr<PaddedCounter[]> decode_expert_ready_;  // [num_experts_per_tok]
  std::unique_ptr<PaddedCounter[]> decode_block_done_;    // [decode_block_capacity_]
  int decode_block_capacity_ = 0;
  std::unique_ptr<PaddedCounter[]> decode_pair_done_;     // [decode_pair_capacity_], per (expert, gate/up range)
  int decode_pair_capacity_ = 0;
  std::vector<const ggml_bf16_t*> decode_sources_;
  std::vector<float> decode_route_weights_;

  AMX_MOE_BASE(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void set_staged_weight_pointers(void* gate, void* up, void* down, void* gate_scale, void* up_scale,
                                  void* down_scale) {
    config_.gate_proj = gate;
    config_.up_proj = up;
    config_.down_proj = down;
    config_.gate_scale = gate_scale;
    config_.up_scale = up_scale;
    config_.down_scale = down_scale;
  }

  void clear_staged_weight_pointers() {
    set_staged_weight_pointers(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    MemoryRequest mem_requests;
    const size_t ml = config_.max_len;
    const size_t k_tok = config_.num_experts_per_tok;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;
    mem_requests.append_pointer(&m_local_input_, sizeof(ggml_bf16_t) * k_tok * ml * H);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * k_tok * ml * H);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    decode_expert_done_.reset(new PaddedCounter[config_.num_experts_per_tok]);
    decode_expert_ready_.reset(new PaddedCounter[config_.num_experts_per_tok]);
    decode_sources_.reserve(config_.num_experts_per_tok);
    decode_route_weights_.reserve(config_.num_experts_per_tok);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.hidden_size, nullptr));

      // Keep global expert indexing for routing, but allow backends to omit
      // resident weight buffers for experts served by another device.
      if (!derived_const()->should_allocate_expert_weights(i)) {
        gate_bb_.push_back(nullptr);
        up_bb_.push_back(nullptr);
        down_bb_.push_back(nullptr);
        continue;
      }

      void* gate_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(make_buffer_b(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
    }
    // TODO: need update to all *.hpp
    // (config_.expert_num * T::M_STEP) in pool_count_ is to ensure padding for each experts.
    pool_count_ = (size_t)config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    // BufferA implementations may alias prefill inputs and require only a
    // one-row staging tile per active expert. Reserve for every expert even
    // when required_size(pool_count_) no longer scales with pool_count_.
    gate_up_ba_pool_bytes_ = std::max(buffer_a_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64,
                                      config_.expert_num * (buffer_a_required_size(1, config_.hidden_size) + 64));
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = std::max(buffer_a_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64,
                                   config_.expert_num * (buffer_a_required_size(1, config_.intermediate_size) + 64));
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AMX_MOE_BASE() { shared_mem_buffer_numa.dealloc(tp_part_idx, this); }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

  template <typename... Args>
  void load_weights(Args&&... args) {
    derived()->load_weights(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void write_weights_to_buffer(Args&&... args) const {
    derived_const()->write_weights_to_buffer(std::forward<Args>(args)...);
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) {
        continue;
      }

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }

    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_a <= gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->gate_up_gemm_to_mat(do_up, expert_idx, ith, nth, qlen);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    apply_activation(activated_expert, nth, qlen);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->down_gemm_to_mat(expert_idx, ith, nth, qlen);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          // Resolve the routing once per token: the hidden loop below would
          // otherwise reload expert_ids/weights and re-test should_skip for
          // every 32-element chunk (output may alias them as far as the
          // compiler knows).
          std::vector<const ggml_bf16_t*> sources;
          std::vector<float> route_weights;
          sources.reserve(k);
          route_weights.reserve(k);
          for (int j = 0; j < k; j++) {
            const int64_t expert = expert_ids[i * k + j];
            if (config_.should_skip_expert(expert)) {
              continue;
            }
            sources.push_back(m_local_down_output_ptr_[expert] + m_local_pos_[i][j] * config_.hidden_size);
            route_weights.push_back(weights[i * k + j] * derived_const()->down_global_scale(expert));
          }
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (size_t j = 0; j < sources.size(); j++) {
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i*)(sources[j] + e), &down_output0, &down_output1);
              const __m512 weight = _mm512_set1_ps(route_weights[j]);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long q_input_time = 0, up_gate_time = 0, act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      // Every expert reads the same input row; a backend whose conversion does not
      // depend on the expert can point the others at the first expert's copy.
      if (i == 0 || !derived()->share_decode_gate_input(expert_idx, m_expert_id_map_[0], qlen)) {
        derived()->prepare_decode_gate_input(expert_idx, qlen, input);
      }
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    if (fused_decode_enabled()) {
      run_fused_decode(activated_expert, k, expert_ids, weights, output);
#ifdef FORWARD_TIME_PROFILE
      auto fused_end = std::chrono::high_resolution_clock::now();
      printf("Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, fused: %ld us, total: %ld us\n",
             tp_part_idx, activated_expert, q_input_time,
             (long)std::chrono::duration_cast<std::chrono::microseconds>(fused_end - last).count(),
             (long)std::chrono::duration_cast<std::chrono::microseconds>(fused_end - start_time).count());
#endif
      return;
    }

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    derived()->apply_decode_activation(activated_expert, nth, qlen);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          derived()->prepare_decode_down_input(expert_idx, qlen);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    std::vector<const ggml_bf16_t*> sources;
    std::vector<float> route_weights;
    sources.reserve(k);
    route_weights.reserve(k);
    for (int j = 0; j < k; j++) {
      if (config_.should_skip_expert(expert_ids[j])) {
        continue;
      }
      sources.push_back(m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size);
      route_weights.push_back(weights[j] * derived_const()->down_global_scale(expert_ids[j]));
    }
    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (size_t j = 0; j < sources.size(); j++) {
        __m512 weight = _mm512_set1_ps(route_weights[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32((__m512i*)(sources[j] + e), &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }
      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  void prepare_decode_gate_input(int expert_idx, int qlen, const void* input) {
    gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
  }

  // Makes expert_idx's gate/up input alias source_expert's converted one; false keeps
  // the per-expert conversion.
  bool share_decode_gate_input(int expert_idx, int source_expert, int qlen) { return false; }

  // One prefill gate/up GEMM task, leaving its BF16 rows in the expert's gate or up
  // output. A backend whose kernel writes those rows itself overrides this.
  void gate_up_gemm_to_mat(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
    if (do_up) {
      up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
    } else {
      gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
    }
  }

  // The same for one prefill down GEMM task and the expert's down output.
  void down_gemm_to_mat(int expert_idx, int ith, int nth, int qlen) {
    derived()->do_down_gemm(expert_idx, ith, nth, qlen);
    down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
  }

  void prepare_decode_down_input(int expert_idx, int qlen) {
    down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
  }

  void apply_decode_activation(int activated_expert, int nth, int qlen) {
    apply_activation(activated_expert, nth, qlen);
  }

  // One gate/up range of one expert, for the fused decode: the same arithmetic as
  // apply_activation's task for (expert_idx, ith).
  void apply_decode_activation_range(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
    const __m512 gate_scale = _mm512_set1_ps(derived_const()->gate_global_scale(expert_idx));
    const __m512 up_scale = _mm512_set1_ps(derived_const()->up_global_scale(expert_idx));
    for (int i = 0; i < m_local_num_[expert_idx]; i++) {
      ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
      ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
      for (int j = n_start; j < n_end; j += 32) {
        __m512 gate_val0, gate_val1, up_val0, up_val1;
        avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
        avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
        gate_val0 = _mm512_mul_ps(gate_val0, gate_scale);
        gate_val1 = _mm512_mul_ps(gate_val1, gate_scale);
        up_val0 = _mm512_mul_ps(up_val0, up_scale);
        up_val1 = _mm512_mul_ps(up_val1, up_scale);
        __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit, config_.swiglu_alpha);
        __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit, config_.swiglu_alpha);
        avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
      }
    }
  }

  // Runs once per expert, after its last gate/up range: whole-row work for the down input.
  void finish_decode_expert(int expert_idx, int qlen) { derived()->prepare_decode_down_input(expert_idx, qlen); }

  static bool fused_decode_enabled() {
    static const bool enabled = cached_env_flag("KT_DECODE_FUSED");
    return enabled;
  }

  // Decode in one pool dispatch instead of three. A gate/up unit also applies the
  // activation for its range; an expert's down units start once its gate/up units
  // are done, filling the tail of the gate/up phase; the unit that finishes an
  // output range's last expert sums that range. Each output element keeps the phased
  // path's kernels and expert order, so the results are bit-identical.
  void run_fused_decode(int activated_expert, int k, const int64_t* expert_ids, const float* weights, void* output) {
    constexpr int qlen = 1;
    const int hidden = config_.hidden_size;
    float* out = static_cast<float*>(output);
    if (activated_expert == 0) {
      std::fill(out, out + hidden, 0.0f);
      return;
    }
    decode_sources_.clear();
    decode_route_weights_.clear();
    for (int j = 0; j < k; j++) {
      if (config_.should_skip_expert(expert_ids[j])) continue;
      decode_sources_.push_back(m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * hidden);
      decode_route_weights_.push_back(weights[j] * derived_const()->down_global_scale(expert_ids[j]));
    }

    const int nth_gu = T::recommended_nth(config_.intermediate_size);
    const int nth_down = T::recommended_nth(hidden);
    // The in-job sum walks whole 32-float groups; otherwise sum after the job.
    bool sum_in_job = true;
    for (int d = 0; d < nth_down && sum_in_job; d++) {
      auto [n_start, n_end] = T::split_range_n(hidden, d, nth_down);
      sum_in_job = n_start % 32 == 0 && (n_end - n_start) % 32 == 0;
    }
    if (decode_block_capacity_ < nth_down) {
      decode_block_done_.reset(new PaddedCounter[nth_down]);
      decode_block_capacity_ = nth_down;
    }
    if (decode_pair_capacity_ < activated_expert * nth_gu) {
      decode_pair_done_.reset(new PaddedCounter[activated_expert * nth_gu]);
      decode_pair_capacity_ = activated_expert * nth_gu;
    }
    for (int i = 0; i < activated_expert; i++) {
      decode_expert_done_[i].v.store(0, std::memory_order_relaxed);
      decode_expert_ready_[i].v.store(0, std::memory_order_relaxed);
    }
    for (int p = 0; p < activated_expert * nth_gu; p++) decode_pair_done_[p].v.store(0, std::memory_order_relaxed);
    for (int d = 0; d < nth_down; d++) decode_block_done_[d].v.store(0, std::memory_order_relaxed);

    // Gate and up as separate units (the second of the pair to finish applies the
    // activation) halve each unit. Kernels with few ranges per expert (256-wide FP8 / INT4
    // blocks, DSv3/Kimi: 2) need that to occupy the pool at all. Fine ranges (FP4: 16-18)
    // split only while activated_expert * nth_gu units are under 1.5 per thread; past that
    // the pool is saturated and splitting costs ~1%. Measured with 32 threads per node:
    // -13% on DSv4.1 at 2 CPU experts, -14% / -6% on MiMo at 1 / 2.
    static const int split_below = [] {
      const char* value = std::getenv("KT_DECODE_SPLIT_GATE_UP_BELOW");
      return value ? std::atoi(value) : 8;
    }();
    static const bool split_by_occupancy = cached_env_flag("KT_DECODE_SPLIT_GATE_UP_OCCUPANCY");
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const bool split_gate_up =
        nth_gu < split_below || (split_by_occupancy && 2 * activated_expert * nth_gu < 3 * pool->worker_capacity());
    const int units_per_range = split_gate_up ? 2 : 1;
    const int units_gate_up = activated_expert * nth_gu * units_per_range;
    const int units_down = activated_expert * nth_down;
    pool->do_work_stealing_job(
        units_gate_up + units_down, [](int) { T::config(); },
        [&, this](int unit) {
          if (unit < units_gate_up) {
            const int pair = unit / units_per_range;
            const int i = pair / nth_gu;
            const int ith = pair % nth_gu;
            const int expert_idx = m_expert_id_map_[i];
            const bool run_gate = !split_gate_up || unit % 2 == 0;
            const bool run_up = !split_gate_up || unit % 2 == 1;
            if (run_gate) {
              derived()->do_gate_up_gemm(false, expert_idx, ith, nth_gu, qlen);
              gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth_gu);
            }
            if (run_up) {
              derived()->do_gate_up_gemm(true, expert_idx, ith, nth_gu, qlen);
              up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth_gu);
            }
            if (split_gate_up && decode_pair_done_[pair].v.fetch_add(1, std::memory_order_acq_rel) == 0) return;
            derived()->apply_decode_activation_range(expert_idx, ith, nth_gu, qlen);
            if (decode_expert_done_[i].v.fetch_add(1, std::memory_order_acq_rel) + 1 == nth_gu) {
              derived()->finish_decode_expert(expert_idx, qlen);
              decode_expert_ready_[i].v.store(1, std::memory_order_release);
            }
            return;
          }
          // Every gate/up unit was claimed before any down unit, so this wait only
          // covers units already running on other workers.
          const int down_unit = unit - units_gate_up;
          const int i = down_unit / nth_down;
          const int d = down_unit % nth_down;
          const int expert_idx = m_expert_id_map_[i];
          while (decode_expert_ready_[i].v.load(std::memory_order_acquire) == 0) {
            _mm_pause();
          }
          derived()->do_down_gemm(expert_idx, d, nth_down, qlen);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], d, nth_down);
          if (sum_in_job &&
              decode_block_done_[d].v.fetch_add(1, std::memory_order_acq_rel) + 1 == activated_expert) {
            auto [n_start, n_end] = T::split_range_n(hidden, d, nth_down);
            sum_decode_range(n_start, n_end, out);
          }
        },
        nullptr);
    if (!sum_in_job) sum_decode_range(0, hidden, out);
  }

  // Weighted sum over the routed experts in route order, as the phased path does.
  void sum_decode_range(int n_start, int n_end, float* out) const {
    for (int e = n_start; e < n_end; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (size_t j = 0; j < decode_sources_.size(); j++) {
        __m512 weight = _mm512_set1_ps(decode_route_weights_[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32((__m512i*)(decode_sources_[j] + e), &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }
      auto f32out = (__m512*)(out + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }
  }

  // ============================================================================
  // Derived class initialization hook
  // Called after base class init() completes, allows derived classes to perform
  // their own initialization that depends on base class being fully initialized
  // ============================================================================
  bool should_allocate_expert_weights(int expert_idx) const {
    (void)expert_idx;
    return true;
  }

  // Per-tensor multipliers for the gate/up GEMM outputs (applied before the
  // activation and its clamp) and the down output (folded into the routing
  // weight), by physical expert. Exactly 1.0f, a bit-exact no-op, unless a
  // backend keeps NVFP4 global scales out of its block scales.
  float gate_global_scale(int expert_idx) const {
    (void)expert_idx;
    return 1.0f;
  }
  float up_global_scale(int expert_idx) const {
    (void)expert_idx;
    return 1.0f;
  }
  float down_global_scale(int expert_idx) const {
    (void)expert_idx;
    return 1.0f;
  }

  void derived_init() {
    // Default implementation does nothing - derived classes can override
  }

  // ============================================================================
  // Virtual points for buffer creation and size calculation
  // Default implementations use group_size (for KGroup quantization like K2)
  // Derived classes (like moe.hpp) can override to not use group_size
  // ============================================================================

  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  void apply_activation(int activated_expert, int nth, int qlen) {
    apply_activation_to(activated_expert, nth, qlen, m_local_gate_output_ptr_);
  }

  void apply_activation_to(int activated_expert, int nth, int qlen, const std::vector<ggml_bf16_t*>& destination_ptrs) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth, &destination_ptrs](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      const __m512 gate_scale = _mm512_set1_ps(derived_const()->gate_global_scale(expert_idx));
      const __m512 up_scale = _mm512_set1_ps(derived_const()->up_global_scale(expert_idx));
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* destination_ptr = &destination_ptrs[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          gate_val0 = _mm512_mul_ps(gate_val0, gate_scale);
          gate_val1 = _mm512_mul_ps(gate_val1, gate_scale);
          up_val0 = _mm512_mul_ps(up_val0, up_scale);
          up_val1 = _mm512_mul_ps(up_val1, up_scale);
          __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit, config_.swiglu_alpha);
          __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit, config_.swiglu_alpha);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(destination_ptr + j));
        }
      }
    };

    if (activated_expert == 0) {
      return;
    }

    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
        fn(task_id);
      }
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_BASE derived classes
// ============================================================================

// Parts must be constructed as Derived (as the AVX2 TP does): derived() casts
// to it, so building only the base is UB once Derived has data members.
template <class T, class Derived>
class TP_MOE<AMX_MOE_BASE<T, Derived>> : public TP_MOE_Common<AMX_MOE_BASE<T, Derived>, Derived> {
 public:
  using TP_MOE_Common<AMX_MOE_BASE<T, Derived>, Derived>::TP_MOE_Common;

  // Default load_weights implementation - can be overridden by derived TP_MOE classes
  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    throw std::runtime_error("Not Implemented");
  }

  // Sums elements [e_begin, e_end) of one token's TP partials into output, in TP order,
  // accumulating in registers and converting to BF16 once.
  void merge_token(int token_nth, int e_begin, int e_end, void* output, bool incremental) {
    const float* base = this->local_output_numa[0] + (size_t)token_nth * this->tp_configs[0].hidden_size;
    ggml_bf16_t* dst = (ggml_bf16_t*)output + (size_t)token_nth * this->config.hidden_size;
    for (int e = e_begin; e < e_end; e += 32) {
      __m512 x0, x1;
      if (incremental) {
        avx512_32xbf16_to_32xfp32((__m512i*)(dst + e), &x0, &x1);
        x0 = _mm512_add_ps(x0, _mm512_loadu_ps(base + e));
        x1 = _mm512_add_ps(x1, _mm512_loadu_ps(base + e + 16));
      } else {
        x0 = _mm512_loadu_ps(base + e);
        x1 = _mm512_loadu_ps(base + e + 16);
      }
      for (int i = 1; i < this->tp_count; i++) {
        const float* src = this->local_output_numa[i] + (size_t)token_nth * this->tp_configs[i].hidden_size;
        x0 = _mm512_add_ps(x0, _mm512_loadu_ps(src + e));
        x1 = _mm512_add_ps(x1, _mm512_loadu_ps(src + e + 16));
      }
      avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)(dst + e));
    }
  }

  // Decode-size batches merge inside the NUMA job: one thread summing every remote
  // partial costs about 6 us per layer, a quarter of hidden_size per node about 1.5.
  bool merges_in_numa_job(int qlen) const override { return qlen < 10; }

  void merge_results_part(int qlen, void* output, bool incremental, int part, int parts) override {
    const int groups = this->config.hidden_size / 32;
    const int begin = static_cast<int>(static_cast<int64_t>(groups) * part / parts) * 32;
    const int end = static_cast<int>(static_cast<int64_t>(groups) * (part + 1) / parts) * 32;
    for (int i = 0; i < qlen; i++) {
      merge_token(i, begin, end, output, incremental);
    }
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    const int hidden = this->config.hidden_size;
    auto merge_fn = [this, output, incremental, hidden](int token_nth) {
      merge_token(token_nth, 0, hidden, output, incremental);
    };

    auto pool = this->config.pool;
    const int tp_count = this->tp_count;
    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) {
        merge_fn(i);
      }
      return;
    }
    // Every NUMA pool merges a contiguous token range, so the remote partial
    // reads spread over all nodes instead of node 0's threads.
    const int parts = tp_count;
    pool->dispense_backend()->do_numa_job([&](int numa_id) {
      if (numa_id >= parts) return;
      const int begin = static_cast<int>(static_cast<int64_t>(qlen) * numa_id / parts);
      const int end = static_cast<int>(static_cast<int64_t>(qlen) * (numa_id + 1) / parts);
      if (begin < end) {
        pool->get_subpool(numa_id)->do_work_stealing_job(
            end - begin, nullptr, [&](int i) { merge_fn(begin + i); }, nullptr);
      }
    });
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_MOE_BASE_H
