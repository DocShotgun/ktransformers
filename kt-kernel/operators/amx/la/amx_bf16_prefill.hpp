// The fork's blocked AMX prefill GEMM over BF16 tiles, shared by the FP4 and the
// block-wise FP8 kernels: C = A x B for one task's columns [n_start, n_end).
//
// Blocking: for every K block the task's weight panels are expanded to BF16 VNNI
// tiles once, then each 32-row M block reads its activation rows in place and
// sweeps all panels with a 2x2 tile kernel, so neither operand is re-streamed per
// panel. The FP32 accumulators of an M chunk live in per-worker scratch.
//
// Accumulation, chosen by P::kScaledGroups; neither depends on the blocking:
// - unscaled (scales folded into the weight tiles): the tiles accumulate across K,
//   and between K blocks they round-trip through the FP32 scratch, which is exact;
// - scaled groups (block scales): each scale group's tile sum is folded as
//   c = fma(sum, scale, c), groups in K order, as apply_scale_kgroup does.
//
// P supplies the format (see the FP4 and FP8 kernels' AmxPrefill): kZeroPadTail,
// k_block(), pack(), a_rows(), a_stride() and copy_tail(), then store_tiles() when
// unscaled, or group_size(), scale() and store_block() with scaled groups.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "amx_config.hpp"
#include "llama.cpp/ggml-impl.h"

namespace amx::bf16_prefill {

// Rows of an M block, columns of a weight panel, and K of one VNNI chunk.
constexpr int kTile = 32;
// Rows of one accumulator chunk: bounds the per-worker FP32 scratch.
constexpr int kMChunk = 512;

template <class T>
inline T* align64(T* p) {
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(p) + 63) & ~uintptr_t(63));
}

template <class P>
inline void run(const P& p, int m, int k, int n_start, int n_end) {
  configure_bf16_tiles();
  const int panels = (n_end - n_start) / kTile;
  const int k_block = p.k_block();
  // Per-worker scratch: one K block of the task's weight panels, the accumulators
  // of one M chunk, and the zero-padded rows of a short last M block. No expanded
  // copy of the model is retained in host RAM.
  thread_local std::vector<uint8_t> packed_storage;
  packed_storage.resize(static_cast<size_t>(panels) * k_block * 64 + 63);
  uint32_t* packed = reinterpret_cast<uint32_t*>(align64(packed_storage.data()));
  thread_local std::vector<float> acc_storage;
  thread_local std::vector<ggml_bf16_t> tail_storage;
  alignas(64) float block[kTile][kTile];

  for (int chunk_begin = 0; chunk_begin < m; chunk_begin += kMChunk) {
    const int chunk_end = std::min(m, chunk_begin + kMChunk);
    float* acc = nullptr;
    if (P::kScaledGroups || k > k_block) {
      acc_storage.resize(static_cast<size_t>(chunk_end - chunk_begin + kTile - 1) / kTile * panels * kTile * kTile +
                         16);
      acc = align64(acc_storage.data());
    }
    for (int k_begin = 0; k_begin < k; k_begin += k_block) {
      const int k_len = std::min(k_block, k - k_begin);
      const size_t panel_words = static_cast<size_t>(k_len / kTile) * 16 * 32;
      for (int panel = 0; panel < panels; ++panel) {
        p.pack(n_start + panel * kTile, k_begin, k_len, packed + panel * panel_words);
      }
      const bool first = k_begin == 0;
      const bool last = k_begin + k_len >= k;
      for (int m_pos = chunk_begin; m_pos < chunk_end; m_pos += kTile) {
        const int rows = std::min(kTile, m - m_pos);
        // A short last block of in-place rows copies them as [32-K chunk][32 rows][32 K],
        // so no tile load reads past the expert's rows. Rows past `rows` are zero with
        // P::kZeroPadTail and stale otherwise: they feed only output rows that are
        // discarded, since each output row depends on its own activation row alone.
        const ggml_bf16_t* tail = nullptr;
        if (rows < kTile && p.copy_tail()) {
          tail_storage.resize(static_cast<size_t>(kTile) * k_block + 32);
          ggml_bf16_t* copy = align64(tail_storage.data());
          for (int chunk = 0; chunk < k_len / kTile; ++chunk) {
            const int kk = k_begin + chunk * kTile;
            const ggml_bf16_t* source = p.a_rows(m_pos, kk);
            const size_t stride = p.a_stride(kk);
            ggml_bf16_t* destination = copy + static_cast<size_t>(chunk) * kTile * kTile;
            for (int row = 0; row < rows; ++row) {
              _mm512_store_si512(destination + row * kTile, _mm512_loadu_si512(source + row * stride));
            }
            if constexpr (P::kZeroPadTail) {
              for (int row = rows; row < kTile; ++row) {
                _mm512_store_si512(destination + row * kTile, _mm512_setzero_si512());
              }
            }
          }
          tail = copy;
        }
        // Adds the 32-K chunks [from, to) of this K block into tiles 4..7.
        const auto multiply = [&](const uint32_t* b, int from, int to) {
          for (int chunk = from; chunk < to; ++chunk) {
            const ggml_bf16_t* a;
            int a_stride;
            if (tail) {
              a = tail + static_cast<size_t>(chunk) * kTile * kTile;
              a_stride = kTile * sizeof(ggml_bf16_t);
            } else {
              const int kk = k_begin + chunk * kTile;
              a = p.a_rows(m_pos, kk);
              a_stride = p.a_stride(kk) * sizeof(ggml_bf16_t);
            }
            _tile_loadd(0, a, a_stride);
            _tile_loadd(1, reinterpret_cast<const uint8_t*>(a) + 16 * a_stride, a_stride);
            _tile_loadd(2, b + chunk * 16 * 32, 128);
            _tile_loadd(3, b + chunk * 16 * 32 + 16, 128);
            _tile_dpbf16ps(4, 0, 2);
            _tile_dpbf16ps(5, 0, 3);
            _tile_dpbf16ps(6, 1, 2);
            _tile_dpbf16ps(7, 1, 3);
          }
        };
        const size_t acc_block = (m_pos - chunk_begin) / kTile;
        for (int panel = 0; panel < panels; ++panel) {
          const int n_pos = n_start + panel * kTile;
          float* c = acc ? acc + (acc_block * panels + panel) * kTile * kTile : nullptr;
          const uint32_t* b = packed + panel * panel_words;
          if constexpr (P::kScaledGroups) {
            const int group_size = p.group_size();
            for (int group_begin = k_begin; group_begin < k_begin + k_len; group_begin += group_size) {
              _tile_zero(4);
              _tile_zero(5);
              _tile_zero(6);
              _tile_zero(7);
              const int group_end = std::min(k, group_begin + group_size);
              multiply(b, (group_begin - k_begin) / kTile, (group_end - k_begin) / kTile);
              _tile_stored(4, &block[0][0], kTile * sizeof(float));
              _tile_stored(5, &block[0][16], kTile * sizeof(float));
              _tile_stored(6, &block[16][0], kTile * sizeof(float));
              _tile_stored(7, &block[16][16], kTile * sizeof(float));
              const __m512 scale = _mm512_set1_ps(p.scale(n_pos, group_begin));
              const float* sum = &block[0][0];
              const bool first_group = group_begin == 0;
              for (int i = 0; i < kTile * kTile; i += 16) {
                const __m512 previous = first_group ? _mm512_setzero_ps() : _mm512_load_ps(c + i);
                _mm512_store_ps(c + i, _mm512_fmadd_ps(_mm512_load_ps(sum + i), scale, previous));
              }
            }
            if (last) p.store_block(m_pos, n_pos, c);
          } else {
            if (first) {
              _tile_zero(4);
              _tile_zero(5);
              _tile_zero(6);
              _tile_zero(7);
            } else {
              _tile_loadd(4, c, kTile * sizeof(float));
              _tile_loadd(5, c + 16, kTile * sizeof(float));
              _tile_loadd(6, c + 16 * kTile, kTile * sizeof(float));
              _tile_loadd(7, c + 16 * kTile + 16, kTile * sizeof(float));
            }
            multiply(b, 0, k_len / kTile);
            if (!last) {
              _tile_stored(4, c, kTile * sizeof(float));
              _tile_stored(5, c + 16, kTile * sizeof(float));
              _tile_stored(6, c + 16 * kTile, kTile * sizeof(float));
              _tile_stored(7, c + 16 * kTile + 16, kTile * sizeof(float));
              continue;
            }
            p.store_tiles(m_pos, n_pos, rows, block);
          }
        }
      }
    }
  }
}

}  // namespace amx::bf16_prefill
