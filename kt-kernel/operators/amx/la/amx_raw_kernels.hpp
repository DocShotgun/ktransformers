#ifndef AMX_RAW_KERNELS_HPP
#define AMX_RAW_KERNELS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amx_config.hpp"
#include "amx_raw_buffers.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"

namespace amx {

struct GemmKernel224BF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2;
  static constexpr int TILE_M = 16;
  static constexpr int TILE_K = 32;
  static constexpr int TILE_N = 16;
  static constexpr int VNNI_BLK = 2;

  static constexpr int M_STEP = TILE_M * 2;
  static constexpr int N_STEP = TILE_N * 2;
  static constexpr int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;
  static std::string name() { return "BF16"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
#ifdef HAVE_AMX
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 32
    for (int i = 0; i < 2; i++) tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 32
    for (int i = 2; i < 4; i++) tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
    for (int i = 4; i < 8; i++) tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
#endif
  }

  static void load_a(dt* a, size_t lda) {
#ifdef HAVE_AMX
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
#else
    (void)a;
    (void)lda;
#endif
  }

  static void load_b(dt* b, size_t ldb) {
#ifdef HAVE_AMX
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
#else
    (void)b;
    (void)ldb;
#endif
  }

  static void clean_c() {
#ifdef HAVE_AMX
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
#endif
  }

  static void load_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void store_c(output_t* c, size_t ldc) {
#ifdef HAVE_AMX
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
#else
    (void)c;
    (void)ldc;
#endif
  }

  static void run_tile() {
#ifdef HAVE_AMX
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
#endif
  }
  using BufferA = BufferABF16Impl<GemmKernel224BF16>;
  using BufferB = BufferBBF16Impl<GemmKernel224BF16>;
  using BufferC = BufferCFP32Impl<GemmKernel224BF16>;

  // Basic AVX kernel for BF16: process entire K_BLOCK
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at the start of k_block
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    // Process entire K_BLOCK
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512bh* b512 = (__m512bh*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      for (int m_i = 0; m_i < m_block_end; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma, b512[k_i]);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma, b512[16 + k_i]);
        }
      }
    }
  }

  // Optimized AVX kernel: process 4 k_i at once, unroll m rows by 2
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb) {
    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at the start of k_block
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    // Process entire K_BLOCK
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      int32_t* a32 = (int32_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512bh* b512 = (__m512bh*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      // Process 4 k_i at once - load B vectors and reuse across all m rows
      for (int k_i = 0; k_i < 16; k_i += 4) {
        // Load 4 B vector pairs (lo and hi for each k_i)
        __m512bh b0_lo = b512[k_i];
        __m512bh b0_hi = b512[16 + k_i];
        __m512bh b1_lo = b512[k_i + 1];
        __m512bh b1_hi = b512[16 + k_i + 1];
        __m512bh b2_lo = b512[k_i + 2];
        __m512bh b2_hi = b512[16 + k_i + 2];
        __m512bh b3_lo = b512[k_i + 3];
        __m512bh b3_hi = b512[16 + k_i + 3];

        // Process m rows - unroll by 2 for better ILP
        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          // Load A values for 2 rows, 4 k_i each
          __m512bh ma0_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          __m512bh ma1_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 1]);
          __m512bh ma2_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 2]);
          __m512bh ma3_0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 3]);
          __m512bh ma0_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i]);
          __m512bh ma1_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 1]);
          __m512bh ma2_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 2]);
          __m512bh ma3_1 = (__m512bh)_mm512_set1_epi32(a32[(m_i + 1) * 16 + k_i + 3]);

          // Process row 0
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0_0, b0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0_0, b0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1_0, b1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1_0, b1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2_0, b2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2_0, b2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3_0, b3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3_0, b3_hi);

          // Process row 1
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma0_1, b0_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma0_1, b0_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma1_1, b1_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma1_1, b1_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma2_1, b2_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma2_1, b2_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma3_1, b3_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma3_1, b3_hi);
        }
        // Handle remaining row
        for (; m_i < m_block_end; m_i++) {
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 1]);
          __m512bh ma2 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 2]);
          __m512bh ma3 = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i + 3]);

          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0, b0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0, b0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1, b1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1, b1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2, b2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2, b2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3, b3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3, b3_hi);
        }
      }
    }
  }

  // AMX kernel for BF16: process entire K_BLOCK using AMX tiles
  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                         BufferB* bb) {
    if (k_block_begin == 0) {
      clean_c();
    } else {
      load_c(c, N_STEP * sizeof(float));
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K_STEP * sizeof(ggml_bf16_t));
      load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K_STEP * sizeof(ggml_bf16_t));
      run_tile();
    }

    store_c(c, N_STEP * sizeof(float));
  }
};

// FP8 (e4m3) AMX kernel that mirrors the GemmKernel224BF16 interface.
struct GemmKernel224FP8 {
  using fp8_t = uint8_t;
  using output_t = float;

  static constexpr double ELEMENT_SIZE = 1.0;
  static constexpr int TILE_M = 16;
  static constexpr int TILE_K = 32;
  static constexpr int TILE_N = 16;
  static constexpr int VNNI_BLK = 2;

  static constexpr int M_STEP = TILE_M * 2;
  static constexpr int N_STEP = TILE_N * 2;
  static constexpr int K_STEP = TILE_K;

  static inline const int BLOCK_SIZE = 128;  // 128 x 128 block quantization
  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "FP8"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {}

  // FP8->BF16 conversion lookup tables (public for reuse by GemmKernel224FP8PerChannel)
  alignas(64) static constexpr uint8_t bf16_hi_0_val[64] = {
      0x00, 0x3b, 0x3b, 0x3b, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c, 0x3c,
      0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d,
      0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e,
      0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
  };
  alignas(64) static constexpr uint8_t bf16_hi_1_val[64] = {
      0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
      0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41, 0x41,
      0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42,
      0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43, 0x43,
  };
  alignas(64) static constexpr uint8_t bf16_lo_0_val[64] = {
      0x00, 0x00, 0x80, 0xc0, 0x00, 0x20, 0x40, 0x60, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
  };
  alignas(64) static constexpr uint8_t bf16_lo_1_val[64] = {
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
      0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
  };
  // _mm512_set1_epi8 is not constexpr; keep it as a static cached value
  alignas(64) static const __m512i sign_mask_val;
  static inline __m512i bf16_hi_0_mask() { return _mm512_load_si512((__m512i const*)bf16_hi_0_val); }
  static inline __m512i bf16_hi_1_mask() { return _mm512_load_si512((__m512i const*)bf16_hi_1_val); }
  static inline __m512i bf16_lo_0_mask() { return _mm512_load_si512((__m512i const*)bf16_lo_0_val); }
  static inline __m512i bf16_lo_1_mask() { return _mm512_load_si512((__m512i const*)bf16_lo_1_val); }
  static inline __m512i sign_mask() { return _mm512_set1_epi8(0x80); }
  // Multi-row calls read the expert-grouped row-major BF16 rows in place (the blocked
  // copies cost about 18% of a long prefill); single-row decode stages one padded block.
  struct BufferA : public BufferABF16Impl<GemmKernel224FP8> {
    using Base = BufferABF16Impl<GemmKernel224FP8>;
    using Base::a;
    using Base::k;
    // Only a single-row call writes the pool, as one zero-padded M block.
    static size_t required_size(int /*max_m*/, int k) { return Base::required_size(M_STEP, k); }
    // True while `a` aliases the caller's row-major rows instead of the blocked pool.
    bool row_major = false;
    ggml_bf16_t* pool = nullptr;
    BufferA(int max_m_, int k_, void* ptr) : Base(max_m_, k_, ptr), pool(a) {}
    void set_data(void* ptr) {
      Base::set_data(ptr);
      pool = a;
      row_major = false;
    }
    void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
      if (m > 1) {
        assert(m <= max_m && ith == 0 && nth == 1);
        a = src;
        row_major = true;
      } else {
        a = pool;
        row_major = false;
        Base::from_mat(m, src, ith, nth);
      }
    }
    // Reads the caller's row-major rows (stride k) in place, a single row included. The
    // rows must stay unchanged until the GEMMs that read this buffer are done.
    void alias_rows(ggml_bf16_t* src) {
      a = src;
      row_major = true;
    }
    // Elements between consecutive rows at the pointer get_submat returns.
    int row_stride() const { return row_major ? k : K_STEP; }
    ggml_bf16_t* get_submat(int m, int k_, int m_begin, int k_begin) {
      if (row_major) return a + static_cast<size_t>(m_begin) * k + k_begin;
      return Base::get_submat(m, k_, m_begin, k_begin);
    }
  };
  using BufferB = BufferBFP8Impl<GemmKernel224FP8>;
  using BufferC = BufferCFP32ReduceImpl<GemmKernel224FP8>;

  static inline std::pair<__m512i, __m512i> fp8x64_to_bf16x64(__m512i bfp8_512) {
    // fp8->bf16
    __m512i b_hi = _mm512_permutex2var_epi8(bf16_hi_0_mask(), bfp8_512, bf16_hi_1_mask());
    __m512i b_lo = _mm512_permutex2var_epi8(bf16_lo_0_mask(), bfp8_512, bf16_lo_1_mask());
    b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask(), bfp8_512), b_hi);
    __m512i bbf16_0 = _mm512_unpacklo_epi8(b_lo, b_hi);
    __m512i bbf16_1 = _mm512_unpackhi_epi8(b_lo, b_hi);
    return {bbf16_0, bbf16_1};
  }

  // ==========================================================================
  // FP8 -> BF16 AMX prefill staging. Hosts without AMX-FP8 tiles widen each
  // 32-output-channel FP8 weight block losslessly to BF16 in the FP4 VNNI
  // scratch layout (pair-major uint32 words reused by every 32-token M tile),
  // accumulate unscaled tile sums per k_group into the reduce buffer, and fold
  // the 128x128 scales with apply_scale_kgroup exactly like the AVX path.
  // ==========================================================================

  static void configure_fp8_amx() {
    enable_amx();
    TileConfig config;
    for (int tile = 0; tile < 8; ++tile) config.set_row_col(tile, 16, 64);
    config.set_config();
  }

  // Expand one 32-output-channel FP8 tile to AMX's VNNI B layout: 16 k-pair
  // rows of 32 adjacent BF16 pairs (word pair*32 + channel holds BF16 K
  // 2pair / 2pair+1 of channel, matching pack_fp4_amx_b_scalar's scratch
  // contract). Weight tiles keep BufferBFP8Impl's scramble (word 8*pair +
  // mat_offset[row>>2], lane row&3); unpack once per n_pos block instead of
  // decoding FP8 weights for every token tile.
  //
  // _mm512_unpack{lo,hi}_epi8 inside fp8x64_to_bf16x64 interleaves within each
  // 128-bit lane, which looks like it needs a compensating shuffle. It does not:
  // the required input permutation works out to the identity, because the
  // mat_offset scramble that BufferBFP8Impl::from_mat applies is exactly the
  // inverse of that interleave. So the staged bytes go in unpermuted and the
  // whole 32-output-channel tile is one load + one widen per 64 weights.
  //
  // Templated on the buffer so the per-channel kernel, whose BufferB differs
  // only in how it stores scales, can share the tile walk.
  template <typename B>
  static void pack_fp8_amx_b(int n, int k, int n_pos, B* bb, uint32_t* packed) {
    pack_fp8_amx_b(n, k, n_pos, bb, packed, 0, k);
  }

  // The same for K in [k_begin, k_begin + k_len), packed from `packed` on.
  template <typename B>
  static void pack_fp8_amx_b(int n, int k, int n_pos, B* bb, uint32_t* packed, int k_begin, int k_len) {
    const int chunk_count = k_len / K_STEP;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      uint32_t* destination = packed + static_cast<size_t>(chunk) * 16 * 32;
      const char* source = reinterpret_cast<const char*>(bb->get_submat(n, k, n_pos, k_begin + chunk * K_STEP));
      for (int pair = 0; pair < 16; ++pair) {
        const __m512i raw = _mm512_loadu_si512(source + 64 * pair);
        auto [lo, hi] = GemmKernel224FP8::fp8x64_to_bf16x64(raw);
        _mm512_storeu_si512(destination + pair * 32, lo);
        _mm512_storeu_si512(destination + pair * 32 + 16, hi);
      }
    }
  }

  // K extent of one blocking step of fp8_mat_mat_kgroup_amx, rounded down to whole scale
  // groups, and the rows of one accumulator chunk (bounds the per-worker FP32 scratch).
  static constexpr int AMX_K_BLOCK = 1024;
  static constexpr int AMX_M_CHUNK = 512;

  // AMX prefill GEMM for one task's columns [n_start, n_end) with block scales.
  //
  // Within a k_group the tiles accumulate the unscaled products; after it each output
  // element takes c = fma(sum, scale, c), groups in K order, as apply_scale_kgroup does, so
  // the result does not depend on the blocking.
  //
  // Blocking, as fp4_mat_mat_kgroup_amx: for every K block the task's weight panels are
  // widened once, then each 32-row M block sweeps all panels while its activation slice is
  // still in L2. The FP32 accumulators of an M chunk live in per-worker scratch.
  //
  // Output: with dst, rows go straight to the row-major BF16 matrix (stride n), rounded as
  // BufferC::to_mat rounds, and bc is not touched; otherwise the FP32 blocks land in bc.
  static void fp8_mat_mat_kgroup_amx(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc,
                                     int ith, int nth, ggml_bf16_t* dst = nullptr) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    configure_fp8_amx();

    const int panels = (n_end - n_start) / N_STEP;
    const int k_block = std::min(k, std::max(1, AMX_K_BLOCK / k_group_size) * k_group_size);
    const auto align = [](auto* p) {
      return reinterpret_cast<decltype(p)>((reinterpret_cast<uintptr_t>(p) + 63) & ~uintptr_t(63));
    };
    // Per-worker scratch: one K block of the task's widened weight panels and the FP32
    // accumulators of one M chunk. No expanded copy of the model is retained.
    thread_local std::vector<uint8_t> packed_storage;
    packed_storage.resize(static_cast<size_t>(panels) * k_block * 64 + 63);
    uint32_t* packed = reinterpret_cast<uint32_t*>(align(packed_storage.data()));
    thread_local std::vector<float> acc_storage;
    thread_local std::vector<ggml_bf16_t> tail_storage;
    alignas(64) float sum[M_STEP * N_STEP];

    for (int chunk_begin = 0; chunk_begin < m; chunk_begin += AMX_M_CHUNK) {
      const int chunk_end = std::min(m, chunk_begin + AMX_M_CHUNK);
      const int m_blocks = (chunk_end - chunk_begin + M_STEP - 1) / M_STEP;
      acc_storage.resize(static_cast<size_t>(m_blocks) * panels * M_STEP * N_STEP + 16);
      float* acc = align(acc_storage.data());
      for (int k_begin = 0; k_begin < k; k_begin += k_block) {
        const int k_len = std::min(k_block, k - k_begin);
        const size_t panel_words = static_cast<size_t>(k_len / K_STEP) * 16 * 32;
        for (int panel = 0; panel < panels; ++panel) {
          pack_fp8_amx_b(n, k, n_start + panel * N_STEP, bb, packed + panel * panel_words, k_begin, k_len);
        }
        const bool last = k_begin + k_len >= k;
        for (int m_pos = chunk_begin; m_pos < chunk_end; m_pos += M_STEP) {
          const int m_block = (m_pos - chunk_begin) / M_STEP;
          // Activation tiles of this M block: in place, or for the short last block of an
          // aliased expert a zero-padded [32-K chunk][32 rows][32 K] copy, so no tile load
          // reads past the expert's rows.
          const ggml_bf16_t* a_base = ba->get_submat(m, k, m_pos, k_begin);
          int a_stride = ba->row_stride();
          size_t a_chunk = ba->row_major ? K_STEP : static_cast<size_t>(M_STEP) * K_STEP;
          const int rows = std::min(M_STEP, m - m_pos);
          if (ba->row_major && rows < M_STEP) {
            tail_storage.resize(static_cast<size_t>(M_STEP) * k_len + 32);
            ggml_bf16_t* tail = align(tail_storage.data());
            for (int chunk = 0; chunk < k_len / K_STEP; ++chunk) {
              ggml_bf16_t* destination = tail + static_cast<size_t>(chunk) * M_STEP * K_STEP;
              for (int row = 0; row < M_STEP; ++row) {
                const __m512i value =
                    row < rows ? _mm512_loadu_si512(a_base + static_cast<size_t>(row) * k + chunk * K_STEP)
                               : _mm512_setzero_si512();
                _mm512_store_si512(destination + row * K_STEP, value);
              }
            }
            a_base = tail;
            a_stride = K_STEP;
            a_chunk = static_cast<size_t>(M_STEP) * K_STEP;
          }
          for (int panel = 0; panel < panels; ++panel) {
            const int n_pos = n_start + panel * N_STEP;
            float* c = acc + (static_cast<size_t>(m_block) * panels + panel) * M_STEP * N_STEP;
            const uint32_t* b = packed + panel * panel_words;
            for (int group_begin = k_begin; group_begin < k_begin + k_len; group_begin += k_group_size) {
              _tile_zero(4);
              _tile_zero(5);
              _tile_zero(6);
              _tile_zero(7);
              const int group_end = std::min(k, group_begin + k_group_size);
              for (int chunk_k = group_begin; chunk_k < group_end; chunk_k += K_STEP) {
                const size_t chunk = static_cast<size_t>((chunk_k - k_begin) / K_STEP);
                const ggml_bf16_t* a = a_base + chunk * a_chunk;
                const uint32_t* bt = b + chunk * 16 * 32;
                _tile_loadd(0, a, a_stride * sizeof(ggml_bf16_t));
                _tile_loadd(1, a + 16 * a_stride, a_stride * sizeof(ggml_bf16_t));
                _tile_loadd(2, bt, 128);
                _tile_loadd(3, bt + 16, 128);
                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(5, 0, 3);
                _tile_dpbf16ps(6, 1, 2);
                _tile_dpbf16ps(7, 1, 3);
              }
              _tile_stored(4, sum, N_STEP * sizeof(float));
              _tile_stored(5, sum + TILE_N, N_STEP * sizeof(float));
              _tile_stored(6, sum + 16 * N_STEP, N_STEP * sizeof(float));
              _tile_stored(7, sum + 16 * N_STEP + TILE_N, N_STEP * sizeof(float));
              const __m512 scale = _mm512_set1_ps(*bb->get_scale(n, n_pos, k, group_begin));
              const bool first = group_begin == 0;
              for (int i = 0; i < M_STEP * N_STEP; i += 16) {
                const __m512 previous = first ? _mm512_setzero_ps() : _mm512_load_ps(c + i);
                _mm512_store_ps(c + i, _mm512_fmadd_ps(_mm512_load_ps(sum + i), scale, previous));
              }
            }
            if (last) store_fp8_amx_result(m, n, m_pos, n_pos, c, bc, dst);
          }
        }
      }
    }
  }

  // Epilogue of fp8_mat_mat_kgroup_amx for the 32x32 FP32 block c at (m_pos, n_pos): BF16
  // rows into dst, or the block into bc's layout for to_mat.
  static void store_fp8_amx_result(int m, int n, int m_pos, int n_pos, float* c, BufferC* bc, ggml_bf16_t* dst) {
    const int rows = std::min(M_STEP, m - m_pos);
    if (dst) {
      for (int row = 0; row < rows; ++row) {
        avx512_32xfp32_to_32xbf16(reinterpret_cast<__m512*>(c + row * N_STEP),
                                  reinterpret_cast<__m512*>(c + row * N_STEP + TILE_N),
                                  reinterpret_cast<__m512i*>(dst + static_cast<size_t>(m_pos + row) * n + n_pos));
      }
      return;
    }
    float* result = bc->get_submat(m, n, m_pos, n_pos);
    std::memcpy(result, c, sizeof(float) * M_STEP * N_STEP);
  }

  // Optimized AVX kernel: process entire k_group_size
  // Load all data first, then convert all, then compute all
  // This gives compiler more freedom to schedule instructions
  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_group_begin, float* c, BufferA* ba,
                         BufferB* bb, int k_group_size) {
    const __m512i bf16_hi_0_val = bf16_hi_0_mask();
    const __m512i bf16_hi_1_val = bf16_hi_1_mask();
    const __m512i bf16_lo_0_val = bf16_lo_0_mask();
    const __m512i bf16_lo_1_val = bf16_lo_1_mask();
    const __m512i sign_mask_val = sign_mask();

    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at the start
    for (int m_i = 0; m_i < m_block_end; m_i++) {
      c512[m_i * 2] = _mm512_setzero_ps();
      c512[m_i * 2 + 1] = _mm512_setzero_ps();
    }

    // Process entire k_group_size
    for (int k_begin = 0; k_begin < k_group_size && k_group_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_group_begin + k_begin);
      const int a_stride = ba->row_stride();
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_group_begin + k_begin);

      for (int m_i = 0; m_i < m_block_end; m_i++) {
        // Process 2 k_i per iteration
        for (int k_i = 0; k_i < 16; k_i += 2) {
          // Load A vectors
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + k_i * 2]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 1) * 2]);

          // Load B matrices
          __m512i bfp8_0 = bfp8_512[k_i];
          __m512i bfp8_1 = bfp8_512[k_i + 1];

          // Convert FP8 -> BF16 for all
          __m512i b_hi_0 = _mm512_permutex2var_epi8(bf16_hi_0_val, bfp8_0, bf16_hi_1_val);
          __m512i b_lo_0 = _mm512_permutex2var_epi8(bf16_lo_0_val, bfp8_0, bf16_lo_1_val);
          b_hi_0 = _mm512_or_si512(_mm512_and_si512(sign_mask_val, bfp8_0), b_hi_0);

          __m512i b_hi_1 = _mm512_permutex2var_epi8(bf16_hi_0_val, bfp8_1, bf16_hi_1_val);
          __m512i b_lo_1 = _mm512_permutex2var_epi8(bf16_lo_0_val, bfp8_1, bf16_lo_1_val);
          b_hi_1 = _mm512_or_si512(_mm512_and_si512(sign_mask_val, bfp8_1), b_hi_1);

          // Compute dpbf16 for all
          __m512bh bbf16_0_0 = (__m512bh)_mm512_unpacklo_epi8(b_lo_0, b_hi_0);
          __m512bh bbf16_1_0 = (__m512bh)_mm512_unpackhi_epi8(b_lo_0, b_hi_0);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0, bbf16_0_0);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0, bbf16_1_0);

          __m512bh bbf16_0_1 = (__m512bh)_mm512_unpacklo_epi8(b_lo_1, b_hi_1);
          __m512bh bbf16_1_1 = (__m512bh)_mm512_unpackhi_epi8(b_lo_1, b_hi_1);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1, bbf16_0_1);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1, bbf16_1_1);
        }
      }
    }
  }

  // Optimized AVX kernel: process 4 k_i at once, convert B once and reuse for all m rows
  // This version achieved ~493 GB/s - restoring as baseline for further optimization
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_group_begin, float* c, BufferA* ba,
                           BufferB* bb, int k_group_size) {
    const __m512i bf16_hi_0 = bf16_hi_0_mask();
    const __m512i bf16_hi_1 = bf16_hi_1_mask();
    const __m512i bf16_lo_0 = bf16_lo_0_mask();
    const __m512i bf16_lo_1 = bf16_lo_1_mask();
    const __m512i sign_mask_v = sign_mask();

    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator
    for (int m_i = 0; m_i < m_block_end; m_i++) {
      c512[m_i * 2] = _mm512_setzero_ps();
      c512[m_i * 2 + 1] = _mm512_setzero_ps();
    }

    // Process entire k_group_size
    for (int k_begin = 0; k_begin < k_group_size && k_group_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_group_begin + k_begin);
      const int a_stride = ba->row_stride();
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_group_begin + k_begin);

      // Process 4 k_i at once - convert B and reuse across all m rows
      for (int k_i = 0; k_i < 16; k_i += 4) {
        // Load 4 B vectors
        __m512i bfp8_0 = bfp8_512[k_i];
        __m512i bfp8_1 = bfp8_512[k_i + 1];
        __m512i bfp8_2 = bfp8_512[k_i + 2];
        __m512i bfp8_3 = bfp8_512[k_i + 3];

        // Convert all 4 FP8 -> BF16
        __m512i b_hi, b_lo;

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_0),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_0, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_0, bf16_lo_1);
        __m512bh bbf16_0_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_0_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_1),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_1, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_1, bf16_lo_1);
        __m512bh bbf16_1_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_1_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_2),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_2, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_2, bf16_lo_1);
        __m512bh bbf16_2_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_2_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_3),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_3, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_3, bf16_lo_1);
        __m512bh bbf16_3_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_3_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        // Process m rows - unroll by 2 for better ILP
        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          // Load A values for 2 rows
          __m512bh ma0_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + k_i * 2]);
          __m512bh ma1_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 1) * 2]);
          __m512bh ma2_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 2) * 2]);
          __m512bh ma3_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 3) * 2]);
          __m512bh ma0_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * a_stride + k_i * 2]);
          __m512bh ma1_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * a_stride + (k_i + 1) * 2]);
          __m512bh ma2_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * a_stride + (k_i + 2) * 2]);
          __m512bh ma3_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * a_stride + (k_i + 3) * 2]);

          // Process row 0, then row 1 - sequential to avoid dependencies
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0_0, bbf16_0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0_0, bbf16_0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1_0, bbf16_1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1_0, bbf16_1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2_0, bbf16_2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2_0, bbf16_2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3_0, bbf16_3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3_0, bbf16_3_hi);

          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma0_1, bbf16_0_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma0_1, bbf16_0_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma1_1, bbf16_1_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma1_1, bbf16_1_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma2_1, bbf16_2_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma2_1, bbf16_2_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma3_1, bbf16_3_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma3_1, bbf16_3_hi);
        }
        // Handle remaining row
        for (; m_i < m_block_end; m_i++) {
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + k_i * 2]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 1) * 2]);
          __m512bh ma2 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 2) * 2]);
          __m512bh ma3 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * a_stride + (k_i + 3) * 2]);

          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0, bbf16_0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0, bbf16_0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1, bbf16_1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1, bbf16_1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2, bbf16_2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2, bbf16_2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3, bbf16_3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3, bbf16_3_hi);
        }
      }
    }
  }

  static void apply_scale_kgroup(int m, int n, int m_begin, int n_begin, int k_block_begin, float* c, float* reduce_c,
                                 BufferA* ba, BufferB* bb, int k, int k_group_size) {
    using K = GemmKernel224FP8;
    int to = std::min(m - m_begin, K::M_STEP);

    for (int i = 0; i < to; i++) {
      // Get scale for this k_group
      __m512 bs = _mm512_set1_ps(*bb->get_scale(n, n_begin, k, k_block_begin));
      __m512 now = _mm512_load_ps(reduce_c + i * K::N_STEP);
      __m512 result = _mm512_mul_ps(now, bs);
      __m512 existing = _mm512_load_ps(c + i * K::N_STEP);
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps(c + i * K::N_STEP, result);

      now = _mm512_load_ps(reduce_c + i * K::N_STEP + K::TILE_N);
      result = _mm512_mul_ps(now, bs);
      existing = _mm512_load_ps(c + i * K::N_STEP + K::TILE_N);
      result = _mm512_add_ps(result, existing);
      _mm512_store_ps(c + i * K::N_STEP + K::TILE_N, result);
    }
  }

  // ==========================================================================
  // Small-m GEMV: decode, and prefill experts under 16 rows. It walks the task
  // panel by panel. A 32-column panel's K is contiguous when k <= K_BLOCK, and a
  // task's panels follow each other, so a task reads one prefetched sequential
  // range. Each converted weight vector feeds every row of a block. Per output
  // element the arithmetic is that of avx_kernel plus apply_scale_kgroup (which
  // compiles to an FMA): a dpbf16 chain from zero over each k group in K order,
  // then c = fma(sum, scale, c) for each group in K order, starting from c = 0.
  // The results are therefore bit-identical to float_mat_vec_kgroup's.
  // KT_FP8_DECODE_PANELS=0 restores float_mat_vec_kgroup, the staged decode
  // input and whole-range decode units.
  // ==========================================================================

  static bool decode_panels() {
    static const bool enabled = cached_env_flag("KT_FP8_DECODE_PANELS");
    return enabled;
  }

  // How far ahead of the stream position the GEMV prefetches inside a task's weights,
  // in bytes (KT_FP8_DECODE_PREFETCH; 0 turns prefetching off). On DSv3 decode, 8192 and
  // 16384 measured best: about 1% faster than no prefetch, 6% with one CPU expert; 32768
  // is slower.
  static int decode_prefetch_bytes() {
    static const int bytes = [] {
      const char* value = std::getenv("KT_FP8_DECODE_PREFETCH");
      return value ? std::max(0, std::atoi(value)) / 64 * 64 : 8192;
    }();
    return bytes;
  }

  // G consecutive k groups from g0, for rows row..row+R-1 of the M block at m_begin and
  // the panel at n_pos, accumulated into c. With G = 2 a single row has four independent
  // dpbf16 chains instead of two; dpbf16 latency is about four times its issue interval.
  // prefetch is this block's stream position plus the prefetch distance (nullptr: off).
  template <int R, int G>
  static inline void fp8_vec_groups(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, int m_begin,
                                    int row, int n_pos, int g0, __m512 (&c)[4][2], const uint8_t* prefetch,
                                    const uint8_t* prefetch_end) {
    __m512 sum[G][R][2];
#pragma GCC unroll 2
    for (int gi = 0; gi < G; ++gi) {
#pragma GCC unroll 4
      for (int r = 0; r < R; ++r) sum[gi][r][0] = sum[gi][r][1] = _mm512_setzero_ps();
    }
    const int a_stride = ba->row_stride();
    for (int kk = 0; kk < k_group_size; kk += K_STEP) {
      const __m512i* b[G];
      const ggml_bf16_t* a[G];
#pragma GCC unroll 2
      for (int gi = 0; gi < G; ++gi) {
        const int kc = (g0 + gi) * k_group_size + kk;
        b[gi] = reinterpret_cast<const __m512i*>(bb->get_submat(n, k, n_pos, kc));
        a[gi] = ba->get_submat(m, k, m_begin, kc) + static_cast<size_t>(row) * a_stride;
      }
      if (prefetch) {
        // This step reads G KB of the block; prefetch the G KB as far ahead.
        const uint8_t* ahead = prefetch + static_cast<size_t>(kk) * G * N_STEP;
        for (int line = 0; line < 16 * G && ahead + line * 64 < prefetch_end; ++line) {
          _mm_prefetch(reinterpret_cast<const char*>(ahead + line * 64), _MM_HINT_T0);
        }
      }
      for (int k_i = 0; k_i < K_STEP / 2; ++k_i) {
#pragma GCC unroll 2
        for (int gi = 0; gi < G; ++gi) {
          const auto [lo, hi] = fp8x64_to_bf16x64(_mm512_load_si512(b[gi] + k_i));
#pragma GCC unroll 4
          for (int r = 0; r < R; ++r) {
            int32_t pair;
            std::memcpy(&pair, a[gi] + static_cast<size_t>(r) * a_stride + k_i * 2, sizeof(pair));
            const __m512bh activation = (__m512bh)_mm512_set1_epi32(pair);
            sum[gi][r][0] = _mm512_dpbf16_ps(sum[gi][r][0], activation, (__m512bh)lo);
            sum[gi][r][1] = _mm512_dpbf16_ps(sum[gi][r][1], activation, (__m512bh)hi);
          }
        }
      }
    }
#pragma GCC unroll 2
    for (int gi = 0; gi < G; ++gi) {
      const __m512 scale = _mm512_set1_ps(*bb->get_scale(n, n_pos, k, (g0 + gi) * k_group_size));
#pragma GCC unroll 4
      for (int r = 0; r < R; ++r) {
        c[r][0] = _mm512_fmadd_ps(sum[gi][r][0], scale, c[r][0]);
        c[r][1] = _mm512_fmadd_ps(sum[gi][r][1], scale, c[r][1]);
      }
    }
  }

  // Rows row..row+R-1 of one panel over every k group, into c[0..R-1]. A non-zero
  // prefetch_bytes runs the prefetch stream that far ahead of the panel's reads.
  template <int R, int G>
  static inline void fp8_vec_panel(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, int m_begin,
                                   int row, int n_pos, __m512 (&c)[4][2], int prefetch_bytes,
                                   const uint8_t* prefetch_end) {
#pragma GCC unroll 4
    for (int r = 0; r < R; ++r) c[r][0] = c[r][1] = _mm512_setzero_ps();
    const int groups = k / k_group_size;
    const size_t group_bytes = static_cast<size_t>(k_group_size) * N_STEP;
    const uint8_t* stream = prefetch_bytes > 0 ? bb->get_submat(n, k, n_pos, 0) + prefetch_bytes : nullptr;
    int g = 0;
    for (; g + G <= groups; g += G) {
      fp8_vec_groups<R, G>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, g, c,
                           stream ? stream + g * group_bytes : nullptr, prefetch_end);
    }
    for (; g < groups; ++g) {
      fp8_vec_groups<R, 1>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, g, c,
                           stream ? stream + g * group_bytes : nullptr, prefetch_end);
    }
  }

  // Columns [n_start, n_end) of the small-m GEMV. Writes BF16 rows into dst (row stride
  // n) when it is given, else FP32 blocks into bc for to_mat.
  static void fp8_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc,
                                 int n_start, int n_end, ggml_bf16_t* dst) {
    assert(n % N_STEP == 0 && n_start % N_STEP == 0 && n_end % N_STEP == 0);
    assert(k % k_group_size == 0 && k_group_size % K_STEP == 0);
    // Past K_BLOCK a panel's K is split across blocks and the stream is not sequential.
    const int prefetch_bytes = k <= K_BLOCK ? decode_prefetch_bytes() : 0;
    const uint8_t* prefetch_end = bb->b + static_cast<size_t>(n_end) * k;
    __m512 c[4][2];
    for (int n_pos = n_start; n_pos < n_end; n_pos += N_STEP) {
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        const int rows = std::min(M_STEP, m - m_begin);
        float* block = dst ? nullptr : bc->get_submat(m, n, m_begin, n_pos);
        for (int row = 0; row < rows;) {
          // Later row blocks reread the panel from cache.
          const int prefetch = m_begin == 0 && row == 0 ? prefetch_bytes : 0;
          const int r = std::min(4, rows - row);
          switch (r) {
            case 4:
              fp8_vec_panel<4, 1>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, c, prefetch, prefetch_end);
              break;
            case 3:
              fp8_vec_panel<3, 1>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, c, prefetch, prefetch_end);
              break;
            case 2:
              fp8_vec_panel<2, 1>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, c, prefetch, prefetch_end);
              break;
            default:
              fp8_vec_panel<1, 2>(m, n, k, k_group_size, ba, bb, m_begin, row, n_pos, c, prefetch, prefetch_end);
              break;
          }
          for (int i = 0; i < r; ++i) {
            if (dst) {
              avx512_32xfp32_to_32xbf16(
                  &c[i][0], &c[i][1],
                  reinterpret_cast<__m512i*>(dst + static_cast<size_t>(m_begin + row + i) * n + n_pos));
            } else {
              _mm512_store_ps(block + (row + i) * N_STEP, c[i][0]);
              _mm512_store_ps(block + (row + i) * N_STEP + TILE_N, c[i][1]);
            }
          }
          row += r;
        }
      }
    }
  }
};

// all step = 32
template <typename K, bool amx_or_avx = false>
void float_mat_vec_kgroup(int m, int n, int k, int k_group_size, typename K::BufferA* ba, typename K::BufferB* bb,
                          typename K::BufferC* bc, int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % k_group_size == 0);
  assert(k_group_size % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  // Process by k_groups
  for (int k_group_begin = 0; k_group_begin < k; k_group_begin += k_group_size) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);
        float* reduce_c = bc->get_reduce_submat(m, n, m_begin, n_begin);

        if (k_group_begin == 0) {
          for (int i = 0; i < K::M_STEP && m_begin + i < m; i++) {
            for (int j = 0; j < K::N_STEP; j++) {
              c[i * K::N_STEP + j] = 0.0f;
            }
          }
        }

        // avx_kernel_4 now processes entire k_group_size internally (like INT8's avx_kernel)
        if constexpr (amx_or_avx && AMX_AVAILABLE) {
          for (int k_begin = k_group_begin; k_begin < std::min(k, k_group_begin + k_group_size); k_begin += K::K_STEP) {
            K::amx_kernel(m, n, k, m_begin, n_begin, k_begin, reduce_c, ba, bb, k_group_size);
          }
        } else {
          // Single call processes entire k_group
          K::avx_kernel(m, n, k, m_begin, n_begin, k_group_begin, reduce_c, ba, bb, k_group_size);
        }
        K::apply_scale_kgroup(m, n, m_begin, n_begin, k_group_begin, c, reduce_c, ba, bb, k, k_group_size);
      }
    }
  }
}

// ============================================================================
// GemmKernel224BF16 vec_mul/mat_mul
// ============================================================================

// Template function for BF16 mat_mul/vec_mul with AMX or AVX backend
template <typename K, bool amx_or_avx = true>
void float_mat_vec(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb, typename K::BufferC* bc,
                   int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float* c = bc->get_submat(m, n, m_begin, n_begin);

        if constexpr (amx_or_avx && AMX_AVAILABLE) {
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        } else {
          K::avx_kernel_4(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        }
      }
    }
  }
}

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16::BufferB> bb, std::shared_ptr<GemmKernel224BF16::BufferC> bc,
                    int ith, int nth) {
  float_mat_vec<GemmKernel224BF16, true>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void vec_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF16::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF16::BufferB> bb, std::shared_ptr<GemmKernel224BF16::BufferC> bc,
                    int ith, int nth) {
  float_mat_vec<GemmKernel224BF16, false>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

// Returns true when the kernel wrote the task's BF16 rows into dst itself; otherwise the
// result is in bc for to_mat.
inline bool vec_mul_kgroup(int m, int n, int k, int k_group_size, const std::shared_ptr<GemmKernel224FP8::BufferA>& ba,
                           const std::shared_ptr<GemmKernel224FP8::BufferB>& bb,
                           const std::shared_ptr<GemmKernel224FP8::BufferC>& bc, int ith, int nth,
                           ggml_bf16_t* dst = nullptr) {
  if (GemmKernel224FP8::decode_panels()) {
    auto [n_start, n_end] = GemmKernel224FP8::split_range_n(n, ith, nth);
    GemmKernel224FP8::fp8_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), n_start, n_end, dst);
    return dst != nullptr;
  }
  float_mat_vec_kgroup<GemmKernel224FP8, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
  return false;
}

// Returns true when the kernel wrote the task's BF16 rows into dst itself; otherwise the
// result is in bc for to_mat.
inline bool mat_mul_kgroup(int m, int n, int k, int k_group_size, const std::shared_ptr<GemmKernel224FP8::BufferA>& ba,
                           const std::shared_ptr<GemmKernel224FP8::BufferB>& bb,
                           const std::shared_ptr<GemmKernel224FP8::BufferC>& bc, int ith, int nth,
                           ggml_bf16_t* dst = nullptr) {
#ifdef HAVE_AMX
  // Prefill AMX tiles need a full 16-row staging guard plus FP8->BF16
  // widening; KT_FP8_PREFILL_AMX=0 forces the AVX512-BF16 path for A/B checks.
  static const bool amx_prefill = cached_env_flag("KT_FP8_PREFILL_AMX");
  if (amx_prefill && AMX_AVAILABLE && m >= 16 && n % GemmKernel224FP8::N_STEP == 0 &&
      k % GemmKernel224FP8::K_STEP == 0 && k_group_size % GemmKernel224FP8::K_STEP == 0) {
    GemmKernel224FP8::fp8_mat_mat_kgroup_amx(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth, dst);
    return dst != nullptr;
  }
#endif
  return vec_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, ith, nth, dst);
}

// ============================================================================
// Per-Channel FP8 GEMM (for GLM-4.7-FP8 style quantization)
// ============================================================================

/**
 * @brief FP8 Per-Channel Kernel
 *
 * Similar to GemmKernel224FP8 but with per-channel scaling instead of block-wise scaling.
 * - Block-wise: scale shape = [n/128, k/128], one scale per 128x128 block
 * - Per-channel: scale shape = [n], one scale per output row
 */
struct GemmKernel224FP8PerChannel {
  using fp8_t = uint8_t;
  using output_t = float;

  static constexpr double ELEMENT_SIZE = 1.0;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  // Use smaller N_BLOCK for per-channel to allow efficient scale application
  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "FP8PerChannel"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {}

  using BufferA = BufferABF16Impl<GemmKernel224FP8PerChannel>;
  using BufferB = BufferBFP8PerChannelImpl<GemmKernel224FP8PerChannel>;
  using BufferC = BufferCFP32Impl<GemmKernel224FP8PerChannel>;

  // Reuse FP8->BF16 conversion from GemmKernel224FP8
  static inline std::pair<__m512i, __m512i> fp8x64_to_bf16x64(__m512i bfp8_512) {
    return GemmKernel224FP8::fp8x64_to_bf16x64(bfp8_512);
  }

  /**
   * @brief Apply per-channel scale to result
   *
   * Unlike block-wise scaling, per-channel scaling applies a different scale to each column
   * of the result (each output channel).
   *
   * @param m Total rows
   * @param n Total columns
   * @param m_begin Starting row
   * @param n_begin Starting column
   * @param c Output buffer (M_STEP x N_STEP)
   * @param bb BufferB containing per-channel scales
   */
  static void apply_scale_perchannel(int m, [[maybe_unused]] int n, int m_begin, int n_begin, float* c, BufferB* bb) {
    int to = std::min(m - m_begin, M_STEP);

    // Load N_STEP per-channel scales (32 floats)
    __m512 bs_lo = _mm512_loadu_ps(bb->get_scale(n_begin));           // scale[n_begin..n_begin+15]
    __m512 bs_hi = _mm512_loadu_ps(bb->get_scale(n_begin + TILE_N));  // scale[n_begin+16..n_begin+31]

    for (int i = 0; i < to; i++) {
      // Each row gets multiplied by the same set of per-channel scales
      __m512 c_lo = _mm512_load_ps(c + i * N_STEP);
      __m512 c_hi = _mm512_load_ps(c + i * N_STEP + TILE_N);
      _mm512_store_ps(c + i * N_STEP, _mm512_mul_ps(c_lo, bs_lo));
      _mm512_store_ps(c + i * N_STEP + TILE_N, _mm512_mul_ps(c_hi, bs_hi));
    }
  }

  // ==========================================================================
  // FP8 -> BF16 AMX prefill staging, per-channel scales.
  //
  // Same widening and packed-B tile layout as GemmKernel224FP8, but a single
  // scale covers the whole K extent, so the accumulators are never drained
  // mid-K: the tile registers hold the full dot product and the scale is
  // folded once at the end by apply_scale_perchannel, exactly as the AVX path
  // does. That makes this kernel strictly simpler than the block-wise one --
  // there is no reduce buffer and no k_group loop.
  // ==========================================================================
  static void fp8_perchannel_mat_mat_kgroup_amx(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                                int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    GemmKernel224FP8::configure_fp8_amx();

    // Per-worker scratch is bounded by 32 BF16 output channels times K.
    thread_local std::vector<uint8_t> packed_storage;
    packed_storage.resize(static_cast<size_t>(k) * 64 + 63);
    auto* packed = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(packed_storage.data()) + 63) &
                                               ~uintptr_t(63));
    for (int n_pos = n_start; n_pos < n_end; n_pos += N_STEP) {
      GemmKernel224FP8::pack_fp8_amx_b(n, k, n_pos, bb, packed);
      for (int m_pos = 0; m_pos < m; m_pos += M_STEP) {
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
        for (int chunk_begin = 0; chunk_begin < k; chunk_begin += K_STEP) {
          // Tail activation rows stay zero through BufferABF16Impl padding.
          const ggml_bf16_t* a = ba->get_submat(m, k, m_pos, chunk_begin);
          const uint32_t* b = packed + static_cast<size_t>(chunk_begin / K_STEP) * 16 * 32;
          _tile_loadd(0, a, K_STEP * sizeof(ggml_bf16_t));
          _tile_loadd(1, a + 16 * K_STEP, K_STEP * sizeof(ggml_bf16_t));
          _tile_loadd(2, b, 128);
          _tile_loadd(3, b + 16, 128);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
        }
        float* result = bc->get_submat(m, n, m_pos, n_pos);
        _tile_stored(4, result, N_STEP * sizeof(float));
        _tile_stored(5, result + TILE_N, N_STEP * sizeof(float));
        _tile_stored(6, result + 16 * N_STEP, N_STEP * sizeof(float));
        _tile_stored(7, result + 16 * N_STEP + TILE_N, N_STEP * sizeof(float));
        apply_scale_perchannel(m, n, m_pos, n_pos, result, bb);
      }
    }
  }

  // AVX kernel for per-channel FP8 GEMM - processes entire K dimension
  static void avx_kernel_4(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float* c, BufferA* ba,
                           BufferB* bb) {
    const __m512i bf16_hi_0 = GemmKernel224FP8::bf16_hi_0_mask();
    const __m512i bf16_hi_1 = GemmKernel224FP8::bf16_hi_1_mask();
    const __m512i bf16_lo_0 = GemmKernel224FP8::bf16_lo_0_mask();
    const __m512i bf16_lo_1 = GemmKernel224FP8::bf16_lo_1_mask();
    const __m512i sign_mask_v = GemmKernel224FP8::sign_mask();

    __m512* c512 = (__m512*)c;
    int m_block_end = std::min(m - m_begin, M_STEP);

    // Zero out accumulator at start of K_BLOCK
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m_block_end; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    // Process K_BLOCK
    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      ggml_bf16_t* abf16 = (ggml_bf16_t*)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);

      // Process 4 k_i at once
      for (int k_i = 0; k_i < 16; k_i += 4) {
        // Load 4 B vectors
        __m512i bfp8_0 = bfp8_512[k_i];
        __m512i bfp8_1 = bfp8_512[k_i + 1];
        __m512i bfp8_2 = bfp8_512[k_i + 2];
        __m512i bfp8_3 = bfp8_512[k_i + 3];

        // Convert all 4 FP8 -> BF16
        __m512i b_hi, b_lo;

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_0),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_0, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_0, bf16_lo_1);
        __m512bh bbf16_0_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_0_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_1),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_1, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_1, bf16_lo_1);
        __m512bh bbf16_1_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_1_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_2),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_2, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_2, bf16_lo_1);
        __m512bh bbf16_2_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_2_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        b_hi = _mm512_or_si512(_mm512_and_si512(sign_mask_v, bfp8_3),
                               _mm512_permutex2var_epi8(bf16_hi_0, bfp8_3, bf16_hi_1));
        b_lo = _mm512_permutex2var_epi8(bf16_lo_0, bfp8_3, bf16_lo_1);
        __m512bh bbf16_3_lo = (__m512bh)_mm512_unpacklo_epi8(b_lo, b_hi);
        __m512bh bbf16_3_hi = (__m512bh)_mm512_unpackhi_epi8(b_lo, b_hi);

        // Process m rows
        int m_i = 0;
        for (; m_i + 1 < m_block_end; m_i += 2) {
          __m512bh ma0_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]);
          __m512bh ma1_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]);
          __m512bh ma2_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]);
          __m512bh ma3_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]);
          __m512bh ma0_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + k_i * 2]);
          __m512bh ma1_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 1) * 2]);
          __m512bh ma2_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 2) * 2]);
          __m512bh ma3_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 3) * 2]);

          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0_0, bbf16_0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0_0, bbf16_0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1_0, bbf16_1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1_0, bbf16_1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2_0, bbf16_2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2_0, bbf16_2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3_0, bbf16_3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3_0, bbf16_3_hi);

          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma0_1, bbf16_0_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma0_1, bbf16_0_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma1_1, bbf16_1_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma1_1, bbf16_1_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma2_1, bbf16_2_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma2_1, bbf16_2_hi);
          c512[(m_i + 1) * 2] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2], ma3_1, bbf16_3_lo);
          c512[(m_i + 1) * 2 + 1] = _mm512_dpbf16_ps(c512[(m_i + 1) * 2 + 1], ma3_1, bbf16_3_hi);
        }
        // Handle remaining row
        for (; m_i < m_block_end; m_i++) {
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]);
          __m512bh ma2 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]);
          __m512bh ma3 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]);

          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma0, bbf16_0_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma0, bbf16_0_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma1, bbf16_1_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma1, bbf16_1_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma2, bbf16_2_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma2, bbf16_2_hi);
          c512[m_i * 2] = _mm512_dpbf16_ps(c512[m_i * 2], ma3, bbf16_3_lo);
          c512[m_i * 2 + 1] = _mm512_dpbf16_ps(c512[m_i * 2 + 1], ma3, bbf16_3_hi);
        }
      }
    }
  }
};

/**
 * @brief Per-channel FP8 GEMM function
 *
 * Unlike block-wise FP8 which applies scale per 128x128 block during computation,
 * per-channel FP8 processes entire K dimension first, then applies per-channel scale at the end.
 */
template <typename K>
void float_mat_vec_perchannel(int m, int n, int k, typename K::BufferA* ba, typename K::BufferB* bb,
                              typename K::BufferC* bc, int ith, int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
    for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
      float* c = bc->get_submat(m, n, m_begin, n_begin);

      // Process entire K dimension with K_BLOCKs
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
        K::avx_kernel_4(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
      }

      // Apply per-channel scale once after all K is processed
      K::apply_scale_perchannel(m, n, m_begin, n_begin, c, bb);
    }
  }
}

inline void vec_mul_perchannel(int m, int n, int k, std::shared_ptr<GemmKernel224FP8PerChannel::BufferA> ba,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferB> bb,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferC> bc, int ith, int nth) {
  float_mat_vec_perchannel<GemmKernel224FP8PerChannel>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_perchannel(int m, int n, int k, std::shared_ptr<GemmKernel224FP8PerChannel::BufferA> ba,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferB> bb,
                               std::shared_ptr<GemmKernel224FP8PerChannel::BufferC> bc, int ith, int nth) {
#ifdef HAVE_AMX
  // Same staging contract as the block-wise FP8 kernel: AMX tiles need a full
  // 16-row guard and the FP8->BF16 widening. KT_FP8_PERCHANNEL_PREFILL_AMX=0
  // forces the AVX512-BF16 path for A/B checks.
  static const bool amx_prefill = cached_env_flag("KT_FP8_PERCHANNEL_PREFILL_AMX");
  if (amx_prefill && AMX_AVAILABLE && m >= 16 && n % GemmKernel224FP8PerChannel::N_STEP == 0 &&
      k % GemmKernel224FP8PerChannel::K_STEP == 0) {
    GemmKernel224FP8PerChannel::fp8_perchannel_mat_mat_kgroup_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
  float_mat_vec_perchannel<GemmKernel224FP8PerChannel>(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

#endif  // AMX_RAW_KERNELS_HPP
