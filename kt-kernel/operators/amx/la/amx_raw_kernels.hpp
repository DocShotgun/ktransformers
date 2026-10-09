#ifndef AMX_RAW_KERNELS_HPP
#define AMX_RAW_KERNELS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  using BufferA = BufferABF16Impl<GemmKernel224FP8>;
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

  // Scalar twin of fp8x64_to_bf16x64's lookup decode (bit-exact widening).
  static inline uint16_t fp8_to_bf16_bits(uint8_t fp8) {
    const uint8_t index = fp8 & 0x3F;
    const bool hi_select = (fp8 & 0x40) != 0;
    uint8_t hi = hi_select ? bf16_hi_1_val[index] : bf16_hi_0_val[index];
    const uint8_t lo = hi_select ? bf16_lo_1_val[index] : bf16_lo_0_val[index];
    hi |= fp8 & 0x80;
    return static_cast<uint16_t>((static_cast<uint16_t>(hi) << 8) | lo);
  }

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
  static void pack_fp8_amx_b(int n, int k, int n_pos, BufferB* bb, uint32_t* packed) {
    static constexpr int mat_offset[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const int chunk_count = k / K_STEP;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
      uint32_t* destination = packed + static_cast<size_t>(chunk) * 16 * 32;
      const uint64_t* words = reinterpret_cast<const uint64_t*>(bb->get_submat(n, k, n_pos, chunk * K_STEP));
      for (int pair = 0; pair < 16; ++pair) {
        const uint64_t pair_words[8] = {words[8 * pair], words[8 * pair + 1], words[8 * pair + 2], words[8 * pair + 3],
                                        words[8 * pair + 4], words[8 * pair + 5], words[8 * pair + 6],
                                        words[8 * pair + 7]};
        for (int channel = 0; channel < 32; ++channel) {
          const uint16_t fp8_pair =
              static_cast<uint16_t>(pair_words[mat_offset[channel >> 2]] >> (16 * (channel & 3)));
          destination[pair * 32 + channel] =
              static_cast<uint32_t>(fp8_to_bf16_bits(static_cast<uint8_t>(fp8_pair))) |
              (static_cast<uint32_t>(fp8_to_bf16_bits(static_cast<uint8_t>(fp8_pair >> 8))) << 16);
        }
      }
    }
  }

  static void fp8_mat_mat_kgroup_amx(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc,
                                     int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    configure_fp8_amx();

    // Per-worker scratch is bounded by 32 BF16 output channels times K. No
    // expanded copy of the full model is retained in host RAM.
    thread_local std::vector<uint8_t> packed_storage;
    packed_storage.resize(static_cast<size_t>(k) * 64 + 63);
    auto* packed = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(packed_storage.data()) + 63) &
                                               ~uintptr_t(63));
    for (int n_pos = n_start; n_pos < n_end; n_pos += N_STEP) {
      pack_fp8_amx_b(n, k, n_pos, bb, packed);
      for (int m_pos = 0; m_pos < m; m_pos += M_STEP) {
        float* result = bc->get_submat(m, n, m_pos, n_pos);
        float* reduce_c = bc->get_reduce_submat(m, n, m_pos, n_pos);
        for (int row = 0; row < M_STEP; ++row) {
          for (int column = 0; column < N_STEP; ++column) result[row * N_STEP + column] = 0.0f;
        }
        for (int group_begin = 0; group_begin < k; group_begin += k_group_size) {
          _tile_zero(4);
          _tile_zero(5);
          _tile_zero(6);
          _tile_zero(7);
          for (int chunk_begin = group_begin; chunk_begin < std::min(k, group_begin + k_group_size);
               chunk_begin += K_STEP) {
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
          _tile_stored(4, reduce_c, N_STEP * sizeof(float));
          _tile_stored(5, reduce_c + TILE_N, N_STEP * sizeof(float));
          _tile_stored(6, reduce_c + 16 * N_STEP, N_STEP * sizeof(float));
          _tile_stored(7, reduce_c + 16 * N_STEP + TILE_N, N_STEP * sizeof(float));
          apply_scale_kgroup(m, n, m_pos, n_pos, group_begin, result, reduce_c, ba, bb, k, k_group_size);
        }
      }
    }
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
      __m512i* bfp8_512 = (__m512i*)bb->get_submat(n, k, n_begin, k_group_begin + k_begin);

      for (int m_i = 0; m_i < m_block_end; m_i++) {
        // Process 2 k_i per iteration
        for (int k_i = 0; k_i < 16; k_i += 2) {
          // Load A vectors
          __m512bh ma0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]);
          __m512bh ma1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]);

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
          __m512bh ma0_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + k_i * 2]);
          __m512bh ma1_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 1) * 2]);
          __m512bh ma2_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 2) * 2]);
          __m512bh ma3_0 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[m_i * K_STEP + (k_i + 3) * 2]);
          __m512bh ma0_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + k_i * 2]);
          __m512bh ma1_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 1) * 2]);
          __m512bh ma2_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 2) * 2]);
          __m512bh ma3_1 = (__m512bh)_mm512_set1_epi32(*(int32_t*)&abf16[(m_i + 1) * K_STEP + (k_i + 3) * 2]);

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

inline void vec_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                           std::shared_ptr<GemmKernel224FP8::BufferB> bb, std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                           int ith, int nth) {
  float_mat_vec_kgroup<GemmKernel224FP8, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size, std::shared_ptr<GemmKernel224FP8::BufferA> ba,
                           std::shared_ptr<GemmKernel224FP8::BufferB> bb, std::shared_ptr<GemmKernel224FP8::BufferC> bc,
                           int ith, int nth) {
#ifdef HAVE_AMX
  // Prefill AMX tiles need a full 16-row staging guard plus FP8->BF16
  // widening; KT_FP8_PREFILL_AMX=0 forces the AVX512-BF16 path for A/B checks.
  const char* amx_prefill = std::getenv("KT_FP8_PREFILL_AMX");
  if ((!amx_prefill || amx_prefill[0] != '0') && AMX_AVAILABLE && m >= 16 && n % GemmKernel224FP8::N_STEP == 0 &&
      k % GemmKernel224FP8::K_STEP == 0 && k_group_size % GemmKernel224FP8::K_STEP == 0) {
    GemmKernel224FP8::fp8_mat_mat_kgroup_amx(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
  float_mat_vec_kgroup<GemmKernel224FP8, false>(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
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

}  // namespace amx

#endif  // AMX_RAW_KERNELS_HPP
