/**
 * @Description  : MXFP4 MoE operator — FP4 E2M1 weights × BF16 activations
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-04-20
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Based on k2-moe.hpp (RAWINT4). Key differences from RAWINT4:
 *   Weight:   FP4 E2M1 (nibble-packed, same layout) → PSHUFB lookup → BF16
 *   Act:      BF16 direct (BufferABF16Impl, no online INT8 quantization)
 *   Dot prod: _mm512_dpbf16_ps (BF16×BF16→FP32) instead of _mm512_dpbssd_epi32
 *   Scale:    per-group weight scale; E8M0 is stored as one exponent byte,
 *             with an exact FP32 fallback for other scale layouts
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl
#include "moe_base.hpp"

namespace amx {

// ============================================================================
// MXFP4 kernel: FP4 E2M1 weights × BF16 activations → FP32 output (AVX512)
// ============================================================================
struct GemmKernel224MXFP4SmallKGroup {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "MXFP4_KGROUP"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  // FP4 E2M1 → BF16 LUTs (16 entries each, for PSHUFB within 128-bit lanes)
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,   //  0..7  positive
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};  //  8..15 negative
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
      0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,   //  0..7  positive
      0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};  //  8..15 negative

#if defined(__AVX512BF16__)
  // Natural lane-local order emitted by the grouped decoder below. Keeping
  // decoded values in this order removes per-output-row 16-bit interleaves;
  // the much smaller activation is permuted once before the N-row loop.
  alignas(64) static constexpr uint16_t natural_group_indices[32] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20,
                                                                     22, 24, 26, 28, 30, 1,  3,  5,  7,  9,  11,
                                                                     13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
  alignas(32) static constexpr uint16_t fp4_bf16[16] = {0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                                        0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0};
#endif

  // Convert 16 packed FP4 bytes (32 values = 1 k_group) → 32 BF16 values (__m512i)
  // Output column order: [BF16(lo[0]),BF16(hi[0]), ..., BF16(lo[15]),BF16(hi[15])]
  __attribute__((always_inline)) static inline __m512i mxfp4_to_bf16_32(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    __m128i lut_lo = _mm_load_si128((__m128i*)fp4_bf16_lo);
    __m128i lut_hi = _mm_load_si128((__m128i*)fp4_bf16_hi);

    // Look up low/high bytes for lo nibbles → 16 BF16 values
    __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
    __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
    __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);  // BF16(lo[0..7])
    __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);  // BF16(lo[8..15])

    // Look up low/high bytes for hi nibbles → 16 BF16 values
    __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
    __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
    __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);  // BF16(hi[0..7])
    __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);  // BF16(hi[8..15])

    // Interleave lo/hi at 16-bit: [lo[0],hi[0], lo[1],hi[1], ...] = column order
    __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);  // cols  0..7
    __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);  // cols  8..15
    __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);  // cols 16..23
    __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);  // cols 24..31

    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
  }

#if defined(__AVX512BF16__)
  // Decode one complete 32-value group with a word-indexed LUT. Expanding the
  // low/high nibbles directly to word indices avoids the byte lookups and
  // byte-to-word unpack chain used by the PSHUFB decoder. Two independent
  // widen operations avoid the dependency chain of widening packed bytes once.
  __attribute__((always_inline)) static inline __m512i mxfp4_to_bf16_32_natural(__m128i packed) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(packed, lo_mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);
    const __m256i lo_words = _mm256_cvtepu8_epi16(lo);
    const __m256i hi_words = _mm256_cvtepu8_epi16(hi);
    const __m512i indices = _mm512_inserti64x4(_mm512_castsi256_si512(lo_words), hi_words, 1);
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(fp4_bf16)));
    return _mm512_permutexvar_epi16(indices, lut);
  }

  __attribute__((always_inline)) static inline __m512bh permute_activation_group(__m512bh activation) {
    const __m512i indices = _mm512_load_si512(static_cast<const void*>(natural_group_indices));
    return (__m512bh)_mm512_permutexvar_epi16(indices, (__m512i)activation);
  }
#endif

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
    inline static const __m512i odd_mask = _mm512_set1_epi32(0xFFFF0000);
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, odd_mask));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d;
#else
    __m512 w_even;
    __m512 w_odd;
    inline static const __m128i lo_mask = _mm_set1_epi8(0x0F);
    inline static const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f,
                                                    -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
#endif

    __attribute__((always_inline)) DequantizedWeight(__m128i w) {
#if defined(__AVX512BF16__)
      d = (__m512bh)mxfp4_to_bf16_32(w);
#else
      __m128i lo = _mm_and_si128(w, lo_mask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), lo_mask);

      __m512i lo_32 = _mm512_cvtepu8_epi32(lo);
      __m512i hi_32 = _mm512_cvtepu8_epi32(hi);

      w_even = _mm512_permutexvar_ps(lo_32, lut);
      w_odd = _mm512_permutexvar_ps(hi_32, lut);
#endif
    }
  };

  __attribute__((always_inline)) static inline __m512 mxfp4_dot_bf16(const DequantizedWeight& w,
                                                                     const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(_mm512_setzero_ps(), act.a, w.d);
#else
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
#endif
  }

  // BufferA records whether decode activation has already been converted to
  // the natural grouped order. Prefill and all fallback paths keep the
  // existing logical layout.
  struct BufferA : public BufferABF16Impl<GemmKernel224MXFP4SmallKGroup> {
    using Base = BufferABF16Impl<GemmKernel224MXFP4SmallKGroup>;
    using Base::a;
    using Base::k;
    using Base::max_m;

    // Prefill aliases the expert-grouped BF16 rows below. Only one row per
    // active expert needs staging for decode or a one-token prefill route.
    static size_t required_size(int max_m, int k) {
      return Base::required_size(k <= K_BLOCK ? 1 : max_m, k);
    }

    bool natural_order = false;

    BufferA(int max_m_, int k_, void* ptr) : Base(max_m_, k_, ptr) {}

    void set_data(void* ptr) {
      Base::set_data(ptr);
      natural_order = false;
    }

    void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
      // With M_STEP=1 and a single K block, BufferA's layout is the same
      // row-major BF16 layout already produced by the expert gather. Avoid
      // copying every prefill activation into the staging arena twice (gate
      // and down). Decode still uses its separately allocated, permuted tile.
      if (m > 1 && k <= K_BLOCK) {
        assert(m <= max_m && ith == 0 && nth == 1);
        a = src;
      } else {
        Base::from_mat(m, src, ith, nth);
      }
      natural_order = false;
    }

    void from_mat_natural(int m, ggml_bf16_t* src, int ith, int nth) {
#if defined(__AVX512BF16__)
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);
      assert(k % 32 == 0);
      for (int mi = 0; mi < m; ++mi) {
        const __m512bh* src_row = reinterpret_cast<const __m512bh*>(src + static_cast<size_t>(mi) * k);
        __m512bh* dst_row = reinterpret_cast<__m512bh*>(get_submat(m, k, mi, 0));
        for (int g = 0; g < k / 32; ++g) dst_row[g] = permute_activation_group(src_row[g]);
      }
      natural_order = true;
#else
      from_mat(m, src, ith, nth);
#endif
    }
  };

  // Native MXFP4 scales are positive powers of two. After validating the full
  // tensor, compact their FP32 exponent bytes in-place; reconstructing either
  // FP32 or BF16 is then bit-exact. Non-E8M0 input is left untouched and uses
  // the original FP32 fallback path.
  struct BufferB : public BufferBInt4KGroupImpl<GemmKernel224MXFP4SmallKGroup> {
    using Base = BufferBInt4KGroupImpl<GemmKernel224MXFP4SmallKGroup>;
    using Base::b;
    using Base::d;
    using Base::k;
    using Base::k_group_count;
    using Base::k_group_size;
    using Base::n;

    uint8_t* scale_e8;
    uint8_t* scale_e8_kmajor;
    bool scale_e8_valid = false;
    bool scale_e8_vector_safe = false;
    bool kmajor_weights = false;
    // NVFP4 (k_group_size 16) scales arrive folded to BF16 by the loader.
    // After validation, compact the FP32 slots to BF16 in-place and keep a
    // transposed per-(group, 32-channel) BF16 copy in the freed tail; the
    // compact copy (n*k/16 bf16) plus the kmajor copy (n*k/16 bf16) exactly
    // fill the FP32 slot area (n*k/16 fp32), so host RAM is unchanged.
    ggml_bf16_t* scale_nv_compact = nullptr;
    ggml_bf16_t* scale_nv_kmajor = nullptr;
    bool scale_nv_valid = false;

    static size_t required_size(int n, int k, int k_group_size) { return Base::required_size(n, k, k_group_size); }

    BufferB(int n_, int k_, int k_group_size_, void* ptr) : Base(n_, k_, k_group_size_, ptr) {
      // The compact bytes replace the FP32 contents in-place after validation.
      // Forward compression is overlap-safe: byte i is always written below
      // the first byte of every not-yet-read float j > i.
      scale_e8 = reinterpret_cast<uint8_t*>(d);
      scale_e8_kmajor = scale_e8 + static_cast<size_t>(n) * k_group_count;
      scale_nv_compact = reinterpret_cast<ggml_bf16_t*>(d);
      scale_nv_kmajor = scale_nv_compact + static_cast<size_t>(n) * k_group_count;
      // NVFP4 keeps its own K-major knob (KT_NVFP4_KMAJOR_WEIGHTS) mirroring
      // the MXFP4 one; both default on for AMX/BF16 hosts.
      const char* layout =
          std::getenv(k_group_size_ == 16 ? "KT_NVFP4_KMAJOR_WEIGHTS" : "KT_MXFP4_KMAJOR_WEIGHTS");
#if defined(HAVE_AMX) && defined(__AVX512BF16__)
      // The packed layout is the fast path on AMX/BF16 hosts. Keep an opt-out
      // for comparisons and for machines with unusual routing distributions.
      kmajor_weights = !layout || layout[0] != '0';
#else
      kmajor_weights = false;
#endif
    }

    // SGLang's 32-channel, K-pair-major MXFP4 layout fits in the same
    // allocation as the original row-major weight. Each pair interleaves
    // channels c and c+16, allowing one contiguous AVX512 load per K pair.
    void from_raw_mat(uint8_t* proj, int ith, int nth) {
      if (!kmajor_weights) {
        Base::from_raw_mat(proj, ith, nth);
        return;
      }
      auto [start, end] = GemmKernel224MXFP4SmallKGroup::split_range_n(n, ith, nth);
      const size_t row_bytes = static_cast<size_t>(k) / 2;
      for (int row = start; row < end; row += 32) {
        uint8_t* destination = b + static_cast<size_t>(row) * row_bytes;
        for (size_t pair = 0; pair < row_bytes; ++pair) {
          uint8_t* dst_pair = destination + pair * 32;
          for (int channel = 0; channel < 16; ++channel) {
            const uint8_t first = proj[static_cast<size_t>(row + channel) * row_bytes + pair];
            const uint8_t second = proj[static_cast<size_t>(row + channel + 16) * row_bytes + pair];
            dst_pair[2 * channel] = static_cast<uint8_t>(((second & 15) << 4) | (first & 15));
            dst_pair[2 * channel + 1] = static_cast<uint8_t>((second & 0xF0) | (first >> 4));
          }
        }
      }
    }

    uint8_t weight_byte(int row, size_t pair) const {
      const size_t row_bytes = static_cast<size_t>(k) / 2;
      if (!kmajor_weights) return b[static_cast<size_t>(row) * row_bytes + pair];
      const uint8_t* src = b + static_cast<size_t>(row & ~31) * row_bytes + pair * 32 + 2 * (row & 15);
      return row & 16 ? static_cast<uint8_t>((src[0] >> 4) | (src[1] & 0xF0))
                      : static_cast<uint8_t>((src[0] & 15) | ((src[1] & 15) << 4));
    }

    uint8_t* get_submat(int n_, int k_, int n_begin, int k_begin) {
      if (!kmajor_weights) return Base::get_submat(n_, k_, n_begin, k_begin);
      // Generic fallback kernels ask for up to four rows at once. Keep four independent rows
      // per worker so the pointers remain valid throughout each GEMV tile.
      thread_local std::array<std::vector<uint8_t>, 4> rows;
      thread_local unsigned slot = 0;
      auto& dst = rows[slot++ & 3u];
      const size_t row_bytes = static_cast<size_t>(k) / 2;
      dst.resize(row_bytes);
      for (size_t pair = 0; pair < row_bytes; ++pair) dst[pair] = weight_byte(n_begin, pair);
      return dst.data() + k_begin / 2;
    }

    void copy_weight_bytes(uint8_t* destination, size_t offset, size_t count) const {
      if (!kmajor_weights) {
        std::memcpy(destination, b + offset, count);
        return;
      }
      const size_t row_bytes = static_cast<size_t>(k) / 2;
      size_t position = offset;
      const size_t end = offset + count;
      while (position < end) {
        const int row = static_cast<int>(position / row_bytes);
        const size_t pair = position % row_bytes;
        const size_t segment = std::min(end - position, row_bytes - pair);
        for (size_t i = 0; i < segment; ++i) destination[position - offset + i] = weight_byte(row, pair + i);
        position += segment;
      }
    }

    void finalize_scale_e8() {
      const size_t count = static_cast<size_t>(n) * k_group_count;
      bool valid = true;
      bool vector_safe = true;
      for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, d + i, sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        const bool is_positive_power_of_two =
            (bits & 0x80000000u) == 0 && (bits & 0x007FFFFFu) == 0 && exponent != 0 && exponent != 0xFFu;
        valid = valid && is_positive_power_of_two;
        // The AVX512 BF16 exponent-add conversion is exact only while every
        // nonzero FP4 value remains a normal finite BF16 value.
        vector_safe = vector_safe && exponent >= 2 && exponent <= 252;
      }
      scale_e8_valid = valid;
      scale_e8_vector_safe = valid && vector_safe;
      if (valid) {
        for (size_t i = 0; i < count; ++i) {
          uint32_t bits;
          std::memcpy(&bits, d + i, sizeof(bits));
          scale_e8[i] = static_cast<uint8_t>((bits >> 23) & 0xFFu);
        }
        if (kmajor_weights) {
          // The FP32 allocation is four bytes per group. Keep the original
          // row-major exponent bytes for export/fallback and place the
          // transposed copy in otherwise unused bytes of that allocation.
          for (int row = 0; row < n; row += 32) {
            for (int group = 0; group < k_group_count; ++group) {
              uint8_t* dst = scale_e8_kmajor + static_cast<size_t>(row) * k_group_count + group * 32;
              for (int channel = 0; channel < 32; ++channel)
                dst[channel] = scale_e8[static_cast<size_t>(row + channel) * k_group_count + group];
            }
          }
        }
      }
    }

    // NVFP4 per-16 scales arrive folded to BF16 by the loader (exact for the
    // e4m3 x fp32 products it generates). Compact the FP32 slots to BF16
    // in-place -- forward compression stays overlap-safe, byte 2i is always
    // written below the first byte of every not-yet-read float j > i -- and
    // place a transposed per-(group, 32-channel) BF16 copy in the freed tail.
    // Non-BF16-representable scales (e.g. a future fp34 loader) keep the FP32
    // fallback untouched.
    void finalize_scale_nv() {
      if (k_group_size != 16 || scale_e8_valid) {
        scale_nv_valid = false;
        return;
      }
      const size_t count = static_cast<size_t>(n) * k_group_count;
      bool valid = true;
      for (size_t i = 0; i < count; ++i) {
        valid = valid && GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(d[i])) == d[i];
      }
      scale_nv_valid = valid;
      if (!valid) return;
      for (size_t i = 0; i < count; ++i) scale_nv_compact[i] = GGML_FP32_TO_BF16(d[i]);
      if (kmajor_weights) {
        for (int row = 0; row < n; row += 32) {
          for (int group = 0; group < k_group_count; ++group) {
            ggml_bf16_t* dst = scale_nv_kmajor + static_cast<size_t>(row) * k_group_count + group * 32;
            for (int channel = 0; channel < 32; ++channel)
              dst[channel] = scale_nv_compact[static_cast<size_t>(row + channel) * k_group_count + group];
          }
        }
      }
    }

    const uint8_t* get_scale_e8(int n_, int n_begin, int k_, int k_begin) const {
      (void)n_;
      (void)k_;
      const int k_group_idx = k_begin / k_group_size;
      return scale_e8 + static_cast<size_t>(n_begin) * k_group_count + k_group_idx;
    }

    float* get_scale(int n_, int n_begin, int k_, int k_begin) {
      if (scale_nv_valid) {
        // Compact BF16 NVFP4 scales expand back to FP32 on demand. Group-16
        // doubles the group count, so this scratch is sized for k/16.
        constexpr int max_group_count = K_BLOCK / 16;
        const int group_count = k_ / k_group_size;
        assert(group_count <= max_group_count);
        alignas(64) thread_local float scratch_nv[4][max_group_count];
        thread_local unsigned slot_nv = 0;
        float* destination = scratch_nv[slot_nv++ & 3u];
        const ggml_bf16_t* source =
            scale_nv_compact + static_cast<size_t>(n_begin) * k_group_count + k_begin / k_group_size;
        int group = 0;
#if defined(__AVX512BF16__)
        for (; group + 16 <= group_count; group += 16) {
          const __m256bh packed = (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + group));
          _mm512_store_ps(destination + group, _mm512_cvtpbh_ps(packed));
        }
#endif
        for (; group < group_count; ++group) destination[group] = GGML_BF16_TO_FP32(source[group]);
        return destination;
      }
      if (!scale_e8_valid) return Base::get_scale(n_, n_begin, k_, k_begin);
      constexpr int max_group_count = K_BLOCK / 32;
      const int group_count = k_ / k_group_size;
      assert(group_count <= max_group_count);
      alignas(64) thread_local float scratch[4][max_group_count];
      thread_local unsigned slot = 0;
      float* destination = scratch[slot++ & 3u];
      const uint8_t* source = get_scale_e8(n_, n_begin, k_, k_begin);
      int group = 0;
      for (; group + 16 <= group_count; group += 16) {
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + group));
        const __m512i exponents = _mm512_slli_epi32(_mm512_cvtepu8_epi32(packed), 23);
        _mm512_store_ps(destination + group, _mm512_castsi512_ps(exponents));
      }
      for (; group < group_count; ++group) {
        const uint32_t bits = static_cast<uint32_t>(source[group]) << 23;
        std::memcpy(destination + group, &bits, sizeof(bits));
      }
      return destination;
    }

    void copy_scale_to_bf16(ggml_bf16_t* destination, size_t offset, size_t count) const {
      if (scale_nv_valid) {
        std::memcpy(destination, scale_nv_compact + offset, count * sizeof(ggml_bf16_t));
      } else if (scale_e8_valid) {
        for (size_t i = 0; i < count; ++i) {
          const uint16_t bits = static_cast<uint16_t>(scale_e8[offset + i]) << 7;
          std::memcpy(destination + i, &bits, sizeof(bits));
        }
      } else {
        for (size_t i = 0; i < count; ++i) destination[i] = GGML_FP32_TO_BF16(d[offset + i]);
      }
    }
  };

  using BufferC = BufferCReduceImpl<GemmKernel224MXFP4SmallKGroup>;  // FP32 reduce

  // 4 个 zmm 的 horizontal reduce → 4 个连续 fp32。
  // 4 次 reduce_add_ps 之间无依赖，编译器/CPU 可并行调度。
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // FP4 scale application for weight dots. MXFP4 (group 32) carries one scale
  // per 32-value block; NVFP4 (group 16) carries one per 16, matching the
  // dpbf16 lane halves exactly (lanes 0..7 cover K 0..15, lanes 8..15 cover
  // K 16..31 in both the BF16 and VNNI fallback decoders). A lane-wise scale
  // vector keeps the single-FMA accumulation shape for both encodings.
  __attribute__((always_inline)) static inline __m512 fp4_scale_vec(const float* s, int scales_per_block) {
    return scales_per_block == 1
               ? _mm512_set1_ps(s[0])
               : _mm512_mask_blend_ps(0xFF00u, _mm512_set1_ps(s[0]), _mm512_set1_ps(s[1]));
  }

  // mat-vec: M 个独立 token，N 维 4 行一组累加，摊销 horizontal reduce。
  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    const int spb = 32 / k_group_size;  // scales per 32-K block (1 MXFP4, 2 NVFP4)

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      // 主循环: N 维 4 行一组
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          acc0 = _mm512_fmadd_ps(fp4_scale_vec(s0 + g * spb, spb), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(fp4_scale_vec(s1 + g * spb, spb), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(fp4_scale_vec(s2 + g * spb, spb), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(fp4_scale_vec(s3 + g * spb, spb), mxfp4_dot_bf16(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      // N 尾巴: N % 4 != 0 时单行 fallback
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(fp4_scale_vec(s + g * spb, spb), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

#if defined(__AVX512BF16__)
  // Convert a logical BF16 activation into the grouped decoder's natural
  // order. This is intentionally separate from the GEMV so callers can hoist
  // it out of every N-row worker task.
  static void permute_activation(int m, int k, BufferA* src, BufferA* dst) {
    const int group_count = k / 32;
    for (int mi = 0; mi < m; ++mi) {
      const __m512bh* src_row = reinterpret_cast<const __m512bh*>(src->get_submat(m, k, mi, 0));
      __m512bh* dst_row = reinterpret_cast<__m512bh*>(dst->get_submat(m, k, mi, 0));
      for (int g = 0; g < group_count; ++g) {
        dst_row[g] = permute_activation_group(src_row[g]);
      }
    }
  }

  // SGLang's resident K-pair-major layout also permits GEMV across 32
  // output channels at once. Each 32-byte load contains two adjacent K
  // values for channels 0..15 and 16..31. AMX handles prefill; AVX512 BF16
  // dot products consume those channel vectors during decode and
  // small-token prefill without reconstructing row-major weight rows.
  static void fp4_mat_vec_kmajor(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int group_count = k / 32;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(fp4_bf16)));
    const __m512i nibble_mask = _mm512_set1_epi16(15);
    const bool natural_order = ba->natural_order;
    for (int mi = 0; mi < m; ++mi) {
      const auto* activation = reinterpret_cast<const uint16_t*>(ba->get_submat(m, k, mi, 0));
      float* output = bc->get_submat(m, n, mi, n_start);
      for (int n_pos = n_start; n_pos < n_end; n_pos += 32) {
        const uint8_t* tile = bb->b + static_cast<size_t>(n_pos) * row_bytes;
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        for (int group = 0; group < group_count; ++group) {
          const uint8_t* group_scales = bb->scale_e8_kmajor + static_cast<size_t>(n_pos) * group_count + group * 32;
          const __m512 scale0 = _mm512_castsi512_ps(_mm512_slli_epi32(
              _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(group_scales))), 23));
          const __m512 scale1 = _mm512_castsi512_ps(_mm512_slli_epi32(
              _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(group_scales + 16))), 23));
          const uint8_t* src = tile + static_cast<size_t>(group) * 16 * 32;
          const uint16_t* a = activation + static_cast<size_t>(group) * 32;
          __m512 group0 = _mm512_setzero_ps();
          __m512 group1 = _mm512_setzero_ps();
          for (int pair = 0; pair < 16; ++pair) {
            const __m512i words = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
            const __m512i low = _mm512_permutexvar_epi16(_mm512_and_si512(words, nibble_mask), lut);
            const __m512i high = _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(words, 4), nibble_mask), lut);
            const uint16_t a0 = natural_order ? a[pair] : a[pair * 2];
            const uint16_t a1 = natural_order ? a[16 + pair] : a[pair * 2 + 1];
            const uint32_t pair_activation = static_cast<uint32_t>(a0) | (static_cast<uint32_t>(a1) << 16);
            const __m512bh broadcast = (__m512bh)_mm512_set1_epi32(pair_activation);
            group0 = _mm512_dpbf16_ps(group0, (__m512bh)low, broadcast);
            group1 = _mm512_dpbf16_ps(group1, (__m512bh)high, broadcast);
          }
          sum0 = _mm512_fmadd_ps(group0, scale0, sum0);
          sum1 = _mm512_fmadd_ps(group1, scale1, sum1);
        }
        _mm512_storeu_ps(output + n_pos - n_start, sum0);
        _mm512_storeu_ps(output + n_pos - n_start + 16, sum1);
      }
    }
  }

  // NVFP4 (per-16 BF16 scales) K-pair-major GEMV for decode and small-token
  // prefill. Channels occupy dpbf16 lanes exactly like the E8 variant, but
  // each 32-K block carries two scales (K 0..15 / K 16..31). Pairs 0..7 and
  // 8..15 land in those halves exactly, so each half accumulates separately
  // and folds its per-channel scale lane-wise from the transposed BF16 copy.
  static void fp4_mat_vec_kmajor_nvfp4(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int group_count = k / 16;
    const int group32_count = k / 32;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(fp4_bf16)));
    const __m512i nibble_mask = _mm512_set1_epi16(15);
    const bool natural_order = ba->natural_order;
    for (int mi = 0; mi < m; ++mi) {
      const auto* activation = reinterpret_cast<const uint16_t*>(ba->get_submat(m, k, mi, 0));
      float* output = bc->get_submat(m, n, mi, n_start);
      for (int n_pos = n_start; n_pos < n_end; n_pos += 32) {
        const uint8_t* tile = bb->b + static_cast<size_t>(n_pos) * row_bytes;
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        for (int group = 0; group < group32_count; ++group) {
          const ggml_bf16_t* scales =
              bb->scale_nv_kmajor + static_cast<size_t>(n_pos) * group_count + group * 64;
          const __m512i sv0 = _mm512_loadu_si512(scales);
          const __m512i sv1 = _mm512_loadu_si512(scales + 32);
          // 32 channels per k-group: lanes 0..15 hold channels 0..15, lanes
          // 16..31 hold channels 16..31 (dpbf16 lane halves).
          const __m512 s0a = _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(sv0));
          const __m512 s0b = _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(sv0, 1));
          const __m512 s1a = _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(sv1));
          const __m512 s1b = _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(sv1, 1));
          const uint8_t* src = tile + static_cast<size_t>(group) * 16 * 32;
          const uint16_t* a = activation + static_cast<size_t>(group) * 32;
          __m512 g0h0 = _mm512_setzero_ps();
          __m512 g0h1 = _mm512_setzero_ps();
          __m512 g1h0 = _mm512_setzero_ps();
          __m512 g1h1 = _mm512_setzero_ps();
          for (int pair = 0; pair < 16; ++pair) {
            const __m512i words = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
            const __m512i low = _mm512_permutexvar_epi16(_mm512_and_si512(words, nibble_mask), lut);
            const __m512i high = _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(words, 4), nibble_mask), lut);
            const uint16_t a0 = natural_order ? a[pair] : a[pair * 2];
            const uint16_t a1 = natural_order ? a[16 + pair] : a[pair * 2 + 1];
            const uint32_t pair_activation = static_cast<uint32_t>(a0) | (static_cast<uint32_t>(a1) << 16);
            const __m512bh broadcast = (__m512bh)_mm512_set1_epi32(pair_activation);
            if (pair < 8) {
              g0h0 = _mm512_dpbf16_ps(g0h0, (__m512bh)low, broadcast);
              g1h0 = _mm512_dpbf16_ps(g1h0, (__m512bh)high, broadcast);
            } else {
              g0h1 = _mm512_dpbf16_ps(g0h1, (__m512bh)low, broadcast);
              g1h1 = _mm512_dpbf16_ps(g1h1, (__m512bh)high, broadcast);
            }
          }
          sum0 = _mm512_fmadd_ps(s0a, g0h0, sum0);
          sum0 = _mm512_fmadd_ps(s1a, g0h1, sum0);
          sum1 = _mm512_fmadd_ps(s0b, g1h0, sum1);
          sum1 = _mm512_fmadd_ps(s1b, g1h1, sum1);
        }
        _mm512_storeu_ps(output + n_pos - n_start, sum0);
        _mm512_storeu_ps(output + n_pos - n_start + 16, sum1);
      }
    }
  }

  // Expand compact E8M0 exponents in vectors before entering the weight loop.
  // This preserves the reduced DRAM traffic without putting a byte load,
  // scalar shift and GPR-to-vector dependency on every dot product.
  static void expand_e8_scales(const uint8_t* source, float* destination, int count) {
    int group = 0;
    for (; group + 16 <= count; group += 16) {
      const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + group));
      const __m512i exponents = _mm512_slli_epi32(_mm512_cvtepu8_epi32(packed), 23);
      _mm512_store_ps(destination + group, _mm512_castsi512_ps(exponents));
    }
    for (; group < count; ++group) {
      const uint32_t bits = static_cast<uint32_t>(source[group]) << 23;
      std::memcpy(destination + group, &bits, sizeof(bits));
    }
  }

  // m=1/group32 decode fast path. BufferA must already be in natural order.
  template <bool E8_SCALE>
  static void fp4_mat_vec_kgroup_natural_impl(int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int group_count = k / 32;
    assert(group_count <= K_BLOCK / 32);
    const __m512bh* activation = reinterpret_cast<const __m512bh*>(ba->get_submat(1, k, 0, 0));
    float* output = bc->get_submat(1, n, 0, n_start);
    alignas(64) float scale_scratch[4][K_BLOCK / 32];

    int ni = n_start;
    for (; ni + 4 <= n_end; ni += 4) {
      const __m128i* w0 = reinterpret_cast<const __m128i*>(bb->get_submat(n, k, ni + 0, 0));
      const __m128i* w1 = reinterpret_cast<const __m128i*>(bb->get_submat(n, k, ni + 1, 0));
      const __m128i* w2 = reinterpret_cast<const __m128i*>(bb->get_submat(n, k, ni + 2, 0));
      const __m128i* w3 = reinterpret_cast<const __m128i*>(bb->get_submat(n, k, ni + 3, 0));
      const float* s0;
      const float* s1;
      const float* s2;
      const float* s3;
      if constexpr (E8_SCALE) {
        expand_e8_scales(bb->get_scale_e8(n, ni + 0, k, 0), scale_scratch[0], group_count);
        expand_e8_scales(bb->get_scale_e8(n, ni + 1, k, 0), scale_scratch[1], group_count);
        expand_e8_scales(bb->get_scale_e8(n, ni + 2, k, 0), scale_scratch[2], group_count);
        expand_e8_scales(bb->get_scale_e8(n, ni + 3, k, 0), scale_scratch[3], group_count);
        s0 = scale_scratch[0];
        s1 = scale_scratch[1];
        s2 = scale_scratch[2];
        s3 = scale_scratch[3];
      } else {
        s0 = bb->get_scale(n, ni + 0, k, 0);
        s1 = bb->get_scale(n, ni + 1, k, 0);
        s2 = bb->get_scale(n, ni + 2, k, 0);
        s3 = bb->get_scale(n, ni + 3, k, 0);
      }
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      auto accumulate_group = [&](int g) __attribute__((always_inline)) {
        const __m512bh a = activation[g];
        const __m512bh d0 = (__m512bh)mxfp4_to_bf16_32_natural(w0[g]);
        const __m512bh d1 = (__m512bh)mxfp4_to_bf16_32_natural(w1[g]);
        const __m512bh d2 = (__m512bh)mxfp4_to_bf16_32_natural(w2[g]);
        const __m512bh d3 = (__m512bh)mxfp4_to_bf16_32_natural(w3[g]);
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), _mm512_dpbf16_ps(_mm512_setzero_ps(), a, d0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), _mm512_dpbf16_ps(_mm512_setzero_ps(), a, d1), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), _mm512_dpbf16_ps(_mm512_setzero_ps(), a, d2), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), _mm512_dpbf16_ps(_mm512_setzero_ps(), a, d3), acc3);
      };
      auto prefetch_group = [&](int g) __attribute__((always_inline)) {
        if constexpr (E8_SCALE) {
          constexpr int prefetch_groups = 40;
          const int future = g + prefetch_groups;
          if (future < group_count) {
            _mm_prefetch(reinterpret_cast<const char*>(w0 + future), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w1 + future), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w2 + future), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w3 + future), _MM_HINT_T0);
          }
        }
      };
      int g = 0;
      for (; g + 3 < group_count; g += 4) {
        prefetch_group(g);
        accumulate_group(g);
        accumulate_group(g + 1);
        accumulate_group(g + 2);
        accumulate_group(g + 3);
      }
      for (; g < group_count; ++g) {
        prefetch_group(g);
        accumulate_group(g);
      }
      reduce4(acc0, acc1, acc2, acc3, output + (ni - n_start));
    }

    for (; ni < n_end; ++ni) {
      const __m128i* w = reinterpret_cast<const __m128i*>(bb->get_submat(n, k, ni, 0));
      const float* scales;
      if constexpr (E8_SCALE) {
        expand_e8_scales(bb->get_scale_e8(n, ni, k, 0), scale_scratch[0], group_count);
        scales = scale_scratch[0];
      } else {
        scales = bb->get_scale(n, ni, k, 0);
      }
      __m512 acc = _mm512_setzero_ps();
      auto accumulate_group = [&](int g) __attribute__((always_inline)) {
        const __m512bh d = (__m512bh)mxfp4_to_bf16_32_natural(w[g]);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(scales[g]), _mm512_dpbf16_ps(_mm512_setzero_ps(), activation[g], d), acc);
      };
      auto prefetch_group = [&](int g) __attribute__((always_inline)) {
        if constexpr (E8_SCALE) {
          constexpr int prefetch_groups = 40;
          const int future = g + prefetch_groups;
          if (future < group_count) _mm_prefetch(reinterpret_cast<const char*>(w + future), _MM_HINT_T0);
        }
      };
      int g = 0;
      for (; g + 3 < group_count; g += 4) {
        prefetch_group(g);
        accumulate_group(g);
        accumulate_group(g + 1);
        accumulate_group(g + 2);
        accumulate_group(g + 3);
      }
      for (; g < group_count; ++g) {
        prefetch_group(g);
        accumulate_group(g);
      }
      output[ni - n_start] = _mm512_reduce_add_ps(acc);
    }
  }

  static void fp4_mat_vec_kgroup_natural(int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    if (bb->scale_e8_valid) {
      fp4_mat_vec_kgroup_natural_impl<true>(n, k, ba, bb, bc, ith, nth);
    } else {
      fp4_mat_vec_kgroup_natural_impl<false>(n, k, ba, bb, bc, ith, nth);
    }
  }

#endif

#ifdef HAVE_AMX
  // MXFP4 scales are powers of two. Folding them into the BF16 weights is
  // exact for normal values, and the conversion below also handles the rare
  // underflow/overflow cases with the same BF16 rounding as the other buffers.
  static const std::array<std::array<uint32_t, 256>, 256>& scaled_fp4_lut() {
    static const auto lut = [] {
      std::array<std::array<uint32_t, 256>, 256> table{};
      constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
      for (int exponent = 1; exponent < 255; ++exponent) {
        const float scale = std::ldexp(1.0f, exponent - 127);
        uint16_t scaled[16];
        for (int value = 0; value < 16; ++value) {
          scaled[value] = GGML_FP32_TO_BF16(values[value] * scale).bits;
        }
        for (int pair = 0; pair < 256; ++pair) {
          table[exponent][pair] = static_cast<uint32_t>(scaled[pair & 15]) |
                                  (static_cast<uint32_t>(scaled[pair >> 4]) << 16);
        }
      }
      return table;
    }();
    return lut;
  }

  static void configure_fp4_amx() {
    enable_amx();
    TileConfig config;
    for (int tile = 0; tile < 8; ++tile) config.set_row_col(tile, 16, 64);
    config.set_config();
  }

  // Expand one 32-output-channel tile to AMX's VNNI B layout. A 32-K group
  // becomes 16 rows of 32 adjacent BF16 pairs. Reuse this expanded tile for
  // every 32-token M tile instead of decoding the FP4 weights for every token.
  static void pack_fp4_amx_b_scalar(int n, int k, int n_pos, BufferB* bb, uint32_t* packed) {
    const auto& lut = scaled_fp4_lut();
    const int group_count = k / 32;
    for (int group = 0; group < group_count; ++group) {
      uint32_t* destination = packed + static_cast<size_t>(group) * 16 * 32;
      for (int channel = 0; channel < 32; ++channel) {
        const int output_row = n_pos + channel;
        const auto* source = bb->get_submat(n, k, output_row, group * 32);
        const uint8_t exponent = bb->scale_e8[static_cast<size_t>(output_row) * group_count + group];
        const auto& scale_lut = lut[exponent];
        for (int pair = 0; pair < 16; ++pair) {
          destination[pair * 32 + channel] = scale_lut[source[pair]];
        }
      }
    }
  }

// NVFP4 scalar fallback: fold each per-16 FP32 scale into BF16 pair words.
  // Used when no transposed BF16 scale copy exists (fp32-scale escape hatch
  // or KT_NVFP4_KMAJOR_WEIGHTS=0) and by hosts without AVX512 BF16.
  static void pack_fp4_amx_b_scalar_nvfp4(int n, int k, int n_pos, BufferB* bb, uint32_t* packed) {
    constexpr float values[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    const int group_count = k / 32;
    for (int group = 0; group < group_count; ++group) {
      uint32_t* destination = packed + static_cast<size_t>(group) * 16 * 32;
      for (int channel = 0; channel < 32; ++channel) {
        const int output_row = n_pos + channel;
        const auto* source = bb->get_submat(n, k, output_row, group * 32);
        const float* scales = bb->get_scale(n, output_row, k, 0);
        for (int pair = 0; pair < 16; ++pair) {
          // K 2pair / 2pair+1 of this 32-block live in k-group 2g + pair/8.
          const float scale = scales[2 * group + (pair < 8 ? 0 : 1)];
          const uint8_t byte = source[pair];
          const uint16_t low = GGML_FP32_TO_BF16(values[byte & 15u] * scale).bits;
          const uint16_t high = GGML_FP32_TO_BF16(values[byte >> 4] * scale).bits;
          destination[pair * 32 + channel] = static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
        }
      }
    }
  }

#if defined(__AVX512BF16__)
  // Resident SGLang layout: skip the temporary 512-byte repack for every
  // K group. Transposed E8 scales are loaded as one vector per group.
  static void pack_fp4_amx_b_kmajor(int k, int n_pos, BufferB* bb, uint32_t* packed) {
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(fp4_bf16)));
    const __m512i nibble_mask = _mm512_set1_epi16(15);
    const __m512i abs_mask = _mm512_set1_epi16(0x7FFF);
    const __m512i zero = _mm512_setzero_si512();
    const int group_count = k / 32;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const uint8_t* tile = bb->b + static_cast<size_t>(n_pos) * row_bytes;
    const __m512i exponent_bias = _mm512_set1_epi16(127);
    alignas(64) static constexpr uint16_t duplicate_indices[32] = {
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
        8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
    const __m512i duplicate0 = _mm512_load_si512(duplicate_indices);
    const __m512i duplicate1 = _mm512_add_epi16(duplicate0, _mm512_set1_epi16(16));
    for (int group = 0; group < group_count; ++group) {
      const uint8_t* group_scales = bb->scale_e8_kmajor + static_cast<size_t>(n_pos) * group_count + group * 32;
      const __m512i exponents = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_scales)));
      const __m512i scale_offsets = _mm512_slli_epi16(_mm512_sub_epi16(exponents, exponent_bias), 7);
      const __m512i scale0 = _mm512_permutexvar_epi16(duplicate0, scale_offsets);
      const __m512i scale1 = _mm512_permutexvar_epi16(duplicate1, scale_offsets);
      const uint8_t* src = tile + static_cast<size_t>(group) * 16 * 32;
      uint32_t* destination = packed + static_cast<size_t>(group) * 16 * 32;
      for (int pair = 0; pair < 16; ++pair) {
        const __m512i words = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
        __m512i low = _mm512_permutexvar_epi16(_mm512_and_si512(words, nibble_mask), lut);
        __m512i high = _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(words, 4), nibble_mask), lut);
        const __mmask32 low_nonzero = _mm512_cmpneq_epi16_mask(_mm512_and_si512(low, abs_mask), zero);
        const __mmask32 high_nonzero = _mm512_cmpneq_epi16_mask(_mm512_and_si512(high, abs_mask), zero);
        low = _mm512_mask_add_epi16(low, low_nonzero, low, scale0);
        high = _mm512_mask_add_epi16(high, high_nonzero, high, scale1);
        _mm512_storeu_si512(destination + pair * 32, low);
        _mm512_storeu_si512(destination + pair * 32 + 16, high);
      }
    }
  }
#endif

#if defined(__AVX512BF16__)
  // NVFP4 companion of pack_fp4_amx_b_kmajor: fold the per-16 BF16 scales
  // into the FP4 pair words. bf16 lane j of a nibble vector is channel j>>1
  // at K 2pair+(j&1), so per-channel scales duplicate along lane pairs via
  // the same index trick as the E8 variant; pairs 0..7 / 8..15 select which
  // of the two k-group scales (32-K halves) multiplies the fold.
  static void pack_fp4_amx_b_kmajor_nvfp4(int k, int n_pos, BufferB* bb, uint32_t* packed) {
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(fp4_bf16)));
    const __m512i nibble_mask = _mm512_set1_epi16(15);
    const int nv_group_count = k / 16;
    const int group_count = k / 32;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const uint8_t* tile = bb->b + static_cast<size_t>(n_pos) * row_bytes;
    alignas(64) static constexpr uint16_t duplicate_indices[32] = {
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
        8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
    const __m512i duplicate = _mm512_load_si512(duplicate_indices);
    const __m512i duplicate_hi = _mm512_add_epi16(duplicate, _mm512_set1_epi16(16));
    const auto dup_lane_lo = [](const __m512i v) __attribute__((always_inline)) {
      return _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(v));
    };
    const auto dup_lane_hi = [](const __m512i v) __attribute__((always_inline)) {
      return _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(v, 1));
    };
    // Fold one 32-byte nibble block: bf16(fp4(fp nibbles) * fp32 scale)).
    // cvtpbh_ps fills lanes 0..15 and zeroes lanes 16..31, while the *_hi
    // multiplier carries dup lanes 16..31 in its lanes 0..15; cvtne2ps_pbh
    // consumes lanes 0..15 of each operand and reassembles the pair halves.
    const auto fold_block = [&](const __m512i words, const __m512 m_lo, const __m512 m_lo_hi16,
                                const __m512 m_hi, const __m512 m_hi_hi16, uint32_t* destination) {
      const __m512i low = _mm512_permutexvar_epi16(_mm512_and_si512(words, nibble_mask), lut);
      const __m512i high = _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(words, 4), nibble_mask), lut);
      const __m512 f_lo0 = _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(low)), m_lo);
      const __m512 f_lo1 = _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(low, 1)), m_lo_hi16);
      const __m512 f_hi0 = _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(high)), m_hi);
      const __m512 f_hi1 = _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(high, 1)), m_hi_hi16);
      _mm512_storeu_si512(destination, (__m512i)_mm512_cvtne2ps_pbh(f_lo1, f_lo0));
      _mm512_storeu_si512(destination + 16, (__m512i)_mm512_cvtne2ps_pbh(f_hi1, f_hi0));
    };
    for (int group = 0; group < group_count; ++group) {
      const ggml_bf16_t* scales =
          bb->scale_nv_kmajor + static_cast<size_t>(n_pos) * nv_group_count + group * 64;
      const __m512i sv0 = _mm512_loadu_si512(scales);
      const __m512i sv1 = _mm512_loadu_si512(scales + 32);
      // dup lane l <- channel l>>1 (low nibbles) / 16 + l>>1 (high nibbles),
      // per k-group half 2g (sv0) and 2g+1 (sv1).
      const __m512i dup0a = _mm512_permutexvar_epi16(duplicate, sv0);
      const __m512i dup0b = _mm512_permutexvar_epi16(duplicate_hi, sv0);
      const __m512i dup1a = _mm512_permutexvar_epi16(duplicate, sv1);
      const __m512i dup1b = _mm512_permutexvar_epi16(duplicate_hi, sv1);
      const __m512 dup0a_lo = dup_lane_lo(dup0a), dup0a_hi = dup_lane_hi(dup0a);
      const __m512 dup0b_lo = dup_lane_lo(dup0b), dup0b_hi = dup_lane_hi(dup0b);
      const __m512 dup1a_lo = dup_lane_lo(dup1a), dup1a_hi = dup_lane_hi(dup1a);
      const __m512 dup1b_lo = dup_lane_lo(dup1b), dup1b_hi = dup_lane_hi(dup1b);
      const uint8_t* src = tile + static_cast<size_t>(group) * 16 * 32;
      uint32_t* destination = packed + static_cast<size_t>(group) * 16 * 32;
      for (int pair = 0; pair < 8; ++pair) {
        const __m512i words = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
        fold_block(words, dup0a_lo, dup0a_hi, dup0b_lo, dup0b_hi, destination + pair * 32);
      }
      for (int pair = 8; pair < 16; ++pair) {
        const __m512i words = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
        fold_block(words, dup1a_lo, dup1a_hi, dup1b_lo, dup1b_hi, destination + pair * 32);
      }
    }
  }
#endif

  static void fp4_mat_mat_kgroup_amx(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                     int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    configure_fp4_amx();

    // Per-worker scratch is bounded by 32 BF16 output channels times K. No
    // expanded copy of the full model is retained in host RAM.
    thread_local std::vector<uint8_t> packed_storage;
    packed_storage.resize(static_cast<size_t>(k) * 64 + 63);
    auto* packed = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(packed_storage.data()) + 63) & ~uintptr_t(63));
    alignas(64) ggml_bf16_t tail_a[32][32] = {};
    alignas(64) float tail_c[32][32];
    const int group_count = k / 32;
    for (int n_pos = n_start; n_pos < n_end; n_pos += 32) {
#if defined(__AVX512BF16__)
      if (bb->kmajor_weights && bb->scale_e8_vector_safe) {
        pack_fp4_amx_b_kmajor(k, n_pos, bb, packed);
      } else if (bb->kmajor_weights && bb->scale_nv_valid) {
        pack_fp4_amx_b_kmajor_nvfp4(k, n_pos, bb, packed);
      } else
#endif
      {
        if (bb->k_group_size == 16) {
          pack_fp4_amx_b_scalar_nvfp4(n, k, n_pos, bb, packed);
        } else {
          pack_fp4_amx_b_scalar(n, k, n_pos, bb, packed);
        }
      }
      for (int m_pos = 0; m_pos < m; m_pos += 32) {
        const int rows = std::min(32, m - m_pos);
        _tile_zero(4);
        _tile_zero(5);
        _tile_zero(6);
        _tile_zero(7);
        for (int group = 0; group < group_count; ++group) {
          auto* a = ba->get_submat(m, k, m_pos, group * 32);
          const int k_block_start = (group * 32 / K_BLOCK) * K_BLOCK;
          const int a_stride = std::min(K_BLOCK, k - k_block_start);
          if (rows < 32) {
            for (int row = 0; row < rows; ++row) {
              std::memcpy(tail_a[row], a + static_cast<size_t>(row) * a_stride,
                          32 * sizeof(ggml_bf16_t));
            }
            a = &tail_a[0][0];
          }
          const int tile_stride = rows == 32 ? a_stride * sizeof(ggml_bf16_t) : 64;
          const uint32_t* b = packed + static_cast<size_t>(group) * 16 * 32;
          _tile_loadd(0, a, tile_stride);
          _tile_loadd(1, a + 16 * (tile_stride / sizeof(ggml_bf16_t)), tile_stride);
          _tile_loadd(2, b, 128);
          _tile_loadd(3, b + 16, 128);
          _tile_dpbf16ps(4, 0, 2);
          _tile_dpbf16ps(5, 0, 3);
          _tile_dpbf16ps(6, 1, 2);
          _tile_dpbf16ps(7, 1, 3);
        }

        float* result = rows == 32 ? bc->get_submat(m, n, m_pos, n_pos) : &tail_c[0][0];
        const int stride = rows == 32 ? N_BLOCK * sizeof(float) : 32 * sizeof(float);
        _tile_stored(4, result, stride);
        _tile_stored(5, result + 16, stride);
        _tile_stored(6, result + 16 * (stride / sizeof(float)), stride);
        _tile_stored(7, result + 16 * (stride / sizeof(float)) + 16, stride);
        if (rows < 32) {
          float* destination = bc->get_submat(m, n, m_pos, n_pos);
          for (int row = 0; row < rows; ++row) {
            std::memcpy(destination + static_cast<size_t>(row) * N_BLOCK, tail_c[row], 32 * sizeof(float));
          }
        }
      }
    }
  }
#endif

  // mat-mat: 4×4 register tile (M_TILE=4, N_TILE=4 → 16 累加器)。
  // 每 K-group 解码 4 行 N 一次, 被 4 个 token 共享 → PSHUFB 解码开销 / 4。
  // M / N 尾巴回退到 mat-vec 单 token 内层 (V4 chunked-prefill 16/32/64 整数倍, 极少触发)。
  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    const int spb = 32 / k_group_size;  // scales per 32-K block (1 MXFP4, 2 NVFP4)
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      __m512bh* a_rows[MB] = {
          (__m512bh*)ba->get_submat(m, k, m_pos + 0, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 1, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 2, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          // 4 行权重解码一次, MB 个 token 共享
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = fp4_scale_vec(s0 + g * spb, spb);
          const __m512 sv1 = fp4_scale_vec(s1 + g * spb, spb);
          const __m512 sv2 = fp4_scale_vec(s2 + g * spb, spb);
          const __m512 sv3 = fp4_scale_vec(s3 + g * spb, spb);

#define V_FMA_ROW(M_I)                                                      \
  do {                                                                      \
    const ActivationBF16 a(a_rows[M_I][g]);                                 \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp4_dot_bf16(d0, a), acc[M_I][0]); \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp4_dot_bf16(d1, a), acc[M_I][1]); \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp4_dot_bf16(d2, a), acc[M_I][2]); \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp4_dot_bf16(d3, a), acc[M_I][3]); \
  } while (0)
          V_FMA_ROW(0);
          V_FMA_ROW(1);
          V_FMA_ROW(2);
          V_FMA_ROW(3);
#undef V_FMA_ROW
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      // N 尾巴: 单 N 列 × MB token (V4 不触发)
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(fp4_scale_vec(s + g * spb, spb), mxfp4_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
        }
      }
    }
    // M 尾巴: M 不是 MB 倍数时余下 token, 退回单 token mat-vec 内层 (V4 不触发)
    for (int mi = m_pos; mi < m; mi++) {
      float* c_row = bc->get_submat(m, n, mi, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, mi, 0);
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          a0 = _mm512_fmadd_ps(fp4_scale_vec(s0 + g * spb, spb), mxfp4_dot_bf16(d0, a), a0);
          a1 = _mm512_fmadd_ps(fp4_scale_vec(s1 + g * spb, spb), mxfp4_dot_bf16(d1, a), a1);
          a2 = _mm512_fmadd_ps(fp4_scale_vec(s2 + g * spb, spb), mxfp4_dot_bf16(d2, a), a2);
          a3 = _mm512_fmadd_ps(fp4_scale_vec(s3 + g * spb, spb), mxfp4_dot_bf16(d3, a), a3);
        }
        reduce4(a0, a1, a2, a3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(fp4_scale_vec(s + g * spb, spb), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }
};

// Dispatch functions
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
#if defined(__AVX512BF16__)
  if (k_group_size == 16 && k % 32 == 0 && bb->kmajor_weights && bb->scale_nv_valid) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kmajor_nvfp4(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
  if (k_group_size == 32 && k % 32 == 0 && bb->kmajor_weights && bb->scale_e8_valid) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kmajor(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
  if (m == 1 && k_group_size == 32 && k % 32 == 0 && ba->natural_order) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup_natural(n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
  GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
#if defined(__AVX512BF16__)
  // Sparse routing can leave an expert below the AMX threshold even when
  // the overall prompt is large. Keep those tiles in K-major form too.
  if (m < 16 && k_group_size == 16 && k % 32 == 0 && bb->kmajor_weights && bb->scale_nv_valid) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kmajor_nvfp4(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
  if (m < 16 && bb->kmajor_weights && bb->scale_e8_valid && k_group_size == 32 && k % 32 == 0) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kmajor(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
#ifdef HAVE_AMX
  const char* amx_prefill = std::getenv("KT_MXFP4_PREFILL_AMX");
  if ((!amx_prefill || amx_prefill[0] != '0') &&
      m >= 16 && n >= 512 && k >= 512 &&
      n % GemmKernel224MXFP4SmallKGroup::N_BLOCK == 0 &&
      k_group_size == 32 && k % 32 == 0 && bb->scale_e8_valid) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
  const char* nvfp4_amx = std::getenv("KT_NVFP4_PREFILL_AMX");
  if ((!nvfp4_amx || nvfp4_amx[0] != '0') &&
      m >= 16 && n >= 512 && k >= 512 &&
      n % GemmKernel224MXFP4SmallKGroup::N_BLOCK == 0 &&
      k_group_size == 16 && k % 32 == 0) {
    GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return;
  }
#endif
  GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// AMX_FP4_MOE_TP — CRTP class, identical structure to AMX_K2_MOE_TP
// ============================================================================
template <class T = amx::GemmKernel224MXFP4SmallKGroup>
class AMX_FP4_MOE_TP : public AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP4_MOE_TP() = default;
  AMX_FP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("FP4 (MXFP4/NVFP4) MoE only supports KGroup FP4");
    }
    printf("Creating AMX_FP4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  bool should_allocate_expert_weights(int expert_idx) const {
    return !config_.should_skip_expert(expert_idx);
  }

  ~AMX_FP4_MOE_TP() = default;

  // BufferA: raw BF16, no group_size needed
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  void prepare_decode_gate_input(int expert_idx, int qlen, const void* input) {
    if (qlen == 1 && config_.quant_config.group_size == 32 && config_.hidden_size % 32 == 0) {
      gate_up_ba_[expert_idx]->from_mat_natural(qlen, (ggml_bf16_t*)input, 0, 1);
    } else {
      gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
    }
  }

  void prepare_decode_down_input(int expert_idx, int qlen) {
#if defined(__AVX512BF16__)
    if (qlen != 1 || config_.quant_config.group_size != 32 || config_.intermediate_size % 32 != 0) {
      Base::prepare_decode_down_input(expert_idx, qlen);
      return;
    }
    assert(down_ba_[expert_idx]->natural_order);
#else
    Base::prepare_decode_down_input(expert_idx, qlen);
#endif
  }

  void apply_decode_activation(int activated_expert, int nth, int qlen) {
#if defined(__AVX512BF16__)
    if (qlen != 1 || config_.quant_config.group_size != 32 || config_.intermediate_size % 32 != 0) {
      Base::apply_decode_activation(activated_expert, nth, qlen);
      return;
    }
    for (int task_id = 0; task_id < nth * activated_expert; ++task_id) {
      const int expert_idx = this->m_expert_id_map_[task_id / nth];
      const int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      const ggml_bf16_t* gate = m_local_gate_output_ptr_[expert_idx];
      const ggml_bf16_t* up = this->m_local_up_output_ptr_[expert_idx];
      ggml_bf16_t* destination = down_ba_[expert_idx]->get_submat(1, config_.intermediate_size, 0, n_start);
      for (int j = n_start; j < n_end; j += 32) {
        __m512 gate0, gate1, up0, up1;
        avx512_32xbf16_to_32xfp32((__m512i*)(gate + j), &gate0, &gate1);
        avx512_32xbf16_to_32xfp32((__m512i*)(up + j), &up0, &up1);
        const __m512 result0 = amx::act_fn(gate0, up0, config_.swiglu_limit, config_.swiglu_alpha);
        const __m512 result1 = amx::act_fn(gate1, up1, config_.swiglu_limit, config_.swiglu_alpha);
        const __m512bh logical = _mm512_cvtne2ps_pbh(result1, result0);
        const __m512bh natural = T::permute_activation_group(logical);
        _mm512_storeu_si512((void*)(destination + (j - n_start)), (__m512i)natural);
      }
      down_ba_[expert_idx]->natural_order = true;
    }
#else
    Base::apply_decode_activation(activated_expert, nth, qlen);
#endif
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("FP4 (MXFP4/NVFP4) MoE only support KGroup FP4.");
    if (config_.gate_scale == nullptr) throw std::runtime_error("FP4 (MXFP4/NVFP4) MoE only support load native weight.");

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          if (config_.should_skip_expert(expert_idx)) return;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          if (config_.should_skip_expert(expert_idx)) return;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          if (config_.should_skip_expert(expert_idx)) return;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;
          convert_or_copy(gate_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(up_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.up_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(down_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.down_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          gate_bb_[expert_idx]->finalize_scale_e8();
          gate_bb_[expert_idx]->finalize_scale_nv();
          up_bb_[expert_idx]->finalize_scale_e8();
          up_bb_[expert_idx]->finalize_scale_nv();
          down_bb_[expert_idx]->finalize_scale_e8();
          down_bb_[expert_idx]->finalize_scale_nv();
        },
        nullptr);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;

      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size, offset_in_gpu_weight, offset_in_gpu_scale,
           gpu_tp_weight_bytes, gpu_tp_scale_elem_count, w13_weight_dst, w13_scale_dst, w2_weight_dst, w2_scale_dst,
           group_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_id;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                gate_bb_[expert_id]->copy_weight_bytes(w13_weight_dst + offset_in_gpu_weight + start, start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                up_bb_[expert_id]->copy_weight_bytes(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start, start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; col++) {
                down_bb_[expert_id]->copy_weight_bytes(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset, col * weight_per_col, weight_per_col);
                down_bb_[expert_id]->copy_scale_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                                                        col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              gate_bb_[expert_id]->copy_scale_to_bf16(w13_scale_dst + offset_in_gpu_scale, 0, cpu_tp_scale_elem_count);
            } else {
              up_bb_[expert_id]->copy_scale_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, 0,
                                                    cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;

      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size, gpu_tp_weight_bytes, gpu_tp_scale_elem_count,
           group_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_type;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                gate_bb_[expert_id]->copy_weight_bytes(w13_weight_dst + start, cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                up_bb_[expert_id]->copy_weight_bytes(w13_weight_dst + gpu_tp_weight_bytes + start, cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

                down_bb_[expert_id]->copy_weight_bytes(w2_weight_dst + col * weight_per_gpu_col, col_offset_weight, weight_per_gpu_col);
                down_bb_[expert_id]->copy_scale_to_bf16(w2_scale_dst + col * scale_per_gpu_col, col_offset_scale,
                                                        scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              gate_bb_[expert_id]->copy_scale_to_bf16(w13_scale_dst, cpu_offset_scale, data_per_gpu_tp_scale);
            } else {
              up_bb_[expert_id]->copy_scale_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, cpu_offset_scale,
                                                    data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_FP4_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AMX_FP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("FP4 (MXFP4/NVFP4) MoE only supports Packed FP4 with KGroup Scale");

    printf("From %s\n", use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "Packed FP4 with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              if (tpc.should_skip_expert(expert_id_)) return;
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);
              memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                if (tpc.should_skip_expert(expert_id_)) return;
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);
                memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (tpc.intermediate_size >> 1));
                  memcpy((ggml_bf16_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (ggml_bf16_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done.\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (ggml_bf16_t*)(tpc.gate_scale);
      delete[] (ggml_bf16_t*)(tpc.up_scale);
      delete[] (ggml_bf16_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP4_MOE_H
