// W4A16 int4 MoE: 4-bit integer weights x BF16 activations on the FP4 family's layout,
// kernels and MoE class (AMX_FP4_MOE_TP).
//
// A weight is w = s * (u - z): u is the unsigned 4-bit code as compressed-tensors
// pack-quantized stores it (q + 8, low nibble first along K), s a BF16 scale and z the
// zero code per (output channel, K group); z = 8 for symmetric checkpoints. Any group
// size that is a multiple of 32 and divides K.
//
// Numerics. Fewer than 8 rows (decode, spec-decode verify, sparse prefill experts) are
// exact: (u - 8) is exact in BF16, each group's dpbf16 sum accumulates in FP32 and is
// scaled by one FMA; with zero points, s * (z - 8) times the row's activation sum over the
// group is subtracted. The AMX prefill folds bf16((u - z) * s) into its weight tiles, one
// rounding per weight, like a dequantize-to-BF16 GEMM.
//
// The decode task widths and prefetch distance are FP4's knobs (KT_FP4_DECODE_WIDTH,
// KT_FP4_DECODE_DOWN_WIDTH, KT_FP4_DECODE_PREFETCH).
#ifndef CPUINFER_OPERATOR_AMX_INT4_W4A16_MOE_H
#define CPUINFER_OPERATOR_AMX_INT4_W4A16_MOE_H

#include "fp4-moe.hpp"

#if defined(__AVX512BF16__) && defined(HAVE_AMX)

namespace amx {

struct GemmKernel224Int4W4A16 {
  using FP4 = GemmKernel224MXFP4SmallKGroup;
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;
  // BufferA and BufferC are FP4's, so the task split and C panels are too.
  static inline const int N_BLOCK = FP4::N_BLOCK;
  static inline const int K_BLOCK = FP4::K_BLOCK;

  static std::string name() { return "INT4_W4A16"; }
  // AMX_FP4_MOE_TP accepts zero points for kernels that declare this.
  static constexpr bool kZeroPoints = true;
  static int recommended_nth(int n) { return FP4::recommended_nth(n); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth, int m = 1) {
    return FP4::split_range_n(n, ith, nth, m);
  }
  static void config() {}

  using BufferA = FP4::BufferA;
  using BufferC = FP4::BufferC;

  static inline __m512bh permute_activation_group(__m512bh activation) {
    return FP4::permute_activation_group(activation);
  }

  // BF16 bits of u - 8 for the 16 codes u.
  alignas(32) static constexpr uint16_t kCodeBf16[16] = {0xC100, 0xC0E0, 0xC0C0, 0xC0A0, 0xC080, 0xC040,
                                                         0xC000, 0xBF80, 0x0000, 0x3F80, 0x4000, 0x4040,
                                                         0x4080, 0x40A0, 0x40C0, 0x40E0};

  // Weights in the FP4 K-major layout (see from_raw_mat), scales K-major: for each
  // 32-row block, group by group, the 32 channels' scales (one 64-byte line). Zero codes
  // follow the same order, one byte each.
  struct BufferB {
    uint8_t* b;
    ggml_bf16_t* scale;
    // nullptr for symmetric checkpoints (z = 8); else zero_area.
    uint8_t* zero = nullptr;
    uint8_t* zero_area;
    int n, k, k_group_size, k_group_count;

    static size_t required_size(int n, int k, int k_group_size) {
      return static_cast<size_t>(n) * k / 2 +
             static_cast<size_t>(n) * (k / k_group_size) * (sizeof(ggml_bf16_t) + sizeof(uint8_t));
    }

    BufferB(int n_, int k_, int k_group_size_, void* ptr) : n(n_), k(k_), k_group_size(k_group_size_) {
      if (n % 32 || k % 32 || k_group_size % 32 || k % k_group_size) {
        throw std::runtime_error(
            "INT4 W4A16 needs n and k in multiples of 32 and a group size that is a multiple "
            "of 32 dividing k; got n=" +
            std::to_string(n) + " k=" + std::to_string(k) + " group=" + std::to_string(k_group_size));
      }
      assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
      k_group_count = k / k_group_size;
      b = reinterpret_cast<uint8_t*>(ptr);
      scale = reinterpret_cast<ggml_bf16_t*>(b + static_cast<size_t>(n) * k / 2);
      zero_area = reinterpret_cast<uint8_t*>(scale + static_cast<size_t>(n) * k_group_count);
    }

    // Row-major packed codes [n, k / 2] to the FP4 K-major layout.
    void from_raw_mat(uint8_t* proj, int ith, int nth) {
      auto [start, end] = split_range_n(n, ith, nth);
      FP4::repack_kmajor(b, proj, start, end, static_cast<size_t>(k) / 2);
    }

    // Row-major BF16 scales and, unless symmetric, zero codes [n, k / group] to their
    // K-major copies.
    void load_scales(const ggml_bf16_t* scale_source, const uint8_t* zero_source) {
      FP4::for_each_kmajor(n, k_group_count, [&](size_t to, size_t from) { scale[to] = scale_source[from]; });
      if (zero_source) {
        FP4::for_each_kmajor(n, k_group_count, [&](size_t to, size_t from) { zero_area[to] = zero_source[from]; });
        zero = zero_area;
      } else {
        zero = nullptr;
        // Nothing reads the zero area; return its whole pages.
        static const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        const uintptr_t first = (reinterpret_cast<uintptr_t>(zero_area) + page - 1) & ~(page - 1);
        const uintptr_t last =
            (reinterpret_cast<uintptr_t>(zero_area) + static_cast<size_t>(n) * k_group_count) & ~(page - 1);
        if (last > first) madvise(reinterpret_cast<void*>(first), last - first, MADV_DONTNEED);
      }
    }

    // The FP4 MoE class's export to GPU weight buffers (layerwise prefill) expects the
    // FP4 checkpoint layout; nothing on the GPU side reads int4 that way.
    void copy_weight_bytes(uint8_t*, size_t, size_t) const {
      throw std::runtime_error("INT4 W4A16 does not export weights to GPU buffers");
    }
    void copy_scale_to_bf16(ggml_bf16_t*, size_t, size_t) const {
      throw std::runtime_error("INT4 W4A16 does not export scales to GPU buffers");
    }
  };

  // With zero points, w = s * ((u - 8) - (z - 8)): the shared GEMV sums a * (u - 8), and
  // this subtracts s * (z - 8) times each row's activation sum over the group, in FP32.
  template <int MB>
  struct ZeroPointRows {
    const uint8_t* zero;
    const float* sums[MB];

    ZeroPointRows(const BufferB* bb, const uint16_t* const (&activation)[MB]) : zero(bb->zero) {
      thread_local std::vector<float> storage;
      const int groups = bb->k_group_count;
      const int chunks_per_group = bb->k_group_size / 32;
      storage.resize(static_cast<size_t>(MB) * groups);
      for (int t = 0; t < MB; ++t) {
        float* row = storage.data() + static_cast<size_t>(t) * groups;
        // The order within a 32-K chunk does not matter, so natural-order rows work too.
        for (int group = 0; group < groups; ++group) {
          __m512 sum = _mm512_setzero_ps();
          for (int chunk = group * chunks_per_group; chunk < (group + 1) * chunks_per_group; ++chunk) {
            const __m512i values = _mm512_loadu_si512(activation[t] + static_cast<size_t>(chunk) * 32);
            sum = _mm512_add_ps(sum, _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(values)));
            sum = _mm512_add_ps(sum, _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(values, 1)));
          }
          row[group] = _mm512_reduce_add_ps(sum);
        }
        sums[t] = row;
      }
    }

    void apply(size_t index, int group, __m512 scale0, __m512 scale1, __m512 (&sum0)[MB], __m512 (&sum1)[MB]) const {
      const __m512 eight = _mm512_set1_ps(8.0f);
      const __m512 zero0 =
          _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(zero + index))));
      const __m512 zero1 = _mm512_cvtepi32_ps(
          _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(zero + index + 16))));
      const __m512 offset0 = _mm512_mul_ps(scale0, _mm512_sub_ps(zero0, eight));
      const __m512 offset1 = _mm512_mul_ps(scale1, _mm512_sub_ps(zero1, eight));
      for (int t = 0; t < MB; ++t) {
        const __m512 group_sum = _mm512_set1_ps(sums[t][group]);
        sum0[t] = _mm512_fnmadd_ps(offset0, group_sum, sum0[t]);
        sum1[t] = _mm512_fnmadd_ps(offset1, group_sum, sum1[t]);
      }
    }
  };

  // The format of FP4's shared K-major GEMV (FP4::kmajor_gemv): the u - 8 table, BF16
  // scales per group of k_group_size, and with ZP the zero-point term.
  template <bool ZP>
  struct Kmajor {
    static constexpr int kSegmentPairs = 16;
    static constexpr bool kSegmentIsGroup = false;
    const BufferB* bb;
    // Base pointers copied out of the buffer, so the hot loop reads them from the format.
    const uint8_t* weight_base;
    const ggml_bf16_t* scale_base;
    int groups;
    int group_chunks;
    const ggml_bf16_t* scale_end;
    // Scales take 64 B per group against 512 B per chunk of weights: a lone row prefetches
    // them as many groups ahead as the weights run chunks ahead, at least one.
    int prefetch_groups;

    Kmajor(const BufferB* bb_, int /*k*/, int n_end)
        : bb(bb_),
          weight_base(bb_->b),
          scale_base(bb_->scale),
          groups(bb_->k_group_count),
          group_chunks(bb_->k_group_size / 32),
          scale_end(bb_->scale + static_cast<size_t>(n_end) * bb_->k_group_count) {
      static const int prefetch_bytes = FP4::decode_prefetch_bytes();
      prefetch_groups = std::max(1, prefetch_bytes / (512 * group_chunks));
    }
    const uint8_t* weights() const { return weight_base; }
    int group_count() const { return groups; }
    int chunks_per_group() const { return group_chunks; }
    static __m512i lut() {
      return _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(kCodeBf16)));
    }
    void scales(size_t index, __m512& low, __m512& high) const {
      const __m512i bits = _mm512_loadu_si512(scale_base + index);
      low = _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(bits));
      high = _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(bits, 1));
    }
    void prefetch(size_t tile_index, int /*chunk*/, int group, bool group_starts, int /*prefetch_bytes*/) const {
      if (!group_starts) return;
      const ggml_bf16_t* ahead = scale_base + tile_index + static_cast<size_t>(group + prefetch_groups) * 32;
      if (ahead < scale_end) _mm_prefetch(reinterpret_cast<const char*>(ahead), _MM_HINT_T0);
    }
    template <int MB>
    auto zero_points(const uint16_t* const (&activation)[MB]) const {
      if constexpr (ZP) {
        return ZeroPointRows<MB>(bb, activation);
      } else {
        return FP4::NoZeroPoints{};
      }
    }
  };

  static void mat_vec(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth, m);
    if (n_start >= n_end) return;
    if (bb->zero) {
      FP4::kmajor_gemv(m, n, k, ba, bc, n_start, n_end, Kmajor<true>(bb, k, n_end));
    } else {
      FP4::kmajor_gemv(m, n, k, ba, bc, n_start, n_end, Kmajor<false>(bb, k, n_end));
    }
  }

  // Expand the 32-column panel at n_pos, 32-K chunks [chunk_begin, chunk_end), into
  // AMX VNNI tiles of bf16((u - z) * s), (u - 8) - (z - 8) being exact in FP32. Lane l
  // of a decoded 32-lane vector is channel l >> 1 at K 2 * pair + (l & 1), so each
  // channel's scale and zero are duplicated across its lane pair (as
  // FP4::pack_fp4_amx_b_kmajor_nvfp4 does with its scales).
  template <bool ZP>
  static void pack_amx_b(int k, int n_pos, BufferB* bb, uint32_t* packed, int chunk_begin, int chunk_end) {
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(kCodeBf16)));
    const __m512i nibble_mask = _mm512_set1_epi16(15);
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const uint8_t* tile = bb->b + static_cast<size_t>(n_pos) * row_bytes;
    const ggml_bf16_t* tile_scales = bb->scale + static_cast<size_t>(n_pos) * bb->k_group_count;
    const int chunks_per_group = bb->k_group_size / 32;
    alignas(64) static constexpr uint16_t duplicate_indices[32] = {
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
    const __m512i duplicate_lo = _mm512_load_si512(duplicate_indices);
    const __m512i duplicate_hi = _mm512_add_epi16(duplicate_lo, _mm512_set1_epi16(16));
    const auto lanes_lo = [](__m512i v) __attribute__((always_inline)) {
      return _mm512_cvtpbh_ps((__m256bh)_mm512_castsi512_si256(v));
    };
    const auto lanes_hi = [](__m512i v) __attribute__((always_inline)) {
      return _mm512_cvtpbh_ps((__m256bh)_mm512_extracti64x4_epi64(v, 1));
    };
    // 16-bit integer lanes to FP32, minus 8.
    const auto codes_lo = [](__m512i v) __attribute__((always_inline)) {
      return _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(v))), _mm512_set1_ps(8.0f));
    };
    const auto codes_hi = [](__m512i v) __attribute__((always_inline)) {
      return _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1))),
                           _mm512_set1_ps(8.0f));
    };
    [[maybe_unused]] const uint8_t* tile_zeros =
        ZP ? bb->zero + static_cast<size_t>(n_pos) * bb->k_group_count : nullptr;
    // Scales of channels 0..15 (low nibbles) and 16..31 (high), per lane half, and
    // the matching z - 8.
    __m512 low0 = _mm512_setzero_ps(), low1 = low0, high0 = low0, high1 = low0;
    [[maybe_unused]] __m512 zero_low0 = low0, zero_low1 = low0, zero_high0 = low0, zero_high1 = low0;
    int scale_group = -1;
    for (int chunk = chunk_begin; chunk < chunk_end; ++chunk) {
      const int group = chunk / chunks_per_group;
      if (group != scale_group) {
        const __m512i scales = _mm512_loadu_si512(tile_scales + static_cast<size_t>(group) * 32);
        const __m512i dup_lo = _mm512_permutexvar_epi16(duplicate_lo, scales);
        const __m512i dup_hi = _mm512_permutexvar_epi16(duplicate_hi, scales);
        low0 = lanes_lo(dup_lo);
        low1 = lanes_hi(dup_lo);
        high0 = lanes_lo(dup_hi);
        high1 = lanes_hi(dup_hi);
        if constexpr (ZP) {
          const __m512i zeros = _mm512_cvtepu8_epi16(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile_zeros + static_cast<size_t>(group) * 32)));
          const __m512i zero_dup_lo = _mm512_permutexvar_epi16(duplicate_lo, zeros);
          const __m512i zero_dup_hi = _mm512_permutexvar_epi16(duplicate_hi, zeros);
          zero_low0 = codes_lo(zero_dup_lo);
          zero_low1 = codes_hi(zero_dup_lo);
          zero_high0 = codes_lo(zero_dup_hi);
          zero_high1 = codes_hi(zero_dup_hi);
        }
        scale_group = group;
      }
      const uint8_t* src = tile + static_cast<size_t>(chunk) * 16 * 32;
      uint32_t* destination = packed + static_cast<size_t>(chunk - chunk_begin) * 16 * 32;
      for (int pair = 0; pair < 16; ++pair) {
        const __m512i words =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + pair * 32)));
        const __m512i low = _mm512_permutexvar_epi16(_mm512_and_si512(words, nibble_mask), lut);
        const __m512i high = _mm512_permutexvar_epi16(_mm512_and_si512(_mm512_srli_epi16(words, 4), nibble_mask), lut);
        __m512 l0 = lanes_lo(low), l1 = lanes_hi(low), h0 = lanes_lo(high), h1 = lanes_hi(high);
        if constexpr (ZP) {
          l0 = _mm512_sub_ps(l0, zero_low0);
          l1 = _mm512_sub_ps(l1, zero_low1);
          h0 = _mm512_sub_ps(h0, zero_high0);
          h1 = _mm512_sub_ps(h1, zero_high1);
        }
        l0 = _mm512_mul_ps(l0, low0);
        l1 = _mm512_mul_ps(l1, low1);
        h0 = _mm512_mul_ps(h0, high0);
        h1 = _mm512_mul_ps(h1, high1);
        _mm512_storeu_si512(destination + pair * 32, (__m512i)_mm512_cvtne2ps_pbh(l1, l0));
        _mm512_storeu_si512(destination + pair * 32 + 16, (__m512i)_mm512_cvtne2ps_pbh(h1, h0));
      }
    }
  }

  // Hooks for the shared AMX prefill driver (la/amx_bf16_prefill.hpp); the scales are
  // folded into the tiles, so they accumulate unscaled.
  struct AmxPrefill {
    static constexpr bool kScaledGroups = false;
    // As FP4: the stale rows of a short block feed only discarded output rows.
    static constexpr bool kZeroPadTail = false;
    int m, n, k;
    BufferA* ba;
    BufferB* bb;
    BufferC* bc;
    ggml_bf16_t* dst;

    int k_block() const { return std::min(k, FP4::AMX_K_BLOCK); }
    void pack(int n_pos, int k_begin, int k_len, uint32_t* packed) const {
      if (bb->zero) {
        pack_amx_b<true>(k, n_pos, bb, packed, k_begin / 32, (k_begin + k_len) / 32);
      } else {
        pack_amx_b<false>(k, n_pos, bb, packed, k_begin / 32, (k_begin + k_len) / 32);
      }
    }
    const ggml_bf16_t* a_rows(int m_pos, int kk) const { return ba->get_submat(m, k, m_pos, kk); }
    int a_stride(int kk) const { return ba->k_block_stride(kk / K_BLOCK * K_BLOCK); }
    bool copy_tail() const { return true; }
    void store_tiles(int m_pos, int n_pos, int rows, float (&block)[32][32]) const {
      FP4::store_amx_result(m, n, m_pos, n_pos, rows, bc, dst, block);
    }
  };

  static void mat_mat_amx(int m, int n, int k, BufferA* ba, BufferB* bb, BufferC* bc, int ith, int nth,
                          ggml_bf16_t* dst) {
    auto [n_start, n_end] = split_range_n(n, ith, nth, m);
    if (n_start >= n_end) return;
    bf16_prefill::run(AmxPrefill{m, n, k, ba, bb, bc, dst}, m, k, n_start, n_end);
  }
};

inline void vec_mul_kgroup(int m, int n, int k, int /*k_group_size*/,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferC> bc, int ith, int nth) {
  GemmKernel224Int4W4A16::mat_vec(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
}

// As the FP4 mat_mul_kgroup: returns true when the task's BF16 rows went to dst.
inline bool mat_mul_kgroup(int m, int n, int k, int /*k_group_size*/,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferA> ba,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferB> bb,
                           std::shared_ptr<GemmKernel224Int4W4A16::BufferC> bc, int ith, int nth,
                           ggml_bf16_t* dst = nullptr) {
  // Fewest rows of an expert that take the AMX prefill; the GEMV below it reads the
  // weights once per four rows. 8 measured best on Kimi K2.6 (4 and 6 equal, 12 and 16
  // up to 1.7x slower at 8-14 rows).
  static const int amx_min_rows =
      static_cast<int>(kt_env::integer("KT_W4A16_AMX_MIN_ROWS", 8, [](long long rows) { return rows >= 1; }));
  if (m < amx_min_rows) {
    GemmKernel224Int4W4A16::mat_vec(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth);
    return false;
  }
  GemmKernel224Int4W4A16::mat_mat_amx(m, n, k, ba.get(), bb.get(), bc.get(), ith, nth, dst);
  return dst != nullptr;
}

}  // namespace amx

#endif  // __AVX512BF16__ && HAVE_AMX

#endif  // CPUINFER_OPERATOR_AMX_INT4_W4A16_MOE_H
