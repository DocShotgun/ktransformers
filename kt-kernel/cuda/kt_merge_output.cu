// Merges one MoE layer's CPU-expert output into the GPU experts' output in one kernel.
//
// Decode copied the pinned CPU output to the device, added it to the GPU output and
// multiplied the sum by routed_scaling_factor: a copy node and two elementwise
// kernels per layer. This kernel reads the pinned output through its mapped address
// and rounds as torch's CUDA kernels do (fp32 add, bf16, fp32 multiply, bf16), so the
// result is bit-identical. Prefill-sized outputs keep the copy engine.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Inline PTX because this file is built with --use_fast_math, whose flush-to-zero
// would change subnormal results; torch rounds with cvt.rn and no ftz.
__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ uint32_t to_bf16(float x) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float lo_f32(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_f32(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

template <bool kScale>
__device__ __forceinline__ uint32_t merge_pair(uint32_t gpu, uint32_t host, float scale) {
  uint32_t lo = to_bf16(add_rn(lo_f32(gpu), lo_f32(host)));
  uint32_t hi = to_bf16(add_rn(hi_f32(gpu), hi_f32(host)));
  if (kScale) {
    lo = to_bf16(mul_rn(lo_f32(lo), scale));
    hi = to_bf16(mul_rn(lo_f32(hi), scale));
  }
  return lo | (hi << 16);
}

template <bool kScale>
__global__ void merge_output_kernel(const uint4* gpu, const uint4* host, uint4* out, int64_t vecs, float scale) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < vecs; i += stride) {
    const uint4 g = gpu[i];
    // ld.global.cv: treat any cached line of host memory as stale; the CPU rewrote it.
    const uint4 h = __ldcv(host + i);
    uint4 r;
    r.x = merge_pair<kScale>(g.x, h.x, scale);
    r.y = merge_pair<kScale>(g.y, h.y, scale);
    r.z = merge_pair<kScale>(g.z, h.z, scale);
    r.w = merge_pair<kScale>(g.w, h.w, scale);
    out[i] = r;
  }
}

}  // namespace

// out = gpu + host (bf16), then out *= scale when apply_scale. elems must be a
// multiple of 8 and every pointer 16-byte aligned; the caller checks both. host is
// pinned memory, read through its mapped (UVA) address.
cudaError_t kt_merge_moe_output(cudaStream_t stream, const void* gpu, const void* host, void* out, int64_t elems,
                                float scale, bool apply_scale) {
  const int64_t vecs = elems / 8;
  constexpr int threads = 128;
  int64_t blocks = (vecs + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 1024) blocks = 1024;
  const auto* g = static_cast<const uint4*>(gpu);
  const auto* h = static_cast<const uint4*>(host);
  auto* o = static_cast<uint4*>(out);
  if (apply_scale) {
    merge_output_kernel<true><<<static_cast<int>(blocks), threads, 0, stream>>>(g, h, o, vecs, scale);
  } else {
    merge_output_kernel<false><<<static_cast<int>(blocks), threads, 0, stream>>>(g, h, o, vecs, scale);
  }
  return cudaGetLastError();
}
