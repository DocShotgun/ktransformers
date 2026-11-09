// Stages one MoE layer's CPU-expert inputs into pinned host memory in one kernel.
//
// Decode copies the hidden rows, the routing weights and the expert ids (widened
// to int64) to the host before the CPU experts can start. As three copies and a
// cast that is four stream nodes, ~20 us per layer on the critical path; this is
// one kernel of mapped-memory stores. Prefill-sized inputs keep the copy engine.

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void stage_inputs_kernel(const uint4* hidden, uint4* host_hidden, int64_t hidden_vecs, const void* ids,
                                    bool ids_int32, int64_t* host_ids, const float* weights, float* host_weights,
                                    int64_t routes) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t i = tid; i < hidden_vecs; i += stride) host_hidden[i] = hidden[i];
  for (int64_t i = tid; i < routes; i += stride) {
    host_ids[i] = ids_int32 ? static_cast<int64_t>(static_cast<const int32_t*>(ids)[i])
                            : static_cast<const int64_t*>(ids)[i];
    host_weights[i] = weights[i];
  }
  // Make the stores visible to the host before anything later in the stream
  // (the doorbell write) is.
  __threadfence_system();
}

}  // namespace

// hidden_bytes must be a multiple of 16 and every pointer 16-byte aligned for the
// hidden rows; the caller checks both.
cudaError_t kt_stage_moe_inputs(cudaStream_t stream, const void* hidden, int64_t hidden_bytes, const void* ids,
                                bool ids_int32, const float* weights, int64_t routes, void* host_hidden,
                                int64_t* host_ids, float* host_weights) {
  const int64_t hidden_vecs = hidden_bytes / 16;
  const int64_t work = hidden_vecs > routes ? hidden_vecs : routes;
  constexpr int threads = 256;
  int blocks = static_cast<int>((work + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 64) blocks = 64;
  stage_inputs_kernel<<<blocks, threads, 0, stream>>>(static_cast<const uint4*>(hidden),
                                                      static_cast<uint4*>(host_hidden), hidden_vecs, ids, ids_int32,
                                                      host_ids, weights, host_weights, routes);
  return cudaGetLastError();
}
