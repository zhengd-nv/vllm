#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

constexpr int kMaxWorldSize = 8;
constexpr int kSignalPhases = 8;

enum class CustomAlgo : int {
  kOneshot = 0,
  kTwostageFast = 1,
  kTree = 2,
  kRsag = 3,
  kTopoTree = 4,
  kTopoRsag = 5,
  kTp2Oneshot = 6,
  kFastOneshot = 7,
  kTopoTreeFastFinal = 8,
  kTopoRsagFastFinal = 9,
  kTopoRsagPush = 10,
  kTopoRsagWide2 = 11,
  kTopoRsagWide4 = 12,
  kTopoRsagPipelined = 13,
  kTopoRsagPipelinedDirect = 14,
  kTopoRsagPipelinedCrossPush = 15,
  kTopoRsagPipelinedWide2 = 16,
  kTopoRsagPipelinedChunks8 = 17,
  kTopoRsagPipelinedGatherStart = 18,
  kTopoRsagPipelinedAgPush = 19,
  kTopoRsagPipelinedAgPushSplit = 20,
  kTp2OneshotPairFast = 21,
  kPushOneshot = 22,
  kTp2RemotePush = 23,
  kTp2RemotePushStream = 24,
  kTp2RemotePushWindow4 = 25,
};

struct LaunchPolicy {
  CustomAlgo algo;
  int blocks;
  int threads;
};

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
};

template <typename T>
struct PackTraits {
  static constexpr int kPackElems = 16 / sizeof(T);
  using Pack = Vec<T, kPackElems>;
};

template <typename T, int VecPacks>
struct alignas(16 * VecPacks) PackGroup {
  typename PackTraits<T>::Pack packs[VecPacks];
};

__device__ __forceinline__ void pdl_grid_sync(bool enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (enabled) {
    cudaGridDependencySynchronize();
  }
#else
  (void)enabled;
#endif
}

__device__ __forceinline__ void pdl_grid_release(bool enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (enabled) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
#else
  (void)enabled;
#endif
}

template <bool Enabled>
__device__ __forceinline__ void pdl_grid_sync_const() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Enabled) {
    cudaGridDependencySynchronize();
  }
#endif
}

template <bool Enabled>
__device__ __forceinline__ void pdl_grid_release_const() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (Enabled) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
#endif
}

__device__ __forceinline__ void store_release_i32(int32_t* addr, int32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(value), "l"(addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(value),
               "l"(addr));
#endif
}

__device__ __forceinline__ int32_t load_acquire_i32(int32_t* addr) {
  int32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;"
               : "=r"(value)
               : "l"(addr));
#endif
  return value;
}

__device__ __forceinline__ void store_volatile_i32(int32_t* addr,
                                                   int32_t value) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(value), "l"(addr));
}

__device__ __forceinline__ int32_t load_volatile_i32(int32_t* addr) {
  int32_t value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(addr));
  return value;
}

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

template <>
__device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

__device__ __forceinline__ uint32_t add_half2_u32(uint32_t a, uint32_t b) {
  auto ah = *reinterpret_cast<half2*>(&a);
  auto bh = *reinterpret_cast<half2*>(&b);
  half2 out = __hadd2(ah, bh);
  return *reinterpret_cast<uint32_t*>(&out);
}

__device__ __forceinline__ uint32_t add_bfloat162_u32(uint32_t a, uint32_t b) {
  auto ah = *reinterpret_cast<__nv_bfloat162*>(&a);
  auto bh = *reinterpret_cast<__nv_bfloat162*>(&b);
  __nv_bfloat162 out = __hadd2(ah, bh);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <typename T>
__device__ __forceinline__ uint4 packed_add_u4(uint4 a, uint4 b) {
  if constexpr (std::is_same_v<T, half>) {
    a.x = add_half2_u32(a.x, b.x);
    a.y = add_half2_u32(a.y, b.y);
    a.z = add_half2_u32(a.z, b.z);
    a.w = add_half2_u32(a.w, b.w);
  } else {
    static_assert(std::is_same_v<T, nv_bfloat16>);
    a.x = add_bfloat162_u32(a.x, b.x);
    a.y = add_bfloat162_u32(a.y, b.y);
    a.z = add_bfloat162_u32(a.z, b.z);
    a.w = add_bfloat162_u32(a.w, b.w);
  }
  return a;
}

template <typename T>
__device__ __forceinline__ float2 lane_to_float2(uint32_t lane);

template <>
__device__ __forceinline__ float2 lane_to_float2<half>(uint32_t lane) {
  auto value = *reinterpret_cast<half2*>(&lane);
  return __half22float2(value);
}

template <>
__device__ __forceinline__ float2 lane_to_float2<nv_bfloat16>(uint32_t lane) {
  auto value = *reinterpret_cast<__nv_bfloat162*>(&lane);
  return __bfloat1622float2(value);
}

template <typename T>
__device__ __forceinline__ uint32_t float2_to_lane(float2 value);

template <>
__device__ __forceinline__ uint32_t float2_to_lane<half>(float2 value) {
  half2 out = __float22half2_rn(value);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <>
__device__ __forceinline__ uint32_t float2_to_lane<nv_bfloat16>(float2 value) {
  __nv_bfloat162 out = __float22bfloat162_rn(value);
  return *reinterpret_cast<uint32_t*>(&out);
}

template <typename T, int WorldSize>
__device__ __forceinline__ uint4 reduce_u4_fp32(
    uint4 const (&values)[WorldSize]) {
  float2 acc0 = {0.0f, 0.0f};
  float2 acc1 = {0.0f, 0.0f};
  float2 acc2 = {0.0f, 0.0f};
  float2 acc3 = {0.0f, 0.0f};
#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer) {
    float2 v0 = lane_to_float2<T>(values[peer].x);
    float2 v1 = lane_to_float2<T>(values[peer].y);
    float2 v2 = lane_to_float2<T>(values[peer].z);
    float2 v3 = lane_to_float2<T>(values[peer].w);
    acc0.x += v0.x;
    acc0.y += v0.y;
    acc1.x += v1.x;
    acc1.y += v1.y;
    acc2.x += v2.x;
    acc2.y += v2.y;
    acc3.x += v3.x;
    acc3.y += v3.y;
  }
  uint4 out;
  out.x = float2_to_lane<T>(acc0);
  out.y = float2_to_lane<T>(acc1);
  out.z = float2_to_lane<T>(acc2);
  out.w = float2_to_lane<T>(acc3);
  return out;
}

template <typename T>
struct ZeroBits;

template <>
struct ZeroBits<half> {
  using Raw = uint16_t;
  static constexpr Raw kPos = 0x0000u;
  static constexpr Raw kNeg = 0x8000u;
};

template <>
struct ZeroBits<nv_bfloat16> {
  using Raw = uint16_t;
  static constexpr Raw kPos = 0x0000u;
  static constexpr Raw kNeg = 0x8000u;
};

template <>
struct ZeroBits<float> {
  using Raw = uint32_t;
  static constexpr Raw kPos = 0x00000000u;
  static constexpr Raw kNeg = 0x80000000u;
};

template <typename T>
__device__ __forceinline__ void clear_pos_zero(T& value) {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw* raw = reinterpret_cast<Raw*>(&value);
  if (*raw == Bits::kPos) {
    *raw = Bits::kNeg;
  }
}

template <typename T>
__device__ __forceinline__ bool is_pos_zero(T value) {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw raw = *reinterpret_cast<Raw*>(&value);
  return raw == Bits::kPos;
}

template <typename T>
__device__ __forceinline__ T pos_zero() {
  using Bits = ZeroBits<T>;
  using Raw = typename Bits::Raw;
  Raw raw = Bits::kPos;
  return *reinterpret_cast<T*>(&raw);
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack load_pack_volatile(
    typename PackTraits<T>::Pack const* base, int idx) {
  uint4 raw;
  auto const* addr = reinterpret_cast<uint4 const*>(base + idx);
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
               : "l"(addr));
  return *reinterpret_cast<typename PackTraits<T>::Pack*>(&raw);
}

template <typename T>
__device__ __forceinline__ void store_pack_volatile(
    typename PackTraits<T>::Pack* base, int idx,
    typename PackTraits<T>::Pack value) {
  uint4 raw = *reinterpret_cast<uint4*>(&value);
  auto* addr = reinterpret_cast<uint4*>(base + idx);
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(
                   raw.x),
               "r"(raw.y), "r"(raw.z), "r"(raw.w), "l"(addr));
}

template <typename T>
__device__ __forceinline__ void clear_pos_zero_pack(
    typename PackTraits<T>::Pack& pack) {
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    clear_pos_zero(pack.data[i]);
  }
}

template <typename T>
__device__ __forceinline__ bool has_pos_zero_pack(
    typename PackTraits<T>::Pack const& pack) {
  bool has_zero = false;
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    has_zero |= is_pos_zero(pack.data[i]);
  }
  return has_zero;
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack zero_pack() {
  typename PackTraits<T>::Pack pack;
#pragma unroll
  for (int i = 0; i < PackTraits<T>::kPackElems; ++i) {
    pack.data[i] = pos_zero<T>();
  }
  return pack;
}

__device__ __forceinline__ uint32_t clear_pos_zero_u16x2(uint32_t raw) {
  uint32_t lo = raw & 0xffffu;
  uint32_t hi = raw & 0xffff0000u;
  if (lo == 0u) {
    lo = 0x8000u;
  }
  if (hi == 0u) {
    hi = 0x80000000u;
  }
  return hi | lo;
}

__device__ __forceinline__ bool has_pos_zero_u16x2(uint32_t raw) {
  return (raw & 0xffffu) == 0u || (raw & 0xffff0000u) == 0u;
}

__device__ __forceinline__ uint4 clear_pos_zero_u4_16(uint4 value) {
  value.x = clear_pos_zero_u16x2(value.x);
  value.y = clear_pos_zero_u16x2(value.y);
  value.z = clear_pos_zero_u16x2(value.z);
  value.w = clear_pos_zero_u16x2(value.w);
  return value;
}

__device__ __forceinline__ bool has_pos_zero_u4_16(uint4 value) {
  return has_pos_zero_u16x2(value.x) || has_pos_zero_u16x2(value.y) ||
         has_pos_zero_u16x2(value.z) || has_pos_zero_u16x2(value.w);
}

__device__ __forceinline__ uint4 load_u4_volatile(uint4 const* base, int idx) {
  uint4 value;
  auto const* addr = base + idx;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(addr));
  return value;
}

__device__ __forceinline__ void store_u4_volatile(uint4* base, int idx,
                                                  uint4 value) {
  auto* addr = base + idx;
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(
                   value.x),
               "r"(value.y), "r"(value.z), "r"(value.w), "l"(addr));
}

__device__ __forceinline__ int phase_offset(int phase, int block, int peer,
                                            int max_blocks, int world_size) {
  return phase * max_blocks * world_size + block * world_size + peer;
}

__device__ __forceinline__ int flag_offset(int block, int max_blocks,
                                           int world_size) {
  return kSignalPhases * max_blocks * world_size + block;
}

__device__ __forceinline__ void block_barrier(uint64_t const* signal_ptrs,
                                              int rank, int world_size,
                                              int max_blocks, int phase,
                                              int flag) {
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x < world_size) {
    int peer = threadIdx.x;
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_release_i32(
        peer_signal + phase_offset(phase, block, rank, max_blocks, world_size),
        flag);
    int32_t* self_slot =
        self + phase_offset(phase, block, peer, max_blocks, world_size);
    while (load_acquire_i32(self_slot) < flag) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void tp2_pair_barrier(
    uint64_t const* signal_ptrs, int rank, int max_blocks, int phase,
    int flag) {
  int block = blockIdx.x;
  int peer = rank ^ 1;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x == 0) {
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_release_i32(peer_signal + phase_offset(phase, block, rank,
                                                 max_blocks, 2),
                      flag);
    int32_t* self_slot =
        self + phase_offset(phase, block, peer, max_blocks, 2);
    while (load_acquire_i32(self_slot) < flag) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void tp2_pair_barrier_fast_start(
    uint64_t const* signal_ptrs, int rank, int max_blocks) {
  int block = blockIdx.x;
  int peer = rank ^ 1;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_volatile_i32(self + flag_offset(block, max_blocks, 2)) + 1;
  if (threadIdx.x == 0) {
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_volatile_i32(
        peer_signal + phase_offset(0, block, rank, max_blocks, 2), flag);
    int32_t* self_slot =
        self + phase_offset(0, block, peer, max_blocks, 2);
    while (load_volatile_i32(self_slot) != flag) {
    }
    store_volatile_i32(self + flag_offset(block, max_blocks, 2), flag);
  }
  __syncthreads();
}

__device__ __forceinline__ void tp2_pair_barrier_fast_final(
    uint64_t const* signal_ptrs, int rank, int max_blocks, int phase) {
  __syncthreads();
  int block = blockIdx.x;
  int peer = rank ^ 1;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_volatile_i32(self + flag_offset(block, max_blocks, 2)) + 1;
  if (threadIdx.x == 0) {
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_volatile_i32(
        peer_signal + phase_offset(phase, block, rank, max_blocks, 2), flag);
    int32_t* self_slot =
        self + phase_offset(phase, block, peer, max_blocks, 2);
    while (load_volatile_i32(self_slot) != flag) {
    }
    store_volatile_i32(self + flag_offset(block, max_blocks, 2), flag);
  }
}

__device__ __forceinline__ void block_barrier_mask(
    uint64_t const* signal_ptrs, int rank, int world_size, int max_blocks,
    int phase, int flag, uint32_t participant_mask) {
  if ((participant_mask & (1u << rank)) == 0u) {
    __syncthreads();
    return;
  }
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x < world_size) {
    int peer = threadIdx.x;
    if ((participant_mask & (1u << peer)) != 0u) {
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_release_i32(
          peer_signal + phase_offset(phase, block, rank, max_blocks, world_size),
          flag);
      int32_t* self_slot =
          self + phase_offset(phase, block, peer, max_blocks, world_size);
      while (load_acquire_i32(self_slot) < flag) {
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void block_barrier_mask_fast_final(
    uint64_t const* signal_ptrs, int rank, int world_size, int max_blocks,
    int phase, int flag, uint32_t participant_mask) {
  __syncthreads();
  if ((participant_mask & (1u << rank)) == 0u) {
    return;
  }
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  if (threadIdx.x < world_size) {
    int peer = threadIdx.x;
    if ((participant_mask & (1u << peer)) != 0u) {
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_volatile_i32(
          peer_signal + phase_offset(phase, block, rank, max_blocks,
                                     world_size),
          flag);
      int32_t* self_slot =
          self + phase_offset(phase, block, peer, max_blocks, world_size);
      while (load_volatile_i32(self_slot) != flag) {
      }
    }
  }
}

__device__ __forceinline__ void island_owner_gather(
    uint64_t const* signal_ptrs, int rank, int base, int owner,
    int max_blocks, int phase, int flag) {
  int block = blockIdx.x;
  int32_t* owner_signal = reinterpret_cast<int32_t*>(signal_ptrs[owner]);
  if (threadIdx.x == 0) {
    store_release_i32(
        owner_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
  }
  if (rank == owner && threadIdx.x < 4) {
    int peer = base + threadIdx.x;
    int32_t* self_slot =
        owner_signal + phase_offset(phase, block, peer, max_blocks, 8);
    while (load_acquire_i32(self_slot) < flag) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void owner_pair_barrier(
    uint64_t const* signal_ptrs, int rank, int owner, int cross_owner,
    int max_blocks, int phase, int flag) {
  __syncthreads();
  if (rank == owner && threadIdx.x == 0) {
    int block = blockIdx.x;
    int32_t* cross_signal =
        reinterpret_cast<int32_t*>(signal_ptrs[cross_owner]);
    store_release_i32(
        cross_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
    int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
    int32_t* self_slot =
        self_signal + phase_offset(phase, block, cross_owner, max_blocks, 8);
    while (load_acquire_i32(self_slot) < flag) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void island_owner_ready(
    uint64_t const* signal_ptrs, int rank, int base, int owner,
    int max_blocks, int phase, int flag) {
  __syncthreads();
  int block = blockIdx.x;
  if (rank == owner) {
    if (threadIdx.x < 4) {
      int peer = base + threadIdx.x;
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_release_i32(
          peer_signal + phase_offset(phase, block, owner, max_blocks, 8),
          flag);
    }
  } else if (threadIdx.x == 0) {
    int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
    int32_t* self_slot =
        self_signal + phase_offset(phase, block, owner, max_blocks, 8);
    while (load_acquire_i32(self_slot) < flag) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void island_owner_ack(
    uint64_t const* signal_ptrs, int rank, int base, int owner,
    int max_blocks, int phase, int flag) {
  __syncthreads();
  int block = blockIdx.x;
  int32_t* owner_signal = reinterpret_cast<int32_t*>(signal_ptrs[owner]);
  if (rank != owner && threadIdx.x == 0) {
    store_release_i32(
        owner_signal + phase_offset(phase, block, rank, max_blocks, 8), flag);
  }
  if (rank == owner && threadIdx.x < 4) {
    int peer = base + threadIdx.x;
    if (peer != owner) {
      int32_t* self_slot =
          owner_signal + phase_offset(phase, block, peer, max_blocks, 8);
      while (load_acquire_i32(self_slot) < flag) {
      }
    }
  }
  __syncthreads();
}

template <int WorldSize>
__device__ __forceinline__ void block_barrier_fast_start(
    uint64_t const* signal_ptrs, int rank, int max_blocks) {
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_volatile_i32(self + flag_offset(block, max_blocks,
                                                  WorldSize)) +
             1;
  if (threadIdx.x < WorldSize) {
    int peer = threadIdx.x;
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_volatile_i32(
        peer_signal + phase_offset(0, block, rank, max_blocks, WorldSize),
        flag);
    int32_t* self_slot =
        self + phase_offset(0, block, peer, max_blocks, WorldSize);
    while (load_volatile_i32(self_slot) != flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    store_volatile_i32(self + flag_offset(block, max_blocks, WorldSize), flag);
  }
}

template <int WorldSize>
__device__ __forceinline__ void block_barrier_fast_mid(
    uint64_t const* signal_ptrs, int rank, int max_blocks) {
  __syncthreads();
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_volatile_i32(self + flag_offset(block, max_blocks,
                                                  WorldSize)) +
             1;
  if (threadIdx.x < WorldSize) {
    int peer = threadIdx.x;
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_release_i32(
        peer_signal + phase_offset(1, block, rank, max_blocks, WorldSize),
        flag);
    int32_t* self_slot =
        self + phase_offset(1, block, peer, max_blocks, WorldSize);
    while (load_acquire_i32(self_slot) != flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    store_volatile_i32(self + flag_offset(block, max_blocks, WorldSize), flag);
  }
}

template <int WorldSize>
__device__ __forceinline__ void block_barrier_fast_final(
    uint64_t const* signal_ptrs, int rank, int max_blocks, int phase) {
  __syncthreads();
  int block = blockIdx.x;
  int32_t* self = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_volatile_i32(self + flag_offset(block, max_blocks,
                                                  WorldSize)) +
             1;
  if (threadIdx.x < WorldSize) {
    int peer = threadIdx.x;
    int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
    store_volatile_i32(
        peer_signal + phase_offset(phase, block, rank, max_blocks, WorldSize),
        flag);
    int32_t* self_slot =
        self + phase_offset(phase, block, peer, max_blocks, WorldSize);
    while (load_volatile_i32(self_slot) != flag) {
    }
  }
  if (threadIdx.x == 0) {
    store_volatile_i32(self + flag_offset(block, max_blocks, WorldSize), flag);
  }
}

template <typename T, int WorldSize>
__device__ __forceinline__ typename PackTraits<T>::Pack reduce_pack(
    uint64_t const* input_ptrs, int pack_idx) {
  using Traits = PackTraits<T>;
  using Pack = typename Traits::Pack;
  if constexpr (false &&
                (std::is_same_v<T, half> ||
                 std::is_same_v<T, nv_bfloat16>)) {
    auto* first = reinterpret_cast<uint4 const*>(input_ptrs[0]);
    uint4 acc = first[pack_idx];
#pragma unroll
    for (int peer = 1; peer < WorldSize; ++peer) {
      auto* peer_data = reinterpret_cast<uint4 const*>(input_ptrs[peer]);
      acc = packed_add_u4<T>(acc, peer_data[pack_idx]);
    }
    return *reinterpret_cast<Pack*>(&acc);
  }

  float acc[Traits::kPackElems];

#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int peer = 0; peer < WorldSize; ++peer) {
    auto* peer_data = reinterpret_cast<Pack const*>(input_ptrs[peer]);
    Pack value = peer_data[pack_idx];
#pragma unroll
    for (int i = 0; i < Traits::kPackElems; ++i) {
      acc[i] += to_float(value.data[i]);
    }
  }

  Pack out;
#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    out.data[i] = from_float<T>(acc[i]);
  }
  return out;
}

template <typename T>
__device__ __forceinline__ typename PackTraits<T>::Pack add_pack(
    typename PackTraits<T>::Pack a, typename PackTraits<T>::Pack b) {
  using Traits = PackTraits<T>;
  using Pack = typename Traits::Pack;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 av = *reinterpret_cast<uint4*>(&a);
    uint4 bv = *reinterpret_cast<uint4*>(&b);
    uint4 out = packed_add_u4<T>(av, bv);
    return *reinterpret_cast<Pack*>(&out);
  }

  Pack out;
#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    out.data[i] = from_float<T>(to_float(a.data[i]) + to_float(b.data[i]));
  }
  return out;
}

template <typename T, int VecPacks>
__device__ __forceinline__ PackGroup<T, VecPacks> add_pack_group(
    PackGroup<T, VecPacks> a, PackGroup<T, VecPacks> b) {
  PackGroup<T, VecPacks> out;
#pragma unroll
  for (int i = 0; i < VecPacks; ++i) {
    out.packs[i] = add_pack<T>(a.packs[i], b.packs[i]);
  }
  return out;
}

template <typename T, int WorldSize>
__device__ __forceinline__ typename PackTraits<T>::Pack reduce_loaded_packs(
    typename PackTraits<T>::Pack const (&values)[WorldSize]) {
  using Pack = typename PackTraits<T>::Pack;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 acc = *reinterpret_cast<uint4 const*>(&values[0]);
#pragma unroll
    for (int peer = 1; peer < WorldSize; ++peer) {
      uint4 next = *reinterpret_cast<uint4 const*>(&values[peer]);
      acc = packed_add_u4<T>(acc, next);
    }
    return *reinterpret_cast<Pack*>(&acc);
  } else {
    Pack acc = values[0];
#pragma unroll
    for (int peer = 1; peer < WorldSize; ++peer) {
      acc = add_pack<T>(acc, values[peer]);
    }
    return acc;
  }
}

template <typename T, int WorldSize>
__device__ __forceinline__ typename PackTraits<T>::Pack reduce_pack_rotated(
    uint64_t const* input_ptrs, int pack_idx, int rank) {
  using Traits = PackTraits<T>;
  using Pack = typename Traits::Pack;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    auto* first = reinterpret_cast<uint4 const*>(input_ptrs[rank]);
    uint4 acc = first[pack_idx];
#pragma unroll
    for (int i = 1; i < WorldSize; ++i) {
      int peer = (rank + i) % WorldSize;
      auto* peer_data = reinterpret_cast<uint4 const*>(input_ptrs[peer]);
      acc = packed_add_u4<T>(acc, peer_data[pack_idx]);
    }
    return *reinterpret_cast<Pack*>(&acc);
  }

  float acc[Traits::kPackElems];

#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    acc[i] = 0.0f;
  }

#pragma unroll
  for (int i = 0; i < WorldSize; ++i) {
    int peer = (rank + i) % WorldSize;
    auto* peer_data = reinterpret_cast<Pack const*>(input_ptrs[peer]);
    Pack value = peer_data[pack_idx];
#pragma unroll
    for (int elem = 0; elem < Traits::kPackElems; ++elem) {
      acc[elem] += to_float(value.data[elem]);
    }
  }

  Pack out;
#pragma unroll
  for (int i = 0; i < Traits::kPackElems; ++i) {
    out.data[i] = from_float<T>(acc[i]);
  }
  return out;
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void oneshot_kernel(
    uint64_t const* input_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       WorldSize)) +
             1;

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 0, flag);

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_packs;
       idx += gridDim.x * blockDim.x) {
    reinterpret_cast<Pack*>(output)[idx] =
        reduce_pack<T, WorldSize>(input_ptrs, idx);
  }

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 2, flag);
  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                WorldSize),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, bool PairFast>
__global__ __launch_bounds__(512, 1) void tp2_oneshot_kernel(
    uint64_t const* input_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);

  if constexpr (PairFast) {
    tp2_pair_barrier_fast_start(signal_ptrs, rank, max_blocks);
  } else {
    block_barrier_fast_start<2>(signal_ptrs, rank, max_blocks);
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    auto const* first = reinterpret_cast<uint4 const*>(input_ptrs[0]);
    auto const* second = reinterpret_cast<uint4 const*>(input_ptrs[1]);
    auto* out = reinterpret_cast<uint4*>(output);
    if (stride >= num_packs) {
      if (tid < num_packs) {
        out[tid] = packed_add_u4<T>(first[tid], second[tid]);
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        out[idx] = packed_add_u4<T>(first[idx], second[idx]);
      }
    }
  } else {
    auto const* local = reinterpret_cast<Pack const*>(input_ptrs[rank]);
    auto const* peer = reinterpret_cast<Pack const*>(input_ptrs[rank ^ 1]);
    auto* out = reinterpret_cast<Pack*>(output);
    if (stride >= num_packs) {
      if (tid < num_packs) {
        out[tid] = add_pack<T>(local[tid], peer[tid]);
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        out[idx] = add_pack<T>(local[idx], peer[idx]);
      }
    }
  }

  if constexpr (PairFast) {
    tp2_pair_barrier_fast_final(signal_ptrs, rank, max_blocks, 2);
  } else {
    block_barrier_fast_final<2>(signal_ptrs, rank, max_blocks, 2);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, bool StartBarrier>
__global__ __launch_bounds__(1024, 1) void tp2_input_pull_kernel(
    uint64_t const* input_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  if constexpr (StartBarrier) {
    tp2_pair_barrier_fast_start(signal_ptrs, rank, max_blocks);
  }

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    auto const* local = reinterpret_cast<uint4 const*>(input_ptrs[rank]);
    auto const* peer = reinterpret_cast<uint4 const*>(input_ptrs[rank ^ 1]);
    auto* out = reinterpret_cast<uint4*>(output);
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 local_value = local[idx];
      uint4 peer_value = peer[idx];
      out[idx] = packed_add_u4<T>(local_value, peer_value);
    }
  } else {
    auto const* local = reinterpret_cast<Pack const*>(input_ptrs[rank]);
    auto const* peer = reinterpret_cast<Pack const*>(input_ptrs[rank ^ 1]);
    auto* out = reinterpret_cast<Pack*>(output);
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack local_value = local[idx];
      Pack peer_value = peer[idx];
      out[idx] = add_pack<T>(local_value, peer_value);
    }
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int WorldSize, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void push_oneshot_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, int32_t* epoch_slots,
    T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * WorldSize * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 value = clear_pos_zero_u4_16(input[idx]);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
        int peer_offset = stage_offset + rank * num_packs + idx;
        store_u4_volatile(peer_buffer, peer_offset, value);
      }
    }

    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset = stage_offset + peer * num_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, peer_offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      uint4 acc = values[0];
#pragma unroll
      for (int peer = 1; peer < WorldSize; ++peer) {
        acc = packed_add_u4<T>(acc, values[peer]);
      }
      reinterpret_cast<uint4*>(output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset = stage_offset + peer * num_packs + idx;
        store_u4_volatile(local_buffer, peer_offset, reset);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        Pack* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
        int peer_offset = stage_offset + rank * num_packs + idx;
        store_pack_volatile<T>(peer_buffer, peer_offset, value);
      }
    }

    Pack* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    Pack reset = zero_pack<T>();
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset = stage_offset + peer * num_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, peer_offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset = stage_offset + peer * num_packs + idx;
        store_pack_volatile<T>(local_buffer, peer_offset, reset);
      }
    }
  }

  pdl_grid_release_const<UsePdl>();
  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void tp2_remote_push_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, int32_t* epoch_slots,
    T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
    int peer_write_base = stage_offset + rank * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 value = clear_pos_zero_u4_16(input[idx]);
      store_u4_volatile(peer_buffer, peer_write_base + idx, value);
    }

    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    int local_poll_base = stage_offset + peer * num_packs;
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 peer_value;
      while (true) {
        peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
        if (!has_pos_zero_u4_16(peer_value)) {
          break;
        }
      }
      uint4 local_value = input[idx];
      reinterpret_cast<uint4*>(output)[idx] =
          packed_add_u4<T>(local_value, peer_value);
      store_u4_volatile(local_buffer, local_poll_base + idx, reset);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
    int peer_write_base = stage_offset + rank * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
      store_pack_volatile<T>(peer_buffer, peer_write_base + idx, value);
    }

    auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    int local_poll_base = stage_offset + peer * num_packs;
    Pack reset = zero_pack<T>();
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack peer_value;
      while (true) {
        peer_value = load_pack_volatile<T>(local_buffer, local_poll_base + idx);
        if (!has_pos_zero_pack<T>(peer_value)) {
          break;
        }
      }
      reinterpret_cast<Pack*>(output)[idx] =
          add_pack<T>(input[idx], peer_value);
      store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
    }
  }

  pdl_grid_release_const<UsePdl>();
  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
}

template <typename T, bool UsePdl, bool FastSignal>
__global__ __launch_bounds__(1024, 1) void tp2_remote_signal_push_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, uint64_t const* signal_ptrs,
    int32_t* epoch_slots, T* output, int num_packs, int rank, int max_blocks) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
    int peer_write_base = stage_offset + rank * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      store_u4_volatile(peer_buffer, peer_write_base + idx, input[idx]);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
    int peer_write_base = stage_offset + rank * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      store_pack_volatile<T>(peer_buffer, peer_write_base + idx, input[idx]);
    }
  }

  __syncthreads();
  if constexpr (FastSignal) {
    tp2_pair_barrier_fast_start(signal_ptrs, rank, max_blocks);
  } else {
    int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
    int flag =
        load_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 2)) +
        1;
    if (threadIdx.x == 0) {
      int32_t* peer_signal = reinterpret_cast<int32_t*>(signal_ptrs[peer]);
      store_release_i32(peer_signal +
                            phase_offset(6, blockIdx.x, rank, max_blocks, 2),
                        flag);
      int32_t* self_slot =
          self_signal + phase_offset(6, blockIdx.x, peer, max_blocks, 2);
      while (load_acquire_i32(self_slot) < flag) {
      }
      store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 2),
                         flag);
    }
  }
  __syncthreads();

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    int local_poll_base = stage_offset + peer * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
      reinterpret_cast<uint4*>(output)[idx] =
          packed_add_u4<T>(input[idx], peer_value);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    int local_poll_base = stage_offset + peer * num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack peer_value = load_pack_volatile<T>(local_buffer, local_poll_base + idx);
      reinterpret_cast<Pack*>(output)[idx] =
          add_pack<T>(input[idx], peer_value);
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T, int VecPacks, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void tp2_remote_push_wide_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, int32_t* epoch_slots,
    T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * num_packs;
  int num_groups = num_packs / VecPacks;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  Pack const* input = reinterpret_cast<Pack const*>(input_data);
  Pack* out = reinterpret_cast<Pack*>(output);
  auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
  auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  int peer_write_base = stage_offset + rank * num_packs;
  int local_poll_base = stage_offset + peer * num_packs;
  Pack reset = zero_pack<T>();

  for (int group = tid; group < num_groups; group += stride) {
    int base = group * VecPacks;
#pragma unroll
    for (int item = 0; item < VecPacks; ++item) {
      Pack value = input[base + item];
      clear_pos_zero_pack<T>(value);
      store_pack_volatile<T>(peer_buffer, peer_write_base + base + item, value);
    }
  }

  for (int group = tid; group < num_groups; group += stride) {
    int base = group * VecPacks;
    Pack peer_values[VecPacks];
    while (true) {
      bool waiting = false;
#pragma unroll
      for (int item = 0; item < VecPacks; ++item) {
        peer_values[item] =
            load_pack_volatile<T>(local_buffer, local_poll_base + base + item);
        waiting |= has_pos_zero_pack<T>(peer_values[item]);
      }
      if (!waiting) {
        break;
      }
    }
#pragma unroll
    for (int item = 0; item < VecPacks; ++item) {
      out[base + item] = add_pack<T>(input[base + item], peer_values[item]);
      store_pack_volatile<T>(local_buffer, local_poll_base + base + item, reset);
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void tp2_remote_push_stream_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, int32_t* epoch_slots,
    T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int idx = tid; idx < num_packs; idx += stride) {
      uint4 local_value = input[idx];
      uint4 publish_value = clear_pos_zero_u4_16(local_value);
      store_u4_volatile(peer_buffer, peer_write_base + idx, publish_value);
      uint4 peer_value;
      while (true) {
        peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
        if (!has_pos_zero_u4_16(peer_value)) {
          break;
        }
      }
      reinterpret_cast<uint4*>(output)[idx] =
          packed_add_u4<T>(local_value, peer_value);
      store_u4_volatile(local_buffer, local_poll_base + idx, reset);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    Pack reset = zero_pack<T>();
    for (int idx = tid; idx < num_packs; idx += stride) {
      Pack local_value = input[idx];
      Pack publish_value = local_value;
      clear_pos_zero_pack<T>(publish_value);
      store_pack_volatile<T>(peer_buffer, peer_write_base + idx, publish_value);
      Pack peer_value;
      while (true) {
        peer_value = load_pack_volatile<T>(local_buffer, local_poll_base + idx);
        if (!has_pos_zero_pack<T>(peer_value)) {
          break;
        }
      }
      reinterpret_cast<Pack*>(output)[idx] =
          add_pack<T>(local_value, peer_value);
      store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T, bool Stream, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void tp2_remote_flag_push_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, uint64_t const* flag_ptrs,
    int32_t* epoch_slots, T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int seq = load_volatile_i32(epoch_slot) + 1;
  int epoch = seq & 1;
  int stage_offset = epoch * 2 * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    auto* peer_flags = reinterpret_cast<int32_t*>(flag_ptrs[peer]);
    auto* local_flags = reinterpret_cast<int32_t*>(flag_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    if constexpr (Stream) {
      for (int idx = tid; idx < num_packs; idx += stride) {
        uint4 local_value = input[idx];
        store_u4_volatile(peer_buffer, peer_write_base + idx, local_value);
        store_release_i32(peer_flags + peer_write_base + idx, seq);
        while (load_acquire_i32(local_flags + local_poll_base + idx) < seq) {
        }
        uint4 peer_value =
            load_u4_volatile(local_buffer, local_poll_base + idx);
        reinterpret_cast<uint4*>(output)[idx] =
            packed_add_u4<T>(local_value, peer_value);
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        store_u4_volatile(peer_buffer, peer_write_base + idx, input[idx]);
        store_release_i32(peer_flags + peer_write_base + idx, seq);
      }
      for (int idx = tid; idx < num_packs; idx += stride) {
        while (load_acquire_i32(local_flags + local_poll_base + idx) < seq) {
        }
        uint4 peer_value =
            load_u4_volatile(local_buffer, local_poll_base + idx);
        reinterpret_cast<uint4*>(output)[idx] =
            packed_add_u4<T>(input[idx], peer_value);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    auto* peer_flags = reinterpret_cast<int32_t*>(flag_ptrs[peer]);
    auto* local_flags = reinterpret_cast<int32_t*>(flag_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    if constexpr (Stream) {
      for (int idx = tid; idx < num_packs; idx += stride) {
        Pack local_value = input[idx];
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx,
                               local_value);
        store_release_i32(peer_flags + peer_write_base + idx, seq);
        while (load_acquire_i32(local_flags + local_poll_base + idx) < seq) {
        }
        Pack peer_value =
            load_pack_volatile<T>(local_buffer, local_poll_base + idx);
        reinterpret_cast<Pack*>(output)[idx] = add_pack<T>(local_value,
                                                           peer_value);
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx, input[idx]);
        store_release_i32(peer_flags + peer_write_base + idx, seq);
      }
      for (int idx = tid; idx < num_packs; idx += stride) {
        while (load_acquire_i32(local_flags + local_poll_base + idx) < seq) {
        }
        Pack peer_value =
            load_pack_volatile<T>(local_buffer, local_poll_base + idx);
        reinterpret_cast<Pack*>(output)[idx] =
            add_pack<T>(input[idx], peer_value);
      }
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, seq);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T, int WindowPacks, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void tp2_remote_push_window_kernel(
    T const* input_data, uint64_t const* tmp_ptrs, int32_t* epoch_slots,
    T* output, int num_packs, int rank) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = rank ^ 1;
  int32_t* epoch_slot = epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int window_stride = stride * WindowPacks;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(input_data);
    auto* peer_buffer = reinterpret_cast<uint4*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<uint4*>(tmp_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int base = tid; base < num_packs; base += window_stride) {
      uint4 local_values[WindowPacks];
      bool valid[WindowPacks];
#pragma unroll
      for (int item = 0; item < WindowPacks; ++item) {
        int idx = base + item * stride;
        valid[item] = idx < num_packs;
        if (valid[item]) {
          uint4 local_value = input[idx];
          local_values[item] = local_value;
          uint4 publish_value = clear_pos_zero_u4_16(local_value);
          store_u4_volatile(peer_buffer, peer_write_base + idx, publish_value);
        }
      }
#pragma unroll
      for (int item = 0; item < WindowPacks; ++item) {
        int idx = base + item * stride;
        if (valid[item]) {
          uint4 peer_value;
          while (true) {
            peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
            if (!has_pos_zero_u4_16(peer_value)) {
              break;
            }
          }
          reinterpret_cast<uint4*>(output)[idx] =
              packed_add_u4<T>(local_values[item], peer_value);
          store_u4_volatile(local_buffer, local_poll_base + idx, reset);
        }
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(input_data);
    auto* peer_buffer = reinterpret_cast<Pack*>(tmp_ptrs[peer]);
    auto* local_buffer = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
    int peer_write_base = stage_offset + rank * num_packs;
    int local_poll_base = stage_offset + peer * num_packs;
    Pack reset = zero_pack<T>();
    for (int base = tid; base < num_packs; base += window_stride) {
      Pack local_values[WindowPacks];
      bool valid[WindowPacks];
#pragma unroll
      for (int item = 0; item < WindowPacks; ++item) {
        int idx = base + item * stride;
        valid[item] = idx < num_packs;
        if (valid[item]) {
          Pack local_value = input[idx];
          local_values[item] = local_value;
          Pack publish_value = local_value;
          clear_pos_zero_pack<T>(publish_value);
          store_pack_volatile<T>(peer_buffer, peer_write_base + idx,
                                 publish_value);
        }
      }
#pragma unroll
      for (int item = 0; item < WindowPacks; ++item) {
        int idx = base + item * stride;
        if (valid[item]) {
          Pack peer_value;
          while (true) {
            peer_value =
                load_pack_volatile<T>(local_buffer, local_poll_base + idx);
            if (!has_pos_zero_pack<T>(peer_value)) {
              break;
            }
          }
          reinterpret_cast<Pack*>(output)[idx] =
              add_pack<T>(local_values[item], peer_value);
          store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T>
struct PushOneshotParamData {
  uint64_t tmp_ptrs[kMaxWorldSize];
  uint64_t signal_ptrs[kMaxWorldSize];
  T const* input;
  T* output;
  int32_t* epoch_slots;
  int num_packs;
  int rank_stride_packs;
  int epoch_stride_packs;
  int rank;
  int max_blocks;
};

template <typename T>
struct IpcTp2RemotePushData {
  uint64_t tmp_ptrs[2];
  T const* input;
  T* output;
  int32_t* epoch_slots;
  int num_packs;
  int rank;
};

template <typename T, bool Stream, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_tp2_remote_push_kernel(
    const IpcTp2RemotePushData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int peer = params.rank ^ 1;
  int32_t* epoch_slot = params.epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * 2 * params.num_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
    auto* local_buffer =
        reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    int peer_write_base = stage_offset + params.rank * params.num_packs;
    int local_poll_base = stage_offset + peer * params.num_packs;
    uint4 reset = {0u, 0u, 0u, 0u};
    if constexpr (Stream) {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 local_value = input[idx];
        uint4 publish_value = clear_pos_zero_u4_16(local_value);
        store_u4_volatile(peer_buffer, peer_write_base + idx, publish_value);
        uint4 peer_value;
        while (true) {
          peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_u4_16(peer_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] =
            packed_add_u4<T>(local_value, peer_value);
        store_u4_volatile(local_buffer, local_poll_base + idx, reset);
      }
    } else {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 value = clear_pos_zero_u4_16(input[idx]);
        store_u4_volatile(peer_buffer, peer_write_base + idx, value);
      }
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        uint4 peer_value;
        while (true) {
          peer_value = load_u4_volatile(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_u4_16(peer_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] =
            packed_add_u4<T>(input[idx], peer_value);
        store_u4_volatile(local_buffer, local_poll_base + idx, reset);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
    auto* local_buffer =
        reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    int peer_write_base = stage_offset + params.rank * params.num_packs;
    int local_poll_base = stage_offset + peer * params.num_packs;
    Pack reset = zero_pack<T>();
    if constexpr (Stream) {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack local_value = input[idx];
        Pack publish_value = local_value;
        clear_pos_zero_pack<T>(publish_value);
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx,
                               publish_value);
        Pack peer_value;
        while (true) {
          peer_value =
              load_pack_volatile<T>(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_pack<T>(peer_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] =
            add_pack<T>(local_value, peer_value);
        store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
      }
    } else {
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack value = input[idx];
        clear_pos_zero_pack<T>(value);
        store_pack_volatile<T>(peer_buffer, peer_write_base + idx, value);
      }
      for (int idx = tid; idx < params.num_packs; idx += stride) {
        Pack peer_value;
        while (true) {
          peer_value =
              load_pack_volatile<T>(local_buffer, local_poll_base + idx);
          if (!has_pos_zero_pack<T>(peer_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] =
            add_pack<T>(input[idx], peer_value);
        store_pack_volatile<T>(local_buffer, local_poll_base + idx, reset);
      }
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <int WorldSize>
__device__ __forceinline__ int rsag_owner_for_pack(int idx, int part) {
  int owner = part > 0 ? idx / part : 0;
  return owner < WorldSize ? owner : WorldSize - 1;
}

template <typename T, int WorldSize, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_rsag_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot = params.epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = params.num_packs / WorldSize;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    uint4 reset = {0u, 0u, 0u, 0u};

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      auto* owner_buffer =
          reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      uint4 value = clear_pos_zero_u4_16(input[idx]);
      store_u4_volatile(owner_buffer, offset, value);
    }

    int start = params.rank * part;
    int end = (params.rank == WorldSize - 1) ? params.num_packs : start + part;
    auto* local_buffer =
        reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    for (int idx = start + tid; idx < end; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      uint4 acc = values[0];
#pragma unroll
      for (int peer = 1; peer < WorldSize; ++peer) {
        acc = packed_add_u4<T>(acc, values[peer]);
      }
      reinterpret_cast<uint4*>(params.output)[idx] = acc;
      uint4 publish = clear_pos_zero_u4_16(acc);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int offset = stage_offset + peer * params.rank_stride_packs + idx;
        store_u4_volatile(local_buffer, offset, reset);
        if (peer != params.rank) {
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, publish);
        }
      }
    }

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      uint4 value;
      while (true) {
        value = load_u4_volatile(local_buffer, offset);
        if (!has_pos_zero_u4_16(value)) {
          break;
        }
      }
      reinterpret_cast<uint4*>(params.output)[idx] = value;
      store_u4_volatile(local_buffer, offset, reset);
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    Pack reset = zero_pack<T>();

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      auto* owner_buffer =
          reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
      store_pack_volatile<T>(owner_buffer, offset, value);
    }

    int start = params.rank * part;
    int end = (params.rank == WorldSize - 1) ? params.num_packs : start + part;
    auto* local_buffer =
        reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }
      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(params.output)[idx] = acc;
      Pack publish = acc;
      clear_pos_zero_pack<T>(publish);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int offset = stage_offset + peer * params.rank_stride_packs + idx;
        store_pack_volatile<T>(local_buffer, offset, reset);
        if (peer != params.rank) {
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, publish);
        }
      }
    }

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int owner = rsag_owner_for_pack<WorldSize>(idx, part);
      if (owner == params.rank) {
        continue;
      }
      int offset = stage_offset + owner * params.rank_stride_packs + idx;
      Pack value;
      while (true) {
        value = load_pack_volatile<T>(local_buffer, offset);
        if (!has_pos_zero_pack<T>(value)) {
          break;
        }
      }
      reinterpret_cast<Pack*>(params.output)[idx] = value;
      store_pack_volatile<T>(local_buffer, offset, reset);
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_topo_rsag8_push_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot = params.epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = params.num_packs / 4;
  int base = params.rank < 4 ? 0 : 4;
  int cross_base = base ^ 4;
  int local_rank = params.rank - base;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* local_buffer =
        reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    uint4 reset = {0u, 0u, 0u, 0u};

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int chunk = rsag_owner_for_pack<4>(idx, part);
      int owner = base + chunk;
      auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
      int input_offset =
          stage_offset + params.rank * params.rank_stride_packs + idx;
      uint4 local_value = input[idx];
      uint4 publish_input = clear_pos_zero_u4_16(local_value);
      store_u4_volatile(owner_buffer, input_offset, publish_input);

      if (local_rank == chunk) {
        uint4 values[4];
        while (true) {
          bool waiting = false;
#pragma unroll
          for (int peer_local = 0; peer_local < 4; ++peer_local) {
            int peer = base + peer_local;
            int offset = stage_offset + peer * params.rank_stride_packs + idx;
            values[peer_local] = load_u4_volatile(local_buffer, offset);
            waiting |= has_pos_zero_u4_16(values[peer_local]);
          }
          if (!waiting) {
            break;
          }
        }
        uint4 local_sum = values[0];
#pragma unroll
        for (int peer_local = 1; peer_local < 4; ++peer_local) {
          local_sum = packed_add_u4<T>(local_sum, values[peer_local]);
        }
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          store_u4_volatile(local_buffer, offset, reset);
        }

        int cross_owner = cross_base + chunk;
        auto* cross_buffer =
            reinterpret_cast<uint4*>(params.tmp_ptrs[cross_owner]);
        int cross_write =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        uint4 publish_sum = clear_pos_zero_u4_16(local_sum);
        store_u4_volatile(cross_buffer, cross_write, publish_sum);

        int cross_read =
            stage_offset + cross_owner * params.rank_stride_packs + idx;
        uint4 cross_sum;
        while (true) {
          cross_sum = load_u4_volatile(local_buffer, cross_read);
          if (!has_pos_zero_u4_16(cross_sum)) {
            break;
          }
        }
        uint4 final_value = packed_add_u4<T>(local_sum, cross_sum);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
        store_u4_volatile(local_buffer, cross_read, reset);

        uint4 publish_final = clear_pos_zero_u4_16(final_value);
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, publish_final);
        }
      } else {
        int final_offset =
            stage_offset + owner * params.rank_stride_packs + idx;
        uint4 final_value;
        while (true) {
          final_value = load_u4_volatile(local_buffer, final_offset);
          if (!has_pos_zero_u4_16(final_value)) {
            break;
          }
        }
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
        store_u4_volatile(local_buffer, final_offset, reset);
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* local_buffer =
        reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    Pack reset = zero_pack<T>();

    for (int idx = tid; idx < params.num_packs; idx += stride) {
      int chunk = rsag_owner_for_pack<4>(idx, part);
      int owner = base + chunk;
      auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
      int input_offset =
          stage_offset + params.rank * params.rank_stride_packs + idx;
      Pack local_value = input[idx];
      Pack publish_input = local_value;
      clear_pos_zero_pack<T>(publish_input);
      store_pack_volatile<T>(owner_buffer, input_offset, publish_input);

      if (local_rank == chunk) {
        Pack values[4];
        while (true) {
          bool waiting = false;
#pragma unroll
          for (int peer_local = 0; peer_local < 4; ++peer_local) {
            int peer = base + peer_local;
            int offset = stage_offset + peer * params.rank_stride_packs + idx;
            values[peer_local] = load_pack_volatile<T>(local_buffer, offset);
            waiting |= has_pos_zero_pack<T>(values[peer_local]);
          }
          if (!waiting) {
            break;
          }
        }
        Pack local_sum = values[0];
#pragma unroll
        for (int peer_local = 1; peer_local < 4; ++peer_local) {
          local_sum = add_pack<T>(local_sum, values[peer_local]);
        }
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          int offset = stage_offset + peer * params.rank_stride_packs + idx;
          store_pack_volatile<T>(local_buffer, offset, reset);
        }

        int cross_owner = cross_base + chunk;
        auto* cross_buffer =
            reinterpret_cast<Pack*>(params.tmp_ptrs[cross_owner]);
        int cross_write =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        Pack publish_sum = local_sum;
        clear_pos_zero_pack<T>(publish_sum);
        store_pack_volatile<T>(cross_buffer, cross_write, publish_sum);

        int cross_read =
            stage_offset + cross_owner * params.rank_stride_packs + idx;
        Pack cross_sum;
        while (true) {
          cross_sum = load_pack_volatile<T>(local_buffer, cross_read);
          if (!has_pos_zero_pack<T>(cross_sum)) {
            break;
          }
        }
        Pack final_value = add_pack<T>(local_sum, cross_sum);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
        store_pack_volatile<T>(local_buffer, cross_read, reset);

        Pack publish_final = final_value;
        clear_pos_zero_pack<T>(publish_final);
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, publish_final);
        }
      } else {
        int final_offset =
            stage_offset + owner * params.rank_stride_packs + idx;
        Pack final_value;
        while (true) {
          final_value = load_pack_volatile<T>(local_buffer, final_offset);
          if (!has_pos_zero_pack<T>(final_value)) {
            break;
          }
        }
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
        store_pack_volatile<T>(local_buffer, final_offset, reset);
      }
    }
  }

  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
  pdl_grid_release_const<UsePdl>();
}

template <typename T, bool UsePdl>
__global__ __launch_bounds__(1024, 1) void ipc_topo_rsag8_block_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* self_signal =
      reinterpret_cast<int32_t*>(params.signal_ptrs[params.rank]);
  int flag =
      load_acquire_i32(self_signal +
                       flag_offset(blockIdx.x, params.max_blocks, 8)) +
      1;

  int chunk = blockIdx.x & 3;
  int chunk_block = blockIdx.x >> 2;
  int blocks_per_chunk = gridDim.x >> 2;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;
  int part = params.num_packs >> 2;
  int start = chunk * part;
  int end = (chunk == 3) ? params.num_packs : start + part;
  int base = params.rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;
  int stage_offset = 0;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    auto* owner_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      store_u4_volatile(owner_buffer, offset, input[idx]);
    }
    __threadfence_system();
    island_owner_gather(params.signal_ptrs, params.rank, base, owner,
                        params.max_blocks, 0, flag);

    auto* local_buffer =
        reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 v0 = load_u4_volatile(
            local_buffer, stage_offset + (base + 0) *
                                      params.rank_stride_packs + idx);
        uint4 v1 = load_u4_volatile(
            local_buffer, stage_offset + (base + 1) *
                                      params.rank_stride_packs + idx);
        uint4 v2 = load_u4_volatile(
            local_buffer, stage_offset + (base + 2) *
                                      params.rank_stride_packs + idx);
        uint4 v3 = load_u4_volatile(
            local_buffer, stage_offset + (base + 3) *
                                      params.rank_stride_packs + idx);
        uint4 local_sum = packed_add_u4<T>(packed_add_u4<T>(v0, v1),
                                           packed_add_u4<T>(v2, v3));
        auto* cross_buffer =
            reinterpret_cast<uint4*>(params.tmp_ptrs[cross_owner]);
        int cross_write =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        store_u4_volatile(cross_buffer, cross_write, local_sum);
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    owner_pair_barrier(params.signal_ptrs, params.rank, owner, cross_owner,
                       params.max_blocks, 1, flag);

    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 v0 = load_u4_volatile(
            local_buffer, stage_offset + (base + 0) *
                                      params.rank_stride_packs + idx);
        uint4 v1 = load_u4_volatile(
            local_buffer, stage_offset + (base + 1) *
                                      params.rank_stride_packs + idx);
        uint4 v2 = load_u4_volatile(
            local_buffer, stage_offset + (base + 2) *
                                      params.rank_stride_packs + idx);
        uint4 v3 = load_u4_volatile(
            local_buffer, stage_offset + (base + 3) *
                                      params.rank_stride_packs + idx);
        uint4 local_sum = packed_add_u4<T>(packed_add_u4<T>(v0, v1),
                                           packed_add_u4<T>(v2, v3));
        uint4 cross_sum = load_u4_volatile(
            local_buffer,
            stage_offset + cross_owner * params.rank_stride_packs + idx);
        uint4 final_value = packed_add_u4<T>(local_sum, cross_sum);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_u4_volatile(peer_buffer, final_offset, final_value);
        }
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    island_owner_ready(params.signal_ptrs, params.rank, base, owner,
                       params.max_blocks, 2, flag);

    if (params.rank != owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        uint4 final_value = load_u4_volatile(
            local_buffer,
            stage_offset + owner * params.rank_stride_packs + idx);
        reinterpret_cast<uint4*>(params.output)[idx] = final_value;
      }
    }
    pdl_grid_release_const<UsePdl>();
    island_owner_ack(params.signal_ptrs, params.rank, base, owner,
                     params.max_blocks, 3, flag);
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    auto* owner_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      int offset = stage_offset + params.rank * params.rank_stride_packs + idx;
      store_pack_volatile<T>(owner_buffer, offset, input[idx]);
    }
    __threadfence_system();
    island_owner_gather(params.signal_ptrs, params.rank, base, owner,
                        params.max_blocks, 0, flag);

    auto* local_buffer =
        reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack v0 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 0) *
                                      params.rank_stride_packs + idx);
        Pack v1 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 1) *
                                      params.rank_stride_packs + idx);
        Pack v2 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 2) *
                                      params.rank_stride_packs + idx);
        Pack v3 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 3) *
                                      params.rank_stride_packs + idx);
        Pack local_sum = add_pack<T>(add_pack<T>(v0, v1), add_pack<T>(v2, v3));
        auto* cross_buffer =
            reinterpret_cast<Pack*>(params.tmp_ptrs[cross_owner]);
        int cross_write =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        store_pack_volatile<T>(cross_buffer, cross_write, local_sum);
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    owner_pair_barrier(params.signal_ptrs, params.rank, owner, cross_owner,
                       params.max_blocks, 1, flag);

    if (params.rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack v0 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 0) *
                                      params.rank_stride_packs + idx);
        Pack v1 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 1) *
                                      params.rank_stride_packs + idx);
        Pack v2 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 2) *
                                      params.rank_stride_packs + idx);
        Pack v3 = load_pack_volatile<T>(
            local_buffer, stage_offset + (base + 3) *
                                      params.rank_stride_packs + idx);
        Pack local_sum = add_pack<T>(add_pack<T>(v0, v1), add_pack<T>(v2, v3));
        Pack cross_sum = load_pack_volatile<T>(
            local_buffer,
            stage_offset + cross_owner * params.rank_stride_packs + idx);
        Pack final_value = add_pack<T>(local_sum, cross_sum);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
#pragma unroll
        for (int peer_local = 0; peer_local < 4; ++peer_local) {
          int peer = base + peer_local;
          if (peer == params.rank) {
            continue;
          }
          auto* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
          int final_offset =
              stage_offset + params.rank * params.rank_stride_packs + idx;
          store_pack_volatile<T>(peer_buffer, final_offset, final_value);
        }
      }
    }
    if (params.rank == owner) {
      __threadfence_system();
    }
    island_owner_ready(params.signal_ptrs, params.rank, base, owner,
                       params.max_blocks, 2, flag);

    if (params.rank != owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack final_value = load_pack_volatile<T>(
            local_buffer,
            stage_offset + owner * params.rank_stride_packs + idx);
        reinterpret_cast<Pack*>(params.output)[idx] = final_value;
      }
    }
    pdl_grid_release_const<UsePdl>();
    island_owner_ack(params.signal_ptrs, params.rank, base, owner,
                     params.max_blocks, 3, flag);
  }

  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, params.max_blocks,
                                                8),
                      flag);
  }
}

template <typename T, int WorldSize, bool UsePdl, bool Fp32Reduce>
__global__ __launch_bounds__(1024, 1) void push_oneshot_param_kernel(
    const PushOneshotParamData<T> __grid_constant__ params) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync_const<UsePdl>();

  int32_t* epoch_slot = params.epoch_slots + blockIdx.x;
  int epoch = load_volatile_i32(epoch_slot) & 1;
  int stage_offset = epoch * params.epoch_stride_packs;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    uint4 const* input = reinterpret_cast<uint4 const*>(params.input);
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      uint4 value = clear_pos_zero_u4_16(input[idx]);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        auto* peer_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[peer]);
        int peer_offset =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        store_u4_volatile(peer_buffer, peer_offset, value);
      }
    }

    auto* local_buffer = reinterpret_cast<uint4*>(params.tmp_ptrs[params.rank]);
    uint4 reset = {0u, 0u, 0u, 0u};
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      uint4 values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset =
              stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_u4_volatile(local_buffer, peer_offset);
          waiting |= has_pos_zero_u4_16(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      uint4 acc;
      if constexpr (Fp32Reduce) {
        acc = reduce_u4_fp32<T, WorldSize>(values);
      } else {
        acc = values[0];
#pragma unroll
        for (int peer = 1; peer < WorldSize; ++peer) {
          acc = packed_add_u4<T>(acc, values[peer]);
        }
      }
      reinterpret_cast<uint4*>(params.output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset =
            stage_offset + peer * params.rank_stride_packs + idx;
        local_buffer[peer_offset] = reset;
      }
    }
  } else {
    Pack const* input = reinterpret_cast<Pack const*>(params.input);
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      Pack value = input[idx];
      clear_pos_zero_pack<T>(value);
#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        Pack* peer_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[peer]);
        int peer_offset =
            stage_offset + params.rank * params.rank_stride_packs + idx;
        store_pack_volatile<T>(peer_buffer, peer_offset, value);
      }
    }

    Pack* local_buffer = reinterpret_cast<Pack*>(params.tmp_ptrs[params.rank]);
    Pack reset = zero_pack<T>();
    for (int idx = tid; idx < params.num_packs; idx += stride) {
      Pack values[WorldSize];
      while (true) {
        bool waiting = false;
#pragma unroll
        for (int peer = 0; peer < WorldSize; ++peer) {
          int peer_offset =
              stage_offset + peer * params.rank_stride_packs + idx;
          values[peer] = load_pack_volatile<T>(local_buffer, peer_offset);
          waiting |= has_pos_zero_pack<T>(values[peer]);
        }
        if (!waiting) {
          break;
        }
      }

      Pack acc = reduce_loaded_packs<T, WorldSize>(values);
      reinterpret_cast<Pack*>(params.output)[idx] = acc;

#pragma unroll
      for (int peer = 0; peer < WorldSize; ++peer) {
        int peer_offset =
            stage_offset + peer * params.rank_stride_packs + idx;
        local_buffer[peer_offset] = reset;
      }
    }
  }

  pdl_grid_release_const<UsePdl>();
  if (threadIdx.x == 0) {
    store_volatile_i32(epoch_slot, epoch ^ 1);
  }
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void fast_oneshot_kernel(
    uint64_t const* input_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);

  block_barrier_fast_start<WorldSize>(signal_ptrs, rank, max_blocks);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
    auto* out = reinterpret_cast<uint4*>(output);
    if (stride >= num_packs) {
      if (tid < num_packs) {
        auto const* first = reinterpret_cast<uint4 const*>(input_ptrs[0]);
        uint4 acc = first[tid];
#pragma unroll
        for (int peer = 1; peer < WorldSize; ++peer) {
          auto const* peer_data =
              reinterpret_cast<uint4 const*>(input_ptrs[peer]);
          acc = packed_add_u4<T>(acc, peer_data[tid]);
        }
        out[tid] = acc;
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        auto const* first = reinterpret_cast<uint4 const*>(input_ptrs[0]);
        uint4 acc = first[idx];
#pragma unroll
        for (int peer = 1; peer < WorldSize; ++peer) {
          auto const* peer_data =
              reinterpret_cast<uint4 const*>(input_ptrs[peer]);
          acc = packed_add_u4<T>(acc, peer_data[idx]);
        }
        out[idx] = acc;
      }
    }
  } else {
    auto* out = reinterpret_cast<Pack*>(output);
    if (stride >= num_packs) {
      if (tid < num_packs) {
        out[tid] = reduce_pack<T, WorldSize>(input_ptrs, tid);
      }
    } else {
      for (int idx = tid; idx < num_packs; idx += stride) {
        out[idx] = reduce_pack<T, WorldSize>(input_ptrs, idx);
      }
    }
  }

  block_barrier_fast_final<WorldSize>(signal_ptrs, rank, max_blocks, 2);
  pdl_grid_release(pdl_release);
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void twoshot_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       WorldSize)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = num_packs / WorldSize;
  int start = rank * part;
  int end = (rank == WorldSize - 1) ? num_packs : start + part;
  int largest_part = part + (num_packs % WorldSize);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 0, flag);

  for (int idx = start + tid; idx < end; idx += stride) {
    local_tmp[idx - start] = reduce_pack<T, WorldSize>(input_ptrs, idx);
  }

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 1, flag);

  auto* out_pack = reinterpret_cast<Pack*>(output);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int peer = 0; peer < WorldSize; ++peer) {
      if (peer == WorldSize - 1 || idx < part) {
        auto const* peer_tmp = reinterpret_cast<Pack const*>(tmp_ptrs[peer]);
        out_pack[peer * part + idx] = peer_tmp[idx];
      }
    }
  }

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 2, flag);
  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                WorldSize),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void twostage_fast_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int part = num_packs / WorldSize;
  int start = rank * part;
  int end = (rank == WorldSize - 1) ? num_packs : start + part;
  int largest_part = part + (num_packs % WorldSize);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  block_barrier_fast_start<WorldSize>(signal_ptrs, rank, max_blocks);

  for (int idx = start + tid; idx < end; idx += stride) {
    local_tmp[idx - start] =
        reduce_pack_rotated<T, WorldSize>(input_ptrs, idx, rank);
  }

  block_barrier_fast_mid<WorldSize>(signal_ptrs, rank, max_blocks);

  auto* out_pack = reinterpret_cast<Pack*>(output);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < WorldSize; ++i) {
      int gather_from_rank = (rank + i) % WorldSize;
      if (gather_from_rank == WorldSize - 1 || idx < part) {
        auto const* peer_tmp =
            reinterpret_cast<Pack const*>(tmp_ptrs[gather_from_rank]);
        out_pack[gather_from_rank * part + idx] = peer_tmp[idx];
      }
    }
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void tree_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       WorldSize)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 0, flag);

  int read_buffer = 0;
  int write_buffer = 0;
  int phase = 1;
#pragma unroll
  for (int mask = 1; mask < WorldSize; mask <<= 1) {
    int peer = rank ^ mask;
    bool first_step = mask == 1;
    bool last_step = (mask << 1) >= WorldSize;
    auto const* local_src = first_step
                                ? reinterpret_cast<Pack const*>(input_ptrs[rank])
                                : local_tmp + read_buffer * num_packs;
    auto const* peer_src =
        first_step ? reinterpret_cast<Pack const*>(input_ptrs[peer])
                   : reinterpret_cast<Pack const*>(tmp_ptrs[peer]) +
                         read_buffer * num_packs;
    auto* dst = last_step ? out_pack : local_tmp + write_buffer * num_packs;

    for (int idx = tid; idx < num_packs; idx += stride) {
      dst[idx] = add_pack<T>(local_src[idx], peer_src[idx]);
    }

    block_barrier(signal_ptrs, rank, WorldSize, max_blocks, phase, flag);
    read_buffer = write_buffer;
    write_buffer ^= 1;
    ++phase;
  }

  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                WorldSize),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void rsag_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       WorldSize)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 0, flag);

  int range_start = 0;
  int range_len = num_packs;
  int read_buffer = 0;
  int write_buffer = 0;
  int phase = 1;

#pragma unroll
  for (int mask = 1; mask < WorldSize; mask <<= 1) {
    int peer = rank ^ mask;
    int half = range_len / 2;
    bool keep_upper = (rank & mask) != 0;
    int keep_start = range_start + (keep_upper ? half : 0);
    auto const* local_src = (mask == 1)
                                ? reinterpret_cast<Pack const*>(input_ptrs[rank])
                                : local_tmp + read_buffer * num_packs;
    auto const* peer_src =
        (mask == 1) ? reinterpret_cast<Pack const*>(input_ptrs[peer])
                    : reinterpret_cast<Pack const*>(tmp_ptrs[peer]) +
                          read_buffer * num_packs;
    auto* dst = local_tmp + write_buffer * num_packs;

    for (int pos = keep_start + tid; pos < keep_start + half; pos += stride) {
      dst[pos] = add_pack<T>(local_src[pos], peer_src[pos]);
    }

    range_start = keep_start;
    range_len = half;
    if constexpr (WorldSize == 8) {
      uint32_t sync_mask = (mask == 1) ? (rank < 4 ? 0x0fu : 0xf0u) : 0xffu;
      block_barrier_mask(signal_ptrs, rank, WorldSize, max_blocks, phase, flag,
                         sync_mask);
    } else {
      block_barrier(signal_ptrs, rank, WorldSize, max_blocks, phase, flag);
    }
    read_buffer = write_buffer;
    write_buffer ^= 1;
    ++phase;
  }

  for (int mask = WorldSize >> 1; mask >= 1; mask >>= 1) {
    int peer = rank ^ mask;
    bool have_upper = (rank & mask) != 0;
    int peer_start = have_upper ? range_start - range_len
                                : range_start + range_len;
    int union_start = have_upper ? peer_start : range_start;
    bool last_step = mask == 1;
    auto const* local_src = local_tmp + read_buffer * num_packs;
    auto const* peer_src =
        reinterpret_cast<Pack const*>(tmp_ptrs[peer]) + read_buffer * num_packs;
    auto* dst = last_step ? out_pack : local_tmp + write_buffer * num_packs;

    for (int pos = range_start + tid; pos < range_start + range_len;
         pos += stride) {
      dst[pos] = local_src[pos];
    }
    for (int pos = peer_start + tid; pos < peer_start + range_len;
         pos += stride) {
      dst[pos] = peer_src[pos];
    }

    range_start = union_start;
    range_len *= 2;
    if constexpr (WorldSize == 8) {
      uint32_t sync_mask =
          (mask == 4) ? (rank < 4 ? 0x0fu : 0xf0u)
                      : ((1u << rank) | (1u << (rank ^ 1)));
      block_barrier_mask(signal_ptrs, rank, WorldSize, max_blocks, phase, flag,
                         sync_mask);
    } else {
      block_barrier(signal_ptrs, rank, WorldSize, max_blocks, phase, flag);
    }
    read_buffer = write_buffer;
    write_buffer ^= 1;
    ++phase;
  }

  if constexpr (WorldSize == 8) {
    block_barrier(signal_ptrs, rank, WorldSize, max_blocks, phase, flag);
  }
  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                WorldSize),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, bool FastFinal>
__global__ __launch_bounds__(512, 1) void topo_tree8_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  int peer1 = rank ^ 1;
  int peer2 = rank ^ 2;
  int peer4 = rank ^ 4;
  uint32_t node_pair_mask = (1u << rank) | (1u << peer2);
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t cross_pair_mask = (1u << rank) | (1u << peer4);

  auto const* input_self = reinterpret_cast<Pack const*>(input_ptrs[rank]);
  auto const* input_peer1 = reinterpret_cast<Pack const*>(input_ptrs[peer1]);
  for (int idx = tid; idx < num_packs; idx += stride) {
    local_tmp[idx] = add_pack<T>(input_self[idx], input_peer1[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag,
                     node_pair_mask);

  auto const* tmp_peer2 = reinterpret_cast<Pack const*>(tmp_ptrs[peer2]);
  auto* local_tmp1 = local_tmp + num_packs;
  for (int idx = tid; idx < num_packs; idx += stride) {
    local_tmp1[idx] = add_pack<T>(local_tmp[idx], tmp_peer2[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag, island_mask);
  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag,
                     cross_pair_mask);

  auto const* tmp_peer4 = reinterpret_cast<Pack const*>(tmp_ptrs[peer4]) +
                          num_packs;
  for (int idx = tid; idx < num_packs; idx += stride) {
    out_pack[idx] = add_pack<T>(local_tmp1[idx], tmp_peer4[idx]);
  }

  if constexpr (FastFinal) {
    block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                  cross_pair_mask);
  } else {
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 3, flag,
                       cross_pair_mask);
  }
  if (threadIdx.x == 0) {
    if constexpr (FastFinal) {
      store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                         flag);
    } else {
      store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                        flag);
    }
  }
  pdl_grid_release(pdl_release);
}

template <typename T, bool FastFinal>
__global__ __launch_bounds__(512, 1) void topo_rsag8_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int local_rank = rank - base;
  int cross = rank ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t cross_pair_mask = (1u << rank) | (1u << cross);

  int part = num_packs / 4;
  int start = local_rank * part;
  int end = (local_rank == 3) ? num_packs : start + part;

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag, island_mask);

  Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
  Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
  Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
  Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
  for (int idx = start + tid; idx < end; idx += stride) {
    Pack v = add_pack<T>(d0[idx], d1[idx]);
    v = add_pack<T>(v, d2[idx]);
    v = add_pack<T>(v, d3[idx]);
    local_tmp0[idx] = v;
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     cross_pair_mask);

  auto const* cross_tmp0 = reinterpret_cast<Pack const*>(tmp_ptrs[cross]);
  for (int idx = start + tid; idx < end; idx += stride) {
    local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag, island_mask);

#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    int cs = chunk * part;
    int ce = (chunk == 3) ? num_packs : cs + part;
    auto const* src =
        (chunk == local_rank)
            ? local_tmp1
            : reinterpret_cast<Pack const*>(tmp_ptrs[base + chunk]) +
                  num_packs;
    for (int idx = cs + tid; idx < ce; idx += stride) {
      out_pack[idx] = src[idx];
    }
  }

  if constexpr (FastFinal) {
    block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                  island_mask);
    block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 4, flag,
                                  cross_pair_mask);
  } else {
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 3, flag, island_mask);
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 4, flag,
                       cross_pair_mask);
  }
  if (threadIdx.x == 0) {
    if constexpr (FastFinal) {
      store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                         flag);
    } else {
      store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                        flag);
    }
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int Chunks, bool OwnerGatherStart>
__global__ __launch_bounds__(512, 1) void topo_rsag8_pipelined_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;

  // Pipeline unit: one CTA stream owns one island chunk. Chunks=4 maps one
  // chunk to each island rank; Chunks=8 gives each rank two smaller chunks.
  int chunk = blockIdx.x % Chunks;
  int chunk_block = blockIdx.x / Chunks;
  int blocks_per_chunk = gridDim.x / Chunks;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;

  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int owner_local = chunk & 3;
  int owner = base + owner_local;
  int cross_owner = owner ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t owner_pair_mask = (1u << owner_local) | (1u << (owner_local + 4));

  int part = num_packs / Chunks;
  int start = chunk * part;
  int end = (chunk == Chunks - 1) ? num_packs : start + part;

  if constexpr (OwnerGatherStart) {
    island_owner_gather(signal_ptrs, rank, base, owner, max_blocks, 0, flag);
  } else {
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag,
                       island_mask);
  }

  if (rank == owner) {
    Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
    Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
    Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
    Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Pack v = add_pack<T>(d0[idx], d1[idx]);
      v = add_pack<T>(v, d2[idx]);
      v = add_pack<T>(v, d3[idx]);
      local_tmp0[idx] = v;
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     owner_pair_mask);

  if (rank == owner) {
    auto const* cross_tmp0 =
        reinterpret_cast<Pack const*>(tmp_ptrs[cross_owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag, island_mask);

  auto const* owner_tmp1 = reinterpret_cast<Pack const*>(tmp_ptrs[owner]) +
                           num_packs;
  for (int idx = start + tid; idx < end; idx += stride) {
    out_pack[idx] = owner_tmp1[idx];
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                island_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int VecPacks>
__global__ __launch_bounds__(512, 1) void topo_rsag8_pipelined_wide_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Group = PackGroup<T, VecPacks>;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;

  int chunk = blockIdx.x & 3;
  int chunk_block = blockIdx.x >> 2;
  int blocks_per_chunk = gridDim.x >> 2;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;
  int num_groups = num_packs / VecPacks;

  auto* out_group = reinterpret_cast<Group*>(output);
  auto* local_tmp0 = reinterpret_cast<Group*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_groups;

  int base = rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t owner_pair_mask = (1u << chunk) | (1u << (chunk + 4));

  int part = num_groups / 4;
  int start = chunk * part;
  int end = (chunk == 3) ? num_groups : start + part;

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag, island_mask);

  if (rank == owner) {
    Group const* d0 = reinterpret_cast<Group const*>(input_ptrs[base + 0]);
    Group const* d1 = reinterpret_cast<Group const*>(input_ptrs[base + 1]);
    Group const* d2 = reinterpret_cast<Group const*>(input_ptrs[base + 2]);
    Group const* d3 = reinterpret_cast<Group const*>(input_ptrs[base + 3]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Group v = add_pack_group<T, VecPacks>(d0[idx], d1[idx]);
      v = add_pack_group<T, VecPacks>(v, d2[idx]);
      v = add_pack_group<T, VecPacks>(v, d3[idx]);
      local_tmp0[idx] = v;
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     owner_pair_mask);

  if (rank == owner) {
    auto const* cross_tmp0 =
        reinterpret_cast<Group const*>(tmp_ptrs[cross_owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      local_tmp1[idx] = add_pack_group<T, VecPacks>(local_tmp0[idx],
                                                   cross_tmp0[idx]);
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag, island_mask);

  auto const* owner_tmp1 = reinterpret_cast<Group const*>(tmp_ptrs[owner]) +
                           num_groups;
  for (int idx = start + tid; idx < end; idx += stride) {
    out_group[idx] = owner_tmp1[idx];
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                island_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T>
__global__ __launch_bounds__(512, 1) void topo_rsag8_pipelined_ag_push_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* output_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;

  int chunk = blockIdx.x & 3;
  int chunk_block = blockIdx.x >> 2;
  int blocks_per_chunk = gridDim.x >> 2;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;

  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t owner_pair_mask = (1u << chunk) | (1u << (chunk + 4));

  int part = num_packs / 4;
  int start = chunk * part;
  int end = (chunk == 3) ? num_packs : start + part;

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag, island_mask);

  if (rank == owner) {
    Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
    Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
    Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
    Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Pack v = add_pack<T>(d0[idx], d1[idx]);
      v = add_pack<T>(v, d2[idx]);
      v = add_pack<T>(v, d3[idx]);
      local_tmp0[idx] = v;
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     owner_pair_mask);

  if (rank == owner) {
    auto const* cross_tmp0 =
        reinterpret_cast<Pack const*>(tmp_ptrs[cross_owner]);
    for (int idx = start + tid; idx < end; idx += stride) {
      local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
    }

#pragma unroll
    for (int peer_local = 0; peer_local < 4; ++peer_local) {
      auto* peer_output =
          reinterpret_cast<Pack*>(output_ptrs[base + peer_local]);
      for (int idx = start + tid; idx < end; idx += stride) {
        peer_output[idx] = local_tmp1[idx];
      }
    }
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 2, flag,
                                island_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  (void)output;
  pdl_grid_release(pdl_release);
}

template <typename T>
__global__ __launch_bounds__(512, 1)
void topo_rsag8_pipelined_ag_push_split_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* output_ptrs, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;

  int unit = blockIdx.x & 15;
  int chunk = unit >> 2;
  int peer_local = unit & 3;
  int lane0_block = blockIdx.x & ~3;
  int chunk_block = blockIdx.x >> 4;
  int blocks_per_lane = gridDim.x >> 4;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_lane * blockDim.x;

  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t owner_pair_mask = (1u << chunk) | (1u << (chunk + 4));

  int part = num_packs / 4;
  int start = chunk * part;
  int end = (chunk == 3) ? num_packs : start + part;

  if (peer_local == 0) {
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag,
                       island_mask);

    if (rank == owner) {
      Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
      Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
      Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
      Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
      for (int idx = start + tid; idx < end; idx += stride) {
        Pack v = add_pack<T>(d0[idx], d1[idx]);
        v = add_pack<T>(v, d2[idx]);
        v = add_pack<T>(v, d3[idx]);
        local_tmp0[idx] = v;
      }
    }

    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                       owner_pair_mask);

    if (rank == owner) {
      auto const* cross_tmp0 =
          reinterpret_cast<Pack const*>(tmp_ptrs[cross_owner]);
      for (int idx = start + tid; idx < end; idx += stride) {
        local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
      }
    }

    __syncthreads();
    if (rank == owner && threadIdx.x == 0) {
      store_release_i32(
          self_signal + phase_offset(2, lane0_block, owner, max_blocks, 8),
          flag);
    }
  } else {
    if (rank == owner && threadIdx.x == 0) {
      int32_t* ready_slot =
          self_signal + phase_offset(2, lane0_block, owner, max_blocks, 8);
      while (load_acquire_i32(ready_slot) < flag) {
      }
    }
    __syncthreads();
  }

  if (rank == owner) {
    auto* peer_output = reinterpret_cast<Pack*>(output_ptrs[base + peer_local]);
    for (int idx = start + tid; idx < end; idx += stride) {
      peer_output[idx] = local_tmp1[idx];
    }
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                island_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  (void)output;
  pdl_grid_release(pdl_release);
}

template <typename T, bool CrossPush>
__global__ __launch_bounds__(512, 1) void topo_rsag8_pipelined_direct_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;

  int chunk = blockIdx.x & 3;
  int chunk_block = blockIdx.x >> 2;
  int blocks_per_chunk = gridDim.x >> 2;
  int tid = chunk_block * blockDim.x + threadIdx.x;
  int stride = blocks_per_chunk * blockDim.x;

  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int owner = base + chunk;
  int cross_owner = owner ^ 4;

  int part = num_packs / 4;
  int start = chunk * part;
  int end = (chunk == 3) ? num_packs : start + part;

  island_owner_gather(signal_ptrs, rank, base, owner, max_blocks, 0, flag);

  if (rank == owner) {
    Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
    Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
    Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
    Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
    for (int idx = start + tid; idx < end; idx += stride) {
      Pack v = add_pack<T>(d0[idx], d1[idx]);
      v = add_pack<T>(v, d2[idx]);
      v = add_pack<T>(v, d3[idx]);
      local_tmp0[idx] = v;
    }
  }

  if constexpr (CrossPush) {
    if (rank == owner) {
      auto* cross_tmp1 =
          reinterpret_cast<Pack*>(tmp_ptrs[cross_owner]) + num_packs;
      for (int idx = start + tid; idx < end; idx += stride) {
        cross_tmp1[idx] = local_tmp0[idx];
      }
    }

    owner_pair_barrier(signal_ptrs, rank, owner, cross_owner, max_blocks, 1,
                       flag);

    if (rank == owner) {
      for (int idx = start + tid; idx < end; idx += stride) {
        local_tmp1[idx] = add_pack<T>(local_tmp0[idx], local_tmp1[idx]);
      }
    }
  } else {
    owner_pair_barrier(signal_ptrs, rank, owner, cross_owner, max_blocks, 1,
                       flag);

    if (rank == owner) {
      auto const* cross_tmp0 =
          reinterpret_cast<Pack const*>(tmp_ptrs[cross_owner]);
      for (int idx = start + tid; idx < end; idx += stride) {
        local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
      }
    }
  }

  island_owner_ready(signal_ptrs, rank, base, owner, max_blocks, 2, flag);

  auto const* owner_tmp1 = reinterpret_cast<Pack const*>(tmp_ptrs[owner]) +
                           num_packs;
  for (int idx = start + tid; idx < end; idx += stride) {
    out_pack[idx] = owner_tmp1[idx];
  }

  island_owner_ack(signal_ptrs, rank, base, owner, max_blocks, 3, flag);

  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T>
__global__ __launch_bounds__(512, 1) void topo_rsag8_push_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp0 = reinterpret_cast<Pack*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_packs;

  int base = rank < 4 ? 0 : 4;
  int local_rank = rank - base;
  int cross = rank ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t cross_pair_mask = (1u << rank) | (1u << cross);

  int part = num_packs / 4;
  int start = local_rank * part;
  int end = (local_rank == 3) ? num_packs : start + part;

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag, island_mask);

  Pack const* d0 = reinterpret_cast<Pack const*>(input_ptrs[base + 0]);
  Pack const* d1 = reinterpret_cast<Pack const*>(input_ptrs[base + 1]);
  Pack const* d2 = reinterpret_cast<Pack const*>(input_ptrs[base + 2]);
  Pack const* d3 = reinterpret_cast<Pack const*>(input_ptrs[base + 3]);
  for (int idx = start + tid; idx < end; idx += stride) {
    Pack v = add_pack<T>(d0[idx], d1[idx]);
    v = add_pack<T>(v, d2[idx]);
    v = add_pack<T>(v, d3[idx]);
    local_tmp0[idx] = v;
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     cross_pair_mask);

  auto const* cross_tmp0 = reinterpret_cast<Pack const*>(tmp_ptrs[cross]);
  for (int idx = start + tid; idx < end; idx += stride) {
    local_tmp1[idx] = add_pack<T>(local_tmp0[idx], cross_tmp0[idx]);
  }

  // Hybrid push/pull all-gather: each rank pushes its reduced shard to the
  // other three ranks in the same island, then every rank copies local tmp1.
  for (int peer_local = 0; peer_local < 4; ++peer_local) {
    int peer = base + peer_local;
    if (peer == rank) {
      continue;
    }
    auto* peer_tmp1 = reinterpret_cast<Pack*>(tmp_ptrs[peer]) + num_packs;
    for (int idx = start + tid; idx < end; idx += stride) {
      peer_tmp1[idx] = local_tmp1[idx];
    }
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag, island_mask);

  for (int idx = tid; idx < num_packs; idx += stride) {
    out_pack[idx] = local_tmp1[idx];
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                island_mask);
  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 4, flag,
                                cross_pair_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T, int VecPacks>
__global__ __launch_bounds__(512, 1) void topo_rsag8_wide_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Group = PackGroup<T, VecPacks>;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int num_groups = num_packs / VecPacks;
  auto* out_group = reinterpret_cast<Group*>(output);
  auto* local_tmp0 = reinterpret_cast<Group*>(tmp_ptrs[rank]);
  auto* local_tmp1 = local_tmp0 + num_groups;

  int base = rank < 4 ? 0 : 4;
  int local_rank = rank - base;
  int cross = rank ^ 4;
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t cross_pair_mask = (1u << rank) | (1u << cross);

  int part_groups = num_groups / 4;
  int start = local_rank * part_groups;
  int end = (local_rank == 3) ? num_groups : start + part_groups;

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag, island_mask);

  Group const* d0 = reinterpret_cast<Group const*>(input_ptrs[base + 0]);
  Group const* d1 = reinterpret_cast<Group const*>(input_ptrs[base + 1]);
  Group const* d2 = reinterpret_cast<Group const*>(input_ptrs[base + 2]);
  Group const* d3 = reinterpret_cast<Group const*>(input_ptrs[base + 3]);
  for (int idx = start + tid; idx < end; idx += stride) {
    Group v = add_pack_group<T, VecPacks>(d0[idx], d1[idx]);
    v = add_pack_group<T, VecPacks>(v, d2[idx]);
    v = add_pack_group<T, VecPacks>(v, d3[idx]);
    local_tmp0[idx] = v;
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag,
                     cross_pair_mask);

  auto const* cross_tmp0 = reinterpret_cast<Group const*>(tmp_ptrs[cross]);
  for (int idx = start + tid; idx < end; idx += stride) {
    local_tmp1[idx] = add_pack_group<T, VecPacks>(local_tmp0[idx],
                                                 cross_tmp0[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag, island_mask);

#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    int cs = chunk * part_groups;
    int ce = (chunk == 3) ? num_groups : cs + part_groups;
    auto const* src =
        (chunk == local_rank)
            ? local_tmp1
            : reinterpret_cast<Group const*>(tmp_ptrs[base + chunk]) +
                  num_groups;
    for (int idx = cs + tid; idx < ce; idx += stride) {
      out_group[idx] = src[idx];
    }
  }

  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 3, flag,
                                island_mask);
  block_barrier_mask_fast_final(signal_ptrs, rank, 8, max_blocks, 4, flag,
                                cross_pair_mask);
  if (threadIdx.x == 0) {
    store_volatile_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                       flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T>
__global__ __launch_bounds__(512, 1) void island_leader8_kernel(
    uint64_t const* input_ptrs, uint64_t const* tmp_ptrs,
    uint64_t const* signal_ptrs, T* output, int num_packs, int rank,
    int max_blocks, bool pdl_sync, bool pdl_release) {
  using Pack = typename PackTraits<T>::Pack;
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       8)) +
             1;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  auto* out_pack = reinterpret_cast<Pack*>(output);
  auto* local_tmp = reinterpret_cast<Pack*>(tmp_ptrs[rank]);

  int peer1 = rank ^ 1;
  int peer2 = rank ^ 2;
  int leader = rank < 4 ? 0 : 4;
  int peer_leader = rank < 4 ? 4 : 0;
  uint32_t node_pair_mask = (1u << rank) | (1u << peer2);
  uint32_t island_mask = rank < 4 ? 0x0fu : 0xf0u;
  uint32_t leader_pair_mask = (1u << 0) | (1u << 4);

  auto const* input_self = reinterpret_cast<Pack const*>(input_ptrs[rank]);
  auto const* input_peer1 = reinterpret_cast<Pack const*>(input_ptrs[peer1]);
  for (int idx = tid; idx < num_packs; idx += stride) {
    local_tmp[idx] = add_pack<T>(input_self[idx], input_peer1[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 0, flag,
                     node_pair_mask);

  auto const* tmp_peer2 = reinterpret_cast<Pack const*>(tmp_ptrs[peer2]);
  auto* local_tmp1 = local_tmp + num_packs;
  for (int idx = tid; idx < num_packs; idx += stride) {
    local_tmp1[idx] = add_pack<T>(local_tmp[idx], tmp_peer2[idx]);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 1, flag, island_mask);

  if (rank == leader) {
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 2, flag,
                       leader_pair_mask);
    auto const* peer_island_sum =
        reinterpret_cast<Pack const*>(tmp_ptrs[peer_leader]) + num_packs;
    for (int idx = tid; idx < num_packs; idx += stride) {
      local_tmp[idx] = add_pack<T>(local_tmp1[idx], peer_island_sum[idx]);
    }
    block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 3, flag,
                       leader_pair_mask);
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 4, flag, island_mask);

  auto const* leader_sum = reinterpret_cast<Pack const*>(tmp_ptrs[leader]);
  for (int idx = tid; idx < num_packs; idx += stride) {
    out_pack[idx] = leader_sum[idx];
  }

  block_barrier_mask(signal_ptrs, rank, 8, max_blocks, 5, flag, island_mask);
  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks, 8),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <typename T>
__device__ __forceinline__ void multimem_ld_reduce_pack(T const* mc_ptr,
                                                        uint4& out);

template <>
__device__ __forceinline__ void multimem_ld_reduce_pack<nv_bfloat16>(
    nv_bfloat16 const* mc_ptr, uint4& out) {
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(mc_ptr)
      : "memory");
}

template <>
__device__ __forceinline__ void multimem_ld_reduce_pack<half>(
    half const* mc_ptr, uint4& out) {
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
      : "l"(mc_ptr)
      : "memory");
}

template <typename T, int WorldSize>
__global__ __launch_bounds__(512, 1) void multimem_kernel(
    uint64_t input_mc_ptr, uint64_t const* signal_ptrs, T* output,
    int num_packs, int rank, int max_blocks, bool pdl_sync,
    bool pdl_release) {
  static_assert(std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, half>);
  pdl_grid_sync(pdl_sync);
  int32_t* self_signal = reinterpret_cast<int32_t*>(signal_ptrs[rank]);
  int flag = load_acquire_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                       WorldSize)) +
             1;

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 0, flag);

  auto* mc_base = reinterpret_cast<T const*>(input_mc_ptr);
  auto* out_pack = reinterpret_cast<uint4*>(output);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_packs;
       idx += gridDim.x * blockDim.x) {
    uint4 value;
    multimem_ld_reduce_pack<T>(mc_base + idx * PackTraits<T>::kPackElems,
                               value);
    out_pack[idx] = value;
  }

  block_barrier(signal_ptrs, rank, WorldSize, max_blocks, 2, flag);
  if (threadIdx.x == 0) {
    store_release_i32(self_signal + flag_offset(blockIdx.x, max_blocks,
                                                WorldSize),
                      flag);
  }
  pdl_grid_release(pdl_release);
}

template <class Kernel, class... Args>
void launch_custom_kernel(Kernel kernel, dim3 grid, dim3 block,
                          cudaStream_t stream, bool use_pdl_launch,
                          Args const&... args) {
#if CUDART_VERSION >= 12000
  if (use_pdl_launch) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
    return;
  }
#else
  TORCH_CHECK(!use_pdl_launch,
              "PDL launch requires CUDA Runtime 12.0 or newer");
#endif
  kernel<<<grid, block, 0, stream>>>(args...);
}

int clamp_blocks(int blocks, int64_t max_blocks) {
  if (blocks < 1) {
    return 1;
  }
  if (blocks > max_blocks) {
    return static_cast<int>(max_blocks);
  }
  return blocks;
}

LaunchPolicy select_adaptive_policy(int64_t numel, int64_t hidden_size,
                                    int64_t world_size,
                                    int64_t max_blocks) {
  int64_t batch = hidden_size > 0 ? numel / hidden_size : 0;
  if (world_size == 8) {
    if (hidden_size >= 6144) {
      if (batch <= 1) {
        return {CustomAlgo::kTopoRsagPipelinedCrossPush,
                clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 2) {
        return {CustomAlgo::kTopoRsagPipelinedCrossPush,
                clamp_blocks(8, max_blocks), 256};
      }
      if (batch <= 4) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(8, max_blocks),
                512};
      }
      if (batch <= 8) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 12) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                256};
      }
      if (batch <= 16) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 20) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(16, max_blocks),
                512};
      }
      if (batch <= 24) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 28) {
        return {CustomAlgo::kTopoRsagWide2, clamp_blocks(4, max_blocks), 512};
      }
      if (batch <= 32) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 36) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(8, max_blocks),
                512};
      }
      if (batch <= 40) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 44) {
        return {CustomAlgo::kTopoRsagWide2, clamp_blocks(4, max_blocks), 512};
      }
      if (batch <= 48) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 52) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(16, max_blocks),
                512};
      }
      if (batch <= 56) {
        return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(12, max_blocks),
                512};
      }
      if (batch <= 60) {
        return {CustomAlgo::kTopoRsagWide2, clamp_blocks(6, max_blocks), 512};
      }
      return {CustomAlgo::kTopoRsagPipelined, clamp_blocks(16, max_blocks),
              512};
    }
    int blocks = 4;
    if (hidden_size <= 2048) {
      blocks = batch <= 1 ? 1 : (batch <= 8 ? 2 : 4);
    } else if (hidden_size <= 5120) {
      blocks = batch <= 4 ? 2 : 4;
    } else {
      blocks = batch <= 1 ? 1 : (batch <= 2 ? 2 : 4);
    }
    return {CustomAlgo::kTree, clamp_blocks(blocks, max_blocks), 256};
  }

  if (world_size == 4) {
    if (hidden_size == 4096) {
      if (batch <= 1) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(4, max_blocks), 128};
      }
      if (batch <= 2) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 4) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 8) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(96, max_blocks), 128};
      }
      if (batch <= 16) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(8, max_blocks), 512};
      }
      if (batch <= 32) {
        return {CustomAlgo::kTwostageFast, clamp_blocks(48, max_blocks), 128};
      }
      if (batch <= 64) {
        return {CustomAlgo::kTwostageFast, clamp_blocks(64, max_blocks), 128};
      }
      if (batch <= 128) {
        return {CustomAlgo::kTwostageFast, clamp_blocks(96, max_blocks), 128};
      }
      if (batch <= 256) {
        return {CustomAlgo::kTwostageFast, clamp_blocks(48, max_blocks), 512};
      }
      return {CustomAlgo::kRsag, clamp_blocks(64, max_blocks), 512};
    }
    if (hidden_size >= 6144 && batch <= 4) {
      return {CustomAlgo::kOneshot, clamp_blocks(16, max_blocks), 256};
    }
    if (batch >= 512 && hidden_size >= 4096) {
      return {CustomAlgo::kRsag, clamp_blocks(32, max_blocks), 512};
    }
    int blocks = batch <= 8 ? 8 : (batch <= 16 ? 16 : 32);
    int threads = (batch >= 128 || hidden_size == 5120) ? 512 : 256;
    return {CustomAlgo::kTwostageFast, clamp_blocks(blocks, max_blocks),
            threads};
  }

  if (world_size == 2) {
    if (hidden_size <= 2048) {
      if (batch <= 1) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(2, max_blocks), 128};
      }
      if (batch <= 2) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(32, max_blocks), 128};
      }
      if (batch <= 4) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(96, max_blocks), 128};
      }
      if (batch <= 8) {
        return {CustomAlgo::kTp2RemotePushStream, clamp_blocks(96, max_blocks),
                128};
      }
      if (batch <= 12) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 64};
      }
      if (batch <= 16) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(64, max_blocks), 128};
      }
      if (batch <= 20) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(8, max_blocks), 128};
      }
      if (batch <= 24) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 64};
      }
      if (batch <= 28) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 30) {
        return {CustomAlgo::kTp2RemotePushStream, clamp_blocks(32, max_blocks),
                128};
      }
      if (batch <= 32) {
        return {CustomAlgo::kTp2RemotePushStream, clamp_blocks(32, max_blocks),
                128};
      }
      if (batch <= 36) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 40) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 44) {
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 64) {
        if (batch <= 48) {
          return {CustomAlgo::kTp2RemotePushStream,
                  clamp_blocks(32, max_blocks), 128};
        }
        return {CustomAlgo::kTp2RemotePush, clamp_blocks(16, max_blocks), 128};
      }
      if (batch <= 128) {
        return {CustomAlgo::kPushOneshot, clamp_blocks(16, max_blocks), 256};
      }
      if (batch <= 256) {
        return {CustomAlgo::kTp2Oneshot, clamp_blocks(12, max_blocks), 512};
      }
      return {CustomAlgo::kTwostageFast, clamp_blocks(48, max_blocks), 512};
    }
    if (hidden_size >= 4096 && batch >= 512) {
      return {CustomAlgo::kTwostageFast, clamp_blocks(32, max_blocks), 512};
    }
    if (hidden_size >= 5120 && batch >= 128) {
      return {CustomAlgo::kTwostageFast, clamp_blocks(32, max_blocks), 512};
    }
    int blocks = batch >= 128 ? 32 : 16;
    return {CustomAlgo::kTree, clamp_blocks(blocks, max_blocks), 512};
  }

  return {CustomAlgo::kOneshot, clamp_blocks(8, max_blocks), 512};
}

const char* algo_name(CustomAlgo algo) {
  switch (algo) {
    case CustomAlgo::kOneshot:
      return "custom_oneshot";
    case CustomAlgo::kTwostageFast:
      return "custom_twostage_fast";
    case CustomAlgo::kTree:
      return "custom_tree";
    case CustomAlgo::kRsag:
      return "custom_rsag";
    case CustomAlgo::kTopoTree:
      return "custom_topo_tree";
    case CustomAlgo::kTopoRsag:
      return "custom_topo_rsag";
    case CustomAlgo::kTp2Oneshot:
      return "custom_tp2_oneshot";
    case CustomAlgo::kTp2OneshotPairFast:
      return "custom_tp2_oneshot_pairfast";
    case CustomAlgo::kPushOneshot:
      return "custom_push_oneshot";
    case CustomAlgo::kTp2RemotePush:
      return "custom_tp2_remote_push";
    case CustomAlgo::kTp2RemotePushStream:
      return "custom_tp2_remote_push_stream";
    case CustomAlgo::kTp2RemotePushWindow4:
      return "custom_tp2_remote_push_window4";
    case CustomAlgo::kFastOneshot:
      return "custom_fast_oneshot";
    case CustomAlgo::kTopoTreeFastFinal:
      return "custom_topo_tree_fastfinal";
    case CustomAlgo::kTopoRsagFastFinal:
      return "custom_topo_rsag_fastfinal";
    case CustomAlgo::kTopoRsagPush:
      return "custom_topo_rsag_push";
    case CustomAlgo::kTopoRsagWide2:
      return "custom_topo_rsag_wide2";
    case CustomAlgo::kTopoRsagWide4:
      return "custom_topo_rsag_wide4";
    case CustomAlgo::kTopoRsagPipelined:
      return "custom_topo_rsag_pipelined";
    case CustomAlgo::kTopoRsagPipelinedDirect:
      return "custom_topo_rsag_pipelined_direct";
    case CustomAlgo::kTopoRsagPipelinedCrossPush:
      return "custom_topo_rsag_pipelined_crosspush";
    case CustomAlgo::kTopoRsagPipelinedWide2:
      return "custom_topo_rsag_pipelined_wide2";
    case CustomAlgo::kTopoRsagPipelinedChunks8:
      return "custom_topo_rsag_pipelined_chunks8";
    case CustomAlgo::kTopoRsagPipelinedGatherStart:
      return "custom_topo_rsag_pipelined_gatherstart";
    case CustomAlgo::kTopoRsagPipelinedAgPush:
      return "custom_topo_rsag_pipelined_ag_push";
    case CustomAlgo::kTopoRsagPipelinedAgPushSplit:
      return "custom_topo_rsag_pipelined_ag_push_split";
  }
  return "unknown";
}

template <typename T>
void launch_oneshot_typed(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
                          torch::Tensor output, int64_t numel, int64_t rank,
                          int64_t world_size, int64_t max_blocks,
                          int64_t blocks, int64_t threads, bool pdl_sync,
                          bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(oneshot_kernel<T, W>, dim3(static_cast<unsigned>(blocks)), \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        input_ptrs_data, signal_ptrs_data, reinterpret_cast<T*>(output.data_ptr()), \
        num_packs, static_cast<int>(rank), static_cast<int>(max_blocks),       \
        pdl_sync, pdl_release);                                                \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(6);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size: ", world_size);
  }
#undef DISPATCH_WORLD
}

template <typename T, bool PairFast>
void launch_tp2_oneshot_typed(torch::Tensor input_ptrs,
                              torch::Tensor signal_ptrs,
                              torch::Tensor output, int64_t numel,
                              int64_t rank, int64_t world_size,
                              int64_t max_blocks, int64_t blocks,
                              int64_t threads, bool pdl_sync,
                              bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_oneshot requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

  launch_custom_kernel(tp2_oneshot_kernel<T, PairFast>,
                       dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), stream,
                       pdl_sync || pdl_release, input_ptrs_data,
                       signal_ptrs_data,
                       reinterpret_cast<T*>(output.data_ptr()), num_packs,
                       static_cast<int>(rank), static_cast<int>(max_blocks),
                       pdl_sync, pdl_release);
}

template <typename T, bool StartBarrier>
void launch_tp2_input_pull_typed(torch::Tensor input_ptrs,
                                 torch::Tensor signal_ptrs,
                                 torch::Tensor output, int64_t numel,
                                 int64_t rank, int64_t world_size,
                                 int64_t max_blocks, int64_t blocks,
                                 int64_t threads, bool pdl_sync,
                                 bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_input_pull requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

  launch_custom_kernel(
      tp2_input_pull_kernel<T, StartBarrier>,
      dim3(static_cast<unsigned>(blocks)),
      dim3(static_cast<unsigned>(threads)), stream, pdl_sync || pdl_release,
      input_ptrs_data, signal_ptrs_data,
      reinterpret_cast<T*>(output.data_ptr()), num_packs,
      static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,
      pdl_release);
}

template <typename T>
void launch_push_oneshot_typed(torch::Tensor input, torch::Tensor tmp_ptrs,
                               torch::Tensor epoch_slots,
                               torch::Tensor output, int64_t numel,
                               int64_t rank, int64_t world_size,
                               int64_t max_blocks, int64_t blocks,
                               int64_t threads, bool pdl_sync,
                               bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define LAUNCH_PUSH(W, USE_PDL)                                                \
  launch_custom_kernel(                                                        \
      push_oneshot_kernel<T, W, USE_PDL>,                                      \
      dim3(static_cast<unsigned>(blocks)),                                     \
      dim3(static_cast<unsigned>(threads)), stream, USE_PDL,                   \
      reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,             \
      epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,    \
      static_cast<int>(rank))

#define DISPATCH_WORLD(W)                                                      \
  case W:                                                                      \
    if (pdl_sync || pdl_release) {                                             \
      LAUNCH_PUSH(W, true);                                                    \
    } else {                                                                   \
      LAUNCH_PUSH(W, false);                                                   \
    }                                                                          \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size for push_oneshot: ",
                  world_size);
  }
#undef DISPATCH_WORLD
#undef LAUNCH_PUSH
}

template <typename T>
void launch_tp2_remote_push_typed(torch::Tensor input, torch::Tensor tmp_ptrs,
                                  torch::Tensor epoch_slots,
                                  torch::Tensor output, int64_t numel,
                                  int64_t rank, int64_t world_size,
                                  int64_t max_blocks, int64_t blocks,
                                  int64_t threads, bool pdl_sync,
                                  bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_remote_push requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  if (pdl_sync || pdl_release) {
    launch_custom_kernel(
        tp2_remote_push_kernel<T, true>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, true,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  } else {
    launch_custom_kernel(
        tp2_remote_push_kernel<T, false>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, false,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  }
  (void)max_blocks;
}

template <typename T>
void launch_tp2_remote_signal_push_typed(
    torch::Tensor input, torch::Tensor tmp_ptrs, torch::Tensor signal_ptrs,
    torch::Tensor epoch_slots, torch::Tensor output, int64_t numel,
    int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
    int64_t threads, bool pdl_sync, bool pdl_release, bool fast_signal) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2,
              "tp2_remote_signal_push requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  if (pdl_sync || pdl_release) {
    if (fast_signal) {
      launch_custom_kernel(
          tp2_remote_signal_push_kernel<T, true, true>,
          dim3(static_cast<unsigned>(blocks)),
          dim3(static_cast<unsigned>(threads)), stream, true,
          reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
          signal_ptrs_data, epoch_slots_data,
          reinterpret_cast<T*>(output.data_ptr()), num_packs,
          static_cast<int>(rank), static_cast<int>(max_blocks));
    } else {
      launch_custom_kernel(
          tp2_remote_signal_push_kernel<T, true, false>,
          dim3(static_cast<unsigned>(blocks)),
          dim3(static_cast<unsigned>(threads)), stream, true,
          reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
          signal_ptrs_data, epoch_slots_data,
          reinterpret_cast<T*>(output.data_ptr()), num_packs,
          static_cast<int>(rank), static_cast<int>(max_blocks));
    }
  } else {
    if (fast_signal) {
      launch_custom_kernel(
          tp2_remote_signal_push_kernel<T, false, true>,
          dim3(static_cast<unsigned>(blocks)),
          dim3(static_cast<unsigned>(threads)), stream, false,
          reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
          signal_ptrs_data, epoch_slots_data,
          reinterpret_cast<T*>(output.data_ptr()), num_packs,
          static_cast<int>(rank), static_cast<int>(max_blocks));
    } else {
      launch_custom_kernel(
          tp2_remote_signal_push_kernel<T, false, false>,
          dim3(static_cast<unsigned>(blocks)),
          dim3(static_cast<unsigned>(threads)), stream, false,
          reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
          signal_ptrs_data, epoch_slots_data,
          reinterpret_cast<T*>(output.data_ptr()), num_packs,
          static_cast<int>(rank), static_cast<int>(max_blocks));
    }
  }
}

template <typename T, int VecPacks>
void launch_tp2_remote_push_wide_typed(
    torch::Tensor input, torch::Tensor tmp_ptrs, torch::Tensor epoch_slots,
    torch::Tensor output, int64_t numel, int64_t rank, int64_t world_size,
    int64_t max_blocks, int64_t blocks, int64_t threads, bool pdl_sync,
    bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_remote_push_wide requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  int64_t num_packs64 = numel / Traits::kPackElems;
  TORCH_CHECK(num_packs64 % VecPacks == 0,
              "num_packs must be divisible by VecPacks");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(num_packs64);
  if (pdl_sync || pdl_release) {
    launch_custom_kernel(
        tp2_remote_push_wide_kernel<T, VecPacks, true>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, true,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  } else {
    launch_custom_kernel(
        tp2_remote_push_wide_kernel<T, VecPacks, false>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, false,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  }
  (void)max_blocks;
}

template <typename T>
void launch_tp2_remote_push_stream_typed(
    torch::Tensor input, torch::Tensor tmp_ptrs, torch::Tensor epoch_slots,
    torch::Tensor output, int64_t numel, int64_t rank, int64_t world_size,
    int64_t max_blocks, int64_t blocks, int64_t threads, bool pdl_sync,
    bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_remote_push_stream requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  if (pdl_sync || pdl_release) {
    launch_custom_kernel(
        tp2_remote_push_stream_kernel<T, true>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, true,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  } else {
    launch_custom_kernel(
        tp2_remote_push_stream_kernel<T, false>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, false,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  }
  (void)max_blocks;
}

template <typename T, bool Stream>
void launch_tp2_remote_flag_push_typed(
    torch::Tensor input, torch::Tensor tmp_ptrs, torch::Tensor flag_ptrs,
    torch::Tensor epoch_slots, torch::Tensor output, int64_t numel,
    int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
    int64_t threads, bool pdl_sync, bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_remote_flag_push requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* flag_ptrs_data =
      reinterpret_cast<uint64_t const*>(flag_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  if (pdl_sync || pdl_release) {
    launch_custom_kernel(
        tp2_remote_flag_push_kernel<T, Stream, true>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, true,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        flag_ptrs_data, epoch_slots_data,
        reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  } else {
    launch_custom_kernel(
        tp2_remote_flag_push_kernel<T, Stream, false>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, false,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        flag_ptrs_data, epoch_slots_data,
        reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  }
  (void)max_blocks;
}

template <typename T, int WindowPacks>
void launch_tp2_remote_push_window_typed(
    torch::Tensor input, torch::Tensor tmp_ptrs, torch::Tensor epoch_slots,
    torch::Tensor output, int64_t numel, int64_t rank, int64_t world_size,
    int64_t max_blocks, int64_t blocks, int64_t threads, bool pdl_sync,
    bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 2, "tp2_remote_push_window requires world_size 2");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* epoch_slots_data = epoch_slots.data_ptr<int32_t>();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  if (pdl_sync || pdl_release) {
    launch_custom_kernel(
        tp2_remote_push_window_kernel<T, WindowPacks, true>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, true,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  } else {
    launch_custom_kernel(
        tp2_remote_push_window_kernel<T, WindowPacks, false>,
        dim3(static_cast<unsigned>(blocks)),
        dim3(static_cast<unsigned>(threads)), stream, false,
        reinterpret_cast<T const*>(input.data_ptr()), tmp_ptrs_data,
        epoch_slots_data, reinterpret_cast<T*>(output.data_ptr()), num_packs,
        static_cast<int>(rank));
  }
  (void)max_blocks;
}

template <typename T>
void launch_push_oneshot_param_raw(
    torch::Tensor input, const std::vector<uint64_t>& tmp_ptrs_host,
    int32_t* epoch_slots_data, torch::Tensor output, int64_t numel,
    int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
    int64_t threads, bool pdl_sync, bool pdl_release,
    bool fp32_reduce = false, int64_t rank_stride_packs = 0,
    int64_t epoch_stride_packs = 0) {
  using Traits = PackTraits<T>;
  constexpr bool kCanFp32Reduce =
      std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  TORCH_CHECK(static_cast<int64_t>(tmp_ptrs_host.size()) == world_size,
              "world_size must match host tmp pointer list");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

  PushOneshotParamData<T> params{};
  for (int64_t peer = 0; peer < world_size; ++peer) {
    params.tmp_ptrs[peer] = tmp_ptrs_host[peer];
  }
  params.input = reinterpret_cast<T const*>(input.data_ptr());
  params.output = reinterpret_cast<T*>(output.data_ptr());
  params.epoch_slots = epoch_slots_data;
  params.num_packs = num_packs;
  params.rank_stride_packs = static_cast<int>(
      rank_stride_packs > 0 ? rank_stride_packs : num_packs);
  params.epoch_stride_packs = static_cast<int>(
      epoch_stride_packs > 0 ? epoch_stride_packs : world_size * num_packs);
  params.rank = static_cast<int>(rank);

#define LAUNCH_PUSH_PARAM(W, USE_PDL, FP32_REDUCE)                               \
  launch_custom_kernel(                                                          \
      push_oneshot_param_kernel<T, W, USE_PDL, FP32_REDUCE>,                     \
      dim3(static_cast<unsigned>(blocks)), dim3(static_cast<unsigned>(threads)), \
      stream, USE_PDL, params)

#define LAUNCH_PUSH_PARAM_REDUCE(W, USE_PDL)                                     \
  if (fp32_reduce && kCanFp32Reduce) {                                           \
    LAUNCH_PUSH_PARAM(W, USE_PDL, true);                                         \
  } else {                                                                       \
    LAUNCH_PUSH_PARAM(W, USE_PDL, false);                                        \
  }

#define DISPATCH_WORLD(W)                                                        \
  case W:                                                                        \
    if (pdl_sync || pdl_release) {                                               \
      LAUNCH_PUSH_PARAM_REDUCE(W, true);                                         \
    } else {                                                                     \
      LAUNCH_PUSH_PARAM_REDUCE(W, false);                                        \
    }                                                                            \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size for push_oneshot_param: ",
                  world_size);
  }
#undef DISPATCH_WORLD
#undef LAUNCH_PUSH_PARAM_REDUCE
#undef LAUNCH_PUSH_PARAM
  (void)max_blocks;
}

template <typename T>
void launch_push_oneshot_param_typed(
    torch::Tensor input, const std::vector<int64_t>& tmp_ptrs_host,
    torch::Tensor epoch_slots, torch::Tensor output, int64_t numel,
    int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
    int64_t threads, bool pdl_sync, bool pdl_release) {
  std::vector<uint64_t> ptrs;
  ptrs.reserve(tmp_ptrs_host.size());
  for (int64_t ptr : tmp_ptrs_host) {
    ptrs.push_back(static_cast<uint64_t>(ptr));
  }
  launch_push_oneshot_param_raw<T>(
      input, ptrs, epoch_slots.data_ptr<int32_t>(), output, numel, rank,
      world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
}

template <typename T>
void launch_fast_oneshot_typed(torch::Tensor input_ptrs,
                               torch::Tensor signal_ptrs,
                               torch::Tensor output, int64_t numel,
                               int64_t rank, int64_t world_size,
                               int64_t max_blocks, int64_t blocks,
                               int64_t threads, bool pdl_sync,
                               bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(                                                     \
        fast_oneshot_kernel<T, W>, dim3(static_cast<unsigned>(blocks)),        \
        dim3(static_cast<unsigned>(threads)), stream, pdl_sync || pdl_release, \
        input_ptrs_data, signal_ptrs_data,                                     \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(6);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size: ", world_size);
  }
#undef DISPATCH_WORLD
}

template <typename T>
void launch_multimem_typed(int64_t input_mc_ptr, torch::Tensor signal_ptrs,
                           torch::Tensor output, int64_t numel, int64_t rank,
                           int64_t world_size, int64_t max_blocks,
                           int64_t blocks, int64_t threads, bool pdl_sync,
                           bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(multimem_kernel<T, W>, dim3(static_cast<unsigned>(blocks)), \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        static_cast<uint64_t>(input_mc_ptr), signal_ptrs_data,                 \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(6);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size: ", world_size);
  }
#undef DISPATCH_WORLD
}

template <typename T>
void launch_twoshot_typed(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                          torch::Tensor signal_ptrs, torch::Tensor output,
                          int64_t numel, int64_t rank, int64_t world_size,
                          int64_t max_blocks, int64_t blocks,
                          int64_t threads, bool pdl_sync, bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(twoshot_kernel<T, W>, dim3(static_cast<unsigned>(blocks)), \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        input_ptrs_data, tmp_ptrs_data, signal_ptrs_data,                      \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(6);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size: ", world_size);
  }
#undef DISPATCH_WORLD
}

template <typename T>
void launch_twostage_fast_typed(torch::Tensor input_ptrs,
                                torch::Tensor tmp_ptrs,
                                torch::Tensor signal_ptrs,
                                torch::Tensor output, int64_t numel,
                                int64_t rank, int64_t world_size,
                                int64_t max_blocks, int64_t blocks,
                                int64_t threads, bool pdl_sync,
                                bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(twostage_fast_kernel<T, W>,                          \
                         dim3(static_cast<unsigned>(blocks)),                  \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        input_ptrs_data, tmp_ptrs_data, signal_ptrs_data,                      \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(6);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "unsupported world_size: ", world_size);
  }
#undef DISPATCH_WORLD
}

template <typename T>
void launch_tree_typed(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                       torch::Tensor signal_ptrs, torch::Tensor output,
                       int64_t numel, int64_t rank, int64_t world_size,
                       int64_t max_blocks, int64_t blocks, int64_t threads,
                       bool pdl_sync, bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(tree_kernel<T, W>, dim3(static_cast<unsigned>(blocks)), \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        input_ptrs_data, tmp_ptrs_data, signal_ptrs_data,                      \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "custom_tree supports world_size 2, 4, or 8");
  }
#undef DISPATCH_WORLD
}

template <typename T>
void launch_rsag_typed(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                       torch::Tensor signal_ptrs, torch::Tensor output,
                       int64_t numel, int64_t rank, int64_t world_size,
                       int64_t max_blocks, int64_t blocks, int64_t threads,
                       bool pdl_sync, bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  TORCH_CHECK(num_packs % world_size == 0,
              "num_packs must be divisible by world_size");

#define DISPATCH_WORLD(W)                                                     \
  case W:                                                                     \
    launch_custom_kernel(rsag_kernel<T, W>, dim3(static_cast<unsigned>(blocks)), \
                         dim3(static_cast<unsigned>(threads)), stream,        \
                         pdl_sync || pdl_release,                             \
        input_ptrs_data, tmp_ptrs_data, signal_ptrs_data,                      \
        reinterpret_cast<T*>(output.data_ptr()), num_packs,                    \
        static_cast<int>(rank), static_cast<int>(max_blocks), pdl_sync,        \
        pdl_release);                                                         \
    break

  switch (world_size) {
    DISPATCH_WORLD(2);
    DISPATCH_WORLD(4);
    DISPATCH_WORLD(8);
    default:
      TORCH_CHECK(false, "custom_rsag supports world_size 2, 4, or 8");
  }
#undef DISPATCH_WORLD
}

template <typename T, class Kernel>
void launch_topo8_typed(Kernel kernel, torch::Tensor input_ptrs,
                        torch::Tensor tmp_ptrs, torch::Tensor signal_ptrs,
                        torch::Tensor output, int64_t numel, int64_t rank,
                        int64_t world_size, int64_t max_blocks,
                        int64_t blocks, int64_t threads, bool pdl_sync,
                        bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 8, "topology-aware kernels currently require TP8");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  launch_custom_kernel(kernel, dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), stream,
                       pdl_sync || pdl_release, input_ptrs_data, tmp_ptrs_data,
                       signal_ptrs_data, reinterpret_cast<T*>(output.data_ptr()),
                       num_packs, static_cast<int>(rank),
                       static_cast<int>(max_blocks), pdl_sync, pdl_release);
}

template <typename T>
void launch_topo8_ag_push_typed(torch::Tensor input_ptrs,
                                torch::Tensor tmp_ptrs,
                                torch::Tensor output_ptrs,
                                torch::Tensor signal_ptrs,
                                torch::Tensor output, int64_t numel,
                                int64_t rank, int64_t world_size,
                                int64_t max_blocks, int64_t blocks,
                                int64_t threads, bool pdl_sync,
                                bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 8, "AG-push topology-aware kernels require TP8");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* output_ptrs_data =
      reinterpret_cast<uint64_t const*>(output_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  launch_custom_kernel(topo_rsag8_pipelined_ag_push_kernel<T>,
                       dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), stream,
                       pdl_sync || pdl_release, input_ptrs_data, tmp_ptrs_data,
                       output_ptrs_data, signal_ptrs_data,
                       reinterpret_cast<T*>(output.data_ptr()), num_packs,
                       static_cast<int>(rank), static_cast<int>(max_blocks),
                       pdl_sync, pdl_release);
}

template <typename T>
void launch_topo8_ag_push_split_typed(torch::Tensor input_ptrs,
                                      torch::Tensor tmp_ptrs,
                                      torch::Tensor output_ptrs,
                                      torch::Tensor signal_ptrs,
                                      torch::Tensor output, int64_t numel,
                                      int64_t rank, int64_t world_size,
                                      int64_t max_blocks, int64_t blocks,
                                      int64_t threads, bool pdl_sync,
                                      bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 8,
              "split AG-push topology-aware kernels require TP8");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* output_ptrs_data =
      reinterpret_cast<uint64_t const*>(output_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(numel / Traits::kPackElems);
  launch_custom_kernel(topo_rsag8_pipelined_ag_push_split_kernel<T>,
                       dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), stream,
                       pdl_sync || pdl_release, input_ptrs_data, tmp_ptrs_data,
                       output_ptrs_data, signal_ptrs_data,
                       reinterpret_cast<T*>(output.data_ptr()), num_packs,
                       static_cast<int>(rank), static_cast<int>(max_blocks),
                       pdl_sync, pdl_release);
}

template <typename T, int VecPacks>
void launch_topo8_wide_typed(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                             torch::Tensor signal_ptrs, torch::Tensor output,
                             int64_t numel, int64_t rank, int64_t world_size,
                             int64_t max_blocks, int64_t blocks,
                             int64_t threads, bool pdl_sync,
                             bool pdl_release) {
  using Traits = PackTraits<T>;
  TORCH_CHECK(world_size == 8, "wide topology-aware kernels require TP8");
  TORCH_CHECK(numel % Traits::kPackElems == 0,
              "numel must be divisible by the 16-byte pack width");
  int64_t num_packs64 = numel / Traits::kPackElems;
  TORCH_CHECK(num_packs64 % VecPacks == 0,
              "num_packs must be divisible by the wide vector group");
  TORCH_CHECK((num_packs64 / VecPacks) % 4 == 0,
              "wide topo RSAG group count must be divisible by 4");
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto* input_ptrs_data =
      reinterpret_cast<uint64_t const*>(input_ptrs.data_ptr<int64_t>());
  auto* tmp_ptrs_data =
      reinterpret_cast<uint64_t const*>(tmp_ptrs.data_ptr<int64_t>());
  auto* signal_ptrs_data =
      reinterpret_cast<uint64_t const*>(signal_ptrs.data_ptr<int64_t>());
  int num_packs = static_cast<int>(num_packs64);
  launch_custom_kernel(topo_rsag8_wide_kernel<T, VecPacks>,
                       dim3(static_cast<unsigned>(blocks)),
                       dim3(static_cast<unsigned>(threads)), stream,
                       pdl_sync || pdl_release, input_ptrs_data, tmp_ptrs_data,
                       signal_ptrs_data, reinterpret_cast<T*>(output.data_ptr()),
                       num_packs, static_cast<int>(rank),
                       static_cast<int>(max_blocks), pdl_sync, pdl_release);
}

void validate_common(torch::Tensor ptrs, torch::Tensor signal_ptrs,
                     torch::Tensor output, int64_t world_size,
                     int64_t max_blocks, int64_t blocks, int64_t threads) {
  TORCH_CHECK(ptrs.is_cuda(), "pointer tensor must be CUDA");
  TORCH_CHECK(signal_ptrs.is_cuda(), "signal pointer tensor must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(ptrs.scalar_type() == at::ScalarType::Long,
              "pointer tensor must be int64");
  TORCH_CHECK(signal_ptrs.scalar_type() == at::ScalarType::Long,
              "signal pointer tensor must be int64");
  TORCH_CHECK(ptrs.is_contiguous(), "pointer tensor must be contiguous");
  TORCH_CHECK(signal_ptrs.is_contiguous(),
              "signal pointer tensor must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(world_size == ptrs.numel(), "world_size must match ptr tensor");
  TORCH_CHECK(world_size == signal_ptrs.numel(),
              "world_size must match signal pointer tensor");
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 6 ||
                  world_size == 8,
              "world_size must be one of 2, 4, 6, 8");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
}

void validate_tmp_ptrs(torch::Tensor tmp_ptrs, int64_t world_size) {
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(), "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == tmp_ptrs.numel(),
              "world_size must match tmp pointer tensor");
}

size_t align_bytes_128(size_t size) {
  return ((size + 127) / 128) * 128;
}

std::vector<std::vector<int>> p2p_native_atomic_matrix(
    const std::vector<int64_t>& devices) {
  int device_count = 0;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  std::vector<std::vector<int>> matrix(
      devices.size(), std::vector<int>(devices.size(), 0));
  for (size_t src_idx = 0; src_idx < devices.size(); ++src_idx) {
    int src = static_cast<int>(devices[src_idx]);
    TORCH_CHECK(src >= 0 && src < device_count, "invalid source device id");
    for (size_t dst_idx = 0; dst_idx < devices.size(); ++dst_idx) {
      int dst = static_cast<int>(devices[dst_idx]);
      TORCH_CHECK(dst >= 0 && dst < device_count,
                  "invalid destination device id");
      if (src == dst) {
        matrix[src_idx][dst_idx] = 1;
        continue;
      }
      int value = 0;
      cudaError_t status = cudaDeviceGetP2PAttribute(
          &value, cudaDevP2PAttrNativeAtomicSupported, src, dst);
      if (status == cudaSuccess) {
        matrix[src_idx][dst_idx] = value != 0 ? 1 : 0;
      } else {
        cudaGetLastError();
        matrix[src_idx][dst_idx] = 0;
      }
    }
  }
  return matrix;
}

class IpcPushAllreduce {
 public:
  IpcPushAllreduce(int64_t rank, int64_t world_size, int64_t max_numel,
                   int64_t elem_size, int64_t max_blocks)
      : rank_(rank),
        world_size_(world_size),
        max_numel_(max_numel),
        elem_size_(elem_size),
        max_blocks_(max_blocks),
        peer_storage_(static_cast<size_t>(world_size), nullptr),
        peer_signal_ptrs_(static_cast<size_t>(world_size), 0),
        peer_tmp_ptrs_(static_cast<size_t>(world_size), 0),
        peer_v2_pack_tmp_ptrs_(static_cast<size_t>(world_size), 0),
        peer_v2_block_tmp_ptrs_(static_cast<size_t>(world_size), 0) {
    TORCH_CHECK(world_size_ == 2 || world_size_ == 4 || world_size_ == 8,
                "IpcPushAllreduce supports world_size 2, 4, or 8");
    TORCH_CHECK(rank_ >= 0 && rank_ < world_size_, "rank out of range");
    TORCH_CHECK(max_numel_ > 0, "max_numel must be positive");
    TORCH_CHECK(elem_size_ == 2 || elem_size_ == 4,
                "elem_size must be 2 or 4 bytes");
    TORCH_CHECK(max_blocks_ > 0, "max_blocks must be positive");
    signal_slots_ = static_cast<size_t>(kSignalPhases) *
                        static_cast<size_t>(max_blocks_) *
                        static_cast<size_t>(world_size_) +
                    static_cast<size_t>(max_blocks_);
    epoch_bytes_ = align_bytes_128(sizeof(int32_t) * signal_slots_);
    max_payload_bytes_ =
        align_bytes_128(static_cast<size_t>(max_numel_) *
                        static_cast<size_t>(elem_size_));
    push_bytes_ = align_bytes_128(2 * static_cast<size_t>(world_size_) *
                                  max_payload_bytes_);
    v2_pack_bytes_ = push_bytes_;
    v2_block_bytes_ = push_bytes_;
    storage_bytes_ =
        epoch_bytes_ + push_bytes_ + v2_pack_bytes_ + v2_block_bytes_;
    C10_CUDA_CHECK(cudaMalloc(&storage_, storage_bytes_));
    C10_CUDA_CHECK(cudaMemset(storage_, 0, storage_bytes_));
  }

  ~IpcPushAllreduce() {
    close();
  }

  pybind11::bytes share_storage() const {
    TORCH_CHECK(storage_ != nullptr, "storage is closed");
    cudaIpcMemHandle_t handle;
    C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, storage_));
    return pybind11::bytes(reinterpret_cast<const char*>(&handle),
                           sizeof(handle));
  }

  void post_init(const pybind11::list& handles) {
    TORCH_CHECK(static_cast<int64_t>(handles.size()) == world_size_,
                "handle count must match world_size");
    TORCH_CHECK(!initialized_, "IpcPushAllreduce is already initialized");
    for (int64_t peer = 0; peer < world_size_; ++peer) {
      if (peer == rank_) {
        peer_storage_[static_cast<size_t>(peer)] = storage_;
      } else {
        std::string raw = pybind11::cast<std::string>(handles[peer]);
        TORCH_CHECK(raw.size() == sizeof(cudaIpcMemHandle_t),
                    "invalid CUDA IPC handle size");
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, raw.data(), sizeof(handle));
        void* ptr = nullptr;
        C10_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handle,
                                            cudaIpcMemLazyEnablePeerAccess));
        peer_storage_[static_cast<size_t>(peer)] = ptr;
      }
      peer_signal_ptrs_[static_cast<size_t>(peer)] =
          reinterpret_cast<uint64_t>(peer_storage_[static_cast<size_t>(peer)]);
      auto* peer_base =
          static_cast<char*>(peer_storage_[static_cast<size_t>(peer)]) +
          static_cast<ptrdiff_t>(epoch_bytes_);
      peer_tmp_ptrs_[static_cast<size_t>(peer)] =
          reinterpret_cast<uint64_t>(peer_base);
      peer_v2_pack_tmp_ptrs_[static_cast<size_t>(peer)] =
          reinterpret_cast<uint64_t>(peer_base + push_bytes_);
      peer_v2_block_tmp_ptrs_[static_cast<size_t>(peer)] =
          reinterpret_cast<uint64_t>(peer_base + push_bytes_ + v2_pack_bytes_);
    }
    initialized_ = true;
  }

  void all_reduce(torch::Tensor input, torch::Tensor output, int64_t numel,
                  int64_t blocks, int64_t threads, bool pdl_sync = false,
                  bool pdl_release = false, bool fp32_reduce = false,
                  bool fixed_stride = false) {
    const at::cuda::OptionalCUDAGuard guard(device_of(output));
    TORCH_CHECK(initialized_, "IpcPushAllreduce must be post_init()'d first");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                "input and output dtype must match");
    TORCH_CHECK(input.element_size() == elem_size_,
                "input dtype element size does not match constructor");
    TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
    TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");
    TORCH_CHECK(static_cast<size_t>(numel) * static_cast<size_t>(elem_size_) <=
                    max_payload_bytes_,
                "numel exceeds IPC scratch capacity");
    TORCH_CHECK(blocks > 0 && blocks <= max_blocks_,
                "blocks must be in (0, max_blocks]");
    TORCH_CHECK(threads > 0 && threads <= 1024,
                "threads must be in (0, 1024]");

    auto* epoch_slots = reinterpret_cast<int32_t*>(storage_);
#define LAUNCH_IPC_PUSH(DTYPE)                                                   \
  do {                                                                           \
    using Traits = PackTraits<DTYPE>;                                            \
    int64_t rank_stride = 0;                                                     \
    int64_t epoch_stride = 0;                                                    \
    if (fixed_stride) {                                                          \
      TORCH_CHECK(max_numel_ % Traits::kPackElems == 0,                          \
                  "max_numel must be divisible by pack width");                  \
      rank_stride = max_numel_ / Traits::kPackElems;                             \
      epoch_stride = world_size_ * rank_stride;                                  \
    }                                                                            \
    launch_push_oneshot_param_raw<DTYPE>(                                        \
        input, peer_tmp_ptrs_, epoch_slots, output, numel, rank_, world_size_,   \
        max_blocks_, blocks, threads, pdl_sync, pdl_release, fp32_reduce,        \
        rank_stride, epoch_stride);                                              \
  } while (false)

    switch (output.scalar_type()) {
      case at::ScalarType::BFloat16:
        LAUNCH_IPC_PUSH(nv_bfloat16);
        break;
      case at::ScalarType::Half:
        LAUNCH_IPC_PUSH(half);
        break;
      case at::ScalarType::Float:
        LAUNCH_IPC_PUSH(float);
        break;
      default:
        TORCH_CHECK(false, "unsupported dtype");
    }
#undef LAUNCH_IPC_PUSH
  }

  void all_reduce_v2(torch::Tensor input, torch::Tensor output, int64_t numel,
                     int64_t blocks, int64_t threads, bool stream_mode,
                     bool pdl_sync = false, bool pdl_release = false) {
    const at::cuda::OptionalCUDAGuard guard(device_of(output));
    TORCH_CHECK(initialized_, "IpcPushAllreduce must be post_init()'d first");
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(input.scalar_type() == output.scalar_type(),
                "input and output dtype must match");
    TORCH_CHECK(input.element_size() == elem_size_,
                "input dtype element size does not match constructor");
    TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
    TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");
    TORCH_CHECK(static_cast<size_t>(numel) * static_cast<size_t>(elem_size_) <=
                    max_payload_bytes_,
                "numel exceeds IPC scratch capacity");
    TORCH_CHECK(blocks > 0 && blocks <= max_blocks_,
                "blocks must be in (0, max_blocks]");
    TORCH_CHECK(threads > 0 && threads <= 1024,
                "threads must be in (0, 1024]");

    auto* epoch_slots = reinterpret_cast<int32_t*>(storage_);
    if (world_size_ != 2) {
#define LAUNCH_IPC_V2_ONESHOT(DTYPE)                                            \
  do {                                                                          \
    using Traits = PackTraits<DTYPE>;                                           \
    TORCH_CHECK(max_numel_ % Traits::kPackElems == 0,                           \
                "max_numel must be divisible by pack width");                  \
    int64_t rank_stride = max_numel_ / Traits::kPackElems;                      \
    int64_t epoch_stride = world_size_ * rank_stride;                           \
    launch_push_oneshot_param_raw<DTYPE>(                                       \
        input, peer_v2_pack_tmp_ptrs_, epoch_slots, output, numel, rank_, world_size_,\
        max_blocks_, blocks, threads, pdl_sync, pdl_release, false,             \
        rank_stride, epoch_stride);                                             \
  } while (false)

#define LAUNCH_IPC_V2_RSAG_TYPED(DTYPE, W, USE_PDL)                             \
  do {                                                                          \
    if constexpr (W == 8) {                                                     \
      launch_custom_kernel(                                                     \
          ipc_topo_rsag8_block_param_kernel<DTYPE, USE_PDL>,                    \
          dim3(static_cast<unsigned>(blocks)),                                  \
          dim3(static_cast<unsigned>(threads)), stream, USE_PDL, params);       \
    } else {                                                                    \
      launch_custom_kernel(                                                     \
          ipc_rsag_push_param_kernel<DTYPE, W, USE_PDL>,                        \
          dim3(static_cast<unsigned>(blocks)),                                  \
          dim3(static_cast<unsigned>(threads)), stream, USE_PDL, params);       \
    }                                                                           \
  } while (false)

#define LAUNCH_IPC_V2_RSAG_WORLD(DTYPE, W)                                      \
  case W:                                                                       \
    if (pdl_sync || pdl_release) {                                              \
      LAUNCH_IPC_V2_RSAG_TYPED(DTYPE, W, true);                                 \
    } else {                                                                    \
      LAUNCH_IPC_V2_RSAG_TYPED(DTYPE, W, false);                                \
    }                                                                           \
    break

#define LAUNCH_IPC_V2_FALLBACK(DTYPE)                                          \
  do {                                                                         \
    using Traits = PackTraits<DTYPE>;                                          \
    TORCH_CHECK(numel % Traits::kPackElems == 0,                               \
                "numel must be divisible by the 16-byte pack width");         \
    TORCH_CHECK(max_numel_ % Traits::kPackElems == 0,                          \
                "max_numel must be divisible by pack width");                 \
    auto stream =                                                              \
        at::cuda::getCurrentCUDAStream(output.get_device()).stream();          \
    PushOneshotParamData<DTYPE> params{};                                      \
    for (int64_t peer = 0; peer < world_size_; ++peer) {                       \
      params.tmp_ptrs[peer] = peer_v2_block_tmp_ptrs_[static_cast<size_t>(peer)];\
      params.signal_ptrs[peer] =                                               \
          peer_signal_ptrs_[static_cast<size_t>(peer)];                        \
    }                                                                          \
    params.input = reinterpret_cast<DTYPE const*>(input.data_ptr());           \
    params.output = reinterpret_cast<DTYPE*>(output.data_ptr());               \
    params.epoch_slots = epoch_slots;                                          \
    params.num_packs = static_cast<int>(numel / Traits::kPackElems);           \
    params.rank_stride_packs =                                                 \
        static_cast<int>(max_numel_ / Traits::kPackElems);                     \
    params.epoch_stride_packs =                                                \
        static_cast<int>(world_size_ * params.rank_stride_packs);              \
    params.rank = static_cast<int>(rank_);                                     \
    params.max_blocks = static_cast<int>(max_blocks_);                         \
    if (world_size_ == 8) {                                                    \
      TORCH_CHECK(blocks % 4 == 0,                                             \
                  "TP8 IPC V2 topology kernel requires blocks divisible by 4");\
    }                                                                          \
    switch (world_size_) {                                                     \
      LAUNCH_IPC_V2_RSAG_WORLD(DTYPE, 4);                                      \
      LAUNCH_IPC_V2_RSAG_WORLD(DTYPE, 8);                                      \
      default:                                                                 \
        TORCH_CHECK(false, "unsupported world_size for IPC V2 RS/AG: ",        \
                    world_size_);                                              \
    }                                                                          \
  } while (false)

#define LAUNCH_IPC_V2_TOPO_PACK_TYPED(DTYPE, USE_PDL)                          \
  launch_custom_kernel(                                                        \
      ipc_topo_rsag8_push_param_kernel<DTYPE, USE_PDL>,                        \
      dim3(static_cast<unsigned>(blocks)),                                     \
      dim3(static_cast<unsigned>(threads)), stream, USE_PDL, params)

#define LAUNCH_IPC_V2_TOPO_PACK(DTYPE)                                         \
  do {                                                                         \
    using Traits = PackTraits<DTYPE>;                                          \
    TORCH_CHECK(numel % Traits::kPackElems == 0,                               \
                "numel must be divisible by the 16-byte pack width");         \
    TORCH_CHECK(max_numel_ % Traits::kPackElems == 0,                          \
                "max_numel must be divisible by pack width");                 \
    TORCH_CHECK(world_size_ == 8, "topology pack mode requires TP8");          \
    TORCH_CHECK(blocks % 4 == 0,                                               \
                "TP8 IPC V2 topology kernel requires blocks divisible by 4"); \
    auto stream =                                                              \
        at::cuda::getCurrentCUDAStream(output.get_device()).stream();          \
    PushOneshotParamData<DTYPE> params{};                                      \
    for (int64_t peer = 0; peer < world_size_; ++peer) {                       \
      params.tmp_ptrs[peer] = peer_v2_pack_tmp_ptrs_[static_cast<size_t>(peer)];\
      params.signal_ptrs[peer] =                                               \
          peer_signal_ptrs_[static_cast<size_t>(peer)];                        \
    }                                                                          \
    params.input = reinterpret_cast<DTYPE const*>(input.data_ptr());           \
    params.output = reinterpret_cast<DTYPE*>(output.data_ptr());               \
    params.epoch_slots = epoch_slots;                                          \
    params.num_packs = static_cast<int>(numel / Traits::kPackElems);           \
    params.rank_stride_packs =                                                 \
        static_cast<int>(max_numel_ / Traits::kPackElems);                     \
    params.epoch_stride_packs =                                                \
        static_cast<int>(world_size_ * params.rank_stride_packs);              \
    params.rank = static_cast<int>(rank_);                                     \
    params.max_blocks = static_cast<int>(max_blocks_);                         \
    if (pdl_sync || pdl_release) {                                             \
      LAUNCH_IPC_V2_TOPO_PACK_TYPED(DTYPE, true);                              \
    } else {                                                                   \
      LAUNCH_IPC_V2_TOPO_PACK_TYPED(DTYPE, false);                             \
    }                                                                          \
  } while (false)

      switch (output.scalar_type()) {
        case at::ScalarType::BFloat16:
          if (world_size_ == 8 && !stream_mode) {
            LAUNCH_IPC_V2_TOPO_PACK(nv_bfloat16);
          } else if (stream_mode) {
            LAUNCH_IPC_V2_FALLBACK(nv_bfloat16);
          } else {
            LAUNCH_IPC_V2_ONESHOT(nv_bfloat16);
          }
          break;
        case at::ScalarType::Half:
          if (world_size_ == 8 && !stream_mode) {
            LAUNCH_IPC_V2_TOPO_PACK(half);
          } else if (stream_mode) {
            LAUNCH_IPC_V2_FALLBACK(half);
          } else {
            LAUNCH_IPC_V2_ONESHOT(half);
          }
          break;
        case at::ScalarType::Float:
          if (world_size_ == 8 && !stream_mode) {
            LAUNCH_IPC_V2_TOPO_PACK(float);
          } else if (stream_mode) {
            LAUNCH_IPC_V2_FALLBACK(float);
          } else {
            LAUNCH_IPC_V2_ONESHOT(float);
          }
          break;
        default:
          TORCH_CHECK(false, "unsupported dtype");
      }
#undef LAUNCH_IPC_V2_FALLBACK
#undef LAUNCH_IPC_V2_TOPO_PACK
#undef LAUNCH_IPC_V2_TOPO_PACK_TYPED
#undef LAUNCH_IPC_V2_RSAG_WORLD
#undef LAUNCH_IPC_V2_RSAG_TYPED
#undef LAUNCH_IPC_V2_ONESHOT
      return;
    }

    auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();

#define LAUNCH_IPC_V2(DTYPE)                                                    \
  do {                                                                          \
    using Traits = PackTraits<DTYPE>;                                           \
    TORCH_CHECK(numel % Traits::kPackElems == 0,                                \
                "numel must be divisible by the 16-byte pack width");          \
    IpcTp2RemotePushData<DTYPE> params{};                                       \
    params.tmp_ptrs[0] = peer_v2_block_tmp_ptrs_[0];                            \
    params.tmp_ptrs[1] = peer_v2_block_tmp_ptrs_[1];                            \
    params.input = reinterpret_cast<DTYPE const*>(input.data_ptr());            \
    params.output = reinterpret_cast<DTYPE*>(output.data_ptr());                \
    params.epoch_slots = epoch_slots;                                           \
    params.num_packs = static_cast<int>(numel / Traits::kPackElems);            \
    params.rank = static_cast<int>(rank_);                                      \
    if (pdl_sync || pdl_release) {                                              \
      if (stream_mode) {                                                        \
        launch_custom_kernel(                                                   \
            ipc_tp2_remote_push_kernel<DTYPE, true, true>,                      \
            dim3(static_cast<unsigned>(blocks)),                                \
            dim3(static_cast<unsigned>(threads)), stream, true, params);        \
      } else {                                                                  \
        launch_custom_kernel(                                                   \
            ipc_tp2_remote_push_kernel<DTYPE, false, true>,                     \
            dim3(static_cast<unsigned>(blocks)),                                \
            dim3(static_cast<unsigned>(threads)), stream, true, params);        \
      }                                                                         \
    } else {                                                                    \
      if (stream_mode) {                                                        \
        launch_custom_kernel(                                                   \
            ipc_tp2_remote_push_kernel<DTYPE, true, false>,                     \
            dim3(static_cast<unsigned>(blocks)),                                \
            dim3(static_cast<unsigned>(threads)), stream, false, params);       \
      } else {                                                                  \
        launch_custom_kernel(                                                   \
            ipc_tp2_remote_push_kernel<DTYPE, false, false>,                    \
            dim3(static_cast<unsigned>(blocks)),                                \
            dim3(static_cast<unsigned>(threads)), stream, false, params);       \
      }                                                                         \
    }                                                                           \
  } while (false)

    switch (output.scalar_type()) {
      case at::ScalarType::BFloat16:
        LAUNCH_IPC_V2(nv_bfloat16);
        break;
      case at::ScalarType::Half:
        LAUNCH_IPC_V2(half);
        break;
      case at::ScalarType::Float:
        LAUNCH_IPC_V2(float);
        break;
      default:
        TORCH_CHECK(false, "unsupported dtype");
    }
#undef LAUNCH_IPC_V2
  }

  void close_peers() {
    for (int64_t peer = 0; peer < world_size_; ++peer) {
      void* ptr = peer_storage_[static_cast<size_t>(peer)];
      if (ptr != nullptr && ptr != storage_) {
        cudaIpcCloseMemHandle(ptr);
        peer_storage_[static_cast<size_t>(peer)] = nullptr;
      }
    }
    initialized_ = false;
  }

  void free_storage() {
    if (storage_ != nullptr) {
      cudaFree(storage_);
      storage_ = nullptr;
    }
    if (rank_ >= 0 && rank_ < world_size_) {
      peer_storage_[static_cast<size_t>(rank_)] = nullptr;
    }
  }

  void close() {
    close_peers();
    free_storage();
  }

 private:
  int64_t rank_;
  int64_t world_size_;
  int64_t max_numel_;
  int64_t elem_size_;
  int64_t max_blocks_;
  size_t signal_slots_ = 0;
  size_t epoch_bytes_ = 0;
  size_t max_payload_bytes_ = 0;
  size_t push_bytes_ = 0;
  size_t v2_pack_bytes_ = 0;
  size_t v2_block_bytes_ = 0;
  size_t storage_bytes_ = 0;
  void* storage_ = nullptr;
  bool initialized_ = false;
  std::vector<void*> peer_storage_;
  std::vector<uint64_t> peer_signal_ptrs_;
  std::vector<uint64_t> peer_tmp_ptrs_;
  std::vector<uint64_t> peer_v2_pack_tmp_ptrs_;
  std::vector<uint64_t> peer_v2_block_tmp_ptrs_;
};

}  // namespace

void oneshot(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
             torch::Tensor output, int64_t numel, int64_t rank,
             int64_t world_size, int64_t max_blocks, int64_t blocks,
             int64_t threads, bool pdl_sync = false,
             bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_oneshot_typed<nv_bfloat16>(input_ptrs, signal_ptrs, output, numel,
                                        rank, world_size, max_blocks, blocks,
                                        threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_oneshot_typed<half>(input_ptrs, signal_ptrs, output, numel, rank,
                                 world_size, max_blocks, blocks, threads,
                                 pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_oneshot_typed<float>(input_ptrs, signal_ptrs, output, numel, rank,
                                  world_size, max_blocks, blocks, threads,
                                  pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_oneshot(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
                 torch::Tensor output, int64_t numel, int64_t rank,
                 int64_t world_size, int64_t max_blocks, int64_t blocks,
                 int64_t threads, bool pdl_sync = false,
                 bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  TORCH_CHECK(world_size == 2, "tp2_oneshot requires world_size 2");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tp2_oneshot_typed<nv_bfloat16, false>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_tp2_oneshot_typed<half, false>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_tp2_oneshot_typed<float, false>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_oneshot_pairfast(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
                          torch::Tensor output, int64_t numel, int64_t rank,
                          int64_t world_size, int64_t max_blocks,
                          int64_t blocks, int64_t threads,
                          bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  TORCH_CHECK(world_size == 2,
              "tp2_oneshot_pairfast requires world_size 2");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tp2_oneshot_typed<nv_bfloat16, true>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_tp2_oneshot_typed<half, true>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_tp2_oneshot_typed<float, true>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_input_pull(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
                    torch::Tensor output, int64_t numel, int64_t rank,
                    int64_t world_size, int64_t max_blocks, int64_t blocks,
                    int64_t threads, bool start_barrier,
                    bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  TORCH_CHECK(world_size == 2, "tp2_input_pull requires world_size 2");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

#define DISPATCH_PULL(DTYPE)                                                    \
  do {                                                                          \
    if (start_barrier) {                                                        \
      launch_tp2_input_pull_typed<DTYPE, true>(                                 \
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks, \
          blocks, threads, pdl_sync, pdl_release);                             \
    } else {                                                                    \
      launch_tp2_input_pull_typed<DTYPE, false>(                                \
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks, \
          blocks, threads, pdl_sync, pdl_release);                             \
    }                                                                           \
  } while (false)

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      DISPATCH_PULL(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      DISPATCH_PULL(half);
      break;
    case at::ScalarType::Float:
      DISPATCH_PULL(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
#undef DISPATCH_PULL
}

void push_oneshot(torch::Tensor input, torch::Tensor tmp_ptrs,
                  torch::Tensor epoch_slots, torch::Tensor output,
                  int64_t numel, int64_t rank, int64_t world_size,
                  int64_t max_blocks, int64_t blocks, int64_t threads,
                  bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == tmp_ptrs.numel(),
              "world_size must match tmp pointer tensor");
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
              "push_oneshot supports world_size 2, 4, or 8");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_push_oneshot_typed<nv_bfloat16>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_push_oneshot_typed<half>(input, tmp_ptrs, epoch_slots, output,
                                      numel, rank, world_size, max_blocks,
                                      blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_push_oneshot_typed<float>(input, tmp_ptrs, epoch_slots, output,
                                       numel, rank, world_size, max_blocks,
                                       blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_remote_push(torch::Tensor input, torch::Tensor tmp_ptrs,
                     torch::Tensor epoch_slots, torch::Tensor output,
                     int64_t numel, int64_t rank, int64_t world_size,
                     int64_t max_blocks, int64_t blocks, int64_t threads,
                     bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2, "tp2_remote_push requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2, "tp2_remote_push requires two tmp ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tp2_remote_push_typed<nv_bfloat16>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_tp2_remote_push_typed<half>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_tp2_remote_push_typed<float>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_remote_signal_push(torch::Tensor input, torch::Tensor tmp_ptrs,
                            torch::Tensor signal_ptrs,
                            torch::Tensor epoch_slots, torch::Tensor output,
                            int64_t numel, int64_t rank, int64_t world_size,
                            int64_t max_blocks, int64_t blocks,
                            int64_t threads, bool pdl_sync = false,
                            bool pdl_release = false,
                            bool fast_signal = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(signal_ptrs.is_cuda(), "signal pointer tensor must be CUDA");
  TORCH_CHECK(signal_ptrs.scalar_type() == at::ScalarType::Long,
              "signal pointer tensor must be int64");
  TORCH_CHECK(signal_ptrs.is_contiguous(),
              "signal pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2,
              "tp2_remote_signal_push requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2 && signal_ptrs.numel() == 2,
              "tp2_remote_signal_push requires two tmp and signal ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tp2_remote_signal_push_typed<nv_bfloat16>(
          input, tmp_ptrs, signal_ptrs, epoch_slots, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release,
          fast_signal);
      break;
    case at::ScalarType::Half:
      launch_tp2_remote_signal_push_typed<half>(
          input, tmp_ptrs, signal_ptrs, epoch_slots, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release,
          fast_signal);
      break;
    case at::ScalarType::Float:
      launch_tp2_remote_signal_push_typed<float>(
          input, tmp_ptrs, signal_ptrs, epoch_slots, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release,
          fast_signal);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_remote_push_wide(torch::Tensor input, torch::Tensor tmp_ptrs,
                          torch::Tensor epoch_slots, torch::Tensor output,
                          int64_t numel, int64_t rank, int64_t world_size,
                          int64_t max_blocks, int64_t blocks,
                          int64_t threads, int64_t vec_packs,
                          bool pdl_sync = false,
                          bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(vec_packs == 2 || vec_packs == 4,
              "vec_packs must be 2 or 4");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2, "tp2_remote_push_wide requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2,
              "tp2_remote_push_wide requires two tmp ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

#define DISPATCH_WIDE(DTYPE)                                                     \
  do {                                                                           \
    if (vec_packs == 2) {                                                        \
      launch_tp2_remote_push_wide_typed<DTYPE, 2>(                               \
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,         \
          max_blocks, blocks, threads, pdl_sync, pdl_release);                  \
    } else {                                                                     \
      launch_tp2_remote_push_wide_typed<DTYPE, 4>(                               \
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,         \
          max_blocks, blocks, threads, pdl_sync, pdl_release);                  \
    }                                                                            \
  } while (false)

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      DISPATCH_WIDE(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      DISPATCH_WIDE(half);
      break;
    case at::ScalarType::Float:
      DISPATCH_WIDE(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
#undef DISPATCH_WIDE
}

void tp2_remote_push_stream(torch::Tensor input, torch::Tensor tmp_ptrs,
                            torch::Tensor epoch_slots, torch::Tensor output,
                            int64_t numel, int64_t rank, int64_t world_size,
                            int64_t max_blocks, int64_t blocks,
                            int64_t threads, bool pdl_sync = false,
                            bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2, "tp2_remote_push_stream requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2,
              "tp2_remote_push_stream requires two tmp ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tp2_remote_push_stream_typed<nv_bfloat16>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_tp2_remote_push_stream_typed<half>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_tp2_remote_push_stream_typed<float>(
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tp2_remote_flag_push(torch::Tensor input, torch::Tensor tmp_ptrs,
                          torch::Tensor flag_ptrs,
                          torch::Tensor epoch_slots, torch::Tensor output,
                          int64_t numel, int64_t rank, int64_t world_size,
                          int64_t max_blocks, int64_t blocks,
                          int64_t threads, bool stream_mode,
                          bool pdl_sync = false,
                          bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(flag_ptrs.is_cuda(), "flag pointer tensor must be CUDA");
  TORCH_CHECK(flag_ptrs.scalar_type() == at::ScalarType::Long,
              "flag pointer tensor must be int64");
  TORCH_CHECK(flag_ptrs.is_contiguous(),
              "flag pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2, "tp2_remote_flag_push requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2 && flag_ptrs.numel() == 2,
              "tp2_remote_flag_push requires two tmp and flag ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

#define DISPATCH_FLAG(DTYPE)                                                    \
  do {                                                                          \
    if (stream_mode) {                                                          \
      launch_tp2_remote_flag_push_typed<DTYPE, true>(                           \
          input, tmp_ptrs, flag_ptrs, epoch_slots, output, numel, rank,         \
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);     \
    } else {                                                                    \
      launch_tp2_remote_flag_push_typed<DTYPE, false>(                          \
          input, tmp_ptrs, flag_ptrs, epoch_slots, output, numel, rank,         \
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);     \
    }                                                                           \
  } while (false)

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      DISPATCH_FLAG(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      DISPATCH_FLAG(half);
      break;
    case at::ScalarType::Float:
      DISPATCH_FLAG(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
#undef DISPATCH_FLAG
}

void tp2_remote_push_window(torch::Tensor input, torch::Tensor tmp_ptrs,
                            torch::Tensor epoch_slots, torch::Tensor output,
                            int64_t numel, int64_t rank, int64_t world_size,
                            int64_t max_blocks, int64_t blocks,
                            int64_t threads, int64_t window_packs,
                            bool pdl_sync = false,
                            bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(window_packs == 2 || window_packs == 4 || window_packs == 8,
              "window_packs must be 2, 4, or 8");
  TORCH_CHECK(tmp_ptrs.is_cuda(), "tmp pointer tensor must be CUDA");
  TORCH_CHECK(tmp_ptrs.scalar_type() == at::ScalarType::Long,
              "tmp pointer tensor must be int64");
  TORCH_CHECK(tmp_ptrs.is_contiguous(),
              "tmp pointer tensor must be contiguous");
  TORCH_CHECK(world_size == 2, "tp2_remote_push_window requires world_size 2");
  TORCH_CHECK(tmp_ptrs.numel() == 2,
              "tp2_remote_push_window requires two tmp ptrs");
  TORCH_CHECK(rank == 0 || rank == 1, "rank must be 0 or 1");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

#define DISPATCH_WINDOW(DTYPE)                                                  \
  do {                                                                          \
    if (window_packs == 2) {                                                    \
      launch_tp2_remote_push_window_typed<DTYPE, 2>(                            \
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,        \
          max_blocks, blocks, threads, pdl_sync, pdl_release);                 \
    } else if (window_packs == 4) {                                             \
      launch_tp2_remote_push_window_typed<DTYPE, 4>(                            \
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,        \
          max_blocks, blocks, threads, pdl_sync, pdl_release);                 \
    } else {                                                                    \
      launch_tp2_remote_push_window_typed<DTYPE, 8>(                            \
          input, tmp_ptrs, epoch_slots, output, numel, rank, world_size,        \
          max_blocks, blocks, threads, pdl_sync, pdl_release);                 \
    }                                                                           \
  } while (false)

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      DISPATCH_WINDOW(nv_bfloat16);
      break;
    case at::ScalarType::Half:
      DISPATCH_WINDOW(half);
      break;
    case at::ScalarType::Float:
      DISPATCH_WINDOW(float);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
#undef DISPATCH_WINDOW
}

void push_oneshot_param(torch::Tensor input,
                        const std::vector<int64_t>& tmp_ptrs_host,
                        torch::Tensor epoch_slots, torch::Tensor output,
                        int64_t numel, int64_t rank, int64_t world_size,
                        int64_t max_blocks, int64_t blocks, int64_t threads,
                        bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "input and output dtype must match");
  TORCH_CHECK(epoch_slots.is_cuda(), "epoch tensor must be CUDA");
  TORCH_CHECK(epoch_slots.scalar_type() == at::ScalarType::Int,
              "epoch tensor must be int32");
  TORCH_CHECK(epoch_slots.is_contiguous(),
              "epoch tensor must be contiguous");
  TORCH_CHECK(epoch_slots.numel() >= blocks,
              "epoch tensor must have at least one element per block");
  TORCH_CHECK(max_blocks > 0, "max_blocks must be positive");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(threads > 0 && threads <= 1024,
              "threads must be in (0, 1024]");
  TORCH_CHECK(world_size == static_cast<int64_t>(tmp_ptrs_host.size()),
              "world_size must match host tmp pointer list");
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
              "push_oneshot_param supports world_size 2, 4, or 8");
  TORCH_CHECK(rank >= 0 && rank < world_size, "rank out of range");
  TORCH_CHECK(numel <= input.numel(), "numel exceeds input size");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_push_oneshot_param_typed<nv_bfloat16>(
          input, tmp_ptrs_host, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_push_oneshot_param_typed<half>(
          input, tmp_ptrs_host, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_push_oneshot_param_typed<float>(
          input, tmp_ptrs_host, epoch_slots, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void fast_oneshot(torch::Tensor input_ptrs, torch::Tensor signal_ptrs,
                  torch::Tensor output, int64_t numel, int64_t rank,
                  int64_t world_size, int64_t max_blocks, int64_t blocks,
                  int64_t threads, bool pdl_sync = false,
                  bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_fast_oneshot_typed<nv_bfloat16>(
          input_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_fast_oneshot_typed<half>(input_ptrs, signal_ptrs, output, numel,
                                      rank, world_size, max_blocks, blocks,
                                      threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_fast_oneshot_typed<float>(input_ptrs, signal_ptrs, output, numel,
                                       rank, world_size, max_blocks, blocks,
                                       threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void twoshot(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
             torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
             int64_t rank, int64_t world_size, int64_t max_blocks,
             int64_t blocks, int64_t threads, bool pdl_sync = false,
             bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_twoshot_typed<nv_bfloat16>(input_ptrs, tmp_ptrs, signal_ptrs,
                                        output, numel, rank, world_size,
                                        max_blocks, blocks, threads, pdl_sync,
                                        pdl_release);
      break;
    case at::ScalarType::Half:
      launch_twoshot_typed<half>(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                 numel, rank, world_size, max_blocks, blocks,
                                 threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_twoshot_typed<float>(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                  numel, rank, world_size, max_blocks, blocks,
                                  threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void twostage_fast(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                   torch::Tensor signal_ptrs, torch::Tensor output,
                   int64_t numel, int64_t rank, int64_t world_size,
                   int64_t max_blocks, int64_t blocks, int64_t threads,
                   bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_twostage_fast_typed<nv_bfloat16>(
          input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_twostage_fast_typed<half>(input_ptrs, tmp_ptrs, signal_ptrs,
                                       output, numel, rank, world_size,
                                       max_blocks, blocks, threads, pdl_sync,
                                       pdl_release);
      break;
    case at::ScalarType::Float:
      launch_twostage_fast_typed<float>(input_ptrs, tmp_ptrs, signal_ptrs,
                                        output, numel, rank, world_size,
                                        max_blocks, blocks, threads, pdl_sync,
                                        pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void tree(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
          torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
          int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
          int64_t threads, bool pdl_sync = false,
          bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_tree_typed<nv_bfloat16>(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                     numel, rank, world_size, max_blocks,
                                     blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_tree_typed<half>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                              rank, world_size, max_blocks, blocks, threads,
                              pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_tree_typed<float>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                               rank, world_size, max_blocks, blocks, threads,
                               pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void rsag(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
          torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
          int64_t rank, int64_t world_size, int64_t max_blocks, int64_t blocks,
          int64_t threads, bool pdl_sync = false,
          bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_rsag_typed<nv_bfloat16>(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                     numel, rank, world_size, max_blocks,
                                     blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_rsag_typed<half>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                              rank, world_size, max_blocks, blocks, threads,
                              pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_rsag_typed<float>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                               rank, world_size, max_blocks, blocks, threads,
                               pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_tree(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
               torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
               int64_t rank, int64_t world_size, int64_t max_blocks,
               int64_t blocks, int64_t threads, bool pdl_sync = false,
               bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8, "custom_topo_tree currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_tree8_kernel<nv_bfloat16, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_tree8_kernel<half, false>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_tree8_kernel<float, false>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
               torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
               int64_t rank, int64_t world_size, int64_t max_blocks,
               int64_t blocks, int64_t threads, bool pdl_sync = false,
               bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8, "custom_topo_rsag currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_kernel<nv_bfloat16, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_kernel<half, false>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_kernel<float, false>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_tree_fastfinal(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                         torch::Tensor signal_ptrs, torch::Tensor output,
                         int64_t numel, int64_t rank, int64_t world_size,
                         int64_t max_blocks, int64_t blocks, int64_t threads,
                         bool pdl_sync = false,
                         bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_tree_fastfinal currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_tree8_kernel<nv_bfloat16, true>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_tree8_kernel<half, true>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_tree8_kernel<float, true>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_fastfinal(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                         torch::Tensor signal_ptrs, torch::Tensor output,
                         int64_t numel, int64_t rank, int64_t world_size,
                         int64_t max_blocks, int64_t blocks, int64_t threads,
                         bool pdl_sync = false,
                         bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_fastfinal currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_kernel<nv_bfloat16, true>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_kernel<half, true>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_kernel<float, true>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                         torch::Tensor signal_ptrs, torch::Tensor output,
                         int64_t numel, int64_t rank, int64_t world_size,
                         int64_t max_blocks, int64_t blocks, int64_t threads,
                         bool pdl_sync = false,
                         bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined currently supports TP8 only");
  TORCH_CHECK(blocks >= 4 && blocks % 4 == 0,
              "custom_topo_rsag_pipelined requires blocks to be a multiple of 4");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_pipelined_kernel<nv_bfloat16, 4, false>, input_ptrs,
          tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_pipelined_kernel<half, 4, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_pipelined_kernel<float, 4, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined_gatherstart(torch::Tensor input_ptrs,
                                     torch::Tensor tmp_ptrs,
                                     torch::Tensor signal_ptrs,
                                     torch::Tensor output, int64_t numel,
                                     int64_t rank, int64_t world_size,
                                     int64_t max_blocks, int64_t blocks,
                                     int64_t threads,
                                     bool pdl_sync = false,
                                     bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_gatherstart currently supports TP8 only");
  TORCH_CHECK(blocks >= 4 && blocks % 4 == 0,
              "custom_topo_rsag_pipelined_gatherstart requires blocks to be a multiple of 4");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_pipelined_kernel<nv_bfloat16, 4, true>, input_ptrs,
          tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_pipelined_kernel<half, 4, true>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_pipelined_kernel<float, 4, true>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined_ag_push(torch::Tensor input_ptrs,
                                 torch::Tensor tmp_ptrs,
                                 torch::Tensor output_ptrs,
                                 torch::Tensor signal_ptrs,
                                 torch::Tensor output, int64_t numel,
                                 int64_t rank, int64_t world_size,
                                 int64_t max_blocks, int64_t blocks,
                                 int64_t threads, bool pdl_sync = false,
                                 bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  validate_tmp_ptrs(output_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_ag_push currently supports TP8 only");
  TORCH_CHECK(blocks >= 4 && blocks % 4 == 0,
              "custom_topo_rsag_pipelined_ag_push requires blocks to be a multiple of 4");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_ag_push_typed<nv_bfloat16>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_ag_push_typed<half>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_ag_push_typed<float>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined_ag_push_split(torch::Tensor input_ptrs,
                                       torch::Tensor tmp_ptrs,
                                       torch::Tensor output_ptrs,
                                       torch::Tensor signal_ptrs,
                                       torch::Tensor output, int64_t numel,
                                       int64_t rank, int64_t world_size,
                                       int64_t max_blocks, int64_t blocks,
                                       int64_t threads,
                                       bool pdl_sync = false,
                                       bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  validate_tmp_ptrs(output_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_ag_push_split currently supports TP8 only");
  TORCH_CHECK(blocks >= 16 && blocks % 16 == 0,
              "custom_topo_rsag_pipelined_ag_push_split requires blocks to be a multiple of 16");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_ag_push_split_typed<nv_bfloat16>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_ag_push_split_typed<half>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_ag_push_split_typed<float>(
          input_ptrs, tmp_ptrs, output_ptrs, signal_ptrs, output, numel, rank,
          world_size, max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined_chunks8(torch::Tensor input_ptrs,
                                 torch::Tensor tmp_ptrs,
                                 torch::Tensor signal_ptrs,
                                 torch::Tensor output, int64_t numel,
                                 int64_t rank, int64_t world_size,
                                 int64_t max_blocks, int64_t blocks,
                                 int64_t threads, bool pdl_sync = false,
                                 bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_chunks8 currently supports TP8 only");
  TORCH_CHECK(blocks >= 8 && blocks % 8 == 0,
              "custom_topo_rsag_pipelined_chunks8 requires blocks to be a multiple of 8");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_pipelined_kernel<nv_bfloat16, 8, false>, input_ptrs,
          tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_pipelined_kernel<half, 8, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_pipelined_kernel<float, 8, false>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

template <bool CrossPush>
void topo_rsag_pipelined_direct_impl(torch::Tensor input_ptrs,
                                     torch::Tensor tmp_ptrs,
                                     torch::Tensor signal_ptrs,
                                     torch::Tensor output, int64_t numel,
                                     int64_t rank, int64_t world_size,
                                     int64_t max_blocks, int64_t blocks,
                                     int64_t threads,
                                     bool pdl_sync = false,
                                     bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_direct currently supports TP8 only");
  TORCH_CHECK(blocks >= 4 && blocks % 4 == 0,
              "custom_topo_rsag_pipelined_direct requires blocks to be a multiple of 4");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_pipelined_direct_kernel<nv_bfloat16, CrossPush>,
          input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_pipelined_direct_kernel<half, CrossPush>, input_ptrs,
          tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_pipelined_direct_kernel<float, CrossPush>, input_ptrs,
          tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
          blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_pipelined_direct(torch::Tensor input_ptrs,
                                torch::Tensor tmp_ptrs,
                                torch::Tensor signal_ptrs,
                                torch::Tensor output, int64_t numel,
                                int64_t rank, int64_t world_size,
                                int64_t max_blocks, int64_t blocks,
                                int64_t threads, bool pdl_sync = false,
                                bool pdl_release = false) {
  topo_rsag_pipelined_direct_impl<false>(
      input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
      max_blocks, blocks, threads, pdl_sync, pdl_release);
}

void topo_rsag_pipelined_crosspush(torch::Tensor input_ptrs,
                                   torch::Tensor tmp_ptrs,
                                   torch::Tensor signal_ptrs,
                                   torch::Tensor output, int64_t numel,
                                   int64_t rank, int64_t world_size,
                                   int64_t max_blocks, int64_t blocks,
                                   int64_t threads, bool pdl_sync = false,
                                   bool pdl_release = false) {
  topo_rsag_pipelined_direct_impl<true>(
      input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
      max_blocks, blocks, threads, pdl_sync, pdl_release);
}

template <int VecPacks>
void topo_rsag_pipelined_wide(torch::Tensor input_ptrs,
                              torch::Tensor tmp_ptrs,
                              torch::Tensor signal_ptrs,
                              torch::Tensor output, int64_t numel,
                              int64_t rank, int64_t world_size,
                              int64_t max_blocks, int64_t blocks,
                              int64_t threads, bool pdl_sync = false,
                              bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_pipelined_wide currently supports TP8 only");
  TORCH_CHECK(blocks >= 4 && blocks % 4 == 0,
              "custom_topo_rsag_pipelined_wide requires blocks to be a multiple of 4");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  auto launch = [&](auto dtype_tag) {
    using DType = decltype(dtype_tag);
    using Traits = PackTraits<DType>;
    TORCH_CHECK(numel % Traits::kPackElems == 0,
                "numel must be divisible by the 16-byte pack width");
    int64_t num_packs = numel / Traits::kPackElems;
    TORCH_CHECK(num_packs % VecPacks == 0,
                "num_packs must be divisible by the wide vector group");
    TORCH_CHECK((num_packs / VecPacks) % 4 == 0,
                "pipelined wide group count must be divisible by 4");
    launch_topo8_typed<DType>(
        topo_rsag8_pipelined_wide_kernel<DType, VecPacks>, input_ptrs,
        tmp_ptrs, signal_ptrs, output, numel, rank, world_size, max_blocks,
        blocks, threads, pdl_sync, pdl_release);
  };

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch(nv_bfloat16{});
      break;
    case at::ScalarType::Half:
      launch(half{});
      break;
    case at::ScalarType::Float:
      launch(float{});
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void topo_rsag_push(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                    torch::Tensor signal_ptrs, torch::Tensor output,
                    int64_t numel, int64_t rank, int64_t world_size,
                    int64_t max_blocks, int64_t blocks, int64_t threads,
                    bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_push currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          topo_rsag8_push_kernel<nv_bfloat16>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          topo_rsag8_push_kernel<half>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          topo_rsag8_push_kernel<float>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

template <int VecPacks>
void topo_rsag_wide(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                    torch::Tensor signal_ptrs, torch::Tensor output,
                    int64_t numel, int64_t rank, int64_t world_size,
                    int64_t max_blocks, int64_t blocks, int64_t threads,
                    bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_topo_rsag_wide currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_wide_typed<nv_bfloat16, VecPacks>(
          input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_wide_typed<half, VecPacks>(
          input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_wide_typed<float, VecPacks>(
          input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
          max_blocks, blocks, threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void island_leader(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
                   torch::Tensor signal_ptrs, torch::Tensor output,
                   int64_t numel, int64_t rank, int64_t world_size,
                   int64_t max_blocks, int64_t blocks, int64_t threads,
                   bool pdl_sync = false, bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  validate_common(input_ptrs, signal_ptrs, output, world_size, max_blocks,
                  blocks, threads);
  validate_tmp_ptrs(tmp_ptrs, world_size);
  TORCH_CHECK(world_size == 8,
              "custom_island_leader currently supports TP8 only");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_topo8_typed<nv_bfloat16>(
          island_leader8_kernel<nv_bfloat16>, input_ptrs, tmp_ptrs,
          signal_ptrs, output, numel, rank, world_size, max_blocks, blocks,
          threads, pdl_sync, pdl_release);
      break;
    case at::ScalarType::Half:
      launch_topo8_typed<half>(
          island_leader8_kernel<half>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    case at::ScalarType::Float:
      launch_topo8_typed<float>(
          island_leader8_kernel<float>, input_ptrs, tmp_ptrs, signal_ptrs,
          output, numel, rank, world_size, max_blocks, blocks, threads,
          pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype");
  }
}

void multimem(int64_t input_mc_ptr, torch::Tensor signal_ptrs,
              torch::Tensor output, int64_t numel, int64_t rank,
              int64_t world_size, int64_t max_blocks, int64_t blocks,
              int64_t threads, bool pdl_sync = false,
              bool pdl_release = false) {
  const at::cuda::OptionalCUDAGuard guard(device_of(output));
  TORCH_CHECK(signal_ptrs.is_cuda(), "signal pointer tensor must be CUDA");
  TORCH_CHECK(signal_ptrs.scalar_type() == at::ScalarType::Long,
              "signal pointer tensor must be int64");
  TORCH_CHECK(signal_ptrs.is_contiguous(),
              "signal pointer tensor must be contiguous");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(world_size == signal_ptrs.numel(),
              "world_size must match signal pointer tensor");
  TORCH_CHECK(blocks > 0 && blocks <= max_blocks,
              "blocks must be in (0, max_blocks]");
  TORCH_CHECK(numel <= output.numel(), "numel exceeds output size");
  TORCH_CHECK(input_mc_ptr != 0, "input multicast pointer is null");

  switch (output.scalar_type()) {
    case at::ScalarType::BFloat16:
      launch_multimem_typed<nv_bfloat16>(input_mc_ptr, signal_ptrs, output,
                                         numel, rank, world_size, max_blocks,
                                         blocks, threads, pdl_sync,
                                         pdl_release);
      break;
    case at::ScalarType::Half:
      launch_multimem_typed<half>(input_mc_ptr, signal_ptrs, output, numel,
                                  rank, world_size, max_blocks, blocks,
                                  threads, pdl_sync, pdl_release);
      break;
    default:
      TORCH_CHECK(false, "custom multimem only supports bf16/fp16");
  }
}

std::tuple<std::string, int64_t, int64_t> adaptive_policy(
    int64_t numel, int64_t hidden_size, int64_t world_size,
    int64_t max_blocks) {
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  TORCH_CHECK(numel % hidden_size == 0,
              "numel must be divisible by hidden_size");
  LaunchPolicy policy =
      select_adaptive_policy(numel, hidden_size, world_size, max_blocks);
  return {std::string(algo_name(policy.algo)), policy.blocks, policy.threads};
}

void adaptive(torch::Tensor input_ptrs, torch::Tensor tmp_ptrs,
              torch::Tensor signal_ptrs, torch::Tensor output, int64_t numel,
              int64_t rank, int64_t world_size, int64_t max_blocks,
              int64_t hidden_size, bool pdl_sync = false,
              bool pdl_release = false) {
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
  TORCH_CHECK(numel % hidden_size == 0,
              "numel must be divisible by hidden_size");
  LaunchPolicy policy =
      select_adaptive_policy(numel, hidden_size, world_size, max_blocks);
  switch (policy.algo) {
    case CustomAlgo::kOneshot:
      oneshot(input_ptrs, signal_ptrs, output, numel, rank, world_size,
              max_blocks, policy.blocks, policy.threads, pdl_sync,
              pdl_release);
      break;
    case CustomAlgo::kTwostageFast:
      twostage_fast(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                    world_size, max_blocks, policy.blocks, policy.threads,
                    pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTree:
      tree(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
           max_blocks, policy.blocks, policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kRsag:
      rsag(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank, world_size,
           max_blocks, policy.blocks, policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoTree:
      topo_tree(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                world_size, max_blocks, policy.blocks, policy.threads,
                pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsag:
      topo_rsag(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                world_size, max_blocks, policy.blocks, policy.threads,
                pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoTreeFastFinal:
      topo_tree_fastfinal(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                          rank, world_size, max_blocks, policy.blocks,
                          policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagFastFinal:
      topo_rsag_fastfinal(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                          rank, world_size, max_blocks, policy.blocks,
                          policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelined:
      topo_rsag_pipelined(input_ptrs, tmp_ptrs, signal_ptrs, output, numel,
                          rank, world_size, max_blocks, policy.blocks,
                          policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedDirect:
      topo_rsag_pipelined_direct(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                 numel, rank, world_size, max_blocks,
                                 policy.blocks, policy.threads, pdl_sync,
                                 pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedCrossPush:
      topo_rsag_pipelined_crosspush(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                    numel, rank, world_size, max_blocks,
                                    policy.blocks, policy.threads, pdl_sync,
                                    pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedWide2:
      topo_rsag_pipelined_wide<2>(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                  numel, rank, world_size, max_blocks,
                                  policy.blocks, policy.threads, pdl_sync,
                                  pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedChunks8:
      topo_rsag_pipelined_chunks8(input_ptrs, tmp_ptrs, signal_ptrs, output,
                                  numel, rank, world_size, max_blocks,
                                  policy.blocks, policy.threads, pdl_sync,
                                  pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedGatherStart:
      topo_rsag_pipelined_gatherstart(input_ptrs, tmp_ptrs, signal_ptrs,
                                      output, numel, rank, world_size,
                                      max_blocks, policy.blocks,
                                      policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagPipelinedAgPush:
      TORCH_CHECK(false,
                  "custom_topo_rsag_pipelined_ag_push requires output_ptrs and is not selectable by adaptive()");
      break;
    case CustomAlgo::kTopoRsagPipelinedAgPushSplit:
      TORCH_CHECK(false,
                  "custom_topo_rsag_pipelined_ag_push_split requires output_ptrs and is not selectable by adaptive()");
      break;
    case CustomAlgo::kTopoRsagPush:
      topo_rsag_push(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                     world_size, max_blocks, policy.blocks, policy.threads,
                     pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagWide2:
      topo_rsag_wide<2>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                        world_size, max_blocks, policy.blocks, policy.threads,
                        pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTopoRsagWide4:
      topo_rsag_wide<4>(input_ptrs, tmp_ptrs, signal_ptrs, output, numel, rank,
                        world_size, max_blocks, policy.blocks, policy.threads,
                        pdl_sync, pdl_release);
      break;
    case CustomAlgo::kTp2Oneshot:
      tp2_oneshot(input_ptrs, signal_ptrs, output, numel, rank, world_size,
                  max_blocks, policy.blocks, policy.threads, pdl_sync,
                  pdl_release);
      break;
    case CustomAlgo::kTp2OneshotPairFast:
      tp2_oneshot_pairfast(input_ptrs, signal_ptrs, output, numel, rank,
                           world_size, max_blocks, policy.blocks,
                           policy.threads, pdl_sync, pdl_release);
      break;
    case CustomAlgo::kPushOneshot:
      TORCH_CHECK(false,
                  "custom_push_oneshot needs direct input and local epoch; use adaptive_policy() and dispatch it from Python");
      break;
    case CustomAlgo::kTp2RemotePush:
      TORCH_CHECK(false,
                  "custom_tp2_remote_push needs direct input and local epoch; use adaptive_policy() and dispatch it from Python");
      break;
    case CustomAlgo::kTp2RemotePushStream:
      TORCH_CHECK(false,
                  "custom_tp2_remote_push_stream needs direct input and local epoch; use adaptive_policy() and dispatch it from Python");
      break;
    case CustomAlgo::kTp2RemotePushWindow4:
      TORCH_CHECK(false,
                  "custom_tp2_remote_push_window4 needs direct input and local epoch; use adaptive_policy() and dispatch it from Python");
      break;
    case CustomAlgo::kFastOneshot:
      fast_oneshot(input_ptrs, signal_ptrs, output, numel, rank, world_size,
                   max_blocks, policy.blocks, policy.threads, pdl_sync,
                   pdl_release);
      break;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  namespace py = pybind11;
  py::class_<IpcPushAllreduce>(m, "IpcPushAllreduce")
      .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t>(),
           py::arg("rank"), py::arg("world_size"), py::arg("max_numel"),
           py::arg("elem_size"), py::arg("max_blocks"))
      .def("share_storage", &IpcPushAllreduce::share_storage)
      .def("post_init", &IpcPushAllreduce::post_init, py::arg("handles"))
      .def("all_reduce", &IpcPushAllreduce::all_reduce,
           py::arg("input"), py::arg("output"), py::arg("numel"),
           py::arg("blocks"), py::arg("threads"),
           py::arg("pdl_sync") = false, py::arg("pdl_release") = false,
           py::arg("fp32_reduce") = false, py::arg("fixed_stride") = false)
      .def("all_reduce_v2", &IpcPushAllreduce::all_reduce_v2,
           py::arg("input"), py::arg("output"), py::arg("numel"),
           py::arg("blocks"), py::arg("threads"), py::arg("stream_mode"),
           py::arg("pdl_sync") = false, py::arg("pdl_release") = false)
      .def("close_peers", &IpcPushAllreduce::close_peers)
      .def("free_storage", &IpcPushAllreduce::free_storage)
      .def("close", &IpcPushAllreduce::close);
  m.def("p2p_native_atomic_matrix", &p2p_native_atomic_matrix,
        "Return cudaDevP2PAttrNativeAtomicSupported for directed device pairs",
        py::arg("devices"));
  m.def("oneshot", &oneshot,
        "TP allreduce oneshot over symmetric memory",
        py::arg("input_ptrs"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("tp2_oneshot", &tp2_oneshot,
        "TP2 allreduce oneshot with pairwise symmetric-memory barrier",
        py::arg("input_ptrs"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("tp2_oneshot_pairfast", &tp2_oneshot_pairfast,
        "TP2 allreduce oneshot with single-peer fast barriers",
        py::arg("input_ptrs"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("tp2_input_pull", &tp2_input_pull,
        "TP2 allreduce by directly pulling peer symmetric input",
        py::arg("input_ptrs"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("start_barrier"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("push_oneshot", &push_oneshot,
        "TP allreduce one-shot push using peer-visible scratch buffers",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("tp2_remote_push", &tp2_remote_push,
        "TP2 allreduce remote-only push using one peer-visible scratch slot",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("tp2_remote_signal_push", &tp2_remote_signal_push,
        "TP2 allreduce remote-only push with one block-level signal",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("epoch_slots"), py::arg("output"), py::arg("numel"),
        py::arg("rank"), py::arg("world_size"), py::arg("max_blocks"),
        py::arg("blocks"), py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false, py::arg("fast_signal") = false);
  m.def("tp2_remote_push_wide", &tp2_remote_push_wide,
        "TP2 remote-only push processing multiple 16-byte packs per thread",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("vec_packs"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("tp2_remote_push_stream", &tp2_remote_push_stream,
        "TP2 remote-only push with per-pack push/poll/reduce streaming",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("tp2_remote_flag_push", &tp2_remote_flag_push,
        "TP2 remote-only push using a per-pack epoch flag instead of data sentinel",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("flag_ptrs"),
        py::arg("epoch_slots"), py::arg("output"), py::arg("numel"),
        py::arg("rank"), py::arg("world_size"), py::arg("max_blocks"),
        py::arg("blocks"), py::arg("threads"), py::arg("stream_mode"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("tp2_remote_push_window", &tp2_remote_push_window,
        "TP2 remote-only push with a small per-thread in-flight window",
        py::arg("input"), py::arg("tmp_ptrs"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("window_packs"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("push_oneshot_param", &push_oneshot_param,
        "TP allreduce one-shot push with peer pointers passed as launch params",
        py::arg("input"), py::arg("tmp_ptrs_host"), py::arg("epoch_slots"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("fast_oneshot", &fast_oneshot,
        "TP allreduce oneshot with CAR-style fast symmetric-memory barriers",
        py::arg("input_ptrs"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("twoshot", &twoshot,
        "TP allreduce twoshot over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("twostage_fast", &twostage_fast,
        "TP allreduce vLLM-style two-stage over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("tree", &tree, "TP allreduce tree over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("rsag", &rsag,
        "TP allreduce reduce-scatter/allgather over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_tree", &topo_tree,
        "TP8 topology-aware tree allreduce over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag", &topo_rsag,
        "TP8 topology-aware two-level RS/AG allreduce over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_tree_fastfinal", &topo_tree_fastfinal,
        "TP8 topology-aware tree allreduce with fast final barrier",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_fastfinal", &topo_rsag_fastfinal,
        "TP8 topology-aware RS/AG allreduce with fast final barriers",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined", &topo_rsag_pipelined,
        "TP8 topology-aware RS/AG allreduce with chunk-pipelined barriers",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_direct", &topo_rsag_pipelined_direct,
        "TP8 chunk-pipelined RS/AG with owner-directed ready/ack",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_crosspush", &topo_rsag_pipelined_crosspush,
        "TP8 chunk-pipelined RS/AG with owner-directed control and cross push",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_wide2", &topo_rsag_pipelined_wide<2>,
        "TP8 chunk-pipelined RS/AG with 2-pack vector groups",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_chunks8", &topo_rsag_pipelined_chunks8,
        "TP8 chunk-pipelined RS/AG with 8 chunks",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_gatherstart", &topo_rsag_pipelined_gatherstart,
        "TP8 chunk-pipelined RS/AG with owner-gather start barrier",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_ag_push", &topo_rsag_pipelined_ag_push,
        "TP8 chunk-pipelined RS/AG with island-local all-gather push",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("output_ptrs"),
        py::arg("signal_ptrs"), py::arg("output"), py::arg("numel"),
        py::arg("rank"), py::arg("world_size"), py::arg("max_blocks"),
        py::arg("blocks"), py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_pipelined_ag_push_split",
        &topo_rsag_pipelined_ag_push_split,
        "TP8 chunk-pipelined RS/AG with split island-local all-gather push",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("output_ptrs"),
        py::arg("signal_ptrs"), py::arg("output"), py::arg("numel"),
        py::arg("rank"), py::arg("world_size"), py::arg("max_blocks"),
        py::arg("blocks"), py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_push", &topo_rsag_push,
        "TP8 topology-aware RS/AG allreduce with push all-gather",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_wide2", &topo_rsag_wide<2>,
        "TP8 topology-aware RS/AG allreduce with 2-pack vector groups",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("topo_rsag_wide4", &topo_rsag_wide<4>,
        "TP8 topology-aware RS/AG allreduce with 4-pack vector groups",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("island_leader", &island_leader,
        "TP8 two-island leader allreduce over symmetric memory",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"), py::arg("blocks"),
        py::arg("threads"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
  m.def("multimem", &multimem,
        "TP allreduce multimem over symmetric memory",
        py::arg("input_mc_ptr"), py::arg("signal_ptrs"), py::arg("output"),
        py::arg("numel"), py::arg("rank"), py::arg("world_size"),
        py::arg("max_blocks"), py::arg("blocks"), py::arg("threads"),
        py::arg("pdl_sync") = false, py::arg("pdl_release") = false);
  m.def("adaptive_policy", &adaptive_policy,
        "Return (algorithm, blocks, threads) for the adaptive custom allreduce policy",
        py::arg("numel"), py::arg("hidden_size"), py::arg("world_size"),
        py::arg("max_blocks"));
  m.def("adaptive", &adaptive,
        "TP allreduce with adaptive algorithm/block/thread policy",
        py::arg("input_ptrs"), py::arg("tmp_ptrs"), py::arg("signal_ptrs"),
        py::arg("output"), py::arg("numel"), py::arg("rank"),
        py::arg("world_size"), py::arg("max_blocks"),
        py::arg("hidden_size"), py::arg("pdl_sync") = false,
        py::arg("pdl_release") = false);
}
