#include <metal_stdlib>
using namespace metal;

// One simdgroup computes ROWS output rows. Each lane reads 8 contiguous halfs
// (16 bytes) per row per step, so a simdgroup reads 512 contiguous bytes of a row.
template <uint ROWS, typename XT>
inline void gemv_body(device const half* W, device const XT* x, device half* y,
                      uint M, uint K, uint row0, uint lane) {
    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
    for (uint k = lane * 8; k < K; k += 256) {
        float4 xa = float4(*(device const vec<XT,4>*)(x + k));
        float4 xb = float4(*(device const vec<XT,4>*)(x + k + 4));
        for (uint r = 0; r < ROWS; r++) {
            device const half4* w = (device const half4*)(W + (size_t)(row0 + r) * K + k);
            acc[r] += dot(float4(w[0]), xa) + dot(float4(w[1]), xb);
        }
    }
    for (uint r = 0; r < ROWS; r++) {
        float s = simd_sum(acc[r]);
        if (lane == 0) y[row0 + r] = half(s);
    }
}

kernel void gemv_r4(device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
                    device half* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * nsg + sg) * 4;
    if (row0 >= M) return;
    gemv_body<4, half>(W, x, y, M, K, row0, lane);
}
kernel void gemv_r8(device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
                    device half* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * nsg + sg) * 8;
    if (row0 >= M) return;
    gemv_body<8, half>(W, x, y, M, K, row0, lane);
}
kernel void gemv_r2(device const half* W [[buffer(0)]], device const half* x [[buffer(1)]],
                    device half* y [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * nsg + sg) * 2;
    if (row0 >= M) return;
    gemv_body<2, half>(W, x, y, M, K, row0, lane);
}
