#include <metal_stdlib>
using namespace metal;

// Single-token GQA decode attention, flash-decoding style.
// Grid: (num_blocks, NUM_HEADS); 256 threads per threadgroup.
// Each simdgroup is split into 4 "slots" of 8 lanes; a slot owns one key token
// per iteration and each lane in it owns 8 of the 64 head dims. So one
// threadgroup scores 32 tokens at once and keeps an online softmax per slot,
// instead of looping over tokens serially.
// num_blocks == 1: writes the normalized result to output.
// num_blocks > 1 : writes (max, sum, unnormalized numerator) per block for
//                  gqa_attention_block_reduce.
kernel void gqa_decode_attention(device const float* q [[buffer(0)]],
                                 device const half* k_cache [[buffer(1)]],
                                 device const half* v_cache [[buffer(2)]],
                                 device half* output [[buffer(3)]],
                                 device float* block_maxima [[buffer(4)]],
                                 device float* block_sums [[buffer(5)]],
                                 device float* block_outputs [[buffer(6)]],
                                 constant uint& num_tokens [[buffer(7)]],
                                 constant uint& num_blocks [[buffer(8)]],
                                 constant uint& block_size [[buffer(9)]],
                                 threadgroup float* scratch [[threadgroup(0)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_width = 128;
    constexpr uint query_heads_per_kv_head = 7;
    constexpr uint slots = 32;
    constexpr float scale = 0.125f;

    uint block = group.x;
    uint head = group.y;
    uint kv_head = head / query_heads_per_kv_head;
    uint slot = simd_group * 4 + lane / 8;
    uint dim0 = (lane % 8) * 8;

    uint block_start = block * block_size;
    uint block_end = min(block_start + block_size, num_tokens);

    device const float4* q4 = (device const float4*)(q + head * head_dim + dim0);
    float4 qa = q4[0] * scale;
    float4 qb = q4[1] * scale;

    float m = -INFINITY;
    float l = 0.0f;
    float4 acc_a = 0.0f;
    float4 acc_b = 0.0f;

    for (uint token = block_start + slot; token < block_end; token += slots) {
        uint base = token * kv_width + kv_head * head_dim + dim0;
        device const half4* k4 = (device const half4*)(k_cache + base);
        float s = dot(qa, float4(k4[0])) + dot(qb, float4(k4[1]));
        s += simd_shuffle_xor(s, 1);
        s += simd_shuffle_xor(s, 2);
        s += simd_shuffle_xor(s, 4);

        float m_new = max(m, s);
        float correction = exp(m - m_new);
        float p = exp(s - m_new);
        device const half4* v4 = (device const half4*)(v_cache + base);
        acc_a = acc_a * correction + p * float4(v4[0]);
        acc_b = acc_b * correction + p * float4(v4[1]);
        l = l * correction + p;
        m = m_new;
    }

    // scratch layout: [slots] maxima, [slots] sums, [slots * head_dim] numerators
    threadgroup float* slot_max = scratch;
    threadgroup float* slot_sum = scratch + slots;
    threadgroup float* slot_acc = scratch + 2 * slots;
    if ((lane % 8) == 0) {
        slot_max[slot] = m;
        slot_sum[slot] = l;
    }
    threadgroup float4* acc_out = (threadgroup float4*)(slot_acc + slot * head_dim + dim0);
    acc_out[0] = acc_a;
    acc_out[1] = acc_b;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float maximum = -INFINITY;
        for (uint i = 0; i < slots; i++) maximum = max(maximum, slot_max[i]);
        float sum = 0.0f;
        float numerator = 0.0f;
        for (uint i = 0; i < slots; i++) {
            float c = exp(slot_max[i] - maximum);
            sum += slot_sum[i] * c;
            numerator += slot_acc[i * head_dim + tid] * c;
        }
        if (num_blocks == 1) {
            output[head * head_dim + tid] = half(numerator / sum);
        } else {
            uint block_index = head * num_blocks + block;
            block_outputs[block_index * head_dim + tid] = numerator;
            if (tid == 0) {
                block_maxima[block_index] = maximum;
                block_sums[block_index] = sum;
            }
        }
    }
}
