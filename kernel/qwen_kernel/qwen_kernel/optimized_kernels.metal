#include <metal_stdlib>
using namespace metal;

kernel void matvec_q8(device const char* weights [[buffer(0)]],
                      device const half* scales [[buffer(1)]],
                      device const half* x [[buffer(2)]],
                      device half* y [[buffer(3)]],
                      constant uint& K [[buffer(4)]],
                      threadgroup float* partial [[threadgroup(0)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    const uint scale_groups = K / 64;
    float sum = 0.0f;

    for (uint col = tid; col < K; col += threads) {
        float scale = float(scales[row * scale_groups + col / 64]);
        sum += scale * float(weights[row * K + col]) * float(x[col]);
    }

    sum = simd_sum(sum);
    if ((tid & 31u) == 0) {
        partial[tid / 32] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; i++) {
            total += partial[i];
        }
        y[row] = half(total);
    }
}

kernel void gate_up_q8(device const char* gate_weights [[buffer(0)]],
                       device const half* gate_scales [[buffer(1)]],
                       device const char* up_weights [[buffer(2)]],
                       device const half* up_scales [[buffer(3)]],
                       device const float* x [[buffer(4)]],
                       device float* gate_out [[buffer(5)]],
                       device float* up_out [[buffer(6)]],
                       constant uint& K [[buffer(7)]],
                       threadgroup float* partial [[threadgroup(0)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
    const uint scale_groups = K / 64;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint col = tid; col < K; col += threads) {
        uint scale_index = row * scale_groups + col / 64;
        float x_value = x[col];

        gate_sum += float(gate_scales[scale_index]) *
                    float(gate_weights[row * K + col]) * x_value;
        up_sum += float(up_scales[scale_index]) *
                  float(up_weights[row * K + col]) * x_value;
    }

    gate_sum = simd_sum(gate_sum);
    up_sum = simd_sum(up_sum);

    if ((tid & 31u) == 0) {
        uint warp = tid / 32;
        partial[warp * 2] = gate_sum;
        partial[warp * 2 + 1] = up_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gate_total = 0.0f;
        float up_total = 0.0f;
        for (uint i = 0; i < threads / 32; i++) {
            gate_total += partial[i * 2];
            up_total += partial[i * 2 + 1];
        }
        gate_out[row] = gate_total;
        up_out[row] = up_total;
    }
}

kernel void down_q8(device const char* weights [[buffer(0)]],
                    device const half* scales [[buffer(1)]],
                    device const float* x [[buffer(2)]],
                    device float* y [[buffer(3)]],
                    constant uint& K [[buffer(4)]],
                    threadgroup float* partial [[threadgroup(0)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint threads [[threads_per_threadgroup]]) {
    const uint scale_groups = K / 64;
    float sum = 0.0f;

    for (uint col = tid; col < K; col += threads) {
        float scale = float(scales[row * scale_groups + col / 64]);
        sum += scale * float(weights[row * K + col]) * x[col];
    }

    sum = simd_sum(sum);
    if ((tid & 31u) == 0) {
        partial[tid / 32] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; i++) {
            total += partial[i];
        }
        y[row] = total;
    }
}

kernel void qkv_fp16(device const half* q_weight [[buffer(0)]],
                     device const half* k_weight [[buffer(1)]],
                     device const half* v_weight [[buffer(2)]],
                     device const half* x [[buffer(3)]],
                     device half* q_out [[buffer(4)]],
                     device half* k_out [[buffer(5)]],
                     device half* v_out [[buffer(6)]],
                     device const half* q_bias [[buffer(7)]],
                     device const half* k_bias [[buffer(8)]],
                     device const half* v_bias [[buffer(9)]],
                     constant uint& K [[buffer(10)]],
                     threadgroup float* partial [[threadgroup(0)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]]) {
    device const half* weight;
    device half* output;
    device const half* bias;
    uint local_row = row;

    if (row < 896) {
        weight = q_weight;
        output = q_out;
        bias = q_bias;
    } else if (row < 1024) {
        local_row = row - 896;
        weight = k_weight;
        output = k_out;
        bias = k_bias;
    } else {
        local_row = row - 1024;
        weight = v_weight;
        output = v_out;
        bias = v_bias;
    }

    device const half4* weight4 = (device const half4*)(weight + local_row * K);
    device const half4* x4 = (device const half4*)x;
    uint K4 = K / 4;
    float sum = 0.0f;

    for (uint j = tid; j < K4; j += threads) {
        sum += dot(float4(weight4[j]), float4(x4[j]));
    }

    sum = simd_sum(sum);
    if ((tid & 31u) == 0) {
        partial[tid / 32] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; i++) {
            total += partial[i];
        }
        output[local_row] = half(total + float(bias[local_row]));
    }
}

kernel void rope_qkv_cache_append(device const half* q [[buffer(0)]],
                                  device half* k [[buffer(1)]],
                                  device const half* v [[buffer(2)]],
                                  device float* q_rotated [[buffer(3)]],
                                  device half* k_cache [[buffer(4)]],
                                  device half* v_cache [[buffer(5)]],
                                  constant uint& pos [[buffer(6)]],
                                  uint id [[thread_position_in_grid]]) {
    constexpr uint half_head_dim = 32;
    constexpr uint num_q_heads = 14;
    constexpr uint num_kv_heads = 2;
    constexpr uint kv_dim = 128;
    constexpr uint rotate_items = (num_q_heads + num_kv_heads) * half_head_dim;

    if (id < kv_dim) {
        uint offset = pos * kv_dim + id;
        v_cache[offset] = v[id];
    }

    if (id >= rotate_items) {
        return;
    }

    bool is_q = id < num_q_heads * half_head_dim;
    uint local_id = is_q ? id : id - num_q_heads * half_head_dim;
    uint head = local_id / half_head_dim;
    uint dim = local_id % half_head_dim;
    uint base = head * 64;

    float angle = float(pos) * pow(1000000.0f, -float(2 * dim) / 64.0f);
    float c = cos(angle);
    float s = sin(angle);

    float x0 = is_q ? float(q[base + dim]) : float(k[base + dim]);
    float x1 = is_q ? float(q[base + dim + half_head_dim])
                    : float(k[base + dim + half_head_dim]);

    float y0 = x0 * c - x1 * s;
    float y1 = x1 * c + x0 * s;

    if (is_q) {
        q_rotated[base + dim] = y0;
        q_rotated[base + dim + half_head_dim] = y1;
    } else {
        uint cache_base = pos * kv_dim + base;
        half h0 = half(y0);
        half h1 = half(y1);
        k[base + dim] = h0;
        k[base + dim + half_head_dim] = h1;
        k_cache[cache_base + dim] = h0;
        k_cache[cache_base + dim + half_head_dim] = h1;
    }
}

kernel void gqa_attention_fused(device const float* q [[buffer(0)]],
                                device const half* k_cache [[buffer(1)]],
                                device const half* v_cache [[buffer(2)]],
                                device half* output [[buffer(3)]],
                                constant uint& num_tokens [[buffer(4)]],
                                threadgroup float* scratch [[threadgroup(0)]],
                                uint head [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_width = 128;
    constexpr uint query_heads_per_kv_head = 7;
    constexpr float scale = 0.125f;

    uint lane = tid & 31u;
    uint simd_group = tid >> 5;
    uint simd_groups = threads >> 5;
    uint kv_head = head / query_heads_per_kv_head;
    uint q_base = head * head_dim;
    threadgroup float* partial = scratch + num_tokens;

    for (uint token = simd_group; token < num_tokens; token += simd_groups) {
        uint k_base = token * kv_width + kv_head * head_dim;
        float dot_value =
            q[q_base + lane] * float(k_cache[k_base + lane]) +
            q[q_base + lane + 32] * float(k_cache[k_base + lane + 32]);
        dot_value = simd_sum(dot_value);
        if (lane == 0) {
            scratch[token] = dot_value * scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint token = tid; token < num_tokens; token += threads) {
        local_max = max(local_max, scratch[token]);
    }
    local_max = simd_max(local_max);
    if (lane == 0) {
        partial[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maximum = partial[0];
        for (uint group = 1; group < simd_groups; group++) {
            maximum = max(maximum, partial[group]);
        }
        partial[0] = maximum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = partial[0];
    float local_sum = 0.0f;
    for (uint token = tid; token < num_tokens; token += threads) {
        float value = exp(scratch[token] - maximum);
        scratch[token] = value;
        local_sum += value;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) {
        partial[simd_group] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sum = 0.0f;
        for (uint group = 0; group < simd_groups; group++) {
            sum += partial[group];
        }
        partial[0] = 1.0f / sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inverse = partial[0];
    for (uint token = tid; token < num_tokens; token += threads) {
        scratch[token] *= inverse;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float weighted_sum = 0.0f;
        for (uint token = 0; token < num_tokens; token++) {
            uint v_index = token * kv_width + kv_head * head_dim + tid;
            weighted_sum += scratch[token] * float(v_cache[v_index]);
        }
        output[q_base + tid] = half(weighted_sum);
    }
}

kernel void gqa_attention_block(device const float* q [[buffer(0)]],
                                device const half* k_cache [[buffer(1)]],
                                device const half* v_cache [[buffer(2)]],
                                device float* block_maxima [[buffer(3)]],
                                device float* block_sums [[buffer(4)]],
                                device float* block_outputs [[buffer(5)]],
                                constant uint& num_tokens [[buffer(6)]],
                                constant uint& num_blocks [[buffer(7)]],
                                threadgroup float* scratch [[threadgroup(0)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint2 local [[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_width = 128;
    constexpr uint block_size = 256;
    constexpr uint simd_groups = 8;
    constexpr uint query_heads_per_kv_head = 7;
    constexpr float scale = 0.125f;

    uint block = group.x;
    uint head = group.y;
    uint tid = local.x;
    uint lane = tid & 31u;
    uint simd_group = tid >> 5;
    uint block_start = block * block_size;
    uint block_end = min(block_start + block_size, num_tokens);
    uint block_tokens = block_end - block_start;
    uint kv_head = head / query_heads_per_kv_head;
    uint q_base = head * head_dim;
    threadgroup float* partial = scratch + block_size;

    for (uint local_token = simd_group;
         local_token < block_tokens;
         local_token += simd_groups) {
        uint token = block_start + local_token;
        uint k_base = token * kv_width + kv_head * head_dim;
        float dot_value =
            q[q_base + lane] * float(k_cache[k_base + lane]) +
            q[q_base + lane + 32] * float(k_cache[k_base + lane + 32]);
        dot_value = simd_sum(dot_value);
        if (lane == 0) {
            scratch[local_token] = dot_value * scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = tid < block_tokens ? scratch[tid] : -INFINITY;
    local_max = simd_max(local_max);
    if (lane == 0) {
        partial[simd_group] = local_max;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float maximum = partial[0];
        for (uint index = 1; index < simd_groups; index++) {
            maximum = max(maximum, partial[index]);
        }
        partial[0] = maximum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maximum = partial[0];
    float local_sum = 0.0f;
    if (tid < block_tokens) {
        float value = exp(scratch[tid] - maximum);
        scratch[tid] = value;
        local_sum = value;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) {
        partial[simd_group] = local_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sum = 0.0f;
        for (uint index = 0; index < simd_groups; index++) {
            sum += partial[index];
        }
        uint block_index = head * num_blocks + block;
        block_maxima[block_index] = maximum;
        block_sums[block_index] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        float numerator = 0.0f;
        for (uint local_token = 0; local_token < block_tokens; local_token++) {
            uint token = block_start + local_token;
            uint v_index = token * kv_width + kv_head * head_dim + tid;
            numerator += scratch[local_token] * float(v_cache[v_index]);
        }
        uint output_index = (head * num_blocks + block) * head_dim + tid;
        block_outputs[output_index] = numerator;
    }
}

kernel void gqa_attention_block_grouped(device const float* q [[buffer(0)]],
                                        device const half* k_cache [[buffer(1)]],
                                        device const half* v_cache [[buffer(2)]],
                                        device float* block_maxima [[buffer(3)]],
                                        device float* block_sums [[buffer(4)]],
                                        device float* block_outputs [[buffer(5)]],
                                        constant uint& num_tokens [[buffer(6)]],
                                        constant uint& num_blocks [[buffer(7)]],
                                        threadgroup float* scratch [[threadgroup(0)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint2 local [[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_width = 128;
    constexpr uint block_size = 256;
    constexpr uint simd_groups = 8;
    constexpr uint query_heads_per_kv_head = 7;
    constexpr uint query_heads_per_group = 2;
    constexpr uint groups_per_kv_head = 4;
    constexpr float scale = 0.125f;

    uint block = group.x;
    uint kv_head = group.y / groups_per_kv_head;
    uint head_group = group.y % groups_per_kv_head;
    uint tid = local.x;
    uint lane = tid & 31u;
    uint simd_group = tid >> 5;
    uint block_start = block * block_size;
    uint block_end = min(block_start + block_size, num_tokens);
    uint block_tokens = block_end - block_start;
    uint first_kv_head = kv_head * query_heads_per_kv_head;
    uint first_head = first_kv_head + head_group * query_heads_per_group;
    uint query_count = min(query_heads_per_group,
                           first_kv_head + query_heads_per_kv_head - first_head);
    threadgroup float* partial = scratch + query_heads_per_group * block_size;

    for (uint local_token = simd_group;
         local_token < block_tokens;
         local_token += simd_groups) {
        uint token = block_start + local_token;
        uint k_base = token * kv_width + kv_head * head_dim;
        float key_low = float(k_cache[k_base + lane]);
        float key_high = float(k_cache[k_base + lane + 32]);
        for (uint query = 0; query < query_count; query++) {
            uint q_base = (first_head + query) * head_dim;
            float dot_value =
                q[q_base + lane] * key_low +
                q[q_base + lane + 32] * key_high;
            dot_value = simd_sum(dot_value);
            if (lane == 0) {
                scratch[query * block_size + local_token] = dot_value * scale;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint query = 0; query < query_count; query++) {
        threadgroup float* scores = scratch + query * block_size;
        float local_max = tid < block_tokens ? scores[tid] : -INFINITY;
        local_max = simd_max(local_max);
        if (lane == 0) {
            partial[simd_group] = local_max;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float maximum = partial[0];
            for (uint index = 1; index < simd_groups; index++) {
                maximum = max(maximum, partial[index]);
            }
            partial[0] = maximum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float maximum = partial[0];
        float local_sum = 0.0f;
        if (tid < block_tokens) {
            float value = exp(scores[tid] - maximum);
            scores[tid] = value;
            local_sum = value;
        }
        local_sum = simd_sum(local_sum);
        if (lane == 0) {
            partial[simd_group] = local_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float sum = 0.0f;
            for (uint index = 0; index < simd_groups; index++) {
                sum += partial[index];
            }
            uint head = first_head + query;
            uint block_index = head * num_blocks + block;
            block_maxima[block_index] = maximum;
            block_sums[block_index] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < head_dim) {
        float numerators[query_heads_per_group];
        for (uint query = 0; query < query_count; query++) {
            numerators[query] = 0.0f;
        }
        for (uint local_token = 0; local_token < block_tokens; local_token++) {
            uint token = block_start + local_token;
            uint v_index = token * kv_width + kv_head * head_dim + tid;
            float value = float(v_cache[v_index]);
            for (uint query = 0; query < query_count; query++) {
                numerators[query] += scratch[query * block_size + local_token] * value;
            }
        }
        for (uint query = 0; query < query_count; query++) {
            uint head = first_head + query;
            uint output_index = (head * num_blocks + block) * head_dim + tid;
            block_outputs[output_index] = numerators[query];
        }
    }
}

kernel void gqa_attention_block_reduce(device const float* block_maxima [[buffer(0)]],
                                       device const float* block_sums [[buffer(1)]],
                                       device const float* block_outputs [[buffer(2)]],
                                       device half* output [[buffer(3)]],
                                       constant uint& num_blocks [[buffer(4)]],
                                       uint head [[threadgroup_position_in_grid]],
                                       uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = 64;
    uint block_base = head * num_blocks;
    float maximum = -INFINITY;
    for (uint block = 0; block < num_blocks; block++) {
        maximum = max(maximum, block_maxima[block_base + block]);
    }

    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint block = 0; block < num_blocks; block++) {
        uint block_index = block_base + block;
        float correction = exp(block_maxima[block_index] - maximum);
        denominator += block_sums[block_index] * correction;
        numerator += block_outputs[block_index * head_dim + tid] * correction;
    }
    output[head * head_dim + tid] = half(numerator / denominator);
}

kernel void split_qkv_bias_batch(device const half* combined [[buffer(0)]],
                                 device const half* q_bias [[buffer(1)]],
                                 device const half* k_bias [[buffer(2)]],
                                 device const half* v_bias [[buffer(3)]],
                                 device half* q [[buffer(4)]],
                                 device half* k [[buffer(5)]],
                                 device half* v [[buffer(6)]],
                                 constant uint& token_count [[buffer(7)]],
                                 uint id [[thread_position_in_grid]]) {
    constexpr uint q_width = 896;
    constexpr uint kv_width = 128;
    constexpr uint combined_width = q_width + 2 * kv_width;
    uint total = token_count * combined_width;
    if (id >= total) {
        return;
    }

    uint token = id / combined_width;
    uint column = id % combined_width;
    if (column < q_width) {
        q[token * q_width + column] = half(
            float(combined[id]) + float(q_bias[column]));
    } else if (column < q_width + kv_width) {
        uint kv_column = column - q_width;
        k[token * kv_width + kv_column] = half(
            float(combined[id]) + float(k_bias[kv_column]));
    } else {
        uint kv_column = column - q_width - kv_width;
        v[token * kv_width + kv_column] = half(
            float(combined[id]) + float(v_bias[kv_column]));
    }
}

kernel void gate_up_silu_mul_batch(device const half* combined [[buffer(0)]],
                                   device half* output [[buffer(1)]],
                                   constant uint& token_count [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
    constexpr uint intermediate = 4864;
    constexpr uint combined_width = 2 * intermediate;
    uint total = token_count * intermediate;
    if (id >= total) {
        return;
    }

    uint token = id / intermediate;
    uint column = id % intermediate;
    uint base = token * combined_width;
    float gate = float(combined[base + column]);
    float up = float(combined[base + intermediate + column]);
    output[id] = half((gate / (1.0f + exp(-gate))) * up);
}

kernel void gqa_tiled_prefill(device const float* q [[buffer(0)]],
                              device const half* k_cache [[buffer(1)]],
                              device const half* v_cache [[buffer(2)]],
                              device half* output [[buffer(3)]],
                              constant uint& token_count [[buffer(4)]],
                              constant uint& start_pos [[buffer(5)]],
                              threadgroup float* scratch [[threadgroup(0)]],
                              threadgroup half* value_tile [[threadgroup(1)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint2 local [[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_width = 128;
    constexpr uint query_tile_size = 8;
    constexpr uint key_tile_size = 32;
    constexpr float scale = 0.125f;

    uint query_tile = group.x;
    uint head = group.y;
    uint tid = local.x;
    uint simd_group = tid >> 5;
    uint lane = tid & 31u;
    uint first_query = query_tile * query_tile_size;
    uint kv_head = head / 7;
    threadgroup float* query_values = scratch;
    threadgroup float* scores = query_values + query_tile_size * head_dim;
    threadgroup float* block_output = scores + query_tile_size * key_tile_size;
    threadgroup float* accumulator = block_output + query_tile_size * head_dim;
    threadgroup float* maxima = accumulator + query_tile_size * head_dim;
    threadgroup float* denominators = maxima + query_tile_size;
    threadgroup float* old_corrections = denominators + query_tile_size;
    threadgroup float* block_corrections = old_corrections + query_tile_size;

    for (uint index = tid; index < query_tile_size * head_dim; index += 256) {
        uint row = index / head_dim;
        uint dimension = index % head_dim;
        uint query = first_query + row;
        query_values[index] = query < token_count
            ? q[query * 896 + head * head_dim + dimension]
            : 0.0f;
        accumulator[index] = 0.0f;
    }
    if (tid < query_tile_size) {
        maxima[tid] = -INFINITY;
        denominators[tid] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint query = query_tile * query_tile_size + simd_group;
    bool active = query < token_count;
    uint key_count = active ? start_pos + query + 1 : 0;
    uint last_query = min((query_tile + 1) * query_tile_size, token_count) - 1;
    uint group_key_count = start_pos + last_query + 1;

    for (uint block_start = 0; block_start < group_key_count; block_start += key_tile_size) {
        uint block_tokens = min(key_tile_size, group_key_count - block_start);
        if (simd_group < 4) {
            simdgroup_float8x8 result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint dimension = 0; dimension < head_dim; dimension += 8) {
                simdgroup_float8x8 query_matrix;
                simdgroup_half8x8 key_matrix;
                simdgroup_load(query_matrix, query_values, head_dim,
                               ulong2(dimension, 0), false);
                simdgroup_load(key_matrix, k_cache, kv_width,
                               ulong2(kv_head * head_dim + dimension,
                                      block_start + simd_group * 8), true);
                simdgroup_multiply_accumulate(
                    result, query_matrix, key_matrix, result);
            }
            simdgroup_store(result, scores, key_tile_size,
                            ulong2(simd_group * 8, 0), false);
        } else {
            uint loader = tid - 128;
            uint tile_elements = key_tile_size * head_dim;
            for (uint index = loader; index < tile_elements; index += 128) {
                uint token = index / head_dim;
                uint dimension = index % head_dim;
                uint cache_index =
                    (block_start + token) * kv_width + kv_head * head_dim + dimension;
                value_tile[index] = token < block_tokens
                    ? v_cache[cache_index]
                    : half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool valid = active && lane < block_tokens && block_start + lane < key_count;
        float score = valid
            ? scores[simd_group * key_tile_size + lane] * scale
            : -INFINITY;
        float block_maximum = simd_broadcast_first(simd_max(score));
        float weight = valid ? exp(score - block_maximum) : 0.0f;
        scores[simd_group * key_tile_size + lane] = weight;
        float block_sum = simd_broadcast_first(simd_sum(weight));
        if (lane == 0) {
            float old_maximum = maxima[simd_group];
            float next_maximum = max(old_maximum, block_maximum);
            float old_correction = old_maximum == -INFINITY
                ? 0.0f
                : exp(old_maximum - next_maximum);
            float block_correction = block_maximum == -INFINITY
                ? 0.0f
                : exp(block_maximum - next_maximum);
            maxima[simd_group] = next_maximum;
            denominators[simd_group] =
                denominators[simd_group] * old_correction + block_sum * block_correction;
            old_corrections[simd_group] = old_correction;
            block_corrections[simd_group] = block_correction;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint key = 0; key < key_tile_size; key += 8) {
            simdgroup_float8x8 weight_matrix;
            simdgroup_half8x8 value_matrix;
            simdgroup_load(weight_matrix, scores, key_tile_size,
                           ulong2(key, 0), false);
            simdgroup_load(value_matrix, value_tile, head_dim,
                           ulong2(simd_group * 8, key), false);
            simdgroup_multiply_accumulate(
                result, weight_matrix, value_matrix, result);
        }
        simdgroup_store(result, block_output, head_dim,
                        ulong2(simd_group * 8, 0), false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid; index < query_tile_size * head_dim; index += 256) {
            uint row = index / head_dim;
            accumulator[index] = accumulator[index] * old_corrections[row] +
                block_output[index] * block_corrections[row];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = tid; index < query_tile_size * head_dim; index += 256) {
        uint row = index / head_dim;
        uint dimension = index % head_dim;
        uint output_query = first_query + row;
        if (output_query < token_count) {
            output[output_query * 896 + head * head_dim + dimension] =
                half(accumulator[index] / denominators[row]);
        }
    }
}

kernel void argmax_stage1(device const half* logits [[buffer(0)]],
                          device float* block_values [[buffer(1)]],
                          device uint* block_ids [[buffer(2)]],
                          constant uint& vocab_size [[buffer(3)]],
                          threadgroup float* values [[threadgroup(0)]],
                          threadgroup uint* ids [[threadgroup(1)]],
                          uint block [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    uint index = block * threads + tid;
    float value = index < vocab_size ? float(logits[index]) : -INFINITY;
    values[tid] = value;
    ids[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            uint other_id = ids[tid + stride];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_id < ids[tid])) {
                values[tid] = other_value;
                ids[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        block_values[block] = values[0];
        block_ids[block] = ids[0];
    }
}

kernel void argmax_stage2(device const float* block_values [[buffer(0)]],
                          device const uint* block_ids [[buffer(1)]],
                          device uint* selected [[buffer(2)]],
                          constant uint& block_count [[buffer(3)]],
                          threadgroup float* values [[threadgroup(0)]],
                          threadgroup uint* ids [[threadgroup(1)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    float best_value = -INFINITY;
    uint best_id = 0;

    for (uint index = tid; index < block_count; index += threads) {
        float value = block_values[index];
        uint token_id = block_ids[index];
        if (value > best_value || (value == best_value && token_id < best_id)) {
            best_value = value;
            best_id = token_id;
        }
    }

    values[tid] = best_value;
    ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_value = values[tid + stride];
            uint other_id = ids[tid + stride];
            if (other_value > values[tid] ||
                (other_value == values[tid] && other_id < ids[tid])) {
                values[tid] = other_value;
                ids[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        selected[0] = ids[0];
    }
}
