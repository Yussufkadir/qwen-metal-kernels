#include <metal_stdlib>
using namespace metal;

kernel void matvec_gate_up_batched(
    device const half*  gate_weights [[ buffer(0) ]],
    device const half*  up_weights   [[ buffer(1) ]],
    device const float* inputs       [[ buffer(2) ]],
    device float*       gate_outputs [[ buffer(3) ]],
    device float*       up_outputs   [[ buffer(4) ]],
    constant uint&      K            [[ buffer(5) ]],
    threadgroup float*  partial      [[ threadgroup(0) ]],
    uint3 gid  [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 tg_size    [[ threads_per_threadgroup ]])
{
    const uint M = 4864;
    
    uint row = gid.x;
    uint batch = gid.y;
    uint local_id = lid.x;
    uint gsize = tg_size.x;
    
    uint batch_w_offset = batch * M * K;
    uint batch_x_offset = batch * K;
    uint batch_y_offset = batch * M;
    
    device const half4* gate4 = (device const half4*)(gate_weights + batch_w_offset + row * K);
    device const half4* up4 = (device const half4*)(up_weights + batch_w_offset + row * K);
    device const float4* x4 = (device const float4*)(inputs + batch_x_offset);
    uint K4 = K / 4;
    
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    
    for (uint j = local_id; j < K4; j += gsize * 2){
        float4 g = float4(gate4[j]);
        float4 v = x4[j];
        gate_sum += dot(g, v);
        
        float4 u = float4(up4[j]);
        up_sum += dot(u, v);
        
        uint j2 = j + gsize;
        if (j2 < K4) {
            float4 g2 = float4(gate4[j2]);
            float4 v2 = x4[j2];
            gate_sum += dot(g2, v2);
            
            float4 u2 = float4(up4[j2]);
            up_sum += dot(u2, v2);
        }
    }
    
    gate_sum = simd_sum(gate_sum);
    if (local_id % 32 == 0){
        partial[(local_id / 32) * 2] = gate_sum;
    }
    
    up_sum = simd_sum(up_sum);
    if (local_id % 32 == 0) {
        partial[(local_id / 32) * 2 + 1] = up_sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (local_id == 0){
        float total_gate = 0.0f;
        float total_up = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) {
            total_gate += partial[i * 2];
            total_up += partial[i * 2 + 1];
        }
        gate_outputs[batch_y_offset + row] = total_gate;
        up_outputs[batch_y_offset + row] = total_up;
    }
}

kernel void matvec_down_batched(
                                device const half* down_weights [[ buffer(0) ]],
                                device const float* inputs [[ buffer(1) ]],
                                device float* outputs [[ buffer(2) ]],
                                constant uint& K [[ buffer(3)]],
                                threadgroup float* partial [[ threadgroup(0)]],
                                uint3 gid [[ threadgroup_position_in_grid ]],
                                uint3 lid [[ thread_position_in_threadgroup ]],
                                uint3 tg_size [[ threads_per_threadgroup ]]
                                )
{
    const uint M = 896;
    
    uint row = gid.x;
    uint batch = gid.y;
    uint local_id = lid.x;
    uint gsize = tg_size.x;
    
    uint batch_w_offset = batch * M * K;
    uint batch_x_offset = batch * K;
    uint batch_y_offset = batch * M;
    
    device const half4* down4 = (device const half4*)(down_weights + batch_w_offset + row * K);
    device const float4* x4 = (device const float4*)(inputs + batch_x_offset);
    uint K4 = K / 4;
    
    float sum = 0.0f;
    for (uint j = local_id; j < K4; j += gsize){
        float4 d = float4(down4[j]);
        float4 v = x4[j];
        sum += dot(d, v);
    }
    
    sum = simd_sum(sum);
    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) total += partial[i];
        outputs[batch_y_offset + row] = total;
    }
}

kernel void rms_norm(
    device const half*  x       [[ buffer(0) ]],
    device const half*  weight  [[ buffer(1) ]],
    device half*        y       [[ buffer(2) ]],
    constant uint&      D       [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    for (uint i = 0; i < D; i++) sum += float(x[i]) * float(x[i]);
    float rms = sqrt(sum / (float)D + 1e-6f);
    y[id] = half(float(x[id]) * float(weight[id]) / rms);
}

kernel void silu_inplace(device half* x [[ buffer(0) ]], uint id [[ thread_position_in_grid ]]) {
    float v = float(x[id]);
    x[id] = half(v / (1.0f + exp(-v)));
}

kernel void element_mul(
    device const half* a [[ buffer(0) ]],
    device const half* b [[ buffer(1) ]],
    device half* c       [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{
    c[id] = half(float(a[id]) * float(b[id]));
}

kernel void residual_add(
    device half* x       [[ buffer(0) ]],
    device const half* r [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]])
{
    x[id] = half(float(x[id]) + float(r[id]));
}

kernel void matvec_float4(
    device const half*  A       [[ buffer(0) ]],
    device const half*  x       [[ buffer(1) ]],
    device half*        y       [[ buffer(2) ]],
    constant uint&      K       [[ buffer(3) ]],
    threadgroup float*  partial [[ threadgroup(0) ]],
    uint row      [[ threadgroup_position_in_grid ]],
    uint local_id [[ thread_position_in_threadgroup ]],
    uint gsize    [[ threads_per_threadgroup ]])
{
    device const half4* A4 = (device const half4*)(A + row * K);
    device const half4* x4 = (device const half4*)x;
    uint K4 = K / 4;

    float sum = 0.0f;
    for (uint j = local_id; j < K4; j += gsize) {
        float4 a = float4(A4[j]);
        float4 v = float4(x4[j]);
        sum += dot(a, v);
    }

    // SIMD reduction
    sum = simd_sum(sum);

    // Write SIMD result to threadgroup memory
    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction by thread 0
    if (local_id == 0) {
        float total = 0.0f;
        uint n_simds = gsize / 32;
        for (uint i = 0; i < n_simds; i++) {
            total += partial[i];
        }
        y[row] = half(total);
    }
}

kernel void attention_scores(
    device const half* q       [[ buffer(0) ]],  // [1, head_dim] (single query)
    device const half* k       [[ buffer(1) ]],  // [num_tokens, head_dim]
    device float* scores       [[ buffer(2) ]],  // [num_tokens]
    constant uint& num_tokens  [[ buffer(3) ]],
    constant uint& head_dim    [[ buffer(4) ]],
    uint tid [[ thread_position_in_grid ]])
{
    if (tid >= num_tokens) return;
    
    float dot = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        dot += float(q[d]) * float(k[tid * head_dim + d]);
    }
    scores[tid] = dot / sqrt(float(head_dim));
}

// ── Softmax in-place ───────────────────────────────
kernel void softmax_inplace(
    device float* x [[ buffer(0) ]],
    constant uint& N [[ buffer(1) ]],
    uint tid [[ thread_position_in_grid ]])
{
    if (tid != 0) return;
    if (N == 0) return;
    
    float max_val = x[0];
    for (uint i = 1; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < N; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    if (sum > 0.0f) {
        for (uint i = 0; i < N; i++) x[i] /= sum;
    }
}
// ── Weighted sum: output = softmax_scores · V ──────
// Grid: (head_dim, 1, 1)
kernel void attention_weighted_sum(
    device const float* scores  [[ buffer(0) ]],  // [num_tokens]
    device const half*  v       [[ buffer(1) ]],  // [num_tokens, head_dim]
    device half*        out     [[ buffer(2) ]],  // [head_dim]
    constant uint& num_tokens   [[ buffer(3) ]],
    constant uint& head_dim     [[ buffer(4) ]],
    uint d [[ thread_position_in_grid ]])
{
    if (d >= head_dim) return;
    
    float sum = 0.0f;
    for (uint t = 0; t < num_tokens; t++) {
        sum += scores[t] * float(v[t * head_dim + d]);
    }
    out[d] = half(sum);
}
