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
