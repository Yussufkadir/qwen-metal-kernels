#include <metal_stdlib>
using namespace metal;

kernel void matvec_q4(
    device const uchar* packed [[ buffer(0) ]],
    device const half* scales [[ buffer(1) ]],
    device const half* x [[ buffer(2) ]],
    device half* y [[ buffer(3) ]],
    constant uint& K [[ buffer(4) ]],
    threadgroup float* partial [[ threadgroup(0) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint threads [[ threads_per_threadgroup ]])
{
    uint bytes_per_row = K / 2;
    uint groups_per_row = K / 64;
    float sum = 0.0f;
    for (uint b = tid; b < bytes_per_row; b += threads) {
        uchar pair = packed[row * bytes_per_row + b];
        float scale = float(scales[row * groups_per_row + b / 32]);
        int q0 = int(pair & 15) - 8;
        int q1 = int(pair >> 4) - 8;
        sum += scale * (float(q0) * float(x[2*b]) + float(q1) * float(x[2*b+1]));
    }
    sum = simd_sum(sum);
    if ((tid & 31u) == 0) partial[tid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; i++) total += partial[i];
        y[row] = half(total);
    }
}

kernel void gate_up_q4(
    device const uchar* gate [[ buffer(0) ]],
    device const half* gate_scales [[ buffer(1) ]],
    device const uchar* up [[ buffer(2) ]],
    device const half* up_scales [[ buffer(3) ]],
    device const float* x [[ buffer(4) ]],
    device float* gate_out [[ buffer(5) ]],
    device float* up_out [[ buffer(6) ]],
    constant uint& K [[ buffer(7) ]],
    threadgroup float* partial [[ threadgroup(0) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint threads [[ threads_per_threadgroup ]])
{
    uint bytes_per_row = K / 2, groups_per_row = K / 64;
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (uint b = tid; b < bytes_per_row; b += threads) {
        uchar gp = gate[row * bytes_per_row + b];
        uchar upv = up[row * bytes_per_row + b];
        uint group = row * groups_per_row + b / 32;
        float x0 = x[2*b], x1 = x[2*b+1];
        gate_sum += float(gate_scales[group]) *
                    (float(int(gp & 15)-8) * x0 + float(int(gp >> 4)-8) * x1);
        up_sum += float(up_scales[group]) *
                  (float(int(upv & 15)-8) * x0 + float(int(upv >> 4)-8) * x1);
    }
    gate_sum = simd_sum(gate_sum); up_sum = simd_sum(up_sum);
    if ((tid & 31u) == 0) {
        partial[(tid / 32)*2] = gate_sum;
        partial[(tid / 32)*2+1] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float gs=0.0f, us=0.0f;
        for (uint i=0; i<threads/32; i++) { gs+=partial[i*2]; us+=partial[i*2+1]; }
        gate_out[row]=gs; up_out[row]=us;
    }
}

kernel void down_q4(
    device const uchar* packed [[ buffer(0) ]],
    device const half* scales [[ buffer(1) ]],
    device const float* x [[ buffer(2) ]],
    device float* y [[ buffer(3) ]],
    constant uint& K [[ buffer(4) ]],
    threadgroup float* partial [[ threadgroup(0) ]],
    uint row [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint threads [[ threads_per_threadgroup ]])
{
    uint bytes_per_row=K/2, groups_per_row=K/64;
    float sum=0.0f;
    for (uint b=tid; b<bytes_per_row; b+=threads) {
        uchar pair=packed[row*bytes_per_row+b];
        float scale=float(scales[row*groups_per_row+b/32]);
        sum += scale * (float(int(pair&15)-8)*x[2*b] + float(int(pair>>4)-8)*x[2*b+1]);
    }
    sum=simd_sum(sum);
    if ((tid&31u)==0) partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid==0) { float total=0.0f; for(uint i=0;i<threads/32;i++) total+=partial[i]; y[row]=total; }
}

kernel void rope_qk_inplace(
    device const half* q [[ buffer(0) ]],
    device half* k [[ buffer(1) ]],
    device float* q_rotated [[ buffer(2) ]],
    constant uint& position [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    constexpr uint half_dim = 32;
    constexpr uint q_heads = 14;
    constexpr uint kv_heads = 2;
    if (id >= (q_heads + kv_heads) * half_dim) return;
    bool is_q = id < q_heads * half_dim;
    uint local = is_q ? id : id - q_heads * half_dim;
    uint head = local / half_dim;
    uint i = local % half_dim;
    uint base = head * 64;
    float angle = float(position) * pow(1000000.0f, -float(2 * i) / 64.0f);
    float c = cos(angle), s = sin(angle);
    float a = is_q ? float(q[base + i]) : float(k[base + i]);
    float b = is_q ? float(q[base + i + half_dim]) : float(k[base + i + half_dim]);
    if (is_q) {
        q_rotated[base + i] = a * c - b * s;
        q_rotated[base + i + half_dim] = b * c + a * s;
    } else {
        k[base + i] = half(a * c - b * s);
        k[base + i + half_dim] = half(b * c + a * s);
    }
}

kernel void kv_cache_append(
    device const half* k [[ buffer(0) ]],
    device const half* v [[ buffer(1) ]],
    device half* k_cache [[ buffer(2) ]],
    device half* v_cache [[ buffer(3) ]],
    constant uint& position [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]])
{
    constexpr uint kv_width = 128;
    if (id >= kv_width) return;
    uint offset = position * kv_width + id;
    k_cache[offset] = k[id];
    v_cache[offset] = v[id];
}

kernel void gqa_attention_scores(
    device const float* q [[ buffer(0) ]],
    device const half* k_cache [[ buffer(1) ]],
    device float* scores [[ buffer(2) ]],
    constant uint& num_tokens [[ buffer(3) ]],
    uint3 group [[ threadgroup_position_in_grid ]],
    uint3 local [[ thread_position_in_threadgroup ]])
{
    uint token = group.x, head = group.y;
    uint lane = local.x;
    if (token >= num_tokens || head >= 14) return;
    uint kv_head = head / 7;
    float dot_value = 0.0f;
    for (uint d = lane; d < 64; d += 32)
        dot_value += q[head * 64 + d] *
                     float(k_cache[token * 128 + kv_head * 64 + d]);
    dot_value = simd_sum(dot_value);
    if (lane == 0) scores[head * num_tokens + token] = dot_value * 0.125f;
}

kernel void gqa_softmax(
    device float* scores [[ buffer(0) ]],
    constant uint& num_tokens [[ buffer(1) ]],
    threadgroup float* partial [[ threadgroup(0) ]],
    uint head [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint threads [[ threads_per_threadgroup ]])
{
    if (head >= 14 || num_tokens == 0) return;
    uint base = head * num_tokens;
    float local_max = -INFINITY;
    for (uint t = tid; t < num_tokens; t += threads)
        local_max = max(local_max, scores[base + t]);
    local_max = simd_max(local_max);
    if ((tid & 31u) == 0) partial[tid / 32] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float maximum = partial[0];
        for (uint i = 1; i < threads / 32; i++) maximum = max(maximum, partial[i]);
        partial[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = partial[0];

    float local_sum = 0.0f;
    for (uint t = tid; t < num_tokens; t += threads) {
        float value = exp(scores[base + t] - maximum);
        scores[base + t] = value;
        local_sum += value;
    }
    local_sum = simd_sum(local_sum);
    if ((tid & 31u) == 0) partial[tid / 32] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < threads / 32; i++) sum += partial[i];
        partial[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = partial[0];
    for (uint t = tid; t < num_tokens; t += threads) scores[base + t] *= inverse;
}

kernel void gqa_weighted_sum(
    device const float* scores [[ buffer(0) ]],
    device const half* v_cache [[ buffer(1) ]],
    device half* output [[ buffer(2) ]],
    constant uint& num_tokens [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint d = gid.x, head = gid.y;
    if (d >= 64 || head >= 14) return;
    uint kv_head = head / 7;
    float sum = 0.0f;
    for (uint t = 0; t < num_tokens; t++)
        sum += scores[head * num_tokens + t] *
               float(v_cache[t * 128 + kv_head * 64 + d]);
    output[head * 64 + d] = half(sum);
}

kernel void embedding_batch(
    device const uint* token_ids [[ buffer(0) ]],
    device const half* embedding [[ buffer(1) ]],
    device half* output [[ buffer(2) ]],
    constant uint& token_count [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    uint total = token_count * 896;
    if (id >= total) return;
    uint token = id / 896, column = id % 896;
    output[id] = embedding[token_ids[token] * 896 + column];
}

kernel void rms_norm_batch(
    device const half* x [[ buffer(0) ]],
    device const half* weight [[ buffer(1) ]],
    device half* y [[ buffer(2) ]],
    constant uint& D [[ buffer(3) ]],
    threadgroup float* partial [[ threadgroup(0) ]],
    uint token [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_position_in_threadgroup ]],
    uint threads [[ threads_per_threadgroup ]])
{
    uint base = token * D;
    float sum = 0.0f;
    for (uint i=tid; i<D; i+=threads) { float v=float(x[base+i]); sum+=v*v; }
    sum=simd_sum(sum); if((tid&31u)==0) partial[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float total=0;for(uint i=0;i<threads/32;i++)total+=partial[i];partial[0]=rsqrt(total/float(D)+1e-6f);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=partial[0];
    for(uint i=tid;i<D;i+=threads)y[base+i]=half(float(x[base+i])*float(weight[i])*inv);
}

kernel void bias_add_batch(device half* x [[buffer(0)]],device const half* bias [[buffer(1)]],
                           constant uint& D [[buffer(2)]],uint id [[thread_position_in_grid]])
{ x[id]=half(float(x[id])+float(bias[id%D])); }

kernel void residual_add_batch(device half* x [[buffer(0)]],device const half* residual [[buffer(1)]],
                               uint id [[thread_position_in_grid]])
{ x[id]=half(float(x[id])+float(residual[id])); }

kernel void rope_qk_batch(device const half* q [[buffer(0)]],device half* k [[buffer(1)]],
                          device float* q_rotated [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                          uint id [[thread_position_in_grid]])
{
    constexpr uint pairs_per_token=16*32;
    if(id>=tokens*pairs_per_token)return;uint token=id/pairs_per_token,z=id%pairs_per_token;
    bool iq=z<14*32;uint local=iq?z:z-14*32,head=local/32,i=local%32,base=token*(iq?896:128)+head*64;
    float angle=float(token)*pow(1000000.0f,-float(2*i)/64.0f),c=cos(angle),s=sin(angle);
    float a=iq?float(q[base+i]):float(k[base+i]),b=iq?float(q[base+i+32]):float(k[base+i+32]);
    if(iq){q_rotated[token*896+head*64+i]=a*c-b*s;q_rotated[token*896+head*64+i+32]=b*c+a*s;}
    else{k[base+i]=half(a*c-b*s);k[base+i+32]=half(b*c+a*s);}
}

kernel void kv_cache_batch(device const half* k [[buffer(0)]],device const half* v [[buffer(1)]],
                           device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],
                           constant uint& tokens [[buffer(4)]],uint id [[thread_position_in_grid]])
{ if(id<tokens*128){kc[id]=k[id];vc[id]=v[id];} }

kernel void causal_scores_batch(device const float* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                device float* scores [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]])
{
    uint key=gid.x,head=gid.y,query=gid.z;if(query>=tokens||key>query)return;uint kh=head/7;float sum=0;
    for(uint d=0;d<64;d++)sum+=q[query*896+head*64+d]*float(kc[key*128+kh*64+d]);
    scores[(query*14+head)*tokens+key]=sum*0.125f;
}

kernel void causal_softmax_batch(device float* scores [[buffer(0)]],constant uint& tokens [[buffer(1)]],
                                 threadgroup float* p [[threadgroup(0)]],uint2 group [[threadgroup_position_in_grid]],
                                 uint2 local [[thread_position_in_threadgroup]],uint2 size [[threads_per_threadgroup]])
{
    uint head=group.x,query=group.y,tid=local.x,nt=size.x,n=query+1,base=(query*14+head)*tokens;float m=-INFINITY;
    for(uint t=tid;t<n;t+=nt)m=max(m,scores[base+t]);m=simd_max(m);if((tid&31u)==0)p[tid/32]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float z=p[0];for(uint i=1;i<nt/32;i++)z=max(z,p[i]);p[0]=z;}
    threadgroup_barrier(mem_flags::mem_threadgroup);m=p[0];float sum=0;
    for(uint t=tid;t<n;t+=nt){float e=exp(scores[base+t]-m);scores[base+t]=e;sum+=e;}sum=simd_sum(sum);
    if((tid&31u)==0)p[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float z=0;for(uint i=0;i<nt/32;i++)z+=p[i];p[0]=1.0f/z;}threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=p[0];for(uint t=tid;t<n;t+=nt)scores[base+t]*=inv;
}

kernel void causal_weighted_batch(device const float* scores [[buffer(0)]],device const half* vc [[buffer(1)]],
                                  device half* out [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    uint d=gid.x,head=gid.y,query=gid.z;if(d>=64||head>=14||query>=tokens)return;uint kh=head/7;float sum=0;
    uint base=(query*14+head)*tokens;for(uint t=0;t<=query;t++)sum+=scores[base+t]*float(vc[t*128+kh*64+d]);
    out[query*896+head*64+d]=half(sum);
}

kernel void rope_qk_batch_offset(device const half* q [[buffer(0)]],device half* k [[buffer(1)]],
                                 device float* q_rotated [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                                 constant uint& start_pos [[buffer(4)]],uint id [[thread_position_in_grid]])
{
    constexpr uint pairs_per_token=16*32;
    if(id>=tokens*pairs_per_token)return;
    uint token=id/pairs_per_token,z=id%pairs_per_token;
    bool iq=z<14*32;
    uint local=iq?z:z-14*32,head=local/32,i=local%32,base=token*(iq?896:128)+head*64;
    float position=float(start_pos+token);
    float angle=position*pow(1000000.0f,-float(2*i)/64.0f),c=cos(angle),s=sin(angle);
    float a=iq?float(q[base+i]):float(k[base+i]),b=iq?float(q[base+i+32]):float(k[base+i+32]);
    if(iq){q_rotated[token*896+head*64+i]=a*c-b*s;q_rotated[token*896+head*64+i+32]=b*c+a*s;}
    else{k[base+i]=half(a*c-b*s);k[base+i+32]=half(b*c+a*s);}
}

kernel void kv_cache_batch_offset(device const half* k [[buffer(0)]],device const half* v [[buffer(1)]],
                                  device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],
                                  constant uint& tokens [[buffer(4)]],constant uint& start_pos [[buffer(5)]],
                                  uint id [[thread_position_in_grid]])
{
    if(id>=tokens*128)return;
    uint token=id/128,dim=id%128,cache_offset=(start_pos+token)*128+dim;
    kc[cache_offset]=k[id];
    vc[cache_offset]=v[id];
}

kernel void causal_scores_batch_offset(device const float* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                       device float* scores [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                                       constant uint& total_tokens [[buffer(4)]],constant uint& start_pos [[buffer(5)]],
                                       uint3 gid [[thread_position_in_grid]])
{
    uint key=gid.x,head=gid.y,query=gid.z;
    if(query>=tokens||head>=14||key>start_pos+query||key>=total_tokens)return;
    uint kh=head/7;
    float sum=0;
    for(uint d=0;d<64;d++)sum+=q[query*896+head*64+d]*float(kc[key*128+kh*64+d]);
    scores[(query*14+head)*total_tokens+key]=sum*0.125f;
}

kernel void causal_softmax_batch_offset(device float* scores [[buffer(0)]],
                                        constant uint& tokens [[buffer(1)]],
                                        constant uint& total_tokens [[buffer(2)]],
                                        constant uint& start_pos [[buffer(3)]],
                                        threadgroup float* p [[threadgroup(0)]],
                                        uint2 group [[threadgroup_position_in_grid]],
                                        uint2 local [[thread_position_in_threadgroup]],
                                        uint2 size [[threads_per_threadgroup]])
{
    uint head=group.x,query=group.y,tid=local.x,nt=size.x,n=start_pos+query+1;
    if(head>=14||query>=tokens||n==0)return;
    uint base=(query*14+head)*total_tokens;
    float m=-INFINITY;
    for(uint t=tid;t<n;t+=nt)m=max(m,scores[base+t]);
    m=simd_max(m);
    if((tid&31u)==0)p[tid/32]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float z=p[0];for(uint i=1;i<nt/32;i++)z=max(z,p[i]);p[0]=z;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=p[0];
    float sum=0;
    for(uint t=tid;t<n;t+=nt){float e=exp(scores[base+t]-m);scores[base+t]=e;sum+=e;}
    sum=simd_sum(sum);
    if((tid&31u)==0)p[tid/32]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float z=0;for(uint i=0;i<nt/32;i++)z+=p[i];p[0]=1.0f/z;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=p[0];
    for(uint t=tid;t<n;t+=nt)scores[base+t]*=inv;
}

kernel void causal_weighted_batch_offset(device const float* scores [[buffer(0)]],device const half* vc [[buffer(1)]],
                                         device half* out [[buffer(2)]],constant uint& tokens [[buffer(3)]],
                                         constant uint& total_tokens [[buffer(4)]],constant uint& start_pos [[buffer(5)]],
                                         uint3 gid [[thread_position_in_grid]])
{
    uint d=gid.x,head=gid.y,query=gid.z;
    if(d>=64||head>=14||query>=tokens)return;
    uint kh=head/7,n=start_pos+query+1,base=(query*14+head)*total_tokens;
    float sum=0;
    for(uint t=0;t<n;t++)sum+=scores[base+t]*float(vc[t*128+kh*64+d]);
    out[query*896+head*64+d]=half(sum);
}

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

kernel void rms_norm_fast(
    device const half* x          [[ buffer(0) ]],
    device const half* weight     [[ buffer(1) ]],
    device half* y                [[ buffer(2) ]],
    constant uint& D              [[ buffer(3) ]],
    threadgroup float* partial    [[ threadgroup(0) ]],
    uint tid                      [[ thread_position_in_threadgroup ]],
    uint threads                  [[ threads_per_threadgroup ]])
{
    float sum = 0.0f;
    for (uint i = tid; i < D; i += threads) {
        float v = float(x[i]);
        sum += v * v;
    }

    sum = simd_sum(sum);
    if ((tid & 31u) == 0) partial[tid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < (threads + 31) / 32; i++) total += partial[i];
        partial[0] = rsqrt(total / float(D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = partial[0];
    for (uint i = tid; i < D; i += threads)
        y[i] = half(float(x[i]) * float(weight[i]) * inv_rms);
}

kernel void silu_mul_float(
    device const float* gate [[ buffer(0) ]],
    device const float* up   [[ buffer(1) ]],
    device float* out        [[ buffer(2) ]],
    uint id                  [[ thread_position_in_grid ]])
{
    float g = gate[id];
    out[id] = (g / (1.0f + exp(-g))) * up[id];
}

kernel void half_to_float(
    device const half* input [[ buffer(0) ]],
    device float* output     [[ buffer(1) ]],
    uint id                  [[ thread_position_in_grid ]])
{
    output[id] = float(input[id]);
}

kernel void residual_add_float(
    device half* x          [[ buffer(0) ]],
    device const float* r   [[ buffer(1) ]],
    uint id                 [[ thread_position_in_grid ]])
{
    x[id] = half(float(x[id]) + float(half(r[id])));
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

    sum = simd_sum(sum);

    if (local_id % 32 == 0) {
        partial[local_id / 32] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    device const half* q       [[ buffer(0) ]],  
    device const half* k       [[ buffer(1) ]],  
    device float* scores       [[ buffer(2) ]],  
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
