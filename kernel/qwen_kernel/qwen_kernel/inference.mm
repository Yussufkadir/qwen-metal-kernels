#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

constexpr int NUM_LAYERS = 24;
constexpr int HIDDEN_DIM = 896;
constexpr int INTERMEDIATE = 4864;
constexpr int NUM_HEADS = 14;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = 64;
constexpr int VOCAB_SIZE = 151936;

extern "C" {
    int metal_init();
    int run_gate_up_batched(
        const uint16_t* gate_w, const uint16_t* up_w, const float* x,
        float* gate_out, float* up_out, uint32_t B, uint32_t M, uint32_t K);
    int run_down_batched(
        const uint16_t* down_w, const float* x, float* out,
        uint32_t B, uint32_t M, uint32_t K);
}

id<MTLBuffer> loadHalfBuffer(id<MTLDevice> device, const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "Missing: " << path << "\n"; exit(1); }
    size_t size = f.tellg(); f.seekg(0);
    std::vector<uint16_t> data(size / 2);
    f.read((char*)data.data(), size);
    return [device newBufferWithBytes:data.data() length:size options:MTLResourceStorageModeShared];
}

void runKernel(id<MTLDevice> device, id<MTLCommandQueue> queue,
               id<MTLComputePipelineState> pipeline, NSArray* buffers, uint N) {
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < [buffers count]; i++)
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    MTLSize grid = MTLSizeMake(N, 1, 1);
    MTLSize tg = MTLSizeMake(std::min(N, 256u), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

id<MTLBuffer> matvec(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline, id<MTLBuffer> weight,
                     id<MTLBuffer> input, int outDim, int inDim) {
    id<MTLBuffer> output = [device newBufferWithLength:outDim * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:weight offset:0 atIndex:0];
    [enc setBuffer:input  offset:0 atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    uint K = (uint)inDim;
    [enc setBytes:&K length:sizeof(uint) atIndex:3];
    uint gsize = 128;
    size_t tg_bytes = (gsize / 32) * sizeof(float);
    [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
    MTLSize grid = MTLSizeMake((uint)outDim, 1, 1);
    MTLSize group = MTLSizeMake(gsize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    return output;
}

void apply_rope(std::vector<float>& vec, int pos, int head_dim, float theta = 10000.0f) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float a = vec[i];
        float b = vec[i + half];
        vec[i]        = a * cos_val - b * sin_val;
        vec[i + half] = b * cos_val + a * sin_val;
    }
}

std::vector<float> attention_head(const std::vector<float>& q,
                                  const std::vector<float>& k_all,
                                  const std::vector<float>& v_all,
                                  int num_tokens, int head_dim) {
    std::vector<float> scores(num_tokens);
    for (int t = 0; t < num_tokens; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k_all[t * head_dim + d];
        scores[t] = dot / sqrtf((float)head_dim);
    }
    float max_s = *std::max_element(scores.begin(), scores.end());
    float sum_e = 0.0f;
    for (auto& s : scores) { s = expf(s - max_s); sum_e += s; }
    for (auto& s : scores) s /= sum_e;
    std::vector<float> out(head_dim, 0.0f);
    for (int t = 0; t < num_tokens; t++)
        for (int d = 0; d < head_dim; d++)
            out[d] += scores[t] * v_all[t * head_dim + d];
    return out;
}

int main(int argc, char* argv[]) {
    std::cout << "[DEBUG] Starting main" << std::endl;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) { std::cerr << "default.metallib not found\n"; return 1; }

    std::cout << "[DEBUG] Metal device created" << std::endl;

    if (metal_init() != 0) { std::cerr << "bridge init failed\n"; return 1; }

    std::cout << "[DEBUG] Bridge initialized" << std::endl;

    auto loadPipe = [&](NSString* name) -> id<MTLComputePipelineState> {
        return [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:nil];
    };
    auto pipeRmsNorm = loadPipe(@"rms_norm");
    auto pipeResAdd  = loadPipe(@"residual_add");
    auto pipeMatvec  = loadPipe(@"matvec_float4");

    std::cout << "[DEBUG] Pipelines loaded" << std::endl;

    auto loadW = [&](const std::string& fname) {
        return loadHalfBuffer(device, "qwen_weights/" + fname + ".bin");
    };

    std::cout << "[DEBUG] Loading weights..." << std::endl;

    id<MTLBuffer> embed      = loadW("embed_tokens.weight");
    std::cout << "[DEBUG] embed loaded" << std::endl;
    id<MTLBuffer> final_norm = loadW("norm.weight");
    std::cout << "[DEBUG] final_norm loaded" << std::endl;
    id<MTLBuffer> lm_head    = loadW("lm_head.weight");
    std::cout << "[DEBUG] lm_head loaded" << std::endl;

    struct LayerWeights {
        id<MTLBuffer> input_norm, post_norm;
        id<MTLBuffer> q_proj, k_proj, v_proj, o_proj;
        id<MTLBuffer> q_bias, k_bias, v_bias;
        id<MTLBuffer> gate, up, down;
    };
    std::vector<LayerWeights> layers(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        std::string p = "layer" + std::to_string(i);
        layers[i].input_norm = loadW(p + ".input_layernorm.weight");
        layers[i].post_norm  = loadW(p + ".post_attention_layernorm.weight");
        layers[i].q_proj = loadW(p + ".self_attn.q_proj.weight");
        layers[i].k_proj = loadW(p + ".self_attn.k_proj.weight");
        layers[i].v_proj = loadW(p + ".self_attn.v_proj.weight");
        layers[i].o_proj = loadW(p + ".self_attn.o_proj.weight");
        layers[i].q_bias = loadW(p + ".self_attn.q_proj.bias");
        layers[i].k_bias = loadW(p + ".self_attn.k_proj.bias");
        layers[i].v_bias = loadW(p + ".self_attn.v_proj.bias");
        layers[i].gate = loadW(p + ".mlp.gate_proj.weight");
        layers[i].up   = loadW(p + ".mlp.up_proj.weight");
        layers[i].down = loadW(p + ".mlp.down_proj.weight");
        std::cout << "[DEBUG] Layer " << i << " loaded" << std::endl;
    }

    std::cout << "[DEBUG] All weights loaded" << std::endl;

    const size_t PAGE = 4096;
    float* x_mlp    = nullptr;
    float* gate_out = nullptr;
    float* up_out   = nullptr;
    float* mlp_mid  = nullptr;
    float* down_out = nullptr;
    posix_memalign((void**)&x_mlp,    PAGE, HIDDEN_DIM    * sizeof(float));
    posix_memalign((void**)&gate_out, PAGE, INTERMEDIATE  * sizeof(float));
    posix_memalign((void**)&up_out,   PAGE, INTERMEDIATE  * sizeof(float));
    posix_memalign((void**)&mlp_mid,  PAGE, INTERMEDIATE  * sizeof(float));
    posix_memalign((void**)&down_out, PAGE, HIDDEN_DIM    * sizeof(float));

    std::cout << "[DEBUG] MLP buffers allocated" << std::endl;

    std::vector<std::vector<uint16_t>> k_cache(NUM_LAYERS), v_cache(NUM_LAYERS);

    std::vector<int> input_ids;
    if (argc > 1)
        for (int i = 1; i < argc; i++) input_ids.push_back(std::stoi(argv[i]));
    else
        input_ids = {1053};

    std::cout << "[DEBUG] Input tokens: ";
    for (int id : input_ids) std::cout << id << " ";
    std::cout << std::endl;

    int seq_len = 0;
    std::vector<int> generated_ids;
    const int max_new_tokens = 5;

    for (int step = 0; step < (int)input_ids.size() + max_new_tokens; step++) {
        std::cout << "[DEBUG] === Step " << step << " ===" << std::endl;

        @autoreleasepool {
            int token_id = (step < (int)input_ids.size())
                            ? input_ids[step]
                            : generated_ids.back();

            std::cout << "[DEBUG] Step " << step << ": embedding token " << token_id << std::endl;

            id<MTLBuffer> hidden = [device newBufferWithLength:HIDDEN_DIM * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
            {
                uint16_t* hptr = (uint16_t*)[hidden contents];
                uint16_t* eptr = (uint16_t*)[embed contents] + token_id * HIDDEN_DIM;
                memcpy(hptr, eptr, HIDDEN_DIM * sizeof(uint16_t));
            }

            std::cout << "[DEBUG] Step " << step << ": embedding done" << std::endl;

            if (step == 0) {
                uint16_t* hptr = (uint16_t*)[hidden contents];
                std::cout << "[CHECKPOINT] embedding output, first 8 values: ";
                for (int i = 0; i < 8; i++) {
                    __fp16 h; memcpy(&h, &hptr[i], sizeof(h));
                    std::cout << (float)h << " ";
                }
                std::cout << std::endl;
            }

            int current_pos = seq_len++;

            for (int l = 0; l < NUM_LAYERS; l++) {
                std::cout << "[DEBUG]   Layer " << l << " start" << std::endl;

                uint d = HIDDEN_DIM;
                id<MTLBuffer> d_buf = [device newBufferWithBytes:&d length:sizeof(uint)
                                                         options:MTLResourceStorageModeShared];

                id<MTLBuffer> normed = [device newBufferWithLength:HIDDEN_DIM * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
                runKernel(device, queue, pipeRmsNorm,
                         @[hidden, layers[l].input_norm, normed, d_buf], HIDDEN_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": pre-attention RMSNorm done" << std::endl;

                id<MTLBuffer> q_buf = matvec(device, queue, pipeMatvec, layers[l].q_proj, normed,
                                             NUM_HEADS * HEAD_DIM, HIDDEN_DIM);
                id<MTLBuffer> k_buf = matvec(device, queue, pipeMatvec, layers[l].k_proj, normed,
                                             NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
                id<MTLBuffer> v_buf = matvec(device, queue, pipeMatvec, layers[l].v_proj, normed,
                                             NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);

                runKernel(device, queue, pipeResAdd, @[q_buf, layers[l].q_bias], NUM_HEADS * HEAD_DIM);
                runKernel(device, queue, pipeResAdd, @[k_buf, layers[l].k_bias], NUM_KV_HEADS * HEAD_DIM);
                runKernel(device, queue, pipeResAdd, @[v_buf, layers[l].v_bias], NUM_KV_HEADS * HEAD_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": Q/K/V projections done" << std::endl;

                auto toFloatVec = [](id<MTLBuffer> buf, int len) {
                    std::vector<float> vec(len);
                    uint16_t* ptr = (uint16_t*)[buf contents];
                    for (int i = 0; i < len; i++) {
                        __fp16 h; memcpy(&h, &ptr[i], sizeof(h));
                        vec[i] = (float)h;
                    }
                    return vec;
                };
                auto toHalfVec = [](const std::vector<float>& fv) {
                    std::vector<uint16_t> hv(fv.size());
                    for (size_t i = 0; i < fv.size(); i++) {
                        __fp16 h = (__fp16)fv[i];
                        memcpy(&hv[i], &h, sizeof(h));
                    }
                    return hv;
                };

                std::vector<float> q_f = toFloatVec(q_buf, NUM_HEADS * HEAD_DIM);
                std::vector<float> k_f = toFloatVec(k_buf, NUM_KV_HEADS * HEAD_DIM);
                std::vector<float> v_f = toFloatVec(v_buf, NUM_KV_HEADS * HEAD_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": converted to float" << std::endl;

                constexpr float QWEN_ROPE_THETA = 1000000.0f;
                for (int h = 0; h < NUM_HEADS; h++) {
                    std::vector<float> q_h(q_f.begin() + h*HEAD_DIM,
                                          q_f.begin() + (h+1)*HEAD_DIM);
                    apply_rope(q_h, current_pos, HEAD_DIM, QWEN_ROPE_THETA);
                    std::copy(q_h.begin(), q_h.end(), q_f.begin() + h*HEAD_DIM);
                }
                for (int h = 0; h < NUM_KV_HEADS; h++) {
                    std::vector<float> k_h(k_f.begin() + h*HEAD_DIM,
                                          k_f.begin() + (h+1)*HEAD_DIM);
                    apply_rope(k_h, current_pos, HEAD_DIM, QWEN_ROPE_THETA);
                    std::copy(k_h.begin(), k_h.end(), k_f.begin() + h*HEAD_DIM);
                }
                std::cout << "[DEBUG]   Layer " << l << ": RoPE done" << std::endl;

                std::vector<uint16_t> k_half = toHalfVec(k_f);
                std::vector<uint16_t> v_half = toHalfVec(v_f);
                k_cache[l].insert(k_cache[l].end(), k_half.begin(), k_half.end());
                v_cache[l].insert(v_cache[l].end(), v_half.begin(), v_half.end());
                std::cout << "[DEBUG]   Layer " << l << ": KV cache updated, num_tokens="
                          << (k_cache[l].size() / (NUM_KV_HEADS * HEAD_DIM)) << std::endl;

                int num_tokens = (int)k_cache[l].size() / (NUM_KV_HEADS * HEAD_DIM);
                std::vector<float> attn_out(NUM_HEADS * HEAD_DIM, 0.0f);

                std::cout << "[DEBUG]   Layer " << l << ": starting attention loop" << std::endl;
                for (int h = 0; h < NUM_HEADS; h++) {
                    int kv_head = h / (NUM_HEADS / NUM_KV_HEADS);
                    std::vector<float> q_h(q_f.begin() + h*HEAD_DIM,
                                          q_f.begin() + (h+1)*HEAD_DIM);
                    std::vector<float> k_all(num_tokens * HEAD_DIM);
                    std::vector<float> v_all(num_tokens * HEAD_DIM);
                    for (int t = 0; t < num_tokens; t++) {
                        int off = t * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                        for (int d2 = 0; d2 < HEAD_DIM; d2++) {
                            __fp16 hk; memcpy(&hk, &k_cache[l][off+d2], sizeof(hk));
                            k_all[t*HEAD_DIM+d2] = (float)hk;
                            __fp16 hv; memcpy(&hv, &v_cache[l][off+d2], sizeof(hv));
                            v_all[t*HEAD_DIM+d2] = (float)hv;
                        }
                    }
                    auto h_out = attention_head(q_h, k_all, v_all, num_tokens, HEAD_DIM);
                    std::copy(h_out.begin(), h_out.end(), attn_out.begin() + h*HEAD_DIM);
                }
                std::cout << "[DEBUG]   Layer " << l << ": attention done" << std::endl;

                id<MTLBuffer> attn_buf = [device newBufferWithLength:NUM_HEADS*HEAD_DIM*sizeof(uint16_t)
                                                             options:MTLResourceStorageModeShared];
                {
                    uint16_t* ptr = (uint16_t*)[attn_buf contents];
                    for (int i = 0; i < NUM_HEADS*HEAD_DIM; i++) {
                        __fp16 h2 = (__fp16)attn_out[i];
                        memcpy(&ptr[i], &h2, sizeof(h2));
                    }
                }

                id<MTLBuffer> attn_proj = matvec(device, queue, pipeMatvec, layers[l].o_proj, attn_buf,
                                                 HIDDEN_DIM, NUM_HEADS * HEAD_DIM);
                runKernel(device, queue, pipeResAdd, @[hidden, attn_proj], HIDDEN_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": attention output projection done" << std::endl;

                if (step == 0 && l == 0) {
                    uint16_t* hptr = (uint16_t*)[hidden contents];
                    std::cout << "[CHECKPOINT] after attention (before MLP), first 8 values: ";
                    for (int i = 0; i < 8; i++) {
                        __fp16 h; memcpy(&h, &hptr[i], sizeof(h));
                        std::cout << (float)h << " ";
                    }
                    std::cout << std::endl;
                }

                id<MTLBuffer> normed2 = [device newBufferWithLength:HIDDEN_DIM * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
                runKernel(device, queue, pipeRmsNorm,
                         @[hidden, layers[l].post_norm, normed2, d_buf], HIDDEN_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": post-attention RMSNorm done" << std::endl;

                {
                    uint16_t* ptr = (uint16_t*)[normed2 contents];
                    for (int j = 0; j < HIDDEN_DIM; j++) {
                        __fp16 h2; memcpy(&h2, &ptr[j], sizeof(h2));
                        x_mlp[j] = (float)h2;
                    }
                }
                std::cout << "[DEBUG]   Layer " << l << ": calling MLP kernel" << std::endl;

                run_gate_up_batched((const uint16_t*)[layers[l].gate contents],
                                    (const uint16_t*)[layers[l].up contents],
                                    x_mlp, gate_out, up_out,
                                    1, INTERMEDIATE, HIDDEN_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": gate+up done" << std::endl;

                for (int i = 0; i < INTERMEDIATE; i++) {
                    float g = gate_out[i];
                    float silu = g / (1.0f + expf(-g));
                    mlp_mid[i] = silu * up_out[i];
                }
                std::cout << "[DEBUG]   Layer " << l << ": SiLU done" << std::endl;

                run_down_batched((const uint16_t*)[layers[l].down contents], mlp_mid, down_out,
                                 1, HIDDEN_DIM, INTERMEDIATE);
                std::cout << "[DEBUG]   Layer " << l << ": down done" << std::endl;

                id<MTLBuffer> mlp_buf = [device newBufferWithLength:HIDDEN_DIM * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
                {
                    uint16_t* ptr = (uint16_t*)[mlp_buf contents];
                    for (int j = 0; j < HIDDEN_DIM; j++) {
                        __fp16 h2 = (__fp16)down_out[j];
                        memcpy(&ptr[j], &h2, sizeof(h2));
                    }
                }
                runKernel(device, queue, pipeResAdd, @[hidden, mlp_buf], HIDDEN_DIM);
                std::cout << "[DEBUG]   Layer " << l << ": MLP residual done" << std::endl;

                if (step == 0 && l == 0) {
                    uint16_t* hptr = (uint16_t*)[hidden contents];
                    std::cout << "[CHECKPOINT] after layer 0, first 8 values: ";
                    for (int i = 0; i < 8; i++) {
                        __fp16 h; memcpy(&h, &hptr[i], sizeof(h));
                        std::cout << (float)h << " ";
                    }
                    std::cout << std::endl;
                }
            }

            std::cout << "[DEBUG] Step " << step << ": all layers done" << std::endl;

            id<MTLBuffer> final_hidden = [device newBufferWithLength:HIDDEN_DIM * sizeof(uint16_t)
                                                             options:MTLResourceStorageModeShared];
            {
                uint d2 = HIDDEN_DIM;
                id<MTLBuffer> d2_buf = [device newBufferWithBytes:&d2 length:sizeof(uint)
                                                         options:MTLResourceStorageModeShared];
                runKernel(device, queue, pipeRmsNorm,
                         @[hidden, final_norm, final_hidden, d2_buf], HIDDEN_DIM);
            }
            std::cout << "[DEBUG] Step " << step << ": final RMSNorm done" << std::endl;

            id<MTLBuffer> logits_buf = matvec(device, queue, pipeMatvec, lm_head, final_hidden,
                                              VOCAB_SIZE, HIDDEN_DIM);
            std::cout << "[DEBUG] Step " << step << ": LM head done" << std::endl;

            uint16_t* logits = (uint16_t*)[logits_buf contents];
            float max_val = -INFINITY;
            int next_token = 0;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                __fp16 h2; memcpy(&h2, &logits[i], sizeof(h2));
                float val = (float)h2;
                if (val > max_val) { max_val = val; next_token = i; }
            }

            generated_ids.push_back(next_token);

            std::cout << "Token " << step << ": " << next_token
                      << " (score " << max_val << ")\n";
        }
    }

    free(x_mlp); free(gate_out); free(up_out);
    free(mlp_mid); free(down_out);

    std::cout << "Generated IDs: ";
    for (int id : generated_ids) std::cout << id << " ";
    std::cout << "\n";
    return 0;
}
