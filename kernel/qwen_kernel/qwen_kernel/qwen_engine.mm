#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <chrono>
#include <random>
#include <unordered_set>

#include "qwen_engine.h"

namespace {

constexpr int NUM_LAYERS = 24;
constexpr int HIDDEN_DIM = 896;
constexpr int INTERMEDIATE = 4864;
constexpr int NUM_HEADS = 14;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = 64;
constexpr int VOCAB_SIZE = 151936;
constexpr int MAX_SEQ_LEN = 4096;
constexpr float QWEN_ROPE_THETA = 1000000.0f;

struct LayerWeights {
    id<MTLBuffer> input_norm, post_norm;
    id<MTLBuffer> q_proj, k_proj, v_proj, o_proj;
    id<MTLBuffer> q_scale, k_scale, v_scale, o_scale;
    id<MTLBuffer> q_bias, k_bias, v_bias;
    id<MTLBuffer> gate, up, down;
    id<MTLBuffer> gate_scale, up_scale, down_scale;
    MPSMatrix *q_matrix=nil, *k_matrix=nil, *v_matrix=nil, *o_matrix=nil;
    MPSMatrix *gate_matrix=nil, *up_matrix=nil, *down_matrix=nil;
};

struct SamplerState {
    QwenSamplingParams params{0.0f, 0, 1.0f, 1.0f, 0};
    std::mt19937_64 rng;
    std::unordered_set<int> seen;
};

id<MTLBuffer> loadHalfBuffer(id<MTLDevice> device, const std::string& path, bool* ok) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Missing: " << path << "\n";
        *ok = false;
        return nil;
    }
    size_t size = f.tellg(); f.seekg(0);
    std::vector<uint16_t> data(size / 2);
    f.read((char*)data.data(), size);
    return [device newBufferWithBytes:data.data() length:size options:MTLResourceStorageModeShared];
}

void encodeKernel(id<MTLComputeCommandEncoder> enc,
                  id<MTLComputePipelineState> pipeline, NSArray* buffers, uint N) {
    [enc setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < [buffers count]; i++)
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    MTLSize grid = MTLSizeMake(N, 1, 1);
    MTLSize tg = MTLSizeMake(std::min(N, 256u), 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
}

void encodeMatvec(id<MTLComputeCommandEncoder> enc,
                  id<MTLComputePipelineState> pipeline, id<MTLBuffer> weight,
                  id<MTLBuffer> input, id<MTLBuffer> output, int outDim, int inDim) {
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
}

void encodeRmsNorm(id<MTLComputeCommandEncoder> enc,
                   id<MTLComputePipelineState> pipeline, id<MTLBuffer> input,
                   id<MTLBuffer> weight, id<MTLBuffer> output, id<MTLBuffer> dim) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:weight offset:0 atIndex:1];
    [enc setBuffer:output offset:0 atIndex:2];
    [enc setBuffer:dim offset:0 atIndex:3];
    constexpr uint threads = 256;
    [enc setThreadgroupMemoryLength:(threads / 32) * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void finishAndWait(id<MTLCommandBuffer> cmd, id<MTLComputeCommandEncoder> enc) {
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void apply_rope(std::vector<float>& vec, int pos, int head_dim, float theta) {
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

} 
struct QwenEngine {
    bool verbose = false;
    QwenBackend backend = QWEN_BACKEND_METAL_FP16;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeRmsNorm = nil;
    id<MTLComputePipelineState> pipeResAdd = nil;
    id<MTLComputePipelineState> pipeMatvec = nil;
    id<MTLComputePipelineState> pipeGateUp = nil;
    id<MTLComputePipelineState> pipeHalfToFloat = nil;
    id<MTLComputePipelineState> pipeSiluMul = nil;
    id<MTLComputePipelineState> pipeDown = nil;
    id<MTLComputePipelineState> pipeResAddFloat = nil;
    id<MTLComputePipelineState> pipeMatvecQ4 = nil;
    id<MTLComputePipelineState> pipeGateUpQ4 = nil;
    id<MTLComputePipelineState> pipeDownQ4 = nil;
    id<MTLComputePipelineState> pipeMatvecQ8 = nil;
    id<MTLComputePipelineState> pipeGateUpQ8 = nil;
    id<MTLComputePipelineState> pipeDownQ8 = nil;
    id<MTLComputePipelineState> pipeSiluHalf = nil;
    id<MTLComputePipelineState> pipeMulHalf = nil;
    id<MTLComputePipelineState> pipeRopeQK = nil;
    id<MTLComputePipelineState> pipeKVAppend = nil;
    id<MTLComputePipelineState> pipeAttnScores = nil;
    id<MTLComputePipelineState> pipeAttnSoftmax = nil;
    id<MTLComputePipelineState> pipeAttnWeighted = nil;
    id<MTLComputePipelineState> pipeEmbeddingBatch = nil;
    id<MTLComputePipelineState> pipeRmsBatch = nil;
    id<MTLComputePipelineState> pipeBiasBatch = nil;
    id<MTLComputePipelineState> pipeResidualBatch = nil;
    id<MTLComputePipelineState> pipeRopeBatch = nil;
    id<MTLComputePipelineState> pipeKVBatch = nil;
    id<MTLComputePipelineState> pipeCausalScores = nil;
    id<MTLComputePipelineState> pipeCausalSoftmax = nil;
    id<MTLComputePipelineState> pipeCausalWeighted = nil;

    id<MTLBuffer> embed = nil;
    id<MTLBuffer> final_norm = nil;
    id<MTLBuffer> lm_head = nil;
    id<MTLBuffer> lm_head_scale = nil;
    MPSMatrix* lm_head_matrix = nil;
    std::vector<LayerWeights> layers;

    id<MTLBuffer> hidden_dim_const_buf = nil;  
    id<MTLBuffer> hidden = nil;                
    id<MTLBuffer> normed = nil;             
    id<MTLBuffer> normed2 = nil;              
    id<MTLBuffer> q_buf = nil;                
    id<MTLBuffer> k_buf = nil;                 
    id<MTLBuffer> v_buf = nil;                 
    id<MTLBuffer> attn_buf = nil;              
    id<MTLBuffer> attn_proj = nil;             
    id<MTLBuffer> mlp_buf = nil;               
    id<MTLBuffer> final_hidden = nil;          
    id<MTLBuffer> logits_buf = nil;            

    id<MTLBuffer> x_mlp_buf = nil;
    id<MTLBuffer> gate_out_buf = nil;
    id<MTLBuffer> up_out_buf = nil;
    id<MTLBuffer> mlp_mid_buf = nil;
    id<MTLBuffer> down_out_buf = nil;
    float* x_mlp = nullptr;
    float* gate_out = nullptr;
    float* up_out = nullptr;
    float* mlp_mid = nullptr;
    float* down_out = nullptr;

    MPSMatrixVectorMultiplication *mpsHH=nil, *mpsHKV=nil, *mpsHI=nil;
    MPSMatrixVectorMultiplication *mpsIH=nil, *mpsHVocab=nil;
    MPSVector *normedVec=nil, *qVec=nil, *kVec=nil, *vVec=nil;
    MPSVector *attnVec=nil, *attnProjVec=nil, *normed2Vec=nil;
    MPSVector *finalHiddenVec=nil, *logitsVec=nil;
    id<MTLBuffer> gate_half=nil, up_half=nil, mlp_mid_half=nil, down_half=nil;
    MPSVector *gateHalfVec=nil, *upHalfVec=nil, *mlpMidHalfVec=nil, *downHalfVec=nil;
    std::vector<id<MTLBuffer>> k_cache_gpu, v_cache_gpu;
    id<MTLBuffer> attention_scores = nil;
    id<MTLBuffer> q_rotated = nil;
    int session_pos = 0;
    std::vector<int> session_tokens;
};

namespace {

#define LOG(engine, msg) do { if ((engine)->verbose) std::cout << msg << std::endl; } while (0)

bool backendUsesQ4Projections(QwenBackend backend) {
    return backend == QWEN_BACKEND_METAL_INT4 ||
           backend == QWEN_BACKEND_HYBRID ||
           backend == QWEN_BACKEND_INT4_FP16_LM_HEAD;
}

bool backendUsesQ8Projections(QwenBackend backend) {
    return backend == QWEN_BACKEND_METAL_INT8;
}

bool backendUsesBatchedPrefill(QwenBackend backend) {
    return backend == QWEN_BACKEND_METAL_FP16 ||
           backend == QWEN_BACKEND_HYBRID;
}

QwenSamplingParams greedySamplingParams() {
    return QwenSamplingParams{0.0f, 0, 1.0f, 1.0f, 0};
}

bool usesSampling(const QwenSamplingParams& params) {
    return params.temperature > 0.0f;
}

SamplerState makeSampler(const QwenSamplingParams& params,
                         const int* prompt_tokens, int prompt_len) {
    SamplerState state;
    state.params = params;
    uint64_t seed = params.seed;
    if (seed == 0) {
        seed = (uint64_t)std::chrono::high_resolution_clock::now()
                   .time_since_epoch().count();
    }
    state.rng.seed(seed);
    if (params.repetition_penalty > 1.0f && prompt_tokens) {
        for (int i = 0; i < prompt_len; i++) state.seen.insert(prompt_tokens[i]);
    }
    return state;
}

int selectTokenFromLogits(QwenEngine* eng, SamplerState* sampler) {
    uint16_t* raw = (uint16_t*)[eng->logits_buf contents];
    QwenSamplingParams params = sampler ? sampler->params : greedySamplingParams();

    std::vector<float> logits(VOCAB_SIZE);
    float maxValue = -INFINITY;
    int greedyToken = 0;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        __fp16 value; memcpy(&value, &raw[i], sizeof(value));
        float logit = (float)value;
        if (params.repetition_penalty > 1.0f && sampler &&
            sampler->seen.find(i) != sampler->seen.end()) {
            logit = logit > 0.0f ? logit / params.repetition_penalty
                                 : logit * params.repetition_penalty;
        }
        logits[i] = logit;
        if (logit > maxValue) {
            maxValue = logit;
            greedyToken = i;
        }
    }

    if (!usesSampling(params)) {
        if (sampler && params.repetition_penalty > 1.0f) sampler->seen.insert(greedyToken);
        return greedyToken;
    }

    int topK = params.top_k > 0 ? std::min(params.top_k, VOCAB_SIZE) : VOCAB_SIZE;
    std::vector<int> ids(VOCAB_SIZE);
    for (int i = 0; i < VOCAB_SIZE; i++) ids[i] = i;
    if (topK < VOCAB_SIZE) {
        std::nth_element(ids.begin(), ids.begin() + topK, ids.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        ids.resize(topK);
    }
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    float temperature = std::max(params.temperature, 1e-6f);
    float maxLogit = logits[ids[0]] / temperature;
    std::vector<float> weights(ids.size());
    double sum = 0.0;
    for (size_t i = 0; i < ids.size(); i++) {
        double w = std::exp((double)(logits[ids[i]] / temperature - maxLogit));
        weights[i] = (float)w;
        sum += w;
    }

    size_t keep = ids.size();
    if (params.top_p > 0.0f && params.top_p < 1.0f && sum > 0.0) {
        double cumulative = 0.0;
        keep = 0;
        for (; keep < ids.size(); keep++) {
            cumulative += weights[keep] / sum;
            if (cumulative >= params.top_p) {
                keep++;
                break;
            }
        }
        keep = std::max<size_t>(1, keep);
        ids.resize(keep);
        weights.resize(keep);
    }

    double filteredSum = 0.0;
    for (float w : weights) filteredSum += w;
    int selected = ids[0];
    if (filteredSum > 0.0 && std::isfinite(filteredSum)) {
        std::uniform_real_distribution<double> dist(0.0, filteredSum);
        double draw = dist(sampler->rng);
        double cumulative = 0.0;
        for (size_t i = 0; i < ids.size(); i++) {
            cumulative += weights[i];
            if (draw <= cumulative) {
                selected = ids[i];
                break;
            }
        }
    }
    if (sampler && params.repetition_penalty > 1.0f) sampler->seen.insert(selected);
    return selected;
}

void encodeProjection(QwenEngine* eng, id<MTLComputeCommandEncoder> enc,
                      id<MTLBuffer> weight, id<MTLBuffer> scale,
                      id<MTLBuffer> input, id<MTLBuffer> output,
                      int outDim, int inDim) {
    if (eng->backend == QWEN_BACKEND_METAL_FP16 || eng->backend == QWEN_BACKEND_MPS_FP16) {
        encodeMatvec(enc, eng->pipeMatvec, weight, input, output, outDim, inDim);
        return;
    }
    if (eng->backend == QWEN_BACKEND_METAL_INT8) {
        [enc setComputePipelineState:eng->pipeMatvecQ8];
        [enc setBuffer:weight offset:0 atIndex:0];
        [enc setBuffer:scale offset:0 atIndex:1];
        [enc setBuffer:input offset:0 atIndex:2];
        [enc setBuffer:output offset:0 atIndex:3];
        uint K = (uint)inDim;
        [enc setBytes:&K length:sizeof(K) atIndex:4];
        constexpr uint threads = 128;
        [enc setThreadgroupMemoryLength:(threads / 32) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((uint)outDim, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        return;
    }
    [enc setComputePipelineState:eng->pipeMatvecQ4];
    [enc setBuffer:weight offset:0 atIndex:0];
    [enc setBuffer:scale offset:0 atIndex:1];
    [enc setBuffer:input offset:0 atIndex:2];
    [enc setBuffer:output offset:0 atIndex:3];
    uint K = (uint)inDim;
    [enc setBytes:&K length:sizeof(K) atIndex:4];
    constexpr uint threads = 128;
    [enc setThreadgroupMemoryLength:(threads / 32) * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((uint)outDim, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void encodeMPS(id<MTLCommandBuffer> cmd, MPSMatrixVectorMultiplication* kernel,
               MPSMatrix* matrix, MPSVector* input, MPSVector* output) {
    [kernel encodeToCommandBuffer:cmd inputMatrix:matrix inputVector:input resultVector:output];
}

int forward_step_gpu(QwenEngine* eng, int token_id, int current_pos, SamplerState* sampler) {
    memcpy([eng->hidden contents],
           (uint16_t*)[eng->embed contents] + token_id * HIDDEN_DIM,
           HIDDEN_DIM * sizeof(uint16_t));

    id<MTLCommandBuffer> cmd = [eng->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    uint position = (uint)current_pos;
    uint numTokens = position + 1;
    constexpr uint gsize = 128;

    for (int l = 0; l < NUM_LAYERS; l++) {
        auto& lw = eng->layers[l];
        encodeRmsNorm(enc, eng->pipeRmsNorm, eng->hidden, lw.input_norm,
                      eng->normed, eng->hidden_dim_const_buf);
        encodeProjection(eng, enc, lw.q_proj, lw.q_scale, eng->normed, eng->q_buf,
                         NUM_HEADS * HEAD_DIM, HIDDEN_DIM);
        encodeProjection(eng, enc, lw.k_proj, lw.k_scale, eng->normed, eng->k_buf,
                         NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
        encodeProjection(eng, enc, lw.v_proj, lw.v_scale, eng->normed, eng->v_buf,
                         NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
        encodeKernel(enc, eng->pipeResAdd, @[eng->q_buf, lw.q_bias], NUM_HEADS * HEAD_DIM);
        encodeKernel(enc, eng->pipeResAdd, @[eng->k_buf, lw.k_bias], NUM_KV_HEADS * HEAD_DIM);
        encodeKernel(enc, eng->pipeResAdd, @[eng->v_buf, lw.v_bias], NUM_KV_HEADS * HEAD_DIM);

        [enc setComputePipelineState:eng->pipeRopeQK];
        [enc setBuffer:eng->q_buf offset:0 atIndex:0];
        [enc setBuffer:eng->k_buf offset:0 atIndex:1];
        [enc setBuffer:eng->q_rotated offset:0 atIndex:2];
        [enc setBytes:&position length:sizeof(position) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM / 2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:eng->pipeKVAppend];
        [enc setBuffer:eng->k_buf offset:0 atIndex:0];
        [enc setBuffer:eng->v_buf offset:0 atIndex:1];
        [enc setBuffer:eng->k_cache_gpu[l] offset:0 atIndex:2];
        [enc setBuffer:eng->v_cache_gpu[l] offset:0 atIndex:3];
        [enc setBytes:&position length:sizeof(position) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(NUM_KV_HEADS * HEAD_DIM, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:eng->pipeAttnScores];
        [enc setBuffer:eng->q_rotated offset:0 atIndex:0];
        [enc setBuffer:eng->k_cache_gpu[l] offset:0 atIndex:1];
        [enc setBuffer:eng->attention_scores offset:0 atIndex:2];
        [enc setBytes:&numTokens length:sizeof(numTokens) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(numTokens, NUM_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:eng->pipeAttnSoftmax];
        [enc setBuffer:eng->attention_scores offset:0 atIndex:0];
        [enc setBytes:&numTokens length:sizeof(numTokens) atIndex:1];
        uint softmaxThreads = std::min(256u, std::max(32u, ((numTokens + 31) / 32) * 32));
        [enc setThreadgroupMemoryLength:(softmaxThreads / 32) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(NUM_HEADS, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(softmaxThreads, 1, 1)];

        [enc setComputePipelineState:eng->pipeAttnWeighted];
        [enc setBuffer:eng->attention_scores offset:0 atIndex:0];
        [enc setBuffer:eng->v_cache_gpu[l] offset:0 atIndex:1];
        [enc setBuffer:eng->attn_buf offset:0 atIndex:2];
        [enc setBytes:&numTokens length:sizeof(numTokens) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(HEAD_DIM, NUM_HEADS, 1)
             threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];

        encodeProjection(eng, enc, lw.o_proj, lw.o_scale, eng->attn_buf, eng->attn_proj,
                         HIDDEN_DIM, NUM_HEADS * HEAD_DIM);
        encodeKernel(enc, eng->pipeResAdd, @[eng->hidden, eng->attn_proj], HIDDEN_DIM);
        encodeRmsNorm(enc, eng->pipeRmsNorm, eng->hidden, lw.post_norm,
                      eng->normed2, eng->hidden_dim_const_buf);
        encodeKernel(enc, eng->pipeHalfToFloat, @[eng->normed2, eng->x_mlp_buf], HIDDEN_DIM);

        uint gateK = HIDDEN_DIM;
        if (eng->backend == QWEN_BACKEND_METAL_FP16) {
            [enc setComputePipelineState:eng->pipeGateUp];
            [enc setBuffer:lw.gate offset:0 atIndex:0];
            [enc setBuffer:lw.up offset:0 atIndex:1];
            [enc setBuffer:eng->x_mlp_buf offset:0 atIndex:2];
            [enc setBuffer:eng->gate_out_buf offset:0 atIndex:3];
            [enc setBuffer:eng->up_out_buf offset:0 atIndex:4];
            [enc setBytes:&gateK length:sizeof(gateK) atIndex:5];
        } else if (eng->backend == QWEN_BACKEND_METAL_INT8) {
            [enc setComputePipelineState:eng->pipeGateUpQ8];
            [enc setBuffer:lw.gate offset:0 atIndex:0];
            [enc setBuffer:lw.gate_scale offset:0 atIndex:1];
            [enc setBuffer:lw.up offset:0 atIndex:2];
            [enc setBuffer:lw.up_scale offset:0 atIndex:3];
            [enc setBuffer:eng->x_mlp_buf offset:0 atIndex:4];
            [enc setBuffer:eng->gate_out_buf offset:0 atIndex:5];
            [enc setBuffer:eng->up_out_buf offset:0 atIndex:6];
            [enc setBytes:&gateK length:sizeof(gateK) atIndex:7];
        } else {
            [enc setComputePipelineState:eng->pipeGateUpQ4];
            [enc setBuffer:lw.gate offset:0 atIndex:0];
            [enc setBuffer:lw.gate_scale offset:0 atIndex:1];
            [enc setBuffer:lw.up offset:0 atIndex:2];
            [enc setBuffer:lw.up_scale offset:0 atIndex:3];
            [enc setBuffer:eng->x_mlp_buf offset:0 atIndex:4];
            [enc setBuffer:eng->gate_out_buf offset:0 atIndex:5];
            [enc setBuffer:eng->up_out_buf offset:0 atIndex:6];
            [enc setBytes:&gateK length:sizeof(gateK) atIndex:7];
        }
        [enc setThreadgroupMemoryLength:(gsize / 32) * 2 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(gsize, 1, 1)];
        encodeKernel(enc, eng->pipeSiluMul,
                     @[eng->gate_out_buf, eng->up_out_buf, eng->mlp_mid_buf], INTERMEDIATE);

        uint downK = INTERMEDIATE;
        if (eng->backend == QWEN_BACKEND_METAL_FP16) {
            [enc setComputePipelineState:eng->pipeDown];
            [enc setBuffer:lw.down offset:0 atIndex:0];
            [enc setBuffer:eng->mlp_mid_buf offset:0 atIndex:1];
            [enc setBuffer:eng->down_out_buf offset:0 atIndex:2];
            [enc setBytes:&downK length:sizeof(downK) atIndex:3];
        } else if (eng->backend == QWEN_BACKEND_METAL_INT8) {
            [enc setComputePipelineState:eng->pipeDownQ8];
            [enc setBuffer:lw.down offset:0 atIndex:0];
            [enc setBuffer:lw.down_scale offset:0 atIndex:1];
            [enc setBuffer:eng->mlp_mid_buf offset:0 atIndex:2];
            [enc setBuffer:eng->down_out_buf offset:0 atIndex:3];
            [enc setBytes:&downK length:sizeof(downK) atIndex:4];
        } else {
            [enc setComputePipelineState:eng->pipeDownQ4];
            [enc setBuffer:lw.down offset:0 atIndex:0];
            [enc setBuffer:lw.down_scale offset:0 atIndex:1];
            [enc setBuffer:eng->mlp_mid_buf offset:0 atIndex:2];
            [enc setBuffer:eng->down_out_buf offset:0 atIndex:3];
            [enc setBytes:&downK length:sizeof(downK) atIndex:4];
        }
        [enc setThreadgroupMemoryLength:(gsize / 32) * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(HIDDEN_DIM, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(gsize, 1, 1)];
        encodeKernel(enc, eng->pipeResAddFloat, @[eng->hidden, eng->down_out_buf], HIDDEN_DIM);
    }

    encodeRmsNorm(enc, eng->pipeRmsNorm, eng->hidden, eng->final_norm,
                  eng->final_hidden, eng->hidden_dim_const_buf);
    encodeProjection(eng, enc, eng->lm_head, eng->lm_head_scale,
                     eng->final_hidden, eng->logits_buf, VOCAB_SIZE, HIDDEN_DIM);
    finishAndWait(cmd, enc);

    return selectTokenFromLogits(eng, sampler);
}

int prefill_batch_fp16(QwenEngine* eng, const int* token_ids, int token_count, SamplerState* sampler) {
    id<MTLDevice> device = eng->device;
    auto halfBuffer = [&](size_t elements) {
        return [device newBufferWithLength:elements * sizeof(uint16_t)
                                    options:MTLResourceStorageModeShared];
    };
    auto floatBuffer = [&](size_t elements) {
        return [device newBufferWithLength:elements * sizeof(float)
                                    options:MTLResourceStorageModeShared];
    };
    id<MTLBuffer> tokenBuffer = [device newBufferWithBytes:token_ids
                                                    length:token_count * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> hidden = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> normed = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> q = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> k = halfBuffer((size_t)token_count * NUM_KV_HEADS * HEAD_DIM);
    id<MTLBuffer> v = halfBuffer((size_t)token_count * NUM_KV_HEADS * HEAD_DIM);
    id<MTLBuffer> qRotated = floatBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> attention = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> attentionProjected = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> normed2 = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> gate = halfBuffer((size_t)token_count * INTERMEDIATE);
    id<MTLBuffer> up = halfBuffer((size_t)token_count * INTERMEDIATE);
    id<MTLBuffer> middle = halfBuffer((size_t)token_count * INTERMEDIATE);
    id<MTLBuffer> down = halfBuffer((size_t)token_count * HIDDEN_DIM);
    id<MTLBuffer> scores = floatBuffer((size_t)token_count * NUM_HEADS * token_count);

    auto matrix = [&](id<MTLBuffer> buffer, int rows, int columns) -> MPSMatrix* {
        auto* descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
            rowBytes:(NSUInteger)columns * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    };
    MPSMatrix* normedM = matrix(normed, token_count, HIDDEN_DIM);
    MPSMatrix* qM = matrix(q, token_count, HIDDEN_DIM);
    MPSMatrix* kM = matrix(k, token_count, NUM_KV_HEADS * HEAD_DIM);
    MPSMatrix* vM = matrix(v, token_count, NUM_KV_HEADS * HEAD_DIM);
    MPSMatrix* attentionM = matrix(attention, token_count, HIDDEN_DIM);
    MPSMatrix* attentionProjectedM = matrix(attentionProjected, token_count, HIDDEN_DIM);
    MPSMatrix* normed2M = matrix(normed2, token_count, HIDDEN_DIM);
    MPSMatrix* gateM = matrix(gate, token_count, INTERMEDIATE);
    MPSMatrix* upM = matrix(up, token_count, INTERMEDIATE);
    MPSMatrix* middleM = matrix(middle, token_count, INTERMEDIATE);
    MPSMatrix* downM = matrix(down, token_count, HIDDEN_DIM);

    auto multiplication = [&](int output, int input) {
        return [[MPSMatrixMultiplication alloc] initWithDevice:device
            transposeLeft:NO transposeRight:YES resultRows:token_count resultColumns:output
            interiorColumns:input alpha:1.0 beta:0.0];
    };
    MPSMatrixMultiplication* hh = multiplication(HIDDEN_DIM, HIDDEN_DIM);
    MPSMatrixMultiplication* hkv = multiplication(NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
    MPSMatrixMultiplication* hi = multiplication(INTERMEDIATE, HIDDEN_DIM);
    MPSMatrixMultiplication* ih = multiplication(HIDDEN_DIM, INTERMEDIATE);
    auto multiply = [&](id<MTLCommandBuffer> command, MPSMatrixMultiplication* operation,
                        MPSMatrix* input, MPSMatrix* weight, MPSMatrix* output) {
        [operation encodeToCommandBuffer:command leftMatrix:input rightMatrix:weight resultMatrix:output];
    };

    id<MTLCommandBuffer> command = [eng->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    uint tokens = (uint)token_count;
    [encoder setComputePipelineState:eng->pipeEmbeddingBatch];
    [encoder setBuffer:tokenBuffer offset:0 atIndex:0];
    [encoder setBuffer:eng->embed offset:0 atIndex:1];
    [encoder setBuffer:hidden offset:0 atIndex:2];
    [encoder setBytes:&tokens length:sizeof(tokens) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(tokens * HIDDEN_DIM, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];

    auto rmsBatch = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> input,
                        id<MTLBuffer> weight, id<MTLBuffer> output) {
        [enc setComputePipelineState:eng->pipeRmsBatch];
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:weight offset:0 atIndex:1];
        [enc setBuffer:output offset:0 atIndex:2];
        uint D = HIDDEN_DIM; [enc setBytes:&D length:sizeof(D) atIndex:3];
        [enc setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto biasBatch = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> output,
                         id<MTLBuffer> bias, uint D) {
        [enc setComputePipelineState:eng->pipeBiasBatch];
        [enc setBuffer:output offset:0 atIndex:0]; [enc setBuffer:bias offset:0 atIndex:1];
        [enc setBytes:&D length:sizeof(D) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(tokens * D, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };
    auto residualBatch = [&](id<MTLComputeCommandEncoder> enc, id<MTLBuffer> stream,
                             id<MTLBuffer> residual) {
        [enc setComputePipelineState:eng->pipeResidualBatch];
        [enc setBuffer:stream offset:0 atIndex:0]; [enc setBuffer:residual offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(tokens * HIDDEN_DIM, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    };

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        auto& lw = eng->layers[layer];
        encoder = [command computeCommandEncoder];
        rmsBatch(encoder, hidden, lw.input_norm, normed);
        [encoder endEncoding];
        multiply(command, hh, normedM, lw.q_matrix, qM);
        multiply(command, hkv, normedM, lw.k_matrix, kM);
        multiply(command, hkv, normedM, lw.v_matrix, vM);

        encoder = [command computeCommandEncoder];
        biasBatch(encoder, q, lw.q_bias, HIDDEN_DIM);
        biasBatch(encoder, k, lw.k_bias, NUM_KV_HEADS * HEAD_DIM);
        biasBatch(encoder, v, lw.v_bias, NUM_KV_HEADS * HEAD_DIM);
        [encoder setComputePipelineState:eng->pipeRopeBatch];
        [encoder setBuffer:q offset:0 atIndex:0]; [encoder setBuffer:k offset:0 atIndex:1];
        [encoder setBuffer:qRotated offset:0 atIndex:2]; [encoder setBytes:&tokens length:4 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(tokens * (NUM_HEADS + NUM_KV_HEADS) * HEAD_DIM / 2, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:eng->pipeKVBatch];
        [encoder setBuffer:k offset:0 atIndex:0]; [encoder setBuffer:v offset:0 atIndex:1];
        [encoder setBuffer:eng->k_cache_gpu[layer] offset:0 atIndex:2];
        [encoder setBuffer:eng->v_cache_gpu[layer] offset:0 atIndex:3];
        [encoder setBytes:&tokens length:4 atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(tokens * NUM_KV_HEADS * HEAD_DIM, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:eng->pipeCausalScores];
        [encoder setBuffer:qRotated offset:0 atIndex:0];
        [encoder setBuffer:eng->k_cache_gpu[layer] offset:0 atIndex:1];
        [encoder setBuffer:scores offset:0 atIndex:2]; [encoder setBytes:&tokens length:4 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(tokens, NUM_HEADS, tokens)
             threadsPerThreadgroup:MTLSizeMake(8, 2, 2)];
        [encoder setComputePipelineState:eng->pipeCausalSoftmax];
        [encoder setBuffer:scores offset:0 atIndex:0]; [encoder setBytes:&tokens length:4 atIndex:1];
        [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(NUM_HEADS, tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:eng->pipeCausalWeighted];
        [encoder setBuffer:scores offset:0 atIndex:0];
        [encoder setBuffer:eng->v_cache_gpu[layer] offset:0 atIndex:1];
        [encoder setBuffer:attention offset:0 atIndex:2]; [encoder setBytes:&tokens length:4 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(HEAD_DIM, NUM_HEADS, tokens)
             threadsPerThreadgroup:MTLSizeMake(HEAD_DIM, 1, 1)];
        [encoder endEncoding];

        multiply(command, hh, attentionM, lw.o_matrix, attentionProjectedM);
        encoder = [command computeCommandEncoder];
        residualBatch(encoder, hidden, attentionProjected);
        rmsBatch(encoder, hidden, lw.post_norm, normed2);
        [encoder endEncoding];
        multiply(command, hi, normed2M, lw.gate_matrix, gateM);
        multiply(command, hi, normed2M, lw.up_matrix, upM);
        encoder = [command computeCommandEncoder];
        encodeKernel(encoder, eng->pipeSiluHalf, @[gate], tokens * INTERMEDIATE);
        encodeKernel(encoder, eng->pipeMulHalf, @[gate, up, middle], tokens * INTERMEDIATE);
        [encoder endEncoding];
        multiply(command, ih, middleM, lw.down_matrix, downM);
        encoder = [command computeCommandEncoder];
        residualBatch(encoder, hidden, down);
        [encoder endEncoding];
    }

    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:eng->pipeRmsNorm];
    [encoder setBuffer:hidden offset:(token_count - 1) * HIDDEN_DIM * sizeof(uint16_t) atIndex:0];
    [encoder setBuffer:eng->final_norm offset:0 atIndex:1];
    [encoder setBuffer:eng->final_hidden offset:0 atIndex:2];
    [encoder setBuffer:eng->hidden_dim_const_buf offset:0 atIndex:3];
    [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encodeMatvec(encoder, eng->pipeMatvec, eng->lm_head, eng->final_hidden,
                 eng->logits_buf, VOCAB_SIZE, HIDDEN_DIM);
    finishAndWait(command, encoder);

    return selectTokenFromLogits(eng, sampler);
}

int forward_step(QwenEngine* eng,
                  int token_id, int current_pos,
                  std::vector<std::vector<uint16_t>>& k_cache,
                  std::vector<std::vector<uint16_t>>& v_cache,
                  SamplerState* sampler) {
    id<MTLCommandQueue> queue = eng->queue;

    {
        uint16_t* hptr = (uint16_t*)[eng->hidden contents];
        uint16_t* eptr = (uint16_t*)[eng->embed contents] + token_id * HIDDEN_DIM;
        memcpy(hptr, eptr, HIDDEN_DIM * sizeof(uint16_t));
    }

    for (int l = 0; l < NUM_LAYERS; l++) {
        auto& lw = eng->layers[l];

        id<MTLCommandBuffer> qkvCmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> qkvEnc = [qkvCmd computeCommandEncoder];
        encodeRmsNorm(qkvEnc, eng->pipeRmsNorm, eng->hidden, lw.input_norm,
                      eng->normed, eng->hidden_dim_const_buf);
        if (eng->backend == QWEN_BACKEND_MPS_FP16) {
            [qkvEnc endEncoding];
            encodeMPS(qkvCmd, eng->mpsHH, lw.q_matrix, eng->normedVec, eng->qVec);
            encodeMPS(qkvCmd, eng->mpsHKV, lw.k_matrix, eng->normedVec, eng->kVec);
            encodeMPS(qkvCmd, eng->mpsHKV, lw.v_matrix, eng->normedVec, eng->vVec);
            qkvEnc = [qkvCmd computeCommandEncoder];
        } else {
            encodeProjection(eng, qkvEnc, lw.q_proj, lw.q_scale, eng->normed, eng->q_buf,
                             NUM_HEADS * HEAD_DIM, HIDDEN_DIM);
            encodeProjection(eng, qkvEnc, lw.k_proj, lw.k_scale, eng->normed, eng->k_buf,
                             NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            encodeProjection(eng, qkvEnc, lw.v_proj, lw.v_scale, eng->normed, eng->v_buf,
                             NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
        }
        encodeKernel(qkvEnc, eng->pipeResAdd, @[eng->q_buf, lw.q_bias], NUM_HEADS * HEAD_DIM);
        encodeKernel(qkvEnc, eng->pipeResAdd, @[eng->k_buf, lw.k_bias], NUM_KV_HEADS * HEAD_DIM);
        encodeKernel(qkvEnc, eng->pipeResAdd, @[eng->v_buf, lw.v_bias], NUM_KV_HEADS * HEAD_DIM);
        finishAndWait(qkvCmd, qkvEnc);

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

        std::vector<float> q_f = toFloatVec(eng->q_buf, NUM_HEADS * HEAD_DIM);
        std::vector<float> k_f = toFloatVec(eng->k_buf, NUM_KV_HEADS * HEAD_DIM);
        std::vector<float> v_f = toFloatVec(eng->v_buf, NUM_KV_HEADS * HEAD_DIM);

        for (int h = 0; h < NUM_HEADS; h++) {
            std::vector<float> q_h(q_f.begin() + h*HEAD_DIM, q_f.begin() + (h+1)*HEAD_DIM);
            apply_rope(q_h, current_pos, HEAD_DIM, QWEN_ROPE_THETA);
            std::copy(q_h.begin(), q_h.end(), q_f.begin() + h*HEAD_DIM);
        }
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            std::vector<float> k_h(k_f.begin() + h*HEAD_DIM, k_f.begin() + (h+1)*HEAD_DIM);
            apply_rope(k_h, current_pos, HEAD_DIM, QWEN_ROPE_THETA);
            std::copy(k_h.begin(), k_h.end(), k_f.begin() + h*HEAD_DIM);
        }

        std::vector<uint16_t> k_half = toHalfVec(k_f);
        std::vector<uint16_t> v_half = toHalfVec(v_f);
        k_cache[l].insert(k_cache[l].end(), k_half.begin(), k_half.end());
        v_cache[l].insert(v_cache[l].end(), v_half.begin(), v_half.end());

        int num_tokens = (int)k_cache[l].size() / (NUM_KV_HEADS * HEAD_DIM);
        std::vector<float> attn_out(NUM_HEADS * HEAD_DIM, 0.0f);

        for (int h = 0; h < NUM_HEADS; h++) {
            int kv_head = h / (NUM_HEADS / NUM_KV_HEADS);
            std::vector<float> q_h(q_f.begin() + h*HEAD_DIM, q_f.begin() + (h+1)*HEAD_DIM);
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

        {
            uint16_t* ptr = (uint16_t*)[eng->attn_buf contents];
            for (int i = 0; i < NUM_HEADS*HEAD_DIM; i++) {
                __fp16 h2 = (__fp16)attn_out[i];
                memcpy(&ptr[i], &h2, sizeof(h2));
            }
        }


        id<MTLCommandBuffer> postAttnCmd = [queue commandBuffer];
        if (eng->backend == QWEN_BACKEND_MPS_FP16) {
            encodeMPS(postAttnCmd, eng->mpsHH, lw.o_matrix, eng->attnVec, eng->attnProjVec);
            id<MTLComputeCommandEncoder> enc = [postAttnCmd computeCommandEncoder];
            encodeKernel(enc, eng->pipeResAdd, @[eng->hidden, eng->attn_proj], HIDDEN_DIM);
            encodeRmsNorm(enc, eng->pipeRmsNorm, eng->hidden, lw.post_norm,
                          eng->normed2, eng->hidden_dim_const_buf);
            [enc endEncoding];

            encodeMPS(postAttnCmd, eng->mpsHI, lw.gate_matrix, eng->normed2Vec, eng->gateHalfVec);
            encodeMPS(postAttnCmd, eng->mpsHI, lw.up_matrix, eng->normed2Vec, eng->upHalfVec);
            enc = [postAttnCmd computeCommandEncoder];
            encodeKernel(enc, eng->pipeSiluHalf, @[eng->gate_half], INTERMEDIATE);
            encodeKernel(enc, eng->pipeMulHalf,
                         @[eng->gate_half, eng->up_half, eng->mlp_mid_half], INTERMEDIATE);
            [enc endEncoding];

            encodeMPS(postAttnCmd, eng->mpsIH, lw.down_matrix, eng->mlpMidHalfVec, eng->downHalfVec);
            enc = [postAttnCmd computeCommandEncoder];
            encodeKernel(enc, eng->pipeResAdd, @[eng->hidden, eng->down_half], HIDDEN_DIM);
            finishAndWait(postAttnCmd, enc);
            continue;
        }
        id<MTLComputeCommandEncoder> postAttnEnc = [postAttnCmd computeCommandEncoder];
        encodeProjection(eng, postAttnEnc, lw.o_proj, lw.o_scale, eng->attn_buf, eng->attn_proj,
                         HIDDEN_DIM, NUM_HEADS * HEAD_DIM);
        encodeKernel(postAttnEnc, eng->pipeResAdd, @[eng->hidden, eng->attn_proj], HIDDEN_DIM);
        encodeRmsNorm(postAttnEnc, eng->pipeRmsNorm, eng->hidden, lw.post_norm,
                      eng->normed2, eng->hidden_dim_const_buf);
        encodeKernel(postAttnEnc, eng->pipeHalfToFloat,
                     @[eng->normed2, eng->x_mlp_buf], HIDDEN_DIM);

        id<MTLComputeCommandEncoder> mlpEnc = postAttnEnc;
        uint gateK = HIDDEN_DIM;
        constexpr uint gsize = 128;
        if (eng->backend == QWEN_BACKEND_METAL_FP16) {
            [mlpEnc setComputePipelineState:eng->pipeGateUp];
            [mlpEnc setBuffer:lw.gate offset:0 atIndex:0];
            [mlpEnc setBuffer:lw.up offset:0 atIndex:1];
            [mlpEnc setBuffer:eng->x_mlp_buf offset:0 atIndex:2];
            [mlpEnc setBuffer:eng->gate_out_buf offset:0 atIndex:3];
            [mlpEnc setBuffer:eng->up_out_buf offset:0 atIndex:4];
            [mlpEnc setBytes:&gateK length:sizeof(gateK) atIndex:5];
        } else if (eng->backend == QWEN_BACKEND_METAL_INT8) {
            [mlpEnc setComputePipelineState:eng->pipeGateUpQ8];
            [mlpEnc setBuffer:lw.gate offset:0 atIndex:0];
            [mlpEnc setBuffer:lw.gate_scale offset:0 atIndex:1];
            [mlpEnc setBuffer:lw.up offset:0 atIndex:2];
            [mlpEnc setBuffer:lw.up_scale offset:0 atIndex:3];
            [mlpEnc setBuffer:eng->x_mlp_buf offset:0 atIndex:4];
            [mlpEnc setBuffer:eng->gate_out_buf offset:0 atIndex:5];
            [mlpEnc setBuffer:eng->up_out_buf offset:0 atIndex:6];
            [mlpEnc setBytes:&gateK length:sizeof(gateK) atIndex:7];
        } else {
            [mlpEnc setComputePipelineState:eng->pipeGateUpQ4];
            [mlpEnc setBuffer:lw.gate offset:0 atIndex:0];
            [mlpEnc setBuffer:lw.gate_scale offset:0 atIndex:1];
            [mlpEnc setBuffer:lw.up offset:0 atIndex:2];
            [mlpEnc setBuffer:lw.up_scale offset:0 atIndex:3];
            [mlpEnc setBuffer:eng->x_mlp_buf offset:0 atIndex:4];
            [mlpEnc setBuffer:eng->gate_out_buf offset:0 atIndex:5];
            [mlpEnc setBuffer:eng->up_out_buf offset:0 atIndex:6];
            [mlpEnc setBytes:&gateK length:sizeof(gateK) atIndex:7];
        }
        [mlpEnc setThreadgroupMemoryLength:(gsize / 32) * 2 * sizeof(float) atIndex:0];
        [mlpEnc dispatchThreadgroups:MTLSizeMake(INTERMEDIATE, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(gsize, 1, 1)];

        encodeKernel(mlpEnc, eng->pipeSiluMul,
                     @[eng->gate_out_buf, eng->up_out_buf, eng->mlp_mid_buf], INTERMEDIATE);

        uint downK = INTERMEDIATE;
        if (eng->backend == QWEN_BACKEND_METAL_FP16) {
            [mlpEnc setComputePipelineState:eng->pipeDown];
            [mlpEnc setBuffer:lw.down offset:0 atIndex:0];
            [mlpEnc setBuffer:eng->mlp_mid_buf offset:0 atIndex:1];
            [mlpEnc setBuffer:eng->down_out_buf offset:0 atIndex:2];
            [mlpEnc setBytes:&downK length:sizeof(downK) atIndex:3];
        } else if (eng->backend == QWEN_BACKEND_METAL_INT8) {
            [mlpEnc setComputePipelineState:eng->pipeDownQ8];
            [mlpEnc setBuffer:lw.down offset:0 atIndex:0];
            [mlpEnc setBuffer:lw.down_scale offset:0 atIndex:1];
            [mlpEnc setBuffer:eng->mlp_mid_buf offset:0 atIndex:2];
            [mlpEnc setBuffer:eng->down_out_buf offset:0 atIndex:3];
            [mlpEnc setBytes:&downK length:sizeof(downK) atIndex:4];
        } else {
            [mlpEnc setComputePipelineState:eng->pipeDownQ4];
            [mlpEnc setBuffer:lw.down offset:0 atIndex:0];
            [mlpEnc setBuffer:lw.down_scale offset:0 atIndex:1];
            [mlpEnc setBuffer:eng->mlp_mid_buf offset:0 atIndex:2];
            [mlpEnc setBuffer:eng->down_out_buf offset:0 atIndex:3];
            [mlpEnc setBytes:&downK length:sizeof(downK) atIndex:4];
        }
        [mlpEnc setThreadgroupMemoryLength:(gsize / 32) * sizeof(float) atIndex:0];
        [mlpEnc dispatchThreadgroups:MTLSizeMake(HIDDEN_DIM, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(gsize, 1, 1)];
        encodeKernel(mlpEnc, eng->pipeResAddFloat,
                     @[eng->hidden, eng->down_out_buf], HIDDEN_DIM);
        finishAndWait(postAttnCmd, postAttnEnc);
    }

    id<MTLCommandBuffer> outputCmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> outputEnc = [outputCmd computeCommandEncoder];
    encodeRmsNorm(outputEnc, eng->pipeRmsNorm, eng->hidden, eng->final_norm,
                  eng->final_hidden, eng->hidden_dim_const_buf);
    if (eng->backend == QWEN_BACKEND_MPS_FP16) {
        [outputEnc endEncoding];
        encodeMPS(outputCmd, eng->mpsHVocab, eng->lm_head_matrix,
                  eng->finalHiddenVec, eng->logitsVec);
        [outputCmd commit];
        [outputCmd waitUntilCompleted];
    } else if (eng->backend == QWEN_BACKEND_INT4_FP16_LM_HEAD) {
        encodeMatvec(outputEnc, eng->pipeMatvec, eng->lm_head,
                     eng->final_hidden, eng->logits_buf, VOCAB_SIZE, HIDDEN_DIM);
        finishAndWait(outputCmd, outputEnc);
    } else {
        encodeProjection(eng, outputEnc, eng->lm_head, eng->lm_head_scale,
                         eng->final_hidden, eng->logits_buf, VOCAB_SIZE, HIDDEN_DIM);
        finishAndWait(outputCmd, outputEnc);
    }

    return selectTokenFromLogits(eng, sampler);
}

} 

extern "C" {

QwenEngine* qwen_engine_create_with_backend(const char* weights_dir,
                                             QwenBackend backend,
                                             bool verbose) {
    auto* eng = new QwenEngine();
    eng->verbose = verbose;
    eng->backend = backend;

    eng->device = MTLCreateSystemDefaultDevice();
    if (!eng->device) { LOG(eng, "[qwen_engine] no Metal device found"); delete eng; return nullptr; }
    eng->queue = [eng->device newCommandQueue];

    id<MTLLibrary> library = [eng->device newDefaultLibrary];
    if (!library) { LOG(eng, "[qwen_engine] default.metallib not found"); delete eng; return nullptr; }

    static const char* optimizedMetalSource = R"METAL(
        #include <metal_stdlib>
        using namespace metal;
        kernel void matvec_q4(device const uchar* w [[buffer(0)]], device const half* s [[buffer(1)]],
                              device const half* x [[buffer(2)]], device half* y [[buffer(3)]],
                              constant uint& K [[buffer(4)]], threadgroup float* p [[threadgroup(0)]],
                              uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]]) {
            uint nb=K/2, ng=K/64; float sum=0;
            for(uint b=tid;b<nb;b+=nt){uchar v=w[row*nb+b];float z=float(s[row*ng+b/32]);
                sum+=z*(float(int(v&15)-8)*float(x[2*b])+float(int(v>>4)-8)*float(x[2*b+1]));}
            sum=simd_sum(sum);if((tid&31u)==0)p[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float t=0;for(uint i=0;i<nt/32;i++)t+=p[i];y[row]=half(t);}
        }
        kernel void gate_up_q4(device const uchar* gw [[buffer(0)]],device const half* gs [[buffer(1)]],
                               device const uchar* uw [[buffer(2)]],device const half* us [[buffer(3)]],
                               device const float* x [[buffer(4)]],device float* go [[buffer(5)]],
                               device float* uo [[buffer(6)]],constant uint& K [[buffer(7)]],
                               threadgroup float* p [[threadgroup(0)]],uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],uint nt [[threads_per_threadgroup]]) {
            uint nb=K/2,ng=K/64;float a=0,bv=0;
            for(uint j=tid;j<nb;j+=nt){uchar g=gw[row*nb+j],u=uw[row*nb+j];uint z=row*ng+j/32;
                float x0=x[2*j],x1=x[2*j+1];a+=float(gs[z])*(float(int(g&15)-8)*x0+float(int(g>>4)-8)*x1);
                bv+=float(us[z])*(float(int(u&15)-8)*x0+float(int(u>>4)-8)*x1);}
            a=simd_sum(a);bv=simd_sum(bv);if((tid&31u)==0){p[(tid/32)*2]=a;p[(tid/32)*2+1]=bv;}
            threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float ga=0,ua=0;for(uint i=0;i<nt/32;i++){ga+=p[i*2];ua+=p[i*2+1];}go[row]=ga;uo[row]=ua;}
        }
        kernel void down_q4(device const uchar* w [[buffer(0)]],device const half* s [[buffer(1)]],
                            device const float* x [[buffer(2)]],device float* y [[buffer(3)]],
                            constant uint& K [[buffer(4)]],threadgroup float* p [[threadgroup(0)]],
                            uint row [[threadgroup_position_in_grid]],uint tid [[thread_position_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
            uint nb=K/2,ng=K/64;float sum=0;for(uint j=tid;j<nb;j+=nt){uchar v=w[row*nb+j];float z=float(s[row*ng+j/32]);
                sum+=z*(float(int(v&15)-8)*x[2*j]+float(int(v>>4)-8)*x[2*j+1]);}
            sum=simd_sum(sum);if((tid&31u)==0)p[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float t=0;for(uint i=0;i<nt/32;i++)t+=p[i];y[row]=t;}
        }
        kernel void matvec_q8(device const char* w [[buffer(0)]], device const half* s [[buffer(1)]],
                              device const half* x [[buffer(2)]], device half* y [[buffer(3)]],
                              constant uint& K [[buffer(4)]], threadgroup float* p [[threadgroup(0)]],
                              uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]]) {
            uint ng=K/64; float sum=0;
            for(uint j=tid;j<K;j+=nt){float z=float(s[row*ng+j/64]);
                sum+=z*float(w[row*K+j])*float(x[j]);}
            sum=simd_sum(sum);if((tid&31u)==0)p[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float t=0;for(uint i=0;i<nt/32;i++)t+=p[i];y[row]=half(t);}
        }
        kernel void gate_up_q8(device const char* gw [[buffer(0)]],device const half* gs [[buffer(1)]],
                               device const char* uw [[buffer(2)]],device const half* us [[buffer(3)]],
                               device const float* x [[buffer(4)]],device float* go [[buffer(5)]],
                               device float* uo [[buffer(6)]],constant uint& K [[buffer(7)]],
                               threadgroup float* p [[threadgroup(0)]],uint row [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],uint nt [[threads_per_threadgroup]]) {
            uint ng=K/64;float a=0,bv=0;
            for(uint j=tid;j<K;j+=nt){uint z=row*ng+j/64;float xj=x[j];
                a+=float(gs[z])*float(gw[row*K+j])*xj;bv+=float(us[z])*float(uw[row*K+j])*xj;}
            a=simd_sum(a);bv=simd_sum(bv);if((tid&31u)==0){p[(tid/32)*2]=a;p[(tid/32)*2+1]=bv;}
            threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0){float ga=0,ua=0;for(uint i=0;i<nt/32;i++){ga+=p[i*2];ua+=p[i*2+1];}go[row]=ga;uo[row]=ua;}
        }
        kernel void down_q8(device const char* w [[buffer(0)]],device const half* s [[buffer(1)]],
                            device const float* x [[buffer(2)]],device float* y [[buffer(3)]],
                            constant uint& K [[buffer(4)]],threadgroup float* p [[threadgroup(0)]],
                            uint row [[threadgroup_position_in_grid]],uint tid [[thread_position_in_threadgroup]],
                            uint nt [[threads_per_threadgroup]]) {
            uint ng=K/64;float sum=0;for(uint j=tid;j<K;j+=nt){float z=float(s[row*ng+j/64]);
                sum+=z*float(w[row*K+j])*x[j];}
            sum=simd_sum(sum);if((tid&31u)==0)p[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float t=0;for(uint i=0;i<nt/32;i++)t+=p[i];y[row]=t;}
        }
        kernel void rope_qk_inplace(device const half* q [[buffer(0)]],device half* k [[buffer(1)]],
                                    device float* qr [[buffer(2)]],constant uint& pos [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {
            constexpr uint hd=32,qh=14,kh=2;if(id>=(qh+kh)*hd)return;bool iq=id<qh*hd;
            uint z=iq?id:id-qh*hd,h=z/hd,i=z%hd,b=h*64;
            float angle=float(pos)*pow(1000000.0f,-float(2*i)/64.0f),c=cos(angle),s=sin(angle);
            float x=iq?float(q[b+i]):float(k[b+i]),y=iq?float(q[b+i+hd]):float(k[b+i+hd]);
            if(iq){qr[b+i]=x*c-y*s;qr[b+i+hd]=y*c+x*s;}else{k[b+i]=half(x*c-y*s);k[b+i+hd]=half(y*c+x*s);}
        }
        kernel void kv_cache_append(device const half* k [[buffer(0)]],device const half* v [[buffer(1)]],
                                    device half* kc [[buffer(2)]],device half* vc [[buffer(3)]],
                                    constant uint& pos [[buffer(4)]],uint id [[thread_position_in_grid]]) {
            if(id<128){uint o=pos*128+id;kc[o]=k[id];vc[o]=v[id];}
        }
        kernel void gqa_attention_scores(device const float* q [[buffer(0)]],device const half* kc [[buffer(1)]],
                                         device float* scores [[buffer(2)]],constant uint& n [[buffer(3)]],
                                         uint3 g [[threadgroup_position_in_grid]],
                                         uint3 local [[thread_position_in_threadgroup]]) {
            uint t=g.x,h=g.y,lane=local.x;if(t>=n||h>=14)return;uint kh=h/7;float sum=0;
            for(uint d=lane;d<64;d+=32)sum+=q[h*64+d]*float(kc[t*128+kh*64+d]);
            sum=simd_sum(sum);if(lane==0)scores[h*n+t]=sum*0.125f;
        }
        kernel void gqa_softmax(device float* scores [[buffer(0)]],constant uint& n [[buffer(1)]],
                                threadgroup float* p [[threadgroup(0)]],uint h [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],uint nt [[threads_per_threadgroup]]) {
            if(h>=14||n==0)return;uint b=h*n;float m=-INFINITY;for(uint t=tid;t<n;t+=nt)m=max(m,scores[b+t]);
            m=simd_max(m);if((tid&31u)==0)p[tid/32]=m;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float z=p[0];for(uint i=1;i<nt/32;i++)z=max(z,p[i]);p[0]=z;}threadgroup_barrier(mem_flags::mem_threadgroup);m=p[0];
            float s=0;for(uint t=tid;t<n;t+=nt){float v=exp(scores[b+t]-m);scores[b+t]=v;s+=v;}
            s=simd_sum(s);if((tid&31u)==0)p[tid/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
            if(tid==0){float z=0;for(uint i=0;i<nt/32;i++)z+=p[i];p[0]=1.0f/z;}threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv=p[0];for(uint t=tid;t<n;t+=nt)scores[b+t]*=inv;
        }
        kernel void gqa_weighted_sum(device const float* scores [[buffer(0)]],device const half* vc [[buffer(1)]],
                                     device half* out [[buffer(2)]],constant uint& n [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]]) {
            uint d=gid.x,h=gid.y;if(d>=64||h>=14)return;uint kh=h/7;float sum=0;
            for(uint t=0;t<n;t++)sum+=scores[h*n+t]*float(vc[t*128+kh*64+d]);out[h*64+d]=half(sum);
        }
        kernel void rms_norm_fast(device const half* x [[buffer(0)]],
                                  device const half* weight [[buffer(1)]],
                                  device half* y [[buffer(2)]],
                                  constant uint& D [[buffer(3)]],
                                  threadgroup float* partial [[threadgroup(0)]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]]) {
            float sum = 0.0f;
            for (uint i = tid; i < D; i += threads) { float v = float(x[i]); sum += v * v; }
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
        kernel void silu_mul_float(device const float* gate [[buffer(0)]],
                                   device const float* up [[buffer(1)]],
                                   device float* out [[buffer(2)]],
                                   uint id [[thread_position_in_grid]]) {
            float g = gate[id];
            out[id] = (g / (1.0f + exp(-g))) * up[id];
        }
        kernel void half_to_float(device const half* input [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
            output[id] = float(input[id]);
        }
        kernel void residual_add_float(device half* x [[buffer(0)]],
                                       device const float* r [[buffer(1)]],
                                       uint id [[thread_position_in_grid]]) {
            x[id] = half(float(x[id]) + float(half(r[id])));
        }
    )METAL";
    NSError* compileError = nil;
    id<MTLLibrary> optimizedLibrary = [eng->device
        newLibraryWithSource:[NSString stringWithUTF8String:optimizedMetalSource]
        options:nil error:&compileError];
    if (!optimizedLibrary) {
        LOG(eng, "[qwen_engine] optimized Metal compile failed: " <<
            [[compileError localizedDescription] UTF8String]);
        delete eng; return nullptr;
    }

    id<MTLLibrary> sourceLibrary = nil;
    if (backendUsesBatchedPrefill(backend)) {
        std::ifstream sourceFile("kernels.metal");
        std::string sourceText((std::istreambuf_iterator<char>(sourceFile)),
                               std::istreambuf_iterator<char>());
        if (sourceText.empty()) {
            LOG(eng, "[qwen_engine] kernels.metal is required for batched prefill");
            delete eng; return nullptr;
        }
        NSError* sourceError = nil;
        sourceLibrary = [eng->device
            newLibraryWithSource:[NSString stringWithUTF8String:sourceText.c_str()]
            options:nil error:&sourceError];
        if (!sourceLibrary) {
            LOG(eng, "[qwen_engine] kernels.metal compile failed: " <<
                [[sourceError localizedDescription] UTF8String]);
            delete eng; return nullptr;
        }
    }

    auto loadPipe = [&](NSString* name) -> id<MTLComputePipelineState> {
        return [eng->device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:nil];
    };
    auto loadOptimizedPipe = [&](NSString* name) -> id<MTLComputePipelineState> {
        return [eng->device newComputePipelineStateWithFunction:
                [optimizedLibrary newFunctionWithName:name] error:nil];
    };
    auto loadSourcePipe = [&](NSString* name) -> id<MTLComputePipelineState> {
        if (!sourceLibrary) return nil;
        return [eng->device newComputePipelineStateWithFunction:
                [sourceLibrary newFunctionWithName:name] error:nil];
    };
    eng->pipeRmsNorm = loadOptimizedPipe(@"rms_norm_fast");
    eng->pipeResAdd  = loadPipe(@"residual_add");
    eng->pipeMatvec  = loadPipe(@"matvec_float4");
    eng->pipeGateUp = loadPipe(@"matvec_gate_up_batched");
    eng->pipeHalfToFloat = loadOptimizedPipe(@"half_to_float");
    eng->pipeSiluMul = loadOptimizedPipe(@"silu_mul_float");
    eng->pipeDown = loadPipe(@"matvec_down_batched");
    eng->pipeResAddFloat = loadOptimizedPipe(@"residual_add_float");
    eng->pipeMatvecQ4 = loadOptimizedPipe(@"matvec_q4");
    eng->pipeGateUpQ4 = loadOptimizedPipe(@"gate_up_q4");
    eng->pipeDownQ4 = loadOptimizedPipe(@"down_q4");
    eng->pipeMatvecQ8 = loadOptimizedPipe(@"matvec_q8");
    eng->pipeGateUpQ8 = loadOptimizedPipe(@"gate_up_q8");
    eng->pipeDownQ8 = loadOptimizedPipe(@"down_q8");
    eng->pipeSiluHalf = loadPipe(@"silu_inplace");
    eng->pipeMulHalf = loadPipe(@"element_mul");
    eng->pipeRopeQK = loadOptimizedPipe(@"rope_qk_inplace");
    eng->pipeKVAppend = loadOptimizedPipe(@"kv_cache_append");
    eng->pipeAttnScores = loadOptimizedPipe(@"gqa_attention_scores");
    eng->pipeAttnSoftmax = loadOptimizedPipe(@"gqa_softmax");
    eng->pipeAttnWeighted = loadOptimizedPipe(@"gqa_weighted_sum");
    if (backendUsesBatchedPrefill(backend)) {
        eng->pipeEmbeddingBatch = loadSourcePipe(@"embedding_batch");
        eng->pipeRmsBatch = loadSourcePipe(@"rms_norm_batch");
        eng->pipeBiasBatch = loadSourcePipe(@"bias_add_batch");
        eng->pipeResidualBatch = loadSourcePipe(@"residual_add_batch");
        eng->pipeRopeBatch = loadSourcePipe(@"rope_qk_batch");
        eng->pipeKVBatch = loadSourcePipe(@"kv_cache_batch");
        eng->pipeCausalScores = loadSourcePipe(@"causal_scores_batch");
        eng->pipeCausalSoftmax = loadSourcePipe(@"causal_softmax_batch");
        eng->pipeCausalWeighted = loadSourcePipe(@"causal_weighted_batch");
    }
    if (!eng->pipeRmsNorm || !eng->pipeResAdd || !eng->pipeMatvec ||
        !eng->pipeGateUp || !eng->pipeHalfToFloat || !eng->pipeSiluMul ||
        !eng->pipeDown || !eng->pipeResAddFloat || !eng->pipeMatvecQ4 ||
        !eng->pipeGateUpQ4 || !eng->pipeDownQ4 || !eng->pipeMatvecQ8 ||
        !eng->pipeGateUpQ8 || !eng->pipeDownQ8 || !eng->pipeSiluHalf || !eng->pipeMulHalf ||
        !eng->pipeRopeQK || !eng->pipeKVAppend || !eng->pipeAttnScores ||
        !eng->pipeAttnSoftmax || !eng->pipeAttnWeighted) {
        LOG(eng, "[qwen_engine] failed to load one or more Metal pipelines");
        delete eng; return nullptr;
    }
    if (backendUsesBatchedPrefill(backend) &&
        (!eng->pipeEmbeddingBatch || !eng->pipeRmsBatch || !eng->pipeBiasBatch ||
         !eng->pipeResidualBatch || !eng->pipeRopeBatch || !eng->pipeKVBatch ||
         !eng->pipeCausalScores || !eng->pipeCausalSoftmax || !eng->pipeCausalWeighted)) {
        LOG(eng, "[qwen_engine] failed to load batched prefill pipelines");
        delete eng; return nullptr;
    }

    std::string dir = weights_dir;
    auto loadW = [&](const std::string& fname, bool* ok) {
        return loadHalfBuffer(eng->device, dir + "/" + fname + ".bin", ok);
    };
    auto loadProjection = [&](const std::string& name, bool* ok) {
        std::pair<id<MTLBuffer>, id<MTLBuffer>> result;
        if (backendUsesQ4Projections(backend)) {
            result.first = loadW(name + ".q4", ok);
            result.second = loadW(name + ".scales", ok);
        } else if (backendUsesQ8Projections(backend)) {
            result.first = loadW(name + ".q8", ok);
            result.second = loadW(name + ".q8.scales", ok);
        } else {
            result.first = loadW(name, ok);
            result.second = nil;
        }
        return result;
    };
    auto makeMatrix = [&](id<MTLBuffer> buffer, int rows, int columns) -> MPSMatrix* {
        MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)columns
            rowBytes:(NSUInteger)columns * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    };

    bool ok = true;
    eng->embed      = loadW("embed_tokens.weight", &ok);
    eng->final_norm = loadW("norm.weight", &ok);
    if (backend == QWEN_BACKEND_INT4_FP16_LM_HEAD) {
        eng->lm_head = loadW("lm_head.weight", &ok);
        eng->lm_head_scale = nil;
    } else {
        auto lmHead = loadProjection("lm_head.weight", &ok);
        eng->lm_head = lmHead.first; eng->lm_head_scale = lmHead.second;
    }
    if (backend == QWEN_BACKEND_MPS_FP16)
        eng->lm_head_matrix = makeMatrix(eng->lm_head, VOCAB_SIZE, HIDDEN_DIM);
    if (!ok) { delete eng; return nullptr; }

    eng->layers.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        std::string p = "layer" + std::to_string(i);
        auto& lw = eng->layers[i];
        lw.input_norm = loadW(p + ".input_layernorm.weight", &ok);
        lw.post_norm  = loadW(p + ".post_attention_layernorm.weight", &ok);
        auto q = loadProjection(p + ".self_attn.q_proj.weight", &ok);
        auto k = loadProjection(p + ".self_attn.k_proj.weight", &ok);
        auto v = loadProjection(p + ".self_attn.v_proj.weight", &ok);
        auto o = loadProjection(p + ".self_attn.o_proj.weight", &ok);
        lw.q_proj=q.first; lw.q_scale=q.second;
        lw.k_proj=k.first; lw.k_scale=k.second;
        lw.v_proj=v.first; lw.v_scale=v.second;
        lw.o_proj=o.first; lw.o_scale=o.second;
        lw.q_bias = loadW(p + ".self_attn.q_proj.bias", &ok);
        lw.k_bias = loadW(p + ".self_attn.k_proj.bias", &ok);
        lw.v_bias = loadW(p + ".self_attn.v_proj.bias", &ok);
        auto gate = loadProjection(p + ".mlp.gate_proj.weight", &ok);
        auto up = loadProjection(p + ".mlp.up_proj.weight", &ok);
        auto down = loadProjection(p + ".mlp.down_proj.weight", &ok);
        lw.gate=gate.first; lw.gate_scale=gate.second;
        lw.up=up.first; lw.up_scale=up.second;
        lw.down=down.first; lw.down_scale=down.second;
        if (backend == QWEN_BACKEND_HYBRID) {
            id<MTLBuffer> q16 = loadW(p + ".self_attn.q_proj.weight", &ok);
            id<MTLBuffer> k16 = loadW(p + ".self_attn.k_proj.weight", &ok);
            id<MTLBuffer> v16 = loadW(p + ".self_attn.v_proj.weight", &ok);
            id<MTLBuffer> o16 = loadW(p + ".self_attn.o_proj.weight", &ok);
            id<MTLBuffer> gate16 = loadW(p + ".mlp.gate_proj.weight", &ok);
            id<MTLBuffer> up16 = loadW(p + ".mlp.up_proj.weight", &ok);
            id<MTLBuffer> down16 = loadW(p + ".mlp.down_proj.weight", &ok);
            lw.q_matrix = makeMatrix(q16, HIDDEN_DIM, HIDDEN_DIM);
            lw.k_matrix = makeMatrix(k16, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            lw.v_matrix = makeMatrix(v16, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            lw.o_matrix = makeMatrix(o16, HIDDEN_DIM, HIDDEN_DIM);
            lw.gate_matrix = makeMatrix(gate16, INTERMEDIATE, HIDDEN_DIM);
            lw.up_matrix = makeMatrix(up16, INTERMEDIATE, HIDDEN_DIM);
            lw.down_matrix = makeMatrix(down16, HIDDEN_DIM, INTERMEDIATE);
        } else if (backend == QWEN_BACKEND_METAL_FP16 || backend == QWEN_BACKEND_MPS_FP16) {
            lw.q_matrix = makeMatrix(lw.q_proj, HIDDEN_DIM, HIDDEN_DIM);
            lw.k_matrix = makeMatrix(lw.k_proj, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            lw.v_matrix = makeMatrix(lw.v_proj, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            lw.o_matrix = makeMatrix(lw.o_proj, HIDDEN_DIM, HIDDEN_DIM);
            lw.gate_matrix = makeMatrix(lw.gate, INTERMEDIATE, HIDDEN_DIM);
            lw.up_matrix = makeMatrix(lw.up, INTERMEDIATE, HIDDEN_DIM);
            lw.down_matrix = makeMatrix(lw.down, HIDDEN_DIM, INTERMEDIATE);
        }
        if (!ok) { delete eng; return nullptr; }
        LOG(eng, "[qwen_engine] layer " << i << " loaded");
    }

    uint hd = HIDDEN_DIM;
    eng->hidden_dim_const_buf = [eng->device newBufferWithBytes:&hd length:sizeof(uint)
                                                          options:MTLResourceStorageModeShared];

    auto allocHalf = [&](int n) {
        return [eng->device newBufferWithLength:n * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    };
    eng->hidden      = allocHalf(HIDDEN_DIM);
    eng->normed      = allocHalf(HIDDEN_DIM);
    eng->normed2     = allocHalf(HIDDEN_DIM);
    eng->q_buf       = allocHalf(NUM_HEADS * HEAD_DIM);
    eng->k_buf       = allocHalf(NUM_KV_HEADS * HEAD_DIM);
    eng->v_buf       = allocHalf(NUM_KV_HEADS * HEAD_DIM);
    eng->attn_buf    = allocHalf(NUM_HEADS * HEAD_DIM);
    eng->attn_proj   = allocHalf(HIDDEN_DIM);
    eng->mlp_buf     = allocHalf(HIDDEN_DIM);
    eng->final_hidden = allocHalf(HIDDEN_DIM);
    eng->logits_buf  = allocHalf(VOCAB_SIZE);

    auto allocFloat = [&](int n) {
        return [eng->device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    };
    eng->x_mlp_buf = allocFloat(HIDDEN_DIM);
    eng->gate_out_buf = allocFloat(INTERMEDIATE);
    eng->up_out_buf = allocFloat(INTERMEDIATE);
    eng->mlp_mid_buf = allocFloat(INTERMEDIATE);
    eng->down_out_buf = allocFloat(HIDDEN_DIM);
    eng->x_mlp = (float*)[eng->x_mlp_buf contents];
    eng->gate_out = (float*)[eng->gate_out_buf contents];
    eng->up_out = (float*)[eng->up_out_buf contents];
    eng->mlp_mid = (float*)[eng->mlp_mid_buf contents];
    eng->down_out = (float*)[eng->down_out_buf contents];

    if (backend != QWEN_BACKEND_MPS_FP16) {
        eng->k_cache_gpu.resize(NUM_LAYERS);
        eng->v_cache_gpu.resize(NUM_LAYERS);
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            eng->k_cache_gpu[layer] = allocHalf(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM);
            eng->v_cache_gpu[layer] = allocHalf(MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM);
        }
        eng->attention_scores = allocFloat(NUM_HEADS * MAX_SEQ_LEN);
        eng->q_rotated = allocFloat(NUM_HEADS * HEAD_DIM);
    }

    if (backend == QWEN_BACKEND_MPS_FP16) {
        eng->gate_half = allocHalf(INTERMEDIATE);
        eng->up_half = allocHalf(INTERMEDIATE);
        eng->mlp_mid_half = allocHalf(INTERMEDIATE);
        eng->down_half = allocHalf(HIDDEN_DIM);
        auto makeVector = [&](id<MTLBuffer> buffer, int length) -> MPSVector* {
            MPSVectorDescriptor* descriptor = [MPSVectorDescriptor
                vectorDescriptorWithLength:(NSUInteger)length dataType:MPSDataTypeFloat16];
            return [[MPSVector alloc] initWithBuffer:buffer descriptor:descriptor];
        };
        eng->normedVec = makeVector(eng->normed, HIDDEN_DIM);
        eng->qVec = makeVector(eng->q_buf, HIDDEN_DIM);
        eng->kVec = makeVector(eng->k_buf, NUM_KV_HEADS * HEAD_DIM);
        eng->vVec = makeVector(eng->v_buf, NUM_KV_HEADS * HEAD_DIM);
        eng->attnVec = makeVector(eng->attn_buf, HIDDEN_DIM);
        eng->attnProjVec = makeVector(eng->attn_proj, HIDDEN_DIM);
        eng->normed2Vec = makeVector(eng->normed2, HIDDEN_DIM);
        eng->finalHiddenVec = makeVector(eng->final_hidden, HIDDEN_DIM);
        eng->logitsVec = makeVector(eng->logits_buf, VOCAB_SIZE);
        eng->gateHalfVec = makeVector(eng->gate_half, INTERMEDIATE);
        eng->upHalfVec = makeVector(eng->up_half, INTERMEDIATE);
        eng->mlpMidHalfVec = makeVector(eng->mlp_mid_half, INTERMEDIATE);
        eng->downHalfVec = makeVector(eng->down_half, HIDDEN_DIM);

        eng->mpsHH = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:eng->device rows:HIDDEN_DIM columns:HIDDEN_DIM];
        eng->mpsHKV = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:eng->device rows:NUM_KV_HEADS * HEAD_DIM columns:HIDDEN_DIM];
        eng->mpsHI = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:eng->device rows:INTERMEDIATE columns:HIDDEN_DIM];
        eng->mpsIH = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:eng->device rows:HIDDEN_DIM columns:INTERMEDIATE];
        eng->mpsHVocab = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:eng->device rows:VOCAB_SIZE columns:HIDDEN_DIM];
        if (!eng->mpsHH || !eng->mpsHKV || !eng->mpsHI || !eng->mpsIH || !eng->mpsHVocab) {
            LOG(eng, "[qwen_engine] failed to initialize MPS GEMV kernels");
            delete eng; return nullptr;
        }
    }

    LOG(eng, "[qwen_engine] ready (backend=" <<
        (backend == QWEN_BACKEND_METAL_INT4 ? "int4" :
         backend == QWEN_BACKEND_INT4_FP16_LM_HEAD ? "mixed" :
         backend == QWEN_BACKEND_METAL_INT8 ? "int8" :
         backend == QWEN_BACKEND_MPS_FP16 ? "mps" :
         backend == QWEN_BACKEND_HYBRID ? "hybrid" : "fp16") << ")");
    return eng;
}

QwenEngine* qwen_engine_create(const char* weights_dir, bool verbose) {
    return qwen_engine_create_with_backend(weights_dir, QWEN_BACKEND_METAL_FP16, verbose);
}

void qwen_engine_destroy(QwenEngine* engine) {
    delete engine;
}

void qwen_session_reset(QwenEngine* engine) {
    if (!engine) return;
    engine->session_pos = 0;
    engine->session_tokens.clear();
}

static bool qwen_is_stop_token(int token, const int* stop_tokens, int stop_count) {
    if (!stop_tokens || stop_count <= 0) return false;
    for (int i = 0; i < stop_count; i++) {
        if (token == stop_tokens[i]) return true;
    }
    return false;
}

int qwen_session_generate_streaming_sampled_until(QwenEngine* engine,
                                                  const int* new_prompt_tokens,
                                                  int new_prompt_len,
                                                  int max_new_tokens,
                                                  const int* stop_tokens,
                                                  int stop_count,
                                                  QwenSamplingParams sampling,
                                                  qwen_token_callback callback,
                                                  void* user_data) {
    if (!engine || !new_prompt_tokens || new_prompt_len <= 0 || max_new_tokens <= 0) return -1;
    if (engine->backend == QWEN_BACKEND_MPS_FP16) {
        LOG(engine, "[qwen_engine] session cache is not supported for MPS backend");
        return -1;
    }
    if (engine->session_pos + new_prompt_len + max_new_tokens - 1 > MAX_SEQ_LEN) {
        LOG(engine, "[qwen_engine] session exceeds GPU KV-cache capacity of " << MAX_SEQ_LEN);
        return -1;
    }

    for (int i = 0; i < new_prompt_len; i++) {
        engine->session_tokens.push_back(new_prompt_tokens[i]);
    }
    SamplerState sampler = makeSampler(
        sampling,
        engine->session_tokens.empty() ? nullptr : engine->session_tokens.data(),
        (int)engine->session_tokens.size());

    int next_token = 0;
    for (int i = 0; i < new_prompt_len; i++) {
        SamplerState* step_sampler = (i == new_prompt_len - 1) ? &sampler : nullptr;
        next_token = forward_step_gpu(engine, new_prompt_tokens[i], engine->session_pos++, step_sampler);
    }

    int generated = 0;
    if (callback) callback(next_token, user_data);
    generated++;

    while (generated < max_new_tokens && !qwen_is_stop_token(next_token, stop_tokens, stop_count)) {
        engine->session_tokens.push_back(next_token);
        next_token = forward_step_gpu(engine, next_token, engine->session_pos++, &sampler);
        if (callback) callback(next_token, user_data);
        generated++;
    }
    return generated;
}

int qwen_generate_streaming_sampled_until(QwenEngine* engine,
                                          const int* prompt_tokens, int prompt_len,
                                          int max_new_tokens,
                                          const int* stop_tokens, int stop_count,
                                          QwenSamplingParams sampling,
                                          qwen_token_callback callback, void* user_data) {
    if (!engine || !prompt_tokens || prompt_len <= 0 || max_new_tokens <= 0) return -1;
    if (prompt_len + max_new_tokens - 1 > MAX_SEQ_LEN) {
        LOG(engine, "[qwen_engine] sequence exceeds GPU KV-cache capacity of " << MAX_SEQ_LEN);
        return -1;
    }
    SamplerState sampler = makeSampler(sampling, prompt_tokens, prompt_len);

    if ((engine->backend == QWEN_BACKEND_METAL_FP16 ||
         engine->backend == QWEN_BACKEND_HYBRID) && prompt_len > 1) {
        auto started = std::chrono::high_resolution_clock::now();
        int nextToken = prefill_batch_fp16(engine, prompt_tokens, prompt_len, &sampler);
        auto prefillFinished = std::chrono::high_resolution_clock::now();
        double prefillMs = std::chrono::duration<double, std::milli>(prefillFinished - started).count();
        if (callback) callback(nextToken, user_data);
        int generated = 1;
        int position = prompt_len;
        while (generated < max_new_tokens && !qwen_is_stop_token(nextToken, stop_tokens, stop_count)) {
            nextToken = forward_step_gpu(engine, nextToken, position++, &sampler);
            if (callback) callback(nextToken, user_data);
            generated++;
        }
        auto finished = std::chrono::high_resolution_clock::now();
        double totalMs = std::chrono::duration<double, std::milli>(finished - started).count();
        double decodeMs = totalMs - prefillMs;
        int decodeTokens = generated - 1;
        if (engine->verbose) {
            std::cout << "[timing] prefill: " << prefillMs << " ms for " << prompt_len << " tokens ("
                      << (prompt_len / (prefillMs / 1000.0)) << " tok/s)" << std::endl;
            if (decodeTokens > 0)
                std::cout << "[timing] decode: " << decodeMs << " ms for " << decodeTokens << " tokens ("
                          << (decodeTokens / (decodeMs / 1000.0)) << " tok/s)" << std::endl;
            std::cout << "[timing] total: " << totalMs << " ms for " << generated << " output tokens" << std::endl;
        }
        return generated;
    }

    std::vector<std::vector<uint16_t>> k_cache(NUM_LAYERS), v_cache(NUM_LAYERS);
    std::vector<int> generated_ids;
    int seq_len = 0;

    int total_steps = prompt_len + max_new_tokens - 1;
    int n_out = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    double prefill_ms = 0.0;

    @autoreleasepool {
        for (int step = 0; step < total_steps; step++) {
            int token_id = (step < prompt_len) ? prompt_tokens[step] : generated_ids.back();
            int current_pos = seq_len++;

            int next_token = engine->backend == QWEN_BACKEND_MPS_FP16
                ? forward_step(engine, token_id, current_pos, k_cache, v_cache, &sampler)
                : forward_step_gpu(engine, token_id, current_pos, &sampler);
            generated_ids.push_back(next_token);

            if (step == prompt_len - 1) {
                auto t_prefill_end = std::chrono::high_resolution_clock::now();
                prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_start).count();
            }

            if (step >= prompt_len - 1) {
                if (callback) callback(next_token, user_data);
                n_out++;
                if (qwen_is_stop_token(next_token, stop_tokens, stop_count)) break;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double decode_ms = total_ms - prefill_ms;
    int decode_tokens = n_out - 1;  

    if (engine->verbose) {
        std::cout << "[timing] prefill: " << prefill_ms << " ms for " << prompt_len << " tokens ("
                  << (prompt_len / (prefill_ms / 1000.0)) << " tok/s)" << std::endl;
        if (decode_tokens > 0) {
            std::cout << "[timing] decode: " << decode_ms << " ms for " << decode_tokens << " tokens ("
                      << (decode_tokens / (decode_ms / 1000.0)) << " tok/s)" << std::endl;
        }
        std::cout << "[timing] total: " << total_ms << " ms for " << n_out << " output tokens" << std::endl;
    }

    return n_out;
}

int qwen_generate_streaming_until(QwenEngine* engine,
                                  const int* prompt_tokens, int prompt_len,
                                  int max_new_tokens,
                                  const int* stop_tokens, int stop_count,
                                  qwen_token_callback callback, void* user_data) {
    return qwen_generate_streaming_sampled_until(
        engine, prompt_tokens, prompt_len, max_new_tokens,
        stop_tokens, stop_count, greedySamplingParams(), callback, user_data);
}

int qwen_generate_streaming(QwenEngine* engine,
                             const int* prompt_tokens, int prompt_len,
                             int max_new_tokens,
                             qwen_token_callback callback, void* user_data) {
    return qwen_generate_streaming_until(engine, prompt_tokens, prompt_len,
                                         max_new_tokens, nullptr, 0,
                                         callback, user_data);
}

int qwen_generate_sampled_until(QwenEngine* engine,
                                const int* prompt_tokens, int prompt_len,
                                int max_new_tokens,
                                const int* stop_tokens, int stop_count,
                                QwenSamplingParams sampling,
                                int* out_tokens) {
    if (!out_tokens) return -1;
    struct Ctx { int* out; int i; } ctx{out_tokens, 0};
    auto cb = [](int token_id, void* user_data) {
        auto* c = (Ctx*)user_data;
        c->out[c->i++] = token_id;
    };
    return qwen_generate_streaming_sampled_until(
        engine, prompt_tokens, prompt_len, max_new_tokens,
        stop_tokens, stop_count, sampling, cb, &ctx);
}

int qwen_generate_until(QwenEngine* engine,
                        const int* prompt_tokens, int prompt_len,
                        int max_new_tokens,
                        const int* stop_tokens, int stop_count,
                        int* out_tokens) {
    return qwen_generate_sampled_until(engine, prompt_tokens, prompt_len,
                                       max_new_tokens, stop_tokens, stop_count,
                                       greedySamplingParams(), out_tokens);
}

int qwen_generate(QwenEngine* engine,
                   const int* prompt_tokens, int prompt_len,
                   int max_new_tokens,
                   int* out_tokens) {
    return qwen_generate_until(engine, prompt_tokens, prompt_len,
                               max_new_tokens, nullptr, 0, out_tokens);
}

}
