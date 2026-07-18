#ifndef QWEN_ENGINE_H
#define QWEN_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QwenEngine QwenEngine;

typedef enum QwenBackend {
    QWEN_BACKEND_METAL_FP16 = 0,
    QWEN_BACKEND_METAL_INT4 = 1,
    QWEN_BACKEND_MPS_FP16 = 2,
    QWEN_BACKEND_HYBRID = 3,
    QWEN_BACKEND_INT4_FP16_LM_HEAD = 4,
    QWEN_BACKEND_METAL_INT8 = 5,
    QWEN_BACKEND_INT8_FP16_LM_HEAD = 6,
} QwenBackend;

typedef struct QwenSamplingParams {
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} QwenSamplingParams;

QwenEngine* qwen_engine_create(const char* weights_dir, bool verbose);


QwenEngine* qwen_engine_create_with_backend(const char* weights_dir,
                                             QwenBackend backend,
                                             bool verbose);

void qwen_engine_destroy(QwenEngine* engine);


int qwen_generate(QwenEngine* engine,
                   const int* prompt_tokens, int prompt_len,
                   int max_new_tokens,
                   int* out_tokens);

typedef void (*qwen_token_callback)(int token_id, void* user_data);


int qwen_generate_streaming(QwenEngine* engine,
                             const int* prompt_tokens, int prompt_len,
                             int max_new_tokens,
                             qwen_token_callback callback, void* user_data);

int qwen_generate_streaming_until(QwenEngine* engine,
                                  const int* prompt_tokens, int prompt_len,
                                  int max_new_tokens,
                                  const int* stop_tokens, int stop_count,
                                  qwen_token_callback callback, void* user_data);

int qwen_generate_until(QwenEngine* engine,
                        const int* prompt_tokens, int prompt_len,
                        int max_new_tokens,
                        const int* stop_tokens, int stop_count,
                        int* out_tokens);

int qwen_generate_streaming_sampled_until(QwenEngine* engine,
                                          const int* prompt_tokens, int prompt_len,
                                          int max_new_tokens,
                                          const int* stop_tokens, int stop_count,
                                          QwenSamplingParams sampling,
                                          qwen_token_callback callback, void* user_data);

int qwen_generate_sampled_until(QwenEngine* engine,
                                const int* prompt_tokens, int prompt_len,
                                int max_new_tokens,
                                const int* stop_tokens, int stop_count,
                                QwenSamplingParams sampling,
                                int* out_tokens);

void qwen_session_reset(QwenEngine* engine);

int qwen_session_generate_streaming_sampled_until(QwenEngine* engine,
                                                  const int* new_prompt_tokens,
                                                  int new_prompt_len,
                                                  int max_new_tokens,
                                                  const int* stop_tokens,
                                                  int stop_count,
                                                  QwenSamplingParams sampling,
                                                  qwen_token_callback callback,
                                                  void* user_data);

#ifdef __cplusplus
}
#endif

#endif 
