#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "qwen_engine.h"

int main(int argc, char* argv[]) {
    int max_new_tokens = 6;
    QwenBackend backend = QWEN_BACKEND_METAL_FP16;
    QwenSamplingParams sampling{0.0f, 0, 1.0f, 1.0f, 0};
    std::vector<int> input_ids;
    std::vector<int> stop_tokens;
    bool quiet = false;
    bool stream_tokens = false;
    bool server_mode = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./inference_engine [options] token_id ...\n\n"
                << "Options:\n"
                << "  --backend fp16|int4|int8|mixed|mps|hybrid\n"
                << "  --max-tokens N, -n N\n"
                << "  --stop-token ID     Stop generation after producing this token (repeatable)\n"
                << "  --temperature T     0 means greedy, >0 enables sampling\n"
                << "  --top-k K           0 disables top-k filtering\n"
                << "  --top-p P           1 disables nucleus filtering\n"
                << "  --repetition-penalty R\n"
                << "  --seed S            0 uses a time-based seed\n"
                << "  --stream-tokens     Print TOKEN <id> as each token is generated\n"
                << "  --server            Keep engine loaded and read GENERATE requests from stdin\n"
                << "  --quiet             Suppress headers and engine timing/log output\n";
            return 0;
        } else if (arg == "--max-tokens" || arg == "-n") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << arg << "\n";
                return 1;
            }
            max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature") {
            if (i + 1 >= argc) { std::cerr << "Missing value after --temperature\n"; return 1; }
            sampling.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k") {
            if (i + 1 >= argc) { std::cerr << "Missing value after --top-k\n"; return 1; }
            sampling.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p") {
            if (i + 1 >= argc) { std::cerr << "Missing value after --top-p\n"; return 1; }
            sampling.top_p = std::stof(argv[++i]);
        } else if (arg == "--repetition-penalty") {
            if (i + 1 >= argc) { std::cerr << "Missing value after --repetition-penalty\n"; return 1; }
            sampling.repetition_penalty = std::stof(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 >= argc) { std::cerr << "Missing value after --seed\n"; return 1; }
            sampling.seed = std::stoull(argv[++i]);
        } else if (arg == "--stop-token") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --stop-token\n";
                return 1;
            }
            stop_tokens.push_back(std::stoi(argv[++i]));
        } else if (arg == "--backend") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --backend (fp16, int4, int8, mixed, mps, or hybrid)\n";
                return 1;
            }
            std::string value = argv[++i];
            if (value == "fp16") backend = QWEN_BACKEND_METAL_FP16;
            else if (value == "int4") backend = QWEN_BACKEND_METAL_INT4;
            else if (value == "int8") backend = QWEN_BACKEND_METAL_INT8;
            else if (value == "mixed") backend = QWEN_BACKEND_INT4_FP16_LM_HEAD;
            else if (value == "mps") backend = QWEN_BACKEND_MPS_FP16;
            else if (value == "hybrid") backend = QWEN_BACKEND_HYBRID;
            else {
                std::cerr << "Unknown backend: " << value << " (expected fp16, int4, int8, mixed, mps, or hybrid)\n";
                return 1;
            }
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--stream-tokens") {
            stream_tokens = true;
        } else if (arg == "--server") {
            server_mode = true;
        } else {
            input_ids.push_back(std::stoi(arg));
        }
    }
    if (input_ids.empty() && !server_mode) input_ids = {1053};

    const char* backendName = backend == QWEN_BACKEND_METAL_INT4 ? "int4" :
                              backend == QWEN_BACKEND_METAL_INT8 ? "int8" :
                              backend == QWEN_BACKEND_INT4_FP16_LM_HEAD ? "mixed" :
                              backend == QWEN_BACKEND_MPS_FP16 ? "mps" :
                              backend == QWEN_BACKEND_HYBRID ? "hybrid" : "fp16";
    if (!quiet) {
        std::cout << "Input tokens: ";
        for (int id : input_ids) std::cout << id << " ";
        std::cout << std::endl;
        std::cout << "max_new_tokens: " << max_new_tokens << std::endl;
        std::cout << "backend: " << backendName << std::endl;
    }

    QwenEngine* engine = qwen_engine_create_with_backend("qwen_weights", backend, /*verbose=*/!quiet && !server_mode);
    if (!engine) {
        std::cerr << "Failed to initialize engine\n";
        return 1;
    }

    if (server_mode) {
        std::cout << "READY " << backendName << std::endl;
        std::string line;
        std::vector<int> cached_tokens;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT" || line == "EXIT") break;
            if (line == "RESET") {
                qwen_session_reset(engine);
                cached_tokens.clear();
                std::cout << "RESET_OK" << std::endl;
                continue;
            }
            if (line.empty()) continue;

            std::istringstream request(line);
            std::string command;
            int request_max_tokens = max_new_tokens;
            std::vector<int> request_tokens;
            std::vector<int> request_stop_tokens;
            request >> command;
            QwenSamplingParams request_sampling = sampling;
            bool cached_request = command == "GENERATE_CACHE_SAMPLE";
            bool has_sampling = command == "GENERATE_SAMPLE" || cached_request;
            if (command != "GENERATE" && command != "GENERATE_STOP" &&
                command != "GENERATE_SAMPLE" && command != "GENERATE_CACHE_SAMPLE") {
                std::cout << "ERROR expected GENERATE, GENERATE_STOP, GENERATE_SAMPLE, or GENERATE_CACHE_SAMPLE" << std::endl;
                continue;
            }
            if (!(request >> request_max_tokens) || request_max_tokens <= 0) {
                std::cout << "ERROR missing or invalid max token count" << std::endl;
                continue;
            }
            if (has_sampling) {
                if (!(request >> request_sampling.temperature >> request_sampling.top_k >>
                      request_sampling.top_p >> request_sampling.repetition_penalty >>
                      request_sampling.seed)) {
                    std::cout << "ERROR missing sampling parameters" << std::endl;
                    continue;
                }
            }
            if (command == "GENERATE_STOP" || command == "GENERATE_SAMPLE" || cached_request) {
                int stop_count = 0;
                if (!(request >> stop_count) || stop_count < 0) {
                    std::cout << "ERROR missing or invalid stop token count" << std::endl;
                    continue;
                }
                for (int i = 0; i < stop_count; i++) {
                    int stop = 0;
                    if (!(request >> stop)) {
                        std::cout << "ERROR missing stop token" << std::endl;
                        request_stop_tokens.clear();
                        break;
                    }
                    request_stop_tokens.push_back(stop);
                }
                if ((int)request_stop_tokens.size() != stop_count) continue;
            }
            int token = 0;
            while (request >> token) request_tokens.push_back(token);
            if (request_tokens.empty()) {
                std::cout << "ERROR no prompt tokens" << std::endl;
                continue;
            }

            std::cout << "BEGIN" << std::endl;
            struct ServerCtx { int count; std::vector<int> tokens; } ctx{0, {}};
            auto callback = [](int token_id, void* user_data) {
                auto* ctx = static_cast<ServerCtx*>(user_data);
                ctx->count++;
                ctx->tokens.push_back(token_id);
                std::cout << "TOKEN " << token_id << std::endl;
            };
            int n = -1;
            if (cached_request) {
                size_t common = 0;
                bool cache_reset = false;
                while (common < cached_tokens.size() &&
                       common < request_tokens.size() &&
                       cached_tokens[common] == request_tokens[common]) {
                    common++;
                }
                if (common < cached_tokens.size()) {
                    qwen_session_reset(engine);
                    cached_tokens.clear();
                    common = 0;
                    cache_reset = true;
                }
                if (common >= request_tokens.size()) {
                    std::cout << "ERROR no new prompt tokens for cached generation" << std::endl;
                    continue;
                }
                std::vector<int> delta(request_tokens.begin() + (long)common, request_tokens.end());
                std::cout << "CACHE " << common << " " << delta.size() << " "
                          << (cache_reset ? 1 : 0) << std::endl;
                n = qwen_session_generate_streaming_sampled_until(
                    engine, delta.data(), (int)delta.size(),
                    request_max_tokens,
                    request_stop_tokens.empty() ? nullptr : request_stop_tokens.data(),
                    (int)request_stop_tokens.size(),
                    request_sampling,
                    callback, &ctx);
                if (n >= 0) {
                    cached_tokens = request_tokens;
                    if (ctx.tokens.size() > 1) {
                        cached_tokens.insert(cached_tokens.end(), ctx.tokens.begin(), ctx.tokens.end() - 1);
                    }
                }
            } else {
                n = qwen_generate_streaming_sampled_until(
                    engine, request_tokens.data(), (int)request_tokens.size(),
                    request_max_tokens,
                    request_stop_tokens.empty() ? nullptr : request_stop_tokens.data(),
                    (int)request_stop_tokens.size(),
                    request_sampling,
                    callback, &ctx);
            }
            if (n < 0) {
                std::cout << "ERROR generation failed" << std::endl;
            } else {
                std::cout << "END " << n << std::endl;
            }
        }
        qwen_engine_destroy(engine);
        return 0;
    }

    std::vector<int> out_tokens(max_new_tokens);

    int n = 0;
    if (stream_tokens) {
        struct StreamCtx {
            std::vector<int>* tokens;
        } ctx{&out_tokens};
        auto callback = [](int token_id, void* user_data) {
            auto* ctx = static_cast<StreamCtx*>(user_data);
            ctx->tokens->push_back(token_id);
            std::cout << "TOKEN " << token_id << std::endl;
        };
        out_tokens.clear();
        n = qwen_generate_streaming_sampled_until(
            engine, input_ids.data(), (int)input_ids.size(),
            max_new_tokens,
            stop_tokens.empty() ? nullptr : stop_tokens.data(),
            (int)stop_tokens.size(),
            sampling,
            callback, &ctx);
    } else {
        n = qwen_generate_sampled_until(
            engine, input_ids.data(), (int)input_ids.size(),
            max_new_tokens,
            stop_tokens.empty() ? nullptr : stop_tokens.data(),
            (int)stop_tokens.size(),
            sampling,
            out_tokens.data());
    }

    if (n < 0) {
        std::cerr << "Generation failed\n";
        qwen_engine_destroy(engine);
        return 1;
    }

    if (!stream_tokens || !quiet) {
        std::cout << "Generated continuation: ";
        for (int i = 0; i < n; i++) std::cout << out_tokens[i] << " ";
        std::cout << std::endl;
    }

    qwen_engine_destroy(engine);
    return 0;
}
