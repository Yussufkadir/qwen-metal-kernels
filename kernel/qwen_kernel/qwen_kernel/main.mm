#pragma clang language objective-c++

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>

extern "C" {
    int metal_init(id<MTLDevice> device);
    int run_gate_up_batched(
        const uint16_t* gate_w, const uint16_t* up_w, const float* x,
        float* gate_out, float* up_out, uint32_t B, uint32_t M, uint32_t K);
    int run_down_batched(
        const uint16_t* down_w, const float* x, float* out,
        uint32_t B, uint32_t M, uint32_t K);
}

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"GPU: %@", device.name);

    if (metal_init(device) != 0) {
        NSLog(@"Metal init failed");
        return 1;
    }

    const uint BATCH      = 24;
    const uint M_GATE     = 4864;
    const uint K_GATE     = 896;
    const uint M_DOWN     = 896;
    const uint K_DOWN     = 4864;
    const int  REPS       = 100;

    std::vector<uint16_t> gate_w(BATCH * M_GATE * K_GATE);
    std::vector<uint16_t> up_w  (BATCH * M_GATE * K_GATE);
    std::vector<float>    x_gate(BATCH * K_GATE);
    std::vector<uint16_t> down_w(BATCH * M_DOWN * K_DOWN);
    std::vector<float>    x_down(BATCH * K_DOWN);

    srand(42);
    for (size_t i = 0; i < gate_w.size(); i++) {
        gate_w[i] = (uint16_t)(__fp16)((float)rand() / RAND_MAX);
        up_w[i]   = (uint16_t)(__fp16)((float)rand() / RAND_MAX);
    }
    for (size_t i = 0; i < x_gate.size(); i++)
        x_gate[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < down_w.size(); i++)
        down_w[i] = (uint16_t)(__fp16)((float)rand() / RAND_MAX);
    for (size_t i = 0; i < x_down.size(); i++)
        x_down[i] = (float)rand() / RAND_MAX;

    std::vector<float> gate_out(BATCH * M_GATE);
    std::vector<float> up_out  (BATCH * M_GATE);
    std::vector<float> down_out(BATCH * M_DOWN);

    run_gate_up_batched(gate_w.data(), up_w.data(), x_gate.data(),
                        gate_out.data(), up_out.data(), BATCH, M_GATE, K_GATE);
    run_down_batched(down_w.data(), x_down.data(), down_out.data(),
                     BATCH, M_DOWN, K_DOWN);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; i++) {
        run_gate_up_batched(gate_w.data(), up_w.data(), x_gate.data(),
                            gate_out.data(), up_out.data(), BATCH, M_GATE, K_GATE);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_gate = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; i++) {
        run_down_batched(down_w.data(), x_down.data(), down_out.data(),
                         BATCH, M_DOWN, K_DOWN);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_down = std::chrono::duration<double, std::milli>(t1 - t0).count() / REPS;

    double mlp_total = ms_gate + ms_down;
    double est_tok_sec = 1000.0 / (mlp_total * 1.3);

    NSLog(@"\n═══ QWEN MLP: gate+up batched ═══");
    NSLog(@"  time:         %.4f ms", ms_gate);
    NSLog(@"═══ QWEN MLP: down batched ═══");
    NSLog(@"  time:         %.4f ms", ms_down);
    NSLog(@"╔══════════════════════════════════╗");
    NSLog(@"║  MLP TOTAL: %.3f ms              ║", mlp_total);
    NSLog(@"║  Est tok/sec: %.0f               ║", est_tok_sec);
    NSLog(@"╚══════════════════════════════════╝");

    if (est_tok_sec > 106.0)
        NSLog(@"\n🔥 FASTER than MLX baseline (106 tok/sec)");
    else
        NSLog(@"\n🐢 SLOWER than MLX baseline");

    return 0;
}
