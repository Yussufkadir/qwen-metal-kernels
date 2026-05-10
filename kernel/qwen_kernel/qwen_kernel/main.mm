//
//  main.m
//  qwen_kernel
//
//  Created by Yusuf Surmen on 27/04/2026.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSLog(@"GPU: %@", device.name);
    NSLog(@"Max threadgroup memory: %lu bytes",
          (unsigned long)device.maxThreadgroupMemoryLength);

    id<MTLLibrary> library = [device newDefaultLibrary];
    if (!library) { NSLog(@"FAILED to load default library"); return 1; }

    NSError* error = nil;

    id<MTLFunction> func_gate_up = [library newFunctionWithName:@"matvec_gate_up_batched"];
    id<MTLFunction> func_down    = [library newFunctionWithName:@"matvec_down_batched"];

    id<MTLComputePipelineState> pipe_gate_up =
        [device newComputePipelineStateWithFunction:func_gate_up error:&error];
    id<MTLComputePipelineState> pipe_down =
        [device newComputePipelineStateWithFunction:func_down error:&error];

    if (!pipe_gate_up) {
        NSLog(@"ERROR: matvec_gate_up_batched — %@", error.localizedDescription);
        return 1;
    }
    if (!pipe_down) {
        NSLog(@"ERROR: matvec_down_batched — %@", error.localizedDescription);
        return 1;
    }
    NSLog(@"All kernels loaded");

    const uint BATCH      = 24;
    const uint M_GATE     = 4864;
    const uint K_GATE     = 896;
    const uint M_DOWN     = 896;
    const uint K_DOWN     = 4864;
    const uint GROUP_SIZE = 128;
    const int  REPS       = 100;

    std::vector<uint16_t> gate_w(BATCH * M_GATE * K_GATE);
    std::vector<uint16_t> up_w  (BATCH * M_GATE * K_GATE);
    std::vector<float>    x_gate(BATCH * K_GATE);
    std::vector<uint16_t> down_w(BATCH * M_DOWN * K_DOWN);
    std::vector<float>    x_down(BATCH * K_DOWN);

    srand(42);
    for (size_t i = 0; i < gate_w.size(); i++) {
        gate_w[i] = (uint16_t)(__fp16)((float)rand() / (float)RAND_MAX);
        up_w[i]   = (uint16_t)(__fp16)((float)rand() / (float)RAND_MAX);
    }
    for (size_t i = 0; i < x_gate.size(); i++)
        x_gate[i] = (float)rand() / (float)RAND_MAX;
    for (size_t i = 0; i < down_w.size(); i++)
        down_w[i] = (uint16_t)(__fp16)((float)rand() / (float)RAND_MAX);
    for (size_t i = 0; i < x_down.size(); i++)
        x_down[i] = (float)rand() / (float)RAND_MAX;

    id<MTLBuffer> buf_gate_w = [device newBufferWithBytes:gate_w.data()
                            length:gate_w.size() * sizeof(uint16_t)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_up_w   = [device newBufferWithBytes:up_w.data()
                            length:up_w.size() * sizeof(uint16_t)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_gate_x = [device newBufferWithBytes:x_gate.data()
                            length:x_gate.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_gate_y = [device newBufferWithLength:BATCH * M_GATE * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_up_y   = [device newBufferWithLength:BATCH * M_GATE * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_Kg     = [device newBufferWithBytes:&K_GATE
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    id<MTLBuffer> buf_down_w = [device newBufferWithBytes:down_w.data()
                            length:down_w.size() * sizeof(uint16_t)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_down_x = [device newBufferWithBytes:x_down.data()
                            length:x_down.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_down_y = [device newBufferWithLength:BATCH * M_DOWN * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_Kd     = [device newBufferWithBytes:&K_DOWN
                            length:sizeof(uint)
                           options:MTLResourceStorageModeShared];

    id<MTLCommandQueue> queue = [device newCommandQueue];

    double ms_gate_up = 0.0;
    {
        auto run_once = [&]() {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe_gate_up];
            [enc setBuffer:buf_gate_w offset:0 atIndex:0];
            [enc setBuffer:buf_up_w   offset:0 atIndex:1];
            [enc setBuffer:buf_gate_x offset:0 atIndex:2];
            [enc setBuffer:buf_gate_y offset:0 atIndex:3];
            [enc setBuffer:buf_up_y   offset:0 atIndex:4];
            [enc setBuffer:buf_Kg     offset:0 atIndex:5];

            size_t tg_bytes = (GROUP_SIZE / 32) * 2 * sizeof(float);
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];

            MTLSize grid  = MTLSizeMake(M_GATE, BATCH, 1);
            MTLSize group = MTLSizeMake(GROUP_SIZE, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int r = 0; r < 5; r++) run_once();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; r++) run_once();
        auto t1 = std::chrono::high_resolution_clock::now();

        ms_gate_up = std::chrono::duration<double,
                     std::milli>(t1 - t0).count() / REPS;

        double bytes = BATCH * (2.0 * M_GATE * K_GATE * 2
                                + K_GATE * 4
                                + 2.0 * M_GATE * 4);
        double gbs = (bytes / 1e9) / (ms_gate_up / 1000.0);

        NSLog(@"\n═══ QWEN MLP: gate+up batched ═══");
        NSLog(@"  matvecs:      %u (24 blocks × 2)", BATCH * 2);
        NSLog(@"  grid:         %u × %u threadgroups", M_GATE, BATCH);
        NSLog(@"  time:         %.4f ms", ms_gate_up);
        NSLog(@"  bandwidth:    %.1f GB/s", gbs);
        NSLog(@"  util:         %.1f%% of 400 GB/s", (gbs / 400.0) * 100.0);
        NSLog(@"  per-matvec:   %.3f ms", ms_gate_up / (BATCH * 2));
    }

    double ms_down = 0.0;
    {
        auto run_once = [&]() {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe_down];
            [enc setBuffer:buf_down_w offset:0 atIndex:0];
            [enc setBuffer:buf_down_x offset:0 atIndex:1];
            [enc setBuffer:buf_down_y offset:0 atIndex:2];
            [enc setBuffer:buf_Kd     offset:0 atIndex:3];

            size_t tg_bytes = (GROUP_SIZE / 32) * sizeof(float);
            [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];

            MTLSize grid  = MTLSizeMake(M_DOWN, BATCH, 1);
            MTLSize group = MTLSizeMake(GROUP_SIZE, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        };

        for (int r = 0; r < 5; r++) run_once();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < REPS; r++) run_once();
        auto t1 = std::chrono::high_resolution_clock::now();

        ms_down = std::chrono::duration<double,
                  std::milli>(t1 - t0).count() / REPS;

        double bytes = BATCH * (M_DOWN * K_DOWN * 2
                                + K_DOWN * 4
                                + M_DOWN * 4);
        double gbs = (bytes / 1e9) / (ms_down / 1000.0);

        NSLog(@"\n═══ QWEN MLP: down batched ═══");
        NSLog(@"  matvecs:      %u (24 blocks)", BATCH);
        NSLog(@"  grid:         %u × %u threadgroups", M_DOWN, BATCH);
        NSLog(@"  time:         %.4f ms", ms_down);
        NSLog(@"  bandwidth:    %.1f GB/s", gbs);
        NSLog(@"  util:         %.1f%% of 400 GB/s", (gbs / 400.0) * 100.0);
        NSLog(@"  per-matvec:   %.3f ms", ms_down / BATCH);
    }

    double mlp_total = ms_gate_up + ms_down;
    double est_tok_sec = 1000.0 / (mlp_total * 1.3);

    NSLog(@"\n╔══════════════════════════════════╗");
    NSLog(@"║  gate+up:  %.3f ms               ║", ms_gate_up);
    NSLog(@"║  down:     %.3f ms               ║", ms_down);
    NSLog(@"║  MLP TOTAL: %.3f ms              ║", mlp_total);
    NSLog(@"║  Est tok/sec: %.0f               ║", est_tok_sec);
    NSLog(@"║  MLX baseline: 106 tok/sec       ║");
    NSLog(@"╚══════════════════════════════════╝");

    if (est_tok_sec > 106.0) {
        NSLog(@"\n🔥 FASTER than MLX baseline");
    } else {
        NSLog(@"\n🐢 SLOWER than MLX baseline — needs work");
    }

    return 0;
}
