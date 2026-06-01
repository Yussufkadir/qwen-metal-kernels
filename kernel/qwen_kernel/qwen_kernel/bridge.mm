//
//  bridge.mm
//  qwen_kernel
//
//  Created by Yusuf Surmen on 30/04/2026.
//

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <chrono>

static id<MTLDevice> _device = nil;
static id<MTLComputePipelineState> _pipe_gate_up = nil;
static id<MTLComputePipelineState> _pipe_down = nil;

static void _ensure_initialized() {
    if (_device) return;
    
    _device = MTLCreateSystemDefaultDevice();
    NSLog(@"GPU: %@", _device.name);
    
    id<MTLLibrary> library = [_device newLibraryWithFile:@"default.metallib" error:nil];
    if (!library) {
        NSLog(@"FATAL: cannot load default.metallib");
        return;
    }
    
    id<MTLFunction> f1 = [library newFunctionWithName:@"matvec_gate_up_batched"];
    id<MTLFunction> f2 = [library newFunctionWithName:@"matvec_down_batched"];
    
    _pipe_gate_up = [_device newComputePipelineStateWithFunction:f1 error:nil];
    _pipe_down = [_device newComputePipelineStateWithFunction:f2 error:nil];
    
    NSLog(@"Metal kernels loaded: %@ / %@",
          _pipe_gate_up ? @"gate+up OK": @"gate+up Failed",
          _pipe_down ? @"down OK": @"down Failed");
    
}

extern "C" {

int metal_init() {                      
    _ensure_initialized();
    return (_pipe_gate_up && _pipe_down) ? 0 : -1;
}

int run_gate_up_batched(
    const uint16_t* gate_w,
    const uint16_t* up_w,
    const float*    x,
    float*          gate_out,
    float*          up_out,
    uint32_t BATCH,
    uint32_t M,
    uint32_t K)
{
    _ensure_initialized();

    static id<MTLCommandQueue> _queue = nil;
    static id<MTLBuffer> _cached_gate_w = nil;
    static id<MTLBuffer> _cached_up_w   = nil;
    static id<MTLBuffer> _cached_x      = nil;
    static id<MTLBuffer> _cached_gate_y = nil;
    static id<MTLBuffer> _cached_up_y   = nil;
    static size_t _cached_size = 0;

    if (!_queue) _queue = [_device newCommandQueue];

    size_t weight_bytes = (size_t)BATCH * M * K * sizeof(uint16_t);
    size_t input_bytes  = (size_t)BATCH * K * sizeof(float);
    size_t output_bytes = (size_t)BATCH * M * sizeof(float);
    size_t total = weight_bytes + input_bytes + output_bytes;

    if (total != _cached_size) {
        _cached_gate_w = [_device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        _cached_up_w   = [_device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        _cached_x      = [_device newBufferWithLength:input_bytes options:MTLResourceStorageModeShared];
        _cached_gate_y = [_device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        _cached_up_y   = [_device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        _cached_size = total;
    }

    memcpy([_cached_gate_w contents], gate_w, weight_bytes);
    memcpy([_cached_up_w contents], up_w, weight_bytes);
    memcpy([_cached_x contents], x, input_bytes);

    id<MTLCommandBuffer> cmd = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:_pipe_gate_up];
    [enc setBuffer:_cached_gate_w offset:0 atIndex:0];
    [enc setBuffer:_cached_up_w   offset:0 atIndex:1];
    [enc setBuffer:_cached_x      offset:0 atIndex:2];
    [enc setBuffer:_cached_gate_y offset:0 atIndex:3];
    [enc setBuffer:_cached_up_y   offset:0 atIndex:4];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];

    uint32_t gsize = 128;
    size_t tg_bytes = (gsize / 32) * 2 * sizeof(float);
    [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];

    MTLSize grid  = MTLSizeMake(M, BATCH, 1);
    MTLSize group = MTLSizeMake(gsize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(gate_out, [_cached_gate_y contents], output_bytes);
    memcpy(up_out,   [_cached_up_y   contents], output_bytes);
    return 0;
}

int run_down_batched(
    const uint16_t* down_w,
    const float*    x,
    float*          out,
    uint32_t BATCH,
    uint32_t M,
    uint32_t K)
{
    _ensure_initialized();
    
    static id<MTLCommandQueue> _queue = nil;
    static id<MTLBuffer> _cached_down_w = nil;
    static id<MTLBuffer> _cached_x      = nil;
    static id<MTLBuffer> _cached_out    = nil;
    static size_t _cached_size = 0;

    if (!_queue) _queue = [_device newCommandQueue];

    size_t weight_bytes = (size_t)BATCH * M * K * sizeof(uint16_t);
    size_t input_bytes  = (size_t)BATCH * K * sizeof(float);
    size_t output_bytes = (size_t)BATCH * M * sizeof(float);
    size_t total = weight_bytes + input_bytes + output_bytes;

    if (total != _cached_size) {
        _cached_down_w = [_device newBufferWithLength:weight_bytes options:MTLResourceStorageModeShared];
        _cached_x      = [_device newBufferWithLength:input_bytes  options:MTLResourceStorageModeShared];
        _cached_out    = [_device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        _cached_size = total;
    }

    memcpy([_cached_down_w contents], down_w, weight_bytes);
    memcpy([_cached_x contents], x, input_bytes);

    id<MTLCommandBuffer> cmd = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:_pipe_down];
    [enc setBuffer:_cached_down_w offset:0 atIndex:0];
    [enc setBuffer:_cached_x      offset:0 atIndex:1];
    [enc setBuffer:_cached_out    offset:0 atIndex:2];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:3];

    uint32_t gsize = 128;
    size_t tg_bytes = (gsize / 32) * sizeof(float);
    [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];

    MTLSize grid  = MTLSizeMake(M, BATCH, 1);
    MTLSize group = MTLSizeMake(gsize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [_cached_out contents], output_bytes);
    return 0;
}

int run_gate_up_once(
    const uint16_t* gate_w,
    const uint16_t* up_w,
    const float*    x,
    float*          gate_out,
    float*          up_out,
    uint32_t BATCH,
    uint32_t M,
    uint32_t K)
{
    static id<MTLCommandQueue> _queue = nil;
    static id<MTLBuffer> _buf_gate_y = nil;
    static id<MTLBuffer> _buf_up_y = nil;
    
    _ensure_initialized();
    
    if (!_queue) _queue = [_device newCommandQueue];
    
    size_t weight_bytes = (size_t)BATCH * M * K * sizeof(uint16_t);
    size_t input_bytes  = (size_t)BATCH * K * sizeof(float);
    size_t output_bytes = (size_t)BATCH * M * sizeof(float);
    
    id<MTLBuffer> buf_gate_w = [_device newBufferWithBytesNoCopy:(void*)gate_w
                                length:weight_bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_up_w   = [_device newBufferWithBytesNoCopy:(void*)up_w
                                length:weight_bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_x      = [_device newBufferWithBytesNoCopy:(void*)x
                                length:input_bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_gate_y = [_device newBufferWithBytesNoCopy:gate_out
                                length:output_bytes options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> buf_up_y   = [_device newBufferWithBytesNoCopy:up_out
                                length:output_bytes options:MTLResourceStorageModeShared deallocator:nil];
    
    id<MTLCommandBuffer> cmd = [_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:_pipe_gate_up];
    [enc setBuffer:buf_gate_w offset:0 atIndex:0];
    [enc setBuffer:buf_up_w   offset:0 atIndex:1];
    [enc setBuffer:buf_x      offset:0 atIndex:2];
    [enc setBuffer:buf_gate_y offset:0 atIndex:3];
    [enc setBuffer:buf_up_y   offset:0 atIndex:4];
    [enc setBytes:&K length:sizeof(uint32_t) atIndex:5];
    
    uint32_t gsize = 128;
    size_t tg_bytes = (gsize / 32) * 2 * sizeof(float);
    [enc setThreadgroupMemoryLength:tg_bytes atIndex:0];
    
    MTLSize grid  = MTLSizeMake(M, BATCH, 1);
    MTLSize group = MTLSizeMake(gsize, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    return 0;
}

}
