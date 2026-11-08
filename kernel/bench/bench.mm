#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
static std::string slurp(const char* p){ std::ifstream f(p); std::stringstream s; s<<f.rdbuf(); return s.str(); }
int main(int argc, char** argv){ @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q = [dev newCommandQueue];
  std::string src = slurp(argv[1]) + "\n" + slurp(argv[2]) + "\n" + slurp(argv[3]);
  // strip duplicate headers
  NSError* err=nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&err];
  if(!lib){ printf("%s\n", err.localizedDescription.UTF8String); return 1; }
  auto pipe=[&](const char* n){ return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:nil]; };
  struct Shape{const char* name; uint M,K;};
  Shape shapes[] = {{"qkv(1152x896)",1152,896},{"o(896x896)",896,896},{"gate+up(9728x896)",9728,896},{"down(896x4864)",896,4864},{"lm_head(151936x896)",151936,896}};
  for (auto& s: shapes){
    size_t wbytes=(size_t)s.M*s.K*2;
    id<MTLBuffer> W=[dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> x=[dev newBufferWithLength:s.K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> y=[dev newBufferWithLength:s.M*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> y2=[dev newBufferWithLength:s.M*2 options:MTLResourceStorageModeShared];
    __fp16* wp=(__fp16*)W.contents; for(size_t i=0;i<(size_t)s.M*s.K;i++) wp[i]=(__fp16)(((i*2654435761u)>>20)%200/1000.0-0.1);
    __fp16* xp=(__fp16*)x.contents; for(uint i=0;i<s.K;i++) xp[i]=(__fp16)(((i*40503u)>>4)%100/100.0-0.5);
    auto run=[&](const char* kname, bool old, uint simdgroups, uint rows, id<MTLBuffer> out)->double{
      id<MTLComputePipelineState> p=pipe(kname);
      double best=1e9;
      for(int it=0;it<20;it++){
        id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:x offset:0 atIndex:1]; [e setBuffer:out offset:0 atIndex:2];
        if(old){ uint K=s.K; [e setBytes:&K length:4 atIndex:3]; [e setThreadgroupMemoryLength:16 atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake(s.M,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; }
        else { uint M=s.M,K=s.K; [e setBytes:&M length:4 atIndex:3]; [e setBytes:&K length:4 atIndex:4];
          uint perTg=simdgroups*rows; [e dispatchThreadgroups:MTLSizeMake((s.M+perTg-1)/perTg,1,1) threadsPerThreadgroup:MTLSizeMake(32*simdgroups,1,1)]; }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double t=cb.GPUEndTime-cb.GPUStartTime; if(it>2&&t<best)best=t;
      }
      return best;
    };
    double told=run("matvec_float4",true,0,0,y);
    printf("%-22s old matvec_float4: %7.1f us  %6.1f GB/s\n", s.name, told*1e6, wbytes/told/1e9);
    struct V{const char* k; uint sg, r;} vs[]={{"gemv_r2",4,2},{"gemv_r4",2,4},{"gemv_r4",4,4},{"gemv_r4",8,4},{"gemv_r8",4,8}};
    for(auto& v: vs){
      double t=run(v.k,false,v.sg,v.r,y2);
      double maxerr=0; __fp16* a=(__fp16*)y.contents; __fp16* b=(__fp16*)y2.contents;
      for(uint i=0;i<s.M;i++) maxerr=fmax(maxerr,fabs((double)a[i]-(double)b[i]));
      printf("%-22s %s sg=%u: %7.1f us  %6.1f GB/s  (maxdiff %.4f)\n","",v.k,v.sg,t*1e6,wbytes/t/1e9,maxerr);
    }
  }
}}
