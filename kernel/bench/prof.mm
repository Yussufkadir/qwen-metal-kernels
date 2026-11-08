// Per-op decode profile with cold weights: each op is dispatched 24x (one per
// layer) on 24 distinct weight copies, like the real forward pass.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <vector>
#include <functional>
#include <cstdio>
#include <string>
static const int L=24,H=896,I=4864,NH=14,KVH=2,HD=64,V=151936,MAXS=4096;
id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLLibrary> lib;
id<MTLBuffer> B(size_t bytes){ id<MTLBuffer> b=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  uint16_t* p=(uint16_t*)b.contents; for(size_t i=0;i<bytes/2;i++) p[i]=0x2000+(i*7919)%0x800; return b; }
id<MTLComputePipelineState> P(const char* n){ id<MTLFunction> f=[lib newFunctionWithName:@(n)]; if(!f){printf("missing %s\n",n);exit(1);} return [dev newComputePipelineStateWithFunction:f error:nil]; }
double timeOp(const char* name, std::function<void(id<MTLComputeCommandEncoder>,int)> enc, int reps=L){
  double best=1e9;
  for(int it=0;it<6;it++){ @autoreleasepool{
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    for(int l=0;l<reps;l++) enc(e,l);
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    double t=(cb.GPUEndTime-cb.GPUStartTime)*1e3; if(it>0&&t<best)best=t; }}
  printf("  %-34s %8.3f ms/token  (%6.1f us/dispatch)\n", name, best, best*1000/reps);
  return best;
}
int main(int argc,char**argv){ @autoreleasepool{
  dev=MTLCreateSystemDefaultDevice(); q=[dev newCommandQueue];
  lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:nil];
  int ctx = argc>2? atoi(argv[2]) : 384;
  std::vector<id<MTLBuffer>> qw,kw,vw,ow,gw,uw,dw,kc,vc,qb,kb,vb,nw;
  for(int l=0;l<L;l++){ qw.push_back(B(H*H*2)); kw.push_back(B(128*H*2)); vw.push_back(B(128*H*2)); ow.push_back(B(H*H*2));
    gw.push_back(B((size_t)I*H*2)); uw.push_back(B((size_t)I*H*2)); dw.push_back(B((size_t)I*H*2));
    kc.push_back(B(MAXS*128*2)); vc.push_back(B(MAXS*128*2)); qb.push_back(B(H*2)); kb.push_back(B(256)); vb.push_back(B(256)); nw.push_back(B(H*2)); }
  id<MTLBuffer> lm=B((size_t)V*H*2), hidden=B(H*2), normed=B(H*2), qbuf=B(H*2), kbuf=B(256), vbuf=B(256), qrot=B(H*4),
    attn=B(H*2), aproj=B(H*2), xf=B(H*4), go=B(I*4), uo=B(I*4), mid=B(I*4), dn=B(H*4), logits=B(V*2),
    bmax=B(NH*16*4), bsum=B(NH*16*4), bout=B(NH*16*HD*4), av=B(1024*4), ai=B(1024*4), sel=B(16), dimc=B(4);
  *(uint*)dimc.contents=H;
  auto rmsP=P("rms_norm_fast"), qkvP=P("qkv_fp16"), ropeP=P("rope_qkv_cache_append"), fusedP=P("gqa_attention_fused"),
    blockP=P("gqa_attention_block"), redP=P("gqa_attention_block_reduce"), mvP=P("matvec_float4"), resP=P("residual_add"),
    h2fP=P("half_to_float"), guP=P("matvec_gate_up_batched"), siluP=P("silu_mul_float"), downP=P("matvec_down_batched"),
    resfP=P("residual_add_float"), a1=P("argmax_stage1"), a2=P("argmax_stage2");
  uint K896=H, K4864=I, pos=ctx-1, nt=ctx;
  printf("context = %d tokens\n", ctx);
  double tot=0;
  auto rms=[&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:rmsP]; [e setBuffer:hidden offset:0 atIndex:0]; [e setBuffer:nw[l] offset:0 atIndex:1];
    [e setBuffer:normed offset:0 atIndex:2]; [e setBuffer:dimc offset:0 atIndex:3]; [e setThreadgroupMemoryLength:32 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; };
  auto ew=[&](id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p, std::vector<id<MTLBuffer>> bs, uint n){ [e setComputePipelineState:p];
    for(size_t i=0;i<bs.size();i++)[e setBuffer:bs[i] offset:0 atIndex:i]; [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(n,256u),1,1)]; };
  tot+=2*timeOp("rms_norm x2 (counted twice)", rms);
  tot+=timeOp("qkv_fp16", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:qkvP];
    id<MTLBuffer> bs[]={qw[l],kw[l],vw[l],normed,qbuf,kbuf,vbuf,qb[l],kb[l],vb[l]}; for(int i=0;i<10;i++)[e setBuffer:bs[i] offset:0 atIndex:i];
    [e setBytes:&K896 length:4 atIndex:10]; [e setThreadgroupMemoryLength:16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1152,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; });
  tot+=timeOp("rope_qkv_cache_append", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:ropeP];
    id<MTLBuffer> bs[]={qbuf,kbuf,vbuf,qrot,kc[l],vc[l]}; for(int i=0;i<6;i++)[e setBuffer:bs[i] offset:0 atIndex:i];
    [e setBytes:&pos length:4 atIndex:6]; [e dispatchThreads:MTLSizeMake(512,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; });
  if(nt<=512){
    tot+=timeOp("gqa_attention_fused", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:fusedP];
      id<MTLBuffer> bs[]={qrot,kc[l],vc[l],attn}; for(int i=0;i<4;i++)[e setBuffer:bs[i] offset:0 atIndex:i];
      [e setBytes:&nt length:4 atIndex:4]; [e setThreadgroupMemoryLength:(nt+8)*4 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(NH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; });
  } else {
    uint nb=(nt+255)/256;
    tot+=timeOp("gqa_attention_block + reduce", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:blockP];
      id<MTLBuffer> bs[]={qrot,kc[l],vc[l],bmax,bsum,bout}; for(int i=0;i<6;i++)[e setBuffer:bs[i] offset:0 atIndex:i];
      [e setBytes:&nt length:4 atIndex:6]; [e setBytes:&nb length:4 atIndex:7]; [e setThreadgroupMemoryLength:(256+8)*4 atIndex:0];
      [e dispatchThreadgroups:MTLSizeMake(nb,NH,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [e setComputePipelineState:redP]; id<MTLBuffer> rs[]={bmax,bsum,bout,attn}; for(int i=0;i<4;i++)[e setBuffer:rs[i] offset:0 atIndex:i];
      [e setBytes:&nb length:4 atIndex:4]; [e dispatchThreadgroups:MTLSizeMake(NH,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; });
  }
  tot+=timeOp("o_proj matvec_float4", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:mvP];
    [e setBuffer:ow[l] offset:0 atIndex:0]; [e setBuffer:attn offset:0 atIndex:1]; [e setBuffer:aproj offset:0 atIndex:2];
    [e setBytes:&K896 length:4 atIndex:3]; [e setThreadgroupMemoryLength:16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(H,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; });
  tot+=timeOp("residual_add", [&](id<MTLComputeCommandEncoder> e,int){ ew(e,resP,{hidden,aproj},H); });
  tot+=timeOp("half_to_float", [&](id<MTLComputeCommandEncoder> e,int){ ew(e,h2fP,{normed,xf},H); });
  tot+=timeOp("matvec_gate_up_batched", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:guP];
    id<MTLBuffer> bs[]={gw[l],uw[l],xf,go,uo}; for(int i=0;i<5;i++)[e setBuffer:bs[i] offset:0 atIndex:i];
    [e setBytes:&K896 length:4 atIndex:5]; [e setThreadgroupMemoryLength:32 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(I,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; });
  tot+=timeOp("silu_mul_float", [&](id<MTLComputeCommandEncoder> e,int){ ew(e,siluP,{go,uo,mid},I); });
  tot+=timeOp("matvec_down_batched", [&](id<MTLComputeCommandEncoder> e,int l){ [e setComputePipelineState:downP];
    [e setBuffer:dw[l] offset:0 atIndex:0]; [e setBuffer:mid offset:0 atIndex:1]; [e setBuffer:dn offset:0 atIndex:2];
    [e setBytes:&K4864 length:4 atIndex:3]; [e setThreadgroupMemoryLength:16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(H,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; });
  tot+=timeOp("residual_add_float", [&](id<MTLComputeCommandEncoder> e,int){ ew(e,resfP,{hidden,dn},H); });
  tot+=timeOp("lm_head matvec_float4 (x1)", [&](id<MTLComputeCommandEncoder> e,int){ [e setComputePipelineState:mvP];
    [e setBuffer:lm offset:0 atIndex:0]; [e setBuffer:normed offset:0 atIndex:1]; [e setBuffer:logits offset:0 atIndex:2];
    [e setBytes:&K896 length:4 atIndex:3]; [e setThreadgroupMemoryLength:16 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(V,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)]; },1);
  tot+=timeOp("argmax stage1+2 (x1)", [&](id<MTLComputeCommandEncoder> e,int){ uint vs=V, bc=(V+255)/256;
    [e setComputePipelineState:a1]; [e setBuffer:logits offset:0 atIndex:0]; [e setBuffer:av offset:0 atIndex:1]; [e setBuffer:ai offset:0 atIndex:2];
    [e setBytes:&vs length:4 atIndex:3]; [e setThreadgroupMemoryLength:1024 atIndex:0]; [e setThreadgroupMemoryLength:1024 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(bc,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [e setComputePipelineState:a2]; [e setBuffer:av offset:0 atIndex:0]; [e setBuffer:ai offset:0 atIndex:1]; [e setBuffer:sel offset:0 atIndex:2];
    [e setBytes:&bc length:4 atIndex:3]; [e setThreadgroupMemoryLength:2048 atIndex:0]; [e setThreadgroupMemoryLength:2048 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(512,1,1)]; },1);
  printf("  %-34s %8.3f ms/token\n","SUM (isolated ops)",tot);
}}
