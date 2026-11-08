#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <vector>
static const int L=24,NH=14,HD=64,MAXS=4096;
int main(int argc,char**argv){ @autoreleasepool{
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  id<MTLLibrary> old=[dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:nil];
  std::ifstream f(argv[2]); std::stringstream ss; ss<<f.rdbuf(); NSError* err=nil;
  id<MTLLibrary> nw=[dev newLibraryWithSource:@(ss.str().c_str()) options:nil error:&err]; if(!nw){printf("%s\n",err.localizedDescription.UTF8String);return 1;}
  auto P=[&](id<MTLLibrary> lib,const char* n){ return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:nil]; };
  auto fusedP=P(old,"gqa_attention_fused"), blockP=P(old,"gqa_attention_block"), redP=P(old,"gqa_attention_block_reduce"), newP=P(nw,"gqa_decode_attention");
  auto B=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  std::vector<id<MTLBuffer>> kc,vc; srand(1);
  for(int l=0;l<L;l++){ kc.push_back(B(MAXS*128*2)); vc.push_back(B(MAXS*128*2));
    for(auto b: {kc[l],vc[l]}){ __fp16* p=(__fp16*)b.contents; for(int i=0;i<MAXS*128;i++) p[i]=(__fp16)((rand()%2000)/500.0-2.0); } }
  id<MTLBuffer> qr=B(NH*HD*4); float* qp=(float*)qr.contents; for(int i=0;i<NH*HD;i++) qp[i]=(rand()%2000)/500.0-2.0;
  id<MTLBuffer> o1=B(NH*HD*2), o2=B(NH*HD*2), bm=B(NH*16*4), bs=B(NH*16*4), bo=B(NH*16*HD*4);
  for(uint nt: {32u,128u,384u,512u,1024u,2048u,4000u}){
    uint nb=(nt+255)/256, bsz=256;
    auto enc=[&](id<MTLComputeCommandEncoder> e,int l,bool useNew, id<MTLBuffer> out){
      if(useNew){
        [e setComputePipelineState:newP]; id<MTLBuffer> a[]={qr,kc[l],vc[l],out,bm,bs,bo}; for(int i=0;i<7;i++)[e setBuffer:a[i] offset:0 atIndex:i];
        [e setBytes:&nt length:4 atIndex:7]; [e setBytes:&nb length:4 atIndex:8]; [e setBytes:&bsz length:4 atIndex:9];
        [e setThreadgroupMemoryLength:(64+32*64)*4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(nb,NH,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      } else if(nt<=512){
        [e setComputePipelineState:fusedP]; id<MTLBuffer> a[]={qr,kc[l],vc[l],out}; for(int i=0;i<4;i++)[e setBuffer:a[i] offset:0 atIndex:i];
        [e setBytes:&nt length:4 atIndex:4]; [e setThreadgroupMemoryLength:(nt+8)*4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(NH,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; return;
      } else {
        [e setComputePipelineState:blockP]; id<MTLBuffer> a[]={qr,kc[l],vc[l],bm,bs,bo}; for(int i=0;i<6;i++)[e setBuffer:a[i] offset:0 atIndex:i];
        [e setBytes:&nt length:4 atIndex:6]; [e setBytes:&nb length:4 atIndex:7]; [e setThreadgroupMemoryLength:(256+8)*4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake(nb,NH,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      }
      if(nb>1){ [e setComputePipelineState:redP]; id<MTLBuffer> r[]={bm,bs,bo,out}; for(int i=0;i<4;i++)[e setBuffer:r[i] offset:0 atIndex:i];
        [e setBytes:&nb length:4 atIndex:4]; [e dispatchThreadgroups:MTLSizeMake(NH,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; }
    };
    auto t=[&](bool useNew, id<MTLBuffer> out){ double best=1e9; for(int it=0;it<6;it++){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      for(int l=0;l<L;l++) enc(e,l,useNew,out); [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; double x=(cb.GPUEndTime-cb.GPUStartTime)*1e3; if(it&&x<best)best=x;} return best; };
    double a=t(false,o1), b=t(true,o2);
    // CPU reference for layer 23 head outputs
    double maxerr=0; __fp16* ko=(__fp16*)kc[L-1].contents; __fp16* vo=(__fp16*)vc[L-1].contents; __fp16* r2=(__fp16*)o2.contents; __fp16* r1=(__fp16*)o1.contents; double maxold=0;
    for(int h=0;h<NH;h++){ int kv=h/7; std::vector<double> s(nt); double mx=-1e30;
      for(uint tk=0;tk<nt;tk++){ double d=0; for(int i=0;i<HD;i++) d+=qp[h*HD+i]*(double)ko[tk*128+kv*64+i]; s[tk]=d*0.125; mx=fmax(mx,s[tk]); }
      double sum=0; for(auto&x:s){x=exp(x-mx); sum+=x;}
      for(int i=0;i<HD;i++){ double o=0; for(uint tk=0;tk<nt;tk++) o+=s[tk]*(double)vo[tk*128+kv*64+i]; o/=sum;
        maxerr=fmax(maxerr,fabs(o-(double)r2[h*HD+i])); maxold=fmax(maxold,fabs(o-(double)r1[h*HD+i])); } }
    printf("ctx %5u: old %6.3f ms/token  new %6.3f ms/token  speedup %4.1fx   maxerr old %.4f new %.4f\n", nt,a,b,a/b,maxold,maxerr);
  }
}}
