#ifndef SCATTER_KER
#define SCATTER_KER

#include <cuda_runtime.h>
#include "misc.cu.h"

// https://stackoverflow.com/a/14038590
int debugGpuAssert(cudaError_t code, int line)
{
  if(code != cudaSuccess) {
    printf("GPU Error: %s as line %d\n",
           cudaGetErrorString(code),
           line);
    return -1;
  }

  return 0;
}

int gpuAssert(cudaError_t code)
{
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }

  return 0;
}

/* struct for holding an index and a value */
struct indval {
  int index;
  int value;
};

/* index function */
__device__ __host__ inline
struct indval
f(int pixel, int his_sz)
{
  struct indval iv;
  iv.index = pixel % his_sz;
  iv.value = pixel;
  return iv;
}

/* sequential scatter */
template<class OP, class IN_T, class OUT_T>
void
scatter_seq(IN_T  *img,
            OUT_T *his,
            int img_sz,
            int his_sz,
            struct timeval *t_start,
            struct timeval *t_end)
{
  gettimeofday(t_start, NULL);

  /* scatter */
  for(int i=0; i < img_sz; i ++) {
    int idx; OUT_T val;
    struct indval iv;

    iv = f(img[i], his_sz);
    idx = iv.index;
    val = iv.value;
    his[idx] = OP::apply(his[idx], val);
  }

  gettimeofday(t_end, NULL);
}

/* Common reduction kernel */
template<class OP, class T>
__global__ void
reduce_kernel(T *d_his,
              T *d_res,
              int img_sz,
              int his_sz,
              int num_hists)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // sum bins
  int sum = 0;
  if(gid < his_sz) {
    for(int i=gid; i<num_hists * his_sz; i+=his_sz) {
      sum = OP::apply(d_his[i], sum);
    }
    d_res[gid] = sum;
  }
}
/* Common reduction kernel */

/* Common initialization kernel */
template<class OP, class OUT_T>
__global__ void
initialization_kernel(OUT_T *d_his,
                      int his_sz,
                      int num_hists)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int sz  = num_hists * his_sz;

  if(gid < his_sz) {
    for(int i=gid; i<sz; i+=his_sz) {
      d_his[i] = OP::identity();
    }
  }
}
/* Common initialization kernel */

/* -- KERNEL ID: 10 --  */
/* Atomic add in global memory - one global histogram */
template<class T>
__global__ void
aadd_noShared_noChunk_fullCorp_kernel(T *d_img,
                                      T *d_his,
                                      int img_sz,
                                      int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    struct indval iv = f(d_img[gid], his_sz);
    int idx = iv.index;
    int val = iv.value;
    atomicAdd(&d_his[idx], val);
  }
}

template<class T>
int
aadd_noShared_noChunk_fullCorp(T *h_img,
                               T *h_his,
                               int img_sz,
                               int his_sz,
                               struct timeval *t_start,
                               struct timeval *t_end,
                               int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  T *d_img, *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid_dim (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(img_sz), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n", grid_dim.x);
    printf("Block: %d\n", block_dim.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  aadd_noShared_noChunk_fullCorp_kernel<T>
    <<<grid_dim, block_dim>>>
    (d_img, d_his, img_sz, his_sz);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);

  return res;
}
/* -- KERNEL ID: 10 -- */


/* -- KERNEL ID: 11 -- */
/* Atomic add, one global histogram, chunking */
template<class T>
__global__ void
aadd_noShared_chunk_fullCorp_kernel(T *d_img,
                                    T *d_his,
                                    int img_sz,
                                    int his_sz,
                                    int num_threads)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval iv = f(d_img[i], his_sz);
        atomicAdd(&d_his[iv.index], iv.value);
    }
  }
}

template<class T>
int
aadd_noShared_chunk_fullCorp(T *h_img,
                             T *h_his,
                             int img_sz,
                             int his_sz,
                             int num_threads,
                             struct timeval *t_start,
                             struct timeval *t_end,
                             int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  T *d_img, *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid (GRID_X_DIM (num_threads), 1, 1);
  dim3 block(BLOCK_X_DIM(num_threads), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n", grid.x);
    printf("Block: %d\n", block.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  aadd_noShared_chunk_fullCorp_kernel<T>
    <<<grid, block>>>
    (d_img, d_his, img_sz, his_sz, num_threads);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);

  return res;
}
/* -- KERNEL ID: 11 -- */


/* -- KERNEL ID: 12 -- */
/* Atomic add in global memory - w. corporation in global mem. */
template<class T>
__global__ void
aadd_noShared_chunk_corp_kernel(T *d_img,
                                T *d_his,
                                int img_sz,
                                int his_sz,
                                int num_threads,
                                int seq_chunk,
                                int corp_lvl,
                                int num_hists,
                                int init_chunk,
                                int init_threads)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / corp_lvl) * his_sz; // global histogram

  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval iv = f(d_img[i], his_sz);
        atomicAdd(&d_his[ghidx + iv.index], iv.value);
    }
  }
}

template<class T>
int
aadd_noShared_chunk_corp(int *h_img,
                         int *h_his,
                         int img_sz,
                         int his_sz,
                         int num_threads,
                         int seq_chunk,
                         int corp_lvl,
                         int num_hists,
                         struct timeval *t_start,
                         struct timeval *t_end,
                         int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  // compute number of threads needed to initialized
  int init_chunk =
    ceil( (num_hists * his_sz) / (float)num_threads );
  int init_threads =
    ((num_hists * his_sz) % init_chunk) == 0 ?
    ((num_hists * his_sz) / init_chunk) :
    ((num_hists * his_sz) / init_chunk) + 1;

  T *d_img, *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz * num_hists);
  cudaMalloc((void **)&d_res, his_mem_sz);;
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel
  dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (num_threads), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(num_threads), 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // should be handled thread-wise in kernel
  cudaMemset(d_his, 0, his_mem_sz * num_hists);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);

  cudaThreadSynchronize();
  */

  aadd_noShared_chunk_corp_kernel<T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl, num_hists,
     init_chunk, init_threads);

  cudaThreadSynchronize();

  reduce_kernel<Add<int>, T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 12 -- */


/* -- KERNEL ID: 13 -- */
/* Atomic add in shared memory - w. corporation in shared mem. */
template<class T>
__global__ void
aadd_shared_chunk_corp_kernel(T *d_img,
                              T *d_his,
                              int img_sz,
                              int his_sz,
                              int num_threads,
                              int corp_lvl,
                              int num_hists,
                              int hists_per_block,
                              int init_threads)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int his_block_sz = hists_per_block * his_sz;
  int lhid = (tid / corp_lvl) * his_sz; // local histogram idx
  int ghid = blockIdx.x * hists_per_block * his_sz;

  // initialize local histograms
  extern __shared__ T sh_his[];
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
        sh_his[i] = 0;
    }
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval iv = f(d_img[i], his_sz);
        atomicAdd(&sh_his[lhid + iv.index],iv.value);
    }
  }
  __syncthreads();

  // copy to global memory
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
      d_his[ghid + i] = sh_his[i];
    }
  }
}

template<class T>
int aadd_shared_chunk_corp(T *h_img,
                           T *h_his,
                           int img_sz,
                           int his_sz,
                           int num_threads,
                           int seq_chunk,
                           int corp_lvl,
                           int num_hists,
                           struct timeval *t_start,
                           struct timeval *t_end,
                           int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  // histograms per block - maximum value is 1024
  int sh_mem_sz = 48 * 1024;
  int hists_per_block = min(
                            (sh_mem_sz / his_mem_sz),
                            (BLOCK_SZ / corp_lvl)
                            );
  int thrds_per_block = hists_per_block * corp_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  // compute numbers needed for initialization and copy steps
  int total_hist = hists_per_block * his_sz;
  int sh_chunk =
    ceil( total_hist / (float)thrds_per_block );
  int sh_chunk_threads =
    (total_hist % sh_chunk) == 0 ?
    (total_hist / sh_chunk) :
    (total_hist / sh_chunk) + 1;

  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
    printf("sh_chunk: %d\n", sh_chunk);
    printf("sh_chunk_threads: %d\n", sh_chunk_threads);
  }

  // d_his contains all histograms from shared memory
  int *d_img, *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his,
             his_mem_sz * num_blocks * hists_per_block);
  cudaMalloc((void **)&d_res, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // first kernel
  dim3 grid_dim_fst (num_blocks, 1, 1);
  dim3 block_dim_fst(thrds_per_block, 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  aadd_shared_chunk_corp_kernel<T>
    <<<grid_dim_fst, block_dim_fst, his_mem_sz * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, corp_lvl, num_hists, hists_per_block,
     sh_chunk_threads);

  cudaThreadSynchronize();

  reduce_kernel<Add<int>, T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- 13 -- */


/* -- 20 --
 * Manual lock - CAS in global memory - one hist. in global mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_noShared_noChunk_fullCorp_kernel(IN_T  *d_img,
                                     OUT_T *d_his,
                                     int img_sz,
                                     int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    int idx; OUT_T val;
    struct indval iv;

    iv = f(d_img[gid], his_sz);
    idx = iv.index;
    val = iv.value;
    OUT_T old = d_his[idx];
    OUT_T assumed;

    do {
      assumed = old;
      old = atomicCAS(&d_his[idx], assumed,
                      OP::apply(val, assumed));
    } while(assumed != old);
  }
}

template<class OP, class IN_T, class OUT_T>
int
CAS_noShared_noChunk_fullCorp(IN_T  *h_img,
                              OUT_T *h_his,
                              int img_sz,
                              int his_sz,
                              struct timeval *t_start,
                              struct timeval *t_end,
                              int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  IN_T *d_img; OUT_T *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid_dim (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(img_sz), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n", grid_dim.x);
    printf("Block: %d\n", block_dim.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  CAS_noShared_noChunk_fullCorp_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_img, d_his, img_sz, his_sz);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);

  return res;
}
/* -- KERNEL ID: 20 -- */


/* -- KERNEL ID: 21 -- */
/* CAS - one histogram in global memory w. chunking */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_noShared_chunk_fullCorp_kernel(IN_T *d_img,
                                   OUT_T *d_his,
                                   int img_sz,
                                   int his_sz,
                                   int num_threads,
                                   int seq_chunk)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    int idx; OUT_T val;
    struct indval iv;

    for(int i=gid; i<img_sz; i+=num_threads) {
      iv = f(d_img[i], his_sz);
      idx = iv.index;
      val = iv.value;

      OUT_T old = d_his[idx];
      OUT_T assumed;
      do {
        assumed = old;
        old = atomicCAS(&d_his[idx], assumed,
                        OP::apply(val, assumed));
      } while(assumed != old);
    }
  }
}

template<class OP, class IN_T, class OUT_T>
int
CAS_noShared_chunk_fullCorp(IN_T  *h_img,
                            OUT_T *h_his,
                            int img_sz,
                            int his_sz,
                            int num_threads,
                            int seq_chunk,
                            struct timeval *t_start,
                            struct timeval *t_end,
                            int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  IN_T *d_img; OUT_T *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid (GRID_X_DIM (num_threads), 1, 1);
  dim3 block(BLOCK_X_DIM(num_threads), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n", grid.x);
    printf("Block: %d\n", block.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  CAS_noShared_chunk_fullCorp_kernel<Add<int>, int>
    <<<grid, block>>>
    (d_img, d_his, img_sz, his_sz, num_threads, seq_chunk);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);

  return res;
}
/* -- KERNEL ID: 21 -- */


/* -- KERNEL ID: 22 -- */
/* Manual lock - CAS in global memory - corp. in global mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_noShared_chunk_corp_kernel(IN_T  *d_img,
                               OUT_T *d_his,
                               int img_sz,
                               int his_sz,
                               int num_threads,
                               int seq_chunk,
                               int corp_lvl)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / corp_lvl) * his_sz;

  if(gid < num_threads) {
    int idx; OUT_T val;
    struct indval iv;

    for(int i=gid; i<img_sz; i+=num_threads) {
      iv = f(d_img[i], his_sz);
      idx = iv.index;
      val = iv.value;

      OUT_T old = d_his[ghidx + idx];
      OUT_T assumed;
      do {
        assumed = old;
        old = atomicCAS(&d_his[ghidx + idx], assumed,
                        OP::apply(val, assumed));
      } while(assumed != old);
    }
  }
}

template<class OP, class IN_T, class OUT_T>
int
CAS_noShared_chunk_corp(IN_T  *h_img,
                        OUT_T *h_his,
                        int img_sz,
                        int his_sz,
                        int num_threads,
                        int seq_chunk,
                        int corp_lvl,
                        int num_hists,
                        struct timeval *t_start,
                        struct timeval *t_end,
                        int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  IN_T *d_img; OUT_T *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz * num_hists);
  cudaMalloc((void **)&d_res, his_mem_sz);;
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel
  dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (num_threads), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(num_threads), 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // should be done thread-wise in kernel
  cudaMemset(d_his, 0, his_mem_sz * num_hists);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);

  cudaThreadSynchronize();
  */

  CAS_noShared_chunk_corp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl);

  cudaThreadSynchronize();

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 22 -- */


/* -- KERNEL ID: 23 -- */
/* Manual lock - CAS in shared memory - corp. in shared mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_shared_chunk_corp_kernel(IN_T  *d_img,
                             OUT_T *d_his,
                             int img_sz,
                             int his_sz,
                             int num_threads,
                             int seq_chunk,
                             int corp_lvl,
                             int num_hists,
                             int hists_per_block,
                             int init_chunk,
                             int init_threads)
{
  // global thread id
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int his_block_sz = hists_per_block * his_sz;
  int lhidx = (tid / corp_lvl) * his_sz;
  int ghidx = blockIdx.x * hists_per_block * his_sz;

  // initialize local histograms
  extern __shared__ OUT_T sh_his[];
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
      sh_his[i] = OP::identity();
    }
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
      int idx; OUT_T val;
      struct indval iv;

      iv = f(d_img[i], his_sz);
      idx = iv.index;
      val = iv.value;
      OUT_T old = sh_his[lhidx + idx];
      OUT_T assumed;

      do {
        assumed = old;
        old = atomicCAS(&sh_his[lhidx + idx],
                        assumed,
                        OP::apply(val, assumed));
      } while(assumed != old);
    }
  }
  __syncthreads();

  // copy to global memory
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
      d_his[ghidx + i] = sh_his[i];
    }
  }

}

template<class OP, class IN_T, class OUT_T>
int
CAS_shared_chunk_corp(IN_T  *h_img,
                      OUT_T *h_his,
                      int img_sz,
                      int his_sz,
                      int num_threads,
                      int seq_chunk,
                      int corp_lvl,
                      int num_hists,
                      struct timeval *t_start,
                      struct timeval *t_end,
                      int PRINT_INFO)
{
  // because of shared memory
  if(corp_lvl > BLOCK_SZ) {
    printf("Error: corporation level cannot exceed block size\n");
    return -1;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  // histograms per block - maximum value is 1024
  int sh_mem_sz = 48 * 1024;
  int hists_per_block = min(
                            (sh_mem_sz / his_mem_sz),
                            (BLOCK_SZ / corp_lvl)
                            );
  int thrds_per_block = hists_per_block * corp_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  // compute numbers needed for initialization and copy steps
  int total_hist = hists_per_block * his_sz;
  int sh_chunk =
    ceil( total_hist / (float)thrds_per_block );
  int sh_chunk_threads =
    (total_hist % sh_chunk) == 0 ?
    (total_hist / sh_chunk) :
    (total_hist / sh_chunk) + 1;

  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
    printf("sh_chunk: %d\n", sh_chunk);
    printf("sh_chunk_threads: %d\n", sh_chunk_threads);
  }

  // d_his contains all histograms from shared memory
  IN_T *d_img; OUT_T *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his,
             his_mem_sz * num_blocks * hists_per_block);
  cudaMalloc((void **)&d_res, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // first kernel
  dim3 grid_dim_fst (num_blocks, 1, 1);
  dim3 block_dim_fst(thrds_per_block, 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  CAS_shared_chunk_corp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst, his_mem_sz * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl, num_hists, hists_per_block,
     sh_chunk, sh_chunk_threads);

  cudaThreadSynchronize();

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 23 -- */


/* -- KERNEL ID: 30 -- */
/* Exch. in global memory - one hist. in global mem.  */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_noShared_noChunk_fullCorp_kernel(IN_T  *d_img,
                                      OUT_T *d_his,
                                      volatile int *locks,
                                      int img_sz,
                                      int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int done, idx; OUT_T val;
  struct indval iv;

  if(gid < img_sz) {
    done = 0;
    iv = f(d_img[gid], his_sz);
    idx = iv.index;
    val = iv.value;
  } else {
    done = 1;
  }

  while(!done) {
    if(atomicExch((int *)&locks[idx], 1) == 0) {
      d_his[idx] = OP::apply(d_his[idx], val);
      locks[idx] = 0;
      done = 1;
    }
  }

  /* Why doesn't this work? */
  /*
  if(gid < img_sz) {
    int done = 0;
    struct indval iv = f(d_img[gid], his_sz);
    int idx = iv.index;
    int val = iv.value;

    while(!done) {
      if( atomicExch(&locks[idx], 1) == 0 ) {
        d_his[idx] = OP::apply(d_his[idx], val);
        __threadfence();
        locks[idx] = 0;
        done = 1;
      }
    }
  }
  */
  /* Why doesn't this work? */

  /* Why doesn't this work? */
  /* // should work if only one block (or X full blocks)
  int done, idx, val;
  struct indval iv;
  done = 0;
  iv = f(d_img[gid], his_sz);
  idx = iv.index;
  val = iv.value;

  while(!done) {
    if( atomicExch(&locks[idx], 1) == 0 ) {
      d_his[idx] = OP::apply(d_his[idx], val);
      __threadfence();
      locks[idx] = 0;
      done = 1;
    }
  }
  */
  /* Why doesn't this work? */
}

template<class OP, class IN_T, class OUT_T>
int
exch_noShared_noChunk_fullCorp(IN_T  *h_img,
                               OUT_T *h_his,
                               int img_sz,
                               int his_sz,
                               struct timeval *t_start,
                               struct timeval *t_end,
                               int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  unsigned int lck_mem_sz = his_sz * sizeof(int);

  IN_T  *d_img;
  OUT_T *d_his;
  int *d_locks;
  cudaMalloc((void **)&d_img,   img_mem_sz);
  cudaMalloc((void **)&d_his,   his_mem_sz);
  cudaMalloc((void **)&d_locks, lck_mem_sz);
  cudaMemcpy(d_img,   h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his,   h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel - for locks
  dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(img_sz), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n",  grid_dim.x);
    printf("Block: %d\n", block_dim.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // Initializing locks with extra kernel (here)
  // - does not add anything
  cudaMemset(d_locks, 0, lck_mem_sz);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, 1);
  cudaThreadSynchronize();
  */

  exch_noShared_noChunk_fullCorp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, d_locks, img_sz, his_sz);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_locks);

  return res;
}
/* -- KERNEL ID: 30 -- */


/* -- KERNEL ID: 31 -- */
/* Lock - Exch. - one histogram in global memory */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_noShared_chunk_fullCorp_kernel(OUT_T *d_img,
                                    IN_T  *d_his,
                                    volatile int *d_locks,
                                    int img_sz,
                                    int his_sz,
                                    int num_threads,
                                    int seq_chunk)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    int done, idx; OUT_T val;

    for(int i=0; i<seq_chunk; i++) {
      if(gid + i * num_threads < img_sz) {
        struct indval iv;
        done = 0;
        iv = f(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&d_locks[idx], 1) == 0 ) {
          d_his[idx] = OP::apply(d_his[idx], val);
          d_locks[idx] = 0;
          done = 1;
        }
      }
    }
  }
}

template<class OP, class IN_T, class OUT_T>
int
exch_noShared_chunk_fullCorp(IN_T  *h_img,
                             OUT_T *h_his,
                             int img_sz,
                             int his_sz,
                             int num_threads,
                             int seq_chunk,
                             struct timeval *t_start,
                             struct timeval *t_end,
                             int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  unsigned int lck_mem_sz = his_sz * sizeof(int);

  IN_T *d_img; OUT_T *d_his; int *d_locks;
  cudaMalloc((void **)&d_img,   img_mem_sz);
  cudaMalloc((void **)&d_his,   his_mem_sz);
  cudaMalloc((void **)&d_locks, lck_mem_sz);
  cudaMemcpy(d_img,   h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his,   h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel - for locks
  dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (num_threads), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(num_threads), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n", grid_dim_fst.x);
    printf("Block: %d\n", block_dim_fst.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // Initializing locks with extra kernel (here)
  // - does not add anything
  cudaMemset(d_locks, 0, his_mem_sz);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, 1);
  cudaThreadSynchronize();
  */

  exch_noShared_chunk_fullCorp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, d_locks,
     img_sz, his_sz, num_threads, seq_chunk);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_locks);

  return res;
}
/* -- KERNEL ID: 31 -- */


/* -- KERNEL ID: 32 -- */
/* Manual lock - Exch. in global memory - corp. in global mem.  */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_noShared_chunk_corp_kernel(IN_T  *d_img,
                                OUT_T *d_his,
                                int img_sz,
                                int his_sz,
                                int num_threads,
                                int seq_chunk,
                                int corp_lvl,
                                volatile int *d_locks)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / corp_lvl) * his_sz;

  if(gid < num_threads) {
    int done, idx; OUT_T val;

    for(int i=0; i<seq_chunk; i++) {
      if(gid + i * num_threads < img_sz) {
        struct indval iv;
        done = 0;
        iv = f(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&d_locks[ghidx + idx], 1) == 0 ) {
          d_his[ghidx + idx] =
            OP::apply(d_his[ghidx + idx], val);
          d_locks[ghidx + idx] = 0;
          done = 1;
        }
      }
    }
  }
}
/* Dunno why this doesn't work
 *
    for(int i=0; i<seq_chunk; i++) {
      if(gid * seq_chunk + i < img_sz) {
        int idx, val, done;
        struct indval iv;
        done = 0;
        iv = f(d_img[gid * seq_chunk + i], his_sz);
        idx = iv.index;
        val = iv.value;

        while(!done) {
          if( atomicExch(&d_locks[hid * his_sz + idx], 1) == 0 ) {
            d_his[hid * his_sz + idx] =
              OP::apply(d_his[hid * his_sz + idx], val);
            __threadfence();
            done = 1;
            d_locks[hid * his_sz + idx] = 0;
          }
        }
      }
    }
*/



template<class OP, class IN_T, class OUT_T>
int
exch_noShared_chunk_corp(IN_T  *h_img,
                         OUT_T *h_his,
                         int img_sz,
                         int his_sz,
                         int num_threads,
                         int seq_chunk,
                         int corp_lvl,
                         int num_hists,
                         struct timeval *t_start,
                         struct timeval *t_end,
                         int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz  = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz  = his_sz * sizeof(OUT_T);
  unsigned int lock_mem_sz = his_sz * sizeof(int);

  int  *d_locks;
  IN_T *d_img;
  OUT_T *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_res, his_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz * num_hists);
  cudaMalloc((void **)&d_locks, lock_mem_sz * num_hists);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel - for locks
  dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (num_threads), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(num_threads), 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // Adds significant overhead
  // pixels: 1mio, his_sz: 1024, corp. level: 64, histos: 977
  // seq. chunk: 16  ==> adds around 0.250 milliseconds.
  cudaMemset(d_his,   0, his_mem_sz * num_hists);
  cudaMemset(d_locks, 0, his_mem_sz * num_hists);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);
  cudaThreadSynchronize();

  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, num_hists);
  cudaThreadSynchronize();
  */

  exch_noShared_chunk_corp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl, d_locks);

  cudaThreadSynchronize();

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_locks);
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 32 -- */


/* -- KERNEL ID: 33 -- */
/* Manual lock - Exch. in shared memory - corp. in shared mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_shared_chunk_corp_kernel(IN_T  *d_img,
                              OUT_T *d_his,
                              int img_sz,
                              int his_sz,
                              int num_threads,
                              int seq_chunk,
                              int corp_lvl,
                              int num_hists,
                              int hists_per_block,
                              int init_chunk,
                              int init_threads)
{
  // global thread id
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int lhidx = (tid / corp_lvl) * his_sz;
  int ghidx = blockIdx.x * hists_per_block * his_sz;
  int his_block_sz = hists_per_block * his_sz;

  // initialize local histograms and locks
  volatile extern __shared__ int sh_mem[];
  volatile OUT_T *sh_his = sh_mem;
  volatile int *sh_lck = (int *)&sh_his[his_block_sz];

  // should this be split in two for better coalescing?
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
      sh_his[i] = OP::identity();
      sh_lck[i] = 0;
    }
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    int done, idx; OUT_T val;
    for(int i=0; i<seq_chunk; i++) {
      if(gid + i * num_threads < img_sz) {
        struct indval iv;
        done = 0;
        iv = f(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&sh_lck[lhidx + idx], 1) == 0 ) {
          sh_his[lhidx + idx] =
            OP::apply(sh_his[lhidx + idx], val);
          //sh_lck[lhidx + idx] = 0;
          atomicExch((int *)&sh_lck[lhidx + idx], 0);
          done = 1;
        }
      }
    }
  }
  __syncthreads();

  // copy to global memory
  if(tid < init_threads) {
    for(int i=tid; i<his_block_sz; i+=init_threads) {
      d_his[ghidx + i] = sh_his[i];
    }
  }
}

template<class OP, class IN_T, class OUT_T>
int
exch_shared_chunk_corp(IN_T  *h_img,
                       OUT_T *h_his,
                       int img_sz,
                       int his_sz,
                       int num_threads,
                       int seq_chunk,
                       int corp_lvl,
                       int num_hists,
                       struct timeval *t_start,
                       struct timeval *t_end,
                       int PRINT_INFO)
{
  // because of shared memory
  if(corp_lvl > BLOCK_SZ) {
    printf("Error: corporation level cannot exceed block size\n");
    return -1;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  unsigned int lck_mem_sz = his_sz * sizeof(int);

  // histograms per block - maximum value is 1024
  int sh_mem_sz = 48 * 1024;
  int hists_per_block = min(
                            (sh_mem_sz /
                             (his_mem_sz + lck_mem_sz)
                            ),
                            (BLOCK_SZ / corp_lvl)
                            );
  int thrds_per_block = hists_per_block * corp_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  // compute numbers needed for initialization and copy steps
  int total_hist = hists_per_block * his_sz;
  int sh_chunk =
    ceil( total_hist / (float)thrds_per_block );
  int sh_chunk_threads =
    (total_hist % sh_chunk) == 0 ?
    (total_hist / sh_chunk) :
    (total_hist / sh_chunk) + 1;

  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
    printf("sh_chunk: %d\n", sh_chunk);
    printf("sh_chunk_threads: %d\n", sh_chunk_threads);
  }

  if(hists_per_block * (his_mem_sz + lck_mem_sz) > sh_mem_sz) {
    printf("Error: Histograms and locks exceed "
           "shared memory size\n");
    return -1;
  }

  // d_his contains all histograms from shared memory
  IN_T *d_img; OUT_T *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_res, his_mem_sz);
  cudaMalloc((void **)&d_his,
             his_mem_sz * num_blocks * hists_per_block);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // first kernel
  dim3 grid_dim_fst (num_blocks, 1, 1);
  dim3 block_dim_fst(thrds_per_block, 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  if(PRINT_INFO) {
    printf("First - grid: %d\n", grid_dim_fst.x);
    printf("First - block: %d\n", block_dim_fst.x);
    printf("Second - grid: %d\n", grid_dim_snd.x);
    printf("Second - block: %d\n", block_dim_snd.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  exch_shared_chunk_corp_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst,
    (lck_mem_sz + his_mem_sz) * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl, num_hists, hists_per_block,
     sh_chunk, sh_chunk_threads);

  /* cudaError_t err = cudaGetLastError(); */
  /* if (err != cudaSuccess) */
  /*   printf("Error: %s\n", cudaGetErrorString(err)); */

  cudaThreadSynchronize();

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 33 -- */




/* -- 5 -- */
/* Warp optimized - lock free */
template<class OP, class T>
__global__ void
warp_shared_corp_kernel(T *d_img,
                        T *d_his,
                        int img_sz,
                        int his_sz,
                        int num_threads,
                        int seq_chunk,
                        int corp_lvl,
                        int num_hists,
                        int hists_per_block,
                        int init_chunk,
                        int init_threads)
{
  // global thread id
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  //int hid_local  = tid / corp_lvl;
  int h_start = blockIdx.x * hists_per_block * his_sz;
  int laneid = tid & (32 - 1);
  //  int warpid = tid >> 5;

  // initialize local histograms
  extern __shared__ char sh_hisT[];
  volatile T *sh_his = (volatile T*)sh_hisT;
  if(tid < init_threads) {
    for(int i=0; i < init_chunk; i++) {
      if(tid * init_chunk + i < hists_per_block * his_sz) {
        sh_his[tid * init_chunk + i] = OP::identity();
      }
    }
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    for(int i=0; i<seq_chunk; i++) {
      if(gid * seq_chunk + i < img_sz) {
        //        if(laneid < hists_per_block) {

          int idx, val;
          struct indval iv;
          iv = f(d_img[gid * seq_chunk + i], his_sz);
          idx = iv.index;
          val = iv.value;

          //atomicAdd(&sh_his[laneid * his_sz + idx], val);
          sh_his[laneid * his_sz + idx] =
            OP::apply(sh_his[laneid * his_sz + idx], val);

          /* if(blockIdx.x == 0) { */
          /*   printf("warpid: %d -- sh_his[%d * %d + %d]\n", */
          /*          warpid, laneid, his_sz, idx); */
          /* } */

          //        }
      }
    }
  }
  __syncthreads();


  // copy to global memory
  if(tid < init_threads) {
    for(int i=0; i<init_chunk; i++) {
      if(tid * init_chunk + i < hists_per_block * his_sz) {
        d_his[h_start + tid * init_chunk + i] =
          sh_his[tid * init_chunk + i];
      }
    }
  }
}

template<class OP, class T>
void
warp_shared_corp(int *h_img,
                 int *h_his,
                 int img_sz,
                 int his_sz,
                 int num_threads,
                 int seq_chunk,
                 int corp_lvl,
                 int num_hists,
                 struct timeval *t_start,
                 struct timeval *t_end)
{
  if(corp_lvl > 32) {
    printf("Error: corporation level cannot exceed maximum"
           "number of warps in block\n");
    return;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  // histograms per block - maximum value is 1024
  int sh_mem_sz = 48 * 1024;
  int hists_per_block = min((sh_mem_sz / his_mem_sz), 32);
  int thrds_per_block = 32 * corp_lvl; // warp size := 32
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  // compute numbers needed for initialization and copy steps
  int total_hist = hists_per_block * his_sz;
  int sh_chunk =
    ceil(total_hist / (float)thrds_per_block);
  int sh_chunk_threads =
    (total_hist % sh_chunk) == 0 ?
    (total_hist / sh_chunk) :
    (total_hist / sh_chunk) + 1;

  /* printf("Histograms per block: %d\n", hists_per_block); */
  /* printf("Threads per block: %d\n", thrds_per_block); */
  /* printf("Number of blocks: %d\n", num_blocks); */
  /* printf("sh_chunk: %d\n", sh_chunk); */
  /* printf("sh_chunk_threads: %d\n", sh_chunk_threads); */

  if(hists_per_block * his_mem_sz > sh_mem_sz) {
    printf("Error: histograms does not fit in shared memory\n");
    return;
  }

  // d_his contains all histograms from shared memory
  int *d_img, *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his,
             his_mem_sz * num_blocks * hists_per_block);
  cudaMalloc((void **)&d_res, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // first kernel
  dim3 grid_dim_fst (num_blocks, 1, 1);
  dim3 block_dim_fst(thrds_per_block, 1, 1);
  // second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  /* printf("First - grid: %d\n", grid_dim_fst.x); */
  /* printf("First - block: %d\n", block_dim_fst.x); */
  /* printf("Second - grid: %d\n", grid_dim_snd.x); */
  /* printf("Second - block: %d\n", block_dim_snd.x); */

  // execute kernel
  gettimeofday(t_start, NULL);

  warp_shared_corp_kernel<OP, T>
    <<<grid_dim_fst, block_dim_fst, his_mem_sz * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, corp_lvl, num_hists, hists_per_block,
     sh_chunk, sh_chunk_threads);

  cudaThreadSynchronize();

  reduce_kernel<OP, T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);
}



/* -- 5a -- */
/* Warp optimized - lock-free */

/* Warp optimized - assuming one block for now. */
/* You cannot have (hist. sz > block. sz) anyway. */
template<class OP, class T>
__global__ void
warp_optimized_kernel(T *d_img, T *d_his, int img_sz, int his_sz)
{
  const unsigned int warp_size = 32;
  const unsigned int tid = threadIdx.x;
  const unsigned int hid = tid % warp_size; // histogram id

  // initialize local histograms
  extern __shared__ T sh_his[];
  if(tid < warp_size * his_sz) { sh_his[tid] = OP::identity(); }
  __syncthreads();

  // new solution
  /* if(tid < warp_size * his_sz) { */
  /*   for(int i=tid; i<his_sz; i+=blockDim.x) { */
  /*     sh_his[i] = OP::identity(); */
  /*   } */
  /* } */
  /* __syncthreads(); */

  // compute histograms - assume: img_sz < 1024
  if(tid < img_sz) {
    struct indval iv = f(d_img[tid], his_sz);
    int idx = iv.index;
    int val = iv.value;

    // This works
    atomicAdd(&sh_his[hid * his_sz + idx], val);

    // But this doesn't -> race conditions
    /*
    sh_his[hid * his_sz + idx] =
      OP::apply(sh_his[hid * his_sz + idx], val);
    */
  }
  __syncthreads();

  // sum sub-histogram bins and write to global memory
  int sum = 0;
  if(tid < his_sz) {
    for(int i=tid; i<warp_size * his_sz; i += his_sz) {
      sum += sh_his[i];
    }
    d_his[tid] = sum;
  }
}

template<class OP, class T>
int warp_optimized(int *h_img,
                   int *h_his,
                   int img_sz,
                   int his_sz,
                   struct timeval *t_start,
                   struct timeval *t_end)
{
  if(img_sz > 1024) {
    printf("ERROR: Grid dimension can be at most one!\n");
    return -1;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(int); // sizeof(T)?
  unsigned int his_mem_sz = his_sz * sizeof(int); // sizeof(T)?

  int *d_img, *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid_dim (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(img_sz), 1, 1);

  // execute kernel
  gettimeofday(t_start, NULL);

  warp_optimized_kernel<Add<int>, int>
    <<<grid_dim, block_dim, 32 * his_mem_sz>>>
    (d_img, d_his, img_sz, his_sz);
  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);

  return 0;
}
/* -- 5a --
 * Warp optimized - lock-free */


/* -- 5a -- */
/* Warp optimized - lock-free */
/* General warp-optimized version - whatever that means */
template<class OP, class T>
__global__ void
warp_optimized_kernel2(T *d_img,
                        T *d_his,
                        int img_sz,
                        int his_sz)
{
  const unsigned int warp_size = 32;
  const unsigned int tid = threadIdx.x;
  //const unsigned int gid = blockIdx.x * blockDim.x + tid;
  const unsigned int hid = tid % warp_size; // histogram id

  // initialize local histograms
  extern __shared__ T sh_his[];
  if(tid < warp_size * his_sz) { sh_his[tid] = OP::identity(); }
  __syncthreads();

  // compute histograms - assume: img_sz < 1024
  if(tid < img_sz) {
    struct indval iv = f(d_img[tid], his_sz);
    int idx = iv.index;
    int val = iv.value;

    // This works
    atomicAdd(&sh_his[hid * his_sz + idx], val);

    // But this doesn't - race conditions
    /*
    sh_his[hid * his_sz + idx] =
      OP::apply(sh_his[hid * his_sz + idx], val);
    */
  }
  __syncthreads();

  // sum sub-histogram bins and write to global memory
  int sum = 0;
  if(tid < his_sz) {
    for(int i=tid; i < warp_size * his_sz; i += his_sz) {
      sum += sh_his[i];
    }

    d_his[blockIdx.x * his_sz + tid] = sum;
  }
}

template<class OP, class T>
__global__ void
warp_optimized_reduce_kernel(T *d_his_tmp,
                             T *d_his,
                             int img_sz,
                             int his_sz,
                             int tmp_his_sz)
{
  const unsigned int tid = threadIdx.x;

  // reduce bins
  int sum = 0;
  if(tid < his_sz) {
    for(int i=tid; i < tmp_his_sz; i += his_sz) {
      sum = OP::apply(d_his_tmp[i], sum);
    }
  }

  // write to global histogram
  if(tid < his_sz) {
    d_his[tid] = sum;
  }
}

template<class OP, class T>
void warp_optimized2(T *h_img,
                     T *h_his,
                     int img_sz,
                     int his_sz,
                     struct timeval *t_start,
                     struct timeval *t_end)
{
  unsigned int img_mem_sz = img_sz * sizeof(int); // sizeof(T)?
  unsigned int his_mem_sz = his_sz * sizeof(int); // sizeof(T)?

  // allocate device memory
  int *d_img, *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);

  int *d_his_tmp;
  int tmp_his_sz     = GRID_X_DIM(img_sz) * his_sz;
  int tmp_his_mem_sz = GRID_X_DIM(img_sz) * his_mem_sz;
  cudaMalloc((void **)&d_his_tmp, tmp_his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // First kernel
  dim3 grid_dim_fst (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(img_sz), 1, 1);
  // Second kernel
  dim3 grid_dim_snd (GRID_X_DIM (his_sz), 1, 1);
  dim3 block_dim_snd(BLOCK_X_DIM(his_sz), 1, 1);

  // execute kernel
  gettimeofday(t_start, NULL);

  warp_optimized_kernel<Add<int>, int>
    <<<grid_dim_fst, block_dim_fst, 32 * his_mem_sz>>>
    (d_img, d_his, img_sz, his_sz);

  cudaThreadSynchronize();

  warp_optimized_reduce_kernel<Add<int>, int>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his_tmp, d_his, img_sz, his_sz, tmp_his_sz);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_his_tmp);
}
/* -- 5a -- */
/* Warp optimized - lock-free */






/* Indeterministic in global memory */
template<class OP, class IN_T, class OUT_T>
__global__ void
noAtomic_noShared_kernel(IN_T *d_img,
                         OUT_T* d_his,
                         int img_sz,
                         int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    struct indval iv = f(d_img[gid], his_sz);
    int idx = iv.index;
    int val = iv.value;
    d_his[idx] = OP::apply(d_his[idx], val);
  }
}

template<class OP, class IN_T, class OUT_T>
void noAtomic_noShared(IN_T *h_img,
                       OUT_T *h_his,
                       int img_sz,
                       int his_sz,
                       struct timeval *t_start,
                       struct timeval *t_end)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  int *d_img, *d_his;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz);
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_his, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  dim3 grid_dim (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim(BLOCK_X_DIM(img_sz), 1, 1);

  // execute kernel
  gettimeofday(t_start, NULL);

  noAtomic_noShared_kernel<OP, IN_T, OUT_T>
    <<<grid_dim, block_dim>>>
    (d_img, d_his, img_sz, his_sz);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL);

  // copy result from device to host memory
  cudaMemcpy(h_his, d_his, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his);
}
/* Indeterministic in global memory */



/* Atomic update - i.e., the non-combining case (indeterministic) */
template<class T>
__global__ void
scatter_atomicUpdate(T *d_img, T *d_his, int img_sz, int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    struct indval iv = f(d_img[gid], his_sz);
    int idx = iv.index;
    int val = iv.value;
    atomicExch(&d_img[idx], val);
  }
}

#endif // SCATTER_KER
