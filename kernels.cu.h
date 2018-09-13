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

/* struct for index-value pair */
template<class T>
struct indval {
  int index;
  T value;
};

/* index function */
template<class T>
__device__ __host__ inline
struct indval<T>
f(T pixel, int his_sz)
{
  struct indval<T> iv;
  iv.index = ((int)pixel) % his_sz;
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
    struct indval<OUT_T> iv;

    iv = f<OUT_T>(img[i], his_sz);
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

/* -- KERNEL ID: 10 --  */
/* Atomic add in global memory - one global histogram */
template<class T>
__global__ void
aadd_noShared_noChunk_fullCoop_kernel(T *d_img,
                                      T *d_his,
                                      int img_sz,
                                      int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    struct indval<T> iv = f<T>(d_img[gid], his_sz);
    int idx = iv.index;
    int val = iv.value;
    atomicAdd(&d_his[idx], val);
  }
}

template<class T>
int
aadd_noShared_noChunk_fullCoop(T *h_img,
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

  aadd_noShared_noChunk_fullCoop_kernel<T>
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
aadd_noShared_chunk_fullCoop_kernel(T *d_img,
                                    T *d_his,
                                    int img_sz,
                                    int his_sz,
                                    int num_threads)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval<T> iv = f<T>(d_img[i], his_sz);
        atomicAdd(&d_his[iv.index], iv.value);
    }
  }
}

template<class T>
int
aadd_noShared_chunk_fullCoop(T *h_img,
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

  aadd_noShared_chunk_fullCoop_kernel<T>
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
/* Atomic add in global memory - w. cooporation in global mem. */
template<class T>
__global__ void
aadd_noShared_chunk_coop_kernel(T *d_img,
                                T *d_his,
                                int img_sz,
                                int his_sz,
                                int num_threads,
                                int coop_lvl)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / coop_lvl) * his_sz; // global histogram

  // if-stm could be avoided if we only launch num_threads threads
  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval<T> iv = f<T>(d_img[i], his_sz);
        atomicAdd(&d_his[ghidx + iv.index], iv.value);
    }
  }
}

template<class T>
int
aadd_noShared_chunk_coop(int *h_img,
                         int *h_his,
                         int img_sz,
                         int his_sz,
                         int num_threads,
                         int coop_lvl,
                         int num_hists,
                         struct timeval *t_start,
                         struct timeval *t_end,
                         int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  T *d_img, *d_his, *d_res;
  cudaMalloc((void **)&d_img, img_mem_sz);
  cudaMalloc((void **)&d_his, his_mem_sz * num_hists);
  cudaMalloc((void **)&d_res, his_mem_sz);;
  cudaMemcpy(d_img, h_img, img_mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_his, his_mem_sz, cudaMemcpyHostToDevice);

  // compute grid and block dimensions
  // zeroth kernel -- not needed if we use cudaMemset
  //dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  //dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);

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
  // adds larger overhead than cudaMemset for small histograms
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);

  cudaThreadSynchronize();
  */

  aadd_noShared_chunk_coop_kernel<T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz, num_threads, coop_lvl);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<Add<int>, T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 12 -- */


/* -- KERNEL ID: 13 -- */
/* Atomic add in shared memory - w. cooporation in shared mem. */
template<class T>
__global__ void
aadd_shared_chunk_coop_kernel(T *d_img,
                              T *d_his,
                              int img_sz,
                              int his_sz,
                              int num_threads,
                              int coop_lvl,
                              int num_hists,
                              int hists_per_block)
{
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int his_block_sz = hists_per_block * his_sz;
  int lhid = (tid / coop_lvl) * his_sz; // local histogram idx
  int ghid = blockIdx.x * hists_per_block * his_sz;

  // initialize local histograms
  extern __shared__ T sh_his[];
  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    sh_his[i] = 0;
  }
  __syncthreads();

  // compute local histograms
  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
        struct indval<T> iv = f<T>(d_img[i], his_sz);
        atomicAdd(&sh_his[lhid + iv.index],iv.value);
    }
  }
  __syncthreads();

  // copy local histograms to global memory
  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    d_his[ghid + i] = sh_his[i];
  }
}

template<class T>
int aadd_shared_chunk_coop(T *h_img,
                           T *h_his,
                           int img_sz,
                           int his_sz,
                           int num_threads,
                           int coop_lvl,
                           int num_hists,
                           struct timeval *t_start,
                           struct timeval *t_end,
                           int PRINT_INFO)
{
  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(T);
  unsigned int his_mem_sz = his_sz * sizeof(T);

  // histograms per block - maximum value is 1024
  int hists_per_block = min(
                            (SH_MEM_SZ / his_mem_sz),
                            (BLOCK_SZ / coop_lvl)
                            );
  int thrds_per_block = hists_per_block * coop_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  // For debugging.
  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
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

  aadd_shared_chunk_coop_kernel<T>
    <<<grid_dim_fst, block_dim_fst, his_mem_sz * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, coop_lvl, num_hists, hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<Add<int>, T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

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
CAS_noShared_noChunk_fullCoop_kernel(IN_T  *d_img,
                                     OUT_T *d_his,
                                     int img_sz,
                                     int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < img_sz) {
    int idx; OUT_T val;
    struct indval<OUT_T> iv;

    iv = f<OUT_T>(d_img[gid], his_sz);
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
CAS_noShared_noChunk_fullCoop(IN_T  *h_img,
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

  CAS_noShared_noChunk_fullCoop_kernel<OP, IN_T, OUT_T>
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
CAS_noShared_chunk_fullCoop_kernel(IN_T *d_img,
                                   OUT_T *d_his,
                                   int img_sz,
                                   int his_sz,
                                   int num_threads)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    int idx; OUT_T val;
    struct indval<OUT_T> iv;

    for(int i=gid; i<img_sz; i+=num_threads) {
      iv = f<OUT_T>(d_img[i], his_sz);
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
CAS_noShared_chunk_fullCoop(IN_T  *h_img,
                            OUT_T *h_his,
                            int img_sz,
                            int his_sz,
                            int num_threads,
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

  CAS_noShared_chunk_fullCoop_kernel<OP, IN_T, OUT_T>
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
/* -- KERNEL ID: 21 -- */


/* -- KERNEL ID: 22 -- */
/* Manual lock - CAS in global memory - coop. in global mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_noShared_chunk_coop_kernel(IN_T  *d_img,
                               OUT_T *d_his,
                               int img_sz,
                               int his_sz,
                               int num_threads,
                               int coop_lvl)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / coop_lvl) * his_sz;

  if(gid < num_threads) {
    int idx; OUT_T val;
    struct indval<OUT_T> iv;

    for(int i=gid; i<img_sz; i+=num_threads) {
      iv = f<OUT_T>(d_img[i], his_sz);
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
CAS_noShared_chunk_coop(IN_T  *h_img,
                        OUT_T *h_his,
                        int img_sz,
                        int his_sz,
                        int num_threads,
                        int coop_lvl,
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
  // zeroth kernel - unnecessary if we use cudaMemset
  //dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  //dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
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
  // seems not to make a big difference even for large histograms
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);

  cudaThreadSynchronize();
  */

  CAS_noShared_chunk_coop_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz, num_threads, coop_lvl);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 22 -- */


/* -- KERNEL ID: 23 -- */
/* Manual lock - CAS in shared memory - coop. in shared mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
CAS_shared_chunk_coop_kernel(IN_T  *d_img,
                             OUT_T *d_his,
                             int img_sz,
                             int his_sz,
                             int num_threads,
                             int coop_lvl,
                             int num_hists, // it this needed?
                             int hists_per_block)
{
  // global thread id
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int his_block_sz = hists_per_block * his_sz;
  int lhidx = (tid / coop_lvl) * his_sz;
  int ghidx = blockIdx.x * hists_per_block * his_sz;

  // initialize local histograms
  extern __shared__ OUT_T sh_his[];
  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    sh_his[i] = OP::identity();
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    for(int i=gid; i<img_sz; i+=num_threads) {
      int idx; OUT_T val;
      struct indval<OUT_T> iv;

      iv = f<OUT_T>(d_img[i], his_sz);
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
  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    d_his[ghidx + i] = sh_his[i];
  }
}

template<class OP, class IN_T, class OUT_T>
int
CAS_shared_chunk_coop(IN_T  *h_img,
                      OUT_T *h_his,
                      int img_sz,
                      int his_sz,
                      int num_threads,
                      int coop_lvl,
                      int num_hists,
                      struct timeval *t_start,
                      struct timeval *t_end,
                      int PRINT_INFO)
{
  // because of shared memory -- should be avoided from host.cu
  if(coop_lvl > BLOCK_SZ) {
    printf("Error: cooporation level cannot exceed block size\n");
    return -1;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);

  // histograms per block - maximum value is 1024
  int hists_per_block = min(
                            (SH_MEM_SZ / his_mem_sz),
                            (BLOCK_SZ / coop_lvl)
                            );
  int thrds_per_block = hists_per_block * coop_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
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

  CAS_shared_chunk_coop_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst, his_mem_sz * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, coop_lvl, num_hists, hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

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
exch_noShared_noChunk_fullCoop_kernel(IN_T  *d_img,
                                      OUT_T *d_his,
                                      volatile int *locks,
                                      int img_sz,
                                      int his_sz)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  int done, idx; OUT_T val;
  struct indval<OUT_T> iv;

  if(gid < img_sz) {
    done = 0;
    iv = f<OUT_T>(d_img[gid], his_sz);
    idx = iv.index;
    val = iv.value;
  } else {
    done = 1;
  }

  while(!done) {
    if(atomicExch((int *)&locks[idx], 1) == 0) {
      d_his[idx] = OP::apply(d_his[idx], val);
      //__threadfence();
      //atomicExch((int *)&locks[idx], 0);
      locks[idx] = 0;
      done = 1;
    }
    __threadfence(); // the code also works without this
                     // threadfence -- but it is slower
                     // without (haven't thought about why)
  }
}

template<class OP, class IN_T, class OUT_T>
int
exch_noShared_noChunk_fullCoop(IN_T  *h_img,
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
  // zeroth kernel - for locks -- not needed if using cudaMemset
  //dim3 grid_dim (GRID_X_DIM (his_sz), 1, 1);
  //dim3 block_dim(BLOCK_X_DIM(his_sz), 1, 1);
  // first kernel
  dim3 grid_dim_fst (GRID_X_DIM (img_sz), 1, 1);
  dim3 block_dim_fst(BLOCK_X_DIM(img_sz), 1, 1);

  if(PRINT_INFO) {
    printf("Grid: %d\n",  grid_dim_fst.x);
    printf("Block: %d\n", block_dim_fst.x);
  }

  // execute kernel
  gettimeofday(t_start, NULL);

  // Initializing locks
  // - doesn't matter if we use memset or an extra kernel
  cudaMemset(d_locks, 0, lck_mem_sz);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, 0);
  cudaThreadSynchronize();
  */

  exch_noShared_noChunk_fullCoop_kernel<OP, IN_T, OUT_T>
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
exch_noShared_chunk_fullCoop_kernel(OUT_T *d_img,
                                    IN_T  *d_his,
                                    volatile int *d_locks,
                                    int img_sz,
                                    int his_sz,
                                    int num_threads,
                                    int seq_chunk)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < num_threads) {
    for(int i=0; i<seq_chunk; i++) {
      int done, idx; OUT_T val;
      struct indval<OUT_T> iv;

      if(gid + i * num_threads < img_sz) {
        done = 0;
        iv = f<OUT_T>(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&d_locks[idx], 1) == 0 ) {
          d_his[idx] = OP::apply(d_his[idx], val);
          //__threadfence();
          //atomicExch((int *)&d_locks[idx], 0);
          d_locks[idx] = 0;
          done = 1;
        }
        __threadfence();
      }
    }
  }
}

template<class OP, class IN_T, class OUT_T>
int
exch_noShared_chunk_fullCoop(IN_T  *h_img,
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

  // Initialize locks
  cudaMemset(d_locks, 0, his_mem_sz);

  /*
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, 1);
  cudaThreadSynchronize();
  */

  exch_noShared_chunk_fullCoop_kernel<OP, IN_T, OUT_T>
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
/* Manual lock - Exch. in global memory - coop. in global mem.  */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_noShared_chunk_coop_kernel(IN_T  *d_img,
                                OUT_T *d_his,
                                int img_sz,
                                int his_sz,
                                int num_threads,
                                int seq_chunk,
                                int coop_lvl,
                                volatile int *d_locks)
{
  const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int ghidx = (gid / coop_lvl) * his_sz;

  if(gid < num_threads) {
    int done, idx; OUT_T val;

    for(int i=0; i<seq_chunk; i++) {
      if(gid + i * num_threads < img_sz) {
        struct indval<OUT_T> iv;
        done = 0;
        iv = f<OUT_T>(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&d_locks[ghidx + idx], 1) == 0 ) {
          d_his[ghidx + idx] =
            OP::apply(d_his[ghidx + idx], val);
          //__threadfence(); // necessary if not atomicExch
          //atomicExch((int *)&d_locks[ghidx + idx], 0);
          d_locks[ghidx + idx] = 0;
          done = 1;
        }
        __threadfence();
      }
    }
  }
}


template<class OP, class IN_T, class OUT_T>
int
exch_noShared_chunk_coop(IN_T  *h_img,
                         OUT_T *h_his,
                         int img_sz,
                         int his_sz,
                         int num_threads,
                         int seq_chunk,
                         int coop_lvl,
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

  // Initialize histogram and locks. Obviously, this won't
  // work if the identity element is different from zero.
  cudaMemset(d_his,   0, his_mem_sz * num_hists);
  cudaMemset(d_locks, 0, his_mem_sz * num_hists);

  /*
  // seems to add a significant amount of time
  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_his, his_sz, num_hists);
  cudaThreadSynchronize();

  initialization_kernel<Add<int>, int>
    <<<grid_dim, block_dim>>>
    (d_locks, his_sz, num_hists);
  cudaThreadSynchronize();
  */

  exch_noShared_chunk_coop_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, coop_lvl, d_locks);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_hists);

  cudaThreadSynchronize();

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
/* Manual lock - Exch. in shared memory - coop. in shared mem. */
template<class OP, class IN_T, class OUT_T>
__global__ void
exch_shared_chunk_coop_kernel(IN_T  *d_img,
                              OUT_T *d_his,
                              int img_sz,
                              int his_sz,
                              int num_threads,
                              int seq_chunk,
                              int coop_lvl,
                              int num_hists,
                              int hists_per_block)
{
  // global thread id
  const unsigned int tid = threadIdx.x;
  const unsigned int gid = blockIdx.x * blockDim.x + tid;
  int lhidx = (tid / coop_lvl) * his_sz;
  int ghidx = blockIdx.x * hists_per_block * his_sz;
  int his_block_sz = hists_per_block * his_sz;

  // initialize local histograms and locks
  volatile extern __shared__ int sh_mem[];
  volatile OUT_T *sh_his = sh_mem;
  volatile int *sh_lck = (int *)&sh_his[his_block_sz];

  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    sh_his[i] = OP::identity();
    sh_lck[i] = 0;
  }
  __syncthreads();

  // scatter
  if(gid < num_threads) {
    int done, idx; OUT_T val;
    for(int i=0; i<seq_chunk; i++) {
      if(gid + i * num_threads < img_sz) {
        struct indval<OUT_T> iv;
        done = 0;
        iv = f<OUT_T>(d_img[gid + i * num_threads], his_sz);
        idx = iv.index;
        val = iv.value;
      } else {
        done = 1;
      }

      while(!done) {
        if( atomicExch((int *)&sh_lck[lhidx + idx], 1) == 0 ) {
          sh_his[lhidx + idx] =
            OP::apply(sh_his[lhidx + idx], val);
          __threadfence();
          atomicExch((int *)&sh_lck[lhidx + idx], 0);
          //sh_lck[lhidx + idx] = 0;
          done = 1;
        }
        //__threadfence(); // this stalls (threadfence_block
        //doesn't help)
      }
    }
  }
  __syncthreads();

  // copy to global memory
  for(int i=tid; i<his_block_sz; i+=blockDim.x) {
    d_his[ghidx + i] = sh_his[i];
  }
}

template<class OP, class IN_T, class OUT_T>
int
exch_shared_chunk_coop(IN_T  *h_img,
                       OUT_T *h_his,
                       int img_sz,
                       int his_sz,
                       int num_threads,
                       int seq_chunk,
                       int coop_lvl,
                       int num_hists,
                       struct timeval *t_start,
                       struct timeval *t_end,
                       int PRINT_INFO)
{
  if(coop_lvl > BLOCK_SZ) {
    printf("Error: cooporation level cannot exceed block size\n");
    return -1;
  }

  // allocate device memory
  unsigned int img_mem_sz = img_sz * sizeof(IN_T);
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  unsigned int lck_mem_sz = his_sz * sizeof(int);

  // histograms per block - maximum value is 1024
  int hists_per_block = min(
                            (SH_MEM_SZ /
                             (his_mem_sz + lck_mem_sz)
                            ),
                            (BLOCK_SZ / coop_lvl)
                            );
  int thrds_per_block = hists_per_block * coop_lvl;
  int num_blocks = ceil(num_hists / (float)hists_per_block);

  if(PRINT_INFO) {
    printf("Histograms per block: %d\n", hists_per_block);
    printf("Threads per block: %d\n", thrds_per_block);
    printf("Number of blocks: %d\n", num_blocks);
  }

  if(hists_per_block * (his_mem_sz + lck_mem_sz) > SH_MEM_SZ) {
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

  exch_shared_chunk_coop_kernel<OP, IN_T, OUT_T>
    <<<grid_dim_fst, block_dim_fst,
    (lck_mem_sz + his_mem_sz) * hists_per_block>>>
    (d_img, d_his, img_sz, his_sz,
     num_threads, seq_chunk, coop_lvl, num_hists, hists_per_block);

  cudaThreadSynchronize();

  gettimeofday(t_end, NULL); // do not time reduction

  reduce_kernel<OP, OUT_T>
    <<<grid_dim_snd, block_dim_snd>>>
    (d_his, d_res, img_sz, his_sz, num_blocks * hists_per_block);

  cudaThreadSynchronize();

  int res = gpuAssert( cudaPeekAtLastError() );

  // copy result from device to host memory
  cudaMemcpy(h_his, d_res, his_mem_sz, cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_img); cudaFree(d_his); cudaFree(d_res);

  return res;
}
/* -- KERNEL ID: 33 -- */


#endif // SCATTER_KER
