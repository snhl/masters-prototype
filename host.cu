#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include "kernels.cu.h"
#include "misc.cu.h"

/* x0: One histogram in global memory. One pixel per thread.
 * x1: One histogram in global memory. Chunking.
 * x2: Corporation in global memory.   Chunking.
 * x3: Corp. in sh. and glob. memory.  Chunking.
 */
#define NOATOMIC_NOSHARED  0 // non-deterministic - for completeness

#define AADD_NOSHARED_NOCHUNK_FULLCORP  10
#define AADD_NOSHARED_CHUNK_FULLCORP    11
#define AADD_NOSHARED_CHUNK_CORP        12
#define AADD_SHARED_CHUNK_CORP          13

#define ACAS_NOSHARED_NOCHUNK_FULLCORP  20
#define ACAS_NOSHARED_CHUNK_FULLCORP    21
#define ACAS_NOSHARED_CHUNK_CORP        22
#define ACAS_SHARED_CHUNK_CORP          23

#define AEXCH_NOSHARED_NOCHUNK_FULLCORP 30
#define AEXCH_NOSHARED_CHUNK_FULLCORP   31
#define AEXCH_NOSHARED_CHUNK_CORP       32
#define AEXCH_SHARED_CHUNK_CORP         33

// debugging
#define PRINT_INFO     0
#define PRINT_SEQ_TIME 0
#define PRINT_INVALIDS 0

// runtime
#define MICROS 1 // 0 will give runtime in millisecs.
#define PRINT_RUNTIME(time) (MICROS ? \
  printf("%lu\n", time) : printf("%.3f\n", time / 1000.0))

// misc
#define MY_TYPE int
#define MY_OP   Add<MY_TYPE>
#define SHARED_MEMORY_SIZE (48 * 1024) // default is 48KB

int main(int argc, const char* argv[])
{
  /* validate and parse cmd-line arguments */
  int his_sz, kernel, corp_lvl_tmp;
  if(validate_input(argc, argv,
                    &his_sz, &kernel, &corp_lvl_tmp) != 0) {
    return -1;
  }

  /* abort as soon as possible */
  unsigned int his_mem_sz = his_sz * sizeof(MY_TYPE);
  int restrict = (kernel == 13 || kernel == 23 || kernel == 33);
  if(restrict && (his_mem_sz > SHARED_MEMORY_SIZE)) {
    printf("Error: Histogram exceeds shared memory size\n");
    return -1;
  } else if(restrict && corp_lvl_tmp > BLOCK_SZ) {
    printf("Error: Corporation level exceeds block size\n");
    return -1;
  }

  /* check that data file exists */
  if( access(argv[4], F_OK) == -1 ) {
    printf("Error: file '%s' does not exist\n", argv[2]);
    return 2;
  }

  /* get read handle */
  FILE *fp = fopen(argv[4], "r");
  if(fp == NULL) {
    printf("Error: Did not obtain read handle\n");
    return 3;
  }

  /* parse data file size (first number in file) */
  int img_sz;
  if(fscanf(fp, "%d", &img_sz) != 1) {
    printf("Error: Did not read data size\n");
    fclose(fp);
    return 4;
  }

  /* malloc host memory */
  MY_TYPE *h_img = (MY_TYPE *)malloc(img_sz * sizeof(MY_TYPE));
  MY_TYPE *h_his = (MY_TYPE *)malloc(his_sz * sizeof(MY_TYPE));
  MY_TYPE *h_seq = (MY_TYPE *)malloc(his_sz * sizeof(MY_TYPE));

  /* parse data */
  int pixel;
  for(int i = 0; i < img_sz; i++) {
    if( fscanf(fp, "%d", &pixel) != 1) {
      printf("Error: Incorrect read\n");
      free(h_img); free(h_his); free(h_seq);
      fclose(fp);
      return 7;
    } else {
      h_img[i] = pixel;
    }
  }

  /* close file handle */
  fclose(fp);

  /* initialize result histograms with neutral element */
  initialize_histogram<MY_OP, MY_TYPE>(h_seq, his_sz);
  initialize_histogram<MY_OP, MY_TYPE>(h_his, his_sz);

  /* compute seq. chunk, corp. level and num. histos */
  // 1) N number of threads.

  // 2) varying corp. level
  int num_threads = NUM_THREADS(img_sz);
  int seq_chunk   = SEQ_CHUNK(img_sz, num_threads);
  num_threads = ceil(img_sz / (float)seq_chunk);

  int corp_lvl = 0;
  if(corp_lvl_tmp > num_threads) {
    corp_lvl = num_threads;
  } else if(corp_lvl_tmp == 0) {
    corp_lvl = CORP_LEVEL(his_sz, seq_chunk);
  } else {
    corp_lvl = corp_lvl_tmp;
  }
  int num_hists   = NUM_HISTOS(num_threads, corp_lvl);

  if(PRINT_INFO) {
    printf("== Cosmin's formulas ==\n");
    if(kernel == 10 || kernel == 20 || kernel == 30) {
      printf("Number of threads:    %d\n", img_sz);
      printf("Sequential chunk:     %d\n", 1);
      printf("Corporation level:    %d\n", img_sz);
      printf("Number of histograms: %d\n", 1);
    } else if(kernel == 11 || kernel == 21 || kernel == 31) {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Corporation level:    %d\n", num_threads);
      printf("Number of histograms: %d\n", 1);
    } else {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Corporation level:    %d\n", corp_lvl);
      printf("Number of histograms: %d\n", num_hists);
    }
    printf("====\n");
  }

  /** Kernel versions **/
  int res = 0;
  unsigned long int elapsed;
  struct timeval t_start, t_end, t_diff;

  switch(kernel) {
    /* Indeterministic - should never be used */
  case NOATOMIC_NOSHARED:
    printf("Kernel: NOATOMIC_NOSHARED\n");
    noAtomic_noShared<Add<int>, int>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;

    /* Atomic add */
  case AADD_NOSHARED_NOCHUNK_FULLCORP: // 10
    printf("Kernel: AADD_NOSHARED_NOCHUNK_FULLCORP\n");
    res = aadd_noShared_noChunk_fullCorp<MY_TYPE>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
  case AADD_NOSHARED_CHUNK_FULLCORP: // 11
    printf("Kernel: AADD_NOSHARED_CHUNK_FULLCORP\n");
    res = aadd_noShared_chunk_fullCorp<MY_TYPE>
      (h_img, h_his, img_sz, his_sz, num_threads,
       &t_start, &t_end);
    break;
  case AADD_NOSHARED_CHUNK_CORP: // 12
    printf("Kernel: AADD_NOSHARED_CHUNK_CORP\n");
    res = aadd_noShared_chunk_corp<MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;
  case AADD_SHARED_CHUNK_CORP: // 13
    printf("Kernel: AADD_SHARED_CHUNK_CORP\n");
    res = aadd_shared_chunk_corp<MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;

    /* Locking - CAS */
  case ACAS_NOSHARED_NOCHUNK_FULLCORP: // 20
    printf("Kernel: ACAS_NOSHARED_NOCHUNK_FULLCORP\n");
    res = CAS_noShared_noChunk_fullCorp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
  case ACAS_NOSHARED_CHUNK_FULLCORP: // 21
    printf("Kernel: ACAS_NOSHARED_CHUNK_FULLCORP\n");
    res = CAS_noShared_chunk_fullCorp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       &t_start, &t_end);
    break;
  case ACAS_NOSHARED_CHUNK_CORP: // 22
    printf("Kernel: ACAS_NOSHARED_CHUNK_CORP\n");
    res = CAS_noShared_chunk_corp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;
  case ACAS_SHARED_CHUNK_CORP: // 23
    printf("Kernel: ACAS_SHARED_CHUNK_CORP\n");
    res = CAS_shared_chunk_corp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;

    /* Locking - Exch */
  case AEXCH_NOSHARED_NOCHUNK_FULLCORP: // 30
    printf("Kernel: AEXCH_NOSHARED_NOCHUNK_FULLCORP\n");
    res = exch_noShared_noChunk_fullCorp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
  case AEXCH_NOSHARED_CHUNK_FULLCORP: // 31
    printf("Kernel: AEXCH_NOSHARED_CHUNK_FULLCORP\n");
    res = exch_noShared_chunk_fullCorp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       &t_start, &t_end);
    break;
  case AEXCH_NOSHARED_CHUNK_CORP: // 32
    printf("Kernel: AEXCH_NOSHARED_CHUNK_CORP\n");
    res = exch_noShared_chunk_corp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;
  case AEXCH_SHARED_CHUNK_CORP: // 33
    printf("Kernel: AEXCH_SHARED_CHUNK_CORP\n");
    res = exch_shared_chunk_corp<MY_OP, MY_TYPE>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;

    /* There should be only one warp version */
    /*
  case WARP_SHARED_CORP:
    printf("Kernel: WARP_SHARED_CORP\n");
    warp_shared_corp<Add<int>, int>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, corp_lvl, num_hists,
       &t_start, &t_end);
    break;
  case WARP_OPTIMIZED:
    printf("Kernel: WARP_SHARED_CORP_0\n");
    warp_shared_corp_0<Add<int>, int>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
  case WARP_OPTIMIZED_DBL:
    printf("Kernel: WARP_SHARED_CORP_1\n");
    warp_shared_corp_1<Add<int>, int>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
    */
  }

  if(res != 0) {
    free(h_img); free(h_his); free(h_seq);
    return res;
  }

  /* compute elapsed time for parallel version */
  timeval_subtract(&t_diff, &t_end, &t_start);
  elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;

  /* execute sequential scatter */
  unsigned long int seq_elapsed;
  struct timeval seq_start, seq_end, seq_diff;

  scatter_seq<MY_OP, MY_TYPE>
    (h_img, h_seq, img_sz, his_sz, &seq_start, &seq_end);

  /* compute elapsed time for sequential version */
  timeval_subtract(&seq_diff, &seq_end, &seq_start);
  seq_elapsed = seq_diff.tv_sec * 1e6 + seq_diff.tv_usec;

  /* validate */
  int valid = validate_array<MY_TYPE>(h_his, h_seq, his_sz);
  if(valid) { PRINT_RUNTIME(elapsed); }
  else      { printf("ERROR: Invalid!\n"); res = -1; }

  if(!valid && PRINT_INVALIDS) {
    print_invalid_indices<MY_OP, MY_TYPE>(h_his, h_seq, his_sz);
  }

  if(PRINT_SEQ_TIME) { PRINT_RUNTIME(seq_elapsed); }

  /* free host memory */
  free(h_img); free(h_his); free(h_seq);

  return 0;
}
