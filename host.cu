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
 * x2: Cooporation in global memory.   Chunking.
 * x3: Coop. in sh. and glob. memory.  Chunking.
 */
#define NOATOMIC_NOSHARED  0 // non-deterministic - for completeness

#define AADD_NOSHARED_NOCHUNK_FULLCOOP  10
#define AADD_NOSHARED_CHUNK_FULLCOOP    11
#define AADD_NOSHARED_CHUNK_COOP        12
#define AADD_SHARED_CHUNK_COOP          13

#define ACAS_NOSHARED_NOCHUNK_FULLCOOP  20
#define ACAS_NOSHARED_CHUNK_FULLCOOP    21
#define ACAS_NOSHARED_CHUNK_COOP        22
#define ACAS_SHARED_CHUNK_COOP          23

#define AEXCH_NOSHARED_NOCHUNK_FULLCOOP 30
#define AEXCH_NOSHARED_CHUNK_FULLCOOP   31
#define AEXCH_NOSHARED_CHUNK_COOP       32
#define AEXCH_SHARED_CHUNK_COOP         33

// debugging
#define PRINT_INFO     0
#define PRINT_SEQ_TIME 0
#define PRINT_INVALIDS 0

// runtime
#define MICROS 1 // 0 will give runtime in millisecs.
#define PRINT_RUNTIME(time) (MICROS ? \
  printf("%lu\n", time) : printf("%.3f\n", time / 1000.0))

// misc
#define IN_T  int
#define OUT_T int
#define MY_OP Add<OUT_T>
#define SHARED_MEMORY_SIZE (48 * 1024) // default is 48KB

int main(int argc, const char* argv[])
{
  /* validate and parse cmd-line arguments */
  int his_sz, kernel, coop_lvl_tmp;
  if(validate_input(argc, argv,
                    &his_sz, &kernel, &coop_lvl_tmp) != 0) {
    return -1;
  }

  /* abort as soon as possible */
  unsigned int his_mem_sz = his_sz * sizeof(OUT_T);
  int restrict = (kernel == 13 || kernel == 23 || kernel == 33);
  if(restrict && (his_mem_sz > SHARED_MEMORY_SIZE)) {
    printf("Error: Histogram exceeds shared memory size\n");
    return -1;
  } else if(restrict && coop_lvl_tmp > BLOCK_SZ) {
    printf("Error: Cooporation level exceeds block size\n");
    return -1;
  }

  /* check that data file exists */
  if( access(argv[4], F_OK) == -1 ) {
    printf("Error: file '%s' does not exist\n", argv[4]);
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
  IN_T  *h_img = (IN_T  *)malloc(img_sz * sizeof(IN_T));
  OUT_T *h_his = (OUT_T *)malloc(his_sz * sizeof(OUT_T));
  OUT_T *h_seq = (OUT_T *)malloc(his_sz * sizeof(OUT_T));

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
  initialize_histogram<MY_OP, OUT_T>(h_seq, his_sz);
  initialize_histogram<MY_OP, OUT_T>(h_his, his_sz);

  /* compute seq. chunk, coop. level and num. histos */
  // 1) N number of threads.

  // 2) varying coop. level
  int num_threads = NUM_THREADS(img_sz);
  int seq_chunk   = SEQ_CHUNK(img_sz, num_threads);
  num_threads = ceil(img_sz / (float)seq_chunk);

  int coop_lvl = 0;
  if(coop_lvl_tmp > num_threads) {
    coop_lvl = num_threads;
  } else if(coop_lvl_tmp == 0) {
    coop_lvl = COOP_LEVEL(his_sz, seq_chunk);
  } else {
    coop_lvl = coop_lvl_tmp;
  }
  int num_hists   = NUM_HISTOS(num_threads, coop_lvl);

  if(PRINT_INFO) {
    printf("== Cosmin's formulas ==\n");
    if(kernel == 10 || kernel == 20 || kernel == 30) {
      printf("Number of threads:    %d\n", img_sz);
      printf("Sequential chunk:     %d\n", 1);
      printf("Cooporation level:    %d\n", img_sz);
      printf("Number of histograms: %d\n", 1);
    } else if(kernel == 11 || kernel == 21 || kernel == 31) {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooporation level:    %d\n", num_threads);
      printf("Number of histograms: %d\n", 1);
    } else {
      printf("Number of threads:    %d\n", num_threads);
      printf("Sequential chunk:     %d\n", seq_chunk);
      printf("Cooporation level:    %d\n", coop_lvl);
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
    noAtomic_noShared<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;

    /* Atomic add */
  case AADD_NOSHARED_NOCHUNK_FULLCOOP: // 10
    printf("Kernel: AADD_NOSHARED_NOCHUNK_FULLCOOP\n");
    res = aadd_noShared_noChunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end,
       PRINT_INFO);
    break;
  case AADD_NOSHARED_CHUNK_FULLCOOP: // 11
    printf("Kernel: AADD_NOSHARED_CHUNK_FULLCOOP\n");
    res = aadd_noShared_chunk_fullCoop<IN_T>
      (h_img, h_his, img_sz, his_sz, num_threads,
       &t_start, &t_end, PRINT_INFO);
    break;
  case AADD_NOSHARED_CHUNK_COOP: // 12
    printf("Kernel: AADD_NOSHARED_CHUNK_COOP\n");
    res = aadd_noShared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;
  case AADD_SHARED_CHUNK_COOP: // 13
    printf("Kernel: AADD_SHARED_CHUNK_COOP\n");
    res = aadd_shared_chunk_coop<IN_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;

    /* Locking - CAS */
  case ACAS_NOSHARED_NOCHUNK_FULLCOOP: // 20
    printf("Kernel: ACAS_NOSHARED_NOCHUNK_FULLCOOP\n");
    res = CAS_noShared_noChunk_fullCoop
      <MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end,
       PRINT_INFO);
    break;
  case ACAS_NOSHARED_CHUNK_FULLCOOP: // 21
    printf("Kernel: ACAS_NOSHARED_CHUNK_FULLCOOP\n");
    res = CAS_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       &t_start, &t_end, PRINT_INFO);
    break;
  case ACAS_NOSHARED_CHUNK_COOP: // 22
    printf("Kernel: ACAS_NOSHARED_CHUNK_COOP\n");
    res = CAS_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;
  case ACAS_SHARED_CHUNK_COOP: // 23
    printf("Kernel: ACAS_SHARED_CHUNK_COOP\n");
    res = CAS_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;

    /* Locking - Exch */
  case AEXCH_NOSHARED_NOCHUNK_FULLCOOP: // 30
    printf("Kernel: AEXCH_NOSHARED_NOCHUNK_FULLCOOP\n");
    res = exch_noShared_noChunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end,
       PRINT_INFO);
    break;
  case AEXCH_NOSHARED_CHUNK_FULLCOOP: // 31
    printf("Kernel: AEXCH_NOSHARED_CHUNK_FULLCOOP\n");
    res = exch_noShared_chunk_fullCoop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz, num_threads, seq_chunk,
       &t_start, &t_end, PRINT_INFO);
    break;
  case AEXCH_NOSHARED_CHUNK_COOP: // 32
    printf("Kernel: AEXCH_NOSHARED_CHUNK_COOP\n");
    res = exch_noShared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;
  case AEXCH_SHARED_CHUNK_COOP: // 33
    printf("Kernel: AEXCH_SHARED_CHUNK_COOP\n");
    res = exch_shared_chunk_coop<MY_OP, IN_T, OUT_T>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end, PRINT_INFO);
    break;

    /* There should be only one warp version */
    /*
  case WARP_SHARED_COOP:
    printf("Kernel: WARP_SHARED_COOP\n");
    warp_shared_coop<Add<int>, int>
      (h_img, h_his, img_sz, his_sz,
       num_threads, seq_chunk, coop_lvl, num_hists,
       &t_start, &t_end);
    break;
  case WARP_OPTIMIZED:
    printf("Kernel: WARP_SHARED_COOP_0\n");
    warp_shared_coop_0<Add<int>, int>
      (h_img, h_his, img_sz, his_sz, &t_start, &t_end);
    break;
  case WARP_OPTIMIZED_DBL:
    printf("Kernel: WARP_SHARED_COOP_1\n");
    warp_shared_coop_1<Add<int>, int>
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

  scatter_seq<MY_OP, IN_T, OUT_T>
    (h_img, h_seq, img_sz, his_sz, &seq_start, &seq_end);

  /* compute elapsed time for sequential version */
  timeval_subtract(&seq_diff, &seq_end, &seq_start);
  seq_elapsed = seq_diff.tv_sec * 1e6 + seq_diff.tv_usec;

  /* validate */
  int valid = validate_array<OUT_T>(h_his, h_seq, his_sz);
  if(valid) { PRINT_RUNTIME(elapsed); }
  else      { printf("ERROR: Invalid!\n"); res = -1; }

  if(!valid && PRINT_INVALIDS) {
    print_invalid_indices<MY_OP, OUT_T>(h_his, h_seq, his_sz);
  }

  if(PRINT_SEQ_TIME) { PRINT_RUNTIME(seq_elapsed); }

  /* free host memory */
  free(h_img); free(h_his); free(h_seq);

  return 0;
}
