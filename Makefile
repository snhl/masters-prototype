NVCC=nvcc
# if -arch or -code is not specified, the default (CUDA 8.0) is
# compute_20 and (minimum supported cc).
# -arch=compute_35 -code=sm_35 or just -arch=sm_35

NVCC_FLAGS:=-arch=compute_20 -code=sm_30 -x cu #-Wno-deprecated-gpu-targets
C_OPTIONS:=-O3 -Wall -Werror -Wunused
C_FLAGS:= $(foreach option, $(C_OPTIONS), --compiler-options $(option))
LIBS=-lm

REQS=kernels.cu.h misc.cu.h
FILE=host

.PHONY: all clean

all: $(FILE)

$(FILE): $(FILE).cu $(REQS)
	$(NVCC) $(NVCC_FLAGS) $(C_FLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(FILE)
