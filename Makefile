NVCC=nvcc
NVCC_FLAGS :=-Wno-deprecated-gpu-targets -arch=sm_35
C_OPTIONS :=-O3 -Wall -Werror -Wunused
C_FLAGS := $(foreach option, $(C_OPTIONS), --compiler-options $(option))
LIBS=-lm

REQS=kernels.cu.h misc.cu.h
FILE=host

.PHONY: all clean

all:		$(FILE)

$(FILE):	$(FILE).cu $(REQS)
		$(NVCC) $(NVCC_FLAGS) $(C_FLAGS) $< -o $@ $(LIBS)

clean:
		rm -f $(FILE)
