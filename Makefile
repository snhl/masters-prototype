NVCC=nvcc
# if -arch or -code is not specified, the default (CUDA 8.0) is
# compute_20 and (minimum supported cc).
# -arch=compute_35 -code=sm_35 or just -arch=sm_35
#NVCC_FLAGS:= -arch=compute_20 -code=sm_35 -x cu -Wno-deprecated-gpu-targets
NVCC_FLAGS:= -x cu -Wno-deprecated-gpu-targets
C_OPTIONS:=-O3 -Wall -Werror -Wunused
C_FLAGS:= $(foreach option, $(C_OPTIONS), --compiler-options $(option))
LIBS=-lm

REQS=kernels.cu.h misc.cu.h
CU_FILE=host

FUT_FILE=reduce
FC=futhark-opencl

# For experiment
DATA_PATH=data/cuda
RUNT_PATH=runtimes
DATA_SIZE  =10000000
ITERATIONS =5
COOP_LEVELS=1 4 16 64 256 1024 4096 16384 61440 # last is max threads
HISTO_SIZES=16 64 256 1024 4096 16384 61440

.PHONY: all plot clean_runtimes clean_data clean_pfds clean_bins

.PRECIOUS: $(RUNT_PATH)/hist-%.json hist-%-full.json $(DATA_PATH)/%-$(DATA_SIZE).dat $(DATA_PATH)/futhark/%.dat

all:	$(CU_FILE)

# Compile CUDA prototype
$(CU_FILE): $(CU_FILE).cu $(REQS)
	$(NVCC) $(NVCC_FLAGS) $(C_FLAGS) $< -o $@ $(LIBS)

# Compile Futhark reduction program
$(FUT_FILE):	$(FUT_FILE).fut
	$(FC) $<

# Run experiment
plot: 	$(HISTO_SIZES:%=hist-%.pdf) $(HISTO_SIZES:%=hist-%-full.pdf)

# Generate CUDA data (Futhark data should be created manually!)
$(DATA_PATH)/%-$(DATA_SIZE).dat:
	@echo '=== Generating data'
	python generate_image.py $* $(DATA_SIZE)

# Run actual programs
$(RUNT_PATH)/hist-%.json: $(CU_FILE) $(DATA_PATH)/%-$(DATA_SIZE).dat
	@echo '=== Running CUDA experiment'
	python experiment.py $(ITERATIONS) $* \
		$(DATA_PATH)/$*-$(DATA_SIZE).dat $(COOP_LEVELS)

$(RUNT_PATH)/fut_times.json: $(FUT_FILE).fut
	@echo '=== Running Futhark experiment'
	futhark-bench --runs=$(ITERATIONS) --compiler=$(FC) --json $@ $<

# Create graphs
hist-%.pdf hist-%-full.pdf: $(RUNT_PATH)/hist-%.json $(RUNT_PATH)/fut_times.json
	@echo '=== Generating graphs'
	python plot.py $* $(DATA_SIZE) $(COOP_LEVELS)

clean_runtimes:
	rm -f $(RUNT_PATH)/*

clean_data:
#rm -f $(DATA_PATH)/* # you probably don't want to do this

clean_pdfs:
	rm -f hist-*.pdf hist-*-full.pdf

clean_bins:
	rm -f $(CU_FILE)
	rm -f $(FUT_FILE) $(FUT_FILE).c

clean_all:	clean_runtimes clean_data clean_pdfs clean_bins
