.PHONY: all plot rmpdf clean

PGM=../host.cu

DATA_PATH  =data
RUNT_PATH  =runtimes

DATA_SIZE  =10000000
ITERATIONS =5
CORP_LEVELS=1 4 16 64 256 1024 4096 16384 61440 # last is max threads
HISTO_SIZES=16 64 256 1024 4096 16384 61440

.PRECIOUS: $(RUNT_PATH)/hist-%.json hist-%-full.json $(DATA_PATH)/%-$(DATA_SIZE).txt

all:	plot

plot: 	$(HISTO_SIZES:%=hist-%.pdf) $(HISTO_SIZES:%=hist-%-full.pdf)

$(DATA_PATH)/%-$(DATA_SIZE).txt:
	@echo '=== Generating data'
	python generate_image.py $* $(DATA_SIZE)

$(RUNT_PATH)/hist-%.json: $(PGM) $(DATA_PATH)/%-$(DATA_SIZE).txt
	@echo '=== Running experiment'
	python experiment.py $(ITERATIONS) $* \
		$(DATA_PATH)/$*-$(DATA_SIZE).txt $(CORP_LEVELS)

hist-%.pdf hist-%-full.pdf: $(RUNT_PATH)/hist-%.json
	@echo '=== Generating graphs'
	python plot.py $* $(DATA_SIZE) $(CORP_LEVELS)

rmpdf:
	rm -f hist-*-full.pdf hist-*.pdf

clean:
	rm -f $(DATA_PATH)/*
	rm -f $(RUNT_PATH)/*
	rm -f hist-*-full.pdf hist-*.pdf
