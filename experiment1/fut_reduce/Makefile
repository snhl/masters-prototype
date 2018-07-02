.PHONY:	all clean

PGM:=hist-reduce.fut
HIST_SIZE:=61440 #16 64 256 1024 4096 16384 61440
NUM_HISTS:=1 4 15 60 240 959 3835 #15338 61350

# do not make these - slow to generate and not needed
#H:16384-61350
#H:61440-15338
#H:61440-61350

all:	fut_times.json

.PRECIOUS: *-*

hist_%:
	for i in 1 4 15 60 240 959 3835 15338; do \
		futhark-dataset --binary --i32-bounds=0:$* \
			-g [$$i][$*]i32 > $*-$$i ; \
	done

fut_times.json: $(PGM) # $(HIST_SIZE:%=hist_%)
	futhark-bench --runs=5 --compiler=futhark-opencl --json $@ $<

rmdata:
	for h in $(HIST_SIZE); do \
		rm -f $$h-*  ; \
	done

clean:
	rm -f fut_times.json
