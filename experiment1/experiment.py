from collections import defaultdict
import json
from os import getcwd, listdir
from os.path import join, abspath
from re import findall
from subprocess import check_output, CalledProcessError
import sys

if len(sys.argv) < 5:
    print "Usage: python " \
          "{0} <iterations> <hist. size> <filename> " \
          "<list of corp. levels>".format(sys.argv[0])
    sys.exit()
else:
    iterations  = int(sys.argv[1])
    his_sz      = sys.argv[2] # because of subprocess
    filename    = sys.argv[3]
    corp_levels = sys.argv[4:]

# output json file
out_path = join(getcwd(), "runtimes")
out_file = join(out_path, "hist-{0}.json".format(his_sz))

# program to run
pgm = "host"
pgm_path   = "./../{0}".format(pgm)

fullcorp_kernels = {
    '10': 'AADD_NOSHARED_NOCHUNK_FULLCORP',
    '11': 'AADD_NOSHARED_CHUNK_FULLCORP',

    '20': 'ACAS_NOSHARED_NOCHUNK_FULLCORP',
    '21': 'ACAS_NOSHARED_CHUNK_FULLCORP',

    '30': 'AEXCH_NOSHARED_NOCHUNK_FULLCORP',
    '31': 'AEXCH_NOSHARED_CHUNK_FULLCORP',
}

vary_corp_kernels = {
    '12': 'AADD_NOSHARED_CHUNK_CORP',
    '13': 'AADD_SHARED_CHUNK_CORP',

    '22': 'ACAS_NOSHARED_CHUNK_CORP',
    '23': 'ACAS_SHARED_CHUNK_CORP',

    '32': 'AEXCH_NOSHARED_CHUNK_CORP',
    '33': 'AEXCH_SHARED_CHUNK_CORP',
}


# ./scatter_host <kernel> <corp. lvl> <hist. sz> <filename>

timing = defaultdict(lambda: defaultdict(list))

# Some kernels must be run with varying corporation levels
for k in vary_corp_kernels:
    with open(out_file, 'wb') as myfile:
        for clvl in corp_levels:
            for _ in range(iterations):
                print '{0} {1} {2} {3} {4}'.format(
                    pgm, k, clvl, his_sz, filename
                )
                cmd  = [pgm_path, k, clvl, his_sz, filename]
                try:
                    out  = check_output(cmd)
                    t0 = findall("\d+.\d+", out)
                    t1 = int(t0[0]) if len(t0) == 1 else 0
                    if len(t0) == 0:
                        print 'Did not fail, but got this: '
                        print t0
                except CalledProcessError as err:
                    print err.output
                    t1 = 0
                timing[
                    vary_corp_kernels[k]
                ][clvl].append(t1)
        json.dump(timing, myfile)

# .. other kernels always produce one histogram
for k in fullcorp_kernels:
    with open(out_file, 'wb') as myfile:
        for _ in range(iterations):
            print '{0} {1} {2} {3} {4}'.format(
                pgm, k, clvl, his_sz, filename
            )
            cmd  = [pgm_path, k, clvl, his_sz, filename]
            try:
                out  = check_output(cmd)
                t0 = findall("\d+.\d+", out)
                t1 = int(t0[0]) if len(t0) == 1 else 0
                if len(t0) == 0:
                    print 'Did not fail, but got this: '
                    print t0
            except CalledProcessError as err:
                print err.output
                t1 = 0
            timing[
                fullcorp_kernels[k]
            ][clvl].append(int(t1))
        json.dump(timing, myfile)

# What 'timing' looks like:
#
# {
#     "AADD_NOSHARED_NOCHUNK_FULLCORP":  { "16": [853, 852] },
#     "AADD_NOSHARED_CHUNK_FULLCORP":    { "16": [954, 949] },
#     "ACAS_NOSHARED_NOCHUNK_FULLCORP":  { "16": [18633, 18629] }
#     "ACAS_NOSHARED_CHUNK_FULLCORP":    { "16": [17673, 18174] },
#     "AEXCH_NOSHARED_NOCHUNK_FULLCORP": { "16": [1, 1] },
#     "AEXCH_NOSHARED_CHUNK_FULLCORP":   { "16": [28579, 29807]},

#     "AADD_NOSHARED_CHUNK_CORP": {
#         "1": [32638, 32643],
#         "4": [14273, 14284],
#         "16": [7729, 7694]
#     },
#     "AADD_SHARED_CHUNK_CORP": {
#         "1": [12, 12],
#         "4": [12, 12],
#         "16": [12, 12]
#     },

#     "ACAS_NOSHARED_CHUNK_CORP": {
#         "1": [39102, 39150],
#         "4": [20172, 20118],
#         "16": [12866, 12887]
#     },
#     "ACAS_SHARED_CHUNK_CORP": {
#         "1": [12, 12],
#         "4": [12, 12],
#         "16": [12, 12]
#     },
#     "AEXCH_NOSHARED_CHUNK_CORP": {
#         "1": [48209, 48073],
#         "4": [28165, 28205],
#         "16": [19232, 19223]
#     },
#     "AEXCH_SHARED_CHUNK_CORP": {
#         "1": [6, 6],
#         "4": [6, 6],
#         "16": [6, 6]
#     },
# }
