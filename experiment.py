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
          "<list of coop. levels>".format(sys.argv[0])
    sys.exit()
else:
    iterations  = int(sys.argv[1])
    his_sz      = sys.argv[2] # because of subprocess
    filename    = sys.argv[3]
    coop_levels = sys.argv[4:]

# output json file
out_path = join(getcwd(), "runtimes")
out_file = join(out_path, "hist-{0}.json".format(his_sz))

# program to run
pgm = "host"
pgm_path   = "./{0}".format(pgm)

fullcoop_kernels = {
    '10': 'AADD_NOSHARED_NOCHUNK_FULLCOOP',
    '11': 'AADD_NOSHARED_CHUNK_FULLCOOP',

    '20': 'ACAS_NOSHARED_NOCHUNK_FULLCOOP',
    '21': 'ACAS_NOSHARED_CHUNK_FULLCOOP',

    '30': 'AEXCH_NOSHARED_NOCHUNK_FULLCOOP',
    '31': 'AEXCH_NOSHARED_CHUNK_FULLCOOP',
}

vary_coop_kernels = {
    '12': 'AADD_NOSHARED_CHUNK_COOP',
    '13': 'AADD_SHARED_CHUNK_COOP',

    '22': 'ACAS_NOSHARED_CHUNK_COOP',
    '23': 'ACAS_SHARED_CHUNK_COOP',

    '32': 'AEXCH_NOSHARED_CHUNK_COOP',
    '33': 'AEXCH_SHARED_CHUNK_COOP',
}


# ./scatter_host <kernel> <coop. lvl> <hist. sz> <filename>

timing = defaultdict(lambda: defaultdict(list))

# Some kernels must be run with varying cooporation levels
for k in vary_coop_kernels:
    with open(out_file, 'wb') as myfile:
        for clvl in coop_levels:
            for _ in range(iterations):
                print '{0} {1} {2} {3} {4}'.format(
                    pgm_path, k, clvl, his_sz, filename
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
                    vary_coop_kernels[k]
                ][clvl].append(t1)
        json.dump(timing, myfile)

# .. other kernels always produce one histogram
for k in fullcoop_kernels:
    with open(out_file, 'wb') as myfile:
        for _ in range(iterations):
            print '{0} {1} {2} {3} {4}'.format(
                pgm_path, k, clvl, his_sz, filename
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
                fullcoop_kernels[k]
            ][clvl].append(int(t1))
        json.dump(timing, myfile)


# finally we time the sequential version
with open(out_file, 'wb') as myfile:
    for _ in range(iterations):
        print '{0} {1} {2} {3} {4}'.format(
            pgm_path, 'Sequential version', 0, his_sz, filename)
        cmd  = [pgm_path, 00, 0, his_sz, filename]
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
        timing['SEQUENTIAL'].append(int(t1))
    json.dump(timing, myfile)
