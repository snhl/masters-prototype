import sys
import numpy as np
from os import getcwd
from os.path import join

if len(sys.argv) != 3:
    print "Usage: python {0} <upper limit> <image size>".format(
        sys.argv[0])
    sys.exit()

# data path
path = join(getcwd(), "data/cuda")

# parse cmd line arguments
upper, img_sz = int(sys.argv[1]), int(sys.argv[2])

# generate random data for image
lower = 0
img = np.random.randint(low=lower, high=upper, size=img_sz)
img = np.insert(img, 0, img_sz)

# generate absolute path
fname = '{0}-{1}.dat'.format(upper, img_sz)
fname = join(path, fname)

# sloooow - should probably generate data as it is outputted
np.savetxt(fname, img, fmt="%d", delimiter="\n")
