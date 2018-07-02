import json
import sys
import numpy as np
import matplotlib
from math import ceil

matplotlib.use('Agg')

import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print "Usage: python {0} <hist. size> <img. size> " \
          "<list of coop. levels>".format(sys.argv[0])
    sys.exit()
else:
    hist_sz = int(sys.argv[1])
    img_sz  = int(sys.argv[2])
    coop_levels = map(int, sys.argv[3:])

# Futhark runtimes
with open("fut_times.json") as infile:
    fut_json  = json.load(infile)
    fut_times = fut_json['hist-reduce.fut']['datasets']

# num. hists. -> coop. lvl.
conversion = { 61350: 1,
               15338: 4,
               3835: 16,
               959: 64,
               240: 256,
               60: 1024,
               15: 4096,
               4: 16384,
               1: 61440,
}

# (hist. size, cooperation level, mean)
fut_means = [ (int(name.split('-')[0]),
               conversion[int(name.split('-')[1])],
               float(np.mean(fut_times[name]['runtimes'])))
              for name in fut_times
]

fut_ = [ (x[1], x[2]) for x in fut_means if x[0] == hist_sz ]

with open("runtimes/hist-{0}.json".format(hist_sz)) as infile:
    runtimes = json.load(infile)

colors = {
    'AADD_NOSHARED_NOCHUNK_FULLCOOP':  'darkblue',
    'AADD_NOSHARED_CHUNK_FULLCOOP':    'cyan',
    'ACAS_NOSHARED_NOCHUNK_FULLCOOP':  'red',
    'ACAS_NOSHARED_CHUNK_FULLCOOP':    'orange',
    'AEXCH_NOSHARED_NOCHUNK_FULLCOOP': 'green',
    'AEXCH_NOSHARED_CHUNK_FULLCOOP':   'lime',
    'AADD_NOSHARED_CHUNK_COOP':  'darkblue',
    'AADD_SHARED_CHUNK_COOP':    'cyan',
    'ACAS_NOSHARED_CHUNK_COOP':  'red',
    'ACAS_SHARED_CHUNK_COOP':    'orange',
    'AEXCH_NOSHARED_CHUNK_COOP': 'green',
    'AEXCH_SHARED_CHUNK_COOP':   'lime',
    'FUTHARK REDUCTION': 'black',
}

# non-coop-varying kernels
non_varying_kernels = [
    'AADD_NOSHARED_NOCHUNK_FULLCOOP',
    'AADD_NOSHARED_CHUNK_FULLCOOP',
    'ACAS_NOSHARED_NOCHUNK_FULLCOOP',
    'ACAS_NOSHARED_CHUNK_FULLCOOP',
    'AEXCH_NOSHARED_NOCHUNK_FULLCOOP',
    'AEXCH_NOSHARED_CHUNK_FULLCOOP',
]

# coop-varying kernels
varying_kernels = [
    'AADD_NOSHARED_CHUNK_COOP',
    'AADD_SHARED_CHUNK_COOP',
    'ACAS_NOSHARED_CHUNK_COOP',
    'ACAS_SHARED_CHUNK_COOP',
    'AEXCH_NOSHARED_CHUNK_COOP',
    'AEXCH_SHARED_CHUNK_COOP',
]

non_varying_kernels = {
    name: runtimes[name] for name in non_varying_kernels
}
varying_kernels = {
    name: runtimes[name] for name in varying_kernels
}

def sort_fst(xs):
    """Sorts a list of tuples according to the first element."""
    return sorted(xs, key=lambda pair: pair[0])

def compute_means(runtimes):
    """Returns a list of tuples containing cooporation level
    and mean runtime."""
#    tmp  = runtimes[kernel_name]
    tmp_ = [ (int(key), float(np.mean(val)))
             for key, val in runtimes.iteritems()
    ]
    return sort_fst(tmp_)

def all_means(runtimes):
    """Returns a dictionary containing kernel names as keys
    and a list of coop.lvl-mean-time tuples."""
    tmp = {}
    for name in runtimes:
        tmp[name] = compute_means(runtimes[name])
    return tmp

## do the graphs
def autolabel(rects, ylim):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        label_y = 1.05 * height if 1.10 * height < ylim else 0.75 * ylim
        ax.text(rect.get_x() + rect.get_width()/2., label_y,
                '%d' % int(height),
                ha='center', va='bottom',
                rotation='vertical')

def plot_all(inds, kernel_means, axis, width, space, ylim):
    tmp_kernels = sort_fst(kernel_means.items())
    tmp = []
    for i, info in enumerate(tmp_kernels):
        name, data = info
        _, time = zip(*data)
        _inds = inds - (1 - space) / 2. + i * width + 0.5 * width
        _tmp = axis.bar(_inds, np.array(time), width,
                        label=name, color=colors[name])
        autolabel(_tmp, ylim)
        tmp += _tmp
    return tmp

# plot for coop-varying kernels
ylim = 60000

kernel_means = all_means(varying_kernels)
kernel_means['FUTHARK REDUCTION'] = sort_fst(fut_)

space = 0.25
width = (1 - space) / len(kernel_means)
indices = np.arange(1, len(coop_levels) + 1)
fig, ax = plt.subplots()


plots = plot_all(indices, kernel_means, ax, width, space, ylim)

# add formula cooporation level -- only if time allows
#indices = np.append(indices, 1.5)

lgd = ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
ax.set_xticks(indices)
#ax.set_xticklabels(coop_levels + [1.5])
ax.set_xticklabels(coop_levels)
ax.set_xlabel('Cooperation level')

ax.set_ylim([0, ylim])

ax.set_ylabel('Runtime ($\mu s$)')
ax.tick_params('y', colors='k')
ax.set_title('N:{0}, H:{1}'.format(img_sz, hist_sz))

#ax.get_xticklabels()[-1].set_color('red')
#ax.get_tick_params()[-1]

hardware  = 61440
threads0  = min(img_sz, hardware)
seq_chunk = int(ceil(img_sz   / float(threads0)))
threads1  = int(ceil(img_sz   / float(seq_chunk)))
coop_lvl  = int(ceil(hist_sz  / float(seq_chunk)))
num_hists = int(ceil(threads1 / float(coop_lvl)))


c1 = "Hardware:\nSeq. chunk:\nThreads:\n" \
      "Coop. level:\nNum. hists:"

c2 = "{0}\n{1}\n{2}\n{3}\n{4}".format(
    threads0, seq_chunk, threads1, coop_lvl, num_hists
)

col1 = 0.675
colspace = .160
plt.figtext(col1           , -0.15, c1, fontweight='bold')
plt.figtext(col1 + colspace, -0.15, c2)

plt.show()
plt.savefig('hist-{0}.pdf'.format(hist_sz),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
)
plt.close()


# plot for non-varying kernels
ylim = 10000

kernel_means = all_means(non_varying_kernels)
space = 0.10
width = (1 - space) / len(kernel_means)
indices = np.arange(1)
fig, ax = plt.subplots()
plots = plot_all(indices, kernel_means, ax, width, space, ylim)

lgd = ax.legend(loc='upper left', bbox_to_anchor=(0, -0.10))
ax.set_xticks(indices)
ax.set_xticklabels(['maximum'])
ax.set_xlabel('Cooperation level')

ax.set_ylim([0, ylim])

ax.set_ylabel('Runtime ($\mu s$)')
ax.tick_params('y', colors='k')
ax.set_title('N:{0}, H:{1}'.format(img_sz, hist_sz))

plt.show()
plt.savefig('hist-{0}-full.pdf'.format(hist_sz),
            bbox_extra_artists=(lgd,),
            bbox_inches='tight')
plt.close()
