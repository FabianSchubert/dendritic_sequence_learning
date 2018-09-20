#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from plot_settings import *

figdim = (5.,2.5)

w_dist_rec = np.load(simfold + "w_dist.npy")

fig_w_dist, ax_w_dist = plt.subplots(figsize=figdim)

ax_w_dist.plot(w_dist_rec)
ax_w_dist.set_xlabel("#t")
ax_w_dist.set_ylabel("$w_{d}$")

plt.tight_layout()

fig_w_dist.savefig(plotsfold + "w_dist." + dat_format)

plt.show()