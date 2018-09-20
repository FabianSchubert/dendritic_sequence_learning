#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('../../../misc/custom_style.mplstyle')

simfold = "../../../sim_data/single_neuron/"

figdim = (5.,2.5)

w_dist_rec = np.load(simfold + "w_dist.npy")

fig_w_dist, ax_w_dist = plt.subplots(figsize=figdim)

ax_w_dist.plot(w_dist_rec)
ax_w_dist.set_xlabel("#t")
ax_w_dist.set_ylabel("$w_{p}$")

plt.tight_layout()

plt.show()