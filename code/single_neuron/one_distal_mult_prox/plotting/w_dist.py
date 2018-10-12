#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import SIMFOLD, PLOTSFOLD
SIMFOLD_SINGLE_NEURON = SIMFOLD + "single_neuron/one_distal_mult_prox/"
PLOTSFOLD_SINGLE_NEURON = PLOTSFOLD + "single_neuron/one_distal_mult_prox/"

figdim = (5.,2.5)

w_dist_rec = np.load(SIMFOLD_SINGLE_NEURON + "w_dist.npy")

fig_w_dist, ax_w_dist = plt.subplots(figsize=figdim)

ax_w_dist.plot(w_dist_rec)
ax_w_dist.set_xlabel("#t")
ax_w_dist.set_ylabel("$w_{d}$")

plt.tight_layout()

fig_w_dist.savefig(PLOTSFOLD_SINGLE_NEURON + "w_dist." + DAT_FORMAT)

plt.show()