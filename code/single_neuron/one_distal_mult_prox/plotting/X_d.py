#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import SIMFOLD, PLOTSFOLD
SIMFOLD_SINGLE_NEURON = SIMFOLD + "single_neuron/one_distal_mult_prox/"
PLOTSFOLD_SINGLE_NEURON = PLOTSFOLD + "single_neuron/one_distal_mult_prox/"

X_d = np.load(SIMFOLD_SINGLE_NEURON + "X_d.npy")

with open(SIMFOLD_SINGLE_NEURON + "params.p","rb") as reader:
	params = pickle.load(reader)

n_t_learn = params['n_t_learn']

figdim = (5.,2.5)

fig_X_d, ax_X_d = plt.subplots(figsize=figdim)
t_wind = int(n_t_learn*0.0025)

ax_X_d.plot(X_d[:t_wind])
ax_X_d.set_xlabel("#t")
ax_X_d.set_ylabel("$x_{d}$")

plt.tight_layout()

fig_X_d.savefig(PLOTSFOLD_SINGLE_NEURON + "X_d." + DAT_FORMAT)

plt.show()