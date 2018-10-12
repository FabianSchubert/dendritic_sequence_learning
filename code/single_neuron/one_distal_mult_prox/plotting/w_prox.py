#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import SIMFOLD, PLOTSFOLD
SIMFOLD_SINGLE_NEURON = SIMFOLD + "single_neuron/one_distal_mult_prox/"
PLOTSFOLD_SINGLE_NEURON = PLOTSFOLD + "single_neuron/one_distal_mult_prox/"

figdim = (5.,2.5)

w_prox_rec = np.load(SIMFOLD_SINGLE_NEURON + "w_prox.npy")

fig_w_prox, ax_w_prox = plt.subplots(figsize=figdim)

ax_w_prox.plot(w_prox_rec)
ax_w_prox.set_xlabel("#t")
ax_w_prox.set_ylabel("$w_{p}$")

#ax_w_prox.get_xaxis().set_ticks(np.linspace(0,n_t_learn,3))

#ax_w_prox.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
#ax_w_prox.get_xaxis().set_major_formatter(scalform)

plt.tight_layout()

fig_w_prox.savefig(PLOTSFOLD_SINGLE_NEURON + "w_prox." + DAT_FORMAT)

plt.show()