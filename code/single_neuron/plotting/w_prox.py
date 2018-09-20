#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('../../../misc/custom_style.mplstyle')

simfold = "../../../sim_data/single_neuron/"

figdim = (5.,2.5)

w_prox_rec = np.load(simfold + "w_prox.npy")

fig_w_prox, ax_w_prox = plt.subplots(figsize=figdim)

ax_w_prox.plot(w_prox_rec)
ax_w_prox.set_xlabel("#t")
ax_w_prox.set_ylabel("$w_{p}$")

#ax_w_prox.get_xaxis().set_ticks(np.linspace(0,n_t_learn,3))

#ax_w_prox.ticklabel_format(style='sci',axis='x',scilimits=(0,0),useMathText=True)
#ax_w_prox.get_xaxis().set_major_formatter(scalform)

plt.tight_layout()

plt.show()