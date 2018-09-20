#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('../../../misc/custom_style.mplstyle')

simfold = "../../../sim_data/single_neuron/"

X_p = np.load(simfold + "X_p.npy")

with open(simfold + "params.p","rb") as reader:
	params = pickle.load(reader)

n_t_learn = params['n_t_learn']

figdim = (5.,2.5)

fig_X_p, ax_X_p = plt.subplots(figsize=figdim)
t_wind = int(n_t_learn*0.01)

ax_X_p.plot(X_p[:t_wind])
ax_X_p.set_xlabel("#t")
ax_X_p.set_ylabel("$x_{p}$")

plt.tight_layout()

plt.show()