#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from plot_settings import *

X_d = np.load(simfold + "X_d.npy")

with open(simfold + "params.p","rb") as reader:
	params = pickle.load(reader)

n_t_learn = params['n_t_learn']

figdim = (5.,2.5)

fig_X_d, ax_X_d = plt.subplots(figsize=figdim)
t_wind = int(n_t_learn*0.0025)

ax_X_d.plot(X_d[:t_wind])
ax_X_d.set_xlabel("#t")
ax_X_d.set_ylabel("$x_{d}$")

plt.tight_layout()

fig_X_d.savefig(plotsfold + "X_d." + dat_format)

plt.show()