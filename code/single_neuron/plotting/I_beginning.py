#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from plot_settings import *

I_p_rec = np.load(simfold + "I_p.npy")
I_d_rec = np.load(simfold + "I_d.npy")

with open(simfold + "params.p","rb") as reader:
	params = pickle.load(reader)

n_t_learn = params['n_t_learn']

figdim = (5.,2.5)

fig_I_beginning, ax_I_beginning = plt.subplots(1,1,figsize = figdim)
t_wind = int(n_t_learn*0.005)

ax_I_beginning.plot(I_d_rec[:t_wind],label="$I_d$")
ax_I_beginning.plot(I_p_rec[:t_wind],label="$I_p$")

ax_I_beginning.legend(loc='upper right')

ax_I_beginning.set_xlabel("#t")
ax_I_beginning.set_ylabel("$I_{p}$, $I_{d}$")
ax_I_beginning.set_title("First " + str(round(100.*t_wind/n_t_learn,1))+"% of learning phase")

plt.tight_layout()

fig_I_beginning.savefig(plotsfold + "I_beginning." + dat_format)

plt.show()