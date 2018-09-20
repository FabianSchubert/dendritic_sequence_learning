#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('../../../misc/custom_style.mplstyle')

simfold = "../../../sim_data/single_neuron/"

I_p_rec = np.load(simfold + "I_p.npy")
I_d_rec = np.load(simfold + "I_d.npy")

with open(simfold + "params.p","rb") as reader:
	params = pickle.load(reader)

n_t_learn = params['n_t_learn']

figdim = (5.,2.5)

fig_I_end, ax_I_end = plt.subplots(1,1,figsize = figdim)
t_ax = np.array(range(n_t_learn))
t_wind = int(n_t_learn*0.005)

ax_I_end.plot(t_ax[-t_wind:],I_d_rec[-t_wind:],label="$I_d$")
ax_I_end.plot(t_ax[-t_wind:],I_p_rec[-t_wind:],label="$I_p$")

ax_I_end.legend(loc='upper right')

ax_I_end.set_xlabel("#t")
ax_I_end.set_ylabel("$I_{p}$, $I_{d}$")
ax_I_end.set_title("Last " + str(round(100.*t_wind/n_t_learn,1))+"% of learning phase")

plt.tight_layout()

plt.show()