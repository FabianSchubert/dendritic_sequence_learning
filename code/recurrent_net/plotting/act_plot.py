#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import ROOT, SIMFOLD
SAVE_PATH = SIMFOLD + "recurrent_net/"

if not(os.path.isdir(SAVE_PATH)):
	print("Data folder not found!")
	sys.exit()



x_e = np.load(SAVE_PATH + "recurrent_net_data.npz")["x_e"]
x_i = np.load(SAVE_PATH + "recurrent_net_data.npz")["x_i"]

n_t, N_layers, N_neurons_layer = x_e.shape

if len(sys.argv) == 1:
	range_t_plot = [ 0, 1000 ]
elif len(sys.argv) == 3:
	try:
		t_start = int(sys.argv[1])
		t_end = int(sys.argv[2])

		if not (t_start < t_end) or not(0 <= t_start < n_t) or not(0 <= t_end < n_t):
			print("Invalid plot range!")
			sys.exit()

		range_t_plot = [t_start, t_end]

	except ValueError:
	    print("Parameters are not integers!")
	    sys.exit()

else:
	print("Invalid number of plotting range parameters! Give either 0 or 2.")
	sys.exit()


t_ax = np.array(range(t_start, t_end))

### Individual Neurons
figdim = (5., 3.)

fig, ax = plt.subplots(N_layers, 1, figsize = figdim, sharex = True)

for k in range(N_layers):
	ax_ind =  N_layers - 1 - k
	ax[ax_ind].pcolormesh(t_ax, range(N_neurons_layer), x[t_start:t_end,k,:].T, cmap = "Greys")
	ax[ax_ind].ticklabel_format(axis = "x", style = "sci", useOffset = t_start)
	ax[ax_ind].set_title("Layer " + str(k), loc = "left")
	ax[ax_ind].set_ylabel("Neuron #")
	
ax[-1].set_xlabel("time steps")

fig.tight_layout()
###

### Populations
figdim_pop = (5., 1.5)

fig_pop, ax_pop = plt.subplots(1,1, figsize = figdim_pop)

ax_pop.plot(t_ax, x[t_start:t_end,:,:].mean(axis=2))

ax_pop.set_xlabel("time steps")
ax_pop.ticklabel_format(axis = "x", style = "sci", useOffset = t_start)

ax_pop.set_ylabel("Population Activity")

fig_pop.tight_layout()
###



plt.show()
