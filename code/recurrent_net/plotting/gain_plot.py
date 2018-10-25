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


gain = np.load(SAVE_PATH + "recurrent_net_data.npz")["gain"]

n_t, N_layers, N_neurons_layer = gain.shape

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
n_colors = len(colors)

figdim = (5., 3.)

fig, ax = plt.subplots(1, 1, figsize = figdim)

for k in range(N_layers):

	col = colors[k%n_colors]

	ax.plot(gain[::100,k,:], color = col, alpha = 0.25)

ax.set_xlabel("Time Step")
ax.set_ylabel("Gain Factor")

plt.show()


