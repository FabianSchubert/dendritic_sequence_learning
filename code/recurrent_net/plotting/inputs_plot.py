#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

import pdb

import os
import sys

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import ROOT, SIMFOLD
SAVE_PATH = SIMFOLD + "recurrent_net/"

def sigm(x):
	return ( np.tanh( x/2. ) + 1. ) / 2.

def act_pd(p,d,alpha,gain):

	return sigm(gain*d) + alpha*sigm(gain*p)*sigm(-gain*d)


if not(os.path.isdir(SAVE_PATH)):
	print("Data folder not found!")
	sys.exit()


dat = np.load(SAVE_PATH + "recurrent_net_data.npz")

I_f = dat["I_f"]
I_b = dat["I_b"]
I_ei = dat["I_ei"]
I_ie = dat["I_ie"]
I_ii = dat["I_ii"]
I_sens = dat["I_sens"]
I_top_down = dat["I_top_down"]

th_e_p = dat["th_e_p"]
th_e_d = dat["th_e_d"]

params = dat["params"][()]

#pdb.set_trace()

locals().update(params)

#pdb.set_trace()

I_p = I_f + I_ei
I_d = I_b

I_i = I_ie + I_ii

n_t, N_layers, N_e_layer = I_f.shape
N_i_layer = I_ie.shape[2]

cols = mpl.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(N_layers,1)

plot_start = n_t - 50
plot_length = 50

for k in range(N_layers):
	ax[k].set_title("Layer " + str(k))
	ax[k].set_xlabel("Time Step")
	ax[k].set_ylabel("Input")
	
	ax[k].plot(I_p[:,k,:],c=cols[0],alpha=0.2,label="Proximal Input to Exc.")
	ax[k].plot(I_d[:,k,:],c=cols[1],alpha=0.2,label="Distal Input to Exc.")
	ax[k].plot(I_i[:,k,:],c=cols[2],alpha=0.2,label="Input to Inh.")
	
	ax[k].set_xlim([plot_start,plot_start+plot_length])


fig_pd, ax_pd = plt.subplots()

I_p_example = I_p[plot_start:plot_start+plot_length,0,0]
I_d_example = I_d[plot_start:plot_start+plot_length,0,0]

pdb.set_trace()

th_p_example = th_e_p[plot_start:plot_start+plot_length,0,0]
th_d_example = th_e_d[plot_start:plot_start+plot_length,0,0]

i_p = np.linspace(I_p_example.min(),I_p_example.max(),100)
i_d = np.linspace(I_d_example.min(),I_d_example.max(),100)

I_P, I_D = np.meshgrid(i_p,i_d)

x_mesh = act_pd(I_P, I_D, alpha, gain)

ax_pd.pcolormesh(I_P, I_D, x_mesh)

plt.show()
