#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle

from dendritic_sequence_learn.plot_settings import *
from dendritic_sequence_learn.general_settings import SIMFOLD, PLOTSFOLD
SIMFOLD_SINGLE_NEURON = SIMFOLD + "single_neuron/one_distal_mult_prox/"
PLOTSFOLD_SINGLE_NEURON = PLOTSFOLD + "single_neuron/one_distal_mult_prox/"

def sigm(x):
	#return np.tanh(x/2.)
	return (np.tanh(x/2.)+1.)/2.

def act_pd(p,d,alpha,gain):

	return (sigm(gain*d) + alpha*sigm(gain*p)*sigm(-gain*d))*2.-1.

I_p_rec = np.load(SIMFOLD_SINGLE_NEURON + "I_p.npy")
I_d_rec = np.load(SIMFOLD_SINGLE_NEURON + "I_d.npy")

with open(SIMFOLD_SINGLE_NEURON + "params.p","rb") as reader:
	params = pickle.load(reader)

alpha_pd = params['alpha_pd']
gain_pd = params['gain_pd']
n_t_learn = params['n_t_learn']

figdim = (5.,5.)

fig_act_pd_beginning, ax_act_pd_beginning = plt.subplots(1,1,figsize=figdim)
i_p = np.linspace(-1.,1.,400)
i_d = np.linspace(-1.,1.,400)
Ip,Id = np.meshgrid(i_p,i_d)

act_pd_p_beginning = ax_act_pd_beginning.pcolormesh(i_p,i_d,act_pd(Ip,Id,alpha_pd,gain_pd),rasterized=True)

ax_act_pd_beginning.set_xlim([-1.,1.])
ax_act_pd_beginning.set_ylim([-1.,1.])

t_wind = int(n_t_learn*0.02)
ax_act_pd_beginning.plot(I_p_rec[:t_wind],I_d_rec[:t_wind],'.',c=(1.,0.4,0.),alpha=0.5,rasterized=True)
ax_act_pd_beginning.set_xlabel("$I_{p}$")
ax_act_pd_beginning.set_ylabel("$I_{d}$")

ax_act_pd_beginning.set_title("First " + str(int(100.*t_wind/n_t_learn))+"% of learning phase")

plt.tight_layout()

fig_act_pd_beginning.savefig(PLOTSFOLD_SINGLE_NEURON + "act_pd_beginning." + DAT_FORMAT)

plt.show()