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

if not(os.path.isdir(SAVE_PATH)):
	print("Data folder not found!")
	sys.exit()



x_e = np.load(SAVE_PATH + "recurrent_net_data.npz")["x_e"]
x_i = np.load(SAVE_PATH + "recurrent_net_data.npz")["x_i"]

n_t, N_layers, N_e_layer = x_e.shape
N_i_layer = x_i.shape[2]


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)

plot_start = n_t - 50
plot_length = 50

#t = np.linspace(plot_start, plot_end,plot_end - plot_start)

cols = mpl.rcParams['axes.prop_cycle'].by_key()['color']

l_e = ax.plot(x_e[:,0,:],c=cols[0],alpha=0.2)
l_i = ax.plot(x_i[:,0,:],c=cols[1],alpha=0.2)

ax.set_xlim([plot_start,plot_start+plot_length])

ax.set_xlabel("Time Steps")
ax.set_ylabel("Output Activity")

buttonaxes = []
layerbuttons = []
callbackfuncs = []

def change_layer(event,index):
	#print(event)
		
	for k in range(N_e_layer):
		l_e[k].set_ydata(x_e[:,index,k])
	for k in range(N_i_layer):
		l_i[k].set_ydata(x_i[:,index,k])
	
	
for k in range(N_layers):
	buttonaxes.append(plt.axes([(1.+k)/(1.+N_layers),0.05,0.1,0.075]))
	layerbuttons.append(Button(buttonaxes[k],'Layer '+str(k)))
	callbackfuncs.append(lambda x,i=k: change_layer(x,i))
	layerbuttons[k].on_clicked(callbackfuncs[k])

def tb_submit(text):
	plot_start = int(text)
	ax.set_xlim([plot_start,plot_start+plot_length])


tb_ax = plt.axes([0.5,0.2,0.3,0.075])

plot_start_tb = TextBox(tb_ax, 'Plot Start', initial = str(plot_start))
plot_start_tb.on_submit(tb_submit)


plt.show()