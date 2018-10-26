#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys

from dendritic_sequence_learn.general_settings import ROOT, SIMFOLD
SAVE_PATH = SIMFOLD + "recurrent_net/"

if not(os.path.isdir(SAVE_PATH)):
	os.mkdir(SAVE_PATH)

import pdb

def sigm(x):
	return ( np.tanh( x/2. ) + 1. ) / 2.

def act_pd(p,d,alpha,gain):

	return sigm(gain*d) + alpha*sigm(gain*p)*sigm(-gain*d)


def target_signal(t):
	#return np.sin(0.05 * t*2.*np.pi)**2/2.
	return 0.

def main(	N_layers = 2,
			N_e_layer = 200,
			N_i_layer = 100,
			N_neurons_sens = 10,
			N_neurons_top_down = 10,
			mean_w_forward  = 1.,
			mean_w_backward = 1.,
			mean_w_ie_layer = 1.,
			mean_w_ei_layer = -1.,
			cf_forward = 0.02,
			cf_backward = 0.02,
			cf_ie_layer = 0.05,
			cf_ei_layer = 0.05,
			mean_w_sens_forward = 0.5,
			mean_w_top_down_backward = 0.5,
			inp_target_prox = 0.,
			inp_target_dist = 0.,
			inp_target_i = 0.,
			act_var_target = 0.1**2,
			alpha = 0.25,
			gain = 60.,
			mu_learn = 0.0001,
			mu_thresholds = 0.001,
			mu_act_filter = 0.001,
			mu_gain = 0.01,
			n_t_skip_rec = 1,
			n_t = 100000	):
	
	if n_t%n_t_skip_rec != 0:
		print("Total number of simulation steps is not a multiple of the recording interval!")
		sys.exit()
	
	### Init Gain
	gain_arr = np.ones((N_layers, N_e_layer)) * gain
	###
	


	### Init Weights
	W_f = np.random.rand(N_layers, N_e_layer, N_e_layer) * mean_w_forward
	W_f *= (np.random.rand(N_layers, N_e_layer, N_e_layer) <= cf_forward ) / ( cf_forward * N_e_layer )
		
	W_b = np.random.rand(N_layers, N_e_layer, N_e_layer) * mean_w_backward
	W_b *= ( np.random.rand(N_layers, N_e_layer, N_e_layer) <= cf_backward ) / ( cf_backward * N_e_layer )
		
	W_ie = np.random.rand(N_layers, N_i_layer, N_e_layer) * mean_w_ie_layer
	W_ie *= ( np.random.rand(N_layers, N_i_layer, N_e_layer) <= cf_ie_layer ) / ( cf_ie_layer * N_e_layer )
		
	W_ei = np.random.rand(N_layers, N_e_layer, N_i_layer) * mean_w_ei_layer
	W_ei *= ( np.random.rand(N_layers, N_e_layer, N_i_layer) <= cf_ei_layer ) / ( cf_ei_layer * N_i_layer )
		
	W_f_sens = np.random.rand(N_e_layer, N_neurons_sens) * mean_w_sens_forward

	W_b_top_down = np.random.rand(N_e_layer, N_neurons_top_down) * mean_w_top_down_backward
	###

	### Init Thresholds
	th_e_p = np.zeros((N_layers, N_e_layer))
	th_e_d = np.zeros((N_layers, N_e_layer))
	
	th_i = np.zeros((N_layers, N_i_layer))
	###

	### Init Output Rate Variables
	x_e = np.zeros((N_layers, N_e_layer))
	x_i = np.zeros((N_layers, N_i_layer))
	x_sens = np.zeros((N_neurons_sens))
	x_top_down = np.zeros((N_neurons_top_down))
	###

	### Init Input Currents
	
	## proximal and distal to excitatory neurons
	I_p = np.zeros((N_layers, N_e_layer))
	I_d = np.zeros((N_layers, N_e_layer))
	
	# proximal consists of:
	# forward connections
	I_f = np.zeros((N_layers, N_e_layer))
	# inhibition from inh. interneurons
	I_ei = np.zeros((N_layers, N_e_layer))
	# feed forward "sensory", external
	I_sens = np.zeros((N_layers, N_e_layer))
	
	# distal consists of:
	# backward connections
	I_b = np.zeros((N_layers, N_e_layer))
	# "top down", external
	I_top_down = np.zeros((N_layers, N_e_layer))
	
	# input from exc. to inh. interneurons
	I_i = np.zeros((N_layers, N_i_layer))
	# consists of excitatory input from neurons in layer
	I_ie = np.zeros((N_layers, N_i_layer))
	###

		
	'''
	### Init Filtered Standard Deviation of Activity
	x_e_avg = np.zeros((N_layers, N_e_layer))
	x_var_avg = np.zeros((N_layers, N_e_layer))	 
	###
	'''
	
	### Init Recorders
	
	n_t_rec = int(n_t/n_t_skip_rec)
	
	x_e_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	x_i_rec = np.ndarray((n_t_rec, N_layers, N_i_layer))
	
	x_top_down_rec = np.ndarray((n_t_rec, N_neurons_top_down))
	x_sens_rec = np.ndarray((n_t_rec, N_neurons_sens))
	
	I_f_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	I_b_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	I_ei_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	
	I_ie_rec = np.ndarray((n_t_rec, N_layers, N_i_layer))
	
	I_sens_rec = np.ndarray((n_t_rec, N_e_layer))
	I_top_down_rec = np.ndarray((n_t_rec, N_e_layer))
	
	th_e_p_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	th_e_d_rec = np.ndarray((n_t_rec, N_layers, N_e_layer))
	
	th_i_rec = np.ndarray((n_t_rec, N_layers, N_i_layer))
	
	#x_e_avg_rec = np.ndarray((n_t, N_layers, N_e_layer))

	#gain_rec = np.ndarray((n_t, N_layers, N_e_layer))	
	###



	for t in tqdm(range(n_t)):
		
		
		### Generate Top Down Input
		#x_top_down = 1.*(np.random.rand(N_neurons_top_down) <= target_signal(t))
		x_top_down = target_signal(t) * np.ones((N_neurons_top_down))
		###
		
		### Calculate Inputs
		I_sens = np.dot(W_f_sens, x_sens)
		I_top_down = np.dot(W_b_top_down, x_top_down)
		
		for k in range(N_layers):
			I_ie[k,:] = np.dot(W_ie[k,:,:], x_e[k,:])
			I_ei[k,:] = np.dot(W_ei[k,:,:], x_i[k,:])
		
		I_f[0,:] = I_sens
		I_b[-1,:] = I_top_down
		
		for k in range(1, N_layers):
			I_f[k,:] = np.dot(W_f[k,:,:], x_e[k-1,:])
		for k in range(0, N_layers-1):
			I_b[k,:] = np.dot(W_b[k,:,:], x_e[k+1,:])
			
		I_p = I_f + I_ei
		I_d = I_b
		
		I_i = I_ie
		###
		
		
		### Filter Inputs
		#I_avg_p += mu_input_filter * ( I_p - I_avg_p )
		#I_avg_d += mu_input_filter * ( I_d - I_avg_d )
		###
		
		### Update Activities
		#prob_x = act_pd( I_p - I_avg_p, I_d - I_avg_d, alpha, gain_arr )
		#x_e = 1.*(np.random.rand(N_layers, N_e_layer) <= prob_x)
		x_e = act_pd( I_p - th_e_p, I_d - th_e_d, alpha, gain_arr )
		x_i = sigm( (I_i - th_i) * gain )
		###
		
		
		### Update Thresholds
		#if t <= 0.8 * n_t:
		th_e_p += mu_thresholds * ( (I_p - th_e_p) - inp_target_prox )
		th_e_d += mu_thresholds * ( (I_d - th_e_d) - inp_target_dist )
		
		th_i += mu_thresholds * ( (I_i - th_i) - inp_target_i )
		###
		
		
		### Filter Activity
		#x_e_avg += mu_act_filter * ( x_e - x_e_avg )
		###
		
		### Update Gain
		#x_e_var = ( x_e - x_e_avg )**2
		#gain_arr += mu_gain * ( act_var_target - x_e_var )
		###
		
		
		
		
		### Record
		
		if t%n_t_skip_rec == 0:
			
			t_rec = int(t/n_t_skip_rec)
			
			x_e_rec[t_rec,:,:] = x_e
			x_i_rec[t_rec,:,:] = x_i
			
			x_top_down_rec[t_rec,:] = x_top_down
			x_sens_rec[t_rec,:] = x_sens
			
			I_f_rec[t_rec,:,:] = I_f
			I_b_rec[t_rec,:,:] = I_b
			I_ei_rec[t_rec,:,:] = I_ei
			
			I_ie_rec[t_rec,:,:] = I_ie
			
			
				
			#I_avg_p_rec[t,:,:] = I_avg_p
			#I_avg_d_rec[t,:,:] = I_avg_d
			
			I_sens_rec[t_rec,:] = I_sens
			I_top_down_rec[t_rec,:] = I_top_down
			
			th_e_p_rec[t_rec,:,:] = th_e_p
			th_e_d_rec[t_rec,:,:] = th_e_d

			th_i_rec[t_rec,:,:] = th_i
			
			#x_e_avg_rec[t,:,:] = x_e_avg
			
			#gain_rec[t,:,:] = gain_arr
		
		###

	if __name__ == "__main__":
		'''
		np.save(SAVE_PATH + "x_e.npy", x_rec)
		np.save(SAVE_PATH + "x_top_down.npy", x_top_down_rec)

		np.save(SAVE_PATH + "x_var_avg.npy", x_var_avg_rec)
		np.save(SAVE_PATH + "gain_arr.npy", gain_rec)

		np.save(SAVE_PATH + "I_p.npy", I_p_rec)
		np.save(SAVE_PATH + "I_d.npy", I_d_rec)

		np.save(SAVE_PATH + "I_sens.npy", I_sens_rec)
		np.save(SAVE_PATH + "I_top_down.npy", I_top_down_rec)
		'''

		#pdb.set_trace()
		
		np.savez_compressed( SAVE_PATH + "recurrent_net_data.npz",
			x_e = x_e_rec,
			x_i = x_i_rec,
			x_top_down = x_top_down_rec,
			x_sens = x_sens_rec,
			I_f = I_f_rec,
			I_b = I_b_rec,
			I_ei = I_ei_rec,
			I_ie = I_ie_rec,
			I_sens = I_sens_rec,
			I_top_down = I_top_down_rec,
			th_e_p = th_e_p_rec,
			th_e_d = th_e_d_rec,
			th_i = th_i_rec	)
		
		#plt.ion()
		#pdb.set_trace()
	
if __name__ == '__main__':
	
	main() 


