#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import pickle

import pdb

os.chdir("../../../")

def synnorm(w,w_total):
	return w_total*w/np.linalg.norm(w)
	#return w_total*w/w.sum()

def main(n_t_sequ = 500000, n_t = 2000, X_p = np.random.normal(0.,1.,(100000,10)),
	X_d = np.random.normal(0.,1.,(100000,1)),
	w_prox_total = 1.,
	w_dist_total = 1.,
	mu_learn = 0.001,
	mu_mean = 0.00001,
	alpha = 0.2):
	
	if X_p.shape[0] != X_d.shape[0]:
		print("Proximal and distul input arrays do not match on time axis.")
		sys.exit()

	n_prox = X_p.shape[1]
	n_dist = X_d.shape[1]

	X_p_mean = np.zeros((n_prox))#X_p[0,:]
	X_d_mean = np.zeros((n_dist))#X_d[0,:]

	w_prox = np.ones((n_prox))
	w_dist = np.ones((n_dist))

	w_prox = synnorm(w_prox,w_prox_total)
	w_dist = synnorm(w_dist,w_dist_total)

	w_prox_rec = np.ndarray((n_t,n_prox))

	#X_p_mean_rec = np.ndarray((n_t,n_prox))
	#X_d_mean_rec = np.ndarray((n_t,n_dist))

	C_pp_mean = np.zeros((n_prox,n_prox))
	C_pd_mean = np.zeros((n_prox,n_dist))

	for k in tqdm(range(n_t_sequ)):
		C_pp_mean += np.outer(X_p[k,:],X_p[k,:])/n_t
		C_pd_mean += np.outer(X_p[k,:],X_d[k,:])/n_t

	n_sweep = 100
	alpha_sweep = np.linspace(0.,1.,n_sweep)
	MSE = np.ndarray((n_sweep))
	rho_Ipd = np.ndarray((n_sweep))
	for k in tqdm(range(n_sweep)):
		alpha = alpha_sweep[k]

		for t in tqdm(range(n_t)):

			#C_pp = np.outer(X_p[t-1,:] - X_p_mean,X_p[t-1,:] - X_p_mean)
			#C_pd = np.outer(X_p[t-1,:] - X_p_mean,X_d[t-1,:] - X_d_mean)

			w_prox += mu_learn*( alpha*np.dot(C_pp_mean,w_prox) + (2.-alpha)*np.dot(C_pd_mean,w_dist) )  
			w_prox = synnorm(w_prox,w_prox_total)

			#X_p_mean += mu_mean * (X_p[t,:] - X_p_mean)
			#X_d_mean += mu_mean * (X_d[t,:] - X_d_mean)

			###
			#w_prox_rec[t,:] = w_prox

		#X_p_mean_rec[t,:] = X_p_mean
		#X_d_mean_rec[t,:] = X_d_mean
		###

		I_p = np.dot(w_prox,X_p.T)
		I_d = np.dot(w_dist,X_d.T)

		MSE[k] = ((I_p - I_d)**2).mean()
		rho_Ipd[k] = ((I_p - I_p.mean())*(I_d - I_d.mean())/(I_p.std()*I_d.std())).mean()


	plt.ion()
	pdb.set_trace()



if __name__ == "__main__":

	X_p_sequ = np.load("sim_data/single_neuron/rand_chaotic_sequ.npy")[:,:10]
	X_d_sequ = np.array([np.load("sim_data/single_neuron/rand_chaotic_sequ.npy")[:,0]]).T
	#X_d_sequ = np.array([X_p_sequ[:,0] + X_p_sequ[:,1]]).T
	X_p_sequ[:,2] *= 2.

	X_p_sequ -= X_p_sequ.mean(axis=0)
	X_d_sequ -= X_d_sequ.mean(axis=0)

	main(X_p = X_p_sequ, X_d = X_d_sequ)