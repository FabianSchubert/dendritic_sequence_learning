#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pickle

import pdb


path_rel = "../../../../sim_data/single_neuron/one_distal_mult_prox/"

def sigm(x):
	#return np.tanh(x/2.)
	return (np.tanh(x/2.)+1.)/2.


def act_pd(p,d,alpha,gain):

	return (sigm(gain*d) + alpha*sigm(gain*p)*sigm(-gain*d))*2.-1.


def gen_rand_sequ(N,T,dt,n_out,g=2.):

	n_t = int(1.2*T/dt)

	W = np.random.normal(0.,g/N**.5,(N,N))

	x = np.random.normal(0.,.5,(N))

	x_rec = np.ndarray((n_t,n_out))

	for t in tqdm(range(n_t)):

		x += dt*(-x + np.tanh(np.dot(W,x)))

		x_rec[t,:] = x[:n_out]

	return x_rec[int(0.2/1.2*n_t):,:]

def synnorm(w,w_total):
	#return w_total * w/w.sum()
	return w_total * w/np.linalg.norm(w)



def main(n_t_learn = 500000, X_p = np.random.normal(0.,1.,(1000000,10)), X_d = np.random.normal(0.,1.,(1000000,1)), alpha_pd = 0.25, gain_pd = 20., gain_d_sign_inv = 1., w_dist = 1., w_prox_total = 1., w_prox_max = 1., w_prox_min = 0.0001, w_dist_total = 1., w_dist_max = 1., w_dist_min = 0.0001, mu_learn = 0.00005, mu_hom = 0.00002,  mu_avg = 0.00002):
	
	n_prox = X_p.shape[1]
	n_dist = X_d.shape[1]

	# initialize proximal weights
	if n_prox > 1:
		w_prox = np.ones(n_prox)
		w_prox = w_prox_total * w_prox/w_prox.sum()
	else:
		w_prox = 1

	# initialize distal weights
	if n_dist > 1:
		w_dist = np.ones(n_dist)
		w_dist = w_dist_total * w_dist/w_dist.sum()
	else:
		w_dist = 1

	# initialize weights for "analytic" time evolution, by covariances
	w_prox_analytic = np.ones(n_prox)
	w_prox_analytic = w_prox_total * w_prox_analytic/w_prox_analytic.sum()

	gamma = (1.-alpha_pd/2.)*alpha_pd*gain_pd/4.

	'''
	# Calc covariance matrices
	C_xx = np.cov(X_p.T)
	C_xd = np.cov(X_p.T,X_d)[-1,:-1]

	C_xxd = np.ndarray((n_prox,n_prox))

	for i in range(n_prox):
		for j in range(n_prox):
			
			C_xxd[i,j] = ((X_d - X_d.mean())*(X_p[:,i] - X_p[:,i].mean())*(X_p[:,j] - X_p[:,j].mean())).mean()

	'''

	th_p = 0.
	th_d = 0.

	# initialize running averages
	x_mean = 0.5
	X_p_mean = X_p[0,:]
	
	X_d_mean = X_d[0,:]
	

	# initialize recordings
	x_rec = np.ndarray((n_t_learn))
	x_mean_rec = np.ndarray((n_t_learn))

	w_prox_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_rec = np.ndarray((n_t_learn,n_prox))

	'''
	w_prox_analytic_cxx_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_cxd_rec = np.ndarray((n_t_learn,n_prox))
	w_prox_analytic_cxxd_rec = np.ndarray((n_t_learn,n_prox))
	'''

	w_dist_rec = np.ndarray((n_t_learn,n_dist))

	X_p_mean_rec = np.ndarray((n_t_learn,n_prox))
	X_d_mean_rec = np.ndarray((n_t_learn,n_dist))	

	I_p_rec = np.ndarray((n_t_learn))
	I_d_rec = np.ndarray((n_t_learn))

	th_p_rec = np.ndarray((n_t_learn))
	th_d_rec = np.ndarray((n_t_learn))

	for t in tqdm(range(n_t_learn)):

		I_p = np.dot(w_prox,X_p[t,:]) - th_p
		
		I_d = np.dot(w_dist,X_d[t,:]) - th_d
		
		
		th_p += mu_hom*I_p
		th_d += mu_hom*I_d
		
		x = act_pd(I_p,I_d,alpha_pd,gain_pd)

		x_mean += mu_avg*(x - x_mean)
		
		X_p_mean += mu_avg*(X_p[t,:] - X_p_mean)
		
		X_d_mean += mu_avg*(X_d[t,:] - X_d_mean)
		


		## plasticity
		
		w_prox += mu_learn * (x-x_mean)*(X_p[t,:]-X_p_mean)
		w_prox = synnorm(w_prox,w_prox_total)
		
		#w_prox = np.maximum(w_prox_min,w_prox)
		#w_prox = np.minimum(w_prox_max,w_prox)
		#w_prox = w_prox_total * w_prox/w_prox.sum()
		
		w_dist += mu_learn * (x-x_mean)*(X_d[t,:]-X_d_mean)
		w_dist = synnorm(w_dist,w_dist_total)
		
		#w_dist = np.maximum(w_dist_min,w_dist)
		#w_dist = np.minimum(w_dist_max,w_dist)
		#w_dist = w_dist_total * w_dist/w_dist.sum()
		#w_dist = synnorm(w_dist,w_dist_total)
		##


		'''
		## plasticity-analytic
		w_prox_analytic += mu_learn * (alpha_pd*np.dot(C_xx,w_prox_analytic) + (2.-alpha_pd)*C_xd + gamma*np.dot(C_xxd,w_prox_analytic)) * gain_pd/8.

		#w_prox_analytic += mu_learn * (alpha_pd*np.dot(C_xx,w_prox_analytic) + (2.-alpha_pd)*C_xd) * gain_pd/8.

		w_prox_analytic_cxx_rec[t,:] = alpha_pd*np.dot(C_xx,w_prox_analytic)
		w_prox_analytic_cxd_rec[t,:] = (2.-alpha_pd)*C_xd
		w_prox_analytic_cxd_rec[t,:] = gamma*np.dot(C_xxd,w_prox_analytic)

		w_prox_analytic = np.maximum(w_prox_min,w_prox_analytic)
		w_prox_analytic = np.minimum(w_prox_max,w_prox_analytic)
		w_prox_analytic = w_prox_total * w_prox_analytic/w_prox_analytic.sum()
		##
		'''

		x_rec[t] = x
		x_mean_rec[t] = x_mean

		w_prox_rec[t,:] = w_prox
		w_prox_analytic_rec[t,:] = w_prox_analytic

		w_dist_rec[t,:] = w_dist

		X_p_mean_rec[t,:] = X_p_mean
		X_d_mean_rec[t,:] = X_d_mean
		
		I_p_rec[t] = I_p
		I_d_rec[t] = I_d

		th_p_rec[t] = th_p
		th_d_rec[t] = th_d

	if __name__ == "__main__":
		
		np.save(path_rel + "X_p.npy",X_p)
		np.save(path_rel + "X_d.npy",X_d)

		np.save(path_rel + "X_p_mean.npy",X_p_mean_rec)
		np.save(path_rel + "X_d_mean.npy",X_d_mean_rec)

		np.save(path_rel + "x.npy",x_rec)
		np.save(path_rel + "x_mean.npy",x_mean_rec)

		np.save(path_rel + "w_prox.npy",w_prox_rec)
		np.save(path_rel + "w_dist.npy",w_dist_rec)

		np.save(path_rel + "I_p.npy",I_p_rec)
		np.save(path_rel + "I_d.npy",I_d_rec)

		np.save(path_rel + "th_p.npy",th_p_rec)
		np.save(path_rel + "th_d.npy",th_d_rec)

		params = {	'n_t_learn':n_t_learn,
					'alpha_pd':alpha_pd,
					'gain_pd':gain_pd,
					'gain_d_sign_inv':gain_d_sign_inv,
					'w_prox_total':w_prox_total,
					'w_dist_total':w_dist_total,
					'mu_learn':mu_learn,
					'mu_hom':mu_hom,
					'mu_avg':mu_avg}
		with open(path_rel + "params.p","wb") as writer:
			pickle.dump(params,writer)

		pdb.set_trace()


if __name__ == "__main__":
	
	'''
	n_rand_sequ = 11
	X_rand_sequ = np.ndarray((2000000,n_rand_sequ))
	for k in tqdm(range(n_rand_sequ)):

		X_rand_sequ[:,k] = gen_rand_sequ(500,2000000*0.1,0.1,1,2.)[:,0]

	
	np.save(path_rel + "rand_chaotic_sequ.npy",X_rand_sequ)
	'''	

	X_p_sequ = np.load(path_rel + "../rand_chaotic_sequ.npy")[:,:10]

	#X_d_sequ = np.ndarray((2000000,10))

	X_d_sequ = np.array([np.load(path_rel + "../rand_chaotic_sequ.npy")[:,0]]).T

	#X_d_sequ[:,:2] *= 2.
	#X_d_sequ[:,1] = X_d_sequ[:,0]*0.9 +  X_d_sequ[:,1]*0.1

	X_p_sequ[:,1] *= 3.

	X_p_sequ -= X_p_sequ.mean(axis=0)
	X_d_sequ -= X_d_sequ.mean(axis=0)
		
	main(X_p = X_p_sequ, X_d = X_d_sequ)







	



