#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

path_rel = "../../../sim_data/single_neuron/one_distal_mult_prox/"

def sigm(x):

	return (np.tanh(x/2.)+1.)/2.

def main(ϕ_som_e = np.random.rand(1000000,1), ϕ_som_i = np.zeros((1000000,1)), ϕ_dnd = np.random.rand(1000000,1), dt = 0.001):

	n_som_e = ϕ_som_e.shape[1]
	n_som_i = ϕ_som_i.shape[1]
	n_dnd = ϕ_dnd.shape[1]

	if not(ϕ_som_e.shape[0] == ϕ_som_i.shape[0] and ϕ_som_i.shape[0] == ϕ_dnd.shape[0]):
		print("Input arrays do not match on time axis.")
		sys.exit()
	
	n_t = ϕ_som_e.shape[0]

	T = n_t*dt

	τ_L = 10.
	τ_s = 3.

	g_L = 1./τ_L
	g_D = 2.

	g_E = 0.
	g_I = 0.

	E_e = 4.66
	E_i = -1./3.

	ϕ_max = 0.15
	k = 0.5
	β = 5.
	θ = 1.

	U = 0.
	I_dnd = 0.
	V_dnd = 0.

	w_dnd = np.ones((n_dnd))*0.1
	w_som_e = np.ones((n_som_e))*0.1
	w_som_i = np.ones((n_som_i))*0.1

	U_rec = np.ndarray((n_t))
	U_spike_rec = 

	for t in tqdm(range(n_t)):

		I_som = g_E*(E_e - U) + g_I*(E_i - U)

		U += dt*(-g_L*U + g_D*(V_dnd - U) + I_som)

		g_E += dt*(-g_E/τ_s) + np.dot(w_som_e,1.*(np.random.rand(n_som_i) <= dt*ϕ_som_e[t,:]))
		g_I += dt*(-g_I/τ_s) + np.dot(w_som_i,1.*(np.random.rand(n_som_i) <= dt*ϕ_som_i[t,:]))

		V_dnd += dt*(-V_dnd + I_dnd)/τ_L
		I_dnd += dt*(-I_dnd/τ_s) + np.dot(w_dnd,1.*(np.random.rand(n_dnd) <= dt*ϕ_dnd[t,:]))/τ_s

		U_rec[t] = U

	plt.plot(U_rec)
	plt.plot(ϕ_som_e[:,0])
	plt.plot(ϕ_dnd[:,0])
	plt.show()


if __name__ == '__main__':

	#ϕ_som_sequ = np.array([np.load(path_rel + "../rand_chaotic_sequ.npy")[:1000000,0]]).T
	#ϕ_som_sequ = np.array([10.*(np.sin(np.linspace(0.,100.,1000000))+1.)/2.]).T
	ϕ_som_sequ = np.zeros((1000000,1))
	#ϕ_dnd_sequ = np.load(path_rel + "../rand_chaotic_sequ.npy")[:1000000,:10]*0.
	ϕ_dnd_sequ = np.array([10.*(np.sin(np.linspace(0.,100.,1000000))+1.)/2.]).T
	#ϕ_dnd_sequ = np.zeros((1000000,1))
	main(ϕ_som_e = ϕ_som_sequ, ϕ_dnd = ϕ_dnd_sequ)
