#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb


def main(	N_e = 300,
			N_i = 60,
			cf_ee = 0.1,
			cf_ie = 0.1,
			cf_ei = 0.1,
			cf_ii = 0.1,
			g = 0.8):
	
	N = N_e + N_i

	W_ee = (np.random.rand(N_e,N_e) <= cf_ee)
	W_ee[range(N_e),range(N_e)] = 0.

	#W_ee = 

	W_ie = (np.random.rand(N_i,N_e) <= cf_ie)
	
	W_ei = -1.*(np.random.rand(N_e,N_i) <= cf_ei)

	W_ii = -1.*(np.random.rand(N_i,N_i) <= cf_ii)
	W_ii[range(N_i),range(N_i)] = 0.

	W = np.ndarray((N_e+N_i,N_e+N_i))
	
	W[:N_e,:N_e] = W_ee
	W[N_e:,:N_e] = W_ie
	W[:N_e,N_e:] = W_ei
	W[N_e:,N_e:] = W_ii

	l = np.linalg.eigvals(W)

	plt.plot(np.real(l),np.imag(l),'.')

	plt.show()

	#pdb.set_trace()


if __name__ == '__main__':
	
	main() 


