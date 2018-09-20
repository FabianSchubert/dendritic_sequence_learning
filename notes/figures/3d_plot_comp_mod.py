#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def act_pos(x):

	return (np.tanh(2.*x)+1.)/2.

def act_pd(p,d,gain,theta_p0,theta_p1,theta_d,alpha):

	return act_pos(gain*(p-theta_p1))*act_pos(gain*(d-theta_d)) + alpha*act_pos(gain*(p-theta_p0))*act_pos(-gain*(d-theta_d))

i_p = np.linspace(0.,1.,200)
i_d = np.linspace(0.,1.,200)
I_p, I_d = np.meshgrid(i_p,i_d)


alpha_pd = 0.25

gain_pd = 5.

th_p0 = 0.5
th_p1 = 0.1*0.5
th_d = 0.5

fig,ax = plt.subplots(figsize=(6.*0.7,5*0.7))

pcm = ax.pcolormesh(i_p,i_d,act_pd(I_p,I_d,gain_pd,th_p0,th_p1,th_d,alpha_pd))

ax.set_xlabel("$I_p$")
ax.set_ylabel("$I_d$")

plt.colorbar(pcm)

plt.tight_layout()

plt.savefig("plot_comp_mod.png",dpi=200)

plt.show()