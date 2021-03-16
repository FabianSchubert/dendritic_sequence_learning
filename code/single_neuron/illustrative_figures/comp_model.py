#! /usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set()

def s(x):
    return 1./(1.+np.exp(-4.*g*x))
    
def phi(x,y):
    return alpha * s(x)*(1.-s(y)) + s(x-thp)*s(y)

plt.style.use("../../mpl_style.mplstyle")

alpha = 0.3
thp = -1.
g=1.

p = np.linspace(-2.,2.,100)
d = np.linspace(-2.,2.,100)

p_ax = np.array(p) - (p[1] - p[0])/2.
p_ax = np.append(p_ax,2.*p_ax[-1] - p_ax[-2])

d_ax = np.array(d) - (d[1] - d[0])/2.
d_ax = np.append(d_ax,2.*d_ax[-1] - d_ax[-2])

P,D = np.meshgrid(p,d)

fig, ax = plt.subplots(1,1,figsize=(5,4))

pm = ax.pcolormesh(p_ax,d_ax,phi(P,D),rasterized=True)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad="13%")

#fig.add_axes(cax)
fig.colorbar(pm,cax=cax)

ax.set_xlabel(r'$I_{\rm p}$')
ax.set_ylabel(r'$I_{\rm d}$')

ax.plot([0.,0.],[-2.,2.],'--',c='w')
ax.plot([-1.,-1.],[-2.,2.],'--',c='w')
ax.plot([-2.,2.],[0.,0.],'--',c='w')

ax.text(1.,-1.2,r'$\alpha$',color='w',fontsize=14)
ax.text(-.15,2.15,r'$\theta_{p0}$',fontsize=14)
ax.text(-1.15,2.15,r'$\theta_{p1}$',fontsize=14)
ax.text(2.15,-0.1,r'$\theta_{d}$',fontsize=14)

fig.tight_layout(pad=0.1)

fig.savefig("../../../notes/figures/plot_comp_mod_marks.pdf",transparent=True)
fig.savefig("../../../notes/figures/plot_comp_mod_marks.png",dpi=600,transparent=True)
