#! /usr/bin/env python3

import numpy as np
from os import listdir
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

mpl.rcParams['font.family'] = 'Fira Sans'

files_comp = listdir("./data/compartment_model/")
files_point = listdir("./data/point_model/")

dat_comp = []
dat_point = []

perf_comp = []
perf_point = []

for file in files_comp:
    dat_comp.append(np.load("./data/compartment_model/"
                    +file))
    perf_comp.append(dat_comp[-1]["perf"])

perf_comp = np.array(perf_comp)

perf_comp_reshape = np.ndarray((perf_comp.shape[0]*perf_comp.shape[3],
                                        perf_comp.shape[1],
                                        perf_comp.shape[2]))
for k in range(perf_comp.shape[0]):
    for l in range(perf_comp.shape[3]):
        perf_comp_reshape[k*perf_comp.shape[3]+l,:] = perf_comp[k,:,:,l]

perf_comp = np.array(perf_comp_reshape)


for file in files_point:
    dat_point.append(np.load("./data/point_model/"
                    +file))
    perf_point.append(dat_point[-1]["perf"])

perf_point = np.array(perf_point)

perf_point_reshape = np.ndarray((perf_point.shape[0]*perf_point.shape[3],
                                        perf_point.shape[1],
                                        perf_point.shape[2]))
for k in range(perf_point.shape[0]):
    for l in range(perf_point.shape[3]):
        perf_point_reshape[k*perf_point.shape[3]+l,:] = perf_point[k,:,:,l]

perf_point = np.array(perf_point_reshape)

nDistSweepComp = dat_comp[0]["nDistSweep"]
scaleDistSweepComp = dat_comp[0]["scaleDistSweep"]

nDistSweepCompMesh = np.array(nDistSweepComp) - 0.5*(nDistSweepComp[1]-nDistSweepComp[0])
nDistSweepCompMesh = np.append(nDistSweepCompMesh,2.*nDistSweepCompMesh[-1] - nDistSweepCompMesh[-2])

scaleDistSweepCompMesh = np.array(scaleDistSweepComp) - 0.5*(scaleDistSweepComp[1]-scaleDistSweepComp[0])
scaleDistSweepCompMesh = np.append(scaleDistSweepCompMesh,2.*scaleDistSweepCompMesh[-1] - scaleDistSweepCompMesh[-2])

nDistSweepPoint = dat_point[0]["nDistSweep"]
scaleDistSweepPoint = dat_point[0]["scaleDistSweep"]

nDistSweepPointMesh = np.array(nDistSweepPoint) - 0.5*(nDistSweepPoint[1]-nDistSweepPoint[0])
nDistSweepPointMesh = np.append(nDistSweepPointMesh,2.*nDistSweepPointMesh[-1] - nDistSweepPointMesh[-2])

scaleDistSweepPointMesh = np.array(scaleDistSweepPoint) - 0.5*(scaleDistSweepPoint[1]-scaleDistSweepPoint[0])
scaleDistSweepPointMesh = np.append(scaleDistSweepPointMesh,2.*scaleDistSweepPointMesh[-1] - scaleDistSweepPointMesh[-2])


perf_comp_mean = perf_comp.mean(axis=0)
perf_point_mean = perf_point.mean(axis=0)


fig, ax = plt.subplots(1,2,figsize=(8,3))

pmeshcomp = ax[0].pcolormesh(scaleDistSweepCompMesh,nDistSweepCompMesh,perf_comp_mean)

pmeshpoint = ax[1].pcolormesh(scaleDistSweepPointMesh,nDistSweepPointMesh,perf_point_mean)

ax[0].set_xlabel(r'$\sigma_{\rm distr}$')
ax[1].set_xlabel(r'$\sigma_{\rm distr}$')

ax[0].set_ylabel(r'$N_{\rm distr}$')
ax[1].set_ylabel(r'$N_{\rm distr}$')

ax[0].set_title("Class. Accuracy - Comp. Model",loc="left")
ax[1].set_title("Class. Accuracy - Point Model",loc="left")

plt.colorbar(pmeshcomp,ax=ax[0])
plt.colorbar(pmeshpoint,ax=ax[1])

fig.tight_layout(pad=0.1)

plt.show()
