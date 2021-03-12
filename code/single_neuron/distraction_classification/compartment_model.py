#! /usr/bin/env python3

import numpy as np
from tqdm import tqdm
from datetime import datetime

def phi(x):
    return 1./(1.+np.exp(-4.*x))
    
def psi(p,d):
    return alpha*phi(g*p)*(1.-phi(g*d)) + beta*phi(g*d)

alpha = 0.3
beta = 1.
g = 1.

pAvTarg = 0.
pVarTarg = 0.1

dAvTarg = 0.
dVarTarg = 0.1

eps_w = 5e-5
eps_b = 1e-3
eps_norm = 1e-4
eps_av = 5e-3

nIn = 10
nOut = 2

nSweepN = 9
nDistSweep = np.arange(1,nSweepN+1)
nSweepScale = 20#nPatt.shape[0]
scaleDistSweep = np.linspace(0.,20.,nSweepScale)

stdMainDir = .25
distMainDir = 2.

nSamples = 1
nTrain = int(2e5)
nTest = int(1e3)

thP = np.ones((nTrain,nOut)) * 0.5
thD = np.ones((nTrain,nOut)) * 0.5

normP = np.ones((nTrain,nOut)) * 1.
normD = np.ones((nTrain,nOut)) * 1.

wP = np.ones((nTrain,nOut,nIn)) / nIn**.5

p = np.zeros((nTrain,nOut))
pAv = np.zeros((nTrain,nOut))

d = np.zeros((nTrain,nOut))
dAv = np.zeros((nTrain,nOut))

xP = np.zeros((nTrain,nIn))
xPav = np.zeros((nTrain,nIn))

y = np.zeros((nTrain,nOut))
ysquAv = np.zeros((nTrain,nOut))

pTest = np.zeros((nTest,nOut))
yTest = np.zeros((nTest,nOut))
labTest = np.zeros((nTest))

perf = np.ndarray((nSweepN,nSweepScale,nSamples))

for i in tqdm(range(nSweepN)):
    for j in tqdm(range(nSweepScale), leave=False):
        for s in tqdm(range(nSamples), leave=False):

            wP = np.ones((nTrain,nOut,nIn)) / nIn**.5

            linSep = np.random.normal(0.,1.,(nIn))
            linSep /= np.linalg.norm(linSep)
            offsSep = np.ones((nIn)) * 0.

            vDist = np.random.normal(0.,1.,(nIn,nDistSweep[i]))

            orth = np.ndarray((nIn,1+nDistSweep[i]))
            orth[:,0] = linSep
            orth[:,1:] = vDist

            q,r = np.linalg.qr(orth)
            vDist = q[:,1:]

            #vDist = vDist - linSep * np.dot(vDist,linSep)
            #vDist /= np.linalg.norm(vDist)

            #patterns, labels = genPattLab(nPatt[i])

            #pattInd = np.random.randint(nPatt[i])

            patt = np.random.normal((2.*(np.random.rand() < .5)-1.)*distMainDir/2.,
                                     stdMainDir) * linSep
            for k in range(nDistSweep[i]):
                patt += vDist[:,k] * scaleDistSweep[j] * np.random.normal(0.,1.)
            patt += offsSep

            lab = 1. * (np.dot(linSep,patt-offsSep) > 0.)

            xP[0] = patt#patterns[pattInd]
            xPav[0] = xP[0]

            p[0] = (wP[0] @ xP[0]) - thP[0]
            pAv[0] = p[0]

            d[0,0] = (1.-lab)
            d[0,1] = lab
            d[0] = d[0] * normD[0] - thD[0]

            dAv[0] = d[0]

            y[0] = psi(p[0],d[0])

            for k in tqdm(range(1,nTrain), disable=False, leave=False):

                #patterns, labels = genPattLab(nPatt[i])

                #pattInd = np.random.randint(nPatt[i])

                patt = np.random.normal((2.*(np.random.rand() < .5)-1.)*distMainDir/2.,
                                     stdMainDir) * linSep
                for l in range(nDistSweep[i]):
                    patt += vDist[:,l] * scaleDistSweep[j] * np.random.normal(0.,1.)
                patt += offsSep

                lab = 1. * (np.dot(linSep,patt-offsSep) > 0.)

                #pattInd = np.random.randint(nPatt[i])

                xP[k] = patt#patterns[pattInd]
                xPav[k] = xPav[k-1] + eps_av*(-xPav[k-1] + xP[k-1])

                p[k] = (wP[k-1] @ xP[k]) - thP[k-1]
                pAv[k] = pAv[k-1] + eps_av*(-pAv[k-1] + p[k-1])

                d[k,0] = (1.-lab)
                d[k,1] = lab
                d[k] = d[k] * normD[k-1] - thD[k-1]

                dAv[k] = dAv[k-1] + eps_av*(-dAv[k-1] + d[k-1])

                y[k] = psi(p[k],d[k])

                wP[k] = wP[k-1] + eps_w * np.outer(y[k-1]*np.tanh(y[k-1] - (alpha + beta)/2.),
                                                   xP[k-1] - xPav[k-1])
                wP[k] = (wP[k].T * normP[k-1] / np.linalg.norm(wP[k],axis=1)).T

                thP[k] = thP[k-1] + eps_b * (p[k-1] - pAvTarg)
                thD[k] = thD[k-1] + eps_b * (d[k-1] - dAvTarg)

                normP[k] = normP[k-1] + eps_norm * normP[k-1] * (pVarTarg - (p[k-1] - pAv[k-1])**2.)
                normD[k] = normD[k-1] + eps_norm * normD[k-1] * (dVarTarg - (d[k-1] - dAv[k-1])**2.)


            for k in range(nTest):

                patt = np.random.normal((2.*(np.random.rand() < .5)-1.)*distMainDir/2.,
                                     stdMainDir) * linSep
                for l in range(nDistSweep[i]):
                    patt += vDist[:,l] * scaleDistSweep[j] * np.random.normal(0.,1.)
                patt += offsSep

                labTest[k] = 1. * (np.dot(linSep,patt-offsSep) > 0.)

                #pattInd = np.random.randint(nPatt[i])

                #labTest[k] = labels[pattInd]

                pTest[k] = (wP[-1] @ patt) - thP[-1]
                yTest[k] = psi(pTest[k],-thD[-1])

            pred = np.argmax(yTest,axis=1)
            perf[i,j,s] = (1.*(pred == labTest)).mean()

np.savez("./data/compartment_model/class_perf_comp_ndist_scale_"
        +datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        +".npz",
        perf=perf,
        nDistSweep=nDistSweep,
        scaleDistSweep=scaleDistSweep)
