#!/usr/bin/env python3

import matplotlib.pyplot as plt

style = {
	# Seaborn paper context
"figure.figsize": (6.4, 4.4),
"axes.labelsize": 12.,
"axes.titlesize": 12.,
#"axes.edgecolor": None,
"axes.spines.top": False,
"axes.spines.right": False,
"xtick.labelsize": 12,
"ytick.labelsize": 12,
"legend.fontsize": 12,
"grid.linewidth": 0.8,
"lines.linewidth": 1.4,
"patch.linewidth": 0.24,
"lines.markersize": 5.6,
"lines.markeredgewidth": 0,
"xtick.major.width": 0.8,
"ytick.major.width": 0.8,
"xtick.minor.width": 0.4,
"ytick.minor.width": 0.4,
"xtick.major.pad": 5.6,
"ytick.major.pad": 5.6,
"font.family": "Arial"
}


plt.style.use(style)

DAT_FORMAT = "pdf"

'''
simfold = "../../../../sim_data/single_neuron/one_distal_mult_prox/"
plotsfold = "../../../../plots/single_neuron/one_distal_mult_prox/"
dat_format = "pdf"
'''
