# -*- coding: UTF-8 -*-
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob 
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

data_names = sorted(glob.glob(os.path.join('data/', '*.npz')))

for data_path in data_names:
	print(data_path)
	data_1 = np.load(data_path, allow_pickle=True)

	data_1 = data_1.f.arr_0
	data_1 = np.round(data_1, 2)

	for i in range(data_1.shape[0]):
		if i > 0:
			break

		out_name = data_path.split('/')[-1].split('.')[0]

		fig_size = 1.0
		fig = plt.figure(figsize=(12.8 / fig_size, 7.2 / fig_size)) 

		gs = gridspec.GridSpec(2, 4) 
		gs.update(wspace=0.16, hspace=-0.05) 

		for j in range(data_1.shape[1]):
			data = data_1[i, j, :,:]

			ax = plt.subplot(gs[j])
			plt.xticks(fontsize = 9)
			plt.yticks(fontsize = 9)

			index = j + 1
			plt.title('Head ' + str(index), fontsize = 12)
			sns.heatmap(data, vmax = 0.08, vmin = -0.0001, cmap='Blues_r', cbar = False, square=True)

			ticks_number = 30
			ax.set_xticks(range(0, 243, ticks_number))
			ax.set_xticklabels(range(0, 243, ticks_number), rotation=0)

			ax.set_yticks(range(0, 243, ticks_number))
			ax.set_yticklabels(range(0, 243, ticks_number), rotation=0)


		plt.savefig('attention/' + out_name + '.png', bbox_inches = 'tight')



