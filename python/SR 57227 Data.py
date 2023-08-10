import MPNeuro.plotting as MP_plot

feeding_data = np.array([[0.5, 0.6, 0.4, 0.55],[0.29, 0.29, 0.1, 0.29]])
x = np.ones([2, 4]); x[0]=0

fs = 16
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(x, feeding_data, linewidth = 2)
plt.ylim([0, 0.8])
plt.xlim([-0.5, 1.5])
plt.xlabel('Day', fontsize = fs)
plt.ylabel('Food intake (g)', fontsize = fs)
ax.set_xticklabels(['', 'Saline', '', 'SR'])
MP_plot.prettify_axes( ax )



