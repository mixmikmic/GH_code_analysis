import rebound
print(rebound.__build__)
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    get_ipython().magic('matplotlib inline')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

sa = rebound.SimulationArchive("ias15_seed87_new.bin")
sar2 = rebound.SimulationArchive("ias15_seed87_new_restart.bin")

get_ipython().run_cell_magic('timeit', '-n1 -r1 ', 'fig = plt.figure(figsize=(15, 10)) \nax1 = plt.subplot(211)\ndata = np.zeros((1+sa[0].N,len(sa)))\ndate = np.zeros((1+sa[0].N,len(sa)))\nfor i, sim in enumerate(sa):\n    data[0][i] = sim.t/2/np.pi\n    for j in range(1,sim.N):\n        o = sim.particles[j].calculate_orbit()\n        data[j][i] = o.a\n        date[j][i] = o.e\nax1.set_xlim([0,data[0][-1]])            \nax1.set_ylim([3,2000])   \nax1.set_yscale("log")\nax1.set_xlabel("time [yrs]")\nax1.set_ylabel("semi-major axis [AU]")\nfor j in range(1,sa[0].N):\n    y1 = data[j]*(1.+date[j])\n    y2 = data[j]*(1.-date[j])\n    ax1.fill_between(data[0], y1, y2, where=y2 <= y1, facecolor=\'black\',alpha=0.2)\n    ax1.plot(data[0], data[j], color="black");    \nax1.add_patch(Rectangle((sar2.tmin/2/np.pi, 20), (sar2.tmax-sar2.tmin)/2/np.pi,   40,          \n                       facecolor = "None", edgecolor = "red", linewidth = 2))    \nax1.add_patch(Rectangle((sar2.tmin/2/np.pi, 20), (sar2.tmax-sar2.tmin)/2/np.pi,   40,          \n                       facecolor = "red", alpha = 0.2, linewidth = 0))    \nax2 = plt.subplot(212)\ndata = np.zeros((1+sar2[0].N,len(sar2)))\ndate = np.zeros((1+sar2[0].N,len(sar2)))\nfor i, sim in enumerate(sar2):\n    data[0][i] = sim.t/2/np.pi\n    for j in range(1,sim.N):\n        o = sim.particles[j].calculate_orbit(primary=sim.particles[0])\n        data[j][i] = o.a\n        date[j][i] = o.e\nax2.set_xlim([data[0][0],data[0][-1]])            \nax2.set_ylim([20,60]) \n#ax2.set_yscale("log")\nax2.set_xlabel("time [yrs]")\nax2.set_ylabel("semi-major axis [AU]")\nfor j in range(1,sar2[0].N):\n    y1 = data[j]*(1.+date[j])\n    y2 = data[j]*(1.-date[j])\n    ax2.fill_between(data[0], y1, y2, where=y2 <= y1, facecolor=\'black\',alpha=0.2)\n    ax2.plot(data[0], data[j], color="black");    \n\ntransFigure = fig.transFigure.inverted()\n\ncoord1 = transFigure.transform(ax1.transData.transform([sar2.tmin/2/np.pi, 20]))\ncoord2 = transFigure.transform(ax2.transData.transform([sar2.tmin/2/np.pi, 60]))\nline = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),\n                               transform=fig.transFigure, color="red")\nfig.lines.append(line)\n\n\ncoord1 = transFigure.transform(ax1.transData.transform([sar2.tmax/2/np.pi, 20]))\ncoord2 = transFigure.transform(ax2.transData.transform([sar2.tmax/2/np.pi, 60]))\nline = Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]),\n                               transform=fig.transFigure, color="red")\n\n\nfig.lines.append(line)\nplt.savefig("closeencounter.pdf", format=\'pdf\', bbox_inches=\'tight\', pad_inches=0)')



