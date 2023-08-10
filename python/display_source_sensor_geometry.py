from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

sens_meg = sensors.SensorsMEG(load_default=True)
conn = connectivity.Connectivity(load_default=True)
skin = surfaces.SkinAir(load_default=True)
skin.configure()
sens_eeg = sensors.SensorsEEG(load_default=True)
sens_eeg.configure()

figure()
ax = subplot(111, projection='3d')

# ROI centers as black circles
x, y, z = conn.centres.T
ax.plot(x, y, z, 'ko')

# EEG sensors as blue x's
x, y, z = sens_eeg.sensors_to_surface(skin).T
ax.plot(x, y, z, 'bx')

# Plot boundary surface
x, y, z = skin.vertices.T
ax.plot_trisurf(x, y, z, triangles=skin.triangles, alpha=0.1, edgecolor='none')

# MEG sensors as red +'s
x, y, z = sens_meg.locations.T
ax.plot(x, y, z, 'r+')

