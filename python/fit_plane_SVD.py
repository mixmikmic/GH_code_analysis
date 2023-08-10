import os, sys
sys.path.append(r"..")
import importlib
import numpy as np
import mplstereonet
get_ipython().run_line_magic('matplotlib', 'inline')

from googlepicks import kml_to_plane
importlib.reload(kml_to_plane)

# Replace x,y,z with the actual data:
# p_0 = (x, y, z)
# p_1 = (x, y, z)
# p_2 = (x, y, z)
p_0, p_1, p_2 = (1,0,0), (0,0,0), (0,1,1)

points = [p_0, p_1, p_2]
ps = kml_to_plane.PointSet(type='nongeo', zone=0)
for p in points:
    ps.add_point(kml_to_plane.Point(x = p[0], y = p[1], z = p[2], type = 'nongeo', zone=0))

ps.get_orientation()
print("(%03d/%02d)" % (np.round(ps.dip_direction), np.round(ps.dip)))

ps.stereonet()



