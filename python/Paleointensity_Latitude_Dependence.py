import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import pmagpy.ipmag as ipmag
import pmagpy.pmag as pmag

latitudes = range(-90,91)
field = []

for lat in latitudes:
    local_field = pmag.vdm_b(8e22,lat)
    field.append(local_field*1e6)

plt.plot(latitudes,field)
plt.xlim(-90,90)
plt.xlabel('latitude')
plt.ylabel('local field intensity ($\mu$T)')
plt.title('local field intensity for 80 ZAm$^2$ geomagnetic field')
plt.show()

pmag.b_vdm(45e-6,45)

pmag.b_vdm(16e-6,25)



