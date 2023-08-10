get_ipython().magic('matplotlib inline')
import os
import numpy as np
from dicompylercore import dicomparser, dvh, dvhcalc
import matplotlib.pyplot as plt
import urllib.request
import os.path

get_ipython().magic('load_ext watermark')
get_ipython().magic("watermark -a 'Robin Cole' -u -d -v -p numpy,matplotlib")

get_ipython().magic('mkdir -p example_data')
repo_url = 'https://github.com/dicompyler/dicompyler-core/blob/master/tests/testdata/file?raw=true'
# files = ['ct.0.dcm', 'rtss.dcm', 'rtplan.dcm', 'rtdose.dcm']
files = ['example_data/{}'.format(y) for y in ['rtss.dcm', 'rtdose.dcm']]
file_urls = [repo_url.replace('file', x) for x in files]
# Only download if the data is not present
[urllib.request.urlretrieve(x, y) for x, y in zip(file_urls, files) if not os.path.exists(y)]

rtss_dcm = files[0]
rtdose_dcm = files[1]
rtss = dicomparser.DicomParser(rtss_dcm)
rtdose = dicomparser.DicomParser(rtdose_dcm)

key = 5
structures = rtss.GetStructures()
structures[key]

planes =     (np.array(rtdose.ds.GridFrameOffsetVector)     * rtdose.ds.ImageOrientationPatient[0])     + rtdose.ds.ImagePositionPatient[2]
dd = rtdose.GetDoseData()

from ipywidgets import FloatSlider, interactive
w = FloatSlider(
    value=0.56,
    min=planes[0],
    max=planes[-1],
    step=np.diff(planes)[0],
    description='Slice Position (mm):',
)

def showdose(z):
    plt.imshow(rtdose.GetDoseGrid(z) * dd['dosegridscaling'],
               vmin=0,
               vmax=dd['dosemax'] * dd['dosegridscaling'])

interactive(showdose, z=w)

heart = rtdose.GetDVHs()[key]
heart.name = structures[key]['name']
heart.describe()

heart.relative_volume.plot()

lung = rtdose.GetDVHs()[6]
lung.name = structures[6]['name']
lung.rx_dose = 14
lung.plot()

lung.max

lung.relative_volume.V5Gy

lung.relative_dose().D2cc

plt.figure(figsize=(10, 6))
plt.axis([0, 20, 0, 100])
for s in structures.values():
    if not s['empty']:
        dvh.DVH.from_dicom_dvh(rtdose.ds, s['id'], name=s['name']).relative_volume.plot()

def compare_dvh(key=1):
    structure = rtss.GetStructures()[key]
    orig = dvh.DVH.from_dicom_dvh(rtdose.ds, key, name=structure['name'] + ' Orig')
    calc = dvhcalc.get_dvh(rtss_dcm, rtdose_dcm, key)
    calc.name = structure['name'] + ' Calc'
    orig.compare(calc)

compare_dvh(key)



