from glob import glob
from satpy.scene import Scene
scn = Scene(
        filenames=glob("/data/lang/proj/safworks/adam/oktober17smoke/npp/lvl1/npp_20171017_1216_30945/*"),
        reader='viirs_sdr')

scn.load(['true_color'])

newscn = scn.resample('europe2km')

newscn.show('true_color', overlay={'coast_dir': '/home/a000680/data/shapes/', 'color': (255, 140, 100)})

