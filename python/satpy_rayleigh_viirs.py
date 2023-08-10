from satpy.scene import Scene
from glob import glob

scn = Scene(
        sensor="viirs",
        filenames=glob("/home/a000680/data/polar_in/direct_readout/npp/lvl1/npp_20150420_1202_18019/*"),
        reader='viirs_sdr')

from satpy.resample import get_area_def

areaid = 'eurol'

areadef = get_area_def(areaid)

composite = 'true_color_lowres'

scn.load([composite])

newscn = scn.resample(areadef)

newscn.show(composite)

prfx = newscn.start_time.strftime('%Y%m%d%H%M')

newscn.save_dataset(
        composite, './true_color_rayleigh_only_{0}_{1}.png'.format(prfx, areaid))

