from glob import glob
from satpy.scene import Scene

fl_ = glob("/home/a000680/data/polar_in/direct_readout/eos/lvl1/MYD*_A16090_1143*hdf")

scn = Scene(
    platform_name="EOS-Aqua",
    sensor="modis",
    filenames=fl_,
    reader='hdfeos_l1b'
)

composite_name = 'true_color'
scn.load([composite_name])

areaid = 'eurol'
lcn = scn.resample(areaid, radius_of_influence=10000)

lcn.show(composite_name)

lcn.save_dataset(composite_name, './modis_%s_rgb_%s_%s.png' %
                 (composite_name, lcn.info['start_time'].strftime('%Y%m%d%H%M'), areaid))

