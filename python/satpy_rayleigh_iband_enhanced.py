from satpy.scene import Scene

scene = Scene(base_dir="/data/lang/satellit2/polar/npp/lvl1/2017/09/npp_20170923_1126_30604/")

scene.load(['true_color'])

scene.show('true_color')



