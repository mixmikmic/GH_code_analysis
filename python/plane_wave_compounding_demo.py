import setup.pymus_us_dataset as pymus_us_dataset
import setup.pymus_scan_region as pymus_scan_region
import setup.pymus_us_probe as pymus_us_probe
import setup.pymus_us_sequence as pymus_sequence
import result.pymus_image as pymus_image
import processing.pymus_beamforming as pymus_beamforming
import tools.pymus_utils as pymus_utils
import numpy as np

nbPW = 3
pht="in_vitro_type1"

scan = pymus_scan_region.ScanRegion()
scf_name = pymus_utils.TO_DATA_TEST + "scan_region/linear_scan_region.hdf5"
scan.read_file(scf_name,"scan_region")
print(scan)

probe = pymus_us_probe.UsProbe()
prb_name = pymus_utils.TO_DATA_TEST + "probe/linear_probe.hdf5"
probe.read_file(prb_name,"probe")
print(probe)

seq = pymus_sequence.UsPWSequence()
sqc_name = pymus_utils.TO_DATA_TEST + "sequence/sequence_nb_pw_%s.hdf5" % nbPW
seq.read_file(sqc_name,"sequence")
print(seq)

data = pymus_us_dataset.UsDataSet("dataset_%s_pw" % nbPW,probe,seq)
dst_name = pymus_utils.TO_DATA_TEST + "echo/%s_nb_pw_%s.hdf5" % (pht,nbPW) 
data.read_file(dst_name,pht)
print(data)

beamformer = pymus_beamforming.BeamFormer(scan,probe,"none")

beamformer.beamform(seq,data.data)

im_path = pymus_utils.TO_PYMUS + "experiment/output/image_%s_bf_%s_PW.hdf5" % (pht,nbPW) 
beamformer.write_image(im_path)
img = pymus_image.EchoImage(scan)
img.read_file(im_path,None)
img.show_image(dbScale=True,dynamic_range=60)



