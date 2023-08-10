import nipype.interfaces.semtools.diffusion as ukf

from nipype.interfaces.semtools.diffusion import tractography

import nipype

import nipype.interfaces.semtools as tools

test = ukf.dtiestim(dwi_image = "Control258localeq.nii")

convert = ukf.DWIConvert()
convert.input_spec(allowLossyConversion=True, )

convert.output_spec()

from nipype.interfaces.freesurfer import MRIConvert

import nipype.interfaces.freesurfer.MRIConvert

a = tractography.UKFTractography()

a.input_spec()



