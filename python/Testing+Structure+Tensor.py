import tractography as tract

import analysis3 as a3

output = a3.get_raw_brain("s275", "userToken.pem", resolution = 5, save = True)

tract.nii_to_tiff_stack("s275_raw.nii", "s275")

data = tract.tiff_stack_to_array("s275_TIFFs/")

output = tract.generate_FSL_structure_tensor(data, "s275")

print output



