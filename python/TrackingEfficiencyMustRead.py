from IPython.display import Image
pathname = "../files/"
cutVariable = "cutVariables.png"
Image(filename = pathname + cutVariable)

#demo of where the calo matching cut is
calo = "caloMatching.png"
Image(filename = pathname + calo)

efficiencyName = "AbsoluteEfficiency_PbPb_MB_NTT_20160118.png"
fakeName = "FakeRate_PbPb_MB_NTT_20160118.png"
SecondaryName = "SecondaryReconstruction_PbPb_MB_NTT_20160118.png"
MultipleName = "MultipleReconstruction_PbPb_MB_NTT_20160118.png"

Image(filename = pathname + efficiencyName)

Image(filename = pathname + fakeName)

Image(filename = pathname + SecondaryName)

Image(filename = pathname + MultipleName)

efficiencyName3D = "AbsoluteEfficiency3D_20151216.png"
fakeName3D = "FakeRate3D_20151216.png"
SecondaryName3D = "SecondaryReconstruction3D_20151216.png"
MultipleName3D = "MultipleReconstruction3D_20151216.png"
Image(filename = pathname + efficiencyName3D)

Image(filename = pathname + fakeName3D)

Image(filename = pathname + SecondaryName3D)

Image(filename = pathname + MultipleName3D)



