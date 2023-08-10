# import all prerequisites
from PyFoam.Applications.CloneCase import CloneCase
from PyFoam.Applications.ClearCase import ClearCase
from PyFoam.Applications.CreateBoundaryPatches import CreateBoundaryPatches
from PyFoam.Applications.ChangeBoundaryName import ChangeBoundaryName
from PyFoam.Applications.ChangeBoundaryType import ChangeBoundaryType
from PyFoam.Applications.WriteDictionary import WriteDictionary
from PyFoam.Applications.Runner import Runner
from PyFoam.Applications.MeshUtilityRunner import MeshUtilityRunner
from PyFoam.Applications.ClearInternalField import ClearInternalField
from PyFoam.Applications.Decomposer import Decomposer
import shutil, os, sys
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))

sandia_path = "/home/pavel.kholodov/WORK/OpenFOAM/Benchmark/combustion/SandiaFlameD/flame/CaseB/B2_1X"
newcase = "Mixing"
meshfile = "Chamber_2Daxi.msh"

get_ipython().run_cell_magic('capture', 'capt', '# Close case from existing one\nCloneCase(args=["--no-vcs",sandia_path,newcase,"--no-pyfoam","--add-item=0.orig","--force"])')

get_ipython().run_cell_magic('capture', 'capt', '# Clear case\n#ClearCase(args=["-h"])\nClearCase(args=["--remove-analyzed","--no-pyfoam","--additional=Mixing.foam","--additional=0",newcase])')

get_ipython().run_cell_magic('capture', 'capt', 'Runner(args=["--silent","fluentMeshToFoam","-case",os.path.join(os.getcwd(),newcase),meshfile])')

get_ipython().run_cell_magic('capture', 'capt', '#shutil.copytree("./Mixing/0.orig","./Mixing/0")\nClearCase(args=["--remove-analyzed","--no-pyfoam","--additional=Mixing.foam",newcase])')

get_ipython().run_cell_magic('capture', 'capt', '#shutil.copytree(os.path.join(newcase,"0.orig"),os.path.join(newcase,"0"))\n# define BC names\nmesh_bc = {\'fuel_inlet\':\'FUEL_INLET\',\\\n           \'oxidizer_inlet\':\'OX_INLET\',\\\n           \'outlet\':\'OUTLET\',\\\n           \'opening\':\'ATMOSPHERE\',\\\n           \'coflow\':\'COFLOW\',\n           \'walls\':[\'BOTTOM\',\'CHAMBER_WALL\',\'NOZZLE_WALL\',\'JET_WALL\'],\n           \'sides\':[\'FRONT\',\'BACK\']}\ncase_bc = {\'fuel_inlet\':\'fuel_inlet\',\\\n           \'oxidizer_inlet\':\'ox_inlet\',\\\n           \'outlet\':\'outlet\',\\\n           \'opening\':\'atmosphere\',\\\n           \'coflow\':\'coflow\',\\\n           \'walls\':[\'bottom\',\'chamber_wall\',\'nozzle_wall\',\'jet_wall\'],\\\n           \'sides\':[\'front\',\'back\']}\n#Change names\nChangeBoundaryName(args=[newcase,mesh_bc[\'oxidizer_inlet\'],case_bc[\'oxidizer_inlet\']])\nChangeBoundaryName(args=[newcase,mesh_bc[\'fuel_inlet\'],case_bc[\'fuel_inlet\']])\nfor i in range(len(mesh_bc[\'walls\'])):\n    ChangeBoundaryName(args=[newcase,mesh_bc[\'walls\'][i],case_bc[\'walls\'][i]])\n\nChangeBoundaryName(args=[newcase,mesh_bc[\'coflow\'],case_bc[\'coflow\']])\nChangeBoundaryName(args=[newcase,mesh_bc[\'opening\'],case_bc[\'opening\']])\nChangeBoundaryName(args=[newcase,mesh_bc[\'outlet\'],case_bc[\'outlet\']])\nfor i in range(len(mesh_bc[\'sides\'])):\n    ChangeBoundaryName(args=[newcase,mesh_bc[\'sides\'][i],case_bc[\'sides\'][i]])')

get_ipython().run_cell_magic('capture', 'capt', '# Change boundary type to appropriate\nbc_types = {\'fuel_inlet\':\'patch\',\\\n            \'oxidizer_inlet\':\'patch\',\\\n            \'outlet\':\'patch\',\\\n            \'opening\':\'patch\',\\\n            \'coflow\':\'patch\',\\\n            \'sides\':\'wedge\'}\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\'],case_bc[\'outlet\'],case_bc[\'opening\']]:\n    ChangeBoundaryType(args=[newcase,bcname,\'patch\'])\nChangeBoundaryType(args=[newcase,"front","wedge"])\nfor bc in case_bc[\'sides\']:\n    ChangeBoundaryType(args=[newcase,bc,bc_types[\'sides\']])')

get_ipython().run_cell_magic('capture', 'capt', '# Delete unneeded BC files\nfor i in [\'CH4\',\'OH\',\'G\',\'H2O\',\'CO2\',\'CO\',\'O\',\'H\']:\n    os.remove("./"+newcase+"/0/"+i)')

get_ipython().run_cell_magic('capture', 'capt', '# Clear all boundary conditions from old names\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/U")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/p")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/T")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/k")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/epsilon")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/nut")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/alphat")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/O2")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/H2")])\nCreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/N2")])')

CreateBoundaryPatches(args=["--clear-unused","--overwrite",os.path.join(os.getcwd(),newcase,"0/Ydefault")])

get_ipython().run_cell_magic('capture', 'capt', '# BC for velocity\n# Need to add somehow the table entry for inlets\n#Wall BC\nCreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'fixedValue\',\'value\':\'uniform (0 0 0)\'}",os.path.join(os.getcwd(),newcase,"0/U")])\n# Inlet BC\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,"--default={\'type\':\'flowRateInletVelocity\'}",os.path.join(os.getcwd(),newcase,"0/U")])\n# Outlet BC\nCreateBoundaryPatches(args=["--overwrite","--filter=coflow","--default={\'type\':\'zeroGradient\'}",os.path.join(os.getcwd(),newcase,"0/U")])\nCreateBoundaryPatches(args=["--overwrite","--filter=outlet","--default={\'type\':\'pressureInletOutletVelocity\',\'value\':\'uniform (0 0 0)\'}",os.path.join(os.getcwd(),newcase,"0/U")])\nCreateBoundaryPatches(args=["--overwrite","--filter=outlet","--default={\'type\':\'pressureInletOutletVelocity\',\'value\':\'uniform (0 0 0)\'}",os.path.join(os.getcwd(),newcase,"0/U")])\n# Wedges\nCreateBoundaryPatches(args=["--overwrite","--filter=front","--default={\'type\':\'wedge\'}",os.path.join(os.getcwd(),newcase,"0/U")])\nCreateBoundaryPatches(args=["--overwrite","--filter=back","--default={\'type\':\'wedge\'}",os.path.join(os.getcwd(),newcase,"0/U")])')

get_ipython().run_cell_magic('capture', 'capt', '# BC for pressure\n# Wall BC\nCreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'zeroGradient\'}",os.path.join(os.getcwd(),newcase,"0/p")])\n# Inlet BC\nCreateBoundaryPatches(args=["--overwrite","--filter=ox_inlet","--default={\'type\':\'zeroGradient\'}",os.path.join(os.getcwd(),newcase,"0/p")])\n#CreateBoundaryPatches(args=["--overwrite","--filter=fuel_inlet","--default={\'type\':\'zeroGradient\'}",os.path.join(os.getcwd(),newcase,"0/U")])\n# Outlet BC\nCreateBoundaryPatches(args=["--overwrite","--filter=outlet","--default={\'type\':\'pressureInletOutletVelocity\',\'value\':\'uniform 98066.5\'}",os.path.join(os.getcwd(),newcase,"0/p")])\nCreateBoundaryPatches(args=["--overwrite","--filter=coflow","--default={\'type\':\'pressureInletOutletVelocity\',\'value\':\'uniform 98066.5\'}",os.path.join(os.getcwd(),newcase,"0/p")])')

ClearInternalField(args=['--value=0.0',os.path.join(os.getcwd(),newcase,"0/H2")])
ClearInternalField(args=['--value=0.23',os.path.join(os.getcwd(),newcase,"0/O2")])
ClearInternalField(args=['--value=0.77',os.path.join(os.getcwd(),newcase,"0/N2")])
ClearInternalField(args=['--value=0.0',os.path.join(os.getcwd(),newcase,"0/Ydefault")])

ClearInternalField(args=['--value=293.0',os.path.join(os.getcwd(),newcase,"0/T")])
ClearInternalField(args=['--value=(0 0 0)',os.path.join(os.getcwd(),newcase,"0/U")])
ClearInternalField(args=['--value=98066.5',os.path.join(os.getcwd(),newcase,"0/p")])

ClearInternalField(args=['--value=5.0',os.path.join(os.getcwd(),newcase,"0/k")])
ClearInternalField(args=['--value=32.0',os.path.join(os.getcwd(),newcase,"0/epsilon")])

get_ipython().run_cell_magic('capture', 'capt', '# BC for H2,O2,N2 components\nfiles = [\'O2\',\'H2\',\'N2\']\n# Wall BC\nfor i in files:\n    CreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'zeroGradient\'}",\\\n                                os.path.join(os.getcwd(),newcase,"0",i)])\n# Inlet BC\n# oxidizer inlet\nvalues = [\'1.0\',\'0.0\',\'0.0\']\nfor i in range(len(files)):\n    CreateBoundaryPatches(args=["--overwrite","--filter="+case_bc[\'oxidizer_inlet\'],\\\n                                "--default={\'type\':\'fixedValue\',\'value\':\'uniform %s\'}"%(values[i]),\\\n                                os.path.join(os.getcwd(),newcase,"0",files[i])])\n# fuel inlet\nvalues = [\'0.0\',\'1.0\',\'0.0\']\nCreateBoundaryPatches(args=["--overwrite","--filter="+case_bc[\'fuel_inlet\'],\\\n                            "--default={\'type\':\'fixedValue\',\'value\':\'uniform %s\'}"%(values[i]),\\\n                            os.path.join(os.getcwd(),newcase,"0",files[i])])\n# Outlet BC - Atmosphere\nfiles = [\'O2\',\'H2\',\'N2\']\nvalues = [\'0.23\',\'0.0\',\'0.77\']\nfor i in range(len(files)):\n    CreateBoundaryPatches(args=["--overwrite","--filter="+case_bc[\'opening\'],\\\n                                "--default={\'type\':\'fixedValue\',\'value\':\'uniform %s\'}"%(values[i]),\\\n                                os.path.join(os.getcwd(),newcase,"0",files[i])])\n    CreateBoundaryPatches(args=["--overwrite","--filter="+case_bc[\'outlet\'],\\\n                                "--default={\'type\':\'fixedValue\',\'value\':\'uniform %s\'}"%(values[i]),\\\n                                os.path.join(os.getcwd(),newcase,"0",files[i])])')

# BC for Ydefault components
CreateBoundaryPatches(args=["--overwrite","--default={'type':'zeroGradient'}",os.path.join(os.getcwd(),newcase,"0/Ydefault")])
for bcname in [case_bc['oxidizer_inlet'],case_bc['fuel_inlet'],case_bc['opening'],case_bc['outlet']]:
    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,                                "--default={'type':'fixedValue','value':'uniform 0.0'}",                                os.path.join(os.getcwd(),newcase,"0/Ydefault")])

get_ipython().run_cell_magic('capture', 'capt', '# BC for temperature\nCreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'fixedValue\',\'value\':\'uniform 293\'}",\\\n                            os.path.join(os.getcwd(),newcase,"0/T")])\n\nfor bcname in [case_bc[\'opening\'],case_bc[\'outlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,"--default={\'type\':\'inletOutlet\',\'value\':\'uniform 293\'}",\\\n                                os.path.join(os.getcwd(),newcase,"0/T")])\n"""\nWriteDictionary(args=[os.path.join(os.getcwd(),newcase,"0/T"),\\\n                      "boundaryField[\'outlet\'][\'value\']","293"])\nWriteDictionary(args=[os.path.join(os.getcwd(),newcase,"0/T"),\\\n                      "boundaryField[\'atmosphere\'][\'value\']","293"])\n"""')

get_ipython().run_cell_magic('capture', 'capt', '# BC for alphat\nCreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'compressible::alphatWallFunction\',\'value\':\'$internalField\'}",\\\n                            os.path.join(os.getcwd(),newcase,"0/alphat")])\n# BC for inlets/outlets\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\'],case_bc[\'opening\'],case_bc[\'outlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                                "--default={\'type\':\'calculated\',\'value\':\'$internalField\'}",\\\n                                os.path.join(os.getcwd(),newcase,"0/alphat")])')

get_ipython().run_cell_magic('capture', 'capt', '# BC for k, epsilon, nut\nCreateBoundaryPatches(args=["--overwrite","--default={\'type\':\'kqRWallFunction\',\'value\':\'uniform 0\'}",\\\n                            os.path.join(os.getcwd(),newcase,"0/k")])\nk_values = [\'0.05\',\'0.05\',\'0.01\',\'0.01\']\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                                "--default={\'type\':\'turbulentIntensityKineticEnergyInlet\',\'value\':\'1\'}",\\\n                                os.path.join(os.getcwd(),newcase,"0/k")])\nfor bcname in [case_bc[\'opening\'],case_bc[\'outlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                            "--default={\'type\':\'inletOutlet\',\'value\':\'uniform 0.001\'}",\\\n                            os.path.join(os.getcwd(),newcase,"0/k")])\n# BC for epsilon\nmixing_length = [1.5e-4,7.6e-5,0.031,0.00923]\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                                "--default={\'type\':\'turbulentMixingLengthDissipationRateInlet\',\'value\':\'200\'}",\\\n                                os.path.join(os.getcwd(),newcase,"0/epsilon")])\n\nfor bcname in [case_bc[\'opening\'],case_bc[\'outlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                            "--default={\'type\':\'inletOutlet\',\'value\':\'uniform 0.0003\'}",\\\n                            os.path.join(os.getcwd(),newcase,"0/epsilon")])\n# BC for nut\nfor bcname in case_bc[\'walls\']:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                        "--default={\'type\':\'nutkWallFunction\',\'value\':\'uniform 0\'}",\\\n                        os.path.join(os.getcwd(),newcase,"0/nut")])\nfor bcname in [case_bc[\'oxidizer_inlet\'],case_bc[\'fuel_inlet\'],case_bc[\'opening\'],case_bc[\'outlet\']]:\n    CreateBoundaryPatches(args=["--overwrite","--filter="+bcname,\\\n                        "--default={\'type\':\'calculated\',\'value\':\'uniform 0\'}",\\\n                        os.path.join(os.getcwd(),newcase,"0/nut")])')

shutil.rmtree(os.path.join(os.getcwd(),newcase,"0.mixing.orig"))
shutil.copytree(os.path.join(os.getcwd(),newcase,"0"),os.path.join(os.getcwd(),newcase,"0.mixing.orig"))

os.system("foamDictionary -entry 'startFrom' -set 'startTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'stopAt' -set 'endTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'endTime' -set '0.07' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'deltaT' -set '1e-7' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'writeControl' -set 'adjustableRunTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'writeInterval' -set '0.002' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'adjustTimeStep' -set 'yes' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'maxDeltaT' -set '0.0001' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'maxCo' -set '3' %s/system/controlDict 2>err"%(newcase))

os.system("foamDictionary -entry 'functions.minMaxT.writeControl' -set 'adjustableRunTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'functions.minMaxT.writeInterval' -set '1e-5' %s/system/controlDict 2>err"%(newcase))

os.system("foamDictionary -entry 'functions.minMaxP.writeControl' -set 'adjustableRunTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'functions.minMaxP.writeInterval' -set '1e-5' %s/system/controlDict 2>err"%(newcase))

os.system("foamDictionary -entry 'functions.volAveTemp.writeControl' -set 'adjustableRunTime' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'functions.volAveTemp.writeInterval' -set '1e-5' %s/system/controlDict 2>err"%(newcase))
os.system("foamDictionary -entry 'functions.volAveTemp.name' -set 'chamber' %s/system/controlDict 2>err"%(newcase))

os.system("foamDictionary -entry 'chemistry' -set 'off' %s/constant/chemistryProperties 2>err"%(newcase))

os.system("foamDictionary -entry 'ddtSchemes.default' -set 'Euler' %s/system/fvSchemes 2>err"%(newcase))
os.system("foamDictionary -entry 'PIMPLE.nOuterCorrectors' -set '30' %s/system/fvSolution 2>err"%(newcase))

Runner(args=["--silent","topoSet","-case",os.path.join(os.getcwd(),newcase)])

Decomposer(args=["--method=simple","--n=(4,1,1)","--delta=1e-4","--no-decompose","./%s"%(newcase),"4"])

