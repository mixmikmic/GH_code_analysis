# Some steps required until these Python modules are properly installed...
import sys
sys.path.append('../Modules')
sys.path.append('../../../../veneer-py')
# Get the Source scripting components (veneer) and GBR specific stuff
import gbr

# Point the system at a particular output directory...
gbr.init('E:/Beckers/Output/Scenario 1/')

# Initialise the Veneer (Source scripting tool)
v = gbr.veneer()

# Also, lets switch on the performance options
v.configureOptions({'RunNetworksInParallel':True,'PreRunCatchments':True,'ParallelFlowPhase':True})
v.model.sourceScenarioOptions("PerformanceConfiguration","ProcessCatchmentsInParallel",True)

# Its a good idea to set some options in Dynamic Sednet to prevent the results window appearing
# Also, to make it automatically override existing results
#v.configureOptions{'ShowResultsAfterRun':False,'OverwriteResults':True})

# Query terms to find the BankHeight_M parameter...
# (Not easy!)
namespace = 'RiverSystem.Constituents.CatchmentElementConstituentData as CatchmentElementConstituentData\nimport GBR_DynSed_Extension.Models.GBR_CropSed_Wrap_Model as GBR_CropSed_Wrap_Model'
accessor_hillslope_cropping = 'scenario.Network.ConstituentsManagement.Elements.OfType[CatchmentElementConstituentData]().*FunctionalUnitData.*ConstituentModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").Where(lambda x: x.ConstituentSources[0].GenerationModel.GetType().Name=="GBR_CropSed_Wrap_Model").*ConstituentSources.*GenerationModel.ErosionModel.HillslopeFineSDR'#*Provider'#.*Model' #.*Data.ProcessingModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*Model'
#v.model.sourceHelp(accessor_hillslope,namespace=namespace)

# Now run the query and get the current values
orig_hillslope_cropping_sdr = v.model.get(accessor_hillslope_cropping,namespace=namespace)
orig_hillslope_cropping_sdr

accessor_gully_cropping = 'scenario.Network.ConstituentsManagement.Elements.OfType[CatchmentElementConstituentData]().*FunctionalUnitData.*ConstituentModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").Where(lambda x: x.ConstituentSources[0].GenerationModel.GetType().Name=="GBR_CropSed_Wrap_Model").*ConstituentSources.*GenerationModel.GULLYmodel.Gully_SDR_Fine'#*Provider'#.*Model' #.*Data.ProcessingModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*Model'
#v.model.sourceHelp(accessor_gully,namespace=namespace)

orig_gully_cropping_sdr = v.model.get(accessor_gully_cropping,namespace=namespace)
orig_gully_cropping_sdr

#accessor_hillslope_grazing = 'scenario.Network.ConstituentsManagement.Elements.OfType[CatchmentElementConstituentData]().*FunctionalUnitData.*ConstituentModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").Where(lambda x: x.ConstituentSources[0].GenerationModel.GetType().Name=="SedNet_Sediment_Generation").*ConstituentSources.*GenerationModel'#.ErosionModel.HillslopeFineSDR'#*Provider'#.*Model' #.*Data.ProcessingModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*Model'
#v.model.sourceHelp(accessor_hillslope_grazing,namespace=namespace)

#accessor_fine_sediment_model_types = 'scenario.Network.ConstituentsManagement.Elements.OfType[CatchmentElementConstituentData]().*FunctionalUnitData.*ConstituentModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*ConstituentSources.*GenerationModel.GetType().Name'#.ErosionModel.HillslopeFineSDR'#*Provider'#.*Model' #.*Data.ProcessingModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*Model'
#v.model.get(accessor_fine_sediment_model_types,namespace=namespace)



# Run with those original values

# First, set the name of the run
v.model.set('scenario.CurrentConfiguration.runName','RUN_ORIGINAL_SDR')

# Now, lets run the model... When this cell executes in Python, the run window should appear in Source...
v.run_model()

# NOTE: The above output (eg runs/1) is a point to retrieving the 'normal' Source results - ie all the time series recorders...
# We don't need that for GBR/Dynamic Sednet, because we can get to the summarised results

# Lets take a quick look at those results...
results_original = gbr.Results('RUN_ORIGINAL_SDR')
results_original.queries.regional_export('t/y')

# We can set every 'instance' of BankHeight_M - ie in every link- to a single value, with
#
# v.model.set(accessor,2.0)
#
# or we can pass in a list of values
#
# v.model.set(accessor,[0.2,0.3,0.5,0.4,1.0],fromList=True)
#
# Now... If your list of values is shorter than the number of instances... (ie # links),
# then the list will be 'recycled'... That is, the list will be reused repeatedly until values have been assigned to all
# instances...
#
# ie... Given that the Becker's model has 5 Links, [0.2,0.3] to saying [0.2,0.3,0.2,0.3,0.2]

# Set to a constant, 2
v.model.set(accessor_gully_cropping,0,namespace=namespace)
v.model.set(accessor_hillslope_cropping,0,namespace=namespace)

# Check that it took effect
v.model.get(accessor_gully_cropping,namespace=namespace)

# Now change the run name
v.model.set('scenario.CurrentConfiguration.runName','RUN_CHANGED_SDR')

v.run_model()

results_changed = gbr.Results('RUN_CHANGED_SDR')
results_changed.queries.regional_export('t/y')

# Now that we've done both runs, we probably want to put the parameter back to normal...
v.model.set(accessor_hillslope_cropping,orig_hillslope_cropping_sdr,namespace=namespace,fromList=True)

v.model.set(accessor_gully_cropping,orig_gully_cropping_sdr,namespace=namespace,fromList=True)

v.model.get(accessor_hillslope_cropping,namespace=namespace)

time_series_directory = 'E:/Beckers/Output/Scenario 1/RUN_CHANGED_SDR/TimeSeries/'

location = 'Outlet Node1'
import pandas as pd

flows = pd.read_csv(time_series_directory+'Flows/Flow_'+location+'_cubicmetrespersecond.csv',names=['Date','Flow'],index_col=0,parse_dates=True,dayfirst=True)
flows

loads = pd.read_csv(time_series_directory+'Sediment - Fine/Sediment - Fine_'+location+'_kilograms.csv',names=['Date','Sediment - Fine'],index_col=0,parse_dates=True,dayfirst=True)
loads

combined = pd.DataFrame([flows.Flow,loads['Sediment - Fine']]).transpose()
combined

combined.Flow *= 86.4

combined['Sediment - Fine'] *= 1000

combined

combined['concentration'] = combined['Sediment - Fine']/combined.Flow

get_ipython().run_line_magic('pylab', 'inline')
scatter(x=combined.Flow,y=combined.concentration)
xlabel('Flow ML/day')
ylabel('Concentration mg/L')
title('Ideally Bank Erosion Only')

get_ipython().run_line_magic('pinfo', 'scatter')



