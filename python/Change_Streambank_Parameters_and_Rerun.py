# Some steps required until these Python modules are properly installed...
import sys
sys.path.append('../Modules')
sys.path.append('../../../../veneer-py')
# Get the Source scripting components (veneer) and GBR specific stuff
import gbr

# Point the system at a particular output directory...
gbr.init('D:/Beckers/outputs/Scenario 1/')

# Initialise the Veneer (Source scripting tool)
v = gbr.veneer()

# Also, lets switch on the performance options
v.configureOptions({'RunNetworksInParallel':True,'PreRunCatchments':True,'ParallelFlowPhase':True})
v.model.sourceScenarioOptions("PerformanceConfiguration","ProcessCatchmentsInParallel",True)

# Its a good idea to set some options in Dynamic Sednet to prevent the results window appearing
# Also, to make it automatically override existing results
v.configureOptions({'ShowResultsAfterRun':False,'OverwriteResults':True})

# Query terms to find the BankHeight_M parameter...
# (Not easy!)
namespace = 'RiverSystem.Constituents.LinkElementConstituentData as LinkElementConstituentData'
accessor = 'scenario.Network.ConstituentsManagement.Elements.OfType[LinkElementConstituentData]().*Data.ProcessingModels.Where(lambda x: x.Constituent.Name=="Sediment - Fine").*Model.BankHeight_M'
#v.model.sourceHelp(accessor,namespace=namespace)

# Now run the query and get the current values
orig_bankheight_m = v.model.get(accessor,namespace=namespace)
orig_bankheight_m

# Run with those original values

# First, set the name of the run
v.model.set('scenario.CurrentConfiguration.runName','RUN_ORIGINAL_BANKHEIGHT')

# Now, lets run the model... When this cell executes in Python, the run window should appear in Source...
v.run_model()

# NOTE: The above output (eg runs/1) is a point to retrieving the 'normal' Source results - ie all the time series recorders...
# We don't need that for GBR/Dynamic Sednet, because we can get to the summarised results

# Lets take a quick look at those results...
results_original = gbr.Results('RUN_ORIGINAL_BANKHEIGHT')
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
v.model.set(accessor,2,namespace=namespace)

# Check that it took effect
v.model.get(accessor,namespace=namespace)

# Now change the run name
v.model.set('scenario.CurrentConfiguration.runName','RUN_CHANGED_BANKHEIGHT')

v.run_model()

results_changed = gbr.Results('RUN_CHANGED_BANKHEIGHT')
results_changed.queries.regional_export('t/y')

# Now that we've done both runs, we probably want to put the parameter back to normal...
v.model.set(accessor,orig_bankheight_m,namespace=namespace,fromList=True)

v.model.get(accessor,namespace=namespace)

# Now... Lets run a results comparison...
differences = gbr.DifferenceResults('RUN_ORIGINAL_BANKHEIGHT','RUN_CHANGED_BANKHEIGHT')
differences.queries.regional_export('t/y')



