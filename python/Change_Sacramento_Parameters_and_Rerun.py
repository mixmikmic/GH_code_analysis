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

# A path to the variable of interest... .* means 'give me all'
accessor = 'scenario.Network.GetCatchments().*FunctionalUnits.*rainfallRunoffModel.theBaseRRModel.Pctim'

existingValues = v.model.get('scenario.Network.GetCatchments().*FunctionalUnits.*rainfallRunoffModel.theBaseRRModel.Pctim')
existingValues

# Run with those original values

# First, set the name of the run
v.model.set('scenario.CurrentConfiguration.runName','RUN_ORIGINAL_PCTIM')

# Its a good idea to set some options in Dynamic Sednet to prevent the results window appearing
# Also, to make it automatically override existing results
v.configureOptions({'ShowResultsAfterRun':False,'OverwriteResults':True})

# Also, lets switch on the performance options
v.configureOptions({'RunNetworksInParallel':True,'PreRunCatchments':True,'ParallelFlowPhase':True})
v.model.sourceScenarioOptions("PerformanceConfiguration","ProcessCatchmentsInParallel",True)

# Now, lets run the model... When this cell executes in Python, the run window should appear in Source...
v.run_model()

# NOTE: The above output (eg runs/1) is a point to retrieving the 'normal' Source results - ie all the time series recorders...
# We don't need that for GBR/Dynamic Sednet, because we can get to the summarised results

# Lets take a quick look at those results...
results_original = gbr.Results('RUN_ORIGINAL_PCTIM')
results_original.queries.regional_export('t/y')



# NOW... Lets change the Pctim parameter...
# We'll use the same accessor -- but this time we'll use it to set values
accessor

# We can set every 'instance' of Pctim - ie every FU in every subcatchment - to a single value, with
#
# v.model.set(accessor,0.5)
#
# or we can pass in a list of values
#
# v.model.set(accessor,[0.2,0.3,0.5,0.4,1.0],fromList=True)
#
# Now... If your list of values is shorter than the number of instances... (ie # subcatchments x # FUs),
# then the list will be 'recycled'... That is, the list will be reused repeatedly until values have been assigned to all
# instances...
#
# ie... Given that the Becker's model has 5 FUs, [0.2,0.3,0.5,0.4,1.0] is the equivalent of giving those five values to the
# five functional units in each subcatchment...
v.model.set(accessor,[0.2,0.3,0.5,0.4,1.0],fromList=True)

# Lets check what happened...
v.model.get(accessor)

v.model.set('scenario.CurrentConfiguration.runName','RUN_CHANGED_PCTIM')

v.run_model()

results_changed = gbr.Results('RUN_CHANGED_PCTIM')
results_changed.queries.regional_export('t/y')

# Now that we've done both runs, we probably want to put the parameter back to normal...
v.model.set(accessor,existingValues,fromList=True)

# Now... Lets run a results comparison...
differences = gbr.DifferenceResults('RUN_ORIGINAL_PCTIM','RUN_CHANGED_PCTIM')
differences.queries.regional_export('t/y')

# As might be expected, increasing the impervious fraction increases flow and consequently the constituents also increase....



