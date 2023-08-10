import dsed

v = dsed.Veneer(port=9876)

# Set the ProcessCatchmentsInParallel option (Provided by Source)
v.model.source_scenario_options("PerformanceConfiguration","ProcessCatchmentsInParallel",True)

# Set the Dynamic Sednet specific options
v.configure_options({
        'RunNetworksInParallel':True,
        'PreRunCatchments':True,
        'ParallelFlowPhase':True
    })





