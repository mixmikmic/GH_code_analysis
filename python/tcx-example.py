from ghettotcx import HeartRate, LatLong, TCX

h = HeartRate("./example_data/example.tcx")
h.plot_heartrate()  # time series of heart rate readings
h.plot_heartrate_histogram() 
h.plot_heartratezone()  # heart rate zones are hard-coded. TODO: make user-defineable

HeartRate.create_heartrate_panel(TCX.load_directory("./example_data", class_factory=HeartRate), fig_size=(6,2))

l = LatLong("./example_data/example.tcx")
l.plot()

