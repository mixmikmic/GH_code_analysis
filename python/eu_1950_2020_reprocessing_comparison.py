import sqlite3 as lite
import os
import sys
sys.path.insert(0, '../scripts/')
jupyter_directory = os.getcwd()

# import necessary modules
import write_deployinst_input as wr
import analysis as an

# Write the Cyclus Input File
csv_file = '../database/eu_reactors_pris.csv'

reactor_template = '../templates/reactor_template.xml.in'
mox_reactor_template = '../templates/reactor_mox_template.xml.in'
deployinst_template = '../templates/deployinst_template.xml.in'
input_template = '../templates/input_template.xml.in'

# simulation starts at Jan, 01 , 1950, and for 840 months (70 years)
# first one: reprocessing, second one: once-through

# reprocessing case
wr.main(csv_file, 19500101, 840, reactor_template,
        mox_reactor_template, True,
        deployinst_template, input_template, './eu_reprocessing.xml')

#non-reprocessing case
wr.main(csv_file, 19500101, 840, reactor_template,
        mox_reactor_template, False,
        deployinst_template, input_template, './eu.xml')

get_ipython().system('cyclus -o ./eu_reprocessing.sqlite eu_reprocessing.xml')

get_ipython().system('cyclus -o ./eu.sqlite eu.xml')

# Get Final SNF capacity of Reprocessing
# and stacked bar chart of capacity and number of reactors
# Wait for the complete message to move on
import analysis as an
output = 'eu_reprocessing.sqlite'
con = lite.connect(output)
with con:
    cur = con.cursor()
    
    # prints total snf isotopic inventory at the end of simulation
    print(an.snf(cur))
    
    # capacity timeseries
    an.plot_power(cur)
    
    # plot of all the isotope timeseries of source
    plot_in_out_flux(cur, 'source', False, 'source vs time', 'source')
    
    # plot of all the isotope timeseries of sink
    plot_in_out_flux(cur, 'sink', Ture, 'isotope vs time', 'sink')
    
    print('Completed! You may go to the next box')

# Get Final SNF capacity of Reprocessing
# and stacked bar chart of capacity and number of reactors
# Wait for the complete message to move on
import analysis as an
output = 'eu.sqlite'
con = lite.connect(output)
with con:
    cur = con.cursor()
    
    # prints total snf isotopic inventory at the end of simulation
    print(an.snf(cur))
    
    # capacity timeseries
    an.plot_power(cur)
    
    # plot of all the isotope timeseries of source
    plot_in_out_flux(cur, 'source', False, 'source vs time', 'source')
    
    # plot of all the isotope timeseries of sink
    plot_in_out_flux(cur, 'sink', Ture, 'isotope vs time', 'sink')
    
    print('Completed! You may go to the next box')

# Display Net Capacity vs Time
from IPython.display import Image
Image(filename='power_plot.png')

# Display Number of Reactors vs Time
from IPython.display import Image
Image(filename='number_plot.png')



