get_ipython().run_cell_magic('HTML', '', '<button id="launcher">Launch Car Simulator</button>\n<script src="setupLauncher.js"></script>')

# CODE CELL
#
# Write the park function so that it actually parks your vehicle.

from Car import Car
import time

def park(car):
    # TODO: Fix this function!
    #  currently it just drives back and forth
    #  Note that the allowed steering angles are
    #  between -25.0 and 25.0 degrees and the 
    #  allowed values for gas are between -1.0 and 1.0
    
    #  back up with 25.0 right angles for 3.1 seconds
    car.steer(25.0)
    car.gas(-0.5)
    time.sleep(2.9) # note how time.sleep works
    
    # back up for 2.7 seconds to keep the car parall  park 
    car.steer(-25.0)
    car.gas(-0.5)
    time.sleep(2.6) 
    
    
    # slow down the speed 
    car.steer(0.0)
    car.gas(0.1)
    time.sleep(1.8)
    
    car.steer(0.0)
    car.gas(-0.02)
    time.sleep(1.5)
    
    # stop the car 
    car.steer(0.0)
    car.gas(0.000000002)
    time.sleep(600.0)
    
    

car = Car()
park(car)

