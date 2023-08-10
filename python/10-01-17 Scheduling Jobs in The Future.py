# Import required modules
import sched
import time

# setup the scheduler with our time settings
s = sched.scheduler(time.time, time.sleep)

# Create a function we want to run in the future.
def alyprint_time():
    print('Executive Order 66')

# Create a function for the delay
def print_some_times():
    # Create a scheduled job that will run
    # the function called 'print_time'
    # after 10 seconds, and with priority 1.
    s.enter(10, 1, print_time)

    # Run the scheduler
    s.run()

# Run the function for the delay
print_some_times()

