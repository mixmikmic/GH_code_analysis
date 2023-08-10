# Import modules
from datetime import datetime
from datetime import timedelta

# Create a variable with the current time
now = datetime.now()
now

# The current year
now.year

# The current month
now.month

# The current day
now.day

# The current hour
now.hour

# The current minute
now.minute

# The difference between two dates
delta = datetime(2011,1,7) - datetime(2011, 1,6)
delta

# THe difference days
delta.days

# The difference seconds
delta.seconds

# Create time
start = datetime(2011, 1, 7)

# Add twelve days to the time
start + timedelta(12)

