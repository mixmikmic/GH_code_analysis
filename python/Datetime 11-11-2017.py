from datetime import datetime
from datetime import timedelta
now = datetime.now()
print(now)

datetime(2017, 11, 9, 6).strftime("%Y_%m_%d_%H_%M_%S")  # Create datetime manually and format

4 < now.hour < 6

now.day

now.microsecond

now.year

now.weekday() < 5  # is it a weekday?

now.ctime()

print(now.microsecond)
print(now.microsecond + 1)  # Add to times

print(now)
print(now + timedelta(days=1))  # Add to times

now = datetime.now()
now

YEAR = 2017
MONTH = 11
DAY = 27

day_of_interest = datetime(YEAR, MONTH, DAY)

diff = day_of_interest - now

future = now + timedelta(days=89)

future.day

diff

diff.days



