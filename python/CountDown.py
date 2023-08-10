from datetime import datetime
import pytz
from math import floor

end_date = datetime(2014,12,18,19,30,0,0,pytz.UTC)  # explicitly make time zone UTC

time_till_end_date = end_date - datetime.now(pytz.UTC) # explicitly make time zone UTC

total_seconds = time_till_end_date.total_seconds()
days_left_remainder = total_seconds/(60*60*24) - floor(total_seconds/(60*60*24))
hours_left = days_left_remainder * 24
hours_left_remainder = days_left_remainder * 24 - floor(hours_left)
minutes_left = hours_left_remainder * 60
minutes_left_remainder = minutes_left - floor(minutes_left)
seconds_left = minutes_left_remainder * 60

print('Days:', time_till_end_date.days, '|', 'Hours:', int(hours_left), '|', 'Minutes:', int(minutes_left), '|',
      'Seconds:', int(seconds_left))

from datetime import datetime
import pytz
from math import floor

end_date = datetime(2014,12,18,19,30,0,0,pytz.UTC)  # explicitly make time zone UTC

while True:
    time_till_end_date = end_date - datetime.now(pytz.UTC)  # explicitly make time zone UTC

    total_seconds = time_till_end_date.total_seconds()
    days_left_remainder = total_seconds/(60*60*24) - floor(total_seconds/(60*60*24))
    hours_left = days_left_remainder * 24
    hours_left_remainder = days_left_remainder * 24 - floor(hours_left)
    minutes_left = hours_left_remainder * 60
    minutes_left_remainder = minutes_left - floor(minutes_left)
    seconds_left = minutes_left_remainder * 60

    # print on the same line
    print('\r'+'Days: ' + str(time_till_end_date.days) + ' | ' + 'Hours: ' + str(int(hours_left)) + ' | ' + 'Minutes: ' +           str(int(minutes_left)) + ' | ' + 'Seconds: ' + "{0:d}".format(int(seconds_left)),end='')

from datetime import datetime
import pytz
import time

end_date = datetime(2015,5,29,13,0,0,0,pytz.UTC)  # explicitly make time zone UTC

while True:
    time_till_end_date = end_date - datetime.now(pytz.UTC)  # explicitly make time zone UTC
    total_seconds = time_till_end_date.total_seconds()
    
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    if days > 1:
        strDays = ' days'
    else:
        strDays = ' day'

    if int(hours) > 1:
        strHours = ' hours'
    else:
        strHours = ' hour'

    if int(minutes) > 1:
        strMinutes = ' minutes'
    else:
        strMinutes = ' minute'

    if int(seconds) > 1:
        strSeconds = ' seconds'
    else:
        strSeconds = ' second'

    # print on the same line
    print('\r' + str(int(days)) + strDays + ' | ' + str(int(hours)) + strHours + ' | ' + str(int(minutes)) +           strMinutes + ' | ' + "{0:.1f}".format(seconds) + strSeconds,end='')
    
    time.sleep(0.05)

import time

message = 'ALL YOUR BASE ARE BELONG TO US!'

for i in range(1, len(message)+1):
    time.sleep(0.3)
    print('\r'+message[:i],end='')

