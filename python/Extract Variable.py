MONTHS = ('January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December')

import random
month = random.choice(MONTHS)

if (month.lower().endswith('r') or
        month.lower().endswith('ary')):
    print('%s is a good time to eat oysters' % month)
elif 8 > MONTHS.index(month) > 4:
    print('%s is a good time to eat tomatoes' % month)
else:
    print('%s is a good time to eat asparagus' % month)

lowered = month.lower()
ends_in_r = lowered.endswith('r')
ends_in_ary = lowered.endswith('ary')
summer = 8 > MONTHS.index(month) > 4

if ends_in_r or ends_in_ary:
    print('%s is a good time to eat oysters' % month)
elif summer:
    print('%s is a good time to eat tomatoes' % month)
else:
    print('%s is a good time to eat asparagus' % month)

def oysters_good(month):
    month_lowered = month.lower()
    return (
        month_lowered.endswith('r') or
        month_lowered.endswith('ary'))

def tomatoes_good(month):
    index = MONTHS.index(month)
    return 8 > index > 4

time_for_oysters = oysters_good(month)
time_for_tomatoes = tomatoes_good(month)

if time_for_oysters:
    print('%s is a good time to eat oysters' % month)
elif time_for_tomatoes:
    print('%s is a good time to eat tomatoes' % month)
else:
    print('%s is a good time to eat asparagus' % month)

class OystersGood:
    def __init__(self, month):
        month = month
        month_lowered = month.lower()
        self.ends_in_r = month_lowered.endswith('r')
        self.ends_in_ary = month_lowered.endswith('ary')
        self._result = self.ends_in_r or self.ends_in_ary

    def __bool__(self):  # Equivalent to __nonzero__ in Python 2
        return self._result
            

class TomatoesGood:
    def __init__(self, month):
        self.index = MONTHS.index(month)
        self._result = 8 > self.index > 4
    
    def __bool__(self):  # Equivalent to __nonzero__ in Python 2
        return self._result

time_for_oysters = OystersGood(month)
time_for_tomatoes = TomatoesGood(month)

if time_for_oysters:
    print('%s is a good time to eat oysters' % month)
elif time_for_tomatoes:
    print('%s is a good time to eat tomatoes' % month)
else:
    print('%s is a good time to eat asparagus' % month)

test = OystersGood('November')
assert test
assert test.ends_in_r
assert not test.ends_in_ary

test = OystersGood('July')
assert not test
assert not test.ends_in_r
assert not test.ends_in_ary

