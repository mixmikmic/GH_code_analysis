MONTH_NDAYS = ''' 
    0:31
    1:29
    2:31
    3:30
    4:31
    5:30
    6:31
    7:31
    8:30
    9:31
    10:30
    11:31
'''.split()

MONTH_NDAYS

for month_n_days in MONTH_NDAYS:
    month, n_days = map(int, month_n_days.split(':'))
    print(f'{month} has {n_days}')

MONTH_NDAYS = [list(map(int, s.split(':'))) for s in ''' 
    0:31
    1:29
    2:31
    3:30
    4:31
    5:30
    6:31
    7:31
    8:30
    9:31
    10:30
    11:31
'''.split()]

MONTH_NDAYS

for month, n_days in MONTH_NDAYS:
    print(f'{month} has {n_days}')

MONTH_NDAYS = ''' 
    0:31
    1:29
    2:31
    3:30
    4:31
    5:30
    6:31
    7:31
    8:30
    9:31
    10:30
    11:31
'''.split()
MONTH_NDAYS

MONTH_NDAYS = [s.split(':') for s in MONTH_NDAYS]
MONTH_NDAYS

MONTH_NDAYS = [list(map(int, x)) for x in MONTH_NDAYS]
MONTH_NDAYS

for month, n_days in MONTH_NDAYS:
    print(f'{month} has {n_days}')

