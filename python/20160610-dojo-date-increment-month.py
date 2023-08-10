import datetime

date = datetime.date(2009, 9, 28)

date.month

date.month += 1

get_ipython().system('pip install python-dateutil')

from dateutil.relativedelta import relativedelta

date + relativedelta(months=1)

def iter_month(date, increment=relativedelta(months=1)):
    while True:
        yield date
        date += increment

from itertools import islice

tuple(islice(iter_month(date), 5))

