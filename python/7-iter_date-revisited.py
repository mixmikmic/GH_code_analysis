import datetime

def iter_date(date, increment=datetime.timedelta(days=1)):
    while True:
        yield date
        date += increment

n = 5
start_date = datetime.date(2016, 2, 27)
date_increment = datetime.timedelta(days=1)
start_date

from itertools import islice

list(islice(iter_date(start_date), n))

list(islice(iter_date(start_date, datetime.timedelta(days=-7)), n*2))

list(islice(iter_date('a one', ' anna two'), n))

list(islice(iter_date(1, 2), n))

import datetime
from itertools import accumulate, repeat, chain

list(accumulate(islice(chain.from_iterable([['a one'], repeat(' anna two')]), n)))

list(accumulate(islice(chain.from_iterable([[1], repeat(2)]), n)))

list(accumulate(islice(chain.from_iterable([[start_date], repeat(date_increment)]), n)))

list(accumulate(chain.from_iterable([[start_date], islice(repeat(date_increment), n)])))

