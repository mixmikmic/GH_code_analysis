get_ipython().magic('pinfo round')

d = 2.675
c = round(d, ndigits=2)
c   # it rounds down

import decimal as my_dec
my_dec.Decimal(2.675)    # how it is stored

my_dec.getcontext()

get_ipython().magic('pinfo Decimal')

my_dec.getcontext().rounding

d = 2.675
Decimal(d)

Decimal(d).quantize(Decimal('.01'), rounding=ROUND_UP)   # round up

Decimal(d).quantize(Decimal('.01'), rounding=ROUND_DOWN)    # round downD

Decimal(2.203).quantize(Decimal('.01'), rounding=ROUND_UP) 

get_ipython().magic('pinfo Decimal')



