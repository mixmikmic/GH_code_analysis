get_ipython().system('pip install Quandl')

code = 'OFDP/FUTURE_SB1'
description = 'Sugar No. 11 Futures, Continuous Contract #1'
price = 'Cents per Pound'
field = 'Settle'

code = 'OFDP/FUTURE_CC1'
description = 'Cocoa Futures, Continuous Contract #1'
price = 'US$ per metric ton'
field = 'Settle'



code = 'ODA/PPOIL_USD'
description = 'Malaysia Palm Oil Futures (first contract forward)'
price = 'US$ per metric ton'
field = 'Value'

import Quandl

end = datetime.datetime.today().date()
start = end - datetime.timedelta(5*365)

S = Quandl.get(code, collapse='daily',     trim_start=start.isoformat(), trim_end=end.isoformat())[field]

figure(figsize=(10,4))
S.plot()
title(description)
ylabel(price)



