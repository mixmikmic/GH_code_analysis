# auto reload modules for code updates

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# exclude standard libraries for autoreload evaluation

get_ipython().magic('aimport -pandas')
get_ipython().magic('aimport -scipy.stats')
get_ipython().magic('aimport -numpy')
get_ipython().magic('aimport -matplotlib.pyplot')


# import standard Python modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import scipy.stats as stats
import scipy.interpolate as interpolate
import pandas

import xlwings as xw # for interaction with Excel

# import custom modules

import default_calcs as default
import collateral_waterfall as cw
import prepayment_calcs as pc
import bond_pricing as bp
from utils import *

# set table output to 2 decimal places

pd.options.display.float_format = '{:,.2f}'.format

a=pc.cpr_curve_creator()
fig, ax = plt.subplots(figsize=(10,2))
plt.plot(a)
plt.show()

speeds = [.5,1,2,5]
curves = [pc.cpr_curve_creator()*np.array(speed) for speed in speeds]
handles = []

fig, ax = plt.subplots(figsize=(10,2))
for i, curve in enumerate(curves):
    ax.plot(curve)

plt.show()

sched = cw.schedule_of_ending_balances(0.09, 360, 200000)['scheduled_balance']
a=pc.cpr_curve_creator()
fig, ax = plt.subplots(figsize=(10,2))
plt.plot(sched)
plt.show()

waterfall = cw.create_waterfall(original_balance=4e8, pass_thru_cpn=0.055,
                                psa_speed=0,wam=360, wac=0.055)

waterfall.tail()

waterfall = {}

figure, axes = plt.subplots()
for i in range(15):
    for j in range(4):
        waterfall[i,j] = cw.create_waterfall(original_balance=200000, 
                        psa_speed=i + (j * 0.25), 
                        pass_thru_cpn=0.075, 
                        wac=0.075,
                        wam=360)
        waterfall[i,j].beginning_balance.plot(ax=axes)
plt.show()

results = {}

coupon = 0.08/12.
terms = [180,360]
ages = [24,48,72,96,120,144,168]
    
for term in terms:
    for age in ages:
        results[term,age] = cw.schedule_of_ending_balance_percent_for_period(
        coupon, term, age)

pd.DataFrame(results, index=['% of Rem. Balance at period'])

original_balance=150000
pass_thru_coupon=0.08
wac=0.08
wam=360

first_wf = cw.create_waterfall(original_balance=original_balance, 
                               pass_thru_cpn=pass_thru_coupon,
                              wac=wac,
                              wam=wam,
                              cpr_description='7')
first_wf['CPR']=first_wf['SMM'].apply(pc.cpr)
first_wf.head()

original_balance=150000
pass_thru_coupon=0.08
wac=0.08
wam=360

second_wf = cw.create_waterfall(original_balance=original_balance, 
                               pass_thru_cpn=pass_thru_coupon,
                              wac=wac,
                              wam=wam,
                              psa_speed=3)
second_wf['CPR']=second_wf['SMM'].apply(pc.cpr)
second_wf.head()

bonds = pd.DataFrame([
    {'Face': 100, 'Maturity': 0.25, 'Coupon': 0, 'Price': 97.50},
    {'Face': 100, 'Maturity': 0.5, 'Coupon': 0, 'Price': 94.90},
    {'Face': 100, 'Maturity': 1.0, 'Coupon': 0, 'Price': 90.00},
    {'Face': 100, 'Maturity': 1.5, 'Coupon': 8, 'Price': 96.00},
    {'Face': 100, 'Maturity': 2.0, 'Coupon': 12, 'Price': 101.60}
])

bonds['coupon_freq'] = 1

bonds = bp.BondPricing(bonds)
spot_curve = bonds.spot_from_par()
print(spot_curve)
spot_curve.plot(y=['Yield','spot_rate'])
plt.show()

maturity1 = 1
maturity2 = 4

fwd_cont = bp.BondPricing.forward_rate(spot_curve.loc[maturity1,'spot_rate']/100,
                                      maturity1,
                                      spot_curve.loc[maturity2,'spot_rate']/100,
                                      maturity2, continuous=True) * 100
fwd_discrete = bp.BondPricing.forward_rate(spot_curve.loc[maturity1,'spot_rate']/100,
                                      maturity1,
                                      spot_curve.loc[maturity2,'spot_rate']/100,
                                      maturity2, continuous=False) * 100

print('Continuous compounding:\t{0:.2f}%\nDiscrete compounding:\t{1:.2f}%'.format(fwd_cont,fwd_discrete))

wb = xw.Book('/Users/ab4017/Google Drive/Programming/Python/zeusRMBS/mbs waterfall.xlsm')

sheet1 = wb.sheets(1)
values = sheet1.range('B1:B6').value

balance = values[0]
WAM = int(values[1])
WAC = values[2]
ptc = values[3]
speed = values[4]
cpr = values[5]

anchor = sheet1.range('A9')

# Plot multiple principal paydowns at different PSA speeds and add figure to spreadsheet

for speed in np.arange(.5, 3.1, .5):
    waterfall = cw.create_waterfall(original_balance=balance,
                                   wam=WAM,
                                   wac=WAC,
                                   pass_thru_cpn=ptc,
                                   psa_speed=speed,
                                   cpr_description=cpr)
    waterfall.rename(columns={'beginning_balance': str(speed)},
                    inplace=True)
    ax = waterfall[str(speed)].plot(legend=True)
    fig = ax.get_figure()

    sheet1.pictures.add(fig, name='Adam',update=True)

anchor.expand().value = ""
anchor.value = waterfall

