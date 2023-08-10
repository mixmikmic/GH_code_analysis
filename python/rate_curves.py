get_ipython().magic('matplotlib inline')
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pyrate import RateCurves


# Load the holidays
holidays = []
with open("files/holidays.txt", "r") as f:
    dates = f.read().split("\n")
    for date in dates:
        holidays.append(datetime.strptime(date, "%m/%d/%y"))

# Initialize the curve 
# The constant coupon rates for nx1 swaps trading liquid in the market
mid_mkt = np.array([0.0409 ,  0.04315,  0.0446 ,  0.046  ,  0.04765,  0.05035,
                    0.05245,  0.0543 ,   0.05605,  0.05905,  0.06215,  0.06401,
                    0.0668 ,  0.06911,  0.07142])

# 'n' flows of of a variable rate v.s. one fixed rate
nx1_instruments = np.array([1, 3, 6, 9, 13, 26, 39, 52, 65, 91, 130, 156, 195, 260, 390])

val_date = datetime(2016, 6, 27)
tiie28 = RateCurves(val_date, mid_mkt, nx1_instruments, 28, holidays)

print(tiie28.discount_rates)
print(tiie28.fitted_curve)

tiie28.fit_curve()
print(tiie28.discount_rates)
print(tiie28.fitted_curve)

x = np.arange(28, 10921)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), dpi=300)
ax1.plot(x, tiie28.rate_curve(x)) 
ax1.scatter(tiie28.days_to_pillar, tiie28.rate_curve(tiie28.days_to_pillar))
ax1.set_title("Discount Rate")

ax2.plot(x, 1 /( 1 + tiie28.rate_curve(x) * x / 360))
ax2.scatter(tiie28.days_to_pillar, 1 /( 1 + tiie28.rate_curve(tiie28.days_to_pillar) * tiie28.days_to_pillar / 360))
ax2.set_title("Discount Factor");

# Pricing the fixed rate swap
r, n = tiie28.market_coupons[-1], tiie28.number_flows[-1]
coupon_days = tiie28.length_flows(n).cumsum()

tau_k = tiie28.length_flows(n) / 360
rate_k = tiie28.rate_curve(coupon_days)
discount_factor = 1 / (1 + rate_k * coupon_days / 360)

# Fixed Leg: PV
fixed_pv = np.sum(r * tau_k * discount_factor)

# Variable leg: PV
variable_pv = 1 - discount_factor[-1]

fixed_pv

coupon_duration = tiie28.length_flows(390)
tau = coupon_duration / 360
time = coupon_duration.cumsum()

long_period = 1 / (1 + tiie28.rate_curve(time) * time / 360)
short_period = np.append(1, long_period[:-1])

fwd = (short_period / long_period - 1) / tau

plt.figure(figsize=(15,7), dpi=300)
plt.plot(time, fwd);

