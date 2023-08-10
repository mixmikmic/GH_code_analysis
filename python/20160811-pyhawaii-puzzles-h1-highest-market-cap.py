import csv

from itertools import islice

from collections import defaultdict

# Works, but if/else stuff is ugly.

filename = 'AMEX_daily_prices_N.csv'
max_market_caps = {}

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # print(row)
        company = row['stock_symbol']
        volume = float(row['stock_volume'])
        price = float(row['stock_price_close'])
        today_market_cap = volume * price
        # print(company, volume, price, today_market_cap)
        if company in max_market_caps:
            if today_market_cap > max_market_caps[company]:
                max_market_caps[company] = today_market_cap
        else:
            max_market_caps[company] = today_market_cap

# print(max_market_caps)
for company, max_market_cap in sorted(
        max_market_caps.items(),
        key=lambda item: item[1])[-3:]:
    print(company, max_market_cap)
print(max(max_market_caps.values()))

# Used .get() method of dictionary to avoid if/else
# and make code more readable.

filename = 'AMEX_daily_prices_N.csv'
max_market_caps = {}

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # print(row)
        company = row['stock_symbol']
        volume = float(row['stock_volume'])
        price = float(row['stock_price_close'])
        today_market_cap = volume * price
        # print(company, volume, price, today_market_cap)
        if today_market_cap > max_market_caps.get(company, 0.):
            max_market_caps[company] = today_market_cap

# print(max_market_caps)
for company, max_market_cap in sorted(
        max_market_caps.items(),
        key=lambda item: item[1])[-3:]:
    print(company, max_market_cap)
print(max(max_market_caps.values()))

# Used max() to get rid of the if statement altogether.
# Now the ugliest thing is the .get() method call.

filename = 'AMEX_daily_prices_N.csv'
max_market_caps = {}

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # print(row)
        company = row['stock_symbol']
        volume = float(row['stock_volume'])
        price = float(row['stock_price_close'])
        today_market_cap = volume * price
        # print(company, volume, price, today_market_cap)
        max_market_caps[company] = max(
            max_market_caps.get(company, 0.), today_market_cap)
# print(max_market_caps)
for company, max_market_cap in sorted(
        max_market_caps.items(),
        key=lambda item: item[1])[-3:]:
    print(company, max_market_cap)
print(max(max_market_caps.values()))

# Used defaultdict to avoid need for .get() method call.
# That's as readable as I know how to make that part of the code.

filename = 'AMEX_daily_prices_N.csv'
max_market_caps = defaultdict(float)

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # print(row)
        company = row['stock_symbol']
        volume = float(row['stock_volume'])
        price = float(row['stock_price_close'])
        today_market_cap = volume * price
        # print(company, volume, price, today_market_cap)
        max_market_caps[company] = max(
            max_market_caps[company], today_market_cap)
# print(max_market_caps)
for company, max_market_cap in sorted(
        max_market_caps.items(),
        key=lambda item: item[1])[-3:]:
    print(company, max_market_cap)
print(max(max_market_caps.values()))

filename = 'AMEX_dividends_N.csv'

earning_companies = set()

with open(filename) as f:
    reader = csv.DictReader(f)
    for row in reader:
        # print(row)
        company = row['stock_symbol']
        dividends = float(row['dividends'])
        if dividends > 0:
            earning_companies |= {company}
    
print('dividend companies:', earning_companies)
print(
    'dividend companies without market cap:',
    earning_companies - set(max_market_caps))
print(
    'non-dividend companies:',
    set(max_market_caps) - earning_companies)

max(
    max_market_caps[company]
    for company in earning_companies & set(max_market_caps))

best_companies = []
highest_market_cap = 0.
for company, market_cap in max_market_caps.items():
    if company not in earning_companies:
        continue
    if market_cap > highest_market_cap:
        highest_market_cap = market_cap
        best_companies = [company]
    elif market_cap == highest_market_cap:
        best_companies.append(company)

print(best_companies, highest_market_cap)

