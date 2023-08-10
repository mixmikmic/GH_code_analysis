import json
with open("tickerInfo.json", 'r') as j:
    tickerInfo = json.load(j)

sector = {}
category = {}

for each in tickerInfo:
    if 'sector' in each:
        if each['sector'] not in sector:
            sector[each['sector']] = 1
        else:
            sector[each['sector']] += 1
    else:
        print(each)

sector

for each in tickerInfo:
    if 'category' in each:
        if each['category'] not in category:
            category[each['category']] = 1
        else:
            category[each['category']] += 1

category



