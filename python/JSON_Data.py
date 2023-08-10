import json
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

with open("datasets/loans.txt", "r") as f:
    loans = json.load(f)

loans

type(loans)

for tags in loans:
    print(tags)

result = pd.io.json.json_normalize(loans, "data")

result.head(2).transpose()

def flatten_json(input_doc):
    out = {} # empty dictionary
    
    def flatten(x, name = ""):
        if type(x) is dict:
            for a in x:
                flaten(x[a], name + a + '_')
        elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + "_")
                    i += 1
        else:
                    out[name[:-1]] = x

    flatten(input_doc)
    return out

type(loans["data"][0])

type(loans["data"][0][21])

flatten_json(loans["data"][0])

final_data = pd.DataFrame()

for i in np.arange(0, len(loans["data"])):
    final_data = final_data.append(pd.DataFrame(json_normalize(flatten_json(loans["data"][i]))))

final_data.head()

final_data[:1]

final_data.drop(final_data.columns[[0,1,12,14,15,16,17,18,19,20,21,22,23]],inplace=True,axis=1)
# df.columns[[0,1,12,14,15,16,17,18,19,20,21,22,23]]
final_data[:1]

len(loans["meta"]["view"]["columns"])

names = []

length = len(loans["meta"]["view"]["columns"])

for i in range(0, length):
    names.append(loans["meta"]["view"]["columns"][i]["fieldName"])
    
names

names = [word for word in names if not ":" in word]
names

names.pop(len(names) - 1)

final_data.rename(index=str, columns={'8': names[0], '9': names[1], '10': names[2], '11': names[3], '12': names[4], '13': names[5],                              '14':names[6], '15': names[7], '16': names[8], '17': names[9], '18': names[10],                               '19':names[11], '20': names[12]},inplace=True)
final_data[:5]



