import json
import pprint
import walkcompare as wc
get_ipython().magic('matplotlib inline')
pp = pprint.PrettyPrinter(indent=4)

with open('./data/parliamentary_committees.json') as f:
    data = json.load(f)
pp.pprint ({k: data[k] for k in list(data.keys())[:3]}) #show a sample of the json

values = [data['RNNR']['membership'], data['BILI']['membership']] # let's try with two values
names = [data['RNNR']['name'], data['BILI']['name']]
compare1 = wc.compare(values, names)

values = [data['RNNR']['membership'], data['BILI']['membership'], data['TRAN']['membership']] # let's try with THREE values
names = [data['RNNR']['name'], data['BILI']['name'], data['TRAN']['name']]
compare2 = wc.compare(values, names)


values = [y['membership'] for  y in data.values()] #Produce a List of membership lists
names = [q for q in data.keys()]  # the abbreviated names of the committees
compare = wc.compare(values, names)
compare.LABEL_BOTH_FACTORS = True #We want the names of both factors and non
compare.adjust = True #use a text adjuster for the plots
compare.plot_ca() #plot the data.  If you want to also send the image to file include a filename.



