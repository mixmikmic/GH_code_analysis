import shapefile
import datetime

sf = shapefile.Reader("/Users/jeriwieringa/Dissertation/drafts/data/external-data/citiesx020_nt00007/citiesx020.shp")

sf.fields

records = sf.records()

len(records)

records[:3]

placenames = []
for each in records:
    placenames.append(each[2])

len(placenames)

placenames[:10]

with open("/Users/jeriwieringa/Dissertation/drafts/data/word-lists/{}-place-names.txt".format(str(datetime.date.today())), "w") as outfile:
    for name in placenames:
        if len(name.split()) > 1:
            words = name.split()
            for word in words:
                outfile.write("{}\n".format(word.lower()))
        else:
            outfile.write("{}\n".format(name.lower()))

# %load shared_elements/system_info.py
import IPython
print (IPython.sys_info())
get_ipython().system('pip freeze')



