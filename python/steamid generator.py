import pandas as pd
import numpy as np
import io #we will need this when we will output to json or xml file 

steamids = pd.read_csv('../data/steamids.csv', header=0) #reads in csv file
list1 = list(steamids) #wouldn't let me print as a df, so i made it into a list
print list1

#converts the list into a single row dataframe
df1 = pd.DataFrame(np.array(list1).reshape(1, 14), list("A"))
print df1





