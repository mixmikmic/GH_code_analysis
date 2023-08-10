import graphlab
from graphlab import SFrame

data =  SFrame.read_csv("turnstile_160319.txt",
                        column_type_hints=[str,str,str,str,str,str,str,str,str,int,int])



data.show()



data.column_names()

#data['TIME'] = data['TIME'].astype("datetime.datetime")
#data['DATE '] = data['DATE'].astype("datetime.datetime")

dir(data['DATE'])



