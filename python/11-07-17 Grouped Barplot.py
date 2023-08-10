get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

df = pd.DataFrame(data=[[2,6,4],[2,7,3],[9,2,3]], columns=['A', 'B', 'C'], index=['I','II','III'])

df.plot.bar()

df.plot.barh()

