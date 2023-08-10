def hello():
    return 'Hello Jupyter.'

hello()

# グラフ表示を有効化
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

df = pd.DataFrame([1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288])
df.plot()

fibo_df = pd.DataFrame([1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765])
fibo_df.plot()

df.describe()

fibo_df.describe()



