from pytrends.request import TrendReq

get_ipython().run_line_magic('matplotlib', 'inline')

# set a nice plotting style
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# with open('super_secrets.txt', 'r') as f:
#     username, password = f.read().splitlines() 

# connector = TrendReq(username, password)

# Enter your Google username and password

connector = TrendReq('username', 'password')

my_search_terms = ['sklearn']  # list with up to 5 items

connector.build_payload(my_search_terms)
df = connector.interest_over_time()

df.head()

df.plot()

get_ipython().run_line_magic('pinfo', 'connector.build_payload')

# Google Shopping searches for skis and snowboards in the US for last three years
connector.build_payload(['skis', 'snowboards'], geo='US', gprop='froogle')

df = connector.interest_over_time()
df.plot()

connector.build_payload(['whistler', 'snowbird'], geo='US')
df = connector.interest_over_time()
df.plot()

df_regional = connector.interest_by_region()
df_regional

df_top_states = df_regional.loc[df_regional.whistler > df_regional.whistler.quantile(0.75)]

df_top_states.plot.barh()

connector.build_payload(['sklearn', 'pandas', 'python'], cat=5)

related = connector.related_queries()

related.keys()

related['sklearn'].keys()

related['sklearn']['top'].head(10)

related['sklearn']['rising'].head(10)

connector.build_payload(['love', 'life', 'computer', 'software'])

df_benchmarks = connector.interest_over_time()

df_benchmarks.plot()

