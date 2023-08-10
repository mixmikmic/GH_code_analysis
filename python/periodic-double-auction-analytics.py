import pandas as pd

get_ipython().system('sbt "run-main PeriodicDoubleAuctionSimulation -Dsimulation.auction.clearing.interval=0.25 -Dsimulation.results.path=results.json"')

df = pd.read_json("./results.json")

df.shape

# gives you a sense of the range of variance in order flow spread across the different clearings...
df.T.count()

# raw json is messy, will need to pre-process the JSON before loading it into Pandas...
df.head()



