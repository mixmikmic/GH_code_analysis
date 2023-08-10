from transcriptic import api
from transcriptic import Run, Container, Dataset
from transcriptic.analysis import spectrophotometry
from transcriptic.config import Connection
import pandas as pd
import matplotlib as plt
from IPython.display import display, Image, HTML, SVG

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina' # or 'svg'")
plt.style.use('ggplot')

api = Connection.from_file("~/.transcriptic")
api.update_environment(org_id="ellis_lab")

myRun = Run("r18ekx48p9fsm")

myRun.data

dsstart = myRun.data["Datasets"][0].data
dsfinal = myRun.data["Datasets"][8].data
ds = pd.concat([dsstart, dsfinal])
ds

fig = ds.plot(kind="bar",figsize=[12,7]) 
fig.set_ylabel("OD600 / Abs")
fig.set_xticklabels(("0 hours", "16 hours"))

myRun.containers

myPlate = Container('ct18enc333fhzt')
myPlate.aliquots[0:8]

myRun.instructions.head(10)

pipetteInstruction = myRun.instructions["Instructions"][7]

mon = pipetteInstruction.monitoring(data_type="pressure").plot(figsize=[15,7])
mon.set_ylabel("Pipette pressure")



