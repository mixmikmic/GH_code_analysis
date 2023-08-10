# Imports and preliminaries.  
get_ipython().run_line_magic('matplotlib', 'notebook')
import os
import sys

import django
import numpy as np
import quantities as pq
import matplotlib as mpl
import matplotlib.pyplot as plt

import owtests
from channelworm.fitter.initiators import Initiator
from neuronunit.tests.channel import IVCurvePeakTest
from neuronunit.models.channel import ChannelModel

# Setup access to the Django database
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "channelworm.web_app.settings"
)
django.setup()
from channelworm.ion_channel.models import GraphData

# Instantiate the model
channel_model_name = 'EGL-19.channel'
channel_id = 'ca_boyle'
channel_file_path = os.path.join(owtests.CW_HOME,'models','%s.nml' % channel_model_name)
model_name = channel_model_name.split('.')[0]

model = ChannelModel(channel_file_path,channel_index=0,name=model_name)

# Get the experiment data from ChannelWorm and instantiate the test
doi = '10.1083/jcb.200203055' # The DOI of the paper containing the data
fig = '2B' # The figure and panel of the data
sample_data = GraphData.objects.get(graph__experiment__reference__doi=doi, 
                                    graph__figure_ref_address=fig)
obs = list(zip(*sample_data.asarray())) 
observation = {'i/C':obs[1]*pq.A/pq.F, 'v':obs[0]*pq.mV}
cell_capacitance = 1e-13 * pq.F # Capacitance is arbitrary if IV curves are scaled.  
observation['i'] = observation['i/C']*cell_capacitance

test = IVCurvePeakTest(observation, scale=True)

# Judge the model output against the experimental data
score = test.judge(model)
score.summarize()
print("The score was computed according to '%s' with raw value %s and pass cutoff %s"     % (score.description,score.raw,test.converter.cutoff))
print('The scaling factor for the model IV curve was %.3g' % score.related_data['scale_factor'])

mpl.rcParams.update({'font.size':16, 'lines.linewidth':3})
score.plot()
plt.tight_layout()
#plt.savefig('/Users/rgerkin/Desktop/iv_curves.eps',format='eps')

# Cleanup
os.system('rm *.dat; rm *.xml')

