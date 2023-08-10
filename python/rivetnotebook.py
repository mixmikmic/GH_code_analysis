get_ipython().run_cell_magic('javascript', '', 'require.config({paths: {\n        vis: "http://cdnjs.cloudflare.com/ajax/libs/vis/4.17.0/vis",\n        yadage: "https://rawgit.com/lukasheinrich/yadage-ipython/master/yadagealpha"\n    }\n});')

import os
import shutil
from packtivity.statecontexts import poxisfs_context as statecontext
from yadage.yadagemodels import YadageWorkflow
from yadage.workflow_loader import workflow
from yadage.clihelpers import setupbackend_fromstring, prepare_workdir_from_archive
import logging
logging.basicConfig()

toplevel = 'https://raw.githubusercontent.com/lukasheinrich/weinberg-exp/master/example_yadage'

#clean up work directory
workdir = 'fromipython'
try:
    shutil.rmtree(workdir)
#     prepare_workdir_from_archive(workdir, '{}/input.zip'.format(toplevel))
except OSError:
    pass

#load workflow 

#load the JSON wflow spec
wflowspec = workflow('madgraph_rivet.yml','from-github/phenochain') 
#define root workdirectory in which data fragements will end up 
rootcontext = statecontext.make_new_context(workdir)  
#finally create a workflow object
wflow = YadageWorkflow.createFromJSON(wflowspec,rootcontext)



wflow.view().init({'nevents':1000,'rivet_analysis':'MC_GENERIC'})
#set up a backend that we will use
backend = setupbackend_fromstring('multiproc:4')
#options are: multiprocessing pool, ipython cluster, celery cluster

import yadage_widget
ui = yadage_widget.WorkflowWidget(wflow)
ui

from adage import rundag
rundag(wflow, update_interval = 1, backend = backend,  additional_trackers=[ui.adagetracker])

from IPython.display import Image
Image('{}/MC_GENERIC/E.png'.format(wflow.view().getSteps('plots')[0].result['plots']))



