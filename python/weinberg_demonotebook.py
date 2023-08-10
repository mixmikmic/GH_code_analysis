get_ipython().run_cell_magic('javascript', '', 'require.config({paths: {\n        vis: "http://cdnjs.cloudflare.com/ajax/libs/vis/4.17.0/vis",\n        yadage: "https://rawgit.com/lukasheinrich/yadage-ipython/master/yadagealpha"\n    }\n});')

import os
import shutil
from packtivity.statecontexts import poxisfs_context as statecontext
from yadage.yadagemodels import YadageWorkflow
from yadage.workflow_loader import workflow
from yadage.clihelpers import setupbackend_fromstring, prepare_workdir_from_archive
import logging
logging.basicConfig()

repolocation = 'https://raw.githubusercontent.com/lukasheinrich/weinberg-exp/master/example_yadage'

workdir = 'fromipython'
try:
    shutil.rmtree(workdir)
except OSError:
    pass
finally:
    prepare_workdir_from_archive(
        workdir,
        '{}/input.zip'.format(repolocation)
    )

#load the JSON wflow spec
wflowspec = workflow('rootflow.yml',repolocation) 


#define root workdirectory in which data fragements will end up 
rootcontext = statecontext.make_new_context(workdir)  

#finally create a workflow object
wflow = YadageWorkflow.createFromJSON(wflowspec,rootcontext)


#initialize workflow with parameters
wflow.view().init({
        'nevents':10000,
        'seeds':[1,2,3],
        'runcardtempl':'{}/init/run_card.templ'.format(os.path.realpath(workdir)),
        'proccardtempl':'{}/init/sm_proc_card.templ'.format(os.path.realpath(workdir)),
        'sqrtshalf':45,
        'polbeam1':0,
        'polbeam2':0
})

#set up a backend that we will use
backend = setupbackend_fromstring('multiproc:4') #options are: multiprocessing pool, ipython cluster, celery cluster

import yadage_widget
ui = yadage_widget.WorkflowWidget(wflow)
ui

from adage import rundag
rundag(wflow,
       update_interval = 1,
       backend = backend,
       additional_trackers=[ui.adagetracker]
)

ui.reset('madevent','/subchain/0')

wflow.view().getSteps('merge')[0].result

import json
with open(wflow.view().getSteps('merge')[0].result['jsonlinesfile']) as f:
    parsed = map(json.loads,f.readlines())

costhetas = []
for e in parsed:
    els = [p for p in e['particles'] if p['id'] == 11]
    mus = [p for p in e['particles'] if p['id'] == 13]
    assert len(mus) == 1
    assert len(els) == 1
    mu = mus[0]
    el = els[0]
    el_px, el_py, el_pz = [el[x] for x in ['px','py','pz']]
    mu_px, mu_py, mu_pz = [mu[x] for x in ['px','py','pz']]
    costheta = mu_pz/el_pz
    costhetas.append(costheta)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

_,_,_ = plt.hist(costhetas, bins = 100, histtype='stepfilled')

print 'Voila!'



