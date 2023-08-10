import os
import sys

sys.path.append('../../')
sys.path.append('/usr/local/bin')

import numpy as np
from pygeo.segyread import SEGYFile
from pygeo.fullpy import readini
# from pymatsolver import MumpsSolver
from zephyr.backend import MiniZephyrHD, EurusHD, SparseKaiserSource
from zephyr.middleware import Helm2DViscoProblem, Helm2DSurvey, FullwvDatastore
from zephyr.middleware import dftreal, idftreal, dwavelet, TimeMachine

fds = FullwvDatastore('xhlayr')
systemConfig = fds.systemConfig

systemConfig.update({
    'Disc':     MiniZephyrHD,
#     'Solver':   MumpsSolver,
})

TM = TimeMachine(systemConfig)
STF = TM.fSource(TM.keuper())
systemConfig['sterms'] = STF.ravel()

problem = Helm2DViscoProblem(systemConfig)
survey  = Helm2DSurvey(systemConfig)
problem.pair(survey)

print('System Wrapper:\t%s'%problem.SystemWrapper)
print('Discretization:\t%s'%problem.system.Disc)
print('RHS Generator: \t%s'%survey.rhsGenerator.__class__)

uF = problem.fields()[:]['u']

uFs = uF[:,0,:]
res = TM.idft(uFs)

get_ipython().magic('pylab inline')
imshow(res[:,20].reshape((200,100)), cmap=cm.bwr)

get_ipython().run_cell_magic('time', '', "\nprint('Running %(nfreq)d frequencies and %(nsrc)s sources'%{'nfreq': survey.nfreq, 'nsrc': survey.nsrc})\n\ndPred = survey.dpred().reshape((survey.nrec, survey.nsrc, survey.nfreq))")

fds.utoutWrite(dPred)



