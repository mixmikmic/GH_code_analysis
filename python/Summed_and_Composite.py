get_ipython().system('ls -1')

from gt_apps import filter, maketime, expMap, expCube, evtbin, srcMaps

filter['rad'] = 15
filter['evclass'] = 2
filter['infile'] = "L1405221252264C652E7F67_PH00.fits"
filter['outfile'] = "3C279_front_filtered.fits"
filter['ra'] = 194.046527
filter['dec'] = -5.789312
filter['tmin'] = 239557417
filter['tmax'] = 255398400
filter['emin'] = 100
filter['emax'] = 100000
filter['zmax'] = 100
filter['convtype'] = 0

filter.run()

filter['rad'] = 15
filter['evclass'] = 2
filter['infile'] = "L1405221252264C652E7F67_PH00.fits"
filter['outfile'] = "3C279_back_filtered.fits"
filter['ra'] = 194.046527
filter['dec'] = -5.789312
filter['tmin'] = 239557417
filter['tmax'] = 255398400
filter['emin'] = 100
filter['emax'] = 100000
filter['zmax'] = 100
filter['convtype'] = 1

filter.run()

maketime['scfile'] = 'L1405221252264C652E7F67_SC00.fits'
maketime['filter'] = '(DATA_QUAL==1)&&(LAT_CONFIG==1)'
maketime['roicut'] = 'yes'
maketime['evfile'] = '3C279_front_filtered.fits'
maketime['outfile'] = '3C279_front_filtered_gti.fits'

maketime.run()

maketime['scfile'] = 'L1405221252264C652E7F67_SC00.fits'
maketime['filter'] = '(DATA_QUAL==1)&&(LAT_CONFIG==1)'
maketime['roicut'] = 'yes'
maketime['evfile'] = '3C279_back_filtered.fits'
maketime['outfile'] = '3C279_back_filtered_gti.fits'

maketime.run()

expCube['evfile'] = '3C279_front_filtered_gti.fits'
expCube['scfile'] = 'L1405221252264C652E7F67_SC00.fits'
expCube['outfile'] = '3C279_front_ltcube.fits'
expCube['dcostheta'] = 0.025
expCube['binsz'] = 1.0

expCube.run()

import numpy as np
15.*np.sqrt(2)

evtbin['evfile'] = '3C279_front_filtered_gti.fits'
evtbin['outfile'] = '3C279_front_CCUBE.fits'
evtbin['algorithm'] = 'CCUBE'
evtbin['nxpix'] = 100
evtbin['nypix'] = 100
evtbin['binsz'] = 0.2
evtbin['coordsys'] = 'CEL'
evtbin['xref'] = 194.046527
evtbin['yref'] =  -5.789312
evtbin['axisrot'] = 0
evtbin['proj'] = 'AIT'
evtbin['ebinalg'] = 'LOG'
evtbin['emin'] = 100
evtbin['emax'] = 100000
evtbin['enumbins'] = 30
evtbin['scfile'] = 'L1405221252264C652E7F67_SC00.fits'

evtbin.run()

evtbin['evfile'] = '3C279_back_filtered_gti.fits'
evtbin['outfile'] = '3C279_back_CCUBE.fits'
evtbin['algorithm'] = 'CCUBE'
evtbin['nxpix'] = 100
evtbin['nypix'] = 100
evtbin['binsz'] = 0.2
evtbin['coordsys'] = 'CEL'
evtbin['xref'] = 194.046527
evtbin['yref'] =  -5.789312
evtbin['axisrot'] = 0
evtbin['proj'] = 'AIT'
evtbin['ebinalg'] = 'LOG'
evtbin['emin'] = 100
evtbin['emax'] = 100000
evtbin['enumbins'] = 30
evtbin['scfile'] = 'L1405221252264C652E7F67_SC00.fits'

evtbin.run()

from GtApp import GtApp
expCube2 = GtApp('gtexpcube2','Likelihood')

expCube2.pars()

expCube2['infile'] = '3C279_front_ltcube.fits'
expCube2['cmap'] = 'none'
expCube2['outfile'] = '3C279_front_BinnedExpMap.fits'
expCube2['irfs'] = 'P7REP_SOURCE_V15::FRONT'
expCube2['nxpix'] = 300
expCube2['nypix'] = 300
expCube2['binsz'] = 0.2
expCube2['coordsys'] = 'CEL'
expCube2['xref'] = 194.046527
expCube2['yref'] = -5.789312
expCube2['axisrot'] = 0.0
expCube2['proj'] = 'AIT'
expCube2['ebinalg'] = 'LOG'
expCube2['emin'] = 100
expCube2['emax'] = 100000
expCube2['enumbins'] = 30

expCube2.run()

expCube2['infile'] = '3C279_front_ltcube.fits'
expCube2['cmap'] = 'none'
expCube2['outfile'] = '3C279_back_BinnedExpMap.fits'
expCube2['irfs'] = 'P7REP_SOURCE_V15::BACK'
expCube2['nxpix'] = 300
expCube2['nypix'] = 300
expCube2['binsz'] = 0.2
expCube2['coordsys'] = 'CEL'
expCube2['xref'] = 194.046527
expCube2['yref'] = -5.789312
expCube2['axisrot'] = 0.0
expCube2['proj'] = 'AIT'
expCube2['ebinalg'] = 'LOG'
expCube2['emin'] = 100
expCube2['emax'] = 100000
expCube2['enumbins'] = 30

expCube2.run()

srcMaps['scfile'] = 'L1405221252264C652E7F67_SC00.fits'
srcMaps['expcube'] = '3C279_front_ltcube.fits'
srcMaps['cmap'] = '3C279_front_CCUBE.fits'
srcMaps['srcmdl'] = '3C279_input_model_front.xml'
srcMaps['bexpmap'] = '3C279_front_BinnedExpMap.fits'
srcMaps['outfile'] = '3C279_front_srcMap.fits'
srcMaps['irfs'] = 'P7REP_SOURCE_V15::FRONT'

srcMaps.run()

srcMaps['scfile'] = 'L1405221252264C652E7F67_SC00.fits'
srcMaps['expcube'] = '3C279_front_ltcube.fits'
srcMaps['cmap'] = '3C279_back_CCUBE.fits'
srcMaps['srcmdl'] = '3C279_input_model_back.xml'
srcMaps['bexpmap'] = '3C279_back_BinnedExpMap.fits'
srcMaps['outfile'] = '3C279_back_srcMap.fits'
srcMaps['irfs'] = 'P7REP_SOURCE_V15::BACK'

srcMaps.run()

from BinnedAnalysis import *
from SummedLikelihood import *

like_f = binnedAnalysis(irfs='P7REP_SOURCE_V15::FRONT', 
                        expcube='3C279_front_ltcube.fits', 
                        srcmdl='3C279_input_model_front.xml',
                        optimizer='NEWMINUIT',
                        cmap='3C279_front_srcMap.fits',
                        bexpmap='3C279_front_BinnedExpMap.fits')

like_b = binnedAnalysis(irfs='P7REP_SOURCE_V15::BACK', 
                        expcube='3C279_front_ltcube.fits', 
                        srcmdl='3C279_input_model_back.xml',
                        optimizer='NEWMINUIT',
                        cmap='3C279_back_srcMap.fits',
                        bexpmap='3C279_back_BinnedExpMap.fits')

summed_like = SummedLikelihood()
summed_like.addComponent(like_f)
summed_like.addComponent(like_b)

summed_like.fit(0)

summed_like.model

