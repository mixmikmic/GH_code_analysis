import obspy

get_ipython().magic('matplotlib inline')

eclipse_sounds = obspy.read('../data/hydrophone.mseed')
eclipse_sounds_meta = obspy.read('../data/hydrophone.mseed',headonly=True)

eclipse_sounds_meta

eclipse_sounds.merge(fill_value='interpolate')

eclipse_sounds.write('merged_hydrophone.mseed')

eclipse_sounds.plot()



