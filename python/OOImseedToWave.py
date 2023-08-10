from obspy import read
from obspy.core import UTCDateTime
import wave

get_ipython().magic('matplotlib inline')
data_url = 'https://rawdata.oceanobservatories.org/files/RS01SLBS/LJ01A/09-HYDBBA102/2017/10/06/OO-HYVM1--YDH-2017-10-06T20:10:00.000015.mseed'
localFileName = 'OO-HYVM1--YDH-2017-10-06T20_10_00.000015.mseed'

loadFromOOI=True

if loadFromOOI==True :
    stream = read(data_url)
else:
    stream = read(localFileName)  # Read Previoulsy Download local file for speed

# print some stats about this signal
stream

#Documentation about the obspy library is here https://docs.obspy.org/contents.html
# and list of things you can do with a stream now that its loaded is here
#https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html

#plot the entire 5 minute clip
stream.plot()

#zoom in and look at just 5 seconds - The spikes every 1/sec are from a co-located ADCP

dt = UTCDateTime("2017-10-06T20:10:00")
st = stream.slice(dt, dt + 5)
print(st)  
st.plot()

st[0].spectrogram()  

#lets convert it to something easy to play on a PC

trace = stream[0].copy()
trace.filter('highpass', freq=2.0)

#convert to full scale and then make 32 bit

trace.normalize()
trace.data = (trace.data * (2**31-1)).astype('int32')
trace.plot()


#write it to an audio file that can it can be played in like Audacity

trace.write('test.wav', format='WAV', framerate=64000)



