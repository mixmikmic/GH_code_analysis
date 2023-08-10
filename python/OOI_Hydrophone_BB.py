from obspy import read
from obspy.core import UTCDateTime
import wave
import datetime
import matplotlib
from ipywidgets import widgets
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from IPython.display import Audio

get_ipython().magic('matplotlib inline')
data_url = 'https://rawdata.oceanobservatories.org/files/RS01SLBS/LJ01A/09-HYDBBA102/2017/10/06/OO-HYVM1--YDH-2017-10-06T20:10:00.000015.mseed'
localFileName = '../data/merged_hydrophone.mseed'

loadFromOOI=False

if loadFromOOI==True :
    stream = read(data_url)
else:
    stream = read(localFileName)  # Read Previoulsy Download local file for speed

# print some stats about this signal
stream

def getSamplingFreq(s):
    string = str(s).split(" ")
    fs = 0;
    for i in range(0, len(string) - 1) :
        char = string[i] 
        count = 1
        if (char == "|"):
            count = count + 1
        if (count == 2):
            fs = string[i + 1]
    return float(fs)

def playSound(sound):
    sound = sound.copy()
    sound.normalize()
    sound.data = (sound.data * (2**31-1)).astype('int32')
    Fs = getSamplingFreq(sound)
    samplerate = 4*Fs;
    #Audio(data=sound, rate=samplerate)
    sd.play(sound, samplerate)
    
def getSlice(stream, Num):
    dt = UTCDateTime("2017-08-21T09:00:00")
    print(dt)
    st = stream.slice(dt + Num*10, dt + Num*10 + 10) 
    return st

def spec(Num):
    st = getSlice(stream, Num)
    #x=np.array([row[Number] for row in stream])
    fig, ax = plt.subplots()
    pxx, freq, t, cax = ax.specgram(st[0],Fs=64000,noverlap=5, cmap='plasma')
    # c axis
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity dB')
    ax.axis("tight")
    # y axis   
    ax.set_ylabel('Frequency [kHz]')
    scale = 1e3                     # KHz
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)
    # x axis 
    ax.set_xlabel('Time h:mm:ss')
    def timeTicks(x, pos):
        d = UTCDateTime("2017-08-21T09:00:00").second+Num*30+x
        #d = datetime.timedelta(seconds=x)
        return str(d)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)
    # shows spectrogram
    plt.show()
    # time plot
    st.plot()
    # plays recording
    playSound(st[0])
    
widgets.interact(spec, Num=widgets.IntSlider(min=0,max=10,value=0,step =1,continuous_update=False))

stream[0].stats



