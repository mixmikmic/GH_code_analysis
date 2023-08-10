import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from obspy.clients.fdsn import Client

client = Client("IRIS")

from obspy import UTCDateTime

t = UTCDateTime("2017-09-03_03:30:00")
st = client.get_waveforms("IC", "MDJ", "00", "BHZ", t, t + 5*60)
st.plot()  

st

inventory = client.get_stations(network="IC", station="MDJ")
inventory.plot()
plt.show()

inventory

t1 = UTCDateTime("2016-09-09_00:30:00")
st1 = client.get_waveforms("IC", "MDJ", "00", "BHZ", t1, t1 + 5*60)
st1.plot()  

label = "{network}.{station}.{location}.{channel}".format(**st.traces[0].meta)

plt.figure(figsize=(15,5))
plt.plot(st.traces[0], label="3 Sept 2017")
plt.plot(st1.traces[0], label="9 Sept 2016")
plt.text(-200, 2.5e6, label, size=20)
plt.legend()
plt.grid()
plt.show()

st = client.get_waveforms("IC", "MDJ", "*", "HHZ", t, t + 15*60)
st.plot()  

z = st.traces[0]
NFFT = 2048
Fs = st.traces[0].meta.sampling_rate  # Sample rate in Hz

plt.figure(figsize=(15, 5))
Pxx, freqs, bins, im = plt.specgram(z, NFFT=NFFT, Fs=Fs, noverlap=int(0.9*NFFT), cmap='viridis', vmin=-50, vmax=50)
plt.show()

