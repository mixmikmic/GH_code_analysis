# From pynq import the BaseOverlay, video, & audio

from pynq.lib.video import *
from pynq.lib.audio import *
from pynq.overlays.base import BaseOverlay

base = BaseOverlay('base.bit')
hdmi_out = base.video.hdmi_out
pAudio = base.audio

# Define a function to convert matplotlib figures to a numpy array to be loaded to the framebuffer
def fig2numpy ( fig ):

    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw ( )

    # Now we can save it to a numpy array.
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return buf    

Mode = VideoMode(1280,720,24)
hdmi_out.configure(Mode)
hdmi_out.start()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
#import time

# Setup the figure to match the HDMI resolution
fig = plt.figure(figsize=(1280/96, 720/96), dpi=96)

# Continue sampling microphone until PB0 is pressed
while (base.buttons[0].read()==0):

  # Sample microphone
  pAudio.record(0.06773)

  # The following was taken from the base/audio example
  af_uint8 = np.unpackbits(pAudio.buffer.astype(np.int16)
                         .byteswap(True).view(np.uint8))
  af_dec = signal.decimate(af_uint8,8,zero_phase=True)
  af_dec = signal.decimate(af_dec,6,zero_phase=True)
  af_dec = signal.decimate(af_dec,2,zero_phase=True)
  af_dec = (af_dec[10:-10]-af_dec[10:-10].mean())

  del af_uint8

  time_axis = np.arange(0,((len(af_dec))/32000),1/32000)
  
  # Plot the time domain
  plt.subplot(211)
  plt.cla()
  plt.title('Audio Signal in Time & Frequency Domain')
  plt.xlabel('Time in s')
  plt.ylabel('Amplitude')
  plt.ylim((-0.025, 0.025))
  # Truncate beginning and end  
  plt.plot(time_axis[50:-50], af_dec[50:-50])

  # Take the FFT
  yf = fft(af_dec[50:-50])
  yf_2 = yf[1:len(yf)//2]
  xf = np.linspace(0.0, 32000//2, len(yf_2))

  # Plot the freq domain
  plt.subplot(212)
  plt.cla()
  plt.semilogx(xf, abs(yf_2))
  plt.xlabel('Frequency in Hz')
  plt.ylabel('Magnitude')
  plt.ylim((0, 15))

  # Convert figure to numpy array
  buf = fig2numpy (fig)

  # Send the image out the framebuffer
  outframe = hdmi_out.newframe()
  outframe[:] = buf
  hdmi_out.writeframe(outframe)

hdmi_out.stop()
del hdmi_out



