# Grab all of the NumPy functions with namespace (nickname) np
import numpy as np              

# Set up the notebook to display plots inline
get_ipython().magic('matplotlib inline')

# Import the plotting functions 
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 5001)   # Time, 0-5s with 5001 samples in the range

# Define the frequency and amplitude of the three components of the signal
A1 = 1.0                  # amplitude of first sine
w1 = 1.0 * (2 * np.pi)    # frequency of first sine (rad/s)
A2 = 0.5                  # amplitude of second sine
w2 = 2.75 * (2 * np.pi)   # frequency of second sine (rad/s)
A3 = 1.5                  # amplitude of third sine
w3 = 7.5 * (2 * np.pi)    # frequency of third sine (rad/s)

orig_signal = A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) + A3 * np.sin(w3 * t)

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Signal Amplitude', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(t, orig_signal, linewidth=2, linestyle='-', label=r'Original Signal')

# uncomment below and set limits if needed
# plt.xlim(0, 5)
# plt.ylim(0, 10)

# # Create the legend, then fix the fontsize
# leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
# ltext  = leg.get_texts()
# plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('OriginalSignal.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

def CRAWLAB_fft(data, time, plotflag):
    ''' Function to get the FFT for a response
    #
    # Inputs:
    #   time = time array corresponding to the data
    #   data = the response data array (only pass a single dimension/state at at time)
    #   plotflag = will plot the FFT if nonzero
    #   
    # Output:
    #   fft_freq = an array of the freqs used in the FFT
    #   fft_mag = an array of the amplitude of the FFT at each freq in fft_freq
    #
    # Created: 03/28/14
    #   - Joshua Vaughan
    #   - joshua.vaughan@louisiana.edu
    #   - http://www.ucs.louisiana.edu/~jev9637
    ######################################################################################
    '''
    
    from scipy.fftpack import fft
    
    # correct for any DC offset
    offset = np.mean(data) 

    # Get the sampling time
    sample_time = time[1] - time[0]
    
    # Get the length of the dataset
    n = len(data)

    # Calculate the FFT of the data, removing the offset and using a Hanning Window
    fft_mag = fft((data - offset) * np.hanning(len(data)))
    
    # Define the frequency range of the output
    fft_freq = np.linspace(0.0, 1.0 / (2.0*sample_time), int(np.ceil(n/2)))
    
    # Only return the "useful" part of the fft
    fft_mag = 2.0/n * np.abs(fft_mag[0:int(np.ceil(n/2))])
    
    # If plotflag is nonzero (True), plot the FFT before returning the magnitude and phase
    if plotflag:
        # Plot the relationshiop
        #   Many of these setting could also be made default by the .matplotlibrc file
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
        plt.setp(ax.get_ymajorticklabels(),fontsize=18)
        plt.setp(ax.get_xmajorticklabels(),fontsize=18)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.grid(True, linestyle=':', color='0.75')
        ax.set_axisbelow(True)

        plt.xlabel('Frequency (Hz)', fontsize=22, labelpad=8)
        plt.ylabel('FFT magnitude', fontsize=22, labelpad=10)
    
        plt.plot(fft_freq, fft_mag, linewidth=2, linestyle='-')
        
        # Adjust the page layout filling the page using the new tight_layout command
        plt.tight_layout(pad=0.5)
        plt.show()
    
    return fft_freq, fft_mag

# We can also use the FFT to get the natrual frequency
freq, mag = CRAWLAB_fft(orig_signal, t, False)

# Plot the relationshiop
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),fontsize=18)
plt.setp(ax.get_xmajorticklabels(),fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel('Frequency (Hz)',fontsize=22,labelpad=8)
plt.ylabel('FFT magnitude',fontsize=22,labelpad=10)


plt.plot(freq, mag, linewidth=2, linestyle='-', label = 'FFT')

# Let's also annotate the plot with vertical lines at each of the frequencies in our origianl signal
plt.plot([w1 / (2 * np.pi), w1 / (2 * np.pi)], [0, 1.25 * np.max(mag)], linewidth = 2, linestyle = '--', label = r'$\omega_1$')
plt.plot([w2 / (2 * np.pi), w2 / (2 * np.pi)], [0, 1.25 * np.max(mag)], linewidth = 2, linestyle = '-.', label = r'$\omega_2$')
plt.plot([w3 / (2 * np.pi), w3 / (2 * np.pi)], [0, 1.25 * np.max(mag)], linewidth = 2, linestyle = ':', label = r'$\omega_3$')

plt.xlim(0, 10)
plt.ylim(0, 1.25 * np.max(mag))

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('FFT_magnitude_5s.pdf',dpi=600)

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

t = np.linspace(0, 25, 5001)   # Time, 0-5s with 5001 samples in the range

# Now, define the 25s signal, using the same amplitudes and frequencies as the 5s version
orig_signal_25s = A1 * np.sin(w1 * t) + A2 * np.sin(w2 * t) + A3 * np.sin(w3 * t)

# Calculate the FFT for the 25s signal
freq_25s, mag_25s = CRAWLAB_fft(orig_signal_25s, t, False)

# Plot the relationshiop
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),fontsize=18)
plt.setp(ax.get_xmajorticklabels(),fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel('Frequency (Hz)', fontsize=22, labelpad=8)
plt.ylabel('FFT magnitude', fontsize=22, labelpad=10)


plt.plot(freq_25s, mag_25s, linewidth=2, linestyle='-', label = 'FFT')

# Let's also annotate the plot with vertical lines at each of the frequencies in our origianl signal
plt.plot([w1 / (2 * np.pi), w1 / (2 * np.pi)], [0, 1.25 * np.max(mag_25s)], linewidth = 2, linestyle = '--', label = r'$\omega_1$')
plt.plot([w2 / (2 * np.pi), w2 / (2 * np.pi)], [0, 1.25 * np.max(mag_25s)], linewidth = 2, linestyle = '-.', label = r'$\omega_2$')
plt.plot([w3 / (2 * np.pi), w3 / (2 * np.pi)], [0, 1.25 * np.max(mag_25s)], linewidth = 2, linestyle = ':', label = r'$\omega_3$')

plt.xlim(0, 10)
plt.ylim(0, 1.25 * np.max(mag_25s))

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 1, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('FFT_magnitude.pdf',dpi=600)

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

noisy_signal = orig_signal_25s +  np.random.randn(len(orig_signal_25s))

# Set the plot size - 3x2 aspect ratio is best
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17, left=0.17, top=0.96, right=0.96)

# Change the axis units to serif
plt.setp(ax.get_ymajorticklabels(), family='serif', fontsize=18)
plt.setp(ax.get_xmajorticklabels(), family='serif', fontsize=18)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Turn on the plot grid and set appropriate linestyle and color
ax.grid(True,linestyle=':', color='0.75')
ax.set_axisbelow(True)

# Define the X and Y axis labels
plt.xlabel('Time (s)', family='serif', fontsize=22, weight='bold', labelpad=5)
plt.ylabel('Signal Amplitude', family='serif', fontsize=22, weight='bold', labelpad=10)

plt.plot(t, orig_signal_25s, linewidth=2, linestyle='-', label=r'Original Signal')
plt.plot(t, noisy_signal, linewidth=2, linestyle='--', label=r'Noisy Signal')

# uncomment below and set limits if needed
plt.xlim(0, 3)
# plt.ylim(0, 10)

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 2, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# Uncomment to save the figure as a high-res pdf in the current folder
# It's saved at the original 6x4 size
# plt.savefig('OriginalSignal.pdf')

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# Calculate the FFT for the noisy 25s signal
freq_noisy, mag_noisy = CRAWLAB_fft(noisy_signal, t, False)

# Plot the relationshiop
#   Many of these setting could also be made default by the .matplotlibrc file
fig = plt.figure(figsize=(6,4))
ax = plt.gca()
plt.subplots_adjust(bottom=0.17,left=0.17,top=0.96,right=0.96)
plt.setp(ax.get_ymajorticklabels(),fontsize=18)
plt.setp(ax.get_xmajorticklabels(),fontsize=18)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.grid(True,linestyle=':',color='0.75')
ax.set_axisbelow(True)

plt.xlabel('Frequency (Hz)', fontsize=22, labelpad=8)
plt.ylabel('FFT magnitude', fontsize=22, labelpad=10)


plt.plot(freq_25s, mag_25s, linewidth=2, linestyle='-', label = 'FFT - Original')
plt.plot(freq_noisy, mag_noisy, linewidth=2, linestyle='--', label = 'FFT - Noisy')

# Let's also annotate the plot with vertical lines at each of the frequencies in our origianl signal
plt.plot([w1 / (2 * np.pi), w1 / (2 * np.pi)], [0, 1.5 * np.max(mag_25s)], linewidth = 2, linestyle = '--', label = r'$\omega_1$')
plt.plot([w2 / (2 * np.pi), w2 / (2 * np.pi)], [0, 1.5 * np.max(mag_25s)], linewidth = 2, linestyle = '-.', label = r'$\omega_2$')
plt.plot([w3 / (2 * np.pi), w3 / (2 * np.pi)], [0, 1.5 * np.max(mag_25s)], linewidth = 2, linestyle = ':', label = r'$\omega_3$')

plt.xlim(0, 10)
plt.ylim(0, 1.5 * np.max(mag_25s))

# Create the legend, then fix the fontsize
leg = plt.legend(loc='upper right', ncol = 3, fancybox=True)
ltext  = leg.get_texts()
plt.setp(ltext, family='serif', fontsize=20)

# Adjust the page layout filling the page using the new tight_layout command
plt.tight_layout(pad=0.5)

# save the figure as a high-res pdf in the current folder
# plt.savefig('FFT_magnitude.pdf',dpi=600)

fig.set_size_inches(9, 6) # Resize the figure for better display in the notebook

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

