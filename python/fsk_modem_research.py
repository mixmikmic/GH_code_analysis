# Imports and boilerplate to make graphs look better
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import wave
from IPython.display import Audio

def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

samp_rate = 1000
len_in_sec = 1
t = np.linspace(0, 1, samp_rate * len_in_sec)
hz_4 = 1*np.sin(4 * 2 * np.pi * t)
hz_8 = hz_4 * (2 * np.cos(4 * 2 * np.pi * t))
plt.plot(t, hz_4)
plt.show()
plt.plot(t, hz_8)
plt.show()

samp_rate = 1000  # samples/second
len_in_sec = 1
space_freq = 8  # Hz
t = np.linspace(0, 1, samp_rate * len_in_sec)
carrier = np.sin(space_freq * 2 * np.pi * t)

# Note: in FSK venaculare, "space" represents 0 and "mark" 1.
mark_multiplier_array = 2*np.cos(space_freq * 2 * np.pi * t)
modulation_array = np.array([1]*500 + [i for i in mark_multiplier_array[500:]])

setup_graph(title='modulation array for "01"', fig_size=(10,4))
plt.plot(modulation_array)
plt.margins(0.1)

setup_graph(title='modulated "01"', fig_size=(10,4))
plt.plot(carrier * modulation_array)
plt.margins(0.1)

samp_rate = 1000  # samples/second
len_in_sec = 1
space_freq = 8  # Hz ("space" = "0")
mark_freq = 16  # Hz ("mark" = "1")
t = np.linspace(0, .5, samp_rate * len_in_sec)
space = np.sin(space_freq * 2 * np.pi * t)
mark = np.sin(mark_freq * 2 * np.pi * t)
modulated_01 = np.append(space, mark)

setup_graph(title='modulation array for "01"', fig_size=(10,4))
plt.plot(modulated_01)
plt.margins(0.1)

def bfsk_modulate(bit_array, space_freq, mark_freq, baud, sample_rate):
    seconds_per_bit = 1 / baud
    samples_per_bit = sample_rate * seconds_per_bit
    t = np.linspace(0, seconds_per_bit, samples_per_bit)
    space = np.sin(space_freq * 2 * np.pi * t)
    mark = np.sin(mark_freq * 2 * np.pi * t)
    signal = np.array([])
    for bit in bit_array:
        if bit == 0:
            signal = np.append(signal, space)
        elif bit == 1:
            signal = np.append(signal, mark)
    return signal

sig_010110 = bfsk_modulate([0,1,0,1,1,0], 8, 16, 1, 1000)
setup_graph(title='bfsk-modulated "010110"', fig_size=(18,5))
plt.plot(sig_010110)
plt.margins(0.05)

sig_010110 = bfsk_modulate([0,1,0,1,1,0], 400, 800, 5, 3200)

from IPython.display import Audio
Audio(sig_010110, rate=3200)

import scipy
import scipy.io.wavfile

def write_audio_file(filename, filedata, sample_rate):
    scipy.io.wavfile.write(filename, sample_rate, filedata)

write_audio_file('raw_data/bfsk_010110.wav', sig_010110, 3200)

def fsk_modulate(bit_str, bit_freq_map, baud, sample_rate):
    seconds_per_bit = 1 / baud
    samples_per_bit = sample_rate * seconds_per_bit
    t = np.linspace(0, seconds_per_bit, samples_per_bit)
    
    # maps from bit sequence (like "10") to the modulated wave representing that "symbol"
    symbol_map = {bit_seq: np.sin(freq * 2 * np.pi * t) for bit_seq, freq in bit_freq_map.items()}

    signal = np.array([])
    bits_per_symbol = len(list(bit_freq_map.keys())[0])  # Assume all keys are the same length
    for symbol in [bit_str[i:i+bits_per_symbol] for i in range(0, len(bit_str), bits_per_symbol)]:
        symbol_wave = symbol_map[symbol]
        signal = np.append(signal, symbol_wave)
    return signal

bit_freq_map = {
    "00": 5,
    "01": 10,
    "10": 15,
    "11": 20
}

sig_00011011 = fsk_modulate("00011011", bit_freq_map, 1, 5000)

setup_graph(title='fsk-modulated "00011011"', fig_size=(18,5))
plt.plot(sig_00011011)
plt.margins(0.05)

bit_freq_map = {
    "00": 500,
    "01": 1000,
    "10": 1500,
    "11": 2000
}

sig_00011011 = fsk_modulate("00011011", bit_freq_map, 8, 5000)

setup_graph(title='fsk-modulated "00011011"', fig_size=(18,5))
plt.plot(sig_00011011)
plt.margins(0.05)

Audio(sig_00011011, rate=5000)

samp_rate = 1000
len_in_sec = 1
carrier_freq = 20  # Hz
t = np.linspace(0, 1, samp_rate * len_in_sec)
carrier = 1*np.sin(carrier_freq * 2 * np.pi * t)
plt.plot(t, carrier)

zero_crossings = np.where(np.diff(np.sign(carrier)))[0]

zero_crossings

len(zero_crossings)

# Note that in a single wave cycle, there will be 2 zero crossings
frequency_detected = len(zero_crossings) / 2
frequency_detected

hz_10 = 1*np.sin(10 * 2 * np.pi * t)
hz_20 = 1*np.sin(20 * 2 * np.pi * t)
hz_30 = 1*np.sin(30 * 2 * np.pi * t)
hz_40 = 1*np.sin(40 * 2 * np.pi * t)

[np.dot(carrier, hz_10), np.dot(carrier, hz_20), np.dot(carrier, hz_30), np.dot(carrier, hz_40)]

[np.dot(carrier, 1*np.sin(freq * 2 * np.pi * t)) for freq in [19.5, 19.6, 19.7, 19.8, 19.9, 20]]

def dot_between_freqs(f1, f2):
    t = np.linspace(0, 1, samp_rate * len_in_sec)
    f1_samples = 1*np.sin(f1 * 2 * np.pi * t)
    f2_samples = 1*np.sin(f2 * 2 * np.pi * t)
    return np.dot(f1_samples, f2_samples)

center_freq = 20
frequency_diffs = np.linspace(-10, 10, 500)
dots = [dot_between_freqs(center_freq, center_freq+d) for d in frequency_diffs]
setup_graph(title='frequency deviation vs dot product', x_label='frequency deviation (in Hz)', y_label='dot product', fig_size=(14,7))
plt.plot(frequency_diffs, dots)

center_freq = 20000
frequency_diffs = np.linspace(-10, 10, 500)
dots = [dot_between_freqs(center_freq, center_freq+d) for d in frequency_diffs]
setup_graph(title='frequency deviation vs dot product', x_label='frequency deviation (in Hz)', y_label='dot product', fig_size=(14,7))
plt.plot(frequency_diffs, dots)

fft_output = np.fft.rfft(carrier)
_ = plt.plot(fft_output)

len(carrier)

[np.abs(fft_output[10]), np.abs(fft_output[20]), np.abs(fft_output[30])]

fft_output100 = np.fft.rfft(carrier, n=100)
plt.plot(fft_output100)

[np.abs(fft_output100[1]), np.abs(fft_output100[2]), np.abs(fft_output100[3])]

aprs_msg = "W0HAK>NY5N:>Hello World'"
aprs_msg_bits = ''.join(["{0:b}".format(ord(c)).zfill(8) for c in aprs_msg])
aprs_msg_bits

bit_freq_map = {
    "0": 2200,
    "1": 1200
}
baud = 1200
sample_rate = 44100

aprs_msg_signal = fsk_modulate(aprs_msg_bits, {"0": 2200, "1": 1200}, baud=1200, sample_rate=44100)

len(aprs_msg_signal)

Audio(aprs_msg_signal, rate=44100)

def fsk_demodulate(raw_signal, bit_freq_map, baud, sample_rate):
    seconds_per_bit = 1 / baud
    samples_per_bit = int(sample_rate * seconds_per_bit)
    t = np.linspace(0, seconds_per_bit, samples_per_bit)
    
    # maps from bit sequence (like "10") to the modulated wave representing that "symbol"
    wave_to_symbol_map = {bit_seq: np.sin(freq * 2 * np.pi * t) for bit_seq, freq in bit_freq_map.items()}
    
    bit_str = ""
    for index in range(0, len(raw_signal), samples_per_bit):
        best_symbol = ""
        highest_dot_abs = 0
        for symbol, symbol_wave in wave_to_symbol_map.items():
            raw_window = raw_signal[index:index+samples_per_bit]
            dot_abs = np.abs(np.dot(symbol_wave[0:len(raw_window)], raw_window))
            if dot_abs > highest_dot_abs:
                best_symbol = symbol
                highest_dot_abs = dot_abs
        bit_str += best_symbol
    return bit_str

demodulated_aprs_msg = fsk_demodulate(aprs_msg_signal, bit_freq_map, baud, sample_rate)
demodulated_aprs_msg

demodulated_aprs_msg == aprs_msg_bits

(sample_rate, input_signal) = scipy.io.wavfile.read("raw_data/aprs_pocket_packet.wav")
time_array = np.arange(0, len(input_signal)/sample_rate, 1/sample_rate)

setup_graph(fig_size=(14,7))
_ = plt.plot(time_array, input_signal)

Audio(input_signal, rate=44100)

setup_graph(title='Spectrogram of aprs signal', x_label='time (in seconds)', y_label='frequency', fig_size=(14,8))
_ = plt.specgram(input_signal, Fs=44100)

signal_10000hz = [input_signal[i] for i in range(0, len(input_signal), 44100//10000)]
setup_graph(title='Spectrogram (10000Hz sample rate)', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
_ = plt.specgram(signal_10000hz, Fs=10000)

(sample_rate, captured_aprs_signal) = scipy.io.wavfile.read("raw_data/aprs_pocket_packet.wav")

data_time = 1.45-0.8
num_bits = data_time * 1200  # baud
num_bytes = num_bits / 8
num_bytes

demodulated_captured_aprs_msg = fsk_demodulate(captured_aprs_signal, bit_freq_map, baud, sample_rate)

demodulated_captured_aprs_msg

(sample_rate2, captured_aprs_signal2) = scipy.io.wavfile.read("raw_data/aprs_pocket_packet2.wav")
demodulated_captured_aprs_msg2 = fsk_demodulate(captured_aprs_signal2, bit_freq_map, 1200, sample_rate2)
demodulated_captured_aprs_msg2

demodulated_captured_aprs_msg == demodulated_captured_aprs_msg2

signal2_10000hz = [captured_aprs_signal2[i] for i in range(0, len(captured_aprs_signal2), 44100//10000)]
setup_graph(title='Spectrogram (10000Hz sample rate)', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
_ = plt.specgram(signal2_10000hz, Fs=10000)

demodulated_captured_aprs_msg == demodulated_captured_aprs_msg2

len(demodulated_captured_aprs_msg) == len(demodulated_captured_aprs_msg2)

index_mismatches = []

for index in range(len(demodulated_captured_aprs_msg)):
    if demodulated_captured_aprs_msg[index] != demodulated_captured_aprs_msg2[index]:
        index_mismatches.append(index)

index_mismatches

remodulated_aprs_msg = fsk_modulate(demodulated_captured_aprs_msg, bit_freq_map, 1200, 44100)

remodulated_10000hz = [remodulated_aprs_msg[i] for i in range(0, len(remodulated_aprs_msg), 44100//10000)]
setup_graph(title='Spectrogram (10000Hz sample rate)', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
_ = plt.specgram(remodulated_10000hz, Fs=10000)

Audio(remodulated_aprs_msg, rate=44100)



