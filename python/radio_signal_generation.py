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

SAMPLE_BITSIZE = 16
MAX_AMP_16BIT = int(2**SAMPLE_BITSIZE/2 - 1)

def generate_wave(freq, len_in_sec=1, samp_rate=44100, amplitude=MAX_AMP_16BIT):
    t = np.linspace(0, len_in_sec, samp_rate * len_in_sec)
    sig = amplitude * np.sin(freq * 2 * np.pi * t) 
    return sig

def write_wav_file(file_path, wav_data, sample_rate=44100, num_channels=1):
    f = wave.open(file_path, 'wb')
    f.setparams((num_channels, 2, sample_rate, len(wav_data), "NONE", "Uncompressed"))
    f.writeframes(np.array(wav_data, dtype=np.int16))
    f.close()

def write_pcm_file(signal_data, file_path, dtype='complex64'):
    np.array(signal_data).astype(dtype).tofile(file_path)

samp_rate = 1000
len_in_sec = 1
carrier_freq = 20

low_amp = 0.1
high_amp = 1

t = np.linspace(0, 1, samp_rate * len_in_sec)
carrier = 1*np.sin(carrier_freq * 2 * np.pi * t)

# Modulate with the binary signal: ['0', '1']
amp_mult = np.array([low_amp]*500 + [high_amp]*500)
sig = amp_mult * carrier

setup_graph(title='sig', x_label='time', y_label='freq', fig_size=(12,6))
plt.plot(t, sig)

SAMPLE_BITSIZE = 16
MAX_AMP_16BIT = int(2**SAMPLE_BITSIZE/2 - 1)

DEFAULT_RATIOS = {
    '_': 1,
    '0': 1,
    '1': 3
}

DEFAULT_AMP_MAP = {
    '0': MAX_AMP_16BIT * .02,
    '1': MAX_AMP_16BIT
}

def get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype=np.int16):
    data_points_in_bit = int(sample_rate * 1/baud)
    modulation_array = np.array([], dtype=dtype)

    for bit in binary_data:
        bit_amplitude = amp_map[bit]
        modulated_bit = np.full(data_points_in_bit, bit_amplitude, dtype=np.int16)
        modulation_array = np.append(modulation_array, modulated_bit)
    return modulation_array

def generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate,
                               baud, sig_ratios=DEFAULT_RATIOS, amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = len(binary_data) * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    carrier_wave = 1 * np.sin(carrier_wave_freq * 2 * np.pi * t)
    modulation_array = get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype)
    return t, carrier_wave * modulation_array
    

binary_data = '0110100010000000'
carrier_wave_freq = 20 #315e6
sample_rate = 100
baud = 1

t, sig = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud)
setup_graph(title='sig', x_label='time', y_label='freq', fig_size=(12,6))
plt.plot(t, sig)

DEFAULT_AMP_MAP = {
    'LOW': MAX_AMP_16BIT * .02,
    'HIGH': MAX_AMP_16BIT
}

def get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype=np.int16):
    data_points_in_bit = int(sample_rate * 1/baud)
    modulation_array = np.array([], dtype=dtype)

    # To describe this general algorithms, I'll use the specific concrete pulse ratios:
    #    '_': 1,
    #    '0': 1,
    #    '1': 3
    # Meaning that a "1" should be 3x longer than a "0" or a "space" pulse. Now since we need a space
    # between "1"s (as well as "0"), we can calculate that the pulse for a "1" should be 3/4 of the bit
    # and the pulse for a "0" should be 1/4 of the bit (since for the 1, it's 3 parts "1" and 1 part "space")
    one_pulse_len = int((sig_ratios['1'] / (sig_ratios['1'] + sig_ratios['_'])) * data_points_in_bit)
    one_space_len = data_points_in_bit - one_pulse_len
    zero_pulse_len = int((sig_ratios['0'] / (sig_ratios['1'] + sig_ratios['_'])) * data_points_in_bit)
    zero_space_len = data_points_in_bit - zero_pulse_len
    
    modulated_one_bit = np.append(np.full(one_pulse_len, amp_map['HIGH'], dtype=dtype),
                                  np.full(one_space_len, amp_map['LOW'], dtype=dtype))
    modulated_zero_bit = np.append(np.full(zero_pulse_len, amp_map['HIGH'], dtype=dtype),
                                   np.full(zero_space_len, amp_map['LOW'], dtype=dtype))
    
    for bit in binary_data:
        modulated_bit = modulated_one_bit if bit == '1' else modulated_zero_bit
        modulation_array = np.append(modulation_array, modulated_bit)
    return modulation_array

def generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate,
                               baud, sig_ratios=DEFAULT_RATIOS, amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = len(binary_data) * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    carrier_wave = 1 * np.sin(carrier_wave_freq * 2 * np.pi * t)
    modulation_array = get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype)
    
    # Pad (or trim) the modulation array to match the length of the carrier wave
    if len(carrier_wave) > len(modulation_array):
        pad_len = len(carrier_wave) - len(modulation_array)
        modulation_array = np.append(modulation_array, np.full(pad_len, amp_map['LOW'], dtype=dtype))
    elif len(carrier_wave) < len(modulation_array):
        modulation_array = modulation_array[:len(carrier_wave)]
    
    return t, carrier_wave * modulation_array

binary_data = '0110100010000000'
carrier_wave_freq = 20 #315e6
sample_rate = 1000
baud = 1

t, sig = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud)
setup_graph(title='sig', x_label='time', y_label='freq', fig_size=(14,7))
plt.plot(t, sig)

# Estimating baud

bits_transmitted = 16
viewing_sample_rate = 100000  # samples per second
real_sample_rate = 2000000  # samples per second
viewing_transmission_time = 8.219-6.664  # seconds
real_transmission_time = viewing_transmission_time * (viewing_sample_rate / real_sample_rate)
baud = bits_transmitted / real_transmission_time # bits per second
print('Real transmission time: {}\nBaud: {}'.format(real_transmission_time, baud))

binary_data = '0110100010000000'
carrier_wave_freq = 315e6
sample_rate = 2e6
baud = 205

t, sig = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud)
setup_graph(title='sig', x_label='time', y_label='freq', fig_size=(14,7))
plt.plot(t, sig)

write_pcm_file(sig, 'raw_data/generated_sig1.pcm', dtype='int16')

# Generate preamble and repeat pattern
binary_data = '0110100010000000'
carrier_wave_freq = 315e6
sample_rate = 2e6
baud = 205

def generate_pulse(bit_val, carrier_wave_freq, sample_rate, baud, multiple_of_bit_len,
                   amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = multiple_of_bit_len * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    high_or_low = 'HIGH' if bit_val == '1' else 'LOW'
    pulse = amp_map[high_or_low] * np.sin(carrier_wave_freq * 2 * np.pi * t)
    return t, pulse


t1, signal_header = generate_pulse('1', carrier_wave_freq, sample_rate, baud, 3.85)
t2, signal_spacer = generate_pulse('0', carrier_wave_freq, sample_rate, baud, 3.78)
setup_graph(title='sig', x_label='time', y_label='freq', fig_size=(14, 7))
plt.plot(np.append(signal_spacer, signal_header))

def join_all_arrays(array_list):
    joined = array_list[0]
    for a in array_list[1:]:
        joined = np.append(joined, a)
    return joined

full_signal = join_all_arrays([signal_header] + ([sig, signal_spacer] * 12))

setup_graph(title='sig', x_label='samples', y_label='freq', fig_size=(14, 7))
plt.plot(full_signal)

write_pcm_file(full_signal, 'raw_data/generated_sig2.pcm', dtype='int16')

t = np.linspace(0, 1, 1000)
amp = 3
freq = 10 # Hz
simple_sig = amp * np.cos(freq * 2 * np.pi * t)

plt.plot(t, simple_sig)

import cmath
complex_sig = 3 * np.e**(freq * 2 * np.pi * (0+1j) * t)
plt.plot(t, complex_sig)

simple_sig[5]

complex_sig[5]

abs(complex_sig[5])  # This is the amplitude of the wave

def generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate,
                               baud, sig_ratios=DEFAULT_RATIOS, amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = len(binary_data) * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    # Using Euler's formula to generate a complex sinusoidal wave
    carrier_wave = 1 * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    modulation_array = get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype)
    
    # Pad (or trim) the modulation array to match the length of the carrier wave
    if len(carrier_wave) > len(modulation_array):
        pad_len = len(carrier_wave) - len(modulation_array)
        modulation_array = np.append(modulation_array, np.full(pad_len, amp_map['LOW'], dtype=dtype))
    elif len(carrier_wave) < len(modulation_array):
        modulation_array = modulation_array[:len(carrier_wave)]
    
    return t, carrier_wave * modulation_array

def generate_pulse(bit_val, carrier_wave_freq, sample_rate, baud, multiple_of_bit_len,
                   amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = multiple_of_bit_len * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    high_or_low = 'HIGH' if bit_val == '1' else 'LOW'
    pulse = amp_map[high_or_low] * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    return t, pulse

binary_data = '0110100010000000'
carrier_wave_freq = 315e6
sample_rate = 2e6
baud = 205

complex64_amp_map = {
    'LOW': 1.4 * .02,
    'HIGH': 1.4
}

t, complex_signal = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud, amp_map=complex64_amp_map, dtype='complex64')
t2, signal_header = generate_pulse('1', carrier_wave_freq, sample_rate, baud, 3.85, amp_map=complex64_amp_map, dtype='complex64')
t3, signal_spacer = generate_pulse('0', carrier_wave_freq, sample_rate, baud, 3.78, amp_map=complex64_amp_map, dtype='complex64')

full_signal = join_all_arrays([signal_header] + ([complex_signal, signal_spacer] * 12))

write_pcm_file(full_signal, 'raw_data/generated_sig2.pcm', dtype='complex64')

def generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate,
                               baud, sig_ratios=DEFAULT_RATIOS, amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = len(binary_data) * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    # Using Euler's formula to generate a complex sinusoidal wave
    carrier_wave = 1 * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    modulation_array = get_modulation_array(binary_data, sample_rate, baud, sig_ratios, amp_map, dtype)
    
    # Pad (or trim) the modulation array to match the length of the carrier wave
    if len(carrier_wave) > len(modulation_array):
        pad_len = len(carrier_wave) - len(modulation_array)
        modulation_array = np.append(modulation_array, np.full(pad_len, amp_map['LOW'], dtype=dtype))
    elif len(carrier_wave) < len(modulation_array):
        modulation_array = modulation_array[:len(carrier_wave)]
    
    # Modulate by superwave
    super_wave_freq = carrier_wave_freq / (160*2)
    super_wave = 1 * np.e**(super_wave_freq * 2 * np.pi * (0+1j) * t)
    
    return t, carrier_wave * modulation_array * super_wave

def generate_pulse(bit_val, carrier_wave_freq, sample_rate, baud, multiple_of_bit_len,
                   amp_map=DEFAULT_AMP_MAP, dtype=np.int16):
    signal_len_secs = multiple_of_bit_len * (1/baud)
    t = np.linspace(0, signal_len_secs, sample_rate * signal_len_secs)
    
    high_or_low = 'HIGH' if bit_val == '1' else 'LOW'
    pulse = amp_map[high_or_low] * np.e**(carrier_wave_freq * 2 * np.pi * (0+1j) * t)
    return t, pulse

binary_data = '0110100010000000'
carrier_wave_freq = 315e6
sample_rate = 2e6
baud = 205 * 2

complex64_amp_map = {
    'LOW': 1.4 * .02,
    'HIGH': 1.4
}

t, complex_signal = generate_on_off_key_signal(binary_data, carrier_wave_freq, sample_rate, baud, amp_map=complex64_amp_map, dtype='complex64')
t2, signal_header = generate_pulse('1', carrier_wave_freq, sample_rate, baud, 3.85, amp_map=complex64_amp_map, dtype='complex64')
t3, signal_spacer = generate_pulse('0', carrier_wave_freq, sample_rate, baud, 3.78, amp_map=complex64_amp_map, dtype='complex64')

full_signal = join_all_arrays([signal_header] + ([complex_signal, signal_spacer] * 12))
write_pcm_file(full_signal, 'raw_data/generated_sig3.pcm', dtype='complex64')



