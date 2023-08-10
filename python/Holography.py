import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

get_ipython().magic('matplotlib nbagg')

holo0 = hs.load('./datasets/01_holo_Vbp_130V_0V_bin2_crop.hdf5', signal_type='hologram')

hs.plot.plot_images([holo0, holo0.isig[150:250, 200:300]], tight_layout=False)

holo5 = hs.load('./datasets/02_holo_Vbp_130V_5V_bin2_crop.hdf5', signal_type='hologram')

ref = hs.load('./datasets/00_ref_Vbp_130V_0V_bin2_crop.hdf5', signal_type='hologram')

ref.plot()

fft_holo0 = np.log(np.abs(fftshift(fft2(holo0.data))))
plt.imshow(fft_holo0)

fft_holo0 = hs.signals.Signal2D(fft_holo0)
m = hs.plot.markers.rectangle(x1=290, y1=100, x2=350, y2=160, color='red')
fft_holo0.add_marker(m)

sb_position = ref.estimate_sideband_position(sb='upper')
sb_position.data

sb_size = ref.estimate_sideband_size(sb_position)
sb_size.data

wave0 = holo0.reconstruct_phase(ref, sb_position=sb_position, sb_size=sb_size,
                                output_shape=(int(sb_size.data*2), int(sb_size.data*2)))
wave5 = holo5.reconstruct_phase(ref, sb_position=sb_position, sb_size=sb_size,
                                output_shape=(int(sb_size.data*2), int(sb_size.data*2)))
hs.plot.plot_images([wave0.real, wave0.imag, wave5.real, wave5.imag], saturated_pixels=.5,
                    per_row=2, cmap="RdBu_r", tight_layout=True, )

wave5.amplitude.plot(saturated_pixels=0.5, vmin=0)

wave5.phase.plot(cmap="RdBu_r")

wave5.unwrapped_phase().plot(cmap="RdBu_r")

wave_electrostatic = wave5 / wave0
wave_electrostatic.phase.plot(cmap="RdBu_r")

phase_stack = hs.stack([wave0.unwrapped_phase(), wave5.unwrapped_phase()])
shifts = phase_stack.estimate_shift2D()

shifts

wave5a= wave5.deepcopy()
wave5a.map(np.roll, shift=0, axis=0)
wave5a.map(np.roll, shift=1, axis=1)

wave_electrostatic = wave5a / wave0
wave_electrostatic.phase.plot(cmap="RdBu_r")

uphase_electrostatic = wave_electrostatic.unwrapped_phase()
uphase_electrostatic.plot(cmap="RdBu_r")

contours_electrostatic = uphase_electrostatic._deepcopy_with_new_data(np.cos(2*uphase_electrostatic.data))
contours_electrostatic.plot(cmap="RdBu_r")



