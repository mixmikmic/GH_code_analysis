from shared_notebook_utils import *
from analysis_algorithms import *
import essentia
import essentia.standard as estd
import random
dataset = Dataset('toy_dataset')
get_ipython().magic('matplotlib inline')

# This notebook contains excerpts from the article: Font, F., & Serra, X. (2016). Tempo Estimation for Music Loops and a Simple Confidence Measure. In Proceedings of the Int. Conf. on Music Information Retrieval (ISMIR).
# License: CC-BY-4.0

# The following code exemplifies the computation of the confidence measure for two different sound examples 
# and generates Figure 1 of the paper.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 5))
sample_rate=44100


# Top of the figure: loop for which BPM estimation fails
########################################################

# Select and load a sound
# Sound: "edison_140d.wav" by NoiseCollector, got from Freesound at http://www.freesound.org/people/NoiseCollector/sounds/63470/
# License: CC-BY-3.0
selected_sound = dataset.data['63470']
print title('Selected sound id: %s' % selected_sound['id'])
sound_file_path = os.path.join(dataset.dataset_path, selected_sound['wav_sound_path'])
selected_sound['file_path'] = sound_file_path
audio_1 = load_audio_file(file_path=sound_file_path, sample_rate=sample_rate)
bpm_1 = algorithm_rhythm_percival14(selected_sound)['Percival14']['bpm']
bpm_1 = int(round(bpm_1))
IPython.display.display(IPython.display.Audio(sound_file_path), embed=True)
print 'Selected sound ground truth bpm: %.2f' % selected_sound['annotations']['bpm']
print 'Selected sound estimated bpm: %.2f' % bpm_1

# Compute confidence based on "standard" audio signal duration
beat_duration = (60.0 * sample_rate)/bpm_1
L = [beat_duration * n for n in range(1, 128)]
thr_lambda = 0.5 * beat_duration
la = audio_1.shape[0]
delta_l = min([abs(l - la) for l in L])
if delta_l > thr_lambda:
    confidence_la = 0.0
else:
    confidence_la = (1.0 - float(delta_l) / thr_lambda)
print '  Confidence: %.2f' % confidence_la

# Plot
ax1.plot(normalize2(audio_1), color="gray", alpha=0.5)
ax1.vlines(L, -1, 1, color='black', linewidth=1, linestyle="--")
ax1.vlines([350, la], -1, 1, color='red', linewidth=2)
annotate_point_pair(ax1, r'$l^a$', (0, -0.65), (la, -0.65), xycoords='data', text_offset=16, text_size=16)
annotate_point_pair(ax1, r'$l^b$', (L[1], 0.75), (L[2], 0.75), xycoords='data', text_offset=16, text_size=16)
annotate_point_pair(ax1, r'$\Delta l$', (la, -0.35), (la + delta_l, -0.35), xycoords='data', text_offset=16, textx_offset=-2000, text_size=16)
annotate_point_pair(ax1, r'$\lambda$', (la + delta_l - thr_lambda, 0.85), (la + delta_l, 0.85), xycoords='data', text_offset=16, textx_offset=-2000, text_size=16)
confidence_output = list()
for i in range(0, la*2):
    delta = min([abs(l - i) for l in L])
    if delta > thr_lambda:
        confidence_output.append(0.0)
    else:
        value = 1.0 - float(delta) / thr_lambda
        confidence_output.append(value)
ax1.plot(confidence_output, color=COLORS[2])
ax1.set_xlim((0, la + 44100/2))
ax1.set_ylim((-1, 1))
#ax1.set_xlabel('Time (samples)')


# Bottom of the figure: loop for which BPM estimation works but that has silence at the beggining
#################################################################################################

# Select and load sound and add 100 ms silence at the beginning and at the end
# Sound: "91Apat99999.wav" by NoiseCollector, got from Freesound at http://www.freesound.org/people/NoiseCollector/sounds/43209/
# License: CC-BY-3.0
selected_sound = dataset.data['43209']
print title('Selected sound id: %s' % selected_sound['id'])
sound_file_path = os.path.join(dataset.dataset_path, selected_sound['wav_sound_path'])
selected_sound['file_path'] = sound_file_path
audio_1 = load_audio_file(file_path=sound_file_path, sample_rate=sample_rate)
n_samples_silence = 4410
audio_2 = np.concatenate((np.zeros(n_samples_silence), audio_1, np.zeros(n_samples_silence)))
bpm_2 = algorithm_rhythm_percival14(selected_sound)['Percival14']['bpm']
bpm_2 = int(round(bpm_2))
IPython.display.display(IPython.display.Audio(sound_file_path), embed=True)
print 'Selected sound ground truth bpm: %.2f' % selected_sound['annotations']['bpm']
print 'Selected sound estimated bpm: %.2f' % bpm_2

# Compute confidence based on different durations
beat_duration = (60.0 * sample_rate)/bpm_2
L = [beat_duration * n for n in range(1, 128)]  # From 1 beat to 32 beats (would be 32 bars in 4/4)
thr_lambda = 0.5 * beat_duration
z = 0.05  # Percentage of the envelope amplitude that we use to compute start and end of signal
env = estd.Envelope(attackTime=10, releaseTime=10)
envelope = env(essentia.array(audio_2))
env_threshold = envelope.max() * z
envelope_above_threshold = np.where(envelope >= env_threshold)
start_effective_duration = envelope_above_threshold[0][0]
end_effective_duration = envelope_above_threshold[0][-1]
la = audio_2.shape[0]
durations_to_check = [
    ('Standard duration', la),
    ('Removed silence beginning', la - start_effective_duration),
    ('Removed silence end', end_effective_duration),
    ('Removed slience beginning and end', end_effective_duration - start_effective_duration)
]
for name, duration in durations_to_check:
    delta_l = min([abs(l - duration) for l in L])
    if delta_l > thr_lambda:
        confidence = 0.0
    else:
        confidence = (1.0 - float(delta_l) / thr_lambda)
    print '  Confidence for "%s": %.2f' % (name, confidence)

# Plot
ax2.plot(normalize2(audio_2), color="gray", alpha=0.5)
ax2.plot(normalize2(envelope), color=COLORS[1])
ax2.vlines([l + start_effective_duration for l in L], -1, 1, color='black', linewidth=1, linestyle="--")
ax2.vlines([start_effective_duration, end_effective_duration], -1, 1, color='red', linewidth=2, linestyle=":")
ax2.vlines([350, la], -1, 1, color='red', linewidth=2)
annotate_point_pair(ax2, r'$l^a$', (0, -0.65), (la, -0.65), xycoords='data', text_offset=16, text_size=16)
annotate_point_pair(ax2, r'$l_0^a$', (start_effective_duration, -0.25), (la, -0.25), xycoords='data', text_offset=16, text_size=16)
annotate_point_pair(ax2, r'$l_1^a$', (0, 0.15), (end_effective_duration, 0.15), xycoords='data', text_offset=16, text_size=16)
annotate_point_pair(ax2, r'$l_2^a$', (start_effective_duration, 0.55), (end_effective_duration, 0.55), xycoords='data', text_offset=16, text_size=16)
confidence_output = list()
for i in range(0, la*2):
    delta = min([abs(l - i) for l in L])
    if delta > thr_lambda:
        confidence_output.append(0.0)
    else:
        value = 1.0 - float(delta) / thr_lambda
        confidence_output.append(value)
confidence_output = list(np.zeros(start_effective_duration)) + confidence_output
ax2.plot(confidence_output, color=COLORS[2])
ax2.set_xlim((0, la + 44100/2))
ax2.set_ylim((-1, 1))
ax2.set_xlabel('Time (samples)')

plt.show()
figure_caption = """
**Figure 1**: Visualisation of confidence computation output according to BPM estimation and signal duration (green curves). The top figure shows a loop whose annotated tempo is 140 BPM but the predicted tempo is 119 BPM.
The duration of the signal $l^a$ does not closely match any multiple of $l^b$ (dashed vertical lines), and the output confidence is 0.59 (i.e.,~$1 - \Delta l / \lambda$).
The figure at the bottom shows a loop that contains silence at the beginning and at the end, and for which tempo has been correctly estimated as being 91 BPM.
The yellow curve represents its envelope and the vertical dashed red lines the estimated effective start and end points.
Note that $l_2^a$ closely matches a multiple of $l^b$, resulting in a confidence of 0.97. The output confidence computed with $l^{a}$, $l_0^{a}$ and $l_1^{a}$ produces lower values.
"""
IPython.display.display(IPython.display.Markdown(figure_caption))



