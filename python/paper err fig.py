import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
get_ipython().magic('matplotlib inline')

af = np.load("data/final_run/paperslowlong_learning_data.npz")
print(af.keys())

error = np.sum(np.abs(af['p_error']), axis=1)
plt.plot(error)
print(error.shape)

plt.plot(error[20000:22200])

plt.plot(error[:100])

# this might need to be adjusted for the lower entries where the error sits at the minimum of 0.2
filt_err = error[np.where(error > 0.1)]
plt.plot(filt_err)

plt.plot(filt_err[:100])

plt.plot(np.abs(np.diff(filt_err)))

thresh_diff = filt_err[np.where(np.abs(np.diff(filt_err)) < 0.25)]
plt.plot(thresh_diff)

plt.plot(nengo.Lowpass(0.004).filtfilt(thresh_diff[:300]))
plt.plot(thresh_diff[:300])

plt.plot(nengo.Lowpass(0.005).filtfilt(thresh_diff))
plt.plot(thresh_diff, alpha=0.5)

filt_thresh = nengo.Lowpass(0.005).filtfilt(thresh_diff)[:10000]
plt.plot(np.arange(0, filt_thresh.shape[0])*0.01, filt_thresh)
plt.ylim(0, 3)
plt.ylabel("Magnitude of Error")
plt.xlabel("Training Time (s)")
plt.title("Decrease of Error with Training Time")
plt.savefig("error_slow.pdf", format="pdf")

big_diff = np.where(np.abs(np.diff(error)) > 0.50)[0]
print(big_diff[:10])
print(np.diff(big_diff)[:10])

plt.plot(error[20:33])
plt.plot(error[46:62])
plt.plot(error[63:71])
plt.plot(error[71:92])

plt.plot(error[:200])
plt.plot(big_diff[:10], error[big_diff[:10]], marker='o', linestyle='None')

plt.plot(np.arange(error.shape[0])[-200:], error[-200:])
plt.plot(big_diff[-5:], error[big_diff[-5:]], marker='o', linestyle='None')
print(np.diff(big_diff[-5:]))

plt.plot(error[:200])
plt.plot(np.diff(error[:200]))

neg_diff = np.where(np.diff(error) < -0.75)[0]
print(neg_diff[:10])
print(error[neg_diff].shape)

plt.plot(error[:100], alpha=0.5)
plt.plot(neg_diff[:10], error[neg_diff][:10], marker='o')

plt.plot(error, alpha=0.5)
plt.plot(neg_diff, error[neg_diff], marker='o')



