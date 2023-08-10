get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import keras.backend as K
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from keras.models import load_model
from lorenz_gan.gan import Interpolate1D
import pandas as pd
from scipy.stats import expon, lognorm
import pickle

gen_model = load_model("../exp/gan_generator_0000_epoch_0010.h5", custom_objects={"Interpolate1D": Interpolate1D})

def normalize_data(data, scaling_values=None):
    """
    Normalize each channel in the 4 dimensional data matrix independently.

    Args:
        data: 4-dimensional array with dimensions (example, y, x, channel/variable)
        scaling_values: pandas dataframe containing mean and std columns

    Returns:
        normalized data array, scaling_values
    """
    normed_data = np.zeros(data.shape, dtype=data.dtype)
    scale_cols = ["mean", "std"]
    if scaling_values is None:
        scaling_values = pd.DataFrame(np.zeros((data.shape[-1], len(scale_cols)), dtype=np.float32),
                                      columns=scale_cols)
    for i in range(data.shape[-1]):
        scaling_values.loc[i, ["mean", "std"]] = [data[:, :, i].mean(), data[:, :, i].std()]
        normed_data[:, :, i] = (data[:, :, i] - scaling_values.loc[i, "mean"]) / scaling_values.loc[i, "std"]
    return normed_data, scaling_values


def unnormalize_data(normed_data, scaling_values):
    """
    Re-scale normalized data back to original values

    Args:
        normed_data: normalized data
        scaling_values: pandas dataframe of mean and standard deviation from normalize_data

    Returns:
        Re-scaled data
    """
    data = np.zeros(normed_data.shape, dtype=normed_data.dtype)
    for i in range(normed_data.shape[-1]):
        data[:, :, i] = normed_data[:, :, i] * scaling_values.loc[i, "std"] + scaling_values.loc[i, "mean"]
    return data

def fit_condition_distributions(train_cond_data):
    """
    Calculate the scale parameter for the exponential distribution of correlated conditional variables
    for the Lorenz 96 model in time.

    Args:
        train_cond_data: array of conditioning values where the first column is the current X, and each
            other column is a lagged X value

    Returns:
        array of scale values
    """
    train_cond_exp_scale = np.zeros(train_cond_data.shape[1] - 1)
    for i in range(1, train_cond_data.shape[1]):
        train_cond_exp_scale[i - 1] = expon.fit(np.abs(train_cond_data[:, 0] - train_cond_data[:, i]), floc=0)[1]
    return train_cond_exp_scale


def generate_random_condition_data(batch_size, num_cond_inputs, train_cond_scale):
    """
    Generate correlated conditional random numbers to train the generator network.

    Args:
        batch_size: number of random samples
        num_cond_inputs: number of conditional inputs
        train_cond_scale: exponential distribution scale values

    Returns:

    """
    batch_cond_data = np.zeros((batch_size, num_cond_inputs, 1))
    batch_cond_data[:, 0, 0] = np.random.normal(size=batch_size)
    for t in range(1, train_cond_scale.size + 1):
        batch_cond_data[:, t , 0] = batch_cond_data[:, 0, 0] +                                     np.random.choice([-1, 1], size=batch_size) * expon.rvs(loc=0,
                                                                                           scale=train_cond_scale[t-1],
                                                                                           size=batch_size)
    return batch_cond_data

patches.shape

gan_samples = xr.open_dataset("../exp/gan_gen_patches_0000_epoch_0002.nc")
patches = gan_samples["gen_samples"][:, :, 0].values
gan_samples.close()



plt.pcolormesh(patches)

plt.hist(patches[:, 0])

plt.hist(patches.mean(axis=1))

gan_loss = pd.read_csv("../exp/gan_loss_history_0000.csv", index_col="Time")

gan_loss[["Gen Loss", "Disc Loss"]].plot()

gan_loss[["Gen accuracy", "Disc accuracy"]].rolling(1000).mean().plot()

combined_data = pd.read_csv("../exp/lorenz_combined_output.csv")

y_cols = combined_data.columns[combined_data.columns.str.contains("Y")]
x_cols = combined_data.columns[combined_data.columns.str.contains("X")]
print(y_cols)
print(x_cols)

x_norm, x_scaling = normalize_data(np.expand_dims(combined_data[x_cols].values, axis=-1))
y_norm, y_scaling = normalize_data(np.expand_dims(combined_data[y_cols].values, axis=-1))
gan_norm = gen_model.predict([x_norm[:,:,0] , np.random.normal(size=(x_norm.shape[0], 20))])
gan_y = unnormalize_data(gan_norm, y_scaling)

plt.pcolormesh(combined_data[y_cols])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
_ = axes[0].hist2d(combined_data["X_t"], combined_data[y_cols].sum(axis=1), bins=[np.linspace(-30, 30, 30), 
                                                                                   np.linspace(-30, 30, 30)], cmin=1)
axes[0].set_title("Lorenz '96 Truth", fontsize=16)
_ = axes[1].hist2d(combined_data["X_t"], gan_y[:, :, 0].sum(axis=1), bins=[np.linspace(-30, 30, 30), 
                                                                    np.linspace(-30, 30, 30)], cmin=1)
axes[1].set_title("Lorenz '96 GAN", fontsize=16)
axes[0].set_xlabel("X$_t$", fontsize=14)
axes[1].set_xlabel("X$_t$", fontsize=14)
axes[0].grid()
axes[1].grid()
axes[0].set_ylabel("U", fontsize=14)
plt.savefig("../exp/gan_x_ymean_hist.png", dpi=200, bbox_inches="tight")

_ = plt.hist2d(combined_data["X_t"], combined_data[y_cols].sum(axis=1), bins=[np.linspace(-30, 30, 30), 
               np.linspace(-30, 30, 30)], cmin=1, cmap="Reds", alpha=0.6)

_ = plt.hist2d(combined_data["X_t"], gan_y[:, :, 0].sum(axis=1), bins=[np.linspace(-30, 30, 30), 
               np.linspace(-30, 30, 30)], cmap="Blues", cmin=1, alpha=0.6)

y_data = combined_data[y_cols].values

plt.fill_between(np.arange(32), np.percentile(gan_y[:, :, 0], 95, axis=0), 
                 np.percentile(gan_y[:, :, 0], 5, axis=0), alpha=0.5, label="GAN 90% CI", facecolor="blue")
plt.plot(gan_y[:, :, 0].mean(axis=0), label="GAN Mean", color="blue")
plt.fill_between(np.arange(32), np.percentile(y_data, 95, axis=0), 
                 np.percentile(y_data, 5, axis=0), alpha=0.5, facecolor="red", label="Truth 90% CI")
plt.plot(y_data.mean(axis=0), label="Truth Mean", color="red")
plt.legend(loc=3, fontsize=10)
plt.xlabel("Y Index", fontsize=12)
plt.ylabel("Y Statistic", fontsize=12)
plt.title("Lorenz Y Distributions")
plt.savefig("../exp/lorenz_y_dist.png", dpi=200, bbox_inches="tight")

from matplotlib.colors import LogNorm
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(0.05, 0.05, 0.9, 0.9)
cax = fig.add_axes([0.95, 0.05, 0.05, .85])
pc = axes[0].pcolormesh(np.corrcoef(y_data.T) **2, vmin=0.01, vmax=1, norm=LogNorm(0.01, 1))
axes[1].pcolormesh(np.corrcoef(gan_y[:, :, 0].T)**2, vmin=0.01, vmax=1, norm=LogNorm(0.01, 1))
axes[0].set_title("Y Truth Correlations", fontsize=16)
axes[0].set_xticks(np.linspace(0, 32, 9))
axes[1].set_xticks(np.linspace(0, 32, 9))
axes[0].set_yticks(np.linspace(0, 32, 9))
axes[1].set_yticks(np.linspace(0, 32, 9))
axes[0].grid()
axes[1].grid()
axes[1].set_title("Y GAN Correlations", fontsize=16)
plt.colorbar(pc, cax=cax)
plt.savefig("../exp/y_gan_correlations.png", dpi=200, bbox_inches="tight")

plt.plot(rand_y[20])

np.corrcoef(combined_data[x_cols].values.T)

plt.hist(combined_data[y_cols].mean(axis=1), bins=np.arange(-1, 1.1, 0.05), normed=True, histtype="step")
plt.hist(gan_y[:, 0, 0], bins=np.arange(-1, 1.1, 0.05), normed=True, histtype="step")

plt.hist(combined_data[y_cols].std(axis=1), bins=np.arange(0, 1.1, 0.05), normed=True, histtype="step")
plt.hist(gan_y[:, 0, 0], bins=np.arange(0, 1.1, 0.05), normed=True, histtype="step")

plt.hist(np.abs(combined_data["X_t-2"] - combined_data["X_t"]), bins=30)

exp_loc, exp_scale = expon.fit(np.abs(combined_data["X_t-1"] - combined_data["X_t"]), floc=0)
print(exp_loc, exp_scale)

lorenz_series = xr.open_dataset("../exp/lorenz_output.nc")

x_values = lorenz_series["lorenz_x"][:, 0].to_dataframe()

lags = np.arange(5, 2001, 5)
auto_corr = np.zeros(lags.size)
for l, lag in enumerate(lags):
    auto_corr[l] = x_values["lorenz_x"].autocorr(lag)

auto_corr[:5]

plt.plot(lags, auto_corr)
plt.plot(lags, np.zeros(lags.size), 'k--')
plt.gca().set_xscale("log")

x_values["lorenz_x"].values[1000::5]

plt.hist(x_values["lorenz_x"].values[100:900015:5] - x_values["lorenz_x"].values[85:900000:5])

plt.plot(np.arange(1000), x_values.iloc[2000:3000, 1])

x_values.shape

plt.hist((x_values["lorenz_x"] - x_values["lorenz_x"].mean()) / x_values['lorenz_x'].std(), 30)

all_x = lorenz_series["lorenz_x"].values

plt.pcolor(np.corrcoef(all_x.T), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
plt.colorbar()

def fit_condition_distributions(train_cond_data):
    """
    Calculate the scale parameter for the exponential distribution of correlated conditional variables
    for the Lorenz 96 model in time.

    Args:
        train_cond_data: array of conditioning values where the first column is the current X, and each
            other column is a lagged X value

    Returns:
        array of scale values
    """
    train_cond_exp_scale = np.zeros(train_cond_data.shape[1] - 1)
    for i in range(1, train_cond_data.shape[1]):
        train_cond_exp_scale[i - 1] = expon.fit(np.abs(train_cond_data[:, 0] - train_cond_data[:, i]), floc=0)[1]
    return train_cond_exp_scale


def generate_random_condition_data(batch_size, num_cond_inputs, train_cond_scale):
    batch_cond_data = np.zeros((batch_size, num_cond_inputs, 1))
    batch_cond_data[:, 0, 0] = np.random.normal(size=batch_size)
    for t in range(1, train_cond_scale.size + 1):
        batch_cond_data[:, t , 0] = batch_cond_data[:, 0, 0] +                                     np.random.choice([-1, 1], size=batch_size) * expon.rvs(loc=0,
                                                                                           scale=train_cond_scale[t-1],
                                                                                           size=batch_size)
    return batch_cond_data

normed_x = (combined_data[x_cols] - combined_data[x_cols].mean()) / combined_data[x_cols].std()
normed_x = normed_x.values
print(normed_x.shape)

exp_scale = fit_condition_distributions(normed_x)
print(exp_scale)
batch_cond_data = generate_random_condition_data(128, 3, exp_scale)

rf = RandomForestRegressor(n_estimators=100, max_features="auto", max_depth=10)
rf.fit(combined_data[x_cols], combined_data[y_cols].mean(axis=1))

out = rf.predict(combined_data[x_cols])

tree_preds = np.array([t.predict(combined_data[x_cols]) for t in rf.estimators_]).T

print(combined_data.loc[0, x_cols])
plt.hist(tree_preds[0])

resampled_mean_preds = np.zeros(1000)
for i in range(1000):
    resampled_mean_preds[i] = np.random.choice(tree_preds[0], size=100, replace=True).mean()
plt.hist(resampled_mean_preds)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
_ = axes[0].hist2d(combined_data["X_t"], combined_data[y_cols].mean(axis=1), bins=[np.linspace(-30, 30, 30), 
                                                                                   np.linspace(-1, 1, 30)], cmin=1)
_ = axes[1].hist2d(combined_data["X_t"], out, bins=[np.linspace(-30, 30, 30), 
                                            np.linspace(-1, 1, 30)], cmin=1)

rf.feature_importances_

plt.hist(combined_data["X_t"] - combined_data["X_t-2"])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
_ = axes[0].hist2d(combined_data["X_t"], combined_data["X_t"] -combined_data["X_t-1"], bins=[np.linspace(-30, 30, 30), 
                                                                       np.linspace(-5, 5, 60)], cmin=1)
_ = axes[1].hist2d(combined_data[y_cols].sum(axis=1), combined_data["X_t"] - combined_data["X_t-1"], bins=[np.linspace(-30, 30, 60), 
                                                                       np.linspace(-5, 5, 60)], cmin=1)

lm = LinearRegression()
lm.fit(np.expand_dims(combined_data["X_t"].values, axis=-1), combined_data["X_t"] -combined_data["X_t-1"])

lm.coef_

lorenz_data = xr.open_dataset("../exp/lorenz_output.nc")
x_vals = lorenz_data["lorenz_x"].values
lorenz_data.close()

lorenz_forecasts = []
for i in range(10):
    lorenz_forecasts.append(pd.read_csv("../exp/lorenz_forecast_{0:02d}.csv".format(i)))

plt.figure(figsize=(10, 6))

plt.plot(np.mean([lorenz_forecasts[i]["X_0"] for i in range(10)], axis=0))
plt.plot(x_vals[100000:101000, 0], 'k--')

with open("../exp/u_histogram.pkl", "rb") as hist_file:
    hist_obj = pickle.load(hist_file)

plt.pcolormesh(hist_obj.u_bins, hist_obj.x_bins, hist_obj.histogram.T)

hist_obj.histogram.shape

lorenz_data[100000:101000, 0]

b = K.random_uniform_variable(low=0, high=5, shape=(3, 4))

b[:-1]

d = np.arange(4)
e = np.zeros(8)
e[::2] = d

e[1:-1:2] = 0.5 * d[:-1] + 0.5 * d[1:]

e[-1] = 2 * e[-2] - e[-3]

e



