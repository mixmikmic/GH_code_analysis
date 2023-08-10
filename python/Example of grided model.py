import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import open_cp.data
import open_cp.sources.sepp as source_sepp
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
import sepp.sepp_grid
import sepp.grid_nonparam

rates = np.random.random(size=(10,10))
simulation = source_sepp.GridHawkesProcess(rates, 0.5, 10)
points = simulation.sample_to_randomised_grid(0, 365, grid_size=50)
time_unit = source_sepp.make_time_unit(datetime.timedelta(days=1))
timed_points = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1),
    time_unit)

fig, ax = plt.subplots(ncols=2, figsize=(16,7))

ax[0].scatter(timed_points.xcoords, timed_points.ycoords, alpha=0.1)
ax[0].set_title("Space location of simulated data")
ax[0].set(xlim=[0,500], ylim=[0,500])

times = timed_points.times_datetime()
ax[1].scatter(times, timed_points.xcoords, alpha=0.1)
ax[1].set_xlim([datetime.datetime(2017,1,1), datetime.datetime(2018,1,1)])
ax[1].set_title("Date against x location")
fig.autofmt_xdate()
None

mask = [[False] * 10 for _ in range(10)]
grid = open_cp.data.MaskedGrid(50, 50, 0, 0, mask)

trainer = sepp.sepp_grid.SEPPGridTrainer(grid)
trainer.data = timed_points

points = trainer.to_cells(datetime.datetime(2018,1,1))

mu = [[0.5] * 10] * 10
model = sepp.sepp_grid.ExpDecayModel(mu, 365, 0.2, 2)

for _ in range(10):
    opt = sepp.sepp_grid.ExpDecayOptFast(model, points)
    model = opt.optimised_model()
    print(model)

fig, ax = plt.subplots(ncols=1, figsize=(8,6))
ax.plot([0,1], [0,1], linewidth=1, color="r")
ax.scatter(rates.ravel(), model.mu.ravel())
#ax.set(xlim=[0,1], ylim=[0,np.max(predictor.mu)*1440*1.1])
ax.set(xlabel="Actual background rate", ylabel="Predicted background rate")
ax.set_title("Predicted background rates")
None

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = timed_points
model = trainer.train(datetime.datetime(2018,1,1))

simulation = source_sepp.GridHawkesProcess(rates, 0.5, 0.1)
points = simulation.sample_to_randomised_grid(0, 365, grid_size=50)
time_unit = source_sepp.make_time_unit(datetime.timedelta(days=1))

trainer1 = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer1.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)

model1 = trainer1.train(datetime.datetime(2018,1,1))
model2 = trainer1.train(datetime.datetime(2018,1,1), use_fast=False)

model, model1, model2

fig, axes = plt.subplots(ncols=3, figsize=(16,5))

for ax, m in zip(axes, [model, model1, model2]):
    ax.plot([0,1], [0,1], linewidth=1, color="black")
    ax.scatter(rates.ravel(), m.mu.ravel(), color="black", marker="x")
    ax.set(xlabel="Actual background rate", ylabel="Predicted background rate")
axes[0].set_title("$\\theta=0.5, \omega=10$")
axes[1].set_title("$\\theta=0.5, \omega=0.1$")
axes[2].set_title("$\\theta=0.5, \omega=0.1$ with full EM algoritm")
None

fig.savefig("../simdata.pdf")

simulation = source_sepp.GridHawkesProcess(rates, 0.5, 10)
points = simulation.sample_to_randomised_grid(0, 365, grid_size=50)
time_unit = source_sepp.make_time_unit(datetime.timedelta(days=1))

mask = [[False] * 10 for _ in range(10)]
grid = open_cp.data.MaskedGrid(50, 50, 0, 0, mask)

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
trainer.data = trainer.data.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(hours=1))
model = trainer.train(cutoff=datetime.datetime(2018,1,1), iterations=50, use_fast=True)
model

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
trainer.data = trainer.data.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(hours=6))
model1 = trainer.train(cutoff=datetime.datetime(2018,1,1), iterations=50, use_fast=True)
model1

fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

for ax, m in zip(axes, [model, model1]):
    ax.plot([0,1], [0,1], linewidth=1, color="r")
    ax.scatter(rates.ravel(), m.mu.ravel())
    #ax.set(xlim=[0,1], ylim=[0,np.max(predictor.mu)*1440*1.1])
    ax.set(xlabel="Actual background rate", ylabel="Predicted background rate")

axes[0].set_title("Predicted background rates, bin size=1h")
axes[1].set_title("Predicted background rates, bin size=6h")

def plot(model, bk_rate_scale=1):
    fig, axes = plt.subplots(ncols=2, figsize=(16,6))

    ax = axes[0]
    ax.plot([0,1], [0,1], linewidth=1, color="r")
    ax.scatter(rates.ravel(), model.mu.ravel() * bk_rate_scale)
    #ax.set(xlim=[0,1], ylim=[0,np.max(predictor.mu)*1440*1.1])
    ax.set(xlabel="Actual background rate", ylabel="Predicted background rate")
    ax.set_title("Predicted background rates")

    ax = axes[1]
    x = np.arange(20) * model.bandwidth
    ax.plot(x, model.alpha[:len(x)])
    y = np.exp(-x*10) * (1 - np.exp(-10*model.bandwidth))
    ax.plot(x, y)
    ax.set(xlabel="Days", title="Trigger")
    
    return fig

simulation = source_sepp.GridHawkesProcess(rates, 0.5, 10)
points = simulation.sample_to_randomised_grid(0, 365, grid_size=50)
time_unit = source_sepp.make_time_unit(datetime.timedelta(days=1))

mask = [[False] * 10 for _ in range(10)]
grid = open_cp.data.MaskedGrid(50, 50, 0, 0, mask)

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=0.1)
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
model = trainer.train(cutoff=datetime.datetime(2018,1,1), iterations=50, use_fast=True)
fig = plot(model)
model

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=0.5)
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
model = trainer.train(cutoff=datetime.datetime(2018,1,1), iterations=50, use_fast=True)
fig = plot(model)
model

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=1)
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
model = trainer.train(cutoff=datetime.datetime(2018,1,1), iterations=50, use_fast=True)
fig = plot(model)
model

fig = plot(model)
model

rates = np.random.random(size=(2,2))
simulation = source_sepp.GridHawkesProcess(rates, 0.5, 10)
length = 50
points = simulation.sample_to_randomised_grid(0, length, grid_size=50)
time_unit = source_sepp.make_time_unit(datetime.timedelta(days=365 / length))

mask = [[False] * 2 for _ in range(2)]
grid = open_cp.data.MaskedGrid(50, 50, 0, 0, mask)

def plot(length, model):
    fig, axes = plt.subplots(ncols=2, figsize=(16,6))

    ax = axes[0]
    ax.plot([0,1], [0,1], linewidth=1, color="r")
    ax.scatter(rates.ravel(), model.mu.ravel() * 365 / length)
    #ax.set(xlim=[0,1], ylim=[0,np.max(predictor.mu)*1440*1.1])
    ax.set(xlabel="Actual background rate", ylabel="Predicted background rate")
    ax.set_title("Predicted background rates")

    ax = axes[1]
    x = np.linspace(0, 10, 100)
    ax.plot(x, model.trigger_func(x) * model.theta)
    y = 0.5 * 10 * np.exp(-10 * x)
    ax.plot(x, y)
    ax.set(xlabel="Days", title="Trigger", ylim=[0,1])
    
    return fig

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(1))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
model

fig = plot(length, model)

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.2))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
model

fig = plot(length, model)

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.1))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
model

fig = plot(length, model)

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderFixedBandwidth(0.01))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
model

fig = plot(length, model)

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(3))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
model

fig = plot(length, model)

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(6))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
plot(length, model)
model

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(10))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
plot(length, model)
model

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(20))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
plot(length, model)
model

trainer = sepp.grid_nonparam.KDETrainer(grid, sepp.grid_nonparam.KDEProviderKthNearestNeighbour(50))
trainer.data = source_sepp.scale_to_real_time(points, datetime.datetime(2017,1,1), time_unit)
print(trainer.data.time_range)
model = trainer.train(datetime.datetime(2018,1,1), iterations=50)
plot(length, model)
model



