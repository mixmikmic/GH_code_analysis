year = 2017

get_ipython().run_line_magic('matplotlib', 'inline')

import papermill as pm

import bikes

data = bikes.get_weather_data(year)

first_day = '%i0101' % year
data[:first_day].Temp.plot()

data[:first_day].Rain.plot()

data[:"%i0115" % year].Temp.plot();

mean_temp = data.resample("W").mean().Temp
total_rain = data.resample("W").sum().Rain

ax = mean_temp.plot()
ax.set_ylabel('Temperature')
ax2 = total_rain.plot.line(ax=ax, secondary_y=True)
ax2.set_ylabel('Percipitation')
ax2.set_ylim([0, total_rain.max() * 2.])
ax.legend(loc=2)
ax2.legend(loc='best')

pm.record("year", year)
pm.record("temp", list(mean_temp))
pm.record('rain', list(total_rain))
pm.record("index", list(mean_temp.index))



