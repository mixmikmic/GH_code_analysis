import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from helper_functions import linear_model_summary

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# This is not good practice, but is appropriate here for a beginners class.
# This allows us to not fuss with image sizes later in the presentation.
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6

get_ipython().system('head ./insects.csv')

insects = pd.read_csv('./insects.csv', sep='\t')

insects

insects.info()

column_names = {
    "continent": "Continent",
    "latitude": "Latitude",
    "wingsize": "Wing Span",
    "sex": "Sex"
}

fig, axs = plt.subplots(2, 2)
for ax, (column, name) in zip(axs.flatten(), column_names.iteritems()):
    ax.hist(insects[column])
    ax.set_title(name)

fig.tight_layout()

fig, ax = plt.subplots()

ax.scatter(insects.latitude, insects.wingsize, s=40)
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Insect Wing Sizes at Various Latitudes")

fig, ax = plt.subplots()

continent_boolean = insects.continent.astype(bool)
ax.scatter(insects.latitude[continent_boolean], 
           insects.wingsize[continent_boolean], 
           s=40, c="red", label="Continent 1")
ax.scatter(insects.latitude[~continent_boolean], 
           insects.wingsize[~continent_boolean],
           s=40, c="blue", label="Continent 0")
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Are The Two Clusters Associated With Continent?")
ax.legend()

fig, ax = plt.subplots()

sex_boolean = insects.sex.astype(bool)
ax.scatter(insects.latitude[sex_boolean], 
           insects.wingsize[sex_boolean],
           s=40, c="red", label="Male")
ax.scatter(insects.latitude[~sex_boolean], 
           insects.wingsize[~sex_boolean],
           s=40, c="blue", label="Female")
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Insect Wing Sizes at Various Latitudes")
ax.legend()

linear_model = smf.ols(formula='wingsize ~ latitude', data=insects)
insects_model = linear_model.fit()
linear_model_summary(insects_model)

fig, ax = plt.subplots()

# Make a scatterplot of the data.
sex_boolean = insects.sex.astype(bool)
ax.scatter(insects.latitude[sex_boolean], 
           insects.wingsize[sex_boolean],
           s=40, c="red", label="Male")
ax.scatter(insects.latitude[~sex_boolean], 
           insects.wingsize[~sex_boolean],
           s=40, c="blue", label="Female")

# Make a linea graph of the predictions.
x = np.linspace(30, 60, num=250)
ax.plot(x, insects_model.params[0] + insects_model.params[1] * x,
       linewidth=2, c="black")

ax.set_xlim(30, 60)
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Insect Wing Sizes at Various Latitudes")
ax.legend()

linear_model = smf.ols(formula='wingsize ~ latitude + sex', data=insects)
insects_model_with_sex = linear_model.fit()
linear_model_summary(insects_model_with_sex)

fig, ax = plt.subplots()

# Make a scatterplot of the data.
sex_boolean = insects.sex.astype(bool)
ax.scatter(insects.latitude[sex_boolean], 
           insects.wingsize[sex_boolean],
           s=40, c="red", label="Male")
ax.scatter(insects.latitude[~sex_boolean], 
           insects.wingsize[~sex_boolean],
           s=40, c="blue", label="Female")

# Make a linea graph of the predictions.
x = np.linspace(30, 60, num=250)
ax.plot(x, insects_model_with_sex.params[0] + insects_model_with_sex.params[1] * x,
       linewidth=2, c="blue")
ax.plot(x, insects_model_with_sex.params[0] + insects_model_with_sex.params[1] * x + insects_model_with_sex.params[2],
       linewidth=2, c="red")

ax.set_xlim(30, 60)
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Insect Wing Sizes at Various Latitudes")
ax.legend()

linear_model = smf.ols(formula='wingsize ~ latitude + sex + continent', data=insects)
insects_model_full = linear_model.fit()
linear_model_summary(insects_model_full)

insects_model_full.params[3] / insects_model_full.bse[3]

fig, ax = plt.subplots()

# Make a scatterplot of the data.
sex_boolean = insects.sex.astype(bool)
ax.scatter(insects.latitude[sex_boolean], 
           insects.wingsize[sex_boolean],
           s=40, c="red", label="Male")
ax.scatter(insects.latitude[~sex_boolean], 
           insects.wingsize[~sex_boolean],
           s=40, c="blue", label="Female")

# Make a linea graph of the predictions.
model_params = insects_model_with_interaction.params
x = np.linspace(30, 60, num=250)
ax.plot(x, model_params[0] + model_params[1] * x,
       linewidth=2, c="blue")
ax.plot(x, model_params[0] + model_params[1] * x + model_params[2] + model_params[3] * x,
       linewidth=2, c="red")

ax.set_xlim(30, 60)
ax.set_xlabel("Latitude")
ax.set_ylabel("Wing Size")
ax.set_title("Insect Wing Sizes at Various Latitudes")
ax.legend()

linear_model = smf.ols(formula='wingsize ~ latitude + sex + sex*latitude', data=insects)
insects_model_with_interaction = linear_model.fit()
linear_model_summary(insects_model_with_interaction)

