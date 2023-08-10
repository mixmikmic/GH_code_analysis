import numpy as np
import altair as alt
import pandas as pd
from astropy.table import Table

kep_data = Table.read('../data/lc_tgas_1arcsec.fits', format='fits')
data = kep_data.to_pandas()

N_tot = len(data)
subsample_ind = np.random.permutation(np.arange(N_tot))[:4999] # random values, no duplication
subsample = data.iloc[list(subsample_ind)]

alt.Chart(subsample).mark_point().encode(
        alt.X('kepmag', bin=True),
        alt.Y('phot_g_mean_mag', bin=True, axis=alt.Axis(title='gaia mag')),
        size='count(*):Q',
    )

gmk = subsample['phot_g_mean_mag'] - subsample['kepmag']
subsample['gaiamag_minus_kepmag'] = gmk

dist = 1.e3/subsample['parallax'] # distance in pc
subsample['dist'] = dist
abs_gmag = subsample['phot_g_mean_mag'] - 5.*(np.log10(dist) - 1.)
subsample['abs_gmag'] = abs_gmag

alt.Chart(subsample).mark_point().encode(
        alt.X('gaiamag_minus_kepmag', bin=alt.Bin(maxbins=20)),
        alt.Y('abs_gmag', bin=alt.Bin(maxbins=20)),
        size='count(*):Q',
    )

brush = alt.selection(type='interval', resolve='global', 
                      on="[mousedown[event.shiftKey], window:mouseup] > \
                      window:mousemove!", zoom='False',
                      translate="[mousedown[event.shiftKey], window:mouseup] > \
                      window:mousemove!")

pan = alt.selection(type='interval', bind='scales',
                    on="[mousedown[!event.shiftKey], window:mouseup] > \
                    window:mousemove!",
                    translate="[mousedown[!event.shiftKey], window:mouseup] > \
                    window:mousemove!")

scale = alt.Scale(domain=['none', 'cand'],
                  range=['#4a69bd', '#e55039'])
color = alt.Color('planet?:N', scale=scale)

chart1 = alt.Chart(subsample).mark_point().encode(
        alt.X('ra', scale=alt.Scale(domain=[270, 310]), 
              axis=alt.Axis(title='Gaia RA (deg)')),
        alt.Y('dec', scale=alt.Scale(domain=[35, 55]),
              axis=alt.Axis(title='Gaia Dec (deg)')),
        color=alt.condition(brush, color, 
                            alt.ColorValue('gray'))
    ).properties(
    selection=brush+pan,
    projection={'type': 'gnomonic'},
    width=450,
    height=500
)

chart2 = alt.Chart(subsample).mark_point().encode(
        alt.X('parallax', axis=alt.Axis(title='G - K (mag)')),
        alt.Y('parallax_error', axis=alt.Axis(title='abs. G')),
        color=alt.condition(brush, color, 
                            alt.ColorValue('gray'))
    ).properties(
    selection=brush,
    width=450,
    height=500
)

chart = chart1 | chart2

chart

url = 'http://gaia-kepler.fun/data/plot_lc_tgas_1arcsec_subsample.csv'

chart = alt.Chart(url).mark_point().encode(
        alt.X('ra_gaia:Q', scale=alt.Scale(domain=[270, 310]), 
              axis=alt.Axis(title='Gaia RA (deg)')),
        alt.Y('dec_gaia:Q', scale=alt.Scale(domain=[35, 55]),
              axis=alt.Axis(title='Gaia Dec (deg)')),
        color=alt.Color('planet?:N')
    ).interactive()

url

from vega_datasets import data

source = data.cars.url
chart = alt.Chart(source).mark_circle().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
    color='Origin:N'
).interactive()

chart

chart = alt.Chart(source).mark_point().encode(
        alt.X('Horsepower:Q'),
        alt.Y('Miles_per_Gallon:Q'),
        color='Origin:N'
    ).interactive()

chart



