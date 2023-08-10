import plotly.plotly as py
from data_prep import *

py.sign_in('charleyjoo', 'rx9n91vitl')

get_ipython().magic('matplotlib inline')
plt.ioff()

set_directory()
conn = dbi.connect('crash_database')

get_tables(conn)

conn.close()

crash_data = get_data()
crash_data.head()

plot_x_vs_y()

fig = plot_scatter(x='Average AADT', y='Total Accident Count')
plt.xlim(xmin=0)
py.iplot_mpl(fig, strip_style = True, filename='ipython/mpl_example')

