import plotly
plotly.__version__

import plotly.plotly as py
from plotly.tools import FigureFactory as FF

#py.sign_in('user', 'API_key')

df = [dict(Task="Job A", Start='2009-01-01', Finish='2009-02-30'),
      dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
      dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')]

fig = FF.create_gantt(df)
py.iplot(fig, filename='Simple Gantt Chart', world_readable=True)



