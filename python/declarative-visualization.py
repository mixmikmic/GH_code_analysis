# Use this when you want to nbconvert the notebook (used by nbviewer)
from krisk import init_notebook; init_notebook()

from krisk import Chart

chart = Chart()
chart

chart.option

chart.set_title('This is a blank visualization', x_pos='center')
chart.set_theme('vintage')

chart.option['series'] = [{'data': [10, 3, 7, 4, 5], 'name': 'continent', 'type': 'bar'}]
chart.option['xAxis'] =  {'data': ['Americas', 'Asia', 'Africa', 'Oceania', 'Europe']}
chart

