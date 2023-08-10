from bokeh.charts import Bar, output_notebook, show, vplot, hplot, defaults
from bokeh.sampledata.autompg import autompg as df

output_notebook()

df['neg_mpg'] = 0 - df['mpg']

defaults.width = 550
defaults.height = 400

bar_plot = Bar(df, label='cyl', title="label='cyl'")
show(bar_plot)

bar_plot2 = Bar(df, label='cyl', bar_width=0.4, title="label='cyl' bar_width=0.4")
show(bar_plot2)

bar_plot3 = Bar(df, label='cyl', values='mpg', agg='mean',
                title="label='cyl' values='mpg' agg='mean'")
show(bar_plot3)

bar_plot4 = Bar(df, label='cyl', title="label='cyl' color='DimGray'", color='dimgray')
show(bar_plot4)

# multiple columns
bar_plot5 = Bar(df, label=['cyl', 'origin'], values='mpg', agg='mean',
                title="label=['cyl', 'origin'] values='mpg' agg='mean'")
show(bar_plot5)

bar_plot6 = Bar(df, label='origin', values='mpg', agg='mean', stack='cyl',
                title="label='origin' values='mpg' agg='mean' stack='cyl'", legend='top_right')
show(bar_plot6)

bar_plot7 = Bar(df, label='cyl', values='displ', agg='mean', group='origin',
                title="label='cyl' values='displ' agg='mean' group='origin'", legend='top_right')
show(bar_plot7)

bar_plot8 = Bar(df, label='cyl', values='neg_mpg', agg='mean', group='origin',
                color='origin', legend='top_right',
                title="label='cyl' values='neg_mpg' agg='mean' group='origin'")
show(bar_plot8)

