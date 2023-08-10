from bokeh.plotting import figure, output_notebook, show

output_notebook()

p = figure()
p.line([1,2,3,4,5],[6,7,8,9,10], line_width=2)
show(p)

