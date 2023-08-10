import numpy
import toyplot

numpy.random.seed(1234)

data = numpy.random.normal(size=(12, 2))

canvas = toyplot.Canvas(width=600, height=300)
axes = canvas.cartesian()
mark = axes.plot(data)
markers = mark.markers
axes.label.text = markers[0] + "  This year sales  " + markers[1] + "  Last year sales"

canvas = toyplot.Canvas(width=600, height=300)
axes = canvas.cartesian()
mark = axes.plot(data)
markers = [mark + toyplot.marker.create(shape="o") for mark in mark.markers]
axes.label.text = markers[0] + "  This year sales  " + markers[1] + "  Last year sales"

canvas = toyplot.Canvas(width=600, height=300)
axes = canvas.cartesian()
mark = axes.plot(data)
markers = mark.markers

table = canvas.table(rows=1, columns=4, bounds=(150, -150, 20, 40))
for column, content in enumerate([markers[0].to_html(), "This year sales", markers[1].to_html(), "Last year sales"]):
    table.cells.cell[0, column].data = content
table.cells.column[0::2].width = 20
table.cells.column[1::2].width = 100

