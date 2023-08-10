import numpy
numpy.random.seed(1234)
matrix = numpy.random.normal(loc=1.0, size=(20, 20))

import toyplot
toyplot.matrix(matrix, label="A matrix");

toyplot.matrix(matrix, label="A matrix", colorshow=True);

colormap = toyplot.color.brewer.map("BlueRed", domain_min=-4, domain_max=4)
toyplot.matrix((matrix, colormap), label="A matrix", colorshow=True);

toyplot.matrix((matrix, colormap), label="A matrix", tlabel="Top", llabel="Left", rlabel="Right", blabel="Bottom");

big_matrix = numpy.random.normal(loc=1, size=(50, 50))
toyplot.matrix((big_matrix, colormap), step=5, label="A matrix");

toyplot.matrix((big_matrix, colormap), tshow=False, lshow=False, label="A matrix");

toyplot.matrix((matrix, colormap), rshow=True, bshow=True, label="A matrix");

tlocator = toyplot.locator.Explicit([5, 10, 15])
llocator = toyplot.locator.Explicit([1, 5, 13], ["One", "Five", "Evil"])
toyplot.matrix((matrix, colormap), tlocator=tlocator, llocator=llocator, label="A matrix");

i, j = numpy.unravel_index(numpy.argmax(matrix), matrix.shape)

canvas, table = toyplot.matrix((matrix, colormap), label="A matrix")
table.body.cell[i, j].style = {"fill":"yellow"}

x = numpy.arange(-5, 5, 0.2)
y = numpy.arange(-5, 5, 0.2)
xx, yy = numpy.meshgrid(x, y, sparse=True)
z = xx ** 2 - yy ** 2

canvas, table = toyplot.matrix(z, step=5, width=400)
table.left.cell[25, 1].lstyle = {"fill":"red"}
table.top.cell[1, 25].lstyle = {"fill":"red"}
table.right.cell[25, 0].data = "Saddle"
table.bottom.cell[0, 25].data = "Saddle"

canvas = toyplot.Canvas(width=600, height=400)
table = canvas.matrix(matrix, label="Matrix", bounds=(50, -250, 50, -50), step=5)
axes = canvas.cartesian(bounds=(-200, -50, 50, -50), label="Distribution", xlabel="Count", ylabel="Value")
axes.bars(numpy.histogram(matrix, 20), along="y");



