import numpy
import toyplot.color
import toyplot.generate

numpy.random.seed(1234)

colormap = toyplot.color.LinearMap(toyplot.color.Palette(["white", "yellow", "red"]))

G0 = toyplot.generate.prufer_tree(numpy.random.choice(4, 12))
G1 = G0[:-4]
G2 = G0[4:]

canvas = toyplot.Canvas(width=1000, height=500)

axes = canvas.axes(grid=(1, 2, 0), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

vcoordinates = numpy.ma.masked_all((mark.vcount, 2))
vcoordinates[0] = (-1, 0)
vcoordinates[1] = (0, 0)
vcoordinates[2] = (1, 0)

axes = canvas.axes(grid=(1, 2, 1), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    vcoordinates=vcoordinates,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

print "Pinned Vertices:"

canvas = toyplot.Canvas(width=1000, height=500)

axes = canvas.axes(grid=(1, 2, 0), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

axes = canvas.axes(grid=(1, 2, 1), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G1,
    olayout = mark,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

print "Shared Layout (Subset):"

canvas = toyplot.Canvas(width=1000, height=500)

axes = canvas.axes(grid=(1, 2, 0), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G2,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

axes = canvas.axes(grid=(1, 2, 1), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    olayout = mark,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

print "Shared Layout (Superset):"

extra_vids = numpy.arange(50, 55)

canvas = toyplot.Canvas(width=1000, height=500)

axes = canvas.axes(grid=(1, 2, 0), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G2,
    extra_vids,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

axes = canvas.axes(grid=(1, 2, 1), show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    extra_vids,
    olayout = mark,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);

print "Shared Layout (Disconnected Vertices):"



