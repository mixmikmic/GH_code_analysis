import numpy
import toyplot.color
import toyplot.generate

numpy.random.seed(1234)

colormap = toyplot.color.LinearMap(toyplot.color.Palette(["white", "yellow", "red"]))

G0 = toyplot.generate.prufer_tree(numpy.random.choice(4, 12))

canvas = toyplot.Canvas(width=1000, height=500)

axes = canvas.axes(show=False)
axes.aspect = "fit-range"
mark = axes.graph(
    G0,
    vcolor=colormap,
    vstyle={"stroke":"black"},
    vlshow=True,
    vsize=20,
    ecolor="black",
    eopacity=0.2,
);
print "Default vertex labels:"

with open("../docs/dolphins.gml", "r") as file:
    rows = [row.strip() for row in file]
labels = [row[7:-1] for row in rows if row.startswith("label")]
sources = [int(row[6:]) for row in rows if row.startswith("source")]
targets = [int(row[6:]) for row in rows if row.startswith("target")]

colormap = toyplot.color.LinearMap(toyplot.color.Palette(["yellow", "red"]))

canvas = toyplot.Canvas(1000, 500)
axes = canvas.axes(show=False, padding=50)
axes.aspect="fit-range"
axes.graph(
    sources,
    targets,
    layout = toyplot.layout.FruchtermanReingold(area=5),
    vlabel=labels,
    vcolor=colormap,
    vstyle={"stroke":"white", "stroke-width":2.0},
    vlstyle={"font-size":"14px", "text-anchor":"start", "-toyplot-anchor-shift":"10px", "baseline-shift":"10px"},
    vsize=10,
    ecolor="black",
    eopacity=0.2,
);



