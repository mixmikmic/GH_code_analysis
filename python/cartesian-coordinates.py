import numpy
y = numpy.linspace(0, 1, 20) ** 2

import toyplot
canvas, axes, mark = toyplot.plot(y, width=300)

canvas = toyplot.Canvas(width=300)
axes = canvas.cartesian()
axes.plot(y);

x = numpy.linspace(0, 2 * numpy.pi)
y = numpy.sin(x)

import toyplot.locator

canvas = toyplot.Canvas(width=600, height=300)
axes = canvas.cartesian()
axes.label.text = "Trigonometry 101"
axes.x.label.text = "x"
axes.y.label.text = "sin(x)"
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Explicit(
    [0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2, 2 * numpy.pi],
    ["0", u"\u03c0 / 2", u"\u03c0", u"3 \u03c0 / 2", u"2 \u03c0"])
mark = axes.plot(x, y)

x = numpy.linspace(0, 10, 100)
y = 40 + x ** 2

canvas = toyplot.Canvas(300, 300)
axes = canvas.cartesian(label="Toyplot Users", xlabel="Days", ylabel="Users")
mark = axes.plot(x, y)

toyplot.plot(
    x,
    y,
    label="Toyplot Users",
    xlabel="Days",
    ylabel="Users",
    ymin=0,
    width=300);

import toyplot.data
data = toyplot.data.read_csv("deliveries.csv")
data[5:10]

data["Delayed"] = 1.0 - data["On Time"].astype("float64")

canvas = toyplot.Canvas(width=600, height=300)
axes = canvas.cartesian(xlabel="Date", ylabel="Deliveries", ymin=0)
axes.plot(data["Delivered"], color="darkred", marker="o")
axes.y.spine.style = {"stroke":"darkred"}

axes = axes.share("x", ylabel="% Delayed", ymax=0.1)
axes.plot(data["Delayed"].astype("float64"), color="steelblue", marker="o")
axes.y.spine.style = {"stroke":"steelblue"}

