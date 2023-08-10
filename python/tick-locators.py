import numpy
x = numpy.arange(20)
y = numpy.linspace(0, 1, len(x)) ** 2

import toyplot
canvas, axes, mark = toyplot.plot(x, y, width=300)

canvas, axes, mark = toyplot.plot(x, y, width=300)
axes.x.ticks.locator = toyplot.locator.Uniform(count=5)

canvas, axes, mark = toyplot.plot(x, y, width=300)
axes.x.ticks.locator = toyplot.locator.Uniform(count=5, format="{:.2f}")

canvas, axes, mark = toyplot.plot(x, y, xscale="log10", width=300)

canvas, axes, mark = toyplot.plot(x, y, xscale="log10", width=300)
axes.x.ticks.locator = toyplot.locator.Log(base=10, format="{base}^{exponent}")

canvas, axes, mark = toyplot.plot(x, y, xscale="log2", width=300)
axes.x.ticks.locator = toyplot.locator.Log(base=2, format="{:.0f}")

numpy.random.seed(1234)
canvas, table = toyplot.matrix(numpy.random.random((5, 5)), width=300)

canvas, table = toyplot.matrix(numpy.random.random((50, 50)), width=400, step=5)

fruits = ["Apples", "Oranges", "Kiwi", "Miracle Fruit", "Durian"]
counts = [452, 347, 67, 21, 5]

canvas, axes, mark = toyplot.bars(counts, width=400, height=300)
axes.x.ticks.locator = toyplot.locator.Explicit(labels=fruits)

x = numpy.linspace(0, 2 * numpy.pi)
y = numpy.sin(x)
locations=[0, numpy.pi/2, numpy.pi, 3*numpy.pi/2, 2*numpy.pi]

canvas, axes, mark = toyplot.plot(x, y, width=500, height=300)
axes.x.ticks.locator = toyplot.locator.Explicit(locations=locations, format="{:.2f}")

labels = ["0", u"\u03c0 / 2", u"\u03c0", u"3\u03c0 / 2", u"2\u03c0"]

canvas, axes, mark = toyplot.plot(x, y, width=500, height=300)
axes.x.ticks.locator = toyplot.locator.Explicit(locations=locations, labels=labels)

import toyplot.data
data = toyplot.data.read_csv("commute-obd.csv")
data[:6]

import arrow
timestamps = numpy.array([arrow.get(datetime).timestamp for datetime in data["Datetime"]])

observations = numpy.logical_and(data["Name"] == "Vehicle Speed", data["Value"] != "NODATA")
x = timestamps[observations]
y = data["Value"][observations]

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time", ylabel="km/h",
    width=600, height=300)

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time", ylabel="km/h",
    width=600, height=300)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp()

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time (US/Mountain)", ylabel="km/h",
    width=600, height=300)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp(timezone="US/Mountain")

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time (US/Mountain)", ylabel="km/h",
    width=600, height=300)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp(timezone="US/Mountain", count=5)

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time (US/Mountain)", ylabel="km/h",
    width=600, height=300)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp(
    timezone="US/Mountain", format="{0:HH}:{0:mm}")

canvas, axes, mark = toyplot.plot(
    x, y, label="Vehicle Speed", xlabel="Time (US/Mountain)", ylabel="km/h",
    width=600, height=300)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp(
    timezone="US/Mountain", interval=(7, "minutes"))

canvas = toyplot.Canvas(width=600, height=350)
axes = canvas.cartesian(bounds=(80, -80, 50, -120), label="Vehicle Speed", ylabel="km/h")
axes.plot(x, y)
axes.x.ticks.show = True
axes.x.ticks.locator = toyplot.locator.Timestamp(
    timezone="US/Mountain", format="{0:MMMM} {0:d}, {0:YYYY} {0:h}:{0:mm} {0:a}")
axes.x.ticks.labels.angle = 30
canvas.text(300, 320, "Time (US/Mountain)", style={"font-weight":"bold"});

