import numpy
x = numpy.linspace(0, 1)
y = x ** 2

import toyplot
toyplot.plot(x, y, width="3in", height="2in");

toyplot.plot(x, y, width=(3, "in"), height=(2, "in"));

toyplot.plot(x, y, width=600, height=400);

