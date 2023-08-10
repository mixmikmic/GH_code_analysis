get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import *

ctx = cortex.Cortex(load_default=True)
ctx.configure()
ctx

figure(figsize=(10, 5))
subplot(121)
hist(ctx.edge_lengths, bins=100)
title("Distribution of Edge Lengths")
xlabel("mm")

subplot(122)
hist(ctx.triangle_areas, bins=100)
title("Distribution of Triangle Areas")
xlabel("mm$^2$")

