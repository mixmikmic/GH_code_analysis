from plotting_utilities import *

ps = RP(n=200, beta=0.001, delta=1.15, c=lambda x: x**1.1)
ps.set_prices()
draw_graph(ps, figsizes=(15,15), scale_factor=20000)
plt.show()

zipf_with_regression(ps)
plt.show()

vas = np.asarray(get_value_added(ps, scale_factor=20000))

vas.mean()

vas.mean() + 6 * vas.std()

vas.max()



ps = RP(n=200, beta=0.001, delta=1.05, c=lambda x: x**1.1)
ps.set_prices()
draw_graph(ps, figsizes=(15,15), scale_factor=40000)
plt.show()

ps = RP(n=200, beta=0.0001, delta=1.15, c=lambda x: x**1.1)
ps.set_prices()
draw_graph(ps, figsizes=(15,15), scale_factor=20000)
plt.show()

ps = RP(n=200, beta=0.001, delta=1.15, c=lambda x: x**1.2)
ps.set_prices()
draw_graph(ps, figsizes=(15,15), scale_factor=50000)
plt.show()



