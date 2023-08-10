import conx as cx
import math

def sin(x):
    return math.sin(12 * x)

def polynomial(x):
    return -2 * x ** 4 + x ** 3 + x ** 2 - 2 * x + 0

functions = [abs, polynomial, sin]
n_hidden_units = [2, 4, 16] 

get_ipython().run_cell_magic('time', '', 'results = []\nfor n in n_hidden_units:\n    name = "%d hidden units" % (n,)\n    net = cx.Network(name)\n    net.add(cx.Layer("input", 1), \n            cx.Layer("hidden", n, activation="sigmoid"),\n            cx.Layer("output", 1))\n    net.connect()\n    pics = [net.picture(format="image")]\n    for function in functions:\n        net.name = "%s with %d hidden units" % (function.__name__, n)\n        net.compile(error="mse", optimizer="adam", lr=0.003) # lr = learning rate\n        inputs = [[v] for v in cx.frange(-1, 1.02, .02)]\n        targets = [[function(v[0])] for v in inputs]\n        net.reset()\n        net.dataset.load(inputs=inputs, targets=targets)\n        net.train(10000, accuracy=1.0, tolerance=.1, \n                  batch_size=len(inputs), verbose=0, report_rate=1000, plot=True)\n        outputs = [x[0] for x in [net.propagate(v) for v in inputs]]\n        pics.append(cx.plot([["True", targets], ["Predicted", outputs]],\n                            xs=[v[0] for v in inputs],\n                            title="%s(x) with %d hidden unts" % (function.__name__, n),\n                            format="image"))\n    results.append(pics)')

cx.view_image_list(results[0] + results[1] + results[2], 
                   layout=(3, None), pivot=True, scale=20.0)



