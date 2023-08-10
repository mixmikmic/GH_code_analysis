get_ipython().magic('pylab nbagg')
from tvb.simulator.lab import connectivity, plot_connectivity

conn = connectivity.Connectivity(load_default=True)
conn.configure()

def plot_with_weights(weights):
    conn = connectivity.Connectivity(load_default=True)
    conn.configure()
    conn.weights = weights
    plot_connectivity(conn, num="tract_mode", plot_tracts=False)

plot_with_weights(conn.scaled_weights(mode='tract'))

plot_with_weights(conn.scaled_weights(mode='region'))
plot_with_weights(conn.scaled_weights(mode='none'))
plot_with_weights(conn.transform_binarize_matrix())
plot_with_weights(conn.transform_remove_self_connections())



