import dask.array as da

x = da.ones(15, chunks=(5,))

x.visualize()

(x+1).visualize()

(x+1).sum().visualize()

m = da.ones((15, 15), chunks=(5,5))

(m.T + 1).visualize()

(m.T + m).visualize()

(m.dot(m.T + 1) - m.mean(axis=0)).visualize()

(m.dot(m.T + 1) - m.mean(axis=0)).compute()



