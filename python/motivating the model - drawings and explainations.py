get_ipython().magic('pylab inline')
import circlesquare
import pandas as pd

product = circlesquare.CircleSquare()
product.make_pts(50)
product.draw(axes=plt.subplot(1,1,1))
plt.title('Vulnerabilities in the Code Base', fontsize=14);

view1 = product.new_interface('Viewer 1')
view1.make_circles(max_area=.01)
view1.draw(axes=plt.subplot(1,1,1))

view2 = product.new_interface('Viewer 2')
view2.make_circles(max_area=.01)
view2.draw(axes=plt.subplot(1,1,1))

plt.figure(figsize(8,4))
view1.draw(axes=plt.subplot(1,2,1))
plt.title('Actor 1')
view2.draw(axes=plt.subplot(1,2,2))
plt.title('Actor 2')

view1.draw(axes=plt.subplot(1,1,1))

x_seek = np.random.rand()
y_seek = np.random.rand()

plt.hlines(y_seek, 0, 1)
plt.vlines(x_seek, 0, 1)

view1.harden(10)
view1.update()
view1.draw(axes=plt.subplot(1,1,1))

