get_ipython().magic('pylab inline --no-import-all')

from ipywidgets import widgets  
from ipywidgets import interact, interactive, fixed

mean_x = .8
std_x = .15
mean_y = 3.
std_y = .9
N = 1000
x = np.random.normal(mean_x, std_x, N)
y = np.random.normal(mean_y, std_y, N)

counts, bins, patches = plt.hist(x, bins=50, normed=True, alpha=0.3)
gaus_x = mlab.normpdf(bins, mean_x,std_x)

q_for_plot = 1./bins

plt.plot(bins, gaus_x, lw=2)
plt.plot(bins, q_for_plot, lw=2)

plt.xlabel('x')

q_of_x = 1./x

pred_mean_q = 1./mean_x
pred_std_q = np.sqrt((std_x/mean_x)**2)/mean_x

counts, bins, patches = plt.hist(q_of_x, bins=30, normed=True, alpha=0.3)

plt.plot(bins, mlab.normpdf(bins, pred_mean_q, pred_std_q), c='r', lw=2)
plt.legend(('pred','hist'))
plt.xlabel('x')
plt.ylabel('p(x)')

def plot_1_over_x(mean_x, std_x, N):
    x = np.random.normal(mean_x, std_x, N)

    q_of_x = 1./x

    pred_mean_q = 1./mean_x
    pred_std_q = np.sqrt((std_x/mean_x)**2)/mean_x

    counts, bins, patches = plt.hist(q_of_x, 
                                     bins=np.linspace(pred_mean_q-3*pred_std_q,pred_mean_q+3*pred_std_q,30), 
                                     normed=True, alpha=0.3)

    plt.plot(bins, mlab.normpdf(bins, pred_mean_q, pred_std_q), c='r', lw=2)
    plt.legend(('pred','hist'))
    plt.xlabel('x')
    plt.ylabel('p(x)')

# now make the interactive widget
interact(plot_1_over_x,mean_x=(0.,3.,.1), std_x=(.0, 2., .1), N=(0,10000,1000))

def plot_x_plus_y(mean_x, std_x, mean_y, std_y, N):
    x = np.random.normal(mean_x, std_x, N)
    y = np.random.normal(mean_y, std_y, N)

    q_of_x_y = x+y

    pred_mean_q = mean_x+mean_y
    pred_std_q = np.sqrt(std_x**2+std_y**2)

    counts, bins, patches = plt.hist(q_of_x_y, 
                                     bins=np.linspace(pred_mean_q-3*pred_std_q,pred_mean_q+3*pred_std_q,30), 
                                     normed=True, alpha=0.3)

    plt.plot(bins, mlab.normpdf(bins, pred_mean_q, pred_std_q), c='r', lw=2)
    plt.legend(('pred','hist'))
    plt.xlabel('x')
    plt.ylabel('p(x)')


# now make the interactive widget
interact(plot_x_plus_y,
         mean_x=(0.,3.,.1), std_x=(.0, 2., .1), 
         mean_y=(0.,3.,.1), std_y=(.0, 2., .1),
         N=(0,10000,1000))

def plot_x_plus_y(mean_x, std_x, mean_y, std_y, N):
    x = np.random.normal(mean_x, std_x, N)
    y = np.random.normal(mean_y, std_y, N)

    q_of_x_y = x/y

    pred_mean_q = mean_x/mean_y
    pred_std_q = np.sqrt((std_x/mean_x)**2+(std_y/mean_y)**2)*mean_x/mean_y

    counts, bins, patches = plt.hist(q_of_x_y, 
                                     bins=np.linspace(pred_mean_q-3*pred_std_q,pred_mean_q+3*pred_std_q,30), 
                                     normed=True, alpha=0.3)


    plt.plot(bins, mlab.normpdf(bins, pred_mean_q, pred_std_q), c='r', lw=2)
    plt.legend(('pred','hist'))
    plt.xlabel('x')
    plt.ylabel('p(x)')



interact(plot_x_plus_y,mean_x=(0.,3.,.1), std_x=(.0, 2., .1), mean_y=(0.,3.,.1), std_y=(.0, 2., .1),N=(0,100000,1000))



