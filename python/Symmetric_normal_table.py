get_ipython().run_cell_magic('capture', '', 'from scipy.stats import norm as ndist\n%matplotlib inline\nfrom pylab import fill_between\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom ipy_table import make_table\n\ndef symmetric_normal():\n    \n    X = np.linspace(-3.5,3.5,101)\n    D = ndist.pdf(X)\n    z = 1.3\n    fig = plt.figure(figsize=(6,6))\n    ax = fig.gca()\n    ax.plot(X, D, \'k\', linewidth=5)\n    x = np.linspace(-z,z,201)\n    ax.fill_between(x, 0, ndist.pdf(x), facecolor=\'gray\')\n    ax.set_xlabel(\'Standard units\', fontsize=15)\n    ax.set_ylabel(\'Percent per standard units\', fontsize=15)\n    ax.set_ylim([0,.45])\n    ax.annotate(\'Height\\n(%/unit)\', xy=(1, 0.5 * ndist.pdf(z)),\n               arrowprops=dict(facecolor=\'red\'), xytext=(2,0.3),\n               fontsize=15)\n    ax.annotate(\'z=%0.1f\' % z, xy=(1.3, 0),\n               arrowprops=dict(facecolor=\'red\'), xytext=(2,-0.1),\n               fontsize=15)\n    ax.annotate(\'Area (%)\', xy=(0, 0.2),\n               arrowprops=dict(facecolor=\'red\'), xytext=(-3.5,0.3),\n               fontsize=15)\n    return fig \n\ndef symmetric_normal_table(z):\n    """\n    Produce a row of Table A104\n    """\n    if z < 0:\n        raise ValueError(\'z must be nonnegative\')\n    table = [(\'$z$\', \'Height\', \'Area\'),\n             (z, 100*ndist.pdf(z), 100*2*(ndist.cdf(z)-0.5))]\n    return make_table(table)\n\n\n\nwith plt.xkcd():\n    fig = symmetric_normal()')

fig

symmetric_normal_table(1.3)

from ipy_table import make_table
Z = zip(np.linspace(0,1.45,30), np.linspace(1.5,2.95,30), np.linspace(3,4.45,30))
A104 = [('$z$', 'Height', 'Area', '')*3] +     [('%0.2f' % z1, '%0.3f' % (100*ndist.pdf(z1)), '%0.3f' % (100*2*(ndist.cdf(z1)-0.5))) + ('',) +
     ('%0.2f' % z2, '%0.3f' % (100*ndist.pdf(z2)), '%0.3f' % (100*2*(ndist.cdf(z2)-0.5)))+ ('',) +
     ('%0.2f' % z3, '%0.3f' % (100*ndist.pdf(z3)), '%0.3f' % (100*2*(ndist.cdf(z3)-0.5))) + ('',)
                                          for z1, z2, z3 in Z]
Symmetric_Table = make_table(A104)

Symmetric_Table

get_ipython().run_cell_magic('capture', '', "import os\nif not os.path.exists('symmetric_normal.pdf'):\n    fig = symmetric_normal()\n    fig.savefig('symmetric_normal.pdf')")

