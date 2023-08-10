get_ipython().run_cell_magic('capture', '', '%matplotlib inline\nfrom pylab import fill_between\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom scipy.stats import norm as ndist\nfrom ipy_table import make_table\n\ndef tail_normal():\n    \n    X = np.linspace(-3.5,3.5,101)\n    D = ndist.pdf(X)\n    fig = plt.figure(figsize=(6,6))\n    ax = fig.gca()\n    ax.plot(X, D, \'k\', linewidth=5)\n    x = np.linspace(-5,1.3,201)\n    ax.fill_between(x, 0, ndist.pdf(x), facecolor=\'gray\')\n    ax.set_xlabel(\'Standard units\', fontsize=15)\n    ax.set_ylabel(\'Percent per standard units\', fontsize=15)\n    ax.set_ylim([0,.45])\n    ax.annotate(\'Height\\n(%/unit)\', xy=(1, 0.5 * ndist.pdf(1.3)),\n               arrowprops=dict(facecolor=\'red\'), xytext=(2,0.3),\n               fontsize=15)\n    ax.annotate(\'z=1.3\', xy=(1.3, 0),\n               arrowprops=dict(facecolor=\'red\'), xytext=(2,-0.1),\n               fontsize=15)\n    ax.annotate(\'Area (%)\', xy=(0, 0.2),\n               arrowprops=dict(facecolor=\'red\'), xytext=(-3.5,0.3),\n               fontsize=15)\n    ax.set_xlim([-4,4])\n    return fig\n\ndef tail_normal_table(z):\n    """\n    Produce a row of Table A104\n    """\n    table = [(\'$z$\', \'Height\', \'Area\'),\n             (z, 100*ndist.pdf(z), 100*ndist.cdf(z))]\n    return make_table(table)\n\n\nwith plt.xkcd():\n    fig = tail_normal()')

fig

tail_normal_table(1.3)

from ipy_table import make_table
Z = zip(np.linspace(-4,-2.05,40), np.linspace(-2,-0.05,40), 
        np.linspace(0,1.95,40), np.linspace(2,3.95,40))
tail = [('$z$', 'Height', 'Area', '')*4] +     [('%0.2f' % z1, '%0.3f' % (100*ndist.pdf(z1)), '%0.3f' % (100*ndist.cdf(z1))) + ('',) +
     ('%0.2f' % z2, '%0.3f' % (100*ndist.pdf(z2)), '%0.3f' % (100*ndist.cdf(z2)))+ ('',) +
     ('%0.2f' % z3, '%0.3f' % (100*ndist.pdf(z3)), '%0.3f' % (100*ndist.cdf(z3))) + ('',) +
     ('%0.2f' % z4, '%0.3f' % (100*ndist.pdf(z4)), '%0.4f' % (100*ndist.cdf(z4))) + ('',)
                                          for z1, z2, z3, z4 in Z]
Tail_Table = make_table(tail)

tex_table = r'''
\begin{tabular}{ccc|ccc|ccc|ccc}
%s
\end{tabular}
''' % '\n'.join([' & '.join(r[:3] + r[4:7] + r[8:11] + r[12:15] + (r'\\',)) for r in tail])
file('tail_table.tex', 'w').write(tex_table)

Tail_Table

get_ipython().run_cell_magic('capture', '', "import os\nif not os.path.exists('tail_normal.pdf'):\n    fig = tail_normal()\n    fig.savefig('tail_normal.pdf')")



