get_ipython().run_cell_magic('capture', '', "%matplotlib inline\nfrom pylab import fill_between\nimport matplotlib.pyplot as plt\nimport numpy as np\nfrom scipy.stats import t as tdist\n\ndef tail_t():\n    df = 20\n    X = np.linspace(-3.5,3.5,101)\n    D = tdist.pdf(X, df)\n    fig = plt.figure(figsize=(6,6))\n    ax = fig.gca()\n    ax.plot(X, D, 'k', linewidth=5)\n    x = np.linspace(1.72,5,201)\n    ax.fill_between(x, 0, tdist.pdf(x, df), facecolor='gray')\n    ax.set_xlabel('$T$ units', fontsize=15)\n    ax.set_ylabel('Percent per $T$ units', fontsize=15)\n    ax.set_ylim([0,.45])\n    ax.annotate('Area\\n(%/unit): top row in table\\n 5%', xy=(2.0, 0.3 * tdist.pdf(1.72, df)),\n               arrowprops=dict(facecolor='red'), xytext=(2,0.3),\n               fontsize=15)\n    ax.annotate('Quantile: entries in table, (1.72 for df=20)', xy=(1.72, 0),\n               arrowprops=dict(facecolor='red'), xytext=(2,-0.1),\n               fontsize=15)\n    ax.set_xlim([-4,4])\n    return fig\n\nwith plt.xkcd():\n    fig = tail_t()")

fig

from ipy_table import make_table
qs = [0.25,0.1,0.05,0.025,0.01,0.005]
dfs = np.arange(1,26)
tail = ([['df'] + ['%0.1f' % (100*q) for q in qs]] +
        [[df] + ['%0.2f' % tdist.ppf(1-q, df) for q in qs] for df in dfs])
Tail_Table = make_table(tail)
tex_table = r'''
\begin{tabular}{c|cccccc}
%s
\end{tabular}
''' % '\n'.join([' & ' .join([str(s) for s in r]) + r' \\' for r in tail])
file('tail_table_T.tex', 'w').write(tex_table)

Tail_Table

get_ipython().run_cell_magic('capture', '', "import os\nif not os.path.exists('tail_T.pdf'):\n    fig = tail_t()\n    fig.savefig('tail_T.pdf')")



