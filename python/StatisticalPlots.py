import warnings
warnings.filterwarnings('ignore')

import holoviews as hv
from holoviews.util.ops import op
from holoviews.operation.datashader import aggregate, shade, datashade, dynspread
hv.notebook_extension('bokeh','matplotlib')

import numpy as np
import pandas as pd

#import pymc3 as pm

print('holoviews version',hv.__version__)

# def var_with_subscript_string(var,n):
#     return '{0}{1}'.format(var,chr(0x2080 + n))
# var_with_subscript_string("\N{GREEK SMALL LETTER SIGMA}", 3)

import unicodedata as ucd
class DimName(object):
    # https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    @staticmethod
    def lcg(s, n=None):
        l = ucd.lookup('greek small letter ' + s) 
        if n is None: return l
        else:         return '{0}{1}'.format(l,chr(0x2080 + n))
    @staticmethod
    def ucg(s, n=None):
        l = ucd.lookup('greek capital letter ' + s) 
        if n is None: return l
        else:         return '{0}{1}'.format(l,chr(0x2080 + n))
    @staticmethod
    def subscript(s,n): return '{0}{1}'.format(s,chr(0x2080 + n))
                                                                     
DimName.lcg('alpha'), DimName.lcg('alpha',1),DimName.lcg('sigma'),  DimName.ucg('sigma',0), DimName.subscript('x',2)

get_ipython().run_cell_magic('opts', "ErrorBars.Pb2 Points.Pb2 HLine.Pb1 (color='darkgreen')", "\n#%%opts ErrorBars [invert_axes=True shared_axes=False show_grid=True  width=600 height=200] (color='slateblue')  {+axiswise}\n#%%opts Points    [invert_axes=True tools=['hover'] toolbar='above'  shared_axes=False] (color='slateblue' marker='+' size=10 line_width=3)  {+axiswise}\n\n# def forest_plot(\n#     df = pd.DataFrame( \n#         {'mean'       : [2.0, 4.0, 6.0,   8.0 ],\n#          'hpd_2.5'    : [0.0, 2.0, 4.0,   6.0 ],\n#          'hpd_97.5'   : [3.0, 5.0, 7.0,   9.0 ],\n#          'sd'         : [0.5, 0.5, 0.5,   0.5 ],},\n#         index=[DimName.lcg('alpha'), DimName.lcg('beta', 0),DimName.lcg('beta', 1),DimName.lcg('sigma')]\n# )):\n#     '''original implementation uses categorical axes'''\n#     rng = (np.floor(df['hpd_2.5'].min()), np.ceil(df['hpd_97.5'].max()) )\n\n#     e_u = df['hpd_97.5']- df['mean']\n#     e_l = df['mean']    - df['hpd_2.5']\n\n#     # Use invisible (i.e., alpha=0) markers to enable hover information\n#     h1 = \\\n#         hv.ErrorBars( [i for i in zip(df.index,  df['mean' ], e_l, e_u)],\n#                       kdims=['parameter'], vdims=['value','e_l','e_u'])\\\n#           .opts(style=dict(line_width=1) ) \\\n#       * hv.ErrorBars( (df.index,  df['mean' ], df['sd' ] ),\n#                       kdims=['parameter'], vdims=['value','sd' ])\\\n#           .opts(style=dict(line_width=5, lower_head=None, upper_head=None) )\n#     h2 = \\\n#         hv.Points( (df.index,  df['mean'   ]), ['parameter', 'value'] ) \\\n#       * hv.Points( (df.index,  df['hpd_2.5']), ['parameter', 'value'] ).opts(style=dict(alpha=0)) \\\n#       * hv.Points( (df.index,  df['hpd_97.5']),['parameter', 'value'] ).opts(style=dict(alpha=0))\n\n#     return (h1*h2).redim.range(value=rng).relabel('95% Credible Intervals')\n\ndef forest_plot(\n    df = pd.DataFrame( \n        {'mean'       : [2.0, 4.0, 6.0,   8.0 ],\n         'hpd_2.5'    : [0.0, 2.0, 4.0,   6.0 ],\n         'hpd_97.5'   : [3.0, 5.0, 7.0,   9.0 ],\n         'sd'         : [0.5, 0.5, 0.5,   0.5 ],},\n        index=[DimName.lcg('alpha'), DimName.lcg('beta', 0),DimName.lcg('beta', 1),DimName.lcg('sigma')]\n), offset=0., group=None):\n    rng   = (np.floor(df['hpd_2.5'].min()), np.ceil(df['hpd_97.5'].max()) )  # improve: this does not work if all data << 1\n    d_rng = (rng[1]-rng[0])*.01\n    rng   = (rng[0]-d_rng, rng[1]+d_rng)\n\n    e_u = df['hpd_97.5']- df['mean']\n    e_l = df['mean']    - df['hpd_2.5']\n    \n    ticks = [ (i,s) for i,s in enumerate(df.index) ]\n    vals  = np.arange(df.shape[0])+offset\n    \n    specs = {\n        'ErrorBars':{'plot' :dict(invert_axes=True, shared_axes=False, show_grid=True,  width=600, height=200),\n                     'style':dict(color='slateblue'),\n                     'group':dict(axiswise=True)},\n        'Points':   {'plot':dict(invert_axes=True, tools=['hover'], toolbar='above', shared_axes=False),\n                     'style':dict(color='slateblue', marker='+', size=10, line_width=3),\n                     'group':dict(axiswise=True)},\n        'HLine':    {'style':dict(color='coral', alpha=0.4, line_width=1.2)}\n    }\n    if group is not None: g={'group':group}\n    else:                 g={}\n\n    # Use invisible (i.e., alpha=0) markers to enable hover information\n    h = \\\n        hv.ErrorBars( [i for i in zip(vals,  df['mean' ], e_l, e_u)],\n                      kdims=['parameter'], vdims=['value','e_l','e_u'], **g)\\\n          .opts(style=dict(line_width=1) ) \\\n      * hv.ErrorBars( (vals,  df['mean' ], df['sd' ] ),\n                      kdims=['parameter'], vdims=['value','sd' ], **g)\\\n          .opts(style=dict(line_width=5, lower_head=None, upper_head=None) ) \\\n      * hv.Points( (vals,  df['mean'   ]), ['parameter', 'value'], **g ) \\\n      * hv.Points( (vals,  df['hpd_2.5']), ['parameter', 'value'], **g ).opts(style=dict(alpha=0)) \\\n      * hv.Points( (vals,  df['hpd_97.5']),['parameter', 'value'], **g ).opts(style=dict(alpha=0)) \\\n      * hv.HLine(0, **g)\n\n    return h.opts(plot=dict(yticks=ticks)).opts(specs).redim.range(parameter=(-.5,df.shape[0]-.5),value=rng).relabel('95% Credible Intervals')\n\nfrom io import StringIO\n\n(forest_plot(group='Pb1') + \\\nforest_plot(df = pd.read_csv(StringIO(\n''',mean,sd,mc_error,hpd_2.5,hpd_97.5\nα,    -14.4, 7.0, 0.2, -29.3, -0.6\nβ__0,   9.2, 2.5, 0.1,   4.4, 14.0\nβ__1, -11.5, 3.8, 0.2, -18.4, -4.1\n'''),index_col=0),group='Pb2')).cols(1)")

get_ipython().run_cell_magic('opts', 'Path Scatter Spread [xticks=4 yticks=4 width=455  show_grid=True]', "%%opts Path    (alpha=0.05 color='indianred') Curve (color='black')\n%%opts Scatter (alpha=0.03 color='slategray' size=5)\n%%opts Spread  (alpha=0.2  color='slategray')\n%%opts ErrorBars (color='black' line_width=2)\n\ndef regression_plot_examples():\n    # Least squares problem:  measure y = a + b x + gaussian error; use least squares to estimate a and b; repeat\n    N=11; NumExperiments = 1000; a=2.; b=1.; sigma=2.8\n    x_vals = np.linspace(-int(N/2),int(N/2),N)\n    y_meas = a + b*np.repeat(x_vals.reshape(1,N),NumExperiments,axis=0)  + sigma*np.random.randn( NumExperiments, N )\n    # compute the least squares estimates for each experiment\n    A              = np.vstack([np.ones(N),x_vals]).T\n    ab_estim,_,_,_ = np.linalg.lstsq(A,y_meas.T)\n    y_estim        = A.dot(ab_estim).T\n    y_estim_ave    = y_estim.mean(axis=0) # averaged y estimates    at each x value\n    y_estim_std    = y_estim.std (axis=0) # std of   y estimates    at each x value\n    y_meas_ave     = y_meas .mean(axis=0) # averaged y measurements at each x value\n    y_meas_std     = y_meas .std (axis=0) # std of   y measurements at each e value\n\n    # Plot the noisy data and the least squares estimates 4 different ways:\n    # 1) plot each of the experiments overlaid with the average line and eror bars on the line\n    # 2) plot each of the lines estimated by least squares overlaid on each other with ave line and eror bars\n    # 3) plot each of the experimantal values overlaid with std of the data and the line estimates\n    #    jitter the x locations to get a better view of the number of points involved\n    # 4) plot the average estimated line overlaid with estimated line values at each x\n    h_data        = hv.Path((x_vals,y_meas.T))\n    x_jitter      = np.random.uniform(low=-.05,high=.05,size=(N*NumExperiments))\n    h_data_pts    = hv.Scatter( ( np.repeat(x_vals, NumExperiments)+x_jitter, y_meas.T.flatten() ))\\\n                        (style=dict(color='darkblue', alpha=0.5, size=5))\n    h_yestim_line = hv.Path((x_vals,y_estim.T))\n    h_yestim_ave  = hv.Curve((x_vals,y_estim_ave))\n    h_yestim_std  = hv.ErrorBars((x_vals,y_estim_ave,y_estim_std))\n    h_yestim_pts  = hv.Scatter( ( np.repeat(x_vals, NumExperiments), y_estim.T.flatten() ))\n    h_spreads_pts = hv.Spread((x_vals, y_estim_ave,y_estim_std))*hv.Spread((x_vals, y_meas_ave, y_meas_std)) *\\\n                    datashade(h_data_pts)\n    #h*h_yestim_ave\n    h=\\\n    (h_data       *h_yestim_ave*h_yestim_std).relabel('Experiments y = %3.2f  + %3.2f x + N(0,%3.2f)'%(a,b,sigma)) +\\\n    (h_yestim_line*h_yestim_ave*h_yestim_std).relabel('Least Squares Estimate Regression Lines') +\\\n     (h_yestim_ave*h_spreads_pts            ).relabel('Standard Deviations of Measurements and Estimates') +\\\n     (h_yestim_ave*h_yestim_pts             ).relabel('Standard Deviations of the Estimated Lines')\n    h.cols(2)\n    return h\n\nregression_plot_examples().relabel('4 Representations of a Regression')")

get_ipython().run_cell_magic('opts', "Curve [show_grid=True width=500 height=300] (color='black')", "%%opts HLine VLine (color='slategray' alpha=0.3)\n%%opts Scatter (size=8 color='coral')\n\nhv.Store.add_style_opts(hv.ErrorBars, ['lower_head', 'upper_head'], 'bokeh')\n\ndef measurements_and_regression_line():\n    N = 10\n\n    x_vals = np.linspace(-10,10,num=N)\n    y_vals = 2. + 1.5*x_vals\n    y_meas = y_vals + np.random.normal(scale=10.,size=N)\n    \n    rng = np.floor(min( y_vals.min(),y_meas.min()) )-1, np.ceil( max(y_vals.max(),y_meas.max()) )+1\n\n    h = \\\n    hv.Curve((x_vals,y_vals)) *hv.HLine(0)*hv.VLine(0) *\\\n    hv.ErrorBars( (x_vals,y_vals,np.zeros(N), y_meas-y_vals), vdims=['y','0','e'])\\\n       .opts(style={'lower_head': None, 'upper_head': None})*\\\n    hv.Scatter((x_vals,y_meas))\n    return h.redim.range(x=(-N-1,N+1), y=(rng))\n\nmeasurements_and_regression_line().relabel('Measurements and Regression Line')")

get_ipython().run_cell_magic('opts', "Curve [show_grid=True width=500 height=300] (color='black')", "%%opts HLine VLine (color='slategray' alpha=0.3)\n%%opts Scatter (size=8 color='coral')\n\ndef estimates_and_line():\n    N = 10\n\n    x_vals = np.linspace(-10,10,num=N)\n    y_vals = 2. + 1.5*x_vals\n\n    y_meas = np.random.uniform(low=-10., high=15.,size=N)\n    e_u    = np.random.uniform(high=12., size=N)\n    e_l    = np.random.uniform(high= 8., size=N)\n\n    rng = np.floor((y_vals-e_l).min()) -1, np.ceil((y_vals+e_u).max())+1\n\n    h = \\\n    hv.Curve((x_vals,y_vals)) *hv.HLine(0)*hv.VLine(0) *\\\n    hv.ErrorBars( (x_vals,y_meas,e_l, e_u), vdims=['y','e_l','e_u'])*\\\n    hv.Scatter((x_vals,y_meas))\n    return h.redim.range(x=(-N-1,N+1), y=(rng))\n\nestimates_and_line().relabel('Measurements and Line')")

get_ipython().run_cell_magic('opts', "Scatter.Individual   [width=600] (color='blue',  size=6)", "%%opts Scatter.Hierarchical [width=600] (color='green', size=6)\n%%opts Path (color='grey' line_width=1)\n%%opts Overlay [legend_position='left']\n\nN=30\nindv_a = np.random.uniform(size=N)\nindv_b = np.random.uniform(size=N)+1.5\nhier_a = indv_a + 0.3*np.random.uniform(size=N)\nhier_b = np.random.uniform(size=N)\n\n# list of arrays with x,y rows; e.g., hv.Path( [ np.array([[0,0],[1,1],[2,0.5],[0,0]])])\nl_paths=[];np.apply_along_axis(lambda x: l_paths.append(x.reshape((2,2))), 1, np.stack([hier_a,hier_b,indv_a,indv_b],axis=1) )\n\nh=\\\nhv.Scatter((indv_a,indv_b), 'intercept', 'slope', group='Individual',   label='individual'  )*\\\nhv.Scatter((hier_a,hier_b), 'intercept', 'slope', group='Hierarchical', label='hierarchical')*\\\nhv.Path(l_paths, ['intercept', 'slope'], label='shrinkage', group='Shrinkage')\n\nh.relabel('Individual/Hierarchical Shrinkage').redim.range(intercept=(0,1.2),slope=(0.,3))")

get_ipython().run_cell_magic('opts', "Scatter [width=500 tools=['hover'] color_index=0 ] (size=2*op('species')+4 color=Cycle(['green', 'blue', 'red']))", '#color=op(\'species\'))\n# do categorical axes a little better\n#    TODO: get hover to display the category\nimport seaborn as sns\niris            = sns.load_dataset("iris")\niris[\'species\'] = iris[\'species\'].astype(\'category\')\niris[\'color\'  ] = 10+10*np.float64(iris[\'species\'].cat.codes)\n\ndef strip_plot(df,x,y,jitter=.2):\n    ticks = [(i,v) for i,v in enumerate(df[x].cat.categories)]\n    return hv.Scatter((np.array([df[x].cat.codes+np.random.uniform(-jitter,jitter,size=df.shape[0]), df[y], df[\'color\']]).T), kdims=[x],vdims=[y,\'color\'])\\\n             .opts(plot=dict(xticks=ticks))\nstrip_plot( iris, \'species\', \'sepal_length\').redim.range(species=(-.5,2.5),sepal_length=(4,8.5))')

get_ipython().run_cell_magic('opts', "Points [width=500 jitter=.4 tools=['hover']] (size=5  color=hv.Cycle(['green','blue','red']))", '# TODO: get hover to display the category\nimport seaborn as sns\niris            = sns.load_dataset("iris")\niris[\'species\'] = iris[\'species\'].astype(\'category\')\n\ndef strip_plot(df,x,y,jitter=.2):\n    ticks = [(i,v) for i,v in enumerate(df[x].cat.categories)]\n    return hv.Points((np.array([df[x].cat.codes, df[y], df[x].cat.codes.astype(float)]).T), kdims=[x,y],vdims=[\'color\'])\\\n             .opts(plot=dict(xticks=ticks, color_index=x))\nstrip_plot( iris, \'species\', \'sepal_length\').redim.range(species=(-.5,2.5),sepal_length=(4,8.5))')

get_ipython().run_cell_magic('opts', 'Curve [width=500]', 'def distribution_example():\n    def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):\n        """Kernel Density Estimation with Scipy"""\n        # Note that scipy weights its bandwidth by the covariance of the\n        # input data.  To make the results comparable to the other methods,\n        # we divide the bandwidth by the sample standard deviation here.\n\n        from scipy.stats import gaussian_kde\n\n        kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)\n        return kde.evaluate(x_grid)\n\n\n    N      = 1000\n    x      = np.random.normal( size = N )\n    x_grid = np.linspace(-4.5, 4.5, 100)\n    h = \\\n    hv.Histogram( np.histogram(x, 20, normed=True) ).opts(style=dict(fill_color=\'slateblue\', alpha=0.4) ) * \\\n    hv.Curve((x_grid, kde_scipy( x, x_grid ))).opts(style=dict( color=\'red\') )\n    return h\n\ndistribution_example().relabel( \'KDE of samples drawn from a Normal Distribution\')')

get_ipython().run_cell_magic('opts', 'Distribution [width=500]', "def gen_distribution_example():\n    N      = 1000\n    x      = np.random.normal( size = N )\n    x_grid = np.linspace(-4.5, 4.5, 100)\n    def distribution_example(bw):\n        # hv now has hv.Distibution\n        h = \\\n        hv.Histogram( np.histogram(x, 20, normed=True) ).opts(style=dict(fill_color='slateblue', alpha=0.4) ).redim.range(Frequency=(0,.6)) * \\\n        hv.Distribution(x).opts(style=dict( fill_color='red', fill_alpha=0.1), plot=dict(bandwidth=bw) )\n        return h\n    return distribution_example\n\nhv.DynamicMap( gen_distribution_example(), kdims='bw' ).redim.values( bw=np.arange(0.01,1, .001))\\\n  .relabel( 'Samples drawn from a Normal Distribution')")

from bokeh.sampledata.iris import flowers
from holoviews.operation import gridmatrix

def splom_example():
    ds = hv.Dataset(flowers)

    grouped_by_species = ds.groupby('species', container_type=hv.NdOverlay)
    grid               = gridmatrix(grouped_by_species, diagonal_type=hv.Histogram)

    plot_opts = dict() #dict(bgcolor='#fafafa')
    style     = dict(alpha=0.5, size=4)

    return grid({'Scatter': {'plot': plot_opts, 'style': style}})
splom_example()

get_ipython().run_cell_magic('opts', 'Overlay [width=380 height=200]', '%%opts HeatMap [colorbar=True colorbar_position=\'left\'] (cmap=\'Blues\')\n%%opts Text (color=\'lightgreen\')\n%%opts Div [height=200]\n\niris = flowers\n\ncorr = iris[iris[\'species\'] != \'virginica\'].corr().abs()\nvals   = np.linspace(1,4,4)\nm_corr = corr.as_matrix().copy()\nm_corr.T[np.triu_indices(4,1)]=np.NaN\n\nimg  = hv.HeatMap((vals,vals,m_corr.T[::-1]), [\'species\',\'Species\']).opts(plot=dict(xticks=corr.columns))\nimg*hv.Overlay([hv.Text(i+1,j+1, \'%.2f\'%m_corr[i,3-j]) for j in range(4) for i in range(4-j)])+\\\nhv.Div( \'<div style=\\"color:darkblue;text-align:left;font-size:16px;\\">Iris dataset correlations (absolute values)</div>\'+corr.round(2).to_html())')

get_ipython().run_cell_magic('opts', "Scatter [width=350 height=350] (color='darkblue' alpha=0.5) Ellipse (color='grey' line_width=3)", "def error_concentration_ellipse_example( A = np.random.multivariate_normal([3,1], cov=[[1,5],[5,15]], size=1000), deviations=4):\n    h = hv.Scatter(A)\n\n    ave        = A.mean(axis=0)\n    A_centered = (A-ave)\n    P          = np.cov(A_centered.T)\n\n    import scipy.linalg as linalg\n\n    U,s,v       = linalg.svd(P)\n    orientation = np.arctan2(U[1,0],U[0,0])\n\n    width  = deviations*np.sqrt(s[0])\n    height = deviations*np.sqrt(s[1])\n\n    h = h*\\\n    hv.Ellipse( ave[0], ave[1], (width,height), orientation=orientation)\\\n        .redim.range(x=(3-6,3+6),y=(1-10,1+10))\n\n    return h.relabel( '%d σ  Error Concentration Ellipse' % deviations)\n\nerror_concentration_ellipse_example()")

get_ipython().run_cell_magic('opts', "Path (alpha=.5 color='black' line_dash='dotted')", "%%opts Path.AXIS [apply_ranges=False] (color='darkgreen' alpha=1 line_width=2 line_dash='solid')\n%%opts Scatter (size=5 alpha=.3) Scatter.A (color='blue') Scatter.B (alpha=0.6 color='indianred')\n%%opts Overlay [width=250 height=250]\ndef apply_covariance_matrix_to_data(covx = np.array([[1,0.8],[0.8,2]]), u = np.random.uniform(-1, 1, (2, 500))  ):\n    N    = u.shape[1]\n    rng  = dict(x=(u[0,:].min(), u[0,:].max()), y=(u[1,:].min(), u[1,:].max()))\n    \n    y = covx @ u                                # apply the covariance matrix as a linear transform\n    p = [np.stack([u[:,i],y[:,i]]) for i in range(0,y.shape[1],10)] # set up the paths connecting points to their transforms\n\n    # compute the singular vectors of the covariance matrix\n    #    and find the axes\n    origin = y.mean(axis=1)\n    e1, v1 = np.linalg.eig(covx)\n    v1[:,0] *= 500*e1[0]; v1[:,1] *= 500*e1[1]\n    a = [np.stack([v1[:,i], origin]) for i in range(2)]\n\n    return (hv.Path(p)*hv.Path(a, group='AXIS')*hv.Scatter(u.T, group='A')*hv.Scatter(y.T, group='B'))\\\n           .redim.range(**rng)\n\ntheta = np.linspace(0, 2*np.pi, 500)\nh=\\\napply_covariance_matrix_to_data()+\\\napply_covariance_matrix_to_data(u = np.stack( [np.cos(theta), np.sin(theta)]))\nh.relabel('Covariance Matrix used as a linear transform').redim.range(x=(-2.5,2.5),y=(-2.5,2.5))")

from holoviews.operation import Operation

class forestplot(Operation):
    """
    boxwhisker plot.
    """
    
    label = param.String(default='ForestPlot', doc="""
        Defines the label of the returned Element.""")
    
    def _process(self, element, key=None):
        # Get first and second Element in overlay
        el = element.get(0)
        
        # Get x-values and y-values of curves
        xvals  = el1.dimension_values(0)
        yvals  = el1.dimension_values(1)
        yvals2 = el2.dimension_values(1)
        
        # Return new Element with subtracted y-values
        # and new label
        return el1.clone((xvals, yvals-yvals2),
                         vdims=[self.p.label])

