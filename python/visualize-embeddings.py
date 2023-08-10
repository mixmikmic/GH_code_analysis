import projtools
reload(projtools)

features_file = "/data/africa2017/features/features-rot256.json"
fd = projtools.FeatureDict()
fd.load_json(features_file)

fd.ftr_matrix

fd.ftr_matrix.shape

import sklearn
from sklearn.decomposition import PCA

get_ipython().run_cell_magic('time', '', 'proj_algo = PCA(n_components=2)\npca_proj = proj_algo.fit_transform(fd.ftr_matrix)')

pca_proj.shape

import bokeh
from bokeh.io import output_notebook, show
output_notebook()
from bokeh.plotting import figure

from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource

pca_datasource = ColumnDataSource(data={
        "x": pca_proj[:,0],
        "y": pca_proj[:,1],
        "imgname": fd.names,        
    })
tooltip_html = """
        <div>  <span style="font-size:8px">@imgname</span>
            <img src="http://localhost:8000/@imgname.jpg" height=128 width=128></img>
        </div>
    """
f = figure(tools=["pan,wheel_zoom,box_zoom,reset,tap",HoverTool(tooltips = tooltip_html)],
           plot_width=800, plot_height=600, title="PCA")
f.circle('x','y', source=pca_datasource, size=15, alpha=0.3)
show(f)

get_ipython().run_cell_magic('time', '', 'from sklearn.manifold import TSNE, SpectralEmbedding, MDS\n#proj_algo = MDS()\n#proj_algo = SpectralEmbedding()\nproj_algo = TSNE(perplexity=50, random_state=1234)\ntsne_proj = proj_algo.fit_transform(fd.ftr_matrix)')

tsne_datasource = ColumnDataSource(data={
        "x": tsne_proj[:,0],
        "y": tsne_proj[:,1],
        "imgname": fd.names,        
    })
f = figure(tools=["pan,wheel_zoom,box_zoom,reset,tap",HoverTool(tooltips = tooltip_html)],
           plot_width=800, plot_height=600, title="t-SNE")
f.circle('x','y', source=tsne_datasource, size=15, alpha=0.3)
show(f)



