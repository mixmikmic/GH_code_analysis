import pickle
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from IPython.display import SVG

with open('depictions.pkl', 'rb') as f:
    identifier_depictions = pickle.load(f)

model = KeyedVectors.load('model_300dim.pkl')

model.wv['3315826729']

SVG(identifier_depictions['3315826729'][2])

identifiers = list(identifier_depictions)

X = [model.wv[x] for x in identifier_depictions]

tsne_model = TSNE(n_components=2, random_state=0, perplexity=10, n_iter=1000, metric='cosine')
tsne_pca = tsne_model.fit_transform(X)

from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool
output_notebook()

source = ColumnDataSource(data=dict(x=tsne_pca[:,0], y=tsne_pca[:,1], desc=identifiers,
                                   svgs=[identifier_depictions[x][1] for x in identifiers]))

hover = HoverTool(tooltips="""
    <div>
        <div>@svgs{safe}
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
        </div>
    </div>
    """
)
p = figure(plot_width=700, plot_height=700, tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan',hover],
           title="Mouse over the dots")

p.circle('x', 'y', size=10, source=source, fill_alpha=0.2,);

show(p)



