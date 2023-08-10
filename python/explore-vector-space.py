get_ipython().magic('run utility')
get_ipython().magic('run widgets-utility')
get_ipython().magic('run model_utility')
get_ipython().magic('run bokeh_plot_utility')

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from IPython.display import display, clear_output
from IPython.core.interactiveshell import InteractiveShell

import ipywidgets as widgets
import bokeh.models as bm
import bokeh.plotting as bp

get_ipython().magic('autosave 120')
get_ipython().magic('config IPCompleter.greedy=True')

InteractiveShell.ast_node_interactivity = "all"
TOOLS = "pan,wheel_zoom,box_zoom,reset,previewsave"

bp.output_notebook()

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')

class ModelState:
    
    def __init__(self, data_folder):
        
        self.data_folder = data_folder
        self.filenames = ModelUtility.get_model_names(data_folder)
        self.filename = self.filenames[0]
        self.wordvector = None
        
    def set_model(self, filename=None):

        filename = filename or self.filename
        self.filename = filename
        self.wordvectors = ModelUtility.load_model_vector(os.path.join(self.data_folder, filename))
        print('Model {} loaded...'.format(self.filename))

state = ModelState('./data')

wdg_basename = widgets.Dropdown(
    options=state.basenames,
    value=state.basename,
    description='Vector Space Model',
    disabled=False,
    layout=widgets.Layout(width='75%')
)
w_filenames = widgets.interactive(state.set_model, basename=wdg_basename)
#state = state.set_model();
#clear_output()
display(widgets.VBox((w_filenames,) + (wdg_model.children[-1],)))
wdg_model.update()

w2vcompute = ModelUtility.compute_most_similar_expression

w2vcompute(word_vectors, "man - boy + girl")[:10]

w2vcompute(word_vectors, "heaven - good + evil")[:10]

word_vectors.most_similar(positive=['heaven', 'evil'], negative=['good'])

w2vcompute(word_vectors, "christ - good + evil")[:3]

w2vcompute(word_vectors, "italy - rome + london")[:3]


# dimensionality reduction - selected word vectors are converted to 2d vectors
def reduce_dimensions(word_vectors, words_of_interest=None, n_components=2, perplexity=30):
    from sklearn.manifold import TSNE

    vectors = word_vectors if words_of_interest is None else [word_vectors[w] for w in words_of_interest]

    tsne_model = TSNE(n_components=n_components, perplexity=perplexity, verbose=0, random_state=0)
    tsne_w2v = tsne_model.fit_transform(vectors)

    # put everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = words_of_interest
    return tsne_df

''' 
total_tsne_df = reduce_dimensions(word_vectors, list(word_vectors.vocab.keys()), n_components=2)
'''

TITLES = { 'title': "Word vectors reduced to XY using T-SNE", 'xlabel': 'X-component', 'ylabel': 'Y-component' }
@interact(w=(0,100),perplexity=(1,100))
def update(w=10,perplexity=30):
    global word_vectors
    words_of_interest = list(word_vectors.vocab.keys())[:w]
    ids_of_interest = [ word_vectors.vocab[x] for x in words_of_interest ]
    tsne_df = reduce_dimensions(word_vectors, words_of_interest,perplexity=perplexity)
    # tsne_df = total_tsne_df.loc[total_tsne_df.words.isin(words_of_interest)]
    p2 = bokeh_scatter_plot_xy_words(tsne_df, line_data=False, **TITLES)
    show(p2)
    

words_of_interest =  ['heaven', 'hell', 'boy', 'girl', 'husband', 'wife', 'son', 'daughter', 'father', 'mother']
tsne_df = reduce_dimensions(word_vectors, words_of_interest)
p3 = bokeh_scatter_plot_xy_words(tsne_df, line_data=True, **TITLES)
show(p3, notebook_handle=True)

holy_words = ['divine', 'hallowed', 'humble', 'pure', 'revered', 'righteous', 'spiritual', 'sublime', 'believing', 'clean', 'devotional', 'faithful', 'good', 'innocent', 'moral', 'perfect', 'upright', 'angelic', 'blessed', 'chaste', 'consecrated', 'dedicated', 'devoted', 'devout', 'faultless', 'glorified', 'god-fearing', 'godlike', 'godly', 'immaculate', 'just', 'messianic', 'pietistic', 'pious', 'prayerful', 'reverent', 'sacrosanct', 'sainted', 'saintlike', 'saintly', 'sanctified', 'seraphic', 'spotless', 'uncorrupt', 'undefiled', 'untainted', 'unworldly', 'venerable', 'venerated']
holy_antonyms = ['lewd', 'nefarious', 'shameless', 'sinful', 'vicious', 'vile', 'wanton', 'warped', 'wicked', 'abandoned', 'base', 'debased', 'debauched', 'degenerate', 'degraded', 'dirty', 'fast', 'low', 'mean', 'perverted', 'twisted', 'vitiate', 'vitiated', 'bad', 'dirty-minded', 'dissolute', 'evil', 'filthy', 'flagitous', 'gone to the dogs', 'kinky', 'lascivious', 'licentious', 'miscreant', 'profligate', 'putrid', 'rotten', 'unhealthy', 'unnatural', 'villainous']

words_of_interest = [ x for x in holy_words + holy_antonyms if x in word_vectors.vocab.keys() ]
tsne_df = reduce_dimensions(word_vectors, words_of_interest)
tsne_df['color'] = tsne_df.words.apply(lambda x: 'green' if x in holy_words else 'firebrick')
p4 = bokeh_scatter_plot_xy_words(tsne_df, **TITLES)
show(p4)

def compute_similarity_to_anthologies(word_vectors, scale_x_pair, scale_y_pair, word_list):

    scale_x = word_vectors[scale_x_pair[0]] - word_vectors[scale_x_pair[1]]
    scale_y = word_vectors[scale_y_pair[0]] - word_vectors[scale_y_pair[1]]

    word_x_similarity = [1 - spatial.distance.cosine(scale_x, word_vectors[x]) for x in word_list ]
    word_y_similarity = [1 - spatial.distance.cosine(scale_y, word_vectors[x]) for x in word_list ]

    df = pd.DataFrame({ 'words': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

    return df

def compute_similarity_to_single_words(word_vectors, word_x, word_y, word_list):

    word_x_similarity = [ word_vectors.similarity(x, word_x) for x in word_list ]
    word_y_similarity = [ word_vectors.similarity(x, word_y) for x in word_list ]

    df = pd.DataFrame({ 'words': word_list, 'x': word_x_similarity, 'y': word_y_similarity })

    return df

def seed_word_toplist(word_vectors, seed_word, topn=100):
     # return [ seed_word ] + [ z[0] for z in word_vectors.most_similar_cosmul(seed_word, topn=topn) ]
     return [ seed_word ] + [ z[0] for z in word_vectors.most_similar(seed_word, topn=topn) ]
    

import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LabelSet, Label, Arrow, OpenHead
from bokeh.plotting import figure, show, output_notebook, output_file

TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"

def bokeh_plot_xy_words(df_words, title='', xlabel='', ylabel='', line_data=False, filename=None, default_color='blue'):

    plot = bp.figure(plot_width=700, plot_height=600, title=title, tools=TOOLS, toolbar_location="above") #, x_axis_type=None, y_axis_type=None, min_border=1)

    color = 'color' if 'color' in df_words.columns else default_color
    
    plot.xaxis[0].axis_label = xlabel
    plot.yaxis[0].axis_label = ylabel

    source = ColumnDataSource(df_words)

    plot.diamond(x='x', y='y', size=8, source=df_words, alpha=0.5, color=color)

    labels = LabelSet(x='x', y='y', text='words', level='glyph',text_font_size="9pt", x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot.add_layout(labels)

    return plot

def show_similarity_to_anthologies(word_vectors, xpair, ypair, word_list):
    word_list = [ x for x in word_list if x in word_vectors.vocab.keys() ]
    df = compute_similarity_to_anthologies(word_vectors, xpair, ypair, word_list)
    xlabel = '{}{}{}'.format(xpair[1], 50 * ' ', xpair[0])
    ylabel = '{}{}{}'.format(ypair[1], 50 * ' ', ypair[0])
    p5 = bokeh_plot_xy_words(df, xlabel=xlabel, ylabel=ylabel)
    show(p5)

xpair = ('west', 'east')
ypair = ('south', 'north')
word_list = holy_words + holy_antonyms
show_similarity_to_anthologies(word_vectors, xpair, ypair, word_list)



