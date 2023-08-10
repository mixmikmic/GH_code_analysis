# General purpose libraries
# A nice library for reading in csv data
import pandas as pd
# A library which most visualization libraries in Python are built on.
# We will start by using it to make some plots with pandas
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
# A library for doing math
import numpy as np
# A library for turning unicode fields into ASCII fields
import unicodedata
# a regex library
import re
# a class which makes counting the number of times something occurs in a list easier
from collections import Counter

# some functions for displaying html in a notebook
from IPython.core.display import display, HTML

# A library to visualize holes in a dataset
import missingno as msno

# a fancy library for numerical plots
import seaborn as sns

# Libraries for Word Trees
# lets us use graphviz in python
from pydotplus import graphviz
# to display the final Image
from IPython.display import Image

# Libraries interactive charts
from bokeh.io import output_notebook
# display interactive charts inline
output_notebook()
from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, FixedTicker, ColumnDataSource, LogColorMapper
# to make patches into glyphs and treat counties and states differently
from bokeh.models.glyphs import Patches

try:
    # shape files for US counties
    from bokeh.sampledata.us_counties import data as counties
    # shape files for US states
    from bokeh.sampledata import us_states as us_states_data
except RuntimeError as e:
    # comment these two lines out if you have previously run them
    import bokeh.sampledata as sampledata
    sampledata.download()
    
    # shape files for US counties
    from bokeh.sampledata.us_counties import data as counties
    # shape files for US states
    from bokeh.sampledata import us_states as us_states_data

url = 'https://docs.google.com/spreadsheets/d/1Uz3F_jCNHTo84k1jVQGGa_n_ZQpBUNKyI0ij6Bc3AlM/export?format=csv'
# url = 'data/EarlySeptemberPosts.csv' # uncomment this line and set the path if you have downloaded the data locally

# load the data
posts = pd.read_csv(url)
posts.head()

posts['CreationDate'] = pd.to_datetime(posts['CreationDate'])

msno.matrix(posts)

msno.heatmap(posts)

post_type_counts = posts['PostType'].value_counts()
post_type_counts.plot(kind='bar', color='DarkBlue') # use color or each bar will be colored differently
plt.show()

TOOLS = "pan,wheel_zoom,reset,hover,save"

p = figure(
    title="Post Types",
    tools=TOOLS,
    x_range=post_type_counts.index.tolist(),
#     y_axis_type="log",
    plot_height=400
)
p.vbar(x=post_type_counts.index.values, top=post_type_counts.values, width=0.9)

hover = p.select_one(HoverTool)
hover.point_policy = "follow_mouse"
hover.tooltips = [("Number of Posts", "@top")]

show(p)

questions = posts[posts['PostType'] == 'Question']

print("Questions")
msno.bar(questions)

answers = posts[posts['PostType'] == 'Answer']

print("Answers")
msno.bar(answers)

null_user_id_posts = posts[posts['UserId'].isnull()]

# a function to turn a post into a link to that post on stackoverflow.com
def get_link(p, desc='has no UserId'):
    # the link to a post is different for answers and questions
    if p['PostType'] == 'Answer':
        link = '"https://stackoverflow.com/questions/{0}#answer-{1}"'.format(int(p['QuestionId']), int(p['PostId']))
    else:
        link = '"https://stackoverflow.com/questions/{0}"'.format(int(p['PostId']))
    return '<a href='+link+' target="_blank">{2} {0} {1}</a>'.format(int(p['PostId']), desc, p['PostType'])

# Take the first couple posts without user ids, turn them into links
# join those links with the <br/> tag, and display the result as HTML
null_user_id_links = null_user_id_posts.head().apply(lambda p: get_link(p), axis=1)
display(HTML('<br/>'.join(null_user_id_links)))

# get posts tagged 'python'
tag = 'python'
python_titles = questions[questions['Tags'].str.contains('<'+tag+'>')]['Title'].str.lower()
word_counter = Counter([word for phrase in python_titles.tolist() for word in re.split("[^\w]+", phrase)])

# get frequent words without much meaning (stop words) for English
url = 'https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt'
# url = 'data/english.txt' # uncomment this line and add the path if you have downloaded the English stop words
stop_words = pd.read_csv(url, header=None)[0].tolist()

stop_word_count = 0
words_to_use = 50

# a list to collect words in the top 50 that are not stop words
totals = []
# a dataframe to occur whether or not a word appears in each post title
occurs_in_post = pd.DataFrame()
for word, count in word_counter.most_common(words_to_use):
    # also filter out the empty string
    if word in stop_words or len(word) == 0:
        stop_word_count += 1
        continue
    totals.append((word, count))
    occurs_in_post[word] = python_titles.apply(lambda s: 1 if word in re.split("[^\w]+", s) else 0)

print('{0} of the top {1} words were stop words'.format(stop_word_count, words_to_use))

# plot word counts
totals = pd.DataFrame(totals, columns=['word', 'count'])
totals.set_index('word', inplace=True)
totals.plot(kind='bar')
plt.show()

# get the correlation between word occurances
corr = occurs_in_post.corr()

# Generate a mask for the diagonal
mask = np.zeros_like(corr, dtype=np.bool)
diagonal_index = np.diag_indices(len(mask))
mask[diagonal_index] = True

# get the maximum value not on the diagonal
corr_values = corr.values.copy()
corr_values[diagonal_index] = 0
corr_values = [val for row in corr_values for val in row]
max_corr = max(corr_values)

sns.clustermap(corr, mask=mask, cmap="BrBG", vmax=max_corr, center=0)
plt.show()

# a variable to help us mark nodes as distinct when they have the same label
node_counter = 0

# a class to keep track of a node and it's connections
class Node:
    def __init__(self, word, count, matching_strings, graph, reverse=False, branching=3, highlight=False):
        # make sure each node gets a unique key
        global node_counter
        node_counter += 1
        # let's add a ring around the root node to make it very obvious
        if highlight:
            self.node = graphviz.Node(node_counter, label=word+'\n'+str(count), peripheries=2, fontsize=20)
        else:
            self.node = graphviz.Node(node_counter, label=word+'\n'+str(count))
        # add node to graph
        graph.add_node(self.node)
        # build children
        if count > 1 and len(matching_strings) > 0:
            self.generate_children(matching_strings, graph, reverse, branching)
    
    # a helper function for adding children
    def add_edge(self, graph, c_node, reverse):
        if reverse:
            graph.add_edge(graphviz.Edge(c_node.node, self.node))
        else:
            graph.add_edge(graphviz.Edge(self.node, c_node.node))
    
    # a function to generate the children of a node
    def generate_children(self, matching_strings, graph, reverse, branching):
        # filter out empty strings
        matching_strings = matching_strings[matching_strings.apply(len) > 0]
        # get a count of words which come after this one
        all_children = Counter(matching_strings.apply(lambda x:x[-1 if reverse else 0]))
        # get the children up to the branch number
        children = all_children.most_common(branching)
        # add top <branching> children
        for word, count in children:
            # calculate the matching strings for a child
            if not reverse:
                child_matches = matching_strings[matching_strings.apply(lambda x:x[0]) == word].apply(lambda x:x[1:])
            else:
                child_matches = matching_strings[matching_strings.apply(lambda x:x[-1]) == word].apply(lambda x:x[:-1])
            # build child node and add edge
            c_node = Node(word, count, child_matches, graph=graph, reverse=reverse, branching=branching)
            self.add_edge(graph, c_node, reverse)
        # add an edge to represent the left over children
        left_over = sum(all_children.values()) - sum([x[1] for x in children])
        if left_over > 0:
            c_node = Node('...', left_over, [], graph=graph, reverse=reverse, branching=branching)
            self.add_edge(graph, c_node, reverse)

# some functions to build a word tree
def build_tree(root_string, suffixes, prefixes):
    graph = graphviz.Dot()
    root = Node(root_string, len(suffixes), suffixes, graph, reverse=False, highlight=True)
    root.generate_children(prefixes, graph, True, 3)
    return Image(graph.create_png())

def get_end(string, sub_string, reverse):
    side = 0 if reverse else -1
    return [x for x in re.split(r'[^\w]+', string.lower().split(sub_string)[side]) if len(x) > 0]

def select_text(phrase):
    series = questions['Title']
    instances = series[series.str.lower().str.contains(phrase)]
    suffixes = instances.apply(lambda x: get_end(x, phrase, False))
    prefixes = instances.apply(lambda x: get_end(x, phrase, True))
    return build_tree(phrase, suffixes, prefixes)

select_text('using python')

select_text('september')

posts['CreationDate'].apply(lambda x: x.hour).hist(bins=range(24))
plt.xlabel('Hour of day on a 24 hour clock')
plt.ylabel('Number of posts')
plt.show()

# aggregate answers by question id
answers_by_question = answers.groupby('QuestionId')['CreationDate'].agg(min)
# get the earliest creation date for each answer
first_reply = pd.DataFrame({'PostId':answers_by_question.index.values, 'EarliestReply':answers_by_question.values})
# add the time of the earliest answer to the questions data frame (filtering out questions which were not answered)
first_reply = pd.merge(first_reply, questions, how='inner', on=['PostId'])

# get the time it took to get an answer
first_reply['Latency'] = (first_reply['EarliestReply']-first_reply['CreationDate'])
# convert to minutes
first_reply['Latency'] /= pd.Timedelta(minutes=1)

# find the median
print('Median answer time for questions asked and answered in the first two weeks of September 2017 is {0:.2f} min.'.format(first_reply['Latency'].median()))

# Let's plot the data
first_reply['Latency'].hist(bins=50)
plt.yscale('log', nonposy='clip')
plt.ylabel('Number of Questions')
plt.xlabel('Time in Minutes')
plt.show()

weird_questions = first_reply[first_reply['EarliestReply'] < first_reply['CreationDate']]
links = weird_questions.apply(lambda p: get_link(p, desc='answered before question'), axis=1)
display(HTML('<br/>'.join(links)))

some_tags = ['javascript', 'python', 'java', 'r', 'c++', 'c']
df = pd.DataFrame()
for t in some_tags:
    t_search = t.replace('+','\+')
    tag_items = pd.DataFrame(first_reply[first_reply['Tags'].str.contains('<'+t_search+'>')]['Latency'])
    tag_items['Tag'] = t
    df = df.append(tag_items)

# uncomment the following line if you would like the chart sorted by the median latency
# some_tags.sort(key = lambda t: df[df['Tag'] == t]['Latency'].median())

ax = sns.boxplot(x="Tag", y="Latency", order=some_tags, data=df, showfliers=False)
ax.set(yscale="log")
plt.ylabel('Latency in Minutes (Log Scale)')
plt.show()

# filter out null users and get question ids
from_edges = answers.loc[answers['UserId'].notnull(),['UserId', 'QuestionId']]
from_edges.rename(columns={'UserId':'AnswerUID', 'QuestionId':'PostId'}, inplace=True)
# filter out null users and get question ids
to_edges = questions.loc[questions['UserId'].notnull(),['UserId','PostId']]
# merge on question id
links = pd.merge(from_edges, to_edges, on='PostId', how='inner')
# use a counter to merge duplicate edges to get edge weights
edges = Counter(links.apply(lambda x:(x['AnswerUID'], x['UserId']), axis=1).tolist())
edges.most_common(10)

repuatation_map = dict(zip(posts['UserId'], posts['Reputation']))

# a function to get a fill color based on reputation
def get_fill(uid):
    rep = repuatation_map[int(uid)]
    # The distribution of reputations is very lop-sided, so let's use log reputation for our scale
    # the value should be between 0 and 255
    val = np.log(rep)/np.log(max(repuatation_map.values())) * 255
    return to_hex([val, 0, val])

# turn a RGB triplet into a hex color graphviz will understand
def to_hex(triple):
    output = '#'
    for val in triple:
        # the hex function returns a string of the form 0x<number in hex>
        val = hex(int(val)).split('x')[1]
        if len(val) < 2:
            val = '0'+val
        output += val
    return output

# The function to visualize our network graph
# It takes in a list of edges with weights
def build_network(edges_with_weights, prog='neato'):
    # The function which builds each node. You can change the node style here.
    make_node = lambda uid: graphviz.Node(uid, label='', shape='circle', style='filled', fillcolor=get_fill(uid), color='white')
    graph = graphviz.Dot()
    # A dictionary to keep track of node objects
    nodes = {}
    for pair in edges_with_weights:
        e, w = pair
        e = (str(int(e[0])), str(int(e[1])))
        # Add notes to the graph if they don't exist yet
        if e[0] not in nodes:
            nodes[e[0]] = make_node(e[0])
            graph.add_node(nodes[e[0]])
        if e[1] not in nodes:
            nodes[e[1]] = make_node(e[1])
            graph.add_node(nodes[e[1]])
        graph.add_edge(graphviz.Edge(nodes[e[0]], nodes[e[1]], penwidth=(float(w)/2)))
    return Image(graph.create_png(prog=prog))

# Let's build a small network from the edges with the highest weights.
build_network(edges.most_common(10))

def show_self_links(uid):
    self_links = links.loc[(links['AnswerUID'] == uid) & (links['UserId'] == uid),:].copy()
    self_links['PostType'] = 'Question'
    display(HTML('<br/>'.join(self_links.apply(lambda p: get_link(p, 'is a self link for '+str(uid)), axis=1))))

for self_linker in [x[0][0] for x in edges.most_common(10) if x[0][0] == x[0][1]]:
    show_self_links(self_linker)

def find_connected_subgraphs(edges):
    nodes = list(set([n for e in edges for n in e]))
    mappings = dict(zip(nodes, range(len(nodes))))
    flipped_mappings = dict(zip(range(len(nodes)), [[n] for n in nodes]))
    for e in edges:
        c_1 = mappings[e[0]]
        c_2 = mappings[e[1]]
        if c_1 == c_2:
            continue
        if len(flipped_mappings[c_1]) > len(flipped_mappings[c_2]):
            tmp = c_1
            c_1 = c_2
            c_2 = tmp
        for n in flipped_mappings[c_1]:
            mappings[n] = c_2
            flipped_mappings[c_2].append(n)
    return mappings

num_edges = 2000
connection_mapping = find_connected_subgraphs([x[0] for x in edges.most_common(num_edges)])
Counter(connection_mapping.values()).most_common(10)

subgraph_id_number = 874
e_list = [x for x in edges.most_common(num_edges) if connection_mapping.get(x[0][0],None) == subgraph_id_number]
build_network(e_list)

# can stack overflow users be partitioned into questioners and answerers?
answerers = set([e[0] for e in edges.keys() if e[0] != e[1]])
questioners = set([e[1] for e in edges.keys() if e[0] != e[1]])
both_question_and_answer = (answerers & questioners)

print('{0:.2f}% of users both answered and asked questions'.format(len(both_question_and_answer)/len(answerers | questioners)))

# let's find a subgraph with users who both asked and answered questions
condition = lambda e: (e[0] in both_question_and_answer) or (e[1] in both_question_and_answer)
connection_mapping = find_connected_subgraphs([e for e in edges if condition(e)])
Counter(connection_mapping.values()).most_common(10)

subgraph_id_number = 6820
e_list = [(e, edges[e]) for e in edges if condition(e) and connection_mapping.get(e[0],None) == subgraph_id_number]
build_network(e_list)

location_data = pd.read_csv('loc_data.csv')
location_data['Google_Data'] = location_data['Google_Data'].apply(lambda x: eval(x))

location_data['Google_Data'].apply(len).hist()
#plt.yscale('log', nonposy='clip')
plt.title('Was Google Able to Find a Unique Location from the Location String?')
plt.xlabel('Number of Matches')
plt.ylabel('Number of Unique Place Strings')
plt.show()

print("Let's only work with data that had a unique match")
location_data = location_data[location_data['Google_Data'].apply(len) == 1]
posts_with_location = pd.merge(posts, location_data, how='inner', on='Location')

# get the lowest level of location data for each row
def get_lowest_component(address_list):
    components = address_list[0]['address_components']
    all_parts = []
    for c in components:
        if len(c['types']) == 2 and c['types'][1] == 'political':
            all_parts.append(c['types'][0])
    if len(all_parts) > 0:
        return all_parts[0]
    return None

posts_with_location = pd.merge(posts, location_data, how='inner', on='Location')
lowest_components = posts_with_location['Google_Data'].apply(get_lowest_component).value_counts()

lowest_components.plot(kind='bar', color='DarkBlue')
plt.title("How precise is our location information?")
plt.show()

def get_part(address_list, level='country', name_type='long_name'):
    components = address_list[0]['address_components']
    for c in components:
        if c['types'] == [level, 'political']:
            return c[name_type]
    return None

for part in lowest_components.index:
    posts_with_location[part] = posts_with_location['Google_Data'].apply(lambda x: get_part(x, level=part))

posts_with_location.head()

def get_lat_long(address_list):
    location = address_list[0]['geometry']['location']
    return location['lat'], location['lng']

posts_with_location['lat_lng'] = posts_with_location['Google_Data'].apply(get_lat_long)
posts_with_location['lat'] = posts_with_location['lat_lng'].apply(lambda x: x[0])
posts_with_location['lng'] = posts_with_location['lat_lng'].apply(lambda x: x[1])
msno.geoplot(posts_with_location, x='lng', y='lat', by='country')

# get posts which were by someone in the United States
county_posts = posts_with_location.loc[posts_with_location['country'] == 'United States',:]
# group those post by state (administrative_area_level_1) and then county (administrative_area_level_1)
# this structure will be used later to get different stats from this aggregation
county_posts = county_posts.groupby(['administrative_area_level_1', 'administrative_area_level_2'])

# a function to count the number of questions with a given tag
def get_tag_count(series, tag='python'):
    return len(series[series.notnull() & (series.str.find('<'+tag+'>') > -1)])

# get basic stats which don't need to be calculated over and over
county_stats = county_posts.agg({'lat':np.mean, 'lng':np.mean, 'PostId':len, 
                                'PostType':lambda x : len(x[x=='Question'])})

county_stats.reset_index(inplace=True)
county_stats = county_stats.rename(columns={'PostId':'posts', 'PostType':'questions', 'administrative_area_level_1':'State', 'administrative_area_level_2':'County'})

# get shape data for counties and states
counties = {
    code: county for code, county in counties.items()
}

us_states = us_states_data.data.copy()

name_to_code = dict([(counties[code]['detailed name'], code) for code in counties])

# a function to match counties from stack overflow data to data from shape files
def match_county(county):
    state = county['State']
    county_name = county['County']
    # take out non-ascii characters which are not in Bokeh file
    county_name = unicodedata.normalize('NFKD', county_name).encode('ascii','ignore').decode("utf-8")
    full_name = county_name + ', ' + state
    if full_name in name_to_code:
        return name_to_code[full_name]
    # some counties end with county in one dataset but not the other
    # in these cases, just match with the closest string from the same state
    close_matches = [n for n in name_to_code.keys() if n.endswith(state) and n.startswith(county_name.split(' ')[0])]
    if len(close_matches) == 0:
        print(full_name)
        return None
    full_name = min(close_matches, key=len)
    return name_to_code[full_name]

# get a code to match stack overflow data to shape file data
county_stats['code'] = pd.Series(county_stats.apply(match_county, axis=1))

# make a function to build the map in Bokeh
def build_map(county_stats, county_posts, county_slice=None, language='python'):
    color_mapper = LogColorMapper(palette=palette)
    
    if county_slice is not None:
        county_stats = county_stats[county_slice]

    county_xs = county_stats['code'].apply(lambda code: counties[code]["lons"]).tolist()
    county_ys = county_stats['code'].apply(lambda code: counties[code]["lats"]).tolist()
    county_names = (county_stats["County"]+', '+county_stats["State"]).tolist()
    
    language_perc = county_posts['Tags'].agg(lambda x: get_tag_count(x, language))
    language_perc = language_perc.reset_index()
    if county_slice is not None:
        language_perc = language_perc[county_slice]
    language_perc = (language_perc['Tags']/county_stats['questions'])
    language_perc = (language_perc*100).tolist()

    posts_source = ColumnDataSource(data=dict(
        x=county_xs,
        y=county_ys,
        name=county_names,
        posts=county_stats['posts'].tolist(),
        questions=county_stats['questions'].tolist(),
        lang_posts=language_perc
    ))
    
    TOOLS = "pan,wheel_zoom,reset,save"

    p = figure(
        title=language.title() + " Posts by County", tools=TOOLS,
        x_axis_location=None, y_axis_location=None,
        plot_width=900
    )
    p.grid.grid_line_color = None

    county_pathches = Patches(xs="x", ys="y",
              fill_color={'field': 'lang_posts', 'transform': color_mapper},
              fill_alpha=0.7, line_color="white", line_width=0.5)
    county_pathches_render = p.add_glyph(posts_source, county_pathches)
    
    # add hover tooltip
    hover = HoverTool(renderers=[county_pathches_render], tooltips=[
        ("Name", "@name"),
        ("Posts", "@posts"),
        ("Questions", "@questions"),
        ("% "+language.capitalize(), "@lang_posts")])
    p.add_tools(hover)
    
    # -----------
    # Add state outlines
    # -----------
    filter_fun = lambda x : x != 'AK' and x != 'HI'
    # get lat and long as x and y
    state_xs = [us_states[code]["lons"] for code in us_states if filter_fun(code)]
    state_ys = [us_states[code]["lats"] for code in us_states if filter_fun(code)]
    
    # draw state lines
    p.patches(state_xs, state_ys, fill_alpha=0.0, line_color="#"+('9'*6), line_width=0.5)

    show(p)

build_map(county_stats, county_posts)

build_map(county_stats, county_posts, county_stats['questions'] > 7, language='python')



