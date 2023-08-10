get_ipython().magic('matplotlib inline')
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from IPython.display import display
from elections import models as election_models
from parliaments import models as parliament_models
from proceedings import models as proceeding_models
from django_extensions.db.fields.json import JSONDict
from collections import OrderedDict
from federal_common.sources import EN, FR
from django.db.models.base import ModelBase
from IPython.display import HTML
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os

def columns_reorder_left(dataframe, columns):
    unmentioned = [column for column in dataframe.columns if column not in columns]
    dataframe = dataframe[columns + unmentioned]
    return dataframe

def columns_reorder_after(dataframe, pairs):
    columns = list(dataframe.columns)
    for left, right in pairs.items():
        columns.remove(right)
        columns.insert(columns.index(left) + 1, right)
    return dataframe[columns]

dataframe_json_mapper = {
    "government_party": "Library of Parliament, History of Federal Ridings",
    "party": "Library of Parliament, History of Federal Ridings",
    "province": "Library of Parliament, Province / Territory File",
    "riding": "Library of Parliament, History of Federal Ridings",
    "parliamentarian": "Library of Parliament, Parliament File",
}

def get_dataframe(qs, mapping, index=None):
    mapping = OrderedDict(mapping)
    qs = qs.objects.all() if isinstance(qs, ModelBase) else qs
    dataframe = pd.DataFrame(
        {
            k: v.get(EN, {}).get(dataframe_json_mapper[k.split("__")[-2]], "") if isinstance(v, dict) else v
            for k, v in row_dict.items()
        }
        for row_dict in qs.values(*[
            key[0] if isinstance(key, tuple) else key
            for key in mapping.keys()
        ])
    )
    dataframe = columns_reorder_left(dataframe, list(mapping.keys()))
    dataframe.rename(columns=mapping, inplace=True)
    if index:
        dataframe = dataframe.set_index(index)
    dataframe = dataframe.sort_index()
    return dataframe

def select_by_index(dataframe, indexes):
    indexer = [slice(None)] * len(dataframe.index.names)
    for key, value in indexes.items():
        indexer[dataframe.index.names.index(key)] = value
    if len(indexer) > 1:
        return dataframe.loc[tuple(indexer), :]
    else:
        return dataframe.loc[tuple(indexer)]

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # calculate evenly-spaced axis angles
    theta = 2 * np.pi * np.linspace(0, 1 - 1. / num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi / 2

    def draw_poly_frame(self, x0, y0, r):
        # TODO: use transforms to convert (x, y) to (r, theta)
        verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_frame(self, x0, y0, r):
        return plt.Circle((x0, y0), r)

    frame_dict = {'polygon': draw_poly_frame, 'circle': draw_circle_frame}
    if frame not in frame_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):
        """Class for creating a radar chart (a.k.a. a spider or star chart)
        http://en.wikipedia.org/wiki/Radar_chart
        """
        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_frame = frame_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180 / np.pi, labels)

        def _gen_axes_patch(self):
            x0, y0 = (0.5, 0.5)
            r = 0.5
            return self.draw_frame(x0, y0, r)

    register_projection(RadarAxes)
    return theta

def display_toggler(button="Click to toggle"):
    return HTML("""
        <script>
            function toggle_cell(event) {{
                $(event ? event.target : ".toggler")
                    .parents(".output_wrapper")
                    .prev()
                    .toggle(event ? undefined : false)
            }}
            $(document).ready(setTimeout(toggle_cell, 0))
        </script>
        <p><button class="toggler" onClick="javascript:toggle_cell(event)">{}</button></p>
    """.format(button))

display_toggler("Click to toggle the display the initializing scripts")

elections = get_dataframe(election_models.GeneralElection, {
    "number": "Election",
    "date": "Date",
    "population": "Population",
    "registered": "Registered",
    "ballots_total": "Ballots (Total)",
    "parliament__seats": "Seats",
    "parliament__government_party__names": "Party",
}, ["Election"])
elections['Date'] = pd.to_datetime(elections['Date'])  # TODO: Why isn't this happening automatically?
ridings = get_dataframe(election_models.ElectionRiding.objects.filter(general_election__isnull=False), {
    "general_election__number": "Election",
    "general_election__date": "Election: Date",
    "riding__province__names": "Province",
    "riding__names": "Riding",
    "population": "Population",
    "registered": "Registered",
    "ballots_rejected": "Ballots (Rejected)"
}, ["Election", "Province", "Riding"])
ridings['Election: Date'] = pd.to_datetime(ridings['Election: Date'])
candidates = get_dataframe(election_models.ElectionCandidate.objects.filter(election_riding__general_election__isnull=False), {
    "election_riding__general_election__number": "Election",
    "election_riding__general_election__date": "Election: Date",
    "election_riding__riding__province__names": "Province",
    "election_riding__riding__names": "Riding",
    "name": "Candidate",
    "party__names": "Party",
    "party__color": "Party: Color",
    "election_riding__population": "Riding: Population",
    "election_riding__registered": "Riding: Registered",
    "ballots": "Ballots",
    "elected": "Elected",
    "acclaimed": "Acclaimed",
}, ["Election", "Province", "Riding", "Candidate"])
candidates['Election: Date'] = pd.to_datetime(candidates['Election: Date'])
house_vote_participants = get_dataframe(proceeding_models.HouseVoteParticipant.objects.all(), {
    "house_vote__sitting__session__parliament__number": "Parliament",
    "house_vote__sitting__session__number": "Session",
    "house_vote__sitting__number": "Sitting",
    "house_vote__number": "Vote",
    "party__names": "Party",
    "parliamentarian__names": "Parliamentarian",
    "recorded_vote": "Parliamentarian: Vote",
}, ["Parliament", "Session", "Sitting", "Vote", "Party", "Parliamentarian"])
house_vote_participants = house_vote_participants.reset_index().fillna("Independent").set_index(house_vote_participants.index.names)

# Some ridings historically had two seats (Halifax, Victoria, etc). For the purposes of
# calculating voter turnout, we need to define a fractional ballot where a voter's single
# ballot is split in half between two candidates.
ridings = ridings.join(candidates[candidates["Elected"] | candidates["Acclaimed"]].reset_index().groupby(ridings.index.names)["Candidate"].count().to_frame())
ridings = ridings.rename(columns={"Candidate": "Seats"})
candidates = candidates.reset_index().set_index(ridings.index.names).join(ridings["Seats"]).reset_index().set_index(candidates.index.names)
candidates["Ballots (Fractional)"] = candidates["Ballots"] / candidates["Seats"]
del candidates["Seats"]

parties = candidates.reset_index().groupby(["Election", "Party"]).agg({
    "Election: Date": "first",
    'Candidate': 'count',
    'Ballots (Fractional)': 'sum',
    'Elected': 'sum',
    'Acclaimed': 'sum',
    'Party: Color': 'first',
}).rename(columns={
    'Candidate': 'Candidates',
    'Ballots (Fractional)': 'Ballots',
    "Party: Color": "Color",
})
parties["Seats"] = parties["Elected"] + parties["Acclaimed"]
del parties["Elected"]
del parties["Acclaimed"]

# Augment elections with winning party data
elections = elections.reset_index().set_index(parties.index.names).join(parties, rsuffix=" (Party)").reset_index().set_index(elections.index.names).rename(columns={
    'Candidates': 'Party: Candidates',
    'Ballots': 'Party: Ballots',
    'Color': 'Party: Color',
    'Seats (Party)': "Party: Seats",
})
del elections["Election: Date"]
elections = columns_reorder_after(elections, {
    "Seats": "Party",
})

# Copy valid ballots per candidate up to the ridings level
ridings = ridings.join(candidates.reset_index().groupby(ridings.index.names)["Ballots (Fractional)"].sum().to_frame())
ridings = ridings.rename(columns={"Ballots (Fractional)": "Ballots (Valid)"})
ridings = columns_reorder_after(ridings, {"Ballots (Valid)": "Ballots (Rejected)"})

# Copy winning candidate up to the ridings level
ridings_winning_parties = candidates[candidates["Elected"] == True].reset_index().set_index(ridings.index.names)
ridings = ridings.join(ridings_winning_parties[["Candidate", "Party", "Party: Color", "Ballots"]])
ridings = ridings.rename(columns={
    "Candidate": "Elected: Name",
    "Ballots": "Elected: Ballots",
    "Party": "Elected: Party",
    "Color": "Elected: Color",
})

# Copy valid ballots per riding back down to the ridings level
candidates = candidates.reset_index().set_index(ridings.index.names).join(
    ridings.reset_index().groupby(ridings.index.names).agg({"Ballots (Valid)": "first"})
).reset_index().set_index(candidates.index.names)
candidates = columns_reorder_after(candidates, {"Riding: Registered": "Ballots (Valid)"})
candidates = candidates.rename(columns={
    "Ballots (Valid)": "Riding: Ballots (Valid)",
})

# Copy valid ballots per riding up to the elections level
elections_ballot_sums = ridings.reset_index().groupby(ridings.index.names).agg({"Ballots (Valid)": "first"}).reset_index().groupby(["Election"])["Ballots (Valid)"].sum().to_frame()
elections = elections.join(elections_ballot_sums)
elections = columns_reorder_after(elections, {"Ballots (Total)": "Ballots (Valid)"})

# Augment parties with election level data
elections["Party: Ballot %"] = elections["Party: Ballots"] / elections["Ballots (Valid)"]
elections["Party: Seat %"] = elections["Party: Seats"] / elections["Seats"]
parties["Seat %"] = parties["Seats"] / elections["Seats"]
parties["Vote %"] = parties["Ballots"] / elections["Ballots (Valid)"]
parties["Seat % - Vote %"] = parties["Seat %"] - parties["Vote %"]
parties["Seat % / Vote %"] = parties["Seat %"] / parties["Vote %"]
candidates["Ballot %"] = candidates["Ballots"] / candidates["Riding: Ballots (Valid)"]

display_toggler("Click to toggle the display the data initialization")

def get_mp_similarity(df, parliament=None):
    top5 = df.reset_index()
    if parliament:
        top5 = top5[top5["Parliament"] == parliament]
    
    per_party_per_vote = top5[top5["Party"].isin((
        "Liberal",
        "Conservative Party of Canada",
        "New Democratic Party",
        "Bloc Québécois",
    ))].groupby(
        ["Parliament", "Session", "Sitting", "Vote", "Parliamentarian: Vote", "Party"],
    )["Parliamentarian: Vote"].count().to_frame().rename(columns={"Parliamentarian: Vote": "Count"})
    per_party_per_vote = per_party_per_vote.unstack(fill_value=0)   
    per_party = top5.reset_index().groupby(
        ["Parliament", "Session", "Sitting", "Vote", "Party"],
    )["Party"].count().to_frame().rename(columns={"Party": "Total"}).unstack().fillna(100000)

    df3 = per_party_per_vote.unstack(fill_value=0).stack().reset_index()
    df1 = per_party.reset_index()
    df2 = df1
    while len(df2) < len(df3):
        df2 = df2.append(df1)
    df2 = df2.sort_values(
        ["Parliament", "Session", "Sitting", "Vote"],
    )
    df2.index = range(len(df2))

    assert len(df3) == len(df2)
    ratio = (df3["Count"] / df2["Total"]).fillna(0)
    ratio.columns = pd.MultiIndex.from_product([['Ratio'], ratio.columns])

    df3[ratio.columns] = ratio
    del df3["Count"]
    vote_ratios = df3

    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=pd.io.pytables.PerformanceWarning)

    df1 = top5.reset_index()
    dfs = []
    for row in list(df1.groupby(["Parliamentarian"]).agg({
        "Parliamentarian": "first",
    }).iterrows()):
        parliamentarian = row[0]
        dfp = df1[df1["Parliamentarian"] == parliamentarian]
        dfp = pd.merge(dfp, vote_ratios, on=["Parliament", "Session", "Sitting", "Vote", "Parliamentarian: Vote"], how='inner')
        del dfp["Parliamentarian: Vote"]
        dfg = dfp.groupby(["Parliament", "Session", "Parliamentarian", "Party"]).agg({
            ("Ratio", "Bloc Québécois"): "mean",
            ("Ratio", "Conservative Party of Canada"): "mean",
            #("Ratio", "Green Party of Canada"): "mean",
            ("Ratio", "Liberal"): "mean",
            ("Ratio", "New Democratic Party"): "mean",
        }).reset_index().set_index(["Parliament", "Session"])
        dfs.append(dfg)

    mp_party_similarity = pd.concat(dfs, ignore_index=True)
    return mp_party_similarity

mp_party_similarity = get_mp_similarity(house_vote_participants, 42)
spoke_labels = mp_party_similarity.columns.levels[1].values[0:4]
theta = radar_factory(len(spoke_labels))

fig = plt.figure(figsize=(20,20))
# adjust spacing around the subplots
fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
title_list = ['Party Similarity']

rows = []
colors = []
for index, row in mp_party_similarity.iterrows():
    name, party, *values = row
    try:
        color = select_by_index(parties, {"Election": [42], "Party": party})["Color"][0]
    except:
        color = "#666666"
    colors.append(color)
    rows.append(values)

data = {'Party Similarity': rows}
radial_grid = [2., 4., 6., 8.]
for n, title in enumerate(title_list):
    ax = fig.add_subplot(2, 2, n + 1, projection='radar')
    plt.rgrids(radial_grid)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
        horizontalalignment='center',
        verticalalignment='center',
    )
    for d, color in zip(data[title], colors):
        ax.plot(theta, d, color=color)
    ax.set_varlabels(spoke_labels)
plt.show()

# TODO: Augment to show per parliament and perhaps broken down by committee source

df = mp_party_similarity.reset_index()
for source in mp_party_similarity.columns.levels[1].values[0:5]:
    if not source:
        continue
    if source == "Green Party of Canada":
        continue
    for target in mp_party_similarity.columns.levels[1].values[0:5]:
        if target:
            display(HTML("""<h3>{} MPs</h3>""".format(source)))
            display(HTML("""<h4>Most like {}</h4>""".format(target)))
            display(df[df["Party"] == source].sort_values(("Ratio", target), ascending=False).head(5))
            display(HTML("""<h4>Least like {}</h4>""".format(target)))
            display(df[df["Party"] == source].sort_values(("Ratio", target)).head(5))

