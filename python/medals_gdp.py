""" Hide code segments from nbviewer output From http://protips.maxmasnick.com/hide-code-when-sharing-ipython-notebooks """
import IPython.core.display as di
# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
# This line will add a button to toggle visibility of code blocks, for use with the HTML export version
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)

""" Faraz Hossein-Babaei   2016/8/24 ~ 2016/10/22
Correlating The economic and Olympic performances of nations """
""" Data Reading Segment: Reads dataframe from Olympics source table and cleans up the table """

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import ipy_table
import IPython.display as dp
from mpl_toolkits.basemap import Basemap, Polygon
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
pd.set_option('display.float_format', lambda x: '%.4f' % x)


medals_df = pd.read_csv("res_data\olympic_summer_medals_1896_2008_athletes.csv")
medals_df.columns = medals_df.iloc[3,:]   # retitle the columns based on a row in file
medals_df.columns.name = ""   # Blanking the index column.
medals_df.drop(medals_df.index[0:4], axis=0, inplace=True)   # remove residual first file rows 0-3
medals_df.reset_index(drop=True, inplace=True)   # reset the index. 4-... --> back to 0-...
medals_df.rename(columns={"Edition":"Year"}, inplace=True)   # column Edition --> .Year

print ("Table 1: The format of the useful subset of source table for this analysis.")
dp.display (medals_df.head(n=4))

new_medal_types = [1 if x == "Gold" else (2 if x == "Silver" else (3 if x == "Bronze" else 0))   for x in medals_df.Medal]
medals_df["Medal"] = new_medal_types
medals_df["Medal"] = medals_df["Medal"].astype(np.int8)
medals_df.drop(['City','Sport','Athlete','Event_gender'], axis=1, inplace=True)   # Drop some cols

""" Team Events Distinction: Determining which sports are team events to attribute single medals """

# Let us get the team sports in "discipline : event" format.
concise = medals_df.copy()
team_events = set()
concise_groupby = concise.groupby(['Year','Discipline','Event','Gender'])
concise_groupby_count = concise_groupby.count()

for idx, row in concise_groupby_count.iterrows():
    if row["Medal"] > 4:
        team_events.add(idx[1] + " : " + idx[2])

# Testing: List all sports determiend as team events
#print (len(team_events))
#for team_event in team_events:
#    print(team_event)

""" Deriving country medal counts: Converting athletes-based to nation-based. For actual country
totals see [2] """

# Make a table for each unique country's medals each year
new_cols = []
for x in medals_df["Year"].unique():
    new_cols += [x + " G", x + " S", x + " B"]
idx = sorted(medals_df["NOC"].unique())
c_medals_df = pd.DataFrame(data=0, index=idx , columns=new_cols )
c_medals_df.fillna(0, inplace=True)

# Dictionary used to convert medal column to integer type herein
medal_types = {1:"G",2:"S",3:"B"}
# Count the medals, now let us. For team events, country wins one medal per team event
row_iter = medals_df.iterrows()
for idx, row in row_iter:
    team_disc, team_event = row["Discipline"], row["Event"]
    if (team_disc+" : "+team_event in team_events):   # Even if supposed team event actually singles, correct result is calculated
        team_medal_counts_dict = {}   # dict to be filled by keys (countries) and values (arrays of the country's team medals)
        while True:   # incrementing rows, count total team medals
            if row["NOC"] not in team_medal_counts_dict:   # If country isn't in the team event medals dictionary
                team_medal_counts_dict[row["NOC"]] = np.zeros(3, dtype=np.int16)
            team_medal_counts_dict[row["NOC"]][row["Medal"]-1] += 1
            if medals_df.loc[idx+1,"Discipline"] != team_disc or medals_df.loc[idx+1,"Event"] != team_event:
                break
            idx, row = next(row_iter)   # NOTE: Here we're INCREMENTING through rows as well as the base loop hereout
        medal_nums = []   # finding min num medals won by a country in a team event whatever type
        for key, value in team_medal_counts_dict.items():
            for num in value:   # The min gives us a base value to divide other medal nums by to decide ow many medals country won
                medal_nums += [num]
        base_num = min([num for num in medal_nums if num != 0])
        for country in team_medal_counts_dict:   # assign num medals to country
            for medal_type in range(1,4):
                medal_contrib = team_medal_counts_dict[country][medal_type-1] // base_num
                c_medals_df.loc[country , row["Year"]+" "+medal_types[medal_type]] += medal_contrib
    else:
        c_medals_df.loc[ row["NOC"] , row["Year"]+" "+medal_types[row["Medal"]] ] += 1   # adding medal contribution each row

#print ("Table showing a few rows of the nations' Olympics medal counts up to 2008.")
#dp.display (c_medals_df.iloc[50:54,:])

""" Appending New Data: Adding 2012, 2016 Olympics data recorded based on TV coverages at the time """

london2012 = pd.read_csv("res_data\London Olympics medals.tsv", sep='\t')
rio2016 = pd.read_csv("res_data\Rio Olympics medals.tsv", sep='\t')
ioc_codes_df = pd.read_csv("res_data\ioc country codes.tsv", sep='\t')
ioc_codes_df["Int Olympic Committee code"] = ioc_codes_df["Int Olympic Committee code"].replace("ROM", "ROU")   # fixing mistake
ioc_codes_df["Country"] = ioc_codes_df["Country"].str.replace("*", "")   # removig some asterisks at ends of country names
ioc_codes_df.set_index("Country", drop=True, inplace=True)
ioc_codes_df.rename(columns={"Int Olympic Committee code":"IOCode"}, inplace=True)   # column of long name --> .IOCode
ioc_codes_df.to_csv("res/ioc country codes corrected.tsv", sep='\t')
# We write corrected.tsv but read corrected_2.tsv after some manual modification of 20+ specific mismatch cases

london2012.set_index(london2012["Country"].map(ioc_codes_df["IOCode"]), inplace=True)   # Derive NOC codes from name and set_index
rio2016.set_index(rio2016["Country"].map(ioc_codes_df["IOCode"]), inplace=True)   # Derive NOC codes from name and set as index
london2012.index.name = ""
rio2016.index.name = ""
# Now, add the countries new in the last 2 Olympics to the medals table
londonNew = london2012["Country"][~london2012.index.isin(c_medals_df.index)]
rioNew = rio2016["Country"][~rio2016.index.isin(c_medals_df.index)]
missing_countries_df = pd.concat([londonNew, rioNew])
miss_ctr = missing_countries_df.groupby(missing_countries_df.index).first()
for noc in miss_ctr.index:
    c_medals_df.loc[noc,:] = 0
c_medals_df.sort_index(inplace=True)   # reset the index. 4-... --> back to 0-...
c_medals_df
# Now add the contributions at the 2012. 2016 Olympics
c_medals_df["2012 G"], c_medals_df["2012 S"], c_medals_df["2012 B"] =             london2012["2012 G"], london2012["2012 S"], london2012["2012 B"]
c_medals_df["2016 G"], c_medals_df["2016 S"], c_medals_df["2016 B"] =             rio2016["2016 G"], rio2016["2016 S"], rio2016["2016 B"]
c_medals_df.fillna(value=0, inplace=True)   # countries not won medals get an N/A, None, null for the 2 Olympics. Fill with 0
c_medals_df = c_medals_df.astype(np.int)

""" Reattributing medals: Caring for anomalies/exceptions (SU is only one with significant
effect, still not on my specific analyses).
ANZ: Australasia (Australia & New Zealand 1908,1912)       given to Australia
BOH: Bohemia (Czech effectively, many Olympics)            given to Czech
BWI: Brit. West Indies 1960 (Jamc., Trin.&Tobago, Brbds.)  given to Jamaica
EUA: United team of Germany                                given to Germany
EUN: United team (almost former soviet: 1992 only)         given to Russia
FRG: West Germany                                          given to Germany
GDR: East Germany                                          given to Germany
IOP: Indpt. Olympic particp. (Yug 1992, cont individuals)  kept independent
RU1: Russian Empire                                        given to Russia
SRB: Serbia & Montenegro (1996-2000)                       given to Serbia
TCH: Czechoslovakia                                        given to Czech
URS: Soviet Union                                          given to Russia
YUG: Yugoslavia                                            given to Serbia
ZZX: Mixed teams [5]                                       omitted
"""

#print ("Test: Germany's total medal count before and after reattribution:")
#print("GER" + ":  " + str(c_medals_df.loc["GER", :].sum()))   # test pre since this cell is to run only once
c_medals_df.loc["AUS", :] += c_medals_df.loc["ANZ", :]
c_medals_df.loc["CZE", :] += c_medals_df.loc[["BOH", "TCH"], :].sum()
c_medals_df.loc["GER", :] += c_medals_df.loc[["EUA", "FRG", "GDR"], :].sum()
c_medals_df.loc["JAM", :] += c_medals_df.loc["BWI", :]
c_medals_df.loc["RUS", :] += c_medals_df.loc[["EUN", "RU1", "URS"], :].sum()
c_medals_df.loc["SCG", :] += c_medals_df.loc[["SRB", "YUG"], :].sum()
#print("GER" + ":  " + str(c_medals_df.loc["GER", :].sum()))   # test epi since this cell is to run only once
#print ()
c_medals_df.drop(["ANZ", "BOH", "TCH", "EUA", "FRG", "GDR", "BWI", "EUN", "RU1", "URS", "SRB", "YUG"], axis=0, inplace=True)
# Note: Among 20 random nations evaluated, all were within 2% of correct value (!) based on [2]

# Display a segment of the medals table
print ("Table 2: Numbers of different medals for different nations at different Summer Olympics,\n" +        " a small portion of the data frame.")
dp.display (c_medals_df.iloc[125:130,:])

""" Add valid IOC countries to the base medals table even if they have never won any medal,
have them represent 0 on choropleths. """

# The following segment subsection was brought from Cell #~20
#ctr_name_discreps = gdp_df[~gdp_df.index.isin(ioc_df.index)].index.values.tolist()
# Above commented code revealed that 44 country name discrepencies were found. Manually, "ioc country
    # codes corrected_2.tsv" was supplied with gdp names to match
ioc_df = pd.read_csv("res/ioc country codes corrected_2.tsv", sep='\t', index_col=4)   # col 0 is countries
ctr_name_discreps = ioc_df[~ioc_df["IOCode"].isin(c_medals_df.index)]
#print ("Names and IOC codes of countries with IOC on file but not in Olympics medals table (", len(ctr_name_discreps), "):")
#print (list(zip(ctr_name_discreps["IOCode"].values.tolist(), ctr_name_discreps.index.values.tolist())))
#print ()

#ctr_name_discreps_rev = c_medals_df[~c_medals_df.index.isin(ioc_df["IOCode"])].index.values.tolist()
#print ("Countries with IOC in our Olympics medals table but absent from ctr_names file (", len(ctr_name_discreps_rev), "):")
#print (ctr_name_discreps_rev)   # No discrepency this time

#print ("\n", ctr_name_discreps)
col_len = len(c_medals_df.columns)
for row in ctr_name_discreps.index:
    this_ioc = ctr_name_discreps.loc[row,"IOCode"]
    if this_ioc[0] != '-':
        # add a new row that didn't exist
        c_medals_df.loc[this_ioc,:] = np.zeros(col_len)

""" Score calculation """

MDL_KS = [4,2,1]   # relative (g,s,b) medal value coefficients in calculating the score
GLOB_YR_RANGE_TPL = (1960,2016)
NUM_SELECTED_COUNTRIES = 10   # Number of high-ranking countries to select for graph

year_idx_grp_list = range(0, len(c_medals_df.columns), 3)   # col idx of gold medl represents its year
olymp_yrs = [int(j[:4]) for j in c_medals_df.columns[year_idx_grp_list]]
#print (*olymp_yrs, sep='\t')


def yr_idx (yr):
    return c_medals_df.columns.tolist().index(str(yr)+" G")

def ctr_year_score (noc, yr, mdl_coeffs):   # e.g. func("AUS", 2016, [4,2,1]) for rel. score value
    return ctr_idx_score(noc, yr_idx(yr), mdl_coeffs)

def ctr_idx_score (noc, idx, mdl_coeffs):
    mdl_nums = [c_medals_df.loc[noc, c_medals_df.columns[idx+idxR]] for idxR in range(3)]
    return np.dot(mdl_nums, mdl_coeffs)

def ctr_cumul_score (yr_range, noc, mdl_coeffs):
    yr_idx1 = year_idx_grp_list.index(yr_idx(yr_range[0]))
    yr_idx2 = year_idx_grp_list.index(yr_idx(yr_range[1]))
    return sum([ctr_idx_score(noc, idx, mdl_coeffs) for idx in year_idx_grp_list[yr_idx1 : yr_idx2+1]])

def globe_year_score (yr, mdl_coeffs):
    return globe_idx_score(yr_idx(yr), mdl_coeffs)

def globe_idx_score (idx, mdl_coeffs):
    return sum([ctr_idx_score(noc, idx, mdl_coeffs) for noc in c_medals_df.index])

def globe_cumul_score (yr_range, mdl_coeffs):
    yr_idx1 = year_idx_grp_list.index(yr_idx(yr_range[0]))
    yr_idx2 = year_idx_grp_list.index(yr_idx(yr_range[1]))
    return sum([globe_idx_score(idx, mdl_coeffs) for idx in year_idx_grp_list[yr_idx1 : yr_idx2+1]])

def score_share (ctr, idx, mdl_coeffs):
    return ctr_idx_score (noc, idx, mdl_coeffs) / globe_idx_score (idx, mdl_coeffs)


global_score = globe_cumul_score(GLOB_YR_RANGE_TPL, MDL_KS)
yr_idxs = [year_idx_grp_list.index(yr_idx(GLOB_YR_RANGE_TPL[i])) for i in (0,1)]

# all_ctrs_scores_df
all_ctrs_df = pd.DataFrame(data=[[ctr_idx_score(noc, i, MDL_KS) for i in year_idx_grp_list] for                                     noc in c_medals_df.index], index=c_medals_df.index)
annual_global_scores = all_ctrs_df.iloc[:].sum()
# all_shares_df
all_shares_df = pd.DataFrame([all_ctrs_df.iloc[:,i] / annual_global_scores[i] for                                     i in range(len(year_idx_grp_list))]).T


all_ctrs_2 = all_ctrs_df.copy()
all_shares_2 = all_shares_df.copy()
all_ctrs_2.columns = olymp_yrs
all_shares_2.columns = olymp_yrs


# determining top 10 countries during some period
ctr_hist_score_ser = pd.Series(all_ctrs_df.loc[:, yr_idxs[0]:yr_idxs[1]].sum(axis=1))
#(ctr_hist_score_ser/ctr_hist_score_ser.sum(axis=0)).to_csv("res/olymp_shrs_1960_2016_total_scores.csv")
select_ctrs = ctr_hist_score_ser.nlargest(NUM_SELECTED_COUNTRIES)

# select_ctrs_df
select_ctrs_df = pd.DataFrame([all_ctrs_df.loc[i,:] for i in select_ctrs.index])
# select_ctrs_share_df
select_shares_df = pd.DataFrame([all_shares_df.loc[i,:] for i in select_ctrs.index])

# select_2016normal_df
absolute_col2016_ser = select_ctrs_df.loc[:, year_idx_grp_list.index(yr_idx(2016))]
select_2016normal_df = select_ctrs_df.div(absolute_col2016_ser, axis=0)   # dividing cols by 2016's
# select_shares_2016normal_df
shares_col2016_ser = select_shares_df.loc[:, year_idx_grp_list.index(yr_idx(2016))]
select_shares_2016normal_df = select_shares_df.div(shares_col2016_ser, axis=0)   # dividing by 2016 col

# Redefine the x-axis data from [0,27] to [1896,2016]
select_ctrs_df.columns = olymp_yrs
select_shares_df.columns = olymp_yrs
select_2016normal_df.columns = olymp_yrs
select_shares_2016normal_df.columns = olymp_yrs

print ()
year_rng_str = str(GLOB_YR_RANGE_TPL[0]) + "-" + str(GLOB_YR_RANGE_TPL[1])
print ("Table 3: Scores of top 10 nations accumulated over Summer Olympics (" + year_rng_str + "):")
dp.display (pd.DataFrame(select_ctrs, columns=["Score"]))

""" Plotting Function """

def prepare_plot (knd, fig_idx, df, ttl, x_lbl, y_lbl, x_tick_lbls, lgd_lbls, clr_map,                   ln_styl, ln_wd, mrk, mrk_sz, atten, txt_bool, x_x, idxs, ylog_bool):
    """ Plots DataFrame. Each row is forms a curve in cartesian coords w/ x-axis values of .columns. """
    
    # grp: list of DataFrames anyway, either of single multicol df or list of many sigle-col dfs
    if type(df) == pd.DataFrame:
        grp = [df,]
        df_bool = True
    else:   # then presumably it's a list
        grp = df
        df_bool = False
    
    font = {'font.family':'Arial', 'font.weight':'normal', 'font.size':22}
    
    color_map = clr_map   # matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    clrs_dict = {}   # other color maps: brg, spectral, Accent, Paired, gnuplot, bone, hsv, jet
    num_plots = len(idxs)
    cnt = 0
    for idx in idxs:   # Building dict of colors for all unique indices of the df(s)
        cur_num = 1.5*cnt / float(num_plots)
        if cur_num > 1: cur_num -= 1
        clrs_dict[idx] = color_map( cur_num )
        cnt += 1
    
    plt.style.use('ggplot')
    plt.figure(fig_idx, figsize=(20,10))   #other options:, dpi=80, facecolor='w', edgecolor='g')
    plt.rcParams.update(font)
    
    x_ticks_vals = []
    for i,itm in enumerate(grp):
        x_ticks_vals += itm.columns.tolist()
        for j,idx in enumerate(itm.index):
            if ylog_bool:
                plt.semilogy( itm.loc[idx,:], color=clrs_dict[idx], linewidth=ln_wd, linestyle=ln_styl, marker=mrk,                           markerfacecolor=clrs_dict[idx], markersize=mrk_sz, alpha=atten)
            else:
                plt.plot( itm.loc[idx,:], color=clrs_dict[idx], linewidth=ln_wd, linestyle=ln_styl, marker=mrk,                           markerfacecolor=clrs_dict[idx], markersize=mrk_sz, alpha=atten)
            if txt_bool:
                for col,elem in enumerate(itm.loc[idx,:]):
                    if not df_bool: col = i   # col is either the index of the mother list or col index depend. df list or DF
                    plt.annotate(idx, xy=(col,elem), xytext=(10,0), textcoords='offset points',                                  fontsize=16, alpha=0.7, color=clrs_dict[idx], fontweight='bold')
    
    
    plt.suptitle(ttl, fontsize=36)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.ylim(bottom=0)
    plt.xlim(grp[0].columns[0]+x_x[0], grp[-1].columns[-1]+x_x[1])
    plt.xticks(x_ticks_vals, x_tick_lbls, rotation='vertical')
    
    if len(lgd_lbls) > 0:
        legend = plt.legend( lgd_lbls, ncol=2, loc='best', fontsize=16, bbox_to_anchor=[0.4, 0.95],                    columnspacing=1.0, labelspacing=0.5, handletextpad=0.0, handlelength=1.5,                    fancybox=True, shadow=True )
    
    return

""" Plot medal trends: Selected countries. """

# select_ctrs_df: Plotting temporal medal trends as countries' medal scores
prepare_plot("line", 1, select_ctrs_df, "Medal scores of top nations" ,              "Olympics Year" , "Olympics medal scores",              olymp_yrs, select_ctrs_df.index , plt.cm.Accent, '-', 4.0, 'o', 4.0, 0.7,              False, (0,0), select_ctrs_df.index, False)
plt.show()
print ("Figure 1: Medal scores of top 10 scoring nations calculated based on 4, 2, and 1 points\n" +        " points allocated for each gold, silver, and bronze, respectivaly.\n")

# select_shares_df: Plotting temporal medal trends as global shares of countries' medal scores
prepare_plot("line", 2, select_shares_df , "Global shares of medal scores of top nations" ,              "Olympics Year" , "Olympics medal score",              olymp_yrs, select_shares_df.index , plt.cm.Accent, '-', 4.0, 'o', 4.0, 0.7,              False, (0,0), select_shares_df.index, False)
plt.show()
print ("Figure 2: The global shares of the top scoring nations found by dividing the Figure 1\n" +        " data each year by each nation's Summer Olympics score accumulated through [1896,2016].\n")

# select_2016normal_df: Plotting temporal medal trends as scores of countries 2016-normalized
prepare_plot("line", 3, select_2016normal_df , "Medal scores of top nations (2016-normalized)" ,              "Olympics Year" , "Olympics medal score ratio of 2016",              olymp_yrs, select_2016normal_df.index , plt.cm.Accent, '-', 4.0, 'o', 4.0, 0.7,              False, (0,0), select_2016normal_df.index, False)
plt.show()
print ("Figure 3: The plotted data of Figure 1 divided by each nation's 2016 score.\n")

# select_shares_2016normal_df: Plotting temporal medal trends as score shares of countries 2016-normalized
prepare_plot("line", 4, select_shares_2016normal_df , "Global shares of medal scores of top nations (2016-normalized)" ,              "Olympics Year" , "Olympics medal score global share ratio of 2016",              olymp_yrs, select_shares_2016normal_df.index , plt.cm.Accent, '-', 4.0, 'o', 4.0, 0.7,              False, (0,0), select_shares_2016normal_df.index, False)
plt.show()
print ("Figure 4: The plotted data of Figure 2 divided by each nation's 2016 score share.\n")

""" Epochs analysis: Averaged performaces through 5 major epochs and box-whisker plot the trends """

# Epoch bins: 1896-1912 (5), 1920-1936 (5), 1948-1964 (5), 1968-1984 (5), 1988-2016 (8)
epochs_list = ["1896-1912", "1920-1936", "1948-1964", "1968-1984", "1988-2016"]
bins = [olymp_yrs.index(i) for i in [1896, 1920, 1948, 1968, 1988]]
# Grouping into bins the data of scores
all_ctrs_df_transposed = all_ctrs_df.T
binned_scores_df_grpby = all_ctrs_df_transposed.groupby(np.digitize(all_ctrs_df_transposed.index, bins))
# Grouping into bins the data of score shares
all_shares_df_transposed = all_shares_df.T
binned_shares_df_grpby = all_shares_df_transposed.groupby(np.digitize(all_shares_df_transposed.index, bins))

# list of averaged score shares of countries during the 5 temporal epochs
indices1 = pd.Series()
epoch_shares_df = []
epoch_shares_top_nations_df = []
for i,h in binned_shares_df_grpby:
    thisSum = h.T.sum(axis=1)
    thisCnt = h.T.count(axis=1)
    thisAvg = thisSum/thisCnt
    top_nations_for_observation = thisAvg.nlargest(NUM_SELECTED_COUNTRIES+15)
    epoch_shares_df.append(thisAvg.to_frame())
    epoch_shares_top_nations_df.append(top_nations_for_observation.to_frame())
    indices1 = indices1.append(top_nations_for_observation.to_frame())

# list of averaged scores of countries during the 5 temporal epochs
indices2 = pd.Series()
epoch_scores_df = []
epoch_scores_top_nations_df = []
for i,h in binned_scores_df_grpby:
    thisSum = h.T.sum(axis=1)
    thisCnt = h.T.count(axis=1)
    thisAvg = thisSum/thisCnt
    top_nations_for_observation = thisAvg.nlargest(NUM_SELECTED_COUNTRIES+15)
    epoch_scores_df.append(thisAvg.to_frame())
    epoch_scores_top_nations_df.append(top_nations_for_observation.to_frame())
    indices2 = indices2.append(thisAvg.to_frame())

for i,_ in enumerate(epoch_scores_df):
    epoch_scores_df[i].columns = [i]
for i,_ in enumerate(epoch_shares_df):
    epoch_shares_df[i].columns = [i]
for i,_ in enumerate(epoch_scores_top_nations_df):
    epoch_scores_top_nations_df[i].columns = [i]
for i,_ in enumerate(epoch_shares_top_nations_df):
    epoch_shares_top_nations_df[i].columns = [i]

# Do any df-wide calculation here; I'm going to make ew dfs excluding outliers for a clean box-whisker plot, modified
# Determined manually as outliers were for each oclumn in num rows: { 0:6 , 1:7 , 2:7 , 3:10 , 4:10 }
epoch_scores_nontop, epoch_shares_nontop, top_scores, top_shares = [], [], [], []
nCols = len(epoch_scores_df)
phenolog_outlie_cutoff = 6
for i in range(nCols):
    if i==1: phenolog_outlie_cutoff = 7
    elif i==3: phenolog_outlie_cutoff = 10
    top_score_list = epoch_scores_top_nations_df[i].iloc[:phenolog_outlie_cutoff,0].index.values.tolist()
    top_share_list = epoch_shares_top_nations_df[i].iloc[:phenolog_outlie_cutoff,0].index.values.tolist()
    epoch_scores_nontop.append(epoch_scores_df[i].loc[~epoch_scores_df[i].index.isin(top_score_list)])
    epoch_shares_nontop.append(epoch_shares_df[i].loc[~epoch_shares_df[i].index.isin(top_share_list)])
    top_scores.append(epoch_scores_df[i].loc[epoch_scores_df[i].index.isin(top_score_list)])
    top_shares.append(epoch_shares_df[i].loc[epoch_shares_df[i].index.isin(top_share_list)])


def zero_to_nan(values):
    """ Replace every 0 with np.nan and return. """
    return [np.nan if x==0 else x for x in values]

epoch_all_scores_dfs_list, epoch_all_shares_dfs_list = [], []
for i,df in enumerate(epoch_scores_nontop):
    epoch_all_scores_dfs_list.append(pd.DataFrame(zero_to_nan(df.iloc[:,0])).dropna())
for i,df in enumerate(epoch_shares_nontop):
    epoch_all_shares_dfs_list.append(pd.DataFrame(zero_to_nan(df.iloc[:,0])).dropna())

""" Plotting Function """

def prepare_outlier_scatter_plot (fig_idx, df_list, ttl, x_lbl, y_lbl, x_tick_lbls, lgd_lbls, clr_map,                   ln_styl, ln_wd, mrk, mrk_sz, atten, x_x, idxs, ylog_bool):
    """ Plot DataFrame. rows form curves in cartesian. .columns forms x-axis values. """

    font = {'font.family':'Arial', 'font.weight':'normal', 'font.size':22}
    plt.rcParams.update(font)
    
    x_ticks_vals = []
    for i,itm in enumerate(df_list):
        x_ticks_vals += itm.columns.tolist()
        for _,idx in enumerate(itm.index):
            if ylog_bool:
                # Note: changed clrs_dict[idx]   to  clr_map
                plt.semilogy( itm.loc[idx,:], color=clr_map, linewidth=ln_wd, linestyle=ln_styl, marker=mrk,                           markerfacecolor=clr_map, markersize=mrk_sz, alpha=atten)
            else:
                plt.plot( itm.loc[idx,:], color=clr_map, linewidth=ln_wd, linestyle=ln_styl, marker=mrk,                           markerfacecolor=clr_map, markersize=mrk_sz, alpha=atten)

            for col,elem in enumerate(itm.loc[idx,:]):
                col = i   # col is either the idx of original list or df
                plt.annotate(idx, xy=(col,elem), xytext=(10,0), textcoords='offset points',                              fontsize=20, alpha=0.3, color=clr_map, fontweight='normal')
    
    plt.suptitle(ttl, fontsize=36)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.ylim(bottom=0)
    plt.xlim(df_list[0].columns[0]+x_x[0], df_list[-1].columns[-1]+x_x[1])
    plt.xticks(x_ticks_vals, x_tick_lbls, rotation='vertical')
    
    if len(lgd_lbls) > 0:
        legend = plt.legend( lgd_lbls, ncol=2, loc='best', fontsize=16, bbox_to_anchor=[0.4, 0.95],                    columnspacing=1.0, labelspacing=0.5, handletextpad=0.0, handlelength=1.5,                    fancybox=True, shadow=True )
    
    return

""" Olympic hosting data gathering and of numbers of nations participating. """

# Data from [4]
olympic_years = [ 1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936,                  1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984,                  1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016 ]
num_nations = [ 14, 24, 12, 22, 28, 29, 44, 46, 37, 49,                59, 69, 72, 84, 94, 112, 121, 92, 81, 140,                160, 169, 197, 199, 201, 204, 206, 206 ]
host_cities = [ "Athens", "Paris", "St. Louis", "London", "Stockholm", "Antwerp", "Paris",                "Amsterdam", "Los Angeles", "Berlin", "London", "Helsinki", "Melbourne", "Rome",                "Tokyo", "Mexico City", "Munich", "Montreal", "Moscow", "Los Angeles", "Seoul",                "Barcelona", "Atlanta", "Sydney", "Athens", "Beijing", "London", "Rio de Janeiro" ]
host_nocs = [ "GRE", "FRA", "USA", "GBR", "SWE", "BEL", "FRA", "NED", "USA", "GER",              "GBR", "FIN", "AUS", "ITA", "JPN", "MEX", "GER", "CAN", "RUS", "USA",              "KOR", "ESP", "USA", "AUS", "GRE", "CHN", "GBR", "BRA" ]

olymp_info = zip(num_nations, host_cities, host_nocs)
cols = ["Num Particip","Host City","Host Country"]
olymp_info_df = pd.DataFrame(list(olymp_info), index=olympic_years, columns=cols)
olymp_info_df.to_csv("res/olymp_host_info.csv")

""" Deriving medal winner among participants fractions through epochs """

#olymp_info = pd.read_csv("res/olymp_host_info.csv", index_col=0)
pg_lim = len(olymp_info_df)//2

yr_host_zip_list = list(zip(olymp_info_df["Host Country"], olymp_yrs))
yr_host_list = list(map(lambda x: str(x[0])+" "+str(x[1]), yr_host_zip_list))
num_particip_df = pd.DataFrame(olymp_info_df["Num Particip"])
num_particip_df = num_particip_df.T.astype(float)

print ("\n")
# num_particip_df: Plotting the growth of the number of participating nations at the Olympics
prepare_plot("line", 1, num_particip_df, "Growing number of participating nations at Olympics" ,              "Host and year of event" , "Number of participating nations",              yr_host_list, [] , plt.cm.summer, '-', 4.0, 'None', 4.0, 0.7,              False, (0,0), num_particip_df.index, False)
plt.show()
print ("Figure 5: Growing number of participating nations during Summer Olympics. [4]")
print ("\n")

# Epoch bins: 1896-1912 (5), 1920-1936 (5), 1948-1964 (5), 1968-1984 (5), 1988-2016 (8)
epochs_list = ["1896-1912", "1920-1936", "1948-1964", "1968-1984", "1988-2016"]
bins = [1896, 1920, 1948, 1968, 1988]
# Grouping into bins the data of scores
olymp_info_grpby = olymp_info_df.groupby(np.digitize(olymp_info_df.index, bins))
olymp_epoch_info = olymp_info_grpby.sum()
#olymp_epoch_info["Non-0s"] = [len(i) for i in epoch_all_scores_dfs_list]
all_scores_all_yrs = all_ctrs_df.copy()
all_scores_all_yrs.columns = olymp_yrs
non0s = all_scores_all_yrs[all_scores_all_yrs != 0].count(axis=0)
non0s_grpby = non0s.groupby(np.digitize(non0s.index, bins))
olymp_epoch_info["Non-zeros"] = non0s_grpby.sum()
olymp_epoch_info["Fraction Zero"] = 1 - olymp_epoch_info["Non-zeros"]/olymp_epoch_info["Num Particip"]

olymp_epoch_info = olymp_epoch_info.T   # Transposing the table for presentation
olymp_epoch_info.columns = epochs_list

""" Table styles Functions """

def printFloat(x):
    if isinstance(x, float):
        if np.modf(x)[0] == 0:
            return str(int(x))
        else:
            return str("{:.2f}".format(x))   # equivalent old way: '%.2f' % (x)
    else: return str(x)
pd.options.display.float_format = printFloat


def format_table (data):
    """ relies on ipy_table modeule """
    if isinstance(data, pd.DataFrame):
        values_str_list = [list(map(printFloat, i)) for i in data.reset_index().values.tolist()]
        data_list = [['Period']+data.columns.tolist()] + values_str_list
    else: data_list = data.copy()
    tbl = ipy_table.make_table(data_list)
    #tbl.apply_theme('basic_both')
    tbl.set_global_style(align='center')
    return tbl

""" Box-whisker plots showing Olympic performance during 5 epochs """

def stylize_boxplots (bps):
    EDG_CLR = "#552222"
    FAC_CLR = "#aa4488"#"#aa4488"#"#882222"
    MED_CLR = "#ee88bb"
    for bp in bps:
        for box in bp['boxes']:
            box.set(color=EDG_CLR, linewidth=2)
            box.set(facecolor = FAC_CLR)
        for whisker in bp['whiskers']:
            whisker.set(color=EDG_CLR, linewidth=2)
        for cap in bp['caps']:
            cap.set(color=EDG_CLR, linewidth=2)
        for median in bp['medians']:
            median.set(color=MED_CLR, linewidth=4)
        for flier in bp['fliers']:
            flier.set(marker='o', color=EDG_CLR, alpha=0.5)
    return

print()
# Produce the table showing fractions of nations producing no medal among countries each period
fr0_df = pd.DataFrame(olymp_epoch_info.T["Fraction Zero"]).T
print ("Table 4: Fractions of no-medal nations per Olympic event per epoch.")
dp.display (format_table(fr0_df))
print ()

CLR = "#663333"   # chnaged from colormap plt.cm.Accent to single color, used inbox-plots

# Note: Outliers are selected manually and excluded from the box-plot data, so the box-plots
    # reflect distributions of ~140 points, 5-10 reasonable outliers of them excluded
scores_outliers = []
for i,df in enumerate(top_scores):
    scores_outliers.append( df.sort_values([i], ascending=False).index )
scores_outliers_df = pd.DataFrame(scores_outliers).T.fillna("")
scores_outliers_df.columns = epochs_list
score_outlier_list = [list(map(printFloat, i)) for i in scores_outliers_df.values.tolist()]
score_outlier_list = [scores_outliers_df.columns.tolist()] + score_outlier_list
score_outlier_table = format_table(score_outlier_list)
score_outlier_table.set_global_style(no_border='all', )
score_outlier_table.set_row_style(0, bold=True)
print ()
print ("Table 5: Outlier scoring nations, based on average medal scores, each period.")
dp.display (score_outlier_table)
print ()

bps = []
# Scatter plot for complement of previous plot: Averages of top countries' medal scores during epochs
plt.figure(5, figsize=(20,10))
prepare_outlier_scatter_plot(5, top_scores , "Averages of countries' medal scores during epochs",          "Epoch" , "Average Olympics medal score", epochs_list, [],          CLR, 'None', 4.0, 'o', 10.0, 0.5, (-0.5,0.5), indices2.index.unique(), True)
# Box-plot for Averages of non-top countries' medal scores during epochs
bps.append(plt.boxplot(epoch_all_scores_dfs_list, notch=False, positions=range(5),                        whis='range', patch_artist=True))
plt.ylim(bottom=0.1)
plt.xticks(range(5), epochs_list, rotation=10)
stylize_boxplots (bps)
plt.show()
print ("Figure 6: Epochs' box-plots showing nations' Summer Olympic medal score average distributions.")
print ("\n\n")


# TODO: If same behaviour below as above, def function, just input scores and shares dataframes
# Note: Outliers are selected manually and excluded from the box-plot data, so the box-plots
    # reflect distributions of ~140 points, 5-10 reasonable outliers of them excluded
shares_outliers = []
for i,df in enumerate(top_shares):
    shares_outliers.append( df.sort_values([i], ascending=False).index )
shares_outliers_df = pd.DataFrame(shares_outliers).T.fillna("")
shares_outliers_df.columns = epochs_list
share_outlier_list = [list(map(printFloat, i)) for i in shares_outliers_df.values.tolist()]
share_outlier_list = [shares_outliers_df.columns.tolist()] + share_outlier_list
share_outlier_table = format_table(share_outlier_list)
share_outlier_table.set_global_style(no_border='all')
share_outlier_table.set_row_style(0, bold=True)
print ("Table 6: Outlier scoring nations, based on average medal scores, each period.")
dp.display (share_outlier_table)
print ()

# Scatter plot for complement of previous plot: Averages of top countries' medal score global shares during epochs
plt.figure(7, figsize=(20,10))
prepare_outlier_scatter_plot(7, top_shares , "Averages of top countries' medal score global shares during epochs",          "Epoch" , "Average Olympics medal score global share", epochs_list, [],          CLR, 'None', 4.0, 'o', 10.0, 0.8, (-0.5,0.5), indices1.index.unique(), True)
# Box-plot for Averages of non-top countries' medal score global shares during epochs
bps.append(plt.boxplot(epoch_all_shares_dfs_list, notch=False, positions=range(5),                        whis='range', patch_artist=True))
plt.ylim(bottom=0.00001)
plt.xticks(range(5), epochs_list, rotation=10)
stylize_boxplots (bps)
plt.show()
print ("Figure 7: Epochs' box-plots of nations' global medal score share average distributions.")
print ()

""" Plotting Function Segment """

def prepare_plot (knd, ax, df, ttl, x_lbl, y_lbl, x_tick_lbls, lgd_lbls, clr_map,                   ln_styl, ln_wd, mrk, mrk_sz, atten, txt_bool, x_x, idxs):
    """ Plot DataFrame. rows form curves in cartesian. .columns forms x-axis values. """
    
    font = {'font.family':'Arial', 'font.weight':'normal', 'font.size':22}
    
    color_map = clr_map   # matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    clrs_dict = {}   # other color maps: brg, spectral, Accent, Paired, gnuplot, bone, hsv, jet
    num_plots = len(idxs)
    cnt = 0
    for idx in idxs:   # Building dict of colors for all unique df indices
        cur_num = 1.0*cnt / float(num_plots)
        if cur_num > 1: cur_num -= 1
        clrs_dict[idx] = clr_map(cur_num)
        cnt += 1
    
    plt.rcParams.update(font)
    
    for _,idx in enumerate(df.index):
        ax.plot( df.loc[idx,:], color=clrs_dict[idx], linewidth=ln_wd, linestyle=ln_styl, marker=mrk,                   markerfacecolor=clrs_dict[idx], markersize=mrk_sz, alpha=atten)
    
    plt.suptitle(ttl, fontsize=36)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.ylim(bottom=0)
    plt.xlim(df.columns[0]+x_x[0], df.columns[-1]+x_x[1])
    plt.xticks(df.columns, x_tick_lbls, rotation='vertical')
    
    if len(lgd_lbls) > 0:
        legend = plt.legend( lgd_lbls, ncol=1, loc='best', fontsize=16,                    columnspacing=1.5, labelspacing=0.5, handletextpad=0.5, handlelength=1.5,                    fancybox=True, shadow=True )
    
    return (ax, clrs_dict)

""" Matching Olympics and GDP data formats """

# Read table used for mapping country names (Olympics or econ format) to IOC codes
#medal_shares_df = pd.read_csv("all_shares_df.csv", index_col=0)

# Loading Olympics data from files prepared from medals_gdp_2.ipynb
#all_shares_df = pd.read_csv("all_shares_df.csv", index_col=0)
# Olympics data column headings to int
all_shares_2.columns = [int(x) for x in all_shares_2.columns.values.tolist()]

# Plotting 1896-2016 Olympics medal score shares data
all_shares_cmplt_top_df = all_shares_2.sort_values([2016,2004], ascending=[False,False])
olymp_yrs = all_shares_2.columns.values.tolist()

host_info_raw = pd.read_csv("res/olymp_host_info.csv", index_col=0).T   # Data from [4]
print ("Table 7: The hosting nations during the Summer Olympics. [4]")
noc_hosts = pd.DataFrame(host_info_raw.iloc[2,:])
dp.display (pd.DataFrame(noc_hosts[:14]).T, pd.DataFrame(noc_hosts[14:]).T)
print ("\n")

# Plotting temporal medal trends as countries' medal score shares of Summer Olympics hosts
hosts = ["USA","GBR","CHN","RUS","GER","FRA","JPN","AUS","ITA",         "KOR","BRA","ESP","CAN","SWE","NED","GRE","BEL","MEX","FIN"]   # list by max peak, descending
plt.style.use('ggplot')
PLT_Y_TOPS = (0.5, 0.25, 0.08, 0.15, 0.09)
for i in range(0,5):
    conations = [hosts[j] for j in range(4*i,min(4*i+4,len(hosts)))]
    sub_shares_df = all_shares_cmplt_top_df.loc[conations, :]
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    plot_ax = prepare_plot("line", ax, sub_shares_df, "Global shares of medal scores",                  "Olympics Year", "Olympics medal scores",                  olymp_yrs, sub_shares_df.index , plt.cm.Accent, '-', 4.0, 'o', 8.0, 0.7,                  False, (0,0), sub_shares_df.index)
    ylim = PLT_Y_TOPS[i]
    df_shape = sub_shares_df.shape
    for row in range(df_shape[0]):
        noc = sub_shares_df.index[row]
        clr = plot_ax[1][noc]   # color used by both annotations
        for j, host in enumerate(noc_hosts.values):
            if host == noc:
                yr = noc_hosts.T.columns[j]
                yv = sub_shares_df.loc[noc,yr]
                if yv>ylim:
                    plot_ax[0].plot(yr, ylim*(1.03), marker='o', markerfacecolor=clr,                                     markersize=50, alpha=0.4, clip_on=False, zorder=10)
                else:
                    plot_ax[0].plot(yr, yv, marker='o', markerfacecolor=clr,                                     markersize=50, alpha=0.6)
        for col in range(df_shape[1]):
            if (sub_shares_df.iloc[row, col] > ylim):
                pk_val = "{:0.3f}".format(sub_shares_df.iloc[row, col])
                col_val = sub_shares_df.columns[col]
                plot_ax[0].annotate (pk_val, xycoords='data', xy=(col_val,ylim),                                   textcoords='offset points', xytext=(-30,4), color=clr)
                # superpose a low attenuation black text on top, just bcs I want above line darker
                plot_ax[0].annotate (pk_val, xycoords='data', xy=(col_val,ylim),                                   textcoords='offset points', xytext=(-30,4), color='black', alpha=0.3)
    plot_ax[0].set_ybound(upper=ylim)   # manually adjust plot y-axis range
    plt.show()

print ("Figure 8 a-e: Summer Olympics hosts' global medal score shares. Cropped-out peaks' values\n" +        " are indicated at top. Circles indicate hosting by nation at that event.")
print ()

# Back to our main point of formatting the Olympic data for correlation with economies
all_shares_2 = all_shares_2.loc[:, 1960:]   # limiting the time to post-1960 to match GDP data

""" GDP involvement starts here """
""" Data reading: Reads and cleans up dataframe from GDP source table """

# Read and clean large database (info about gdp, pop, gender, internet users, surgeries, ...) [3]
econ_df = pd.read_csv("res_data\countries_econ.csv", index_col=2)
econ_df.index.name = None
new_cols = econ_df.columns.values
new_cols[0] = "Series Name"
for i in range(3,len(new_cols)):
    new_cols[i] = int(new_cols[i].split(" ")[0])
econ_df.columns = new_cols

# Now, groupby sub-parts of the complex table to isolate tables of interest
econ_df_groupby = econ_df.groupby(econ_df["Series Name"], axis=0)
gdp_df = econ_df_groupby.get_group("GDP (current US$)")
gdp_df = gdp_df.drop(["Series Name","Series Code","Country Code"], axis=1)
gdp_df = gdp_df.replace(to_replace="..", value=0)
gdp_df = gdp_df.astype(float)
gdp_df = gdp_df.replace(to_replace=0, value=np.nan)
pop_df = econ_df_groupby.get_group("Population, total")
pop_df = pop_df.drop(["Series Name","Series Code","Country Code"], axis=1)
pop_df = pop_df.replace(to_replace="..", value=0)
pop_df = pop_df.astype(float)   # cast as float 1st; direct to int raised error for non-int poplations
pop_df = pop_df.astype(int)     # oddly, columns with NaN therein stay as float64
pop_df = pop_df.replace(to_replace=0, value=np.nan)

""" Reformatting economy tables to match the Olympic tables formats """

# Convert index of country names to IOC (3-letter codes) matching our Olympics tables' formats
gdp_df["Country"] = gdp_df.index   # This col is for later IOC indexing the table
pop_df["Country"] = gdp_df.index   # This col is for later IOC indexing the table
gdp_df["IOC"] = gdp_df["Country"].map(ioc_df["IOCode"])   # Derive NOC codes from name and set_index
pop_df["IOC"] = pop_df["Country"].map(ioc_df["IOCode"])   # Derive NOC codes from name and set_index
gdp_df["IOC"].fillna(gdp_df["Country"], inplace=True)
pop_df["IOC"].fillna(pop_df["Country"], inplace=True)
gdp_df.set_index(gdp_df["IOC"], inplace=True)
pop_df.set_index(pop_df["IOC"], inplace=True)
gdp_df.index.name, pop_df.index.name = None, None
gdp_df.drop(["Country","IOC"], axis=1, inplace=True)
pop_df.drop(["Country","IOC"], axis=1, inplace=True)

""" Adding an estimate of USSR's 1960 GDP to the economy tables indexed as USSR. Choropleths will color the
 corresponding countries covered by USSR at the time based on its value. The USSR will only have a value for
 1960, the year during its existing term we wish to include in the choropleths. Including the USSR is needed
 because of the effect its absence would have on the relative fractions of the metric for other countries. """

USSR_ISOA3 = "SUN"

# USSR GDP estimate based on https://en.wikipedia.org/wiki/Economy_of_the_Soviet_Union as 9e11 2016 US$ in 1960
    # based on the stated 1950 and 1965 absolute values of the GDP in terms of 1990 US$ and the growth rates then.
    # Changed the value considering https://en.wikipedia.org/wiki/List_of_countries_by_largest_historical_GDP 's
        # "Main GDP Countries" section relating it to the US share in the 1960s
# ISO-Alpha 3 code for USSR is exceptionally reserved as "SUN". It is herbey used as such
ussr_yrs = [1960]   # TODO: This does the work for us, but collect more data for USSR for every year [1960,1991]

for df in [gdp_df, pop_df]:
    df.loc[USSR_ISOA3,:] = np.zeros(len(df.columns))
gdp_df.loc[USSR_ISOA3,ussr_yrs] = 3e11
pop_df.loc[USSR_ISOA3,ussr_yrs] = 2.12e8

""" Deriving global share dataframes from economic and population data """

# Define economy shares matrices across globe tables of global shares of gdp & population each year
gdp_share_df = gdp_df / gdp_df.sum(axis=0)
pop_share_df = pop_df / pop_df.sum(axis=0)
# Create new metrics: gdp/pop ratios of different degrees for empirical observation, trial-error
pw = [1, 0.5, 0.2]   # Powers to excite population to before dividing gdp by it
gdppc1_df = gdp_df / (pop_df**pw[0])   # I assume dfs have same size and index
gdppc2_df = gdp_df / (pop_df**pw[1])
gdppc3_df = gdp_df / (pop_df**pw[2])
gdppc1_share_df = gdp_df / (pop_df**pw[0] * gdppc1_df.sum(axis=0))
gdppc2_share_df = gdp_df / (pop_df**pw[1] * gdppc2_df.sum(axis=0))
gdppc3_share_df = gdp_df / (pop_df**pw[2] * gdppc3_df.sum(axis=0))

# Grouping into bins the data of scores
bins = list(range(1960,2020,4)) # years to which data be grouped-averaged; also for new col headings
binned_yrs_idxs = np.digitize(x=gdp_df.columns.values.tolist(), bins=bins, right=True) # all same cols
gdp_df_grpby = gdp_df.groupby(binned_yrs_idxs, axis=1)   # GDPs of countries
pop_df_grpby = pop_df.groupby(binned_yrs_idxs, axis=1)   # Populations of countries
gdp_share_df_grpby = gdp_share_df.groupby(binned_yrs_idxs, axis=1)   # GDPs of countries
pop_share_df_grpby = pop_share_df.groupby(binned_yrs_idxs, axis=1)   # Populations of countries
gdppc1_df_grpby = gdppc1_df.groupby(binned_yrs_idxs, axis=1)   # GDP/population
gdppc2_df_grpby = gdppc2_df.groupby(binned_yrs_idxs, axis=1)   # GDP/(population ^ 0.5)
gdppc3_df_grpby = gdppc3_df.groupby(binned_yrs_idxs, axis=1)   # GDP/(population ^ 1.5)
gdppc1_share_df_grpby = gdppc1_share_df.groupby(binned_yrs_idxs, axis=1)   # GDP/population
gdppc2_share_df_grpby = gdppc2_share_df.groupby(binned_yrs_idxs, axis=1)   # GDP/(population ^ 0.5)
gdppc3_share_df_grpby = gdppc3_share_df.groupby(binned_yrs_idxs, axis=1)   # GDP/(population ^ 1.5)

""" Medal Trends Data preparation Segment: Preparing data for temporal medal trends of countries. """

# Get some output with economically high-ranking nations
g3_4y_share_df = gdppc3_share_df_grpby.mean()
g3_4y_share_df.columns = bins
g3_4y_share_df = g3_4y_share_df.sort_values([2016,2004], ascending=[False,False])
#g3_4y_share_df.to_csv("res/gdpMetric_shrs_olympyrs.csv")
print ()
print ("Table 8: Top nations' global shares of the GDP/population^(0.2) metric, 2016 ranked. [3]")
dp.display(g3_4y_share_df[[1960,1996,2016]].head(n=8))
print ()
# Get some output with Olympics-wise high-ranking nations
all_shares_top_df = all_shares_2.sort_values([2016,2004], ascending=[False,False])
#all_shares_top_df.to_csv("res/olymp_shrs_scores.csv")
print ()
print ("Table 9: Top nations' and global shares of the Olympic medal score metric, 2016 ranked. [3]")
dp.display (all_shares_top_df[[1960,1996,2016]].head(n=8))

""" Choropleth function """

CLR_PWR = 1./3

def draw_map (ser, clr_map):
    """ Function to draw choropleth based on input pandas Series and color map in "robin" world map projection """
    
    plt.rcdefaults()
    
    lMax = ser.max()    # lMin would be zero

    clr_vals = {}
    for ctr in ser.index:
        if ctr=='-': continue
        score = ser.loc[ctr]
        try:
            # C[C["IOCode"]==ctr]["ISO_3"]  finds ISO_3 col of row where this row's IOCode col has ctr.
            # Segment above returns a Series. So I had to to .tolist()[0] to retrieve its only elem
            key = ioc_df[ioc_df["IOCode"]==ctr]["ISO_3"].tolist()[0]
            #if np.isnan(score):
            #    clr_vals[key] = 0   # 0, a recognisably impossible value
            #else:
            clr_vals[key] = 1.0 * score / lMax   # The 1.0 used to be 0.8, but that made the color bar legend quite wrong
        except (KeyError, IndexError) as e: continue   #print (e) or just ignore

    ini_lon = 0. #10. preferred but horizontal streaks appear from polygons of Bering islands cut off by the 10 degree boundary
    m = Basemap (projection='robin', lon_0=ini_lon, resolution='c')
    m.readshapefile('ne_10m_ctr_shape/ne_10m_admin_0_countries', 'NE_10m_Countries')

    m.drawmapboundary (color='black', fill_color='#001530')   # draw white map background and remove boundary line
    #m.drawcoastlines(color='grey')            # not exactly needed but adds definition, e.g. to Amazon river
    m.fillcontinents(color='#222222', lake_color='#001530')
    m.drawcountries(linewidth=1, color='grey')
    graticule_width = 30                       # spacing (in deg?) btw lon- or lat-lines on map
    graticule_color = '#555555'
    parallels = np.arange (-90, 91, graticule_width)
    meridians = np.arange (-180., 181., graticule_width)
    dashes = [3,5]                             # We wish to change the dashline style to 3-on, 5-off in pixels
    m.drawparallels (parallels, dashes=dashes, color=graticule_color, linewidth=0.4)
    m.drawmeridians (meridians, dashes=dashes, color=graticule_color, linewidth=0.4)
    
    ax = plt.gca()
    for info, shape in zip(m.NE_10m_Countries_info, m.NE_10m_Countries):
        try:
            curVal = clr_vals[info['ADM0_A3']]
            if np.isnan(curVal):
                color = '#222222'
            else:
                color = clr_map(curVal**CLR_PWR)
        except (KeyError, IndexError): continue
        ax.add_patch(Polygon(np.array(shape), facecolor=color, edgecolor='black', linewidth=0.4, zorder=2));

    return


def draw_legend (clr_map):
    """ Draw color bar legend with a cubic root color map;
    (help: http://ramiro.org/notebook/basemap-choropleth/) """
    ax_legend = fig.add_axes([0.26, -0.02, 0.48, 0.016], zorder=3)
    grads = np.linspace(0.,1.,400)
    bins = np.linspace(0.,1.,11)
    scheme = [clr_map((i/400.)**(CLR_PWR)) for i in range(400)]
    cmap = mpl.colors.ListedColormap(scheme)
    cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, orientation='horizontal')  # had # boundaries=grads**CLR_PWR,
    #cb.set_xticklabels([str(round(i, 1)) for i in bins], fontsize=10);
    #cb.draw_all()
    return

""" Nan filler function """

# TODO: This is used in function below. Autoformulate it somehow
OFF_VENTURE = [-1,-2,1,-3,-4,2,-5,-6,3,-7,-8,4,-9,-10,5,-11,-12,6,-13,-14,7]
# ISO Alpha 3 codes of USSR's now constituent nations (https://en.wikipedia.org/wiki/ISO_3166-3)
USSR_NOW_COUNTRIES_ISOA3 = ["ARM","AZE","EST","GEO","KAZ","KGZ","LVA","LTU","MDA","RUS","TJK","TKM","UZB","BLR","UKR"]
USSR_BRK_YR = 1991

def fill_df_nan_prox (df, col, offset, fill_USSR=False):
    """ Returns Series after filling nan values in that given by DataFrame's column by
    non-nan values of neighboring columns, venturing off by (-2*,+1*) offset """

    df_cols = df.columns.values.tolist()
    col_idx = df_cols.index(col)
    col_len = len(df_cols)
    ser = df.iloc[:,col_idx].copy()   # Be CAREFUL! Without the .copy() it really is a view ref, not a copy!
    filled_in_cnt = 0
    for row_idx, elem in enumerate(ser): # enumerate(df[col]):
        if np.isnan(elem):
            for i in OFF_VENTURE[0:3*offset]:
                col2_idx = col_idx + i
                if col2_idx < 0 or col2_idx >= col_len: continue
                if np.isnan(df.iloc[row_idx,col2_idx]): continue
                else:
                    ser.iloc[row_idx] = df.iloc[row_idx,col2_idx]
                    filled_in_cnt += 1
                    break
    
    # If < 1992, it will reassign the USSR value to all modern of sub-nations. This reattribution does not
       # affect the values. It is only placed here in the final stage of deriving ser for coloring on map
    if col <= USSR_BRK_YR:
        for ctr in USSR_NOW_COUNTRIES_ISOA3:
            if ctr in ser.index:# and np.isnan(ser.loc[ctr]):
                ser.loc[ctr] = ser.loc["RUS"] if fill_USSR else ser.loc[USSR_ISOA3]
    
    return ser

""" Draw choropleths """

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# display.set_matplotlib_formats
dp.set_matplotlib_formats('retina')     # turn on retina display mode?!
plt.ioff()                              # turn off interactive mode!?

years = [1960, 1992, 2016]
gs = mpl.gridspec.GridSpec(3,1,height_ratios=[1,1,1])

fig = plt.figure(figsize=(8,12))
fig.patch.set_facecolor('#c0c090')  # map background
for i, yr in enumerate(years):
    plt.subplot(gs[i])
    plt.title('Olympic score shares in %s' % yr, fontsize=14)
    # I think for Olympic choropleths we shouldn't fill in data venturing to other years, so 3rd param of zero
    draw_map(fill_df_nan_prox(all_shares_top_df, yr, 0, fill_USSR=True), plt.cm.plasma)   # Drawing Olympic choropleths
draw_legend (plt.cm.plasma)
fig.tight_layout();
plt.show()
print ("Figure 9: Choropleths showing the geographic variation of the Olympic score in the years 1960, \n" +        "1992, and 2016.")

print ("\n")

fig = plt.figure(figsize=(8,12))
fig.patch.set_facecolor('#c0c090')  # map background
for i, yr in enumerate(years):
    plt.subplot(gs[i])
    plt.title('Economic metric shares in %s' % yr, fontsize=14)
    draw_map(fill_df_nan_prox(g3_4y_share_df, yr, 2), plt.cm.plasma)   # Drawing GDP metric choropleths
draw_legend (plt.cm.plasma)
fig.tight_layout();
plt.show()
print ("Figure 10: Choropleths showing the variation of the economic metric averaged over the 4 years \n"+       " ending in 1960, 1992, and 2016 across nations.")

warnings.filterwarnings("default", category=UserWarning)
warnings.filterwarnings("default", category=RuntimeWarning)

""" Simple Plot Function """

def prepare_simple_plot (fig_idx, ax, x_vals, y_vals, ttl, x_lbl, y_lbl, x_tick_lbls, lgd_lbls, clr,                   ln_styl, ln_wd, mrk, mrk_sz, atten, idxs, ax_clr):
    """ Plot DataFrame. rows form curves in cartesian. .columns forms x-axis values. """
    # TODO: Make a subclass of the class used here, for flexibility and not so many args at once

    font = {'font.family':'Arial', 'font.weight':'normal', 'font.size':22}
    plt.rcParams.update(font)
    
    ax.plot(x_vals, y_vals, color=clr, linewidth=ln_wd, linestyle=ln_styl, marker=mrk,             markerfacecolor=clr, markersize=mrk_sz, alpha=atten)
    
    plt.suptitle(ttl, fontsize=36)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl, color=ax_clr)
    ax.yaxis.label.set_color(ax_clr)
    ax.tick_params(axis='y', colors=ax_clr)
    ax.set_ybound(lower=0)
    ax.set_xbound(x_vals[0], x_vals[-1]+4)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_tick_lbls, rotation='vertical')
    
    if len(lgd_lbls) > 0:
        legend = ax.legend( lgd_lbls, ncol=2, loc='best', fontsize=16, bbox_to_anchor=[0.7, 0.95],                    columnspacing=1.0, labelspacing=0.5, handletextpad=0.0, handlelength=1.5,                    fancybox=True, shadow=True )
    
    return

""" Plotting economic-Olympic correlation: Function """

def plot_ctr_gdp_olymp_jux (ctr_grp):
    """ Plotting trends of and fits to economy and Olympics performace shares of nations. """
    
    finite_mask = np.isfinite(g3_4y_share_df.loc[ctr_grp,:].iloc[0,:].tolist())
    xR_expr = g3_4y_share_df.columns.values.tolist()
    xL_expr = [j for i,j in enumerate(xR_expr) if finite_mask[i]]   # applying the bool mask filter
    yL_expr = g3_4y_share_df.loc[ctr_grp,:].iloc[0,:].tolist()
    yL_expr = [j for i,j in enumerate(yL_expr) if finite_mask[i]]
    yR_expr = all_shares_2.loc[ctr_grp,:].iloc[0,:].tolist()
    x_cont = np.linspace(xR_expr[0], xR_expr[-1]+2, 100)   # a[0] is min(a), a[-1] is max(a); sorted!
    xL_cont = np.linspace(xL_expr[0], xL_expr[-1]+2, 100)
    xR1 = [x for x in x_cont if x <= xR_expr[-1]]
    xL1 = [x for x in xL_cont if x <= xL_expr[-1]]
    xL2 = [x for x in xL_cont if x >= xL_expr[-1]-0.5]   # -0.5 to have continuity
    
    # get curve fit functions
    p_fit_L = np.polyfit(xL_expr, yL_expr, deg=5)
    p_fit_R = np.polyfit(xR_expr, yR_expr, deg=5)
    
    COLOR_1 = "#ee7700"
    COLOR_2 = "#77cc22"
    COLOR_1_AX = "#ee7700"
    COLOR_2_AX = "#55bb11"
    COLOR_1_PTS = "#dd8822"
    COLOR_2_PTS = "#99dd00"
    
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    # polynomial fit to GDP data
    prepare_simple_plot(1, ax1, xL1, np.poly1d(p_fit_L)(xL1), "" ,                  "" , "",                  g3_4y_share_df.columns, [] , COLOR_1, '-', 4.0, 'None', 10.0, 0.7,                  g3_4y_share_df.index, COLOR_1_AX)
    # polynomial fit to Olympic data
    prepare_simple_plot(1, ax2, xR1, np.poly1d(p_fit_R)(xR1), "" ,                  "" , "",                  g3_4y_share_df.columns, [] , COLOR_2, '-', 4.0, 'None', 10.0, 0.7,                  g3_4y_share_df.index, COLOR_2_AX)
    # Extrapolating GDP data
    prepare_simple_plot(1, ax1, xL2, np.poly1d(p_fit_L)(xL2), "" ,                  "" , "",                  g3_4y_share_df.columns, [] , COLOR_1, '--', 4.0, 'None', 10.0, 0.7,                  g3_4y_share_df.index, COLOR_1_AX)
    # Actual GDP data
    prepare_simple_plot(1, ax1, xL_expr, yL_expr, "" ,                  "Olympics Year" , "GDP/pop^0.2 global share",                  xL_expr, [], COLOR_1_PTS, '', 4, 'o', 10, 0.9,                  g3_4y_share_df.index, COLOR_1_AX)
    # Actual Olympics data
    prepare_simple_plot(1, ax2, xR_expr, yR_expr, "Economic and Olympic performances of "+ctr_grp[0] ,                  "Olympics Year" , "Olympics medal score share",                  xR_expr, [], COLOR_2_PTS, '', 4, 'd', 10, 0.9,                  all_shares_2.index, COLOR_2_AX)
    
    return [ax1,ax2]

""" Plotting economic-Olympic correlation: Actuation """

warnings.filterwarnings("ignore", category=np.RankWarning)

# Rescaling the GDP/pop^0.2 share y-axis to reflect the range of the Olympics y-axis. Units are
    # arbitrary and show relative global share; similar range allows visual judgement of correlation.
ADJ_YBOUND_DICT = {"USA":0.6,"GER":0.2,"GBR":0.12,"FRA":0.08,"ITA":0.1,"CAN":0.1,"AUS":0.04,"BRA":0.04,                      "RUS":0.04,"ESP":0.07,"IND":0.03,"NED":0.03,"MEX":0.03,"SWE":0.03,"COL":0.01}

def adjust_ybounds_from_dict (plot_ax, noc):
    if (noc in ADJ_YBOUND_DICT):
        plot_ax[0].set_ybound(upper=ADJ_YBOUND_DICT[noc])
    return


print ()
for noc in g3_4y_share_df.index[:16]:  # magic number: 16
    plot_ax = plot_ctr_gdp_olymp_jux([noc])
    adjust_ybounds_from_dict(plot_ax, noc)
    plt.show()
print ("Figure 11 a-p: The economic and Olympic performance trends of top-economy nations revealing\n" +        " strong correlations of the two metrics for most nations.")

print ("\n\n")
for noc in ["JAM","SWE","COL"]:
    plot_ax = plot_ctr_gdp_olymp_jux([noc])
    adjust_ybounds_from_dict(plot_ax, noc)
    plt.show()
print ("Figure 12 a-c: Economy and Summer Olympic performances trends of a few nations.")

warnings.filterwarnings("default", category=np.RankWarning)

