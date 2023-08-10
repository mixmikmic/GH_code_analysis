import pandas as pd
from collections import defaultdict, namedtuple, Counter
from datetime import datetime, timedelta
from pandas.tseries.offsets import *
import sys, json, re, os
import urllib.request
import urllib.parse
import pickle
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import Tracer; debug_here = Tracer()

# Load, initialize data
rv_data = pd.read_csv('/home/michael/school/research/wp/revert_talk_threads_unique_7days.csv', parse_dates=['revert_timestamp'])
talk_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/ipc_talkpages_byarticle.csv', parse_dates=['post_timestamp'])
crit = talk_data['post_text'].map(lambda x: not re.match(r':+$', str(x)))
talk_data = talk_data[crit]
diff_dir = '/home/michael/school/research/wp/wp_articles/ipc_article_diffs/'
thread_durs = []

# FNS FOR SCORING EDITORS IN REVERT DISCUSSIONS
def dict_diff(orig, chg):
    """ Calculates diff between dictionary a and b based on keys from dict a
        Returns: (edit_score, #tokens changed in edit)
    """
    
    if len(orig) == 0: # no change in original except for stopwords
        return (1.0, 0)
    
    chg_abs_sum = 0 # relevant word changes
    orig_abs_sum = 0
    for k in orig:
        orig_abs_sum += abs(orig[k])
        if orig[k] * chg[k] <= 0: # signs differ
            chg_abs_sum += abs(chg[k])
            
    if orig_abs_sum == 0: 
        if chg_abs_sum == 0: # no sign difference, so revert successful
            return 1.0, abs(sum(orig.values()))
        else:
            debug_here()
            
    return max(1-chg_abs_sum/orig_abs_sum, 0.0), orig_abs_sum

# Test dict_diff
print('reverter loses')
a = {'a': -3, 'b': -5}
b = {'a': 3, 'b': 5}
print(dict_diff(a,b))

print("reverter wins")
a = {'a': -3, 'b': -5}
b = {'a': 0, 'b': 0}
print(dict_diff(a,b))

print("reverter loses badly")
a = {'a': -3, 'b': -5}
b = {'a': 3, 'b': 6}
print(dict_diff(a,b))

print('reverter wins well')
a = {'a': -3, 'b': -5}
b = {'a': -5, 'b': -10}
print(dict_diff(a,b))

print('partial')
a = {'a': -3, 'b': -5, 'c': 4}
b = {'a': 1, 'b': 3, 'c': 6}
print(dict_diff(a,b))

print('reverter gets all changes but one badly')
a = {'a': -3, 'b': -5, 'c': 4}
b = {'a': -1, 'b': -1, 'c': -6}
print(dict_diff(a,b))

# SCORE THREAD-WISE WINNERS FOR EACH DISCUSSION AND EDIT PARTICIPANT
art_data = rv_data.copy() # Final spreadsheet with scores
# art_data.columns[4] = 'edit_timestamp'
art_data.rename(columns={'revert_timestamp':'edit_timestamp'}, inplace=True)
art_data['edit_score'] = pd.Series()
art_data['editor_thread_score'] = pd.Series()
art_data['additions'] = pd.Series()
art_data['deletions'] = pd.Series()
art_data['comparison_timestamp'] = pd.Series()

# threads = sorted(list(set(zip(art_data['article'], art_data['thread_title']))))[740:]
threads = sorted(list(set(zip(art_data['article'], art_data['thread_title']))))

stops = stopwords.words('english') 
prev_art = ''

# Get edits by thread editors in between initial revert and session end
# session end: end of thread or last revert 7 days after last thread entry by thread participants
# for i, t in enumerate(threads[:100]):
for i, t in enumerate(threads):
    
    if i % 20 == 0:
        print(i)
    
    # Revert participants: reverter and reverted--not used
#     parts = set(rv_data[(rv_data['article']==t[0]) & (rv_data['thread_title']==t[1])]['editor'].values) |\
#             set(rv_data[(rv_data['article']==t[0]) & (rv_data['thread_title']==t[1])]['reverted_editor'].values)
        
    # Talk page participants--all of them with the corresponding thread in revert data
    talk_parts = set(talk_data[(talk_data['article_title']==t[0]) & (talk_data['thread_title']==t[1])]['username'].values)
    
    talk_rows = talk_data[(talk_data['article_title'] == t[0]) 
                            & (talk_data['thread_title'] == t[1])
                            ].loc[:, ['article_title', 'thread_title', 'post_timestamp']]
    thread_end = max(talk_rows['post_timestamp'])
    thread_beg = min(talk_rows['post_timestamp'])
    
    # Build edit history from revert to end of thread
    rv_rows = rv_data[(rv_data['article']==t[0]) & (rv_data['thread_title']==t[1]) &
                     (rv_data['editor'].isin(talk_parts))]
    initial_rv = min(rv_rows['revert_timestamp'])
    if prev_art != t[0]:
        diff_data = pd.read_csv(os.path.join(diff_dir, t[0].lower().replace(' ', '_').replace('/', '_') + '_diff.csv'),
                   parse_dates=['timestamp'])
    last_rv = max(rv_rows['revert_timestamp'])
    sess_beg = min(thread_beg, initial_rv)
    sess_end = max(thread_end, last_rv)
#   + DateOffset(days=1)

    # Find edits that are in same timeframe as thread and which thread participants make
    sess_edits = diff_data.loc[(diff_data['timestamp'] >= sess_beg) & (diff_data['timestamp'] < sess_end + DateOffset(days=1))
                         & diff_data['editor'].isin(talk_parts)] # could be intervening edits by non-talk participants
    sess_parts = set(sess_edits['editor'].tolist())
    
    # Recalculate session end and beginning in case talk participants didn't make edits
    if sess_edits.empty:
#         print('No diffs')
        # Remove rows from art_data
        art_data = art_data[(art_data['article'] != t[0]) & (art_data['thread_title'] != t[1])] 
        continue
    sess_beg = min(sess_edits['timestamp'])
    sess_end = max(sess_edits['timestamp'])
    
#     if t[0] == 'Timeline of the Israeli–Palestinian conflict' and t[1] == 'Immigration issue':
#         debug_here()
        
    edscores = defaultdict(lambda: [0,0])
    
    # Calculate success score for each edit compared with end revision
    for row in sess_edits.itertuples():
        edit_text = diff_data.loc[diff_data['timestamp']==row.timestamp]
        diffs = diff_data.loc[(diff_data['timestamp'] > row.timestamp) & (diff_data['timestamp'] <= (sess_end + DateOffset(days=1)))]
        
        # Unigram counter for edit
        if (len(edit_text['deletions'].values) > 0) and (isinstance(edit_text['deletions'].values[0], str)):
            positive_dels = Counter([w.lower() for w in edit_text['deletions'].values[0].split() if w.lower() not in stops])
            edit_diff = Counter({key:-1*positive_dels[key] for key in positive_dels})
        else:
            edit_diff = Counter()
        if (len(edit_text['additions'].values) > 0) and (isinstance(edit_text['additions'].values[0], str)):
            adds = Counter([w.lower() for w in edit_text['additions'].values[0].split() if w.lower() not in stops])
        else:
            adds = Counter()
        edit_diff.update(adds)
        edit_diff = {k: edit_diff[k] for k in edit_diff if edit_diff[k] != 0}
        
#         if row.editor == 'Marsden':
#             debug_here()
        if diffs.empty: # No revisions after thread end
            edit_score = 1.0
            n_wds = abs(sum(edit_diff.values()))
            edscores[row.editor][0] += edit_score * n_wds
            edscores[row.editor][1] += n_wds
                
        else: 
            
            # Unigram counter for revision diffs in window
            next_dels = ' '.join(d.lower() for d in diffs['deletions'].values.tolist() if isinstance(d, str))
            changes = Counter([w for w in next_dels.split() if w not in stops])
            changes = Counter({key:-1*changes[key] for key in changes})
            next_adds = ' '.join(a.lower() for a in diffs['additions'].values.tolist() if isinstance(a, str))
            next_addwds = Counter([w for w in next_adds.split() if w not in stops])

            changes.update(next_addwds)
            
            edit_score, n_wds = dict_diff(edit_diff, changes)
            edscores[row.editor][0] += edit_score * n_wds
            edscores[row.editor][1] += n_wds

        # Add score to dataframe (or row if the edit wasn't a revert)
        match = art_data[(art_data['article']==row.article_name) & 
                         (art_data['thread_title']==t[1]) &
                        (art_data['editor']==row.editor) &
                        (art_data['edit_timestamp']==row.timestamp)]

        if not match.empty:
            art_data.set_value(match.index[0], 'edit_score', edit_score)
            art_data.loc[match.index[0], 'additions'] = row.additions
            art_data.loc[match.index[0], 'deletions'] = row.deletions
            if not diffs.empty: 
                art_data.loc[match.index[0], 'comparison_timestamp'] = max(diffs['timestamp'])
#             art_data.set_value(match.index[0], 'winner', winner)
        else:
            new_row = pd.DataFrame([row])
            new_row['edit_score'] = edit_score
#             new_row['winner'] = winner
            new_row['thread_title'] = t[1]
            if not diffs.empty:
                new_row['comparison_timestamp'] = max(diffs['timestamp'])
            new_row.rename(columns={'timestamp': 'edit_timestamp', 'article_name': 'article'}, inplace=True)
            new_row.drop('Index', axis=1, inplace=True)
            art_data = art_data.append(new_row, ignore_index=True)
            
    art_data.reset_index(drop=True)
    prev_art = t[0]
    
    # Calculate editor thread scores
    for ed in sess_parts:
        sess_finalrows = art_data[(art_data['article']==t[0]) & 
                        (art_data['thread_title']==t[1]) &
                        (art_data['editor']==ed)]
        if edscores[ed][1] == 0: # editor whose only edit was the final one and was of no words
            ed_threadscore = 1.0
        else:
            ed_threadscore = edscores[ed][0]/edscores[ed][1]
        if ed_threadscore > 1.0 or ed_threadscore < 0.0:
            debug_here()
        for idx in sess_finalrows.index:
            art_data.loc[idx, 'editor_thread_score'] = ed_threadscore
#         ed_thread_score = np.mean(sess_finalrows['edit_score'])
#             art_data.loc[idx, 'editor_thread_score'] = ed_thread_score
    
# Sort art_data
art_data.sort_values(['article', 'thread_title', 'edit_timestamp'], inplace=True)

# Select columns
cols = ['article', 'thread_title', 'edit_timestamp', 'editor', 'reverted_editor', 'edit_comment',
       'edit_score', 'editor_thread_score', 'comparison_timestamp', 'additions', 'deletions']
art_data = art_data[cols]

# Remove reverts that don't have diffs (just removed Wikipedia metadata, for instance)
mask = [isinstance(tup[0], str) or isinstance(tup[1], str) for tup in zip(art_data['additions'], art_data['deletions'])]
art_data = art_data[mask]
    
art_data.to_csv('/home/michael/school/research/wp/wikipedia/data/editor_thread_scores.csv', index=False)
len(art_data)

# Load thread-level editor scores

edthread = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/editor_thread_scores.csv')
print(len(edthread))
edthread

# Stats on editor thread scores
print('Number of classified edits')
print(len(edthread))
print('Number of talk threads (sessions)')
n_threads = len(set(zip(edthread['article'], edthread['thread_title'])))
print(n_threads)
print('Number of editors per thread')
print(len(set(zip(edthread['article'], edthread['thread_title'], edthread['editor'])))/n_threads)
print('Number of successful edits (above 50%)')
successful_edits = len(edthread[edthread['edit_score']>.5])
print(successful_edits, end='\t')
print(successful_edits/len(edthread))
print('Number of successful editors in threads (above 50%)')
ed_thread_scores = set(zip(edthread['article'], edthread['thread_title'], edthread['editor'], edthread['edit_score']))
s = len([el for el in ed_thread_scores if el[3]>.5])
print('{0}/{1}'.format(s,len(ed_thread_scores)), end='\t')
print(s/len(ed_thread_scores))

# Get scores for editors
stops = stopwords.words('english')
prev_art = ''
    
# row = thread_data.loc[0, :]
# for row in thread_data.loc[1736:1737,:].itertuples():
for row in thread_data.itertuples():
            
    # Calculate session endpt (last time people talk on thread or revert ea other)
    thread_rows = talk_data[(talk_data['article_title'] == row.article) 
                            & (talk_data['thread_title'] == row.thread_title)
                            & (talk_data['username'].isin([row.editor, row.reverted_editor]))].loc[:, ['article_title', 'thread_title', 'post_timestamp']]
    thread_beg = thread_rows['post_timestamp'].loc[thread_rows['post_timestamp'].idxmin()]
    thread_end = thread_rows['post_timestamp'].loc[thread_rows['post_timestamp'].idxmax()]
#     thread_durs.append(thread_end-thread_beg)
    
    last_rv = max(thread_data[(thread_data['article']==row.article) & 
                          (thread_data['thread_title']==row.thread_title) & 
                          (((thread_data['editor']==row.editor) & (thread_data['reverted_editor']==row.reverted_editor)) |
                          ((thread_data['editor']==row.reverted_editor) & (thread_data['reverted_editor']==row.editor)))
                         ]['revert_timestamp'])
    sess_end = max(thread_end, last_rv)

    if prev_art != row.article:
        diff_data = pd.read_csv(os.path.join(diff_dir, row.article.lower().replace(' ', '_').replace('/', '_') + '_diff.csv'),
                   parse_dates=['timestamp'])

    if not row.revert_timestamp in diff_data['timestamp'].tolist(): # revert not present in diff file if no detected change in wiki markup
        continue
            
    rv_text = diff_data.loc[diff_data['timestamp']==row.revert_timestamp]
    sess_rows = diff_data.loc[(diff_data['timestamp'] > row.revert_timestamp) & (diff_data['timestamp'] < sess_end + DateOffset(days=1))]
#     endrevs = diff_data.loc[diff_data['timestamp'] > thread_end + DateOffset(days=1)]
    if sess_rows.empty: # No revisions after thread end
         rv_score = 1.0
    else: 
        # Unigram counter for revert
        if (len(rv_text['deletions'].values) > 0) and (isinstance(rv_text['deletions'].values[0], str)):
            positive_dels = Counter([w.lower() for w in rv_text['deletions'].values[0].split() if w.lower() not in stops])
            dels = Counter({key:-1*positive_dels[key] for key in positive_dels})
        else:
            dels = Counter()
        if (len(rv_text['additions'].values) > 0) and (isinstance(rv_text['additions'].values[0], str)):
            adds = Counter([w.lower() for w in rv_text['additions'].values[0].split() if w.lower() not in stops])
        else:
            adds = Counter()
        dels.update(adds)
        dels = {k: dels[k] for k in dels if dels[k] != 0}

        # Unigram counter for revision diffs in window
        #         wted_changes = {}

        # for r in sess_rows.itertuples():
        next_dels = ' '.join(d.lower() for d in sess_rows['deletions'].values.tolist() if isinstance(d, str))
        next_delwds = Counter([w for w in next_dels.split() if w not in stops])
        next_delwds = Counter({key:-1*next_delwds[key] for key in next_delwds})
        next_adds = ' '.join(a.lower() for a in sess_rows['additions'].values.tolist() if isinstance(a, str))
        next_addwds = Counter([w for w in next_adds.split() if w not in stops])

        next_delwds.update(next_addwds)

        prev_art = row.article

        #     changes = {k: next_delwds.get(k, 0) for k in dels}
        rv_score = dict_diff(dels, next_delwds)
    #     print(rv_score)

    thread_data.set_value(row.Index, 'revert_score', rv_score)

    winner = np.nan
    if not np.isnan(rv_score):
        if rv_score > 0.5:
            winner = row.editor
        else:
            winner = row.reverted_editor
#     print(winner)

    thread_data.loc[row.Index, 'winner'] = winner

    if row.Index % 100 == 0:
        print(row.Index, rv_score)
        sys.stdout.flush()

thread_data_nonan = thread_data[np.isfinite(thread_data['revert_score'])]
# print(len(thread_data_nonan))
# len(thread_data_nonan)

thread_data_nonan.to_csv('/home/michael/school/research/wp/wikipedia/data/revert_threads_scores.csv', index=False)

# Load labeled revert discussions
labeled_revert_talk = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/revert_threads_scores.csv')

# Count rows with reverter-revert winner ambiguity
print("Number of rows that had exact opposite revert winners", end='\t')
thread_winners_reverters = defaultdict(list)
opps = []
for row in labeled_revert_talk.itertuples():
    thread_winners_reverters[(row.article, row.thread_title)].append((row.winner, row.editor, row.reverted_editor, row.Index))
for t in thread_winners_reverters:
    # winners for that thread
    winners = []
    for el in thread_winners_reverters[t]:
        if el[0] == el[1]:
            winners.append((el[0], el[2]))
        else:
            winners.append((el[0], el[1]))
                           
    # get opposites
    for el in thread_winners_reverters[t]:
        if el[0] == el[1]: # winner is reverter
            if (el[2], el[0]) in winners:
                opps.append(el[-1]) # save row index
        else: # winner is reverted
            if (el[1], el[0]) in winners:
                opps.append(el[-1]) # save row index
print(len(opps), end='\t')
print(len(opps)/len(labeled_revert_talk))

# Look into inconsistent rows
labeled_revert_talk.loc[opps[:10], :]

r = thread_data[(thread_data['article']=='Population transfer') & (thread_data['thread_title']=='Greek-Turkish Population Transfer')]
r

# LABEL PAIRWISE WINNERS OF REVERT DISCUSSION THREAD--JUST DIFF AT END OF DISCUSSION 

# score_window = DateOffset(days=o)
# thread_data['revert_score_{0}days'.format(o)] = pd.Series()
# thread_data['winner_{0}days'.format(o)] = pd.Series()
thread_data['revert_score'] = pd.Series()
thread_data['winner'] = pd.Series()

# Get scores for editors
stops = stopwords.words('english')
prev_art = ''
    
# row = thread_data.loc[0, :]
# for row in thread_data.loc[1736:1737,:].itertuples():
for row in thread_data.itertuples():
            
    # Calculate session endpt (last time people talk on thread or revert ea other)
    thread_rows = talk_data[(talk_data['article_title'] == row.article) 
                            & (talk_data['thread_title'] == row.thread_title)
                            & (talk_data['username'].isin([row.editor, row.reverted_editor]))].loc[:, ['article_title', 'thread_title', 'post_timestamp']]
    thread_beg = thread_rows['post_timestamp'].loc[thread_rows['post_timestamp'].idxmin()]
    thread_end = thread_rows['post_timestamp'].loc[thread_rows['post_timestamp'].idxmax()]
#     thread_durs.append(thread_end-thread_beg)
    
    last_rv = max(thread_data[(thread_data['article']==row.article) & 
                          (thread_data['thread_title']==row.thread_title) & 
                          (((thread_data['editor']==row.editor) & (thread_data['reverted_editor']==row.reverted_editor)) |
                          ((thread_data['editor']==row.reverted_editor) & (thread_data['reverted_editor']==row.editor)))
                         ]['revert_timestamp'])
    sess_end = max(thread_end, last_rv)

    if prev_art != row.article:
        diff_data = pd.read_csv(os.path.join(diff_dir, row.article.lower().replace(' ', '_').replace('/', '_') + '_diff.csv'),
                   parse_dates=['timestamp'])

    if not row.revert_timestamp in diff_data['timestamp'].tolist(): # revert not present in diff file if no detected change in wiki markup
        continue
            
    rv_text = diff_data.loc[diff_data['timestamp']==row.revert_timestamp]
    sess_rows = diff_data.loc[(diff_data['timestamp'] > row.revert_timestamp) & (diff_data['timestamp'] < sess_end + DateOffset(days=1))]
#     endrevs = diff_data.loc[diff_data['timestamp'] > thread_end + DateOffset(days=1)]
    if sess_rows.empty: # No revisions after thread end
         rv_score = 1.0
    else: 
        # Unigram counter for revert
        if (len(rv_text['deletions'].values) > 0) and (isinstance(rv_text['deletions'].values[0], str)):
            positive_dels = Counter([w.lower() for w in rv_text['deletions'].values[0].split() if w.lower() not in stops])
            dels = Counter({key:-1*positive_dels[key] for key in positive_dels})
        else:
            dels = Counter()
        if (len(rv_text['additions'].values) > 0) and (isinstance(rv_text['additions'].values[0], str)):
            adds = Counter([w.lower() for w in rv_text['additions'].values[0].split() if w.lower() not in stops])
        else:
            adds = Counter()
        dels.update(adds)
        dels = {k: dels[k] for k in dels if dels[k] != 0}

        # Unigram counter for revision diffs in window
        #         wted_changes = {}

        # for r in sess_rows.itertuples():
        next_dels = ' '.join(d.lower() for d in sess_rows['deletions'].values.tolist() if isinstance(d, str))
        next_delwds = Counter([w for w in next_dels.split() if w not in stops])
        next_delwds = Counter({key:-1*next_delwds[key] for key in next_delwds})
        next_adds = ' '.join(a.lower() for a in sess_rows['additions'].values.tolist() if isinstance(a, str))
        next_addwds = Counter([w for w in next_adds.split() if w not in stops])

        next_delwds.update(next_addwds)

        prev_art = row.article

        #     changes = {k: next_delwds.get(k, 0) for k in dels}
        rv_score = dict_diff(dels, next_delwds)
    #     print(rv_score)

    thread_data.set_value(row.Index, 'revert_score', rv_score)

    winner = np.nan
    if not np.isnan(rv_score):
        if rv_score > 0.5:
            winner = row.editor
        else:
            winner = row.reverted_editor
#     print(winner)

    thread_data.loc[row.Index, 'winner'] = winner

    if row.Index % 100 == 0:
        print(row.Index, rv_score)
        sys.stdout.flush()

thread_data_nonan = thread_data[np.isfinite(thread_data['revert_score'])]
# print(len(thread_data_nonan))
# len(thread_data_nonan)

thread_data_nonan.to_csv('/home/michael/school/research/wp/wikipedia/data/revert_threads_scores.csv', index=False)

# Stats
print("Number of thread-revert pairs", end='\t\t\t\t\t')
print(len(thread_data_nonan)) # Number of thread-revert pairs
print("Number of reverter-winning thread-revert pairs", end='\t\t\t')
print(len(thread_data_nonan[thread_data_nonan['revert_score'] > 0]), end = "\t")
print(len(thread_data_nonan[thread_data_nonan['revert_score'] > 0])/len(thread_data_nonan))
print("Number of unique threads", end='\t\t\t\t\t')
n_threads = len(set(zip(thread_data_nonan['article'], thread_data_nonan['thread_title'])))
print(n_threads)
print("Number of sessions (same thread, same 2 editors)", end='\t\t')
n_sess = len(set(zip(thread_data_nonan['article'], thread_data_nonan['thread_title'], 
                     thread_data_nonan['editor'], thread_data_nonan['reverted_editor'])))
print(n_sess)

print("Number of threads that had multiple revert winners", end='\t\t')
thread_winners = defaultdict(set)
mismatches = 0
for row in thread_data_nonan.itertuples():
    thread_winners[(row.article, row.thread_title)].add(row.winner)
for t in thread_winners:
    if len(thread_winners[t]) > 1:
        mismatches +=1 
print(mismatches, end='\t')
print(mismatches/n_threads)

# Count rows with reverter-revert winner ambiguity
print("Number of rows that had exact opposite revert winners", end='\t\t')
thread_winners_reverters = defaultdict(list)
opps = []
oppthreads = set()
oppsess = set()
for row in thread_data_nonan.itertuples():
    thread_winners_reverters[(row.article, row.thread_title)].append((row.winner, row.editor, row.reverted_editor, row.Index))
for t in thread_winners_reverters:
    # winners for that thread
    winners = []
    for el in thread_winners_reverters[t]:
        if el[0] == el[1]:
            winners.append((el[0], el[2]))
        else:
            winners.append((el[0], el[1]))
                           
    # get opposites
    for el in thread_winners_reverters[t]:
        if el[0] == el[1]: # winner is reverter
            if (el[2], el[0]) in winners:
                opps.append(el[-1]) # save row index
                oppthreads.add(t)
                oppsess.add((t, row.editor, row.reverted_editor))
        else: # winner is reverted
            if (el[1], el[0]) in winners:
                opps.append(el[-1]) # save row index
                oppthreads.add(t)
                oppsess.add((t, row.editor, row.reverted_editor))
print(len(opps), end='\t')
print(len(opps)/len(thread_data_nonan))

print("Number of threads that had exact opposite revert winners", end='\t')
print(len(oppthreads), end='\t')
print(len(oppthreads)/n_threads)
print("Number of sessions that had exact opposite revert winners", end='\t')
print(len(oppsess), end='\t')
print(len(oppsess)/n_sess)

get_ipython().magic('matplotlib inline')

# Plot thread lengths
thread_dur_h = [t/pd.Timedelta('1 hour') for t in thread_durs]
plt.hist(thread_dur_h, bins = 500)
plt.show()

np.mean(thread_dur_h)

np.median(thread_dur_h)

# LABEL WINNERS OF REVERT DISCUSSION THREAD--WITH TEMPORAL DATA

offsets = [7, 14, 30]

thread_data = pd.read_csv('/home/michael/school/research/wp/revert_talk_threads_unique_7days.csv', parse_dates=['revert_timestamp'])

# for o in offsets[:1]:
for o in offsets:
    print("Calculating {0}-day winners".format(o))
    score_window = DateOffset(days=o)
    thread_data['revert_score_{0}days'.format(o)] = pd.Series()
    thread_data['winner_{0}days'.format(o)] = pd.Series()

    # Get scores for editors
    diff_dir = '/home/michael/school/research/wp/wp_articles/ipc_article_diffs/'
    stops = stopwords.words('english')
    prev_art = ''
    
# row = thread_data.loc[0, :]
#     for row in thread_data.loc[:100,:].itertuples():
    for row in thread_data.itertuples():
            
        if prev_art != row.article:
            diff_data = pd.read_csv(os.path.join(diff_dir, row.article.lower().replace(' ', '_').replace('/', '_') + '_diff.csv'),
                   parse_dates=['timestamp'])

        if not row.revert_timestamp in diff_data['timestamp'].tolist(): # revert not present in diff file if no detected change in wiki markup
            continue
            
        window_rows = diff_data.loc[(diff_data['timestamp'] >= row.revert_timestamp) & (diff_data['timestamp'] <= row.revert_timestamp+score_window)]

        # Calculate duration for window_rows
        next_text = window_rows[window_rows['timestamp']!=row.revert_timestamp].loc[:,['timestamp', 'additions', 'deletions']]
        window_rows['rev_duration'] = pd.Series()
        for r in next_text.itertuples():
            window_rows.loc[r.Index-1, 'rev_duration'] = r.timestamp - window_rows.loc[r.Index-1, 'timestamp']

        rv_text = window_rows[window_rows['timestamp']==row.revert_timestamp].loc[:,['additions', 'deletions', 'rev_duration']]

        # Unigram counter for revert
        if (len(rv_text['deletions'].values) > 0) and (isinstance(rv_text['deletions'].values[0], str)):
            positive_dels = Counter([w.lower() for w in rv_text['deletions'].values[0].split() if w.lower() not in stops])
            dels = Counter({key:-1*positive_dels[key] for key in positive_dels})
        else:
            dels = Counter()
        if (len(rv_text['additions'].values) > 0) and (isinstance(rv_text['additions'].values[0], str)):
            adds = Counter([w.lower() for w in rv_text['additions'].values[0].split() if w.lower() not in stops])
        else:
            adds = Counter()
        dels.update(adds)
        
        # Weight the revert unigram counter, too
        if isinstance(rv_text['rev_duration'].values[0], pd.Timedelta): # no next edit in that window, so nan
            h = rv_text['rev_duration'].values[0]/pd.Timedelta('1 hour')
            wted_rv = {k: dels[k] * h for k in dels}
        else:
            wted_rv = dels

        # Unigram counter for revision diffs in window
        next_text = window_rows[window_rows['timestamp']!=row.revert_timestamp].loc[:,['timestamp', 'additions', 'deletions', 'rev_duration']]

        # Calculate next rows unigrams
        wted_changes = {}

        for r in next_text.iloc[:-1,:].itertuples():

        #         next_dels = ' '.join(d.lower() for d in next_text['deletions'].values.tolist() if isinstance(d, str))
            if isinstance(r.deletions, str):
                positive_dels = Counter([w for w in r.deletions.split() if w not in stops])
                next_delwds = Counter({key:-1*positive_dels[key] for key in positive_dels})
            else:
                next_delwds = Counter()
        #     next_adds = ' '.join(a.lower() for a in next_text['additions'].values.tolist() if isinstance(a, str))

            if isinstance(r.additions, str):
                next_addwds = Counter([w for w in r.additions.split() if w not in stops])
            else:
                next_addwds = Counter()

            next_delwds.update(next_addwds)

            # Apply duration weight    
            h = r.rev_duration/pd.Timedelta('1 hour')
            wted = {k: next_delwds[k] * h for k in next_delwds}

            wted_changes.update(wted)

        #         prev_art = row.article
        #         if len(dels) == 0: # sometimes happens if just stopword changes, for instance
        #             continue

        relevant_changes = {k: wted_changes.get(k, 0) for k in dels}
        rv_score = dict_diff(wted_rv, relevant_changes)
        
        thread_data.set_value(row.Index, 'revert_score_{0}days'.format(o), rv_score)

        winner = np.nan
        if not np.isnan(rv_score):
            if rv_score > 0:
                winner = row.editor
            else:
                winner = row.reverted_editor
        thread_data.loc[row.Index, 'winner_{0}days'.format(o)] = winner

        if row.Index % 20 == 0:
            print(row.Index, rv_score)
            sys.stdout.flush()

thread_data_nonan = thread_data[np.isfinite(thread_data['revert_score_{0}days'.format(offsets[0])])]
print(len(thread_data_nonan))
len(thread_data_nonan)

thread_data_nonan.to_csv('/home/michael/school/research/wp/wikipedia/data/revert_threads_scores.csv', index=False)

# Stats
print("Number of thread-revert pairs", end='\t\t\t\t')
print(len(thread_data_nonan)) # Number of thread-revert pairs
print("Number of reverter-winning thread-revert pairs", end='\t\t')
print(len(thread_data_nonan[thread_data_nonan['revert_score_7days'] > 0]), end = " ")
print(len(thread_data_nonan[thread_data_nonan['revert_score_7days'] > 0])/len(thread_data_nonan))
print("Number of unique threads", end='\t\t\t\t')
print(len(set(zip(thread_data_nonan['article'], thread_data_nonan['thread_title']))))
print("Number of rows where 14day winner different from 7day", end='\t')
print(len(thread_data_nonan[thread_data_nonan['winner_7days'] != thread_data_nonan['winner_14days']]))
print("Number of rows where 30day winner different from 7day", end='\t')
print(len(thread_data_nonan[thread_data_nonan['winner_7days'] != thread_data_nonan['winner_30days']]))

print("Number of threads that had different revert winners", end='\t')
thread_winners = defaultdict(set)
mismatches = 0
for row in thread_data_nonan.itertuples():
#     if (row.article, row.thread_title) not in thread_winners:
#         thread_winners[(row.article, row.thread_title)] = row.winner_7days
#     else:
#         if thread_winners[(row.article, row.thread_title)] != row.winner_7days:
#             mismatches += 1
    thread_winners[(row.article, row.thread_title)].add(row.winner_7days)
for t in thread_winners:
    if len(thread_winners[t]) > 1:
        mismatches +=1 
print(mismatches)

print("Number of threads that had exact opposite revert winners", end='\t')
thread_winners_reverters = defaultdict(set)
opps = 0
for row in thread_data_nonan.itertuples():
    thread_winners_reverters[(row.article, row.thread_title)].add((row.winner_7days, row.editor, row.reverted_editor))
for t in thread_winners_reverters:
    winners = [el[0] for el in thread_winners_reverters[t]]
    for el in thread_winners_reverters[t]:
        if el[1] in winners and el[2] in winners:
            opps += 1
            break
print(opps)


# Stats
print("Number of thread-revert pairs", end='\t\t\t\t')
print(len(thread_data_nonan)) # Number of thread-revert pairs
print("Number of reverter-winning thread-revert pairs", end='\t\t')
print(len(thread_data_nonan[thread_data_nonan['revert_score_7days'] > 0]), end = " ")
print(len(thread_data_nonan[thread_data_nonan['revert_score_7days'] > 0])/len(thread_data_nonan))
print("Number of unique threads", end='\t\t\t\t')
print(len(set(zip(thread_data_nonan['article'], thread_data_nonan['thread_title']))))
print("Number of rows where 14day winner different from 7day", end='\t')
print(len(thread_data_nonan[thread_data_nonan['winner_7days'] != thread_data_nonan['winner_14days']]))
print("Number of rows where 30day winner different from 7day", end='\t')
print(len(thread_data_nonan[thread_data_nonan['winner_7days'] != thread_data_nonan['winner_30days']]))

print("Number of threads that had different revert winners", end='\t')
thread_winners = defaultdict(set)
mismatches = 0
for row in thread_data_nonan.itertuples():
#     if (row.article, row.thread_title) not in thread_winners:
#         thread_winners[(row.article, row.thread_title)] = row.winner_7days
#     else:
#         if thread_winners[(row.article, row.thread_title)] != row.winner_7days:
#             mismatches += 1
    thread_winners[(row.article, row.thread_title)].add(row.winner_7days)
for t in thread_winners:
    if len(thread_winners[t]) > 1:
        mismatches +=1 
print(mismatches)

print("Number of threads that had exact opposite revert winners", end='\t')
thread_winners_reverters = defaultdict(set)
opps = 0
for row in thread_data_nonan.itertuples():
    thread_winners_reverters[(row.article, row.thread_title)].add((row.winner_7days, row.editor, row.reverted_editor))
for t in thread_winners_reverters:
    winners = [el[0] for el in thread_winners_reverters[t]]
    for el in thread_winners_reverters[t]:
        if el[1] in winners and el[2] in winners:
            opps += 1
            break
print(opps)


len(thread_winners)

thread_data_nonan

opps_inds[:10]

thread_data_nonan.loc[21, :]

# GET REVERT DIFF URLS

# Load pickled art_revids, revert_data
with open('/home/michael/school/research/wp/wp_articles/art_revids.pkl', 'rb') as f:
    art_revids = pickle.load(f)
print(len(art_revids.keys()))

rv_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/revert_talk_posts_unique_7days.csv', parse_dates=['revert_timestamp'])
rv_data

rv_data.loc[0, 'revert_timestamp'] + DateOffset(hours=6)

# Get revert urls for timestamps
datetime.strptime(rv_data.loc[0, 'revert_timestamp'], '%Y-%m-%d %H:%M:%S')

art = '1929 Hebron massacre'
rev_url = art_url.format(art, art_revids[(art, datetime(2008, 4, 27, 20, 25, 9))])
rev_url

list(art_revids.keys())[0]

art_url =  'https://en.wikipedia.org/w/index.php?title={0}&oldid={1}'
# rev_urls = pd.Series()
rev_urls = []

for el in zip(rv_data['article_title'], rv_data['revert_timestamp']):
    rev_urls.append(art_url.format(el[0], art_revids.get((el[0], el[1] + DateOffset(hours=6)))))

rv_data['revert_url'] = pd.Series(rev_urls)
rv_data

rv_data.to_csv('/home/michael/school/research/wp/revert_talk_posts_7days_urls.csv', index=False)

# Build article revision dict
rv_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/revert_talk_posts_unique_7days.csv')

arts = set(rv_data['article_title'].values)
print(len(arts))

art_revids = {}  # art_revs[title, ts] = revid
arts_done = []

for i, art in enumerate(arts):
    
    if i % 5 == 0:
        print(i)
        sys.stdout.flush()
    
    # Build article revision table
    baseurl = "https://en.wikipedia.org/w/api.php?action=query&titles={:s}&prop=revisions&rvprop=ids|timestamp&rvlimit=max&format=json"

    url = baseurl.format(urllib.parse.quote(art))
    retrieved =  urllib.request.urlopen(url).read().decode('utf-8')
    pagedict = json.loads(retrieved)

    # page no longer exists with that name
    if not 'revisions' in pagedict['query']['pages'][list(pagedict['query']['pages'].keys())[0]]:
        continue

    revtable = pagedict['query']['pages'][list(pagedict['query']['pages'].keys())[0]]['revisions']

    # Loop thru continues
    while 'continue' in pagedict.keys():
        cnum = re.search(r'rvcontinue":"(.*?)"', retrieved).group(1)

        baseurl = "https://en.wikipedia.org/w/api.php?action=query&titles={:s}&prop=revisions&rvprop=ids|timestamp&rvlimit=max&rvcontinue={:s}&format=json"

        url = baseurl.format(urllib.parse.quote(art), cnum)

        retrieved =  urllib.request.urlopen(url).read().decode('utf-8')

        pagedict = json.loads(retrieved)

        revtable += pagedict['query']['pages'][list(pagedict['query']['pages'].keys())[0]]['revisions']

    # Parse dates in revision table
    for el in revtable:
        el['timestamp'] = datetime.strptime(el['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        art_revids[art,el['timestamp']] = el['revid']
        
    arts_done += art
            
art_revids.keys()

# Pickle art_revids
with open('/home/michael/school/research/wp/wp_articles/art_revids.pkl', 'wb') as f:
    pickle.dump(art_revids, f)

# Get most recent article revision before the talk page date
talk_ts = row['post_timestamp']
if isinstance(talk_ts, str):
    talk_ts = datetime.datetime.strptime(talk_ts, '%Y-%m-%d %H:%M:%S.0')
    
    # Look for most recent article revision in revtable
    most_recent_revid = ""
    
    for el in reversed(revtable):
        if el['timestamp'] <= talk_ts:
            most_recent_revid = el['revid']
        else:
            break

    if most_recent_revid == '': continue # another case where article name changed
        
    diff_url = 'https://en.wikipedia.org/w/index.php?title={0}&diff=prev&oldid={1}'.format(row['article_title'], most_recent_revid)

    talk_data.loc[i, 'article_rev_url'] = rev_url

# talk_data.loc[17580:17600,:]

# Print talk_data
talk_data.to_csv('/home/michael/school/research/wp/wikipedia/data/talk/ipc_talk_article_urls.csv', index=False)

talk_data

# MERGE TALK POST-LEVEL AND THREAD-LEVEL REVERT DOCS
talk_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/ipc_talkpages_byarticle.csv', parse_dates=['post_timestamp'])
rv_data = pd.read_csv('/home/michael/school/research/wp/revert_talk_threads_unique_7days.csv')

rv_talk_data = pd.merge(talk_data, rv_data, left_on=['article_title', 'thread_title'], right_on=['article', 'thread_title'])

rv_talk_data.drop('article', axis=1, inplace=True)

#rv_talk_data.columns = ['talk_page_revision_id', 'article_title', 'thread_title', 'talk_editor', 'post_text',
#                       'post_timestamp', 'reverter', 'reverted', 'revert_timestamp', 'edit_comment', 'discussion_revert_related']

rv_talk_data.columns = ['talk_page_revision_id', 'article_title', 'thread_title', 'talk_editor', 'post_text',
                       'post_timestamp', 'reverter', 'reverted', 'revert_timestamp', 'edit_comment']

rv_talk_data.sort_values(['article_title', 'revert_timestamp'], inplace=True)

rv_talk_data.to_csv('/home/michael/school/research/wp/revert_talk_posts_unique_7days.csv', index=None)
len(rv_talk_data)

# BUILD CSV OF POSSIBLE REVERT DISCUSSIONS (at the thread level)
# Load talk data
talk_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/talk/ipc_talkpages_byarticle.csv', parse_dates=['post_timestamp'])

# Load article revision data (including revisions)
art_data = pd.read_csv('/home/michael/school/research/wp/wikipedia/data/citations_reverts.csv', parse_dates=['timestamp'])
art_data['reverted_editor'] = pd.Series()

# Try to get series of indices and reverted editors in one
reverts = art_data[art_data['revert_md5']==1]
prev_eds = art_data.loc[[i-1 for i in reverts.index],'editor']
art_data['reverted_editor'] = pd.Series(prev_eds.values, index=reverts.index)

# Find discussions around reverts with reverted editor and editor
t_win = 7 # days
relevant_threads = set()

reverts = art_data[art_data['revert_md5']==1]
print(len(reverts))
count = 0
for rv in reverts.itertuples():
    count += 1
    if count % 1000==0:
        print(count/1000, 'k', end='\t')
    
    art_talk = talk_data[talk_data['article_title']==rv.article_name]
    if art_talk.empty: continue
    
    tmin = rv.timestamp - DateOffset(days=t_win)
    tmax = rv.timestamp + DateOffset(days=t_win)
    crit = art_talk['post_timestamp'].map(lambda x: x >= tmin and x <= tmax)
    talk_win = art_talk[crit]
    
    # editor and reverted editor in same thread
    relevant = talk_win[talk_win['username'].isin([rv.editor, rv.reverted_editor])]
    threads = defaultdict(set)
    for row in relevant.itertuples():
        threads[row.article_title, row.thread_title].add(row.username) 
        
    # check for any threads with both editor and reverted editor
    relevant_threads_rv = set()
    for t in threads:
        if len(threads[t]) == 2: # has both editor and reverted editor
            relevant_threads_rv.add((t[0], t[1], rv.editor, rv.reverted_editor, rv.timestamp, rv.edit_comment))
    relevant_threads |= relevant_threads_rv
    
    if count % 1000==0:
        print(len(relevant_threads))

print(len(relevant_threads))

# Print out relevant threads and editors
threadpath = '/home/michael/school/research/wp/revert_talk_threads_{0}days.csv'.format(t_win)

out = pd.DataFrame(list(sorted(relevant_threads)), columns=[
        'article', 'thread_title', 'editor', 'reverted_editor', 'revert_timestamp', 'edit_comment'])
out.to_csv(threadpath, index=False)

# Filter out reverts that match multiple threads
filtered = out.drop_duplicates(subset=['article', 'revert_timestamp'],keep=False)
len(filtered)

filtered.to_csv('/home/michael/school/research/wp/revert_talk_threads_unique_{0}days.csv'.format(t_win), index=False)

# Get threads
threads = defaultdict(list)
prev_thread = ''
for i,row in talk_data.iterrows():
    threads[row['article_title'],row['thread_title']].append((row['post_timestamp'], row['username'], row['post_text']))
len(threads)

# Number of threads with 'revert' in post
count = 0
for t in threads:
    t_revert = False
    for el in threads[t]:
        if 'revert' in str(el[1]):
            t_revert = True
            break
    if t_revert:
        count += 1
        
print(count)

