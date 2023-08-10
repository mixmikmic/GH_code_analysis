# set up all the data for the rest of the notebook
import json
from collections import Counter
from itertools import chain
from IPython.display import HTML

def vote_table(votes):
    """Render a crappy HTML table for easy display. I'd use Pandas, but that seems like
    complete overkill for this simple task.
    """
    base_table = """
    <table>
        <tr><td>Position</td><td>Album</td><td>Votes</td></tr>
        {}
    </table>
    """
    
    base_row = "<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>"
    vote_rows = [base_row.format(idx, name, vote) for idx, (name, vote) in enumerate(votes, 1)]
    return HTML(base_table.format('\n'.join(vote_rows)))

with open('shreddit_q2_votes.json', 'r') as fh:
    ballots = json.load(fh)

with open('tallied_votes.json', 'r') as fh:
    tallied = Counter(json.load(fh))

equal_placement_ballots = Counter(chain.from_iterable(ballots))

vote_table(tallied.most_common(10))

vote_table(equal_placement_ballots.most_common(10))

weighted_ballot = Counter()

for ballot in ballots:
    for item, weight in zip(ballot, range(5, 0, -1)):
        weighted_ballot[item] += weight

sum(1 for _ in filter(lambda x: len(x) < 5, ballots)) / len(ballots)

vote_table(weighted_ballot.most_common(10))

regular_tally_spots = {name.lower(): pos for pos, (name, _) in enumerate(tallied.most_common(), 1)}

base_table = """
<table>
    <tr><td>Album</td><td>Regular Spot</td><td>Weighted Spot</td></tr>
    {}
</table>
"""
base_row = "<tr><td>{0}</td><td>{1}</td><td>{2}</td></tr>"

rows = [base_row.format(name, regular_tally_spots[name], pos) 
        for pos, (name, _) in enumerate(weighted_ballot.most_common(), 1)
        # some albums didn't make it, like Arcturian D:
        if name in regular_tally_spots]

HTML(base_table.format('\n'.join(rows)))

number_one = Counter([b[0] for b in ballots]) 
vote_table(number_one.most_common(10))

#regular tallying
vote_table(equal_placement_ballots.most_common())

#weighted ballot
vote_table(weighted_ballot.most_common())

#number one count
vote_table(number_one.most_common())



