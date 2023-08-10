import timeit

import numpy as np
import pandas as pd

import axelrod as axl

strategies = axl.strategies
reps = 20

def play_matches(s1, s2, reps=20):
    match = axl.Match(players=(s1(), s2()), turns=100)
    for _ in range(reps):
        match.play()

data = []        
        
for s1 in strategies:
    times = []
    for s2 in strategies:
        t = timeit.timeit(lambda: play_matches(s1, s2, reps=reps), number=1)
        times.append(t / float(reps))
    data.append((str(s1()), np.mean(times), np.std(times)))

df = pd.DataFrame(data, columns=["Player Name", "Mean Time", "Std Time"])
df.sort_values(by="Mean Time", inplace=True, ascending=False)

print(df)

df.to_csv("runtimes.csv")



