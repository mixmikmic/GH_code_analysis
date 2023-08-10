values = np.array([  33.35107553,  107.61654307,  126.12128266,  100.70309131,  129.16130288,
  153.28388564,   65.74439386,  203.96304796])

average = np.sum(values)/len(values)
print(average)

R = np.max(values) - np.min(values)
print(R)

dAverage = R/(2*np.sqrt(len(values)))
print(dAverage)

