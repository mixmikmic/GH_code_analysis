import pandas as pd

students = pd.read_csv("./input/san-diego-2016-students.csv")

students['in_coalition'] = False
students['contributions'] = 0
students['at_san_diego_2016_training'] = True

students.info()

participants = pd.read_csv("./output/participants.csv")

participants['at_san_diego_2016_training'] = False

participants.info()

merged = pd.concat([participants, students])

merged.tail(5)

merged.to_csv("./output/participants.csv", index=False)

