get_ipython().magic('pylab --no-import-all')

from os import path
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

file = path.join("..", "data", "interim", "df.csv")
df = pd.read_csv(file, index_col=0)

# Get the min of the next three days' temperatures.
three = df[["TEMP" + str(i + 1) for i in range(99)]].shift(-3, axis=1).rolling(3, axis=1).min()
# Get the max of the six temperatures leading up to today.
six = df[["TEMP" + str(i + 1) for i in range(99)]].rolling(6, axis=1).max()
three_over_six = ((three - six) > 0).idxmax(axis=1)  # First column with positive difference = three-over-six day.
three_over_six = three_over_six.apply(lambda x: int(x[4:])).replace(1, np.nan)  # Remove 'TEMP' and convert 1s to NAs.

print("Total number of cycles: {}".format(len(df)))
print("No Pre-Ov calculated: {}".format(three_over_six.isnull().sum()))
print("Accuracy: {}".format(accuracy_score(y_true=df.L_PREOVULATION, y_pred=three_over_six.fillna(-1))))

# Another method--kind of iffy.
no_calc = 0
diff_calc = 0

for idx, row in tqdm(df.iterrows(), total=len(df), leave=True):
    computed_L_PREOVULATION = None
    for i in range(1, 91):
        six_days = [row['TEMP'+str(i+j)] for j in range(0, 6)]
        three_days = [row['TEMP'+str(i+k)] for k in range(6, 9)]
        if min(three_days) > max(six_days):
            computed_L_PREOVULATION = i + 5
            break
    if computed_L_PREOVULATION is None:
        no_calc += 1
    elif computed_L_PREOVULATION != int(row.L_PREOVULATION):
        diff_calc += 1
total_errors = no_calc + diff_calc
print("Total number of cycles: {}".format(len(df)))
print("Total diffs: {} ({:5.2f}%)".format(total_errors, total_errors / len(df) * 100))
print("No Pre-Ov calculated: {}".format(no_calc))
print("Different calculated value: {}".format(diff_calc))
print("Accuracy: {}".format(100 * (1 - total_errors / len(df))))

no_calc = 0
diff_calc = 0

for idx, row in tqdm(df.iterrows(), total=len(df), leave=True):
    computed_L_PREOVULATION = None
    for i in range(1, 91):
        six_days = [row['TEMP'+str(i+j)] for j in range(0, 6)]
        three_days = [row['TEMP'+str(i+k)] for k in range(6, 9)]
        if min(three_days) > max(six_days):
            computed_L_PREOVULATION = i + 5
            break
    if computed_L_PREOVULATION is None:
        no_calc += 1
    elif abs(computed_L_PREOVULATION - int(row.L_PREOVULATION)) > 1:
        diff_calc += 1
total_errors = no_calc + diff_calc
print("Total number of cycles: {}".format(len(df)))
print("Total big diffs: {} ({:5.2f}%)".format(total_errors, total_errors / len(df) * 100))
print("No Pre-Ov calculated: {}".format(no_calc))
print("Difference greater than 1: {}".format(diff_calc))
print("Softened accuracy: {}".format(100 * (1 - total_errors / len(df))))



