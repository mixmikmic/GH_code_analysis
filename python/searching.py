from pandas import read_csv
df = read_csv('test_data.csv', index_col=None)
df = df[['time', 'UV', 'baromin', 'humidity', 'light', 'rainin', 'temp', 'windgust']]

from dateutil.parser import parse
def get_hour (x):
    y = parse(x)
    return y.hour + y.minute / 60.0

df["time"] = [get_hour(x) / 24.0 for x in df["time"].values]

parameters_map = {
    "normalized"    : {
        "humidity": "rh",
        "windgust": "ws",
        "UV"      : "uv",
        "light"   : "light",
    },
    "not_normalized": {
        "time"  : "time",
        "rainin": "rain",
    },
    "target"    : "temp",
}

framework_parameters = {
    "starting_lr"   : 0.001,
    "max_iterations": int(1e4),
    "momentum"      : 0.95,
}

from athena.framework import Framework
fw = Framework(framework_parameters)

from athena.dataset import Dataset
from athena.helpers import split_dataframe
training_df, testing_df = split_dataframe(df, 0.9)
fw.add_dataset(Dataset(training_df, testing_df, parameters_map))

from athena.searching import RandomSearch
rs = RandomSearch(fw, search_length=50, equation_length=10)
rs.search()

equation = rs.get_best_equations()[0]
equation["testing_pearson"]*100

from sympy import N, nsimplify, init_printing
init_printing()
N(nsimplify(equation["equation"], tolerance=1e-2), 2)

