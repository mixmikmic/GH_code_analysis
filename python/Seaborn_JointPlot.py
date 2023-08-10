get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xlwings import Range, Sheet, Workbook, Plot

# Import data from active worksheet
wb = Workbook.active()

df = Range('A1').table.options(pd.DataFrame).value

df = pd.read_excel(r'D:\your_file.xlsx', sheetname='Claims')

sns.jointplot(df.DAYS_TO_FAIL_MINZERO, df.MILES_TO_FAIL, kind="kde", xlim=(-200,2000), ylim=(-3000, 100000))
fig = plt.gcf()
fig.set_size_inches(8, 8)
plot = Plot(fig)
plot.show('Plot3', top=0, width=400, height=400)

fig2 = plt.figure()
fig2.set_size_inches(10, 6)
sns.boxplot(y="MILES_TO_FAIL", x='MODEL_NAME', data=df)
sns.despine(offset=10, trim=True)
plt.title('Miles To Fail', weight='bold')
plot = Plot(fig2)
plot.show('Plot2', left=400, width=400, height=300)

