get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from xlwings import Range, Sheet, Workbook, Plot

wb = Workbook.active()

df = Range('A1').table.options(pd.DataFrame).value

df = pd.read_excel(r'D:\your_file.xlsx', sheetname='Claims')

fig1 = plt.figure()
fig1.set_size_inches(10, 6)
sns.boxplot(y="DAYS_TO_FAIL_MINZERO", x='GRADE_SHORT', data=df)
sns.despine(offset=10, trim=True)
plt.title('Days To Fail', weight='bold')
plot = Plot(fig1)
plot.show('Plot1', width=400, height=300)

fig2 = plt.figure()
fig2.set_size_inches(10, 6)
sns.boxplot(y="MILES_TO_FAIL", x='GRADE_SHORT', data=df)
sns.despine(offset=10, trim=True)
plt.title('Miles To Fail', weight='bold')
plot = Plot(fig2)
plot.show('Plot2', left=400, width=400, height=300)

