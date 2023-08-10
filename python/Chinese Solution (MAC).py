import matplotlib as mpl
from matplotlib import pyplot as plt
# Needs to modify this path if:
#     1) you are on different system
#     2) you want other font family
path = '/Library/Fonts/儷黑 Pro.ttf'
prop = mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()
get_ipython().magic('matplotlib inline')

plt.figure()
plt.plot([1, 2], [1, 3])
plt.title('測試中文 title')
plt.xlabel('測試中文 x')
plt.ylabel('測試中文 y')
plt.text(1.6, 2, '測試中文 label')
plt.axis([0, 3, 0, 4])

