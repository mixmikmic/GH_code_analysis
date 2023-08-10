import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import randn
import numpy as np
import pandas as pd
from IPython.display import Image

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

plt.title('Title')

# 坐标轴
plt.axis([-5, 5, -10, 10])
# plt.axis('tight') # 紧凑型坐标轴

# x轴
plt.xlabel(r'$\sum_{xlabel}$')
plt.axhline(); # 添加水平线，重要参数包括：y轴位置、xmin、xmax，默认在y=0位置绘制水平线
plt.axhspan(-5, -5.5); # 绘制一条水平带（矩形），需要ymin、ymax参数指定水平带的宽度（ymin、ymax都是实际值）
plt.xticks(range(len(x)), ['a', 'b', r'$\tilde{c}$', '$\mathcal{d}$', 'e', 'f']); # 传入位置和标签参数，以修改坐标轴刻度

# y轴
plt.ylabel('ylabel')
plt.axvline(3, 0.2, 0.8); # 添加垂直线，重要参数包括：x轴位置、ymin、ymax，默认在x=0位置绘制水平线(ymin、ymax都是比例值)
plt.axvspan(-0.95, -0.85); # 绘制一条垂直带（矩形），需要xmin、xmax参数指定水平带的宽度（xmin、xmax都是实际值）
plt.yticks(range(1, 8, 2));

# 曲线 dashes=[2, 5, 5, 2] marker="None"
x = np.arange(-5, 6) # x = [-5,6)
plt.plot(x, x*1.5,  
         linestyle='dashdot', linewidth=4, 
         marker='o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=3, markersize=12,
         color="#cc55dd", alpha=0.8,
         label='Normal')
plt.plot(x, x*3.0, 'k--',lw=2, color=(0.43, 0.8, 0.68), label='Fast')
plt.plot(x, x/3.0, 'k.', lw=3, c="red", alpha=0.9, label='_nolegend_')

# 图例
plt.legend(loc='upper left', ncol=2) # ncol控制图例中有几列
# plt.legend(loc=(0,1)) # loc参数可以是2元素的元组，表示图例左下角的坐标
# plt.legend(loc=(-0.1,0.9)) # 图例也可以超过图的界限
# plt.legend(loc='best') # loc='best'
plt.grid(True)

plt.show()

# 图形选择建议
Image('chart suggestions.png')

