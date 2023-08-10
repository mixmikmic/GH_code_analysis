import a8c_styling
import seaborn as sns
from matplotlib import pylab as plt

plt.figure()
a8c_styling.sinplot(4); #this is how a default plot looks like

plt.figure()
a8c_styling.sinplot(4)
plt.ylabel('this is Y',
           **a8c_styling.ylabelparams) #horizontal label for
                                    #more readability
plt.title('This is an improved title', **a8c_styling.axtitleparams)

a8c_styling.cleanup()



styles = ['a8c_style', 'a8c_style_white', 'a8c_style_gray']
for s in styles:
    plt.figure()
    style = getattr(a8c_styling, s)
    with sns.axes_style(style):
        fig = a8c_styling.sinplot(4)
        gca().set_title("Style: %s" %s, **a8c_styling.axtitleparams)
        a8c_styling.cleanup()
    
    

sns.set_style(a8c_styling.a8c_style)
a8c_styling.sinplot()
sns.despine()



