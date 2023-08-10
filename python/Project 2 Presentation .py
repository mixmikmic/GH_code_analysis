import pandas as pd
B = pd.read_csv('gabapentinData.csv')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
#matplotlib.style.use('ggplot')
B.groupby('Condition').size().sort_values(ascending=True).plot(kind='barh',fontsize=25,figsize=(20,15),sort_columns= True)
plt.xlabel('Review Count', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Review Count By Condition (Gabapentin)',fontsize= 30)


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')



clonazepam = pd.read_csv('clonazepamData.csv')
clonazepam.groupby('Condition').size().sort_values(ascending=True).plot(kind='barh',fontsize=25,figsize=(20,15),sort_columns= True)
plt.xlabel('Review Count', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Review Count By Condition (Clonazepam)',fontsize= 30)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

Topiramate= pd.read_csv('TopiramateData.csv')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
Topiramate.groupby('Condition').size().sort_values(ascending=True).plot(kind='barh',fontsize=20,figsize=(13,10),sort_columns= True)
plt.xlabel('Review Count', fontsize=20)
plt.ylabel('Condition', fontsize=20)
plt.title('Review Count By Condition (Topiramate)',fontsize= 25)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

import pandas as pd
Pregabalin = pd.read_csv('PregabalinData.csv')
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
Pregabalin.groupby('Condition').size().sort_values(ascending=True).plot(kind='barh',fontsize=25,figsize=(13,10),sort_columns= True)
plt.xlabel('Review Count', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Review Count By Condition (Pregabalin)',fontsize= 25)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')



DataWithRating = B[B['Rating'].isnull()==False]
DataWithRating.groupby('Condition').mean().sort_values("Rating", ascending=True).plot(fontsize=22,kind = 'barh',color = 'g',figsize=(20,15))
plt.xlabel('Average Rating', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Average Rating By Condition (Gabapentin)',fontsize= 30)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')



DataWithRating1 = clonazepam[clonazepam['Rating'].isnull()==False]
DataWithRating1.groupby('Condition').mean().sort_values("Rating", ascending=True).plot(fontsize=22,kind = 'barh',color = 'g',figsize=(20,15))
plt.xlabel('Average Rating', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Average Rating By Condition (Clonazepam)',fontsize= 30)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')



DataWithRating2 = Topiramate[Topiramate['Rating'].isnull()==False]
DataWithRating2.groupby('Condition').mean().sort_values("Rating", ascending=True).plot(fontsize=20,kind = 'barh',color = 'g',figsize=(20,15))
plt.xlabel('Average Rating', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Average Rating By Condition (Topiramate)',fontsize= 30)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

DataWithRating3 = Pregabalin[Pregabalin['Rating'].isnull()==False]
DataWithRating3.groupby('Condition').mean().sort_values("Rating", ascending=True).plot(fontsize=22,kind = 'barh',color = 'g',figsize=(20,15))
plt.xlabel('Average Rating', fontsize=25)
plt.ylabel('Condition', fontsize=25)
plt.title('Average Rating By Condition (Pregabalin)',fontsize= 30)

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

AvgRating=[DataWithRating.mean(),DataWithRating1.mean(),DataWithRating2.mean(),DataWithRating3.mean()]
DrugAvgRating = pd.DataFrame(AvgRating, index=['Gabapentin','Clonazepam','Topiramate','Pregabalin'])

DrugAvgRating.plot.barh()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.title('Average Rating Comparison')
plt.xlabel('Average Rating')
plt.ylabel('Drugs')

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')



from os import path
import PIL
import matplotlib.pyplot as plt
from wordcloud import WordCloud
m = B.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(m)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

k = clonazepam.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(k)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()



from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

n = Topiramate.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(n)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

from os import path
import PIL
import matplotlib.pyplot as plt
from wordcloud import WordCloud
n = Pregabalin.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(n)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()

from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

k = clonazepam.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(k)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()


from os import path
import PIL
import matplotlib.pyplot as plt
from wordcloud import WordCloud
m = B.Review.str.cat(sep=', ')
wordcloud = WordCloud().generate(m)
# Open a plot of the generated image.
plt.imshow(wordcloud)
plt.axis("off");
plt.show()

