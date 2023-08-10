from plotly.offline import download_plotlyjs, init_notebook_mode # doesn't help
from IPython.core.display import display, HTML
import json
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

init_notebook_mode()

# Maximize visual real estate
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; margin-left:0%;  margin-right:auto; }</style>"))

def plotlyfromjson(fpath):
    """Render a plotly figure from a json file"""
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        v = json.loads(f.read())

    fig = go.Figure(data=v['data'], layout=v['layout'])
    iplot(fig, show_link=False)

import time; print("Last updated "  + time.strftime("%x"))

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
Toggle <a href="javascript:code_toggle()">code visibility</a>.''')

plotlyfromjson('gallup_chart.json')

plotlyfromjson('reddit_activity_chart.json')

plotlyfromjson('main_chart.json')

