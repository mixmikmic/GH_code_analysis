get_ipython().magic('matplotlib inline')

import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')

len(df)

df.head()

sns.boxplot(x="day", y="total_bill", hue="time", data=df, palette="PRGn")

def plot(x='day', y='total_bill', hue='sex'):
    '''Draws the plot and returns the figure for display.'''
    fig, ax = plt.subplots(figsize=(9,5))
    sns.boxplot(x=x, y=y, hue=hue, data=df, palette="PRGn", ax=ax)
    plt.tight_layout()
    plt.close()
    return fig

import declarativewidgets
declarativewidgets.init()

get_ipython().run_cell_magic('html', '', '<link rel="import" href="urth_components/paper-dropdown-menu/paper-dropdown-menu.html" \n    is=\'urth-core-import\' package=\'PolymerElements/paper-dropdown-menu\'>\n<link rel="import" href="urth_components/paper-menu/paper-menu.html"\n    is=\'urth-core-import\' package=\'PolymerElements/paper-menu\'>\n<link rel="import" href="urth_components/paper-item/paper-item.html"\n    is=\'urth-core-import\' package=\'PolymerElements/paper-item\'>\n    \n<style>\n    div.output_wrapper {\n        z-index: auto; /* fixes menus showing under code cells */\n    }\n    div.controls span {\n        padding: 0 20px;\n    }\n    div.controls h3 {\n        margin-bottom: 20px;\n    }\n    div.controls {\n        text-align: center;\n    }\n    div.plot img {\n        margin-left: auto !important;\n        margin-right: auto !important;\n    }\n</style>')

from urth.widgets.widget_channels import channel

numeric = [name for name in df.columns if df[name].dtype in [float, int]]
channel('default').set('numeric', numeric)
channel('default').set('categorical', [name for name in df.columns if name not in numeric])

get_ipython().run_cell_magic('html', '', '<template is="urth-core-bind">\n  <div class="controls">\n    <h3>Tips Dataset</h3>\n    <span>Plot</span>\n    <paper-dropdown-menu label="Select x-axis" selected-item-label="{{ x }}" noink>\n        <paper-menu class="dropdown-content" selected="[[ x ]]" attr-for-selected="label">\n            <template is="dom-repeat" items="[[ categorical ]]">\n                <paper-item label="[[ item ]]">[[item]]</paper-item>\n            </template>\n        </paper-menu>\n    </paper-dropdown-menu>\n    <span>by</span>\n    <paper-dropdown-menu label="Select y-axis" selected-item-label="{{ y }}" noink>\n        <paper-menu class="dropdown-content" selected="[[ y ]]" attr-for-selected="label">\n            <template is="dom-repeat" items="[[ numeric ]]">\n                <paper-item label="[[ item ]]">[[item]]</paper-item>\n            </template>\n        </paper-menu>\n    </paper-dropdown-menu>\n    <span>colored by</span>\n    <paper-dropdown-menu label="Select hue" selected-item-label="{{ hue }}" noink>\n        <paper-menu class="dropdown-content" selected="[[ hue ]]" attr-for-selected="label">\n            <template is="dom-repeat" items="[[ categorical ]]">\n                <paper-item label="[[ item ]]">[[item]]</paper-item>\n            </template>\n        </paper-menu>\n    </paper-dropdown-menu> \n  </div>\n</template>')

get_ipython().run_cell_magic('html', '', '<template is="urth-core-bind">\n  <div class="plot">\n    <urth-core-function ref="plot" arg-x="{{x}}" arg-y="{{y}}" arg-hue="{{hue}}" result="{{plot}}" auto />\n    <img src="[[plot]]" />\n  </div>\n</template>')

