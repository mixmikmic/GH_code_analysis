# Import required libraries

import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output
import quantmod as qm

# Create layout

# Dash app instantiation
app = dash.Dash("Stock market app")

# External CSS
app.css.append_css({
    'external_url': (
        'https://rawgit.com/chriddyp/0247653a7c52feb4c48437e1c1837f75'
        '/raw/a68333b876edaf62df2efa7bac0e9b3613258851/dash.css'
    )
})

# Create app layout
app.layout = html.Div(
    [
        html.H1('Quantmod Demo | 5-minute App'),
        # Dropdown for stocks
        core.Dropdown(
            id='dropdown',
            options=[
                dict(label='PowerShares QQQ Trust Series 1', value='QQQ'),
                dict(label='SPDR S&P 500 ETF Trust', value='SPY'),
                dict(label='Apple Inc', value='AAPL'),
                dict(label='Goldman Sachs Group Inc', value='GS'),
            ],
            value='SPY',
        ),
        # Dropdown for indicators
        core.Dropdown(
            id='multi',
            options=[
                dict(label='EMA', value='EMA'),
                dict(label='RSI', value='RSI'),
                dict(label='MACD', value='MACD'),
                dict(label='BBANDS', value='BBANDS'),
            ],
            multi=True,
            value=[],
        ),
        # Graph output
        core.Graph(id='output')
    ]
)

# Setup callbacks

# Graph is modified by 2 inputs
@app.callback(Output('output', 'figure'), [Input('dropdown', 'value'),
                                       Input('multi', 'value')])
def update_graph_from_dropdown(dropdown, multi):
    # Get Quantmod Chart
    ch = qm.get_symbol(dropdown, start='2016/01/01')

    if 'EMA' in multi:
        ch.add_EMA()
    if 'RSI' in multi:
        ch.add_RSI()
    if 'MACD' in multi:
        ch.add_MACD()
    if 'BBANDS' in multi:
        ch.add_BBANDS()

    # Return plot as figure
    return ch.to_figure()

if __name__ == '__main__':
    app.run_server(debug=True, port=4001)

