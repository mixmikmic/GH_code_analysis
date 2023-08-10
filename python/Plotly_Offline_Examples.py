import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

df['text'] = df['state'] + '<br>' +    'Beef '+df['beef']+' Dairy '+df['dairy']+'<br>'+    'Fruits '+df['total fruits']+' Veggies ' + df['total veggies']+'<br>'+    'Wheat '+df['wheat']+' Corn '+df['corn']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df['code'],
        z = df['total exports'].astype(float),
        locationmode = 'USA-states',
        text = df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Millions USD"
        )
    ) ]

layout = dict(
        title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        )
    )
    
fig = dict( data=data, layout=layout )

iplot(fig, show_link=False )

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

# Create random data with numpy
import numpy as np

N = 1000
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Create a trace
trace = Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
iplot(data, show_link=False)

import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

trace0 = Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker=dict(
        size=[40, 60, 80, 100],
    )
)
data = [trace0]
layout = Layout(
    showlegend=False,
    height=600,
    width=600
)

fig = dict( data=data, layout=layout )

iplot(fig, show_link=False)

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

trace0 = Scatter(
    x=[6223.367465, 4797.231267, 1441.284873, 12569.851770000001, 1217.032994,
        430.07069160000003, 2042.0952399999999, 706.016537, 1704.0637239999999,
        986.1478792000001, 277.55185869999997, 3632.557798, 1544.750112,
        2082.4815670000003, 5581.180998, 12154.08975, 641.3695236000001,
        690.8055759, 13206.48452, 752.7497265, 1327.60891, 942.6542111,
        579.2317429999999, 1463.249282, 1569.331442, 414.5073415, 12057.49928,
        1044.770126, 759.3499101, 1042.581557, 1803.1514960000002, 10956.99112,
        3820.17523, 823.6856205, 4811.060429, 619.6768923999999,
        2013.9773050000001, 7670.122558, 863.0884639000001, 1598.435089,
        1712.4721359999999, 862.5407561000001, 926.1410683, 9269.657808,
        2602.394995, 4513.480643, 1107.482182, 882.9699437999999, 7092.923025,
        1056.3801210000001, 1271.211593, 469.70929810000007],
    y=[72.301, 42.731, 56.728, 50.728, 52.295, 49.58, 50.43, 44.74100000000001,
       50.651, 65.152, 46.461999999999996, 55.321999999999996, 48.328,
       54.791000000000004, 71.33800000000001, 51.57899999999999, 58.04,
       52.946999999999996, 56.735, 59.448, 60.022, 56.007, 46.388000000000005,
       54.11, 42.592, 45.678000000000004, 73.952, 59.443000000000005,
       48.303000000000004, 54.467, 64.164, 72.801, 71.164, 42.082,
       52.906000000000006, 56.867, 46.858999999999995, 76.442, 46.242,
       65.528, 63.062, 42.568000000000005, 48.159, 49.339, 58.556000000000004,
       39.613, 52.516999999999996, 58.42, 73.923, 51.542, 42.38399999999999,
       43.486999999999995],
    mode='markers',
    name='Africa',
    text=['Country: Algeria<br>Life Expectancy: 72.301<br>GDP per capita: 6223.367465<br>Population: 33333216.0<br>Year: 2007', 'Country: Angola<br>Life Expectancy: 42.731<br>GDP per capita: 4797.231267<br>Population: 12420476.0<br>Year: 2007', 'Country: Benin<br>Life Expectancy: 56.728<br>GDP per capita: 1441.284873<br>Population: 8078314.0<br>Year: 2007', 'Country: Botswana<br>Life Expectancy: 50.728<br>GDP per capita: 12569.85177<br>Population: 1639131.0<br>Year: 2007', 'Country: Burkina Faso<br>Life Expectancy: 52.295<br>GDP per capita: 1217.032994<br>Population: 14326203.0<br>Year: 2007', 'Country: Burundi<br>Life Expectancy: 49.58<br>GDP per capita: 430.0706916<br>Population: 8390505.0<br>Year: 2007', 'Country: Cameroon<br>Life Expectancy: 50.43<br>GDP per capita: 2042.09524<br>Population: 17696293.0<br>Year: 2007', 'Country: Central African Republic<br>Life Expectancy: 44.741<br>GDP per capita: 706.016537<br>Population: 4369038.0<br>Year: 2007', 'Country: Chad<br>Life Expectancy: 50.651<br>GDP per capita: 1704.063724<br>Population: 10238807.0<br>Year: 2007', 'Country: Comoros<br>Life Expectancy: 65.152<br>GDP per capita: 986.1478792<br>Population: 710960.0<br>Year: 2007', 'Country: Congo, Dem. Rep.<br>Life Expectancy: 46.462<br>GDP per capita: 277.5518587<br>Population: 64606759.0<br>Year: 2007', 'Country: Congo, Rep.<br>Life Expectancy: 55.322<br>GDP per capita: 3632.557798<br>Population: 3800610.0<br>Year: 2007', "Country: Cote d'Ivoire<br>Life Expectancy: 48.328<br>GDP per capita: 1544.750112<br>Population: 18013409.0<br>Year: 2007", 'Country: Djibouti<br>Life Expectancy: 54.791<br>GDP per capita: 2082.481567<br>Population: 496374.0<br>Year: 2007', 'Country: Egypt<br>Life Expectancy: 71.338<br>GDP per capita: 5581.180998<br>Population: 80264543.0<br>Year: 2007', 'Country: Equatorial Guinea<br>Life Expectancy: 51.579<br>GDP per capita: 12154.08975<br>Population: 551201.0<br>Year: 2007', 'Country: Eritrea<br>Life Expectancy: 58.04<br>GDP per capita: 641.3695236<br>Population: 4906585.0<br>Year: 2007', 'Country: Ethiopia<br>Life Expectancy: 52.947<br>GDP per capita: 690.8055759<br>Population: 76511887.0<br>Year: 2007', 'Country: Gabon<br>Life Expectancy: 56.735<br>GDP per capita: 13206.48452<br>Population: 1454867.0<br>Year: 2007', 'Country: Gambia<br>Life Expectancy: 59.448<br>GDP per capita: 752.7497265<br>Population: 1688359.0<br>Year: 2007', 'Country: Ghana<br>Life Expectancy: 60.022<br>GDP per capita: 1327.60891<br>Population: 22873338.0<br>Year: 2007', 'Country: Guinea<br>Life Expectancy: 56.007<br>GDP per capita: 942.6542111<br>Population: 9947814.0<br>Year: 2007', 'Country: Guinea-Bissau<br>Life Expectancy: 46.388<br>GDP per capita: 579.231743<br>Population: 1472041.0<br>Year: 2007', 'Country: Kenya<br>Life Expectancy: 54.11<br>GDP per capita: 1463.249282<br>Population: 35610177.0<br>Year: 2007', 'Country: Lesotho<br>Life Expectancy: 42.592<br>GDP per capita: 1569.331442<br>Population: 2012649.0<br>Year: 2007', 'Country: Liberia<br>Life Expectancy: 45.678<br>GDP per capita: 414.5073415<br>Population: 3193942.0<br>Year: 2007', 'Country: Libya<br>Life Expectancy: 73.952<br>GDP per capita: 12057.49928<br>Population: 6036914.0<br>Year: 2007', 'Country: Madagascar<br>Life Expectancy: 59.443<br>GDP per capita: 1044.770126<br>Population: 19167654.0<br>Year: 2007', 'Country: Malawi<br>Life Expectancy: 48.303<br>GDP per capita: 759.3499101<br>Population: 13327079.0<br>Year: 2007', 'Country: Mali<br>Life Expectancy: 54.467<br>GDP per capita: 1042.581557<br>Population: 12031795.0<br>Year: 2007', 'Country: Mauritania<br>Life Expectancy: 64.164<br>GDP per capita: 1803.151496<br>Population: 3270065.0<br>Year: 2007', 'Country: Mauritius<br>Life Expectancy: 72.801<br>GDP per capita: 10956.99112<br>Population: 1250882.0<br>Year: 2007', 'Country: Morocco<br>Life Expectancy: 71.164<br>GDP per capita: 3820.17523<br>Population: 33757175.0<br>Year: 2007', 'Country: Mozambique<br>Life Expectancy: 42.082<br>GDP per capita: 823.6856205<br>Population: 19951656.0<br>Year: 2007', 'Country: Namibia<br>Life Expectancy: 52.906<br>GDP per capita: 4811.060429<br>Population: 2055080.0<br>Year: 2007', 'Country: Niger<br>Life Expectancy: 56.867<br>GDP per capita: 619.6768924<br>Population: 12894865.0<br>Year: 2007', 'Country: Nigeria<br>Life Expectancy: 46.859<br>GDP per capita: 2013.977305<br>Population: 135031164.0<br>Year: 2007', 'Country: Reunion<br>Life Expectancy: 76.442<br>GDP per capita: 7670.122558<br>Population: 798094.0<br>Year: 2007', 'Country: Rwanda<br>Life Expectancy: 46.242<br>GDP per capita: 863.0884639<br>Population: 8860588.0<br>Year: 2007', 'Country: Sao Tome and Principe<br>Life Expectancy: 65.528<br>GDP per capita: 1598.435089<br>Population: 199579.0<br>Year: 2007', 'Country: Senegal<br>Life Expectancy: 63.062<br>GDP per capita: 1712.472136<br>Population: 12267493.0<br>Year: 2007', 'Country: Sierra Leone<br>Life Expectancy: 42.568<br>GDP per capita: 862.5407561<br>Population: 6144562.0<br>Year: 2007', 'Country: Somalia<br>Life Expectancy: 48.159<br>GDP per capita: 926.1410683<br>Population: 9118773.0<br>Year: 2007', 'Country: South Africa<br>Life Expectancy: 49.339<br>GDP per capita: 9269.657808<br>Population: 43997828.0<br>Year: 2007', 'Country: Sudan<br>Life Expectancy: 58.556<br>GDP per capita: 2602.394995<br>Population: 42292929.0<br>Year: 2007', 'Country: Swaziland<br>Life Expectancy: 39.613<br>GDP per capita: 4513.480643<br>Population: 1133066.0<br>Year: 2007', 'Country: Tanzania<br>Life Expectancy: 52.517<br>GDP per capita: 1107.482182<br>Population: 38139640.0<br>Year: 2007', 'Country: Togo<br>Life Expectancy: 58.42<br>GDP per capita: 882.9699438<br>Population: 5701579.0<br>Year: 2007', 'Country: Tunisia<br>Life Expectancy: 73.923<br>GDP per capita: 7092.923025<br>Population: 10276158.0<br>Year: 2007', 'Country: Uganda<br>Life Expectancy: 51.542<br>GDP per capita: 1056.380121<br>Population: 29170398.0<br>Year: 2007', 'Country: Zambia<br>Life Expectancy: 42.384<br>GDP per capita: 1271.211593<br>Population: 11746035.0<br>Year: 2007', 'Country: Zimbabwe<br>Life Expectancy: 43.487<br>GDP per capita: 469.7092981<br>Population: 12311143.0<br>Year: 2007'],
    marker=dict(
        symbol='circle',
        sizemode='diameter',
        sizeref=0.85,
        size=[29.810746602820924, 18.197149567147044, 14.675557544415877,
              6.610603004351287, 19.543385335458176, 14.956442130894114,
              21.72077890062975, 10.792626698654045, 16.52185943835442,
              4.353683242838546, 41.50240100063496, 10.066092062338873,
              21.91453196050797, 3.6377994860079204, 46.258986486204044,
              3.8334450569607683, 11.437310410545528, 45.16465542353964,
              6.227961099314154, 6.709136738617642, 24.694430700391482,
              16.285386604676816, 6.264612285824508, 30.812100863425822,
              7.325179403286266, 9.227791164226492, 12.68649752933601,
              22.60573984618565, 18.849582296257626, 17.910159625556144,
              9.337109185582111, 5.774872714286052, 29.999726284159046,
              23.063420581238734, 7.40199199438875, 18.54140518159347, 60,
              4.612764339536968, 15.369704446995708, 2.3067029222366395,
              18.084735199216812, 12.79910818701753, 15.592022291528775,
              34.24915519732991, 33.57902844158756, 5.496191404660524,
              31.887651824471956, 12.329112567064463, 16.55196774082315,
              27.887232791984047, 17.696194784090615, 18.11688103909921],
        line=dict(
            width=2
        ),
    )
)
trace1 = Scatter(
    x=[12779.379640000001, 3822.1370840000004, 9065.800825, 36319.235010000004,
       13171.63885, 7006.580419, 9645.06142, 8948.102923, 6025.374752000001,
       6873.262326000001, 5728.353514, 5186.050003, 1201.637154,
       3548.3308460000003, 7320.880262000001, 11977.57496, 2749.320965,
       9809.185636, 4172.838464, 7408.905561, 19328.70901, 18008.50924,
       42951.65309, 10611.46299, 11415.805690000001],
    y=[75.32, 65.554, 72.39, 80.653, 78.553, 72.889, 78.782, 78.273, 72.235,
       74.994, 71.878, 70.259, 60.916000000000004, 70.19800000000001, 72.567,
       76.195, 72.899, 75.53699999999999, 71.752, 71.421, 78.74600000000001,
       69.819, 78.242, 76.384, 73.747],
    mode='markers',
    name='Americas',
    text=['Country: Argentina<br>Life Expectancy: 75.32<br>GDP per capita: 12779.37964<br>Population: 40301927.0<br>Year: 2007', 'Country: Bolivia<br>Life Expectancy: 65.554<br>GDP per capita: 3822.137084<br>Population: 9119152.0<br>Year: 2007', 'Country: Brazil<br>Life Expectancy: 72.39<br>GDP per capita: 9065.800825<br>Population: 190010647.0<br>Year: 2007', 'Country: Canada<br>Life Expectancy: 80.653<br>GDP per capita: 36319.23501<br>Population: 33390141.0<br>Year: 2007', 'Country: Chile<br>Life Expectancy: 78.553<br>GDP per capita: 13171.63885<br>Population: 16284741.0<br>Year: 2007', 'Country: Colombia<br>Life Expectancy: 72.889<br>GDP per capita: 7006.580419<br>Population: 44227550.0<br>Year: 2007', 'Country: Costa Rica<br>Life Expectancy: 78.782<br>GDP per capita: 9645.06142<br>Population: 4133884.0<br>Year: 2007', 'Country: Cuba<br>Life Expectancy: 78.273<br>GDP per capita: 8948.102923<br>Population: 11416987.0<br>Year: 2007', 'Country: Dominican Republic<br>Life Expectancy: 72.235<br>GDP per capita: 6025.374752<br>Population: 9319622.0<br>Year: 2007', 'Country: Ecuador<br>Life Expectancy: 74.994<br>GDP per capita: 6873.262326<br>Population: 13755680.0<br>Year: 2007', 'Country: El Salvador<br>Life Expectancy: 71.878<br>GDP per capita: 5728.353514<br>Population: 6939688.0<br>Year: 2007', 'Country: Guatemala<br>Life Expectancy: 70.259<br>GDP per capita: 5186.050003<br>Population: 12572928.0<br>Year: 2007', 'Country: Haiti<br>Life Expectancy: 60.916<br>GDP per capita: 1201.637154<br>Population: 8502814.0<br>Year: 2007', 'Country: Honduras<br>Life Expectancy: 70.198<br>GDP per capita: 3548.330846<br>Population: 7483763.0<br>Year: 2007', 'Country: Jamaica<br>Life Expectancy: 72.567<br>GDP per capita: 7320.880262<br>Population: 2780132.0<br>Year: 2007', 'Country: Mexico<br>Life Expectancy: 76.195<br>GDP per capita: 11977.57496<br>Population: 108700891.0<br>Year: 2007', 'Country: Nicaragua<br>Life Expectancy: 72.899<br>GDP per capita: 2749.320965<br>Population: 5675356.0<br>Year: 2007', 'Country: Panama<br>Life Expectancy: 75.537<br>GDP per capita: 9809.185636<br>Population: 3242173.0<br>Year: 2007', 'Country: Paraguay<br>Life Expectancy: 71.752<br>GDP per capita: 4172.838464<br>Population: 6667147.0<br>Year: 2007', 'Country: Peru<br>Life Expectancy: 71.421<br>GDP per capita: 7408.905561<br>Population: 28674757.0<br>Year: 2007', 'Country: Puerto Rico<br>Life Expectancy: 78.746<br>GDP per capita: 19328.70901<br>Population: 3942491.0<br>Year: 2007', 'Country: Trinidad and Tobago<br>Life Expectancy: 69.819<br>GDP per capita: 18008.50924<br>Population: 1056608.0<br>Year: 2007', 'Country: United States<br>Life Expectancy: 78.242<br>GDP per capita: 42951.65309<br>Population: 301139947.0<br>Year: 2007', 'Country: Uruguay<br>Life Expectancy: 76.384<br>GDP per capita: 10611.46299<br>Population: 3447496.0<br>Year: 2007', 'Country: Venezuela<br>Life Expectancy: 73.747<br>GDP per capita: 11415.80569<br>Population: 26084662.0<br>Year: 2007'],
    marker=dict(
        sizemode='diameter',
        sizeref=0.85,
        size=[21.94976988499517, 10.441052822396196, 47.66021903725089,
              19.979112486875845, 13.95267548575408, 22.993945975228556,
              7.029852430522167, 11.682689085146487, 10.555193870118702,
              12.823544926991564, 9.108293955789053, 12.259853478972317,
              10.082039742103595, 9.458604761285072, 5.765006135966166,
              36.048202790993614, 8.23689670992972, 6.22565654446431,
              8.927648460491556, 18.514711052673302, 6.865187781408511,
              3.5540539239313094, 60, 6.41976234423909, 17.658738378883186],
        line=dict(
            width=2
        ),
    )
)
trace2 = Scatter(
    x=[974.5803384, 29796.048339999998, 1391.253792, 1713.7786859999999,
       4959.1148539999995, 39724.97867, 2452.210407, 3540.6515640000002,
       11605.71449, 4471.061906, 25523.2771, 31656.06806, 4519.461171,
       1593.06548, 23348.139730000003, 10461.05868, 12451.6558,
       3095.7722710000003, 944, 1091.359778, 22316.19287, 2605.94758,
       3190.481016, 21654.83194, 47143.179639999995, 3970.0954070000003,
       4184.548089, 28718.27684, 7458.3963269999995, 2441.576404, 3025.349798,
       2280.769906],
    y=[43.828, 75.635, 64.062, 59.723, 72.961, 82.208, 64.69800000000001,
       70.65, 70.964, 59.545, 80.745, 82.603, 72.535, 67.297, 78.623, 71.993,
       74.241, 66.803, 62.068999999999996, 63.785, 75.64, 65.483, 71.688,
       72.777, 79.972, 72.396, 74.143, 78.4, 70.616, 74.249, 73.422, 62.698],
    mode='markers',
    name='Asia',
    text=['Country: Afghanistan<br>Life Expectancy: 43.828<br>GDP per capita: 974.5803384<br>Population: 31889923.0<br>Year: 2007', 'Country: Bahrain<br>Life Expectancy: 75.635<br>GDP per capita: 29796.04834<br>Population: 708573.0<br>Year: 2007', 'Country: Bangladesh<br>Life Expectancy: 64.062<br>GDP per capita: 1391.253792<br>Population: 150448339.0<br>Year: 2007', 'Country: Cambodia<br>Life Expectancy: 59.723<br>GDP per capita: 1713.778686<br>Population: 14131858.0<br>Year: 2007', 'Country: China<br>Life Expectancy: 72.961<br>GDP per capita: 4959.114854<br>Population: 1318683096.0<br>Year: 2007', 'Country: Hong Kong, China<br>Life Expectancy: 82.208<br>GDP per capita: 39724.97867<br>Population: 6980412.0<br>Year: 2007', 'Country: India<br>Life Expectancy: 64.698<br>GDP per capita: 2452.210407<br>Population: 1110396331.0<br>Year: 2007', 'Country: Indonesia<br>Life Expectancy: 70.65<br>GDP per capita: 3540.651564<br>Population: 223547000.0<br>Year: 2007', 'Country: Iran<br>Life Expectancy: 70.964<br>GDP per capita: 11605.71449<br>Population: 69453570.0<br>Year: 2007', 'Country: Iraq<br>Life Expectancy: 59.545<br>GDP per capita: 4471.061906<br>Population: 27499638.0<br>Year: 2007', 'Country: Israel<br>Life Expectancy: 80.745<br>GDP per capita: 25523.2771<br>Population: 6426679.0<br>Year: 2007', 'Country: Japan<br>Life Expectancy: 82.603<br>GDP per capita: 31656.06806<br>Population: 127467972.0<br>Year: 2007', 'Country: Jordan<br>Life Expectancy: 72.535<br>GDP per capita: 4519.461171<br>Population: 6053193.0<br>Year: 2007', 'Country: Korea, Dem. Rep.<br>Life Expectancy: 67.297<br>GDP per capita: 1593.06548<br>Population: 23301725.0<br>Year: 2007', 'Country: Korea, Rep.<br>Life Expectancy: 78.623<br>GDP per capita: 23348.13973<br>Population: 49044790.0<br>Year: 2007', 'Country: Lebanon<br>Life Expectancy: 71.993<br>GDP per capita: 10461.05868<br>Population: 3921278.0<br>Year: 2007', 'Country: Malaysia<br>Life Expectancy: 74.241<br>GDP per capita: 12451.6558<br>Population: 24821286.0<br>Year: 2007', 'Country: Mongolia<br>Life Expectancy: 66.803<br>GDP per capita: 3095.772271<br>Population: 2874127.0<br>Year: 2007', 'Country: Myanmar<br>Life Expectancy: 62.069<br>GDP per capita: 944.0<br>Population: 47761980.0<br>Year: 2007', 'Country: Nepal<br>Life Expectancy: 63.785<br>GDP per capita: 1091.359778<br>Population: 28901790.0<br>Year: 2007', 'Country: Oman<br>Life Expectancy: 75.64<br>GDP per capita: 22316.19287<br>Population: 3204897.0<br>Year: 2007', 'Country: Pakistan<br>Life Expectancy: 65.483<br>GDP per capita: 2605.94758<br>Population: 169270617.0<br>Year: 2007', 'Country: Philippines<br>Life Expectancy: 71.688<br>GDP per capita: 3190.481016<br>Population: 91077287.0<br>Year: 2007', 'Country: Saudi Arabia<br>Life Expectancy: 72.777<br>GDP per capita: 21654.83194<br>Population: 27601038.0<br>Year: 2007', 'Country: Singapore<br>Life Expectancy: 79.972<br>GDP per capita: 47143.17964<br>Population: 4553009.0<br>Year: 2007', 'Country: Sri Lanka<br>Life Expectancy: 72.396<br>GDP per capita: 3970.095407<br>Population: 20378239.0<br>Year: 2007', 'Country: Syria<br>Life Expectancy: 74.143<br>GDP per capita: 4184.548089<br>Population: 19314747.0<br>Year: 2007', 'Country: Taiwan<br>Life Expectancy: 78.4<br>GDP per capita: 28718.27684<br>Population: 23174294.0<br>Year: 2007', 'Country: Thailand<br>Life Expectancy: 70.616<br>GDP per capita: 7458.396327<br>Population: 65068149.0<br>Year: 2007', 'Country: Vietnam<br>Life Expectancy: 74.249<br>GDP per capita: 2441.576404<br>Population: 85262356.0<br>Year: 2007', 'Country: West Bank and Gaza<br>Life Expectancy: 73.422<br>GDP per capita: 3025.349798<br>Population: 4018332.0<br>Year: 2007', 'Country: Yemen, Rep.<br>Life Expectancy: 62.698<br>GDP per capita: 2280.769906<br>Population: 22211743.0<br>Year: 2007'],
    marker=dict(
        sizemode='diameter',
        sizeref=0.85,
        size=[9.330561207739747, 1.390827697025556, 20.266312242166443,
              6.211273648937339, 60, 4.3653750211924, 55.05795036085951,
              24.703896200017994, 13.769821732555231, 8.664520214956125,
              4.188652530719761, 18.654412200415056, 4.0651192623762835,
              7.975814912067495, 11.57117523159306, 3.271861016562374,
              8.231768913808876, 2.8011347940934943, 11.418845373343052,
              8.882667412223675, 2.9579312056937046, 21.49670117903256,
              15.768343552577761, 8.680479951148044, 3.525577657243318,
              7.4587209016354095, 7.261486641287726, 7.95397619750268,
              13.3280083790662, 15.256667990032932, 3.312103798885452,
              7.787039017632765],
        line=dict(
            width=2
        ),
    )
)
trace3 = Scatter(
    x=[5937.029525999999, 36126.4927, 33692.60508, 7446.298803, 10680.79282,
       14619.222719999998, 22833.30851, 35278.41874, 33207.0844, 30470.0167,
       32170.37442, 27538.41188, 18008.94444, 36180.789189999996, 40675.99635,
       28569.7197, 9253.896111, 36797.93332, 49357.19017, 15389.924680000002,
       20509.64777, 10808.47561, 9786.534714, 18678.31435, 25768.25759,
       28821.0637, 33859.74835, 37506.419069999996, 8458.276384, 33203.26128],
    y=[76.423, 79.829, 79.441, 74.852, 73.005, 75.748, 76.486, 78.332, 79.313,
       80.657, 79.406, 79.483, 73.33800000000001, 81.757, 78.885, 80.546,
       74.543, 79.762, 80.196, 75.563, 78.098, 72.476, 74.002, 74.663, 77.926,
       80.941, 80.884, 81.70100000000001, 71.777, 79.425],
    mode='markers',
    name='Europe',
    text=['Country: Albania<br>Life Expectancy: 76.423<br>GDP per capita: 5937.029526<br>Population: 3600523.0<br>Year: 2007', 'Country: Austria<br>Life Expectancy: 79.829<br>GDP per capita: 36126.4927<br>Population: 8199783.0<br>Year: 2007', 'Country: Belgium<br>Life Expectancy: 79.441<br>GDP per capita: 33692.60508<br>Population: 10392226.0<br>Year: 2007', 'Country: Bosnia and Herzegovina<br>Life Expectancy: 74.852<br>GDP per capita: 7446.298803<br>Population: 4552198.0<br>Year: 2007', 'Country: Bulgaria<br>Life Expectancy: 73.005<br>GDP per capita: 10680.79282<br>Population: 7322858.0<br>Year: 2007', 'Country: Croatia<br>Life Expectancy: 75.748<br>GDP per capita: 14619.22272<br>Population: 4493312.0<br>Year: 2007', 'Country: Czech Republic<br>Life Expectancy: 76.486<br>GDP per capita: 22833.30851<br>Population: 10228744.0<br>Year: 2007', 'Country: Denmark<br>Life Expectancy: 78.332<br>GDP per capita: 35278.41874<br>Population: 5468120.0<br>Year: 2007', 'Country: Finland<br>Life Expectancy: 79.313<br>GDP per capita: 33207.0844<br>Population: 5238460.0<br>Year: 2007', 'Country: France<br>Life Expectancy: 80.657<br>GDP per capita: 30470.0167<br>Population: 61083916.0<br>Year: 2007', 'Country: Germany<br>Life Expectancy: 79.406<br>GDP per capita: 32170.37442<br>Population: 82400996.0<br>Year: 2007', 'Country: Greece<br>Life Expectancy: 79.483<br>GDP per capita: 27538.41188<br>Population: 10706290.0<br>Year: 2007', 'Country: Hungary<br>Life Expectancy: 73.338<br>GDP per capita: 18008.94444<br>Population: 9956108.0<br>Year: 2007', 'Country: Iceland<br>Life Expectancy: 81.757<br>GDP per capita: 36180.78919<br>Population: 301931.0<br>Year: 2007', 'Country: Ireland<br>Life Expectancy: 78.885<br>GDP per capita: 40675.99635<br>Population: 4109086.0<br>Year: 2007', 'Country: Italy<br>Life Expectancy: 80.546<br>GDP per capita: 28569.7197<br>Population: 58147733.0<br>Year: 2007', 'Country: Montenegro<br>Life Expectancy: 74.543<br>GDP per capita: 9253.896111<br>Population: 684736.0<br>Year: 2007', 'Country: Netherlands<br>Life Expectancy: 79.762<br>GDP per capita: 36797.93332<br>Population: 16570613.0<br>Year: 2007', 'Country: Norway<br>Life Expectancy: 80.196<br>GDP per capita: 49357.19017<br>Population: 4627926.0<br>Year: 2007', 'Country: Poland<br>Life Expectancy: 75.563<br>GDP per capita: 15389.92468<br>Population: 38518241.0<br>Year: 2007', 'Country: Portugal<br>Life Expectancy: 78.098<br>GDP per capita: 20509.64777<br>Population: 10642836.0<br>Year: 2007', 'Country: Romania<br>Life Expectancy: 72.476<br>GDP per capita: 10808.47561<br>Population: 22276056.0<br>Year: 2007', 'Country: Serbia<br>Life Expectancy: 74.002<br>GDP per capita: 9786.534714<br>Population: 10150265.0<br>Year: 2007', 'Country: Slovak Republic<br>Life Expectancy: 74.663<br>GDP per capita: 18678.31435<br>Population: 5447502.0<br>Year: 2007', 'Country: Slovenia<br>Life Expectancy: 77.926<br>GDP per capita: 25768.25759<br>Population: 2009245.0<br>Year: 2007', 'Country: Spain<br>Life Expectancy: 80.941<br>GDP per capita: 28821.0637<br>Population: 40448191.0<br>Year: 2007', 'Country: Sweden<br>Life Expectancy: 80.884<br>GDP per capita: 33859.74835<br>Population: 9031088.0<br>Year: 2007', 'Country: Switzerland<br>Life Expectancy: 81.701<br>GDP per capita: 37506.41907<br>Population: 7554661.0<br>Year: 2007', 'Country: Turkey<br>Life Expectancy: 71.777<br>GDP per capita: 8458.276384<br>Population: 71158647.0<br>Year: 2007', 'Country: United Kingdom<br>Life Expectancy: 79.425<br>GDP per capita: 33203.26128<br>Population: 60776238.0<br>Year: 2007'],
    marker=dict(
        sizemode='diameter',
        sizeref=0.85,
        size=[12.542029402681376, 18.92719251331642, 21.30783431755826,
              14.102483219452576, 17.88649832258261, 14.010973368444008,
              21.139571238812916, 15.456246600674588, 15.128185315496781,
              51.65929267153148, 60, 21.627410389852702, 20.855942428523296,
              3.6319417326760695, 13.398544876923102, 50.40242285907865,
              5.469487077232467, 26.90632025621006, 14.2193001873736,
              41.02213342839891, 21.56322451638816, 31.196377737918432,
              21.058319482558733, 15.427079550618533, 9.369177525034539,
              42.03727650225595, 19.863467167731834, 18.167388787784372,
              55.75693095494465, 51.529025209914586],
        line=dict(
            width=2
        ),
    )
)
trace4 = Scatter(
    x=[34435.367439999995, 25185.00911],
    y=[81.235, 80.204],
    mode='markers',
    name='Oceania',
    text=['Country: Australia<br>Life Expectancy: 81.235<br>GDP per capita: 34435.36744<br>Population: 20434176.0<br>Year: 2007', 'Country: New Zealand<br>Life Expectancy: 80.204<br>GDP per capita: 25185.00911<br>Population: 4115771.0<br>Year: 2007'],
    marker=dict(
        sizemode='diameter',
        sizeref=0.85,
        size=[60, 26.92763965464884],
        line=dict(
            width=2
        ),
    )
)
data = [trace0, trace1, trace2, trace3, trace4]
layout = Layout(
    title='Life Expectancy v. Per Capita GDP, 2007',
    xaxis=dict(
        title='GDP per capita (2000 dollars)',
        gridcolor='rgb(255, 255, 255)',
        range=[2.003297660701705, 5.191505530708712],
        type='log',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    yaxis=dict(
        title='Life Expectancy (years)',
        gridcolor='rgb(255, 255, 255)',
        range=[36.12621671352166, 91.72921793264332],
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
    height=600,
    width=800
)
fig = dict( data=data, layout=layout )
iplot(fig, show_link=False)

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode()

x_data = ['Product<br>Revenue', 'Services<br>Revenue',
          'Total<br>Revenue', 'Fixed<br>Costs',
          'Variable<br>Costs', 'Total<br>Costs', 'Total']
y_data = [400, 660, 660, 590, 400, 400, 340]
text = ['$430K', '$260K', '$690K', '$-120K', '$-200K', '$-320K', '$370K']

# Base
trace0 = Bar(
    x=x_data,
    y=[0, 430, 0, 570, 370, 370, 0],
    marker=dict(
        color='rgba(1,1,1, 0.0)',
    )
)
# Revenue
trace1 = Bar(
    x=x_data,
    y=[430, 260, 690, 0, 0, 0, 0],
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)
# Costs
trace2 = Bar(
    x=x_data,
    y=[0, 0, 0, 120, 200, 320, 0],
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    )
)
# Profit
trace3 = Bar(
    x=x_data,
    y=[0, 0, 0, 0, 0, 0, 370],
    marker=dict(
        color='rgba(50, 171, 96, 0.7)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=2,
        )
    )
)
data = [trace0, trace1, trace2, trace3]
layout = Layout(
    title='Annual Profit- 2015',
    barmode='stack',
    paper_bgcolor='rgba(245, 246, 249, 1)',
    plot_bgcolor='rgba(245, 246, 249, 1)',
    width=800,
    height=600,
    showlegend=False,
)

annotations = []

for i in range(0, 7):
    annotations.append(dict(x=x_data[i], y=y_data[i], text=text[i],
                                  font=dict(family='Arial', size=14,
                                  color='rgba(245, 246, 249, 1)'),
                                  showarrow=False,))
    layout['annotations'] = annotations

fig = dict( data=data, layout=layout )
iplot(fig, show_link=False)

