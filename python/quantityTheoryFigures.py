import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Quandl
import wbdata as wb
from scipy import stats
import runProcs
get_ipython().magic('matplotlib inline')

# 0. Preliminaries

# 0.1 general plot settings

font = {'weight' : 'bold',
        'size'   : 15}
axes={'labelweight' : 'bold'}
plt.rc('font', **font)
plt.rc('axes', **axes)
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

# 1. Import country codes and organize

# 1.1 Import country codes and names from the country_codes file from Quandl's WB WDI documentation: https://www.quandl.com/data/WWDI/documentation/documentation
countryCodes = {}

try:
    text_file = open('country_codes', 'r')
    lines = text_file.readlines()
    for line in lines:
        split = line.split('|')
        if len(split)>1:
            if len(split[1])==4:
                countryCodes[split[0]] = split[1][:-1]
except:
    countryCodes = {
 'Afghanistan': 'AFG',
 'Africa': 'AFR',
 'Albania': 'ALB',
 'Algeria': 'DZA',
 'American Samoa': 'ASM',
 'Andorra': 'AND',
 'Angola': 'AGO',
 'Antigua and Barbuda': 'ATG',
 'Arab World': 'ARB',
 'Argentina': 'ARG',
 'Armenia': 'ARM',
 'Aruba': 'ABW',
 'Australia': 'AUS',
 'Austria': 'AUT',
 'Azerbaijan': 'AZE',
 'Bahamas, The': 'BHS',
 'Bahrain': 'BHR',
 'Bangladesh': 'BGD',
 'Barbados': 'BRB',
 'Belarus': 'BLR',
 'Belgium': 'BEL',
 'Belize': 'BLZ',
 'Benin': 'BEN',
 'Bermuda': 'BMU',
 'Bhutan': 'BTN',
 'Bolivia': 'BOL',
 'Bosnia and Herzegovina': 'BIH',
 'Botswana': 'BWA',
 'Brazil': 'BRA',
 'Brunei Darussalam': 'BRN',
 'Bulgaria': 'BGR',
 'Burkina Faso': 'BFA',
 'Burundi': 'BDI',
 'Cabo Verde': 'CPV',
 'Cambodia': 'KHM',
 'Cameroon': 'CMR',
 'Canada': 'CAN',
 'Caribbean small states': 'CSS',
 'Cayman Islands': 'CYM',
 'Central African Republic': 'CAF',
 'Chad': 'TCD',
 'Channel Islands': 'CHI',
 'Chile': 'CHL',
 'China': 'CHN',
 'Colombia': 'COL',
 'Comoros': 'COM',
 'Congo, Dem. Rep.': 'COD',
 'Congo, Rep.': 'COG',
 'Costa Rica': 'CRI',
 "Cote d'Ivoire": 'CIV',
 'Croatia': 'HRV',
 'Cuba': 'CUB',
 'Curacao': 'CUW',
 'Cyprus': 'CYP',
 'Czech Republic': 'CZE',
 'Denmark': 'DNK',
 'Djibouti': 'DJI',
 'Dominica': 'DMA',
 'Dominican Republic': 'DOM',
 'East Asia & Pacific (all income levels)': 'EAS',
 'East Asia & Pacific (developing only)': 'EAP',
 'East Asia and the Pacific (IFC classification)': 'CEA',
 'Ecuador': 'ECU',
 'Egypt, Arab Rep.': 'EGY',
 'El Salvador': 'SLV',
 'Equatorial Guinea': 'GNQ',
 'Eritrea': 'ERI',
 'Estonia': 'EST',
 'Ethiopia': 'ETH',
 'Euro area': 'EMU',
 'Europe & Central Asia (all income levels)': 'ECS',
 'Europe & Central Asia (developing only)': 'ECA',
 'Europe and Central Asia (IFC classification)': 'CEU',
 'European Union': 'EUU',
 'Faeroe Islands': 'FRO',
 'Fiji': 'FJI',
 'Finland': 'FIN',
 'France': 'FRA',
 'French Polynesia': 'PYF',
 'Gabon': 'GAB',
 'Gambia, The': 'GMB',
 'Georgia': 'GEO',
 'Germany': 'DEU',
 'Ghana': 'GHA',
 'Greece': 'GRC',
 'Greenland': 'GRL',
 'Grenada': 'GRD',
 'Guam': 'GUM',
 'Guatemala': 'GTM',
 'Guinea': 'GIN',
 'Guinea-Bissau': 'GNB',
 'Guyana': 'GUY',
 'Haiti': 'HTI',
 'Heavily indebted poor countries (HIPC)': 'HPC',
 'High income': 'HIC',
 'High income: OECD': 'OEC',
 'High income: nonOECD': 'NOC',
 'Honduras': 'HND',
 'Hong Kong SAR, China': 'HKG',
 'Hungary': 'HUN',
 'Iceland': 'ISL',
 'India': 'IND',
 'Indonesia': 'IDN',
 'Iran, Islamic Rep.': 'IRN',
 'Iraq': 'IRQ',
 'Ireland': 'IRL',
 'Isle of Man': 'IMN',
 'Israel': 'ISR',
 'Italy': 'ITA',
 'Jamaica': 'JAM',
 'Japan': 'JPN',
 'Jordan': 'JOR',
 'Kazakhstan': 'KAZ',
 'Kenya': 'KEN',
 'Kiribati': 'KIR',
 'Korea, Dem. Rep.': 'PRK',
 'Korea, Rep.': 'KOR',
 'Kosovo': 'KSV',
 'Kuwait': 'KWT',
 'Kyrgyz Republic': 'KGZ',
 'Lao PDR': 'LAO',
 'Latin America & Caribbean (all income levels)': 'LCN',
 'Latin America & Caribbean (developing only)': 'LAC',
 'Latin America and the Caribbean (IFC classification)': 'CLA',
 'Latvia': 'LVA',
 'Least developed countries: UN classification': 'LDC',
 'Lebanon': 'LBN',
 'Lesotho': 'LSO',
 'Liberia': 'LBR',
 'Libya': 'LBY',
 'Liechtenstein': 'LIE',
 'Lithuania': 'LTU',
 'Low & middle income': 'LMY',
 'Low income': 'LIC',
 'Lower middle income': 'LMC',
 'Luxembourg': 'LUX',
 'Macao SAR, China': 'MAC',
 'Macedonia, FYR': 'MKD',
 'Madagascar': 'MDG',
 'Malawi': 'MWI',
 'Malaysia': 'MYS',
 'Maldives': 'MDV',
 'Mali': 'MLI',
 'Malta': 'MLT',
 'Marshall Islands': 'MHL',
 'Mauritania': 'MRT',
 'Mauritius': 'MUS',
 'Mexico': 'MEX',
 'Micronesia, Fed. Sts.': 'FSM',
 'Middle East & North Africa (all income levels)': 'MEA',
 'Middle East & North Africa (developing only)': 'MNA',
 'Middle East and North Africa (IFC classification)': 'CME',
 'Middle income': 'MIC',
 'Moldova': 'MDA',
 'Monaco': 'MCO',
 'Mongolia': 'MNG',
 'Montenegro': 'MNE',
 'Morocco': 'MAR',
 'Mozambique': 'MOZ',
 'Myanmar': 'MMR',
 'Namibia': 'NAM',
 'Nepal': 'NPL',
 'Netherlands': 'NLD',
 'New Caledonia': 'NCL',
 'New Zealand': 'NZL',
 'Nicaragua': 'NIC',
 'Niger': 'NER',
 'Nigeria': 'NGA',
 'North Africa': 'NAF',
 'North America': 'NAC',
 'Northern Mariana Islands': 'MNP',
 'Norway': 'NOR',
 'OECD members': 'OED',
 'Oman': 'OMN',
 'Other small states': 'OSS',
 'Pacific island small states': 'PSS',
 'Pakistan': 'PAK',
 'Palau': 'PLW',
 'Panama': 'PAN',
 'Papua New Guinea': 'PNG',
 'Paraguay': 'PRY',
 'Peru': 'PER',
 'Philippines': 'PHL',
 'Poland': 'POL',
 'Portugal': 'PRT',
 'Puerto Rico': 'PRI',
 'Qatar': 'QAT',
 'Romania': 'ROU',
 'Russian Federation': 'RUS',
 'Rwanda': 'RWA',
 'Samoa': 'WSM',
 'San Marino': 'SMR',
 'Sao Tome and Principe': 'STP',
 'Saudi Arabia': 'SAU',
 'Senegal': 'SEN',
 'Serbia': 'SRB',
 'Seychelles': 'SYC',
 'Sierra Leone': 'SLE',
 'Singapore': 'SGP',
 'Sint Maarten (Dutch part)': 'SXM',
 'Slovak Republic': 'SVK',
 'Slovenia': 'SVN',
 'Small states': 'SST',
 'Solomon Islands': 'SLB',
 'Somalia': 'SOM',
 'South Africa': 'ZAF',
 'South Asia': 'SAS',
 'South Asia (IFC classification)': 'CSA',
 'South Sudan': 'SSD',
 'Spain': 'ESP',
 'Sri Lanka': 'LKA',
 'St. Kitts and Nevis': 'KNA',
 'St. Lucia': 'LCA',
 'St. Martin (French part)': 'MAF',
 'St. Vincent and the Grenadines': 'VCT',
 'Sub-Saharan Africa (IFC classification)': 'CAA',
 'Sub-Saharan Africa (all income levels)': 'SSF',
 'Sub-Saharan Africa (developing only)': 'SSA',
 'Sub-Saharan Africa excluding South Africa': 'SXZ',
 'Sub-Saharan Africa excluding South Africa and Nigeria': 'XZN',
 'Sudan': 'SDN',
 'Suriname': 'SUR',
 'Swaziland': 'SWZ',
 'Sweden': 'SWE',
 'Switzerland': 'CHE',
 'Syrian Arab Republic': 'SYR',
 'Tajikistan': 'TJK',
 'Tanzania': 'TZA',
 'Thailand': 'THA',
 'Timor-Leste': 'TLS',
 'Togo': 'TGO',
 'Tonga': 'TON',
 'Trinidad and Tobago': 'TTO',
 'Tunisia': 'TUN',
 'Turkey': 'TUR',
 'Turkmenistan': 'TKM',
 'Turks and Caicos Islands': 'TCA',
 'Tuvalu': 'TUV',
 'Uganda': 'UGA',
 'Ukraine': 'UKR',
 'United Arab Emirates': 'ARE',
 'United Kingdom': 'GBR',
 'United States': 'USA',
 'Upper middle income': 'UMC',
 'Uruguay': 'URY',
 'Uzbekistan': 'UZB',
 'Vanuatu': 'VUT',
 'Venezuela, RB': 'VEN',
 'Vietnam': 'VNM',
 'Virgin Islands (U.S.)': 'VIR',
 'West Bank and Gaza': 'PSE',
 'World': 'WLD',
 'Yemen, Rep.': 'YEM',
 'Zambia': 'ZMB',
 'Zimbabwe': 'ZWE'}
    
        
#1.2 Use wbdata to get lists of country codes by income groups
countriesIncomeAll =  [i['id'] for i in wb.get_country(incomelevel=['LIC','MIC','HIC'],display=False)]
countriesIncomeH = [i['id'] for i in wb.get_country(incomelevel=['HIC'],display=False)]
countriesIncomeM = [i['id'] for i in wb.get_country(incomelevel=['MIC'],display=False)]
countriesIncomeL = [i['id'] for i in wb.get_country(incomelevel=['LIC'],display=False)]
countriesIncomeOecd = [i['id'] for i in wb.get_country(incomelevel="OEC", display=False)]

# 4. Graph of quantity theory data without exchange or interest rates

# 4.0 Import data

qtyTheoryData = pd.read_csv('qtyTheoryData.csv')
qtyTheoryDataL = pd.read_csv('qtyTheoryDataL.csv')
qtyTheoryDataM = pd.read_csv('qtyTheoryDataM.csv')
qtyTheoryDataH = pd.read_csv('qtyTheoryDataH.csv')
qtyTheoryDataOecd = pd.read_csv('qtyTheoryDataOecd.csv')


# 4.1 Money growth inflation with country codes
m = qtyTheoryData['money growth']
p = qtyTheoryData['inflation']
y = qtyTheoryData['gdp growth']
codes = qtyTheoryData['iso code']


a_p, b_p, r_p, p_value, std_err = stats.linregress(m,p)
xVals = np.arange(-10,10)
p_pred = b_p + a_p*xVals

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(p_pred,p_pred,lw=1.5,label='OLS Regression')
ax1.legend(loc='lower right',fontsize=12)
for i in range(len(codes)):
    if codes[i] =='USA':
        usaIndex = i
#         plt.text(m[i], p[i], codes[i], color='red',fontsize=12, clip_on=True,horizontalalignment='center',verticalalignment='center',alpha = 1)
    else:
        plt.text(m[i], p[i], codes[i], color="#11557c",fontsize=12, clip_on=True,horizontalalignment='center',verticalalignment='center',alpha = 1)
plt.text(m[usaIndex], p[usaIndex], codes[usaIndex], color='red',fontsize=12, clip_on=True,horizontalalignment='center',verticalalignment='center',alpha = 1)
ax1.axis([-0.2, 1.4, -0.2, 1.4])
ax1.set_xlabel('money growth')
ax1.set_ylabel('inflation')
plt.grid()

plt.tight_layout()
plt.savefig('fig_moneyInflationCoded.png',bbox_inches='tight',dpi=120)


print('regression coefficient:',np.round(a_p,2))

# 4.2 Money growth and inflation

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = 1.4

x45 = np.arange(xmin,xmax,0.001)
y45 = x45 - np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x45,y45)
ax1.plot(qtyTheoryDataH['money growth'],qtyTheoryDataH['inflation'],'bo')
ax1.plot(qtyTheoryDataM['money growth'],qtyTheoryDataM['inflation'],'gs')
ax1.plot(qtyTheoryDataL['money growth'],qtyTheoryDataL['inflation'],'r^')
plt.grid(True)
ax1.set_xlabel('money growth')
ax1.set_ylabel('inflation')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])

plt.legend(['$45^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_moneyInflation.png',bbox_inches='tight',dpi=120)

# 4.3 Money growth and real gdp growth

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = .2

x0 = np.arange(xmin,xmax,0.001)
y0 = x0*0 + np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x0,y0)
plt.plot(qtyTheoryDataH['money growth'],qtyTheoryDataH['gdp growth'],'bo')
plt.plot(qtyTheoryDataM['money growth'],qtyTheoryDataM['gdp growth'],'gs')
plt.plot(qtyTheoryDataL['money growth'],qtyTheoryDataL['gdp growth'],'r^')
plt.grid(True)
# ax1.set_title('Money growth and real GDP growth',fontsize=15)
ax1.set_xlabel('money growth')
ax1.set_ylabel('real GDP growth')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])
plt.legend(['$0^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_moneyGDP.png',bbox_inches='tight',dpi=120)

# 4.4 Money growth and real gdp growth

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = .2

x0 = np.arange(xmin,xmax,0.001)
y0 = x0*0 + np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x0,y0)
plt.plot(qtyTheoryDataH['inflation'],qtyTheoryDataH['gdp growth'],'bo')
plt.plot(qtyTheoryDataM['inflation'],qtyTheoryDataM['gdp growth'],'gs')
plt.plot(qtyTheoryDataL['inflation'],qtyTheoryDataL['gdp growth'],'r^')
plt.grid(True)
# ax1.set_title('Inflation and real GDP growth',fontsize=15)
ax1.set_xlabel('inflation')
ax1.set_ylabel('real GDP growth')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])
plt.legend(['$0^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_inflationGDP.png',bbox_inches='tight',dpi=120)

# 5. Make tex files for figures

# 5.0 Money growth and inflation with codes
nf = open('figure_moneyInflationCoded.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyInflationCoded} \\textbf{Money growth and deflator inflation for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyInflationCoded.png}\n')
nf.write('\\end{figure}')
nf.close()

# 5.1 Money growth and inflation
nf = open('figure_moneyInflation.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyInflation} \\textbf{Money growth and deflator inflation for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyInflation.png}\n')
nf.write('\\end{figure}')
nf.close()

# 5.2 Inflation and real GDP
nf = open('figure_inflationGDP.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyGDP} \\textbf{Deflator inflation and real GDP growth for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_inflationGDP.png}\n')
nf.write('\\end{figure}')
nf.close()

# 5.3 Inflation and real GDP
nf = open('figure_moneyGDP.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_inflationGDP} \\textbf{Money growth and real GDP growth for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyGDP.png}\n')
nf.write('\\end{figure}')
nf.close()

# 6. Correlations
print(qtyTheoryData[['money growth','inflation','gdp growth']].corr())
print(qtyTheoryDataH[['money growth','inflation','gdp growth']].corr())
print(qtyTheoryDataM[['money growth','inflation','gdp growth']].corr())
print(qtyTheoryDataL[['money growth','inflation','gdp growth']].corr())
print(qtyTheoryDataOecd[['money growth','inflation','gdp growth']].corr())

qtyTheoryData = pd.read_csv('qtyTheoryOpenData.csv',index_col=0)
qtyTheoryDataL = pd.read_csv('qtyTheoryOpenDataL.csv',index_col=0)
qtyTheoryDataM = pd.read_csv('qtyTheoryOpenDataM.csv',index_col=0)
qtyTheoryDataH = pd.read_csv('qtyTheoryOpenDataH.csv',index_col=0)
qtyTheoryDataOecd = pd.read_csv('qtyTheoryOpenDataOecd.csv',index_col=0)


# 8.1 Money growth and inflation

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = 1.4

x45 = np.arange(xmin,xmax,0.001)
y45 = x45 - np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x45,y45)
ax1.plot(qtyTheoryDataH['money growth'],qtyTheoryDataH['inflation'],'bo')
ax1.plot(qtyTheoryDataM['money growth'],qtyTheoryDataM['inflation'],'gs')
ax1.plot(qtyTheoryDataL['money growth'],qtyTheoryDataL['inflation'],'r^')
plt.grid(True)
ax1.set_xlabel('money growth')
ax1.set_ylabel('inflation')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])

plt.legend(['$45^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_moneyInflationOpen.png',bbox_inches='tight',dpi=120)

# 8.2 Money growth and real gdp growth

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = .2

x0 = np.arange(xmin,xmax,0.001)
y0 = x0*0 + np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x0,y0)
plt.plot(qtyTheoryDataH['money growth'],qtyTheoryDataH['gdp growth'],'bo')
plt.plot(qtyTheoryDataM['money growth'],qtyTheoryDataM['gdp growth'],'gs')
plt.plot(qtyTheoryDataL['money growth'],qtyTheoryDataL['gdp growth'],'r^')
plt.grid(True)
# ax1.set_title('Money growth and real GDP growth',fontsize=15)
ax1.set_xlabel('money growth')
ax1.set_ylabel('real GDP growth')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])
plt.legend(['$0^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_moneyGDPOpen.png',bbox_inches='tight',dpi=120)

# 8.3 Money growth and real gdp growth

xmin = -0.05
xmax = 1.4
ymin = -0.05
ymax = .2

x0 = np.arange(xmin,xmax,0.001)
y0 = x0*0 + np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x0,y0)
plt.plot(qtyTheoryDataH['inflation'],qtyTheoryDataH['gdp growth'],'bo')
plt.plot(qtyTheoryDataM['inflation'],qtyTheoryDataM['gdp growth'],'gs')
plt.plot(qtyTheoryDataL['inflation'],qtyTheoryDataL['gdp growth'],'r^')
plt.grid(True)
# ax1.set_title('Inflation and real GDP growth',fontsize=15)
ax1.set_xlabel('inflation')
ax1.set_ylabel('real GDP growth')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])
plt.legend(['$0^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_inflationGDPOpen.png',bbox_inches='tight',dpi=120)

# 8.4 Money growth differential depreciation

xmin = -0.2
xmax = 1.4
ymin = -0.2
ymax = 1.4

x45 = np.arange(xmin,xmax,0.001)
y45 = x45 - np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x45,y45)
ax1.plot(qtyTheoryDataH['inflation']- qtyTheoryData['inflation'].loc['United States'],qtyTheoryDataH['nominal interest rate']- qtyTheoryData['nominal interest rate'].loc['United States'],'bo')
ax1.plot(qtyTheoryDataM['inflation']- qtyTheoryData['inflation'].loc['United States'],qtyTheoryDataM['nominal interest rate']- qtyTheoryData['nominal interest rate'].loc['United States'],'gs')
ax1.plot(qtyTheoryDataL['inflation']- qtyTheoryData['inflation'].loc['United States'],qtyTheoryDataL['nominal interest rate']- qtyTheoryData['nominal interest rate'].loc['United States'],'r^')
plt.grid(True)
ax1.set_xlabel('inflation differential')
ax1.set_ylabel('interest differential')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])

plt.legend(['$45^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_inflationInterestDifferentialsOpen.png',bbox_inches='tight',dpi=120)

# 8.5 Money growth differential depreciation

xmin = -0.2
xmax = 1.4
ymin = -0.2
ymax = 1.4

x45 = np.arange(xmin,xmax,0.001)
y45 = x45 - np.mean(qtyTheoryData['gdp growth'])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x45,y45)
ax1.plot(qtyTheoryDataH['money growth']- qtyTheoryData['money growth'].loc['United States'],qtyTheoryDataH['exchange rate depreciation'],'bo')
ax1.plot(qtyTheoryDataM['money growth']- qtyTheoryData['money growth'].loc['United States'],qtyTheoryDataM['exchange rate depreciation'],'gs')
ax1.plot(qtyTheoryDataL['money growth']- qtyTheoryData['money growth'].loc['United States'],qtyTheoryDataL['exchange rate depreciation'],'r^')
plt.grid(True)
ax1.set_xlabel('money growth differential')
ax1.set_ylabel('depreciation')
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])

plt.legend(['$45^\circ$'],loc='lower right',fontsize='15')

plt.tight_layout()
plt.savefig('fig_moneyDifferentialDepreciationOpen.png',bbox_inches='tight',dpi=120)

# 9. Make tex files for figures

# 9.1 Money growth and inflation
nf = open('figure_moneyInflationOpen.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyInflationOpen} \\textbf{Money growth and deflator inflation for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyInflationOpen.png}\n')
nf.write('\\end{figure}')
nf.close()

# 9.2 Inflation and real GDP
nf = open('figure_inflationGDPOpen.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyGDPOpen} \\textbf{Deflator inflation and real GDP growth for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_inflationGDPOpen.png}\n')
nf.write('\\end{figure}')
nf.close()

# 9.3 Inflation and real GDP
nf = open('figure_moneyGDPOpen.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_inflationGDPOpen} \\textbf{Money growth and real GDP growth for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyGDPOpen.png}\n')
nf.write('\\end{figure}')
nf.close()

# 9.4 Inflation and interest rate differentials
nf = open('figure_inflationInterestDifferentialsOpen.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_inflationInterestDifferentialsOpen} \\textbf{Interest and inflation relative to US for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_inflationInterestDifferentialsOpen.png}\n')
nf.write('\\end{figure}')
nf.close()

# 9.5 Money growth and exchange rate depreciation
nf = open('figure_moneyDifferentialDepreciationOpen.tex', 'w')
nf.write('\\begin{figure}[h]\n')
nf.write('\\caption{\\label{fig_moneyDifferentialDepreciationOpen} \\textbf{Money growth and depreciation for '),nf.write(str(len(qtyTheoryData))),nf.write(' countries.} High-income countries: blue circles, medium-income: green squares, and low-income: red triangles. {\\tiny Source: Quandl, World Development Indicators, World Bank}}\n')
nf.write('\\hspace*{-.5cm}\\includegraphics[height = 7.cm]{fig_moneyDifferentialDepreciationOpen.png}\n')
nf.write('\\end{figure}')
nf.close()

# 10. Correlations
print(qtyTheoryData[['money growth','inflation','gdp growth','nominal interest rate','exchange rate depreciation']].corr())
print(qtyTheoryDataH[['money growth','inflation','gdp growth','nominal interest rate','exchange rate depreciation']].corr())
print(qtyTheoryDataM[['money growth','inflation','gdp growth','nominal interest rate','exchange rate depreciation']].corr())
print(qtyTheoryDataL[['money growth','inflation','gdp growth','nominal interest rate','exchange rate depreciation']].corr())
print(qtyTheoryDataOecd[['money growth','inflation','gdp growth','nominal interest rate','exchange rate depreciation']].corr())

# 11. Export notebook to python script
runProcs.exportNb('quantityTheoryFigures')

