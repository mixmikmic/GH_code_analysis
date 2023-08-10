from gpcharts import figure

fig1 = figure()
fig1.plot([8,7,6,5,4])

fig2 = figure(title='Two lines',xlabel='Days',ylabel='Count',height=600,width=600)
xVals = ['Mon','Tues','Wed','Thurs','Fri']
yVals = [[5,4],[8,7],[4,8],[10,10],[3,12]]
fig2.plot(xVals,yVals)

fig3 = figure()
fig3.title = 'Weather over Days'
fig3.ylabel = 'Temperature'
#modify size of graph
fig3.height = 800
fig3.width = 1000

#xVals = ['Dates','2016-03-20 00:00:00','2016-03-21 00:00:00','2016-03-25 00:00:00','2016-04-01 00:00:00']
xVals = ['Dates','2016-03-20','2016-03-21','2016-03-25','2016-04-01']
yVals = [['Shakuras','Korhal','Aiur'],[10,30,40],[12,28,41],[15,34,38],[8,33,47]]
fig3.plot(xVals,yVals)

fig4 = figure(title='Population Growth',ylabel='Population')
xVals = ['Year',1700,1800,1900,2000]
yVals = [['Gotham City', 'Central City'],[0,10],[100,200],[100000,500000],[5000000,10000000]]
fig4.plot(xVals,yVals,logScale=True)

fig5 = figure('Strong Correlation')
fig5.scatter([1,2,3,4,5],[[1,5],[2,4],[3,3],[4,2],[5,1]],trendline=True)

fig6 = figure('Percent Alcohol Consumption')
fig6.bar(['Percentage','Beer','Wine','Liquor'],['Type',40,50,10])

fig7 = figure('Distribution',xlabel='value')
fig7.hist([1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,4,4,5,6,7,8,8,8,8,8,9,9,9,10,11,12,13,13,13,13,14])

