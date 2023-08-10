from IPython.display import HTML

input_form = """
<div style="background-color:gainsboro; border:solid black;">
<div><b>Enter the number of items and bidders for the Auction:</b></div>
<div style="display:inline">
Players: <input type="text" id="var_players" value="1">
Items  : <input type="text" id="var_items" value="2">
</div>
<button onclick="gen_table()">Generate Table</button>
</div>
<style>
    table{
        width:100%;
        height:10)%;
    }
    table td{
        padding:5px;
        margin:5px;
        border:1px solid #ccc;
    }
</style>
<div id="box">
<table id="basicTable">
</table>
</div>
<button onclick="rescale()">Scale Values</button>
<button onclick="set_valuations()">Set Values</button>
<button onclick="reset_valuations()">Reset Values</button>
"""



javascript = """
<script type="text/Javascript">
    function gen_table(){
        var players = document.getElementById('var_players').value;
        var items = document.getElementById('var_items').value;
        
        var parent = document.getElementById('box');
        var child = document.getElementById('basicTable');
        parent.removeChild(child);
        mytable = $('<table></table>').attr({ id: "basicTable" });
        var rows = players;
        var cols = items;
        var tr = [];
        var row = $('<tr><td></td></tr>').attr({ class: ["class1"].join(' ') }).appendTo(mytable);
        for (var j = 0; j < cols; j++) {
                $("<td>Item "+j+"</td>").appendTo(row); 
            }
        for (var i = 0; i < rows; i++) {
            var row = $('<tr></tr>').attr({ class: ["class1"].join(' ') }).appendTo(mytable);
            $("<td>Player "+i+"</td>").appendTo(row)
            for (var j = 0; j < cols; j++) {
                $("<td><input type='text' id='val_"+i+"_"+j+"' value='1'></td>").appendTo(row); 
            }

        }
        console.log("TTTTT:"+mytable.html());
        mytable.appendTo("#box"); 
    }

    function set_valuations(){
        var players = document.getElementById('var_players').value;
        var items = document.getElementById('var_items').value;
        var val = [];
        var x;
        for (var i = 0; i < players; i++) {
            val[i] = [];
            for (var j = 0; j < items; j++) {
                x = 'val_'+i+'_'+j;
                val[i][j] = document.getElementById(x).value;
            }
        }
        
        var kernel = IPython.notebook.kernel;
        var cmd1 = "players = " + players;
        var cmd2 = "items = " + items;
        var cmd3 = "my_val = " + val;
        
        console.log("Executing Command: " + cmd1);     
        kernel.execute(cmd1);
        
        console.log("Executing Command: " + cmd2);
        kernel.execute(cmd2);  
        
        console.log("Executing Command: " + cmd3);
        kernel.execute(cmd3); 

    }
    
    function rescale(){
        var players = document.getElementById('var_players').value;
        var items = document.getElementById('var_items').value;
        var val = [];
        var sum = 0;
        var x;
        for (var i = 0; i < players; i++) {
            val[i] = [];
            for (var j = 0; j < items; j++) {
                x = 'val_'+i+'_'+j;
                val[i][j] = parseInt(document.getElementById(x).value);
                sum = sum + val[i][j];  
            }
            for (var j = 0; j < items; j++) {
                val[i][j] = val[i][j]*100/sum;
                x = 'val_'+i+'_'+j;
                document.getElementById(x).value = val[i][j];
            }
            sum = 0
        }        
    }
    function reset_valuations(){
        var players = document.getElementById('var_players').value;
        var items = document.getElementById('var_items').value;
        var val = [];
        var x;
        for (var i = 0; i < players; i++) {
            val[i] = [];
            for (var j = 0; j < items; j++) {
                x = 'val_'+i+'_'+j;
                document.getElementById(x).value = 1;
            }
        }
    
    }
</script>
"""

HTML(input_form + javascript)

# First compute all different one-to-one allocations of items to bidders. 
import numpy as np
import itertools

k = 0
allocs = np.zeros((items, players, 2),dtype=int)
valuations = np.zeros((players, items),dtype=int)
for i in range(0,players):
    for j in range(0,items):
        allocs[j,i] = [j,i]
        valuations[i,j] = my_val[k]
        k += 1

# Identify all possible allocations of items to bidders
########
# Uncomment the following lines to use the default values for items, players and valuations
# items = 4
# players = 3
# allocs = np.array([[[0,0],[0,1],[0,2]],
#                    [[1,0],[1,1],[1,2]],
#                    [[2,0],[2,1],[2,2]],
#                    [[3,0],[3,1],[3,2]],])
# valuations = np.array([[10,20,30,40],
#                        [20,20,20,40],
#                        [40,10,10,40]])
########

x = list(itertools.product(*allocs))
r = len(x)
c = len(x[0])

p = np.zeros((r,items+players+1),dtype=int)

for i in range(0,r):
    for j in range(0,c):
        plid = x[i][j][1]+items # initial columns are items
        p[i,plid] = p[i,plid] + valuations[x[i][j][1], x[i][j][0]]
        p[i,x[i][j][0]] = x[i][j][1]
    p[i,-1] = np.sum(p[i,-players-1:-1])

# Store the data in the form of a Dataframe
import pandas as pd

auctions_df = pd.DataFrame(p)
auctions_df.columns = ['i'+ str(i) for i in range(0,items)] +                       ['v'+ str(i) for i in range(0,players)] +                       ['v_tot']
#auctions_df.columns = ['i1','i2','i3','i4','v1','v2','v3','v_tot']
v_max_without = np.zeros(players,dtype=int)
for i in range(0,players):
    v_max_without[i] = max(auctions_df.v_tot[(auctions_df.filter(regex='i') != i).all(1)])

vcg = auctions_df[auctions_df.v_tot == max(auctions_df.v_tot)].copy().reset_index(drop=True)
for i in range(0,players):
    vcg.loc[:,'p'+str(i)] = v_max_without[i] - (vcg.v_tot - vcg['v'+str(i)])

col_names = list(vcg)
print('Items: ',[s for s in col_names if 'i' in s[0]])
print('Valuation of Player to their allocation: ',[s for s in col_names if 'v' in s[0] and '_' not in s])
print('Payment of Player for their allocation: ',[s for s in col_names if 'p' in s[0]])
HTML(vcg.to_html())

print('VCG Payments:\n')
for i in range(0,players):
    print('VCG Cost for Player ', i, ': ', vcg['v'+str(i)][0] - vcg['p'+str(i)][0])

import numpy as np
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def get_revenue(pnum,players, items, steps, trials, second_price):
    rev = np.zeros((steps,trials))
    thresh = np.zeros(steps)
      
    for i in range(0,steps):
        for j in range(0,trials):
            valuations_scale = np.random.rand(players,items)
            if np.max(valuations_scale) >= thresh[i]:
                if second_largest(valuations_scale) >= thresh[i]:
                    rev[i,j] = second_largest(valuations_scale)
                else:
                    if second_price:
                        rev[i,j] = thresh[i]
                    else:
                        rev[i,j] = np.max(valuations_scale)
            else:
                rev[i,j] = 0
            f_part.value = j + i*trials
            f_full.value = pnum*steps*trials + j + i*trials
            
        if i<steps-1:
            thresh[i+1] = thresh[i] + 1/steps
    
    return rev, thresh

from decimal import *
import matplotlib.pyplot as plt
import math
from ipywidgets import FloatProgress
from IPython.display import display
get_ipython().magic('matplotlib inline')

#players = [2,4,8,10]
players = [2,4,8,10,20,40,80,100,500,1000]
select_flag = [1,0,0,1,0,0,0,1,0,1]
steps = 10
trials = 1000
items = 1
subplotnum = 1
    
print('Set of Players: ',players)
ax = plt.figure(figsize=(15,10))
ax.suptitle('Box plot showing mean and median over 1000 trials')
trend_data = np.zeros((len(players),2,steps))
trend_rev = np.zeros((len(players)))

f_part = FloatProgress(min=0, max=steps*trials,description='Current Progress:')
f_full = FloatProgress(min=0, max=len(players)*steps*trials,description='Overall Progress:')

display(f_part)
display(f_full)

for i in range(0,len(players)):
    
    print('Current Players Set: ', players[i])
    rev, thresh = get_revenue(i,players[i],items,steps,trials,second_price=True)
    #print(str(players[i])," Players\nMedian: ",[float(Decimal("%.2f" % e)) for e in np.median(rev,axis=1)],
    #      "\nMean  : ",[float(Decimal("%.2f" % e)) for e in np.mean(rev,axis=1)] )
    

    n = players[i]
    n_fact = math.factorial(players[i])
    theo_rev_0 = (n-1)/(n+1)

    #theo_thresh = 1/(1+1/(n-1/math.factorial(n-2)))
    theo_thresh = 1/2

    k = (theo_thresh+0.1)*steps


    theo_rev = theo_rev_0 +                 np.power(theo_thresh,n) -                 (2*n/(n+1))*np.power(theo_thresh,n+1)

    #l2, = plt.plot(k,theo_rev, 'og')
    if select_flag[i] == 1:
        l0 = plt.subplot(2,2,subplotnum)
        subplotnum += 1
        l1 = plt.boxplot(np.transpose(rev),notch=True,showmeans=True)
        l2 = plt.plot(k,theo_rev, 'og',label='Theoretical best')
        x = np.zeros((2,len(l1['means'])))
        for j in range(0,len(l1['means'])):
            x[0,j] = l1['means'][j].get_xdata()
            x[1,j] = l1['means'][j].get_ydata()
        plt.plot(x[0],x[1],'g--',label='Mean Line')
        plt.xticks([i for i in range(1,steps+1)],thresh)
        plt.xlabel('Reserve Price')
        plt.ylabel('Seller Revenue')
        plt.title(str(players[i])+' Players - IID')
        handles, labels = l0.get_legend_handles_labels()
        l0.legend(handles[::-1], labels[::-1],loc='lower left')
        
    trend_data[i,0,:] = np.mean(rev,axis=1)
    trend_data[i,1,:] = np.median(rev,axis=1)
    trend_rev[i] = theo_rev
print('\nAll Runs Complete...')
plt.show()

from matplotlib.pyplot import gca

ax = plt.figure(figsize=(15,5))
a0 = plt.subplot(1,2,1)
plt.plot(trend_data[:,0,:])
plt.plot(trend_rev,'r^-.',linewidth=3.0,label='Theoretical best')
plt.xticks([i for i in range(0,len(players))],players)
plt.xlabel('Number of Players')
plt.ylabel('Seller Revenue')
plt.title('Mean Revenue wrt Number of Players')

handles, labels = a0.get_legend_handles_labels()
a2 = plt.legend(thresh,loc='lower right',title='Reserve Price')
a0.legend(handles[::-1], labels[::-1],loc='lower left')
gca().add_artist(a2)

a1 = plt.subplot(1,2,2)
plt.plot(trend_data[:,1,:])
plt.plot(trend_rev,'r^-.',linewidth=3.0,label='Theoretical best')
plt.xticks([i for i in range(0,len(players))],players)
plt.xlabel('Number of Players')
plt.ylabel('Seller Revenue')
plt.title('Median Revenue wrt Number of Players')

a2 = plt.legend(thresh,loc='lower right',title='Reserve Price')
handles, labels = a1.get_legend_handles_labels()
a1.legend(handles[::-1], labels[::-1],loc='lower left')
gca().add_artist(a2)



