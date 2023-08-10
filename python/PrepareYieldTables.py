get_ipython().magic('matplotlib nbagg')
import read_yields as ry
import sygma as s

Zs = [0.02,0.01] #Metallicity         
Ms = [[1,5,15],[1,5,15]] #Initial masses
# 4 models in total, each of them the same number of isotopes!
isos = ['H-1','He-4','C-12']
yields_star = [0.1,0.2,0.01] #Msun
data_star = [yields_star] #provide only yields 
# structure: [Zs[0],Zs[1]] where Z[0] => Ms[0] and Z[1] => Ms[1]
yields_all = [len(Ms[0])*[data_star],len(Ms[1])*[data_star]]    

#define header data, define always all fields.
table_name = 'Yield table'
units='Msun, year'          

#additional necessary data per stellar model
col_attrs = ['Lifetime', 'Mfinal']
data_star =[2e9,0.9] #as specified with Units, yr and Msun
col_attrs_data = [len(Ms[0])*[data_star],len(Ms[1])*[data_star]]   

#data columns, 'Yields' required but other data can be added (in data_star).
data_cols = ['Yields']

ry.write_tables(data=yields_all,data_cols=data_cols,
                Zs=Zs,Ms=Ms,isos=isos,
                col_attrs=col_attrs,col_attrs_data=col_attrs_data,
                units=units,table_name=table_name,filename='isotope_yield_table_mod.txt')

new_table_name='isotope_yield_table_mod.txt'

table=ry.read_nugrid_yields(new_table_name)

#try tetting H-1 you wrote
print table.metallicities
print table.get(Z=0.02,quantity='masses')
print table.get(M=1,Z=0.02,specie='C-12')

#You can try to run SYGMA with with the yield table for a test
s1=s.sygma(iniZ=0.02,table='Teaching/'+new_table_name)

table1=ry.read_nugrid_yields('../yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt')

table2=ry.read_nugrid_yields('../yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12rapid.txt')

print table1.metallicities
print table1.data_cols
Zs = table1.metallicities
for Z in Zs:
    Ms=table1.get(Z=Z,quantity='masses')
    for M in Ms:
        isos = table1.get(M=M,Z=Z,quantity='Isotopes')
        for iso in isos:
            y_iso1 = table1.get(M=M,Z=Z,specie=iso)
            y_iso2 = table2.get(M=M,Z=Z,specie=iso)
            y_mix = (y_iso1+y_iso2)/2.
            #The set function only modifies an existing entry
            table1.set(M=M,Z=Z,specie=iso,value=y_mix)
            

table1.write_table(filename='agb_and_massive_stars_test_merge.txt')



