from astropy.table import Table
from astropy import coordinates as coords
import astropy.units as u
import pickle

paec = Table.read('../tables/paec_7-present.cat',format='ascii.fixed_width')
paec['coords'] = coords.SkyCoord(paec['ra'],paec['dec'],unit=(u.hour, u.deg),frame='icrs')
fp = open('paec.p','wb')
pickle.dump(paec,fp,protocol=pickle.HIGHEST_PROTOCOL)

def propurl(id):
    base="<a href=http://www.stsci.edu/cgi-bin/get-proposal-info?id="
    suffix="&observatory=HST"
    str = "%s%d%s> %d </a>" % (base,id,suffix,id)  
    return str

def dup_urls(targets,propids):
    urls = ["" for i in range(len(targets))]
    for i in range(len(targets)):
        if i in propids:
            for p in propids[i]:
                urls[i] += propurl(p)+" "
            urls[i] = urls[i][:-1]
    return urls

def duplications(targets,paec):
    idxc, idxcatalog, d2d, d3d = targets['coords'].search_around_sky(paec['coords'],200*u.arcsec)
    propids = {}
    for id_targ,id_paec in zip(idxcatalog,idxc):
        if id_targ not in propids:
            propids[id_targ] = [paec['prop'][id_paec]]
        else: 
            propids[id_targ] += [paec['prop'][id_paec]]
    for p in propids.keys():
        propids[p] = list(set(propids[p]))
    urls = dup_urls(targets,propids)
    return urls, propids

data_rows = [('IC10',5.072250,59.303780),
             ('Abell209',22.95901,-13.591956)
            ]
my_catalog = Table(rows=data_rows,names=['name','RA','Dec'])
my_catalog['coords']=coords.SkyCoord(my_catalog['RA'],my_catalog['Dec'],
                                     unit=(u.deg, u.deg),frame='icrs')

fp = open('paec.p','rb')
paec = pickle.load(fp)
fp.close()

urls,propids = duplications(my_catalog,paec)

propids

my_catalog['paec']=urls
ipy_html = my_catalog.show_in_notebook()
ipy_html.data = ipy_html.data.replace('&lt;','<')
ipy_html.data = ipy_html.data.replace('&gt;','>')
ipy_html



