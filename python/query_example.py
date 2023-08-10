from extcats import CatalogQuery

# initialize the CatalogQuery object pointing it to an existsing database
mqc_query = CatalogQuery.CatalogQuery(
        cat_name = 'milliquas',           # name of the database
        coll_name = 'srcs',               # name of the collection with the sources
        ra_key = 'ra', dec_key = 'dec',   # name of catalog fields for the coordinates
        dbclient = None)


# specify target position (same format as the 'ra_key' and 
# 'dec_key specified at initilization) and serach radius
target_ra, target_dec, rs = 5.458082, 16.035756, 100.
target_ra, target_dec, rs = 321.6639722, -89.48325, 100.

# the 'raw' method does not require any pre-formatting of the catalog.
# It first selects points within a box of radius 'box_scale' times larger than the
# search radius using $gte and $lte operators, then uses the $where expression
# to compute the angular distance of the sources in the box from the target.
out_raw = mqc_query.findwithin(target_ra, target_dec, rs, method = 'raw', box_scale = 2.5)
if not out_raw is None:
    print ("%d sources found around target position using the 'raw' method."%len(out_raw))

# the '2dsphere' method uses instead the use mongodb searches in 
# spherical geometry using "$geoWithin" and "$centerSphere" operators.
# it requires the catalog documents to have been assigned a geoJSON 
# or 'legacy pair' field of type 'Point' (see insert_example notebook).
out_2dsphere = mqc_query.findwithin(target_ra, target_dec, rs, method = '2dsphere')
if not out_2dsphere is None:
    print ("%d sources found around target position using the '2dsphere' method."%len(out_2dsphere))


# finally, the healpix method can be used to speed up queries using a 
# spatial prepartinioning of the data based on a HEALPix grid. In this 
# case, the sources in the catalog should be assigned a field containing
# the ID of the healpix that contains it.
out_healpix = mqc_query.findwithin(target_ra, target_dec, rs, method = 'healpix')
if not out_healpix is None:
    print ("%d sources found around target position using the 'healpix' method."%len(out_healpix))

out_healpix_square = mqc_query.findwithin(target_ra, target_dec, rs, method = 'healpix', circular = False)
if not out_healpix_square is None:
    print ("%d sources found around target position using the 'healpix' (square) method."%len(out_healpix_square))


# ======================================== #
#    make a plot with the query results    #
# ======================================== #
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

# get a random sample from the catalog
cat_pos=[[o['ra'], o['dec']] for o in 
         mqc_query.src_coll.aggregate([{ '$sample': { 'size': 5000 }}])]
cat_ra, cat_dec = zip(*cat_pos)


fig=plt.figure()
ax=fig.add_subplot(111)#, projection="aitoff")

ax.scatter(cat_ra, cat_dec, label="random sample", c="k", s=50, marker="o", zorder=1)
ax.scatter(out_raw['ra'], out_raw['dec'], label="matches (RAW)", c="r", s=100, marker="+")
ax.scatter(out_2dsphere['ra'], out_2dsphere['dec'], label="matches (2D sphere)", c="b", s=100, marker="x")
ax.scatter(out_healpix['ra'], out_healpix['dec'], label="matches (HEALPix)", c="m", s=100, marker="v")
ax.scatter(
out_healpix_square['ra'], out_healpix_square['dec'], label="matches (HEALPix square)", c="g", s=50, marker="v")

ax.scatter(target_ra, target_dec, label='target', s=200, c='y', marker='*', zorder=0)
ax.set_xlim(target_ra-2, target_ra+2)
ax.set_ylim(target_dec-3, target_dec+3)
ax.legend(loc='best')
fig.show()
    

rawcp, rawcp_dist = mqc_query.findclosest(target_ra, target_dec, rs, method = 'raw')
s2dcp, s2d_dist = mqc_query.findclosest(target_ra, target_dec, rs, method = '2dsphere')
hpcp, hpcp_dist = mqc_query.findclosest(target_ra, target_dec, rs, method = 'healpix')

# here we verify that all the counterparts are actually the same
print ('      Database ID        |   cp-dist ["]')
print ("------------------------------------------")
print (rawcp['_id'], "|", rawcp_dist)
print (s2dcp['_id'], "|", s2d_dist)
print (hpcp['_id'], "|", hpcp_dist)

raw_bool = mqc_query.binaryserach(target_ra, target_dec, rs, method = 'raw')
s2d_bool = mqc_query.binaryserach(target_ra, target_dec, rs, method = '2dsphere')
hp_bool = mqc_query.binaryserach(target_ra, target_dec, rs, method = 'healpix')

# here we verify that all the counterparts are actually the same
print (raw_bool, s2d_bool, hp_bool)

# test te three main types of queries with the healpix method
mqc_query.test_queries(query_type = 'within', method = 'healpix', rs_arcsec = 3, npoints=1e4)

# here we don't seed the rng, to avoid mongo using some cached results
mqc_query.test_queries(query_type = 'within', method = 'healpix', rs_arcsec = 3, npoints=1e4, rnd_seed = None)

mqc_query.test_queries(query_type = 'closest', method = 'healpix', rs_arcsec = 3, npoints=1e4)
mqc_query.test_queries(query_type = 'binary', method = 'healpix', rs_arcsec = 3, npoints=1e4)

# and the other query methods as well (they are much slower, since there are not indexes to support them)
mqc_query.test_queries(query_type = 'closest', method ='raw', rs_arcsec = 3, npoints=10)
mqc_query.test_queries(query_type = 'closest', method ='2dsphere', rs_arcsec = 3, npoints=100)



