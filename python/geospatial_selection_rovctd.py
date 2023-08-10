db = 'stoqs_rovctd_mb'

from django.contrib.gis.geos import fromstr
from django.contrib.gis.measure import D

mars = fromstr('POINT(-122.18681000 36.71137000)')
near_mars = Measurement.objects.using(db).filter(geom__distance_lt=(mars, D(km=.1)))

mars_dives = Activity.objects.using(db).filter(instantpoint__measurement=near_mars
                                              ).distinct()
print mars_dives.count()

deep_mars_dives = Activity.objects.using(db
                                        ).filter(instantpoint__measurement=near_mars,
                                                 instantpoint__measurement__depth__gt=800
                                                ).distinct()
print deep_mars_dives.count()

get_ipython().run_cell_magic('time', '', "%matplotlib inline\nimport pylab as plt\nfrom mpl_toolkits.basemap import Basemap\n\nm = Basemap(projection='cyl', resolution='l',\n            llcrnrlon=-122.7, llcrnrlat=36.5,\n            urcrnrlon=-121.7, urcrnrlat=37.0)\nm.arcgisimage(server='http://services.arcgisonline.com/ArcGIS', service='Ocean_Basemap')\n\nfor dive in deep_mars_dives:\n    points = Measurement.objects.using(db).filter(instantpoint__activity=dive,\n                                                  instantpoint__measurement__depth__gt=800\n                                                 ).values_list('geom', flat=True)\n    m.scatter(\n        [geom.x for geom in points],\n        [geom.y for geom in points])")

get_ipython().run_cell_magic('time', '', "# A Python dictionary comprehension for all the Parameters and axis labels we want to plot\nparms = {p.name: '{} ({})'.format(p.long_name, p.units) for \n                 p in Parameter.objects.using(db).filter(name__in=\n                        ('t', 's', 'o2', 'sigmat', 'spice', 'light'))}\n\nplt.rcParams['figure.figsize'] = (18.0, 8.0)\nfig, ax = plt.subplots(1, len(parms), sharey=True)\nax[0].invert_yaxis()\nax[0].set_ylabel('Depth (m)')\n\ndive_names = []\nfor dive in deep_mars_dives.order_by('startdate'):\n    dive_names.append(dive.name)\n    # Use select_related() to improve query performance for the depth lookup\n    # Need to also order by time\n    mps = MeasuredParameter.objects.using(db\n                                ).filter(measurement__instantpoint__activity=dive\n                                ).select_related('measurement'\n                                ).order_by('measurement__instantpoint__timevalue')\n    depth = [mp.measurement.depth for mp in mps.filter(parameter__name='t')]\n    for i, (p, label) in enumerate(parms.iteritems()):\n        ax[i].set_xlabel(label)\n        try:\n            ax[i].plot(mps.filter(parameter__name=p).values_list(\n                    'datavalue', flat=True), depth)\n        except ValueError:\n            pass\n\nfrom IPython.display import display, HTML\ndisplay(HTML('<p>All dives at MARS site: ' + ' '.join(dive_names) + '<p>'))")



