acts = (Activity.objects.using('stoqs_cce2015')
        .filter(name__contains='trajectory')
        .order_by('name'))

fmt = '\t{}: {}, {:.6f}, {:.6f}, {:.2f}'
for activity in acts:
    measuredparameters = (MeasuredParameter.objects.using('stoqs_cce2015')
                           .filter(measurement__instantpoint__activity=activity)
                           .order_by('measurement__instantpoint__timevalue'))
    start = measuredparameters.earliest('measurement__instantpoint__timevalue')
    end = measuredparameters.latest('measurement__instantpoint__timevalue')
    
    print('{}'.format(activity))
    print(fmt.format('Start',
                     start.measurement.instantpoint, 
                     start.measurement.geom.x, 
                     start.measurement.geom.y, 
                     start.measurement.depth))
    print(fmt.format('End  ',
                     end.measurement.instantpoint, 
                     end.measurement.geom.x, 
                     end.measurement.geom.y, 
                     end.measurement.depth))

from coards import from_udunits
print(str(from_udunits(95896909, 'seconds since 2013-01-01 00:00:00')))
print(str(from_udunits(95896956.0000112, 'seconds since 2013-01-01 00:00:00')))

