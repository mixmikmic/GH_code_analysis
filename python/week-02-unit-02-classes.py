class Location:
    pass

kathmandu = Location()

kathmandu

print(kathmandu)

paris = Location()

paris

id(kathmandu)

id(paris)

id(Location)

katmandu = kathmandu

id(katmandu)



katmandu is kathmandu

paris is kathmandu

katmandu == kathmandu

paris == kathmandu



isinstance(kathmandu, Location)

isinstance(paris, Location)

isinstance(kathmandu, dict)

isinstance(kathmandu, object)

isinstance(1, object)

type(kathmandu)



issubclass(Location, object)

issubclass(Location, dict)



class Location:
    # class attributes
    latitude = None
    longitude = None
    
    def __init__(self, lat, long, name):
        print("Init method is called with {}".format(id(self)))
        # instance attributes
        self.latitude = lat
        self.longitude = long
        self.name = name
    
    def get_name(self):
        print("get_name method is called")
        return self.name

kathmandu = Location()

kathmandu = Location(27, 83, 'Kathmandu')

id(kathmandu)

kathmandu.get_name()



Location.latitude

Location.name

kathmandu.name

Location()







