from collections import namedtuple

Vehicle = namedtuple('Vehicle', 'make model wheels manual')

forrester = Vehicle('Forrester', 'Subaru', 4, True)

forrester.model

forrester.wheels

