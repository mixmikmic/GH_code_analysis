# import the environment
import simpy
env = simpy.Environment()

# define the car class
class Car(object):
    def __init__(self, env):
        self.env = env
        # Start the run process everytime an instance is created.
        # This is particularly important for the interrupttion example below
        # The driver is called by environment, and the car starts running as instantiated
        self.action = env.process(self.run())

    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We yield the process that process() returns
            # to wait for it to finish
            yield self.env.process(self.charge(charge_duration))
            # This could not work
            # yield self.charge(charge_duration)
            # This could work, but it is essentially wrong
            # no interaction enabled by process()
            # yield self.env.timeout(charge_duration)
            
            # The charge process has finished and
            # we can start driving again.
            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)
    
    def charge(self, duration):
        yield self.env.timeout(duration)

car = Car(env)
env.run(until=15)

# import the environment
import simpy
env = simpy.Environment()

# define a driver that can access the car and interrupt its charging
def driver(env, car):
    yield env.timeout(3)
    car.action.interrupt()

class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            # We may get interrupted while charging the battery
            try:
                yield self.env.process(self.charge(charge_duration))
            except simpy.Interrupt:
                # When we received an interrupt, we stop charging and
                # switch to the "driving" state
                print('Was interrupted. Hope, the battery is full enough ...')

            print('Start driving at %d' % self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)

    def charge(self, duration):
        yield self.env.timeout(duration)

car = Car(env)
env.process(driver(env, car))
env.run(until=15)



