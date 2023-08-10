import random
import numpy as np
from matplotlib import pyplot as plt

string_list = ['the ', 'quick ', 'brown ', 'fox ', 'jumped ', 'over ', 'the ', 'lazy ', 'dog']

get_ipython().run_cell_magic('timeit', '', 'output = ""\nfor string in string_list:\n    output+=string')

get_ipython().run_cell_magic('timeit', '', '"".join(string_list)')

get_ipython().run_cell_magic('timeit', '', 'output = ""\nfor word in string_list:\n    output = "%s %s" % (output, word)')

get_ipython().run_cell_magic('timeit', '', 'output = []\nfor x in range(1000000): output.append(x**2)')

get_ipython().run_cell_magic('timeit', '', '[x**2 for x in range(1000000)]')

get_ipython().run_cell_magic('timeit', '', '[x**2 for x in np.arange(1000000)]')

get_ipython().run_cell_magic('timeit', '', 'np.arange(1000000)**2')

get_ipython().run_cell_magic('timeit', '', 'map(lambda x: x**2, np.arange(1000000))')

# Sort version1

import random

def create_random_list(n_elements):
    """
    Create a list made up of random elements in random order
    """
    random_list = []
    for i in range(n_elements):
        random_list.append(random.random())
    return random_list

def find_minimum_index(random_list):
    """
    Find the index of the minimum value in the list
    """
    # current minimum
    min_value = 1
    i = 0

    # Find minimum in list
    for element in random_list:
        if element < min_value:
            min_value = element

    # Find value that matches minimum
    for element in random_list:
        if element == min_value:
            return i
        i += 1

def sort_list(random_list):
    """
    Sort a list into ascending order
    """
    output_list = []
    for _ in range(len(random_list)):
        i = find_minimum_index(random_list)
        minimum = random_list[i]
        output_list.append(minimum)
        del random_list[i]
    return output_list

if __name__ == '__main__':
    l = create_random_list(10000)
    o = sort_list(l)

from matplotlib import pyplot as plt
import numpy as np
import random

class Galaxy():
    """
    Galaxy class for simply representing a galaxy.
    """
    def __init__(self, total_mass, cold_gas_mass, stellar_mass, age=0):
        self.total_mass = total_mass
        self.cold_gas_mass = cold_gas_mass
        self.stellar_mass = stellar_mass
        self.age = age
        self.SFR = 0
        self.color = 'red'
        
    def __repr__(self):
        return "Galaxy (m_total = %.1g; m_cold = %.1g; m_stars = %.1g; age = %.1g; SFR = %0.2f)" %                 (self.total_mass, self.cold_gas_mass, self.stellar_mass, self.age, self.SFR)
        
class EvolvingGalaxy(Galaxy):
    """
    Galaxy class for representing a galaxy that can evolve over time.
    """
    def current_state(self):
        """
        Return a tuple of the galaxy's total_mass, cold_gas_mass, stellar_mass, age, and SFR
        """
        return (self.total_mass, self.cold_gas_mass, self.stellar_mass, self.age, self.SFR)
    
    def calculate_star_formation_rate(self):
        """
        Calculate the star formation rate by taking a random number between 0 and 1 
        normalized by the galaxy total mass / 1e12; 
        
        Also updates the galaxy's color to blue if SFR > 0.01, otherwise color = red
        """
        self.SFR = random.random() * (self.total_mass / 1e12)
        if self.SFR > 0.01: 
            self.color = 'blue'
        else:
            self.color = 'red'
            
    def accrete_gas_from_IGM(self, time):
        """
        Allow the galaxy to accrete cold gas from the IGM at a variable rate normalized to
        the galaxy's mass
        """
        cold_gas_accreted = random.random() * 0.1 * time * (self.total_mass / 1e12)
        self.cold_gas_mass += cold_gas_accreted
        self.total_mass += cold_gas_accreted
        
    def form_stars(self, time):
        """
        Form stars according to the current star formation rate and time available
        If unable cold gas, then shut off star formation
        """
        if self.cold_gas_mass > self.SFR * time:
            self.cold_gas_mass -= self.SFR * time
            self.stellar_mass += self.SFR * time
        else:
            self.SFR = 0
            self.color = 'red'
            
    def evolve(self, time):
        """
        Evolve this galaxy forward for a period time
        """
        if random.random() < 0.01:
            self.calculate_star_formation_rate()
        self.accrete_gas_from_IGM(time)
        self.form_stars(time)
        self.age += time       
        
class MovingGalaxy(EvolvingGalaxy):
    """
    This galaxy can move over time in the x,y plane
    """
    def __init__(self, total_mass, cold_gas_mass, stellar_mass, x_position, y_position, x_velocity, y_velocity, idnum, age=0):
        
        # Replace self with super to activate the superclass's methods
        super().__init__(total_mass, cold_gas_mass, stellar_mass)
        
        self.x_position = x_position
        self.y_position = y_position
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
        self.idnum = idnum
        
    def __repr__(self):
        return "Galaxy %i (x = %.0f; y = %.0f)" % (self.idnum, self.x_position, self.y_position)
        
    def move(self, time):
        """
        """
        self.x_position += self.x_velocity * time
        self.y_position += self.y_velocity * time
        
    def calculate_momentum(self):
        return (self.total_mass * self.x_velocity, self.total_mass * self.y_velocity)

    def evolve(self, time):
        self.move(time)
        super().evolve(time)
        
def distance(galaxy1, galaxy2):
    x_diff = galaxy1.x_position - galaxy2.x_position
    y_diff = galaxy1.y_position - galaxy2.y_position
    return (x_diff**2 + y_diff**2)**0.5

class Universe():
    """
    """
    def __init__(self):
        self.xrange = (0,100)
        self.yrange = (0,100)
        self.galaxies = []
        self.added_galaxies = []
        self.removed_galaxies = []
        self.time = 0
        pass
    
    def __repr__(self):
        out = 'Universe: t=%.2g\n' % self.time
        for galaxy in self.galaxies:
            out = "%s%s\n" % (out, galaxy)
        return out
        
    def add_galaxy(self, galaxy=None):
        if galaxy is None:
            stellar_mass = 10**(4*random.random()) * 1e6
            cold_gas_mass = 10**(4*random.random()) * 1e6
            total_mass = (cold_gas_mass + stellar_mass)*1e2
            galaxy = MovingGalaxy(total_mass,
                                  cold_gas_mass,
                                  stellar_mass,
                                  x_position=random.random()*100,
                                  y_position=random.random()*100,
                                  x_velocity=random.uniform(-1,1)*1e-7,
                                  y_velocity=random.uniform(-1,1)*1e-7,
                                  idnum=len(self.galaxies))
        self.galaxies.append(galaxy)
        
    def remove_galaxy(self, galaxy):
        if galaxy in self.galaxies:
            del self.galaxies[self.galaxies.index(galaxy)]
        
    def evolve(self, time):
        for galaxy in self.galaxies:
            galaxy.evolve(time)
            galaxy.x_position %= 100
            galaxy.y_position %= 100
        self.check_for_mergers()
        for galaxy in self.removed_galaxies:
            self.remove_galaxy(galaxy)
        for galaxy in self.added_galaxies:
            self.add_galaxy(galaxy)
        self.removed_galaxies = []
        self.added_galaxies = []
        self.time += time
            
    def merge_galaxies(self, galaxy1, galaxy2):
        print('Merging:\n%s\n%s' % (galaxy1, galaxy2))
        x_mom1, y_mom1 = galaxy1.calculate_momentum()
        x_mom2, y_mom2 = galaxy2.calculate_momentum()
        new_total_mass = galaxy1.total_mass + galaxy2.total_mass
        new_galaxy = MovingGalaxy(total_mass = new_total_mass,
                                  cold_gas_mass = galaxy1.cold_gas_mass + galaxy2.cold_gas_mass,
                                  stellar_mass = galaxy1.stellar_mass + galaxy2.stellar_mass,
                                  x_position = galaxy1.x_position,
                                  y_position = galaxy1.y_position,
                                  x_velocity = (x_mom1 + x_mom2) / new_total_mass,
                                  y_velocity = (y_mom1 + y_mom2) / new_total_mass,
                                  idnum = galaxy1.idnum)
        self.added_galaxies.append(new_galaxy)
        self.removed_galaxies.append(galaxy1)
        self.removed_galaxies.append(galaxy2)
        
    def check_for_mergers(self):
        for i, galaxy1 in enumerate(self.galaxies):
            for j, galaxy2 in enumerate(self.galaxies[i+1:]):
                if distance(galaxy1, galaxy2) <= 2:
                    self.merge_galaxies(galaxy1, galaxy2)
                
    def plot_state(self, frame_id):
        plt.clf()
        x = [galaxy.x_position for galaxy in self.galaxies]
        y = [galaxy.y_position for galaxy in self.galaxies]
        color = [galaxy.color for galaxy in self.galaxies]
        size = [galaxy.total_mass / 1e9 for galaxy in self.galaxies]
        plt.scatter(x,y, color=color, s=size)
        plt.xlim(uni.xrange)
        plt.ylim(uni.yrange)
        plt.savefig('frame%04i.png' % frame_id)

if __name__ == '__main__':
    uni = Universe()
    n_timesteps = 2e2
    n_galaxies = 25
    for i in range(n_galaxies):
        uni.add_galaxy()

    for i in range(int(n_timesteps)):
        uni.evolve(2e9/n_timesteps)
        uni.plot_state(i)

