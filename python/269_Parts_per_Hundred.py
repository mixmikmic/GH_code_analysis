# Calculates number of red balls for a given quantity and pph.
def red_balls(quantity, pph):
    return int((pph/100) * quantity)

red_balls(300, 5)

def pph_calculation(dilution_volume, total_volume):
    '''Calculates the pph given the volume (in cubic centimeters) of the 
    quantity to be diluted and total volume (in cubic centimeters).'''
    return (dilution_volume / total_volume) * 100

pph_calculation(10, 1000)

def brightness(number_of_stars, total_stars):
    '''Calculates brightness in pph of a proportion of stars, given total
    number of stars in the cluster.'''
    return (number_of_stars / total_stars) * 100

brightness(4, 200)

def month_pph(days):
    '''Calculates the pph of a given number of days compared to an average 
    calendar month (30 days).'''
    return (days / 30) * 100

month_pph(1)

# pph of the total atoms
8 / 24 * 100

