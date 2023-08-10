import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt
import os

get_ipython().run_line_magic('matplotlib', 'inline')

# Loading data from text files
# You may have to modify the next line if running the notebook locally
module_dir = os.path.dirname(os.path.abspath('.\\Jupyter\\')) + os.sep

# the ASTMG173.csv file holds the standard AM1.5 solar spectrum
# You can find the original at rredc.nrel.gov/solar/spectra/am1.5
spectrum = np.loadtxt(module_dir + 'ASTMG173.csv', delimiter=',', skiprows=1)

plt.plot(spectrum[:, 0], spectrum[:, 1])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Irradiance ($Wm^{-2}nm^{-1}$)')
plt.title('Spectrum Irradiance')
original_irradiance = np.trapz(spectrum[:, 1], spectrum[:, 0])
print 'Total spectrum irriadiance: ', original_irradiance, 'W/m^2'

# Just some constants for the upcoming math
c = constants.value('speed of light in vacuum')
h = constants.value('Planck constant')
e = constants.value('elementary charge')
k = constants.value('Boltzmann constant')

# Globals
Tcell = 300  # Kelvin
# Energy Gap
# The energy gap enetered here will be highlighted on upcoming plots. 
# Default is set to silicon bandgap 1.1eV
Egap = 1.1  #electron volts
# Silicon = 1.1eV

# A helper function that will do the job for us
def convert_spectrum(spectrum):
    """
    Spectrum input:
        y: Irradiance (W/m2/nm)
        x: Wavelength (nm)
    Converted otput:
        y: Number of photons (Np/m2/s/dE)
        x: Energy (eV)
    """
    converted = np.copy(spectrum)
    converted[:, 0] = converted[:, 0] * 1e-9  # wavelength to m
    converted[:, 1] = converted[:, 1] / 1e-9  # irradiance to W/m2/m (from W/m2/nm)

    E = h * c / converted[:, 0]
    d_lambda_d_E = h * c / E**2
    converted[:, 1] = converted[:, 1] * d_lambda_d_E * e / E
    converted[:, 0] = E / e
    return converted

# Let's use the function, convert the input from the text file and have a look at it
photon_spectrum = convert_spectrum(spectrum)

plt.plot(photon_spectrum[:,0], photon_spectrum[:,1])
plt.xlabel('Energy (eV)')
plt.ylabel('# Photons $m^{-2}s^{-1}dE$')
plt.title('Converted spectrum')

# Irradiance check
photon_irradiance = np.trapz(photon_spectrum[::-1, 1] * photon_spectrum[::-1, 0],
                          photon_spectrum[::-1, 0]) * e


print 'If everything went okay this should be pretty close to the number from before'
print 'Original ', original_irradiance, ' W/m^2\tConverted ', photon_irradiance, ' W/m^2'

def photons_above_bandgap(egap, spectrum):
    """Counts number of photons above given bandgap"""
    indexes = np.where(spectrum[:, 0] > egap)
    y = spectrum[indexes, 1][0]
    x = spectrum[indexes, 0][0]
    return np.trapz(y[::-1], x[::-1])


def photons_above_bandgap_plot(spectrum):
    """Plot of photons above bandgap as a function of bandgap"""
    a = np.copy(spectrum)
    for row in a:
        # print row
        row[1] = photons_above_bandgap(row[0], spectrum)
    plt.plot(a[:, 0], a[:, 1])

    p_above_1_1 = photons_above_bandgap(Egap, spectrum)
    plt.plot([Egap], [p_above_1_1], 'ro')
    plt.text(Egap+0.05, p_above_1_1, '{}eV, {:.4}'.format(Egap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('# Photons $m^{-2}s^{-1}$')
    plt.title('Number of above-bandgap \nphotons as a function of bandgap')
    plt.show()

photons_above_bandgap_plot(photon_spectrum)

def rr0(egap, spectrum):
    k_eV = k / e
    h_eV = h / e
    const = (2 * np.pi) / (c**2 * h_eV**3)

    k_eV = k / e
    E = spectrum[::-1, ]  # in increasing order of bandgap energy
    egap_index = np.where(E[:, 0] >= egap)
    numerator = E[:, 0]**2
    exponential_in = E[:, 0] / (k_eV * Tcell)
    denominator = np.exp(exponential_in) - 1
    integrand = numerator / denominator

    integral = np.trapz(integrand[egap_index], E[egap_index, 0])

    result = const * integral
    return result[0]

def recomb_rate(egap, spectrum, voltage):
    print 'recomb rate'
    return e * rr0(egap, spectrum) * np.exp(e * voltage / (k * Tcell))

def current_density(egap, spectrum, voltage):
    # print 'current_density'
    # print photons_above_bandgap(egap, spectrum), 'photons above bandgap'
    # print e * (photons_above_bandgap(egap, spectrum) - rr0(egap, spectrum)), 'photons above bandgap - rr0'
    return e * (photons_above_bandgap(egap, spectrum) - rr0(egap, spectrum) * np.exp(e * voltage / (k * Tcell)))


def jsc(egap, spectrum):
    # print 'jsc'
    return current_density(egap, spectrum, 0)


def voc(egap, spectrum):
    # print 'voc'
    return (k * Tcell / e) * np.log(photons_above_bandgap(egap, spectrum) / rr0(egap, spectrum))

# For an ideal solar cell these will be
print 'A material with a bandgap of %.2f will have an:' % Egap
print 'Ideal short circuit current: ', jsc(1.1, photon_spectrum), 'A/m^2'
print 'Ideal open circuit  voltage: ', voc(1.1, photon_spectrum), 'V'

def ideal_jsc_plot(spectrum):
    """Plot of photons above bandgap as a function of bandgap"""
    a = np.copy(spectrum)
    for row in a:
        # print row
        row[1] = jsc(row[0], spectrum)
    plt.plot(a[:, 0], a[:, 1])
    e_gap = 1.1
    p_above_1_1 = jsc(e_gap, spectrum)
    plt.plot([e_gap], [p_above_1_1], 'ro')
    plt.text(e_gap+0.05, p_above_1_1, '{}eV, {:.4}'.format(e_gap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('$J_{SC}$ $Am^{-2}$')
    plt.title('Ideal short-circuit current')


def ideal_voc_plot(spectrum):
    """Plot of the ideal open circuit voltage as a function of bandgap"""
    a = np.copy(spectrum)
    for row in a[2:]:
        # print row
        row[1] = voc(row[0], spectrum)
    plt.plot(a[:, 0], a[:, 1])
    plt.plot(a[:, 0], a[:, 0])
    e_gap = 1.1
    p_above_1_1 = voc(e_gap, spectrum)
    plt.plot([e_gap], [p_above_1_1], 'ro')
    plt.text(e_gap+0.05, p_above_1_1, '{}eV, {:.4}'.format(e_gap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('$V_{OC}$ (V)')
    plt.xlim((0.5,3.5))
    plt.title('Ideal open-circuit voltage. Straight line is bandgap.')

ideal_jsc_plot(photon_spectrum)

ideal_voc_plot(photon_spectrum)

def iv_curve_plot(egap, spectrum, power=False):
    """Plots the ideal IV curve, or the ideal power for a given material"""
    v_open = voc(egap, spectrum)
    v = np.linspace(0, v_open)
    if power:
        p =  v * current_density(egap, spectrum, v)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Power generated ($W$)')
        plt.title('Power Curve')
        plt.plot(v, p)
    else:
        i =  current_density(egap, spectrum, v)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density $J$ ($Am^{-2}$)')
        plt.title('IV Curve')
        plt.plot(v, i)
    
iv_curve_plot(Egap, photon_spectrum)

def iv_curve_plot(egap, spectrum, power=False):
    """Plots the ideal IV curve, and the ideal power for a given material"""
    v_open = voc(egap, spectrum)
    v = np.linspace(0, v_open)

    fig, ax1 = plt.subplots()
    p =  v * current_density(egap, spectrum, v)
    i =  current_density(egap, spectrum, v)
    
    ax1.plot(v, i)
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current density $J$ ($Am^{-2}$)')
    ax1.legend(['Current'], loc=2)
    
    ax2 = ax1.twinx()
    ax2.plot(v, p, color='orange')
    ax2.set_ylabel('Power generated ($W$)')
    ax2.legend(['Power'], loc=3)
    return


iv_curve_plot(Egap, photon_spectrum)

def v_at_mpp(egap, spectrum):
    v_open = voc(egap, spectrum)
    # print v_open
    v = np.linspace(0, v_open)
    index = np.where(v * current_density(egap, spectrum, v)==max(v * current_density(egap, spectrum, v)))
    return v[index][0]


def j_at_mpp(egap, spectrum):
    return max_power(egap, spectrum) / v_at_mpp(egap, spectrum)


def max_power(egap, spectrum):
    v_open = voc(egap, spectrum)
    # print v_open
    v = np.linspace(0, v_open)
    index = np.where(v * current_density(egap, spectrum, v)==max(v * current_density(egap, spectrum, v)))
    return max(v * current_density(egap, spectrum, v))


def max_eff(egap, spectrum):
    irradiance =  np.trapz(spectrum[::-1, 1] * e * spectrum[::-1, 0], spectrum[::-1, 0])
    # print photons_above_bandgap(egap, spectrum) * e
    return max_power(egap, spectrum) / irradiance

print 'For a cell with bandgap %.2f eV' % Egap
print 'Ideal efficiency is {:.4}%'.format(max_eff(Egap, photon_spectrum)*100)

def sq_limit_plot(spectrum):
    # Plot the famous SQ limit
    a = np.copy(spectrum)
    # Not for whole array hack to remove divide by 0 errors
    for row in a[2:]:
        # print row
        row[1] = max_eff(row[0], spectrum)
    # Not plotting whole array becase some bad values happen
    plt.plot(a[2:, 0], a[2:, 1])
    e_gap = Egap
    p_above_1_1 = max_eff(e_gap, spectrum)
    plt.plot([e_gap], [p_above_1_1], 'ro')
    plt.text(e_gap+0.05, p_above_1_1, '{}eV, {:.4}'.format(e_gap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('Max efficiency')
    plt.title('SQ Limit')

sq_limit_plot(photon_spectrum)

def excess_beyond_gap(egap, spectrum):
    """Loss due to energy beyond bandgap; lost as heat"""
    # find egap index
    indexes = np.where(spectrum[:, 0] > egap)
    y = spectrum[indexes, 1][0]
    x = spectrum[indexes, 0][0]
    return np.trapz(y[::-1] * (x[::-1] - egap), x[::-1])


def energy_below_gap(egap, spectrum):
    """Energy of photons below the bandgap"""
    # find egap index
    indexes = np.where(spectrum[:, 0] < egap)
    y = spectrum[indexes, 1][0]
    x = spectrum[indexes, 0][0]
    return np.trapz(y[::-1] * x[::-1], x[::-1])


def mpp_recombination(egap, spectrum):
    """Loss due to recombination at the maximum power point"""
    return photons_above_bandgap(egap, spectrum) - j_at_mpp(egap, spectrum) / e


def mpp_v_less_than_gap(egap, spectrum):
    return j_at_mpp(egap, spectrum) * (egap - v_at_mpp(egap, spectrum))


def loss(egap, spectrum):
    """Returns a list with a breakdown of where the energy is lost"""
    solar_constant = np.trapz(spectrum[::-1, 1] * e * spectrum[::-1, 0], spectrum[::-1, 0])
    useful_electricity = max_eff(egap, spectrum)
    below_gap = energy_below_gap(egap, spectrum) * e / solar_constant
    beyond_gap = excess_beyond_gap(egap, spectrum) * e / solar_constant
    recomb = mpp_recombination(egap, spectrum) * e / solar_constant
    v_less = mpp_v_less_than_gap(egap, spectrum) / solar_constant
    return [useful_electricity, below_gap, beyond_gap, recomb, v_less]

def print_losses(egap, spectrum):
    """Print the breakdown of loss() nicely"""
    losses = loss(egap, spectrum)
    print 'Useful electricity: \t\t\t{:.4}%'.format(losses[0]*100)
    print 'Below bandgap losses: \t\t\t{:.4}%'.format(losses[1]*100)
    print 'Excess beyond gaplosses:\t\t{:.4}%'.format(losses[2]*100)
    print 'Recombination losses: \t\t\t{:.4}%'.format(losses[3]*100)
    print 'MPP Voltage less than gap losses: \t{:.4}%'.format(losses[4]*100)
    print 'Total should be close to 100%: \t\t{:.4}%'.format(sum(losses)*100)
    
print_losses(Egap, photon_spectrum)

def loss_plot(spectrum):
    gaps = spectrum[2:,0]

    a_arr = np.empty_like(gaps)
    b_arr = np.empty_like(gaps)
    c_arr = np.empty_like(gaps)
    d_arr = np.empty_like(gaps)
    e_arr = np.empty_like(gaps)
    
    for index, gap in enumerate(gaps):
        q,w,e,r,t = loss(gap, spectrum)
        a_arr[index] = q
        b_arr[index] = w
        c_arr[index] = e
        d_arr[index] = r
        e_arr[index] = t
    
    plt.xlabel('Bandgap energy (eV)')
    plt.stackplot(gaps, [a_arr,b_arr,c_arr,d_arr,e_arr])
    plt.xlim(0.5,3.5)
    plt.ylim(0,1)
    labels = ['Useful electricity', 
          'Below bandgap photons', 
          'Energy beyond bandgap', 
          'e-h recombination', 
          'Voltage less than bandgap']
#     plt.legend(labels, loc='upper left', bbox_to_anchor=(1,1))
    plt.legend(labels, loc=7)
    
loss_plot(photon_spectrum)

def plot_pie_breakdown(spectrum):
    losses_11eV = loss(Egap, spectrum)
    labels = ['Useful electricity', 
              'Below bandgap photons', 
              'Energy beyond bandgap', 
              'e-h recombination', 
              'Voltage less than bandgap']
    plt.pie(losses_11eV, labels=labels, autopct='%.1f%%')
    plt.title('Losses for a material with bandgap {}eV'.format(Egap))
    plt.axes().set_aspect('equal')
    plt.show()
    
plot_pie_breakdown(photon_spectrum)

def normalise_power(spectrum):
    result = np.copy(spectrum)
    power = np.trapz(result[:,1], result[:,0])
    result[:,1] /= power
    return result

def denormalise_power(spectrum, power):
    result = np.copy(spectrum)
    result[:,1] *= power
    return result

spectrum_100W = denormalise_power(normalise_power(spectrum), 100)
spectrum_10W = denormalise_power(normalise_power(spectrum), 10)
spectrum_1W = denormalise_power(normalise_power(spectrum), 1)
photon_spectrum_100W = convert_spectrum(spectrum_100W)
photon_spectrum_10W = convert_spectrum(spectrum_10W)
photon_spectrum_1W = convert_spectrum(spectrum_1W)

print '1000W'
print_losses(1.1, photon_spectrum)
print '\n100W'
print_losses(1.1, photon_spectrum_100W)
print '\n10W'
print_losses(1.1, photon_spectrum_10W)
print '\n1W'
print_losses(1.1, photon_spectrum_1W)

x = [1, 10, 100, 1000]
a_arr = np.empty_like(x)
b_arr = np.empty_like(x)
c_arr = np.empty_like(x)
d_arr = np.empty_like(x)
e_arr = np.empty_like(x)
w1 = loss(1.1, photon_spectrum_1W)
w2 = loss(1.1, photon_spectrum_10W)
w3 = loss(1.1, photon_spectrum_100W)
w4 = loss(1.1, photon_spectrum)

a_arr = [w1[0], w2[0], w3[0], w4[0]]
b_arr = [w1[1], w2[1], w3[1], w4[1]]
c_arr = [w1[2], w2[2], w3[2], w4[2]]
d_arr = [w1[3], w2[3], w3[3], w4[3]]
e_arr = [w1[4], w2[4], w3[4], w4[4]]

plt.stackplot(x, [a_arr, b_arr, c_arr, d_arr, e_arr])
plt.semilogx()
plt.xlabel('Total Irradiance ($Wm^{-2}$)')
plt.title('Losses against total power.\nUseful electricity in blue.')
plt.show()

# Loading spectrum information
led_blue = np.loadtxt(module_dir + 'led_blue.csv', delimiter=',', skiprows=1)
led_red = np.loadtxt(module_dir + 'led_red.csv', delimiter=',', skiprows=1)
led_ir = np.loadtxt(module_dir + 'led_ir.csv', delimiter=',', skiprows=1)
led_green = np.loadtxt(module_dir + 'led_green.csv', delimiter=',', skiprows=1)
led_white = np.loadtxt(module_dir + 'led_white.csv', delimiter=',', skiprows=1)
fluor = np.loadtxt(module_dir + 'fluorescent.csv', delimiter=',', skiprows=1)

# Let's pack it up in a dict
spectra = {'white led': led_white,
           'red led': led_red,
           'green led': led_green,
           'blue led': led_blue,
           'ir led': led_ir,
           'fluorescent': fluor}

# Normalise to a peak of 1
def normalise_peak(spectrum):
    """overwrite original spectrum to have a peak of 1"""
    peak = max(spectrum[:,1])
    spectrum [:,1] = spectrum[:,1] / peak
    return


# And plotting it for comparison
for name, data in spectra.items():
    normalise_peak(data)
    plt.plot(data[:,0], data[:,1])
    
plt.legend(spectra.keys())
plt.xlabel('Wavelength (nm)')
plt.xlim((300,950))
plt.ylabel('Relative Irradiance (a.u.)')
plt.title('Relative irradiance for different light sources')
plt.show()

for name, data in spectra.items():
    print '{} irradiance: {:.4}'.format(name, np.trapz(data[:,1], data[:,0]))

# Conversion to photon spectrum and plot
ledw_photon_spectrum = convert_spectrum(spectra['white led'])

plt.plot(ledw_photon_spectrum[:,0], ledw_photon_spectrum[:,1])
plt.xlabel('Energy (eV)')
plt.ylabel('# Photons $m^{-2}s^{-1}dE$')
plt.title('Converted spectrum')
plt.show()

photons_above_bandgap_plot(ledw_photon_spectrum)

ideal_jsc_plot(ledw_photon_spectrum)

iv_curve_plot(Egap, ledw_photon_spectrum)

sq_limit_plot(ledw_photon_spectrum)

print_losses(1.1, ledw_photon_spectrum)

loss_plot(ledw_photon_spectrum[61:,:])

# Conversion to photon spectrum and plot
fluor_photon_spectrum = convert_spectrum(spectra['fluorescent'])

plt.plot(fluor_photon_spectrum[:,0], fluor_photon_spectrum[:,1])
plt.xlabel('Energy (eV)')
plt.ylabel('# Photons $m^{-2}s^{-1}dE$')
plt.title('Converted spectrum')
plt.show()

photons_above_bandgap_plot(fluor_photon_spectrum)

iv_curve_plot(Egap, fluor_photon_spectrum)

sq_limit_plot(fluor_photon_spectrum)

print_losses(Egap, fluor_photon_spectrum)

loss_plot(fluor_photon_spectrum[61:,:])

photopic = np.loadtxt(module_dir + 'photopic.csv', delimiter=',', skiprows=1)
plt.plot(photopic[:,0], photopic[:,1])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Relative response')
plt.title('Photopic response function $V(\lambda$)')
plt.show()

# Copying only the portion for which the photopic function is defined
led_white_photopic = np.copy(spectra['white led'][:601,:])
led_white_photopic[:,1] = led_white_photopic[:,1] * photopic[:,1] * 683

fluor_photopic = np.copy(spectra['fluorescent'][:601,:])
fluor_photopic[:,1] = fluor_photopic[:,1] * photopic[:,1] * 683

plt.plot(led_white_photopic[:,0], led_white_photopic[:,1])
plt.plot(fluor_photopic[:,0], fluor_photopic[:,1])
plt.xlabel('Wavelength (nm)')
plt.xlim((300,900))
plt.ylabel('Illuminance ($lx*nm^{-1}$)')
plt.title('Illuminance')
plt.show()

plt.plot(spectra['white led'][:,0], spectra['white led'][:,1])
plt.plot(spectra['fluorescent'][:,0], spectra['fluorescent'][:,1])
plt.xlabel('Wavelength (nm)')
plt.xlim((300,900))
plt.ylabel('Irradiacne ($Wm^{-2}nm^{-1}$)')
plt.title('Irradiance')
plt.show()

print 'White LED Brightness: {} lux'.format(np.trapz(led_white_photopic[:,1], led_white_photopic[:,0]))
print 'Fluorescent bulb Brightness: {} lux'.format(np.trapz(fluor_photopic[:,1], fluor_photopic[:,0]))

def find_photopic_coefficient(wavelen):
    """Searches the photopic response function"""
    for row in photopic:
        if row[0] == wavelen:
            coeff = row[1]
            break
        else:
            coeff = 0
    return coeff


def denormalise_brightness(spectrum, illuminance):
    """Returns a spectrum in watts with a given illuminance in lux"""
    result = np.copy(spectrum)
    luminance_spectrum = np.copy(spectrum)
    for row in luminance_spectrum:
        row[1] = row[1] * find_photopic_coefficient(row[0]) * 683
    lux = np.trapz(luminance_spectrum[:,1], luminance_spectrum[:,0])
    coeff = illuminance / lux
    result[:, 1] *= coeff
    return result

sun_1000lux = denormalise_brightness(spectrum, 1000)
led_1000lux = denormalise_brightness(spectra['white led'], 1000)
fluor_1000lux = denormalise_brightness(spectra['fluorescent'], 1000)

print 'Sunlight power: {} W/m^2'.format(np.trapz(sun_1000lux[:,1], sun_1000lux[:,0]))
print 'White LED power: {} W/m^2'.format(np.trapz(led_1000lux[:,1], led_1000lux[:,0]))
print 'Fluorescent power: {} W/m^2'.format(np.trapz(fluor_1000lux[:,1], fluor_1000lux[:,0]))

plt.plot(sun_1000lux[:,0], sun_1000lux[:,1])
plt.plot(led_1000lux[:,0], led_1000lux[:,1])
plt.plot(fluor_1000lux[:,0], fluor_1000lux[:,1])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Irradiance ($Wm^{-2}nm^{-1}$)')
plt.title('Spectral irradiance of different sources with equal illuminance\n(x axis limited to 1500nm)')
plt.xlim(250,1500)
plt.show()

print 'Sunlight:'
print_losses(1.1, convert_spectrum(sun_1000lux))
print '\nWhite LED:'
print_losses(1.1, convert_spectrum(led_1000lux))
print '\nFluorescent:'
print_losses(1.1, convert_spectrum(fluor_1000lux))

def sq_limit_plot(spectrum):
    # Plot the famous SQ limit
    a = np.copy(spectrum)
    # Not for whole array hack to remove divide by 0 errors
    for row in a[2:]:
        # print row
        row[1] = max_eff(row[0], spectrum)
    # Not plotting whole array becase some bad values happen
    plt.plot(a[2:, 0], a[2:, 1])
#     e_gap = Egap
#     p_above_1_1 = max_eff(e_gap, spectrum)
#     plt.plot([e_gap], [p_above_1_1], 'ro')
#     plt.text(e_gap+0.05, p_above_1_1, '{}eV, {:.4}'.format(e_gap, p_above_1_1))

    plt.xlabel('$E_{gap}$ (eV)')
    plt.ylabel('Max efficiency')
    plt.title('Detailed Balance L Limit')

sq_limit_plot(convert_spectrum(sun_1000lux))
sq_limit_plot(convert_spectrum(led_1000lux))
sq_limit_plot(convert_spectrum(fluor_1000lux))
e_gap = Egap
p_above_1_1 = max_eff(e_gap, convert_spectrum(sun_1000lux))
plt.plot([e_gap], [p_above_1_1], 'ro') 

plt.legend(['Solar','White LED','Fluorescent'])
plt.show()

plt.pie(loss(Egap, convert_spectrum(led_1000lux)), 
        autopct='%.1f%%',
        labels=['Useful electricity', 'Below bandgap photons', 'Energy beyond bandgap', 'e-h recombination', 'Voltage less than bandgap'])
plt.title('Losses for a material with bandgap {}eV\nand 1000lux of white LED light'.format(Egap))
plt.axes().set_aspect('equal')
plt.show()

