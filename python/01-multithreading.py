# Packages bundled with Python itself
import os
import subprocess
import timeit

# Installed packages
import cpuinfo  # provided by py-cpuinfo conda package
from IPython.display import SVG, display
import matplotlib.pyplot as plt
import mkl  # provided by mkl-service conda package
import numpy as np
import scipy.linalg
import pyfftw

# Enable nicer Matplotlib styling
plt.style.use('ggplot')

# Automatically show Matplotlib plots in Notebook 
get_ipython().magic('matplotlib inline')

np.show_config()

# NB a 5000x5000 element matrix of 64-bit double-precision floating point numbers requires ~190 MiB of RAM
shape = (5000, 5000)

# Set the pseudo-random number generator (PRNG) seed for reproducibility
np.random.seed(42)
x = np.random.random_sample(shape)
y = np.random.random_sample(shape)
x[:5, :5]

x @ y

def sq_matr_mult_timings(side_length, thread_cnts, plot_results=False, prng_seed=None):
    """Time the multiplication of two generated square matrices using a set number of threads and optionally plot results.
    
    Parameters
    ----------
    side_length : 
        size of either dimension of the two square 
    thread_cnts : array_like
        sequence of thread/core counts for which to capture timings
    plot_results : optional
        Whether to plot timing data (default: False)
    prng_seed : optional
        pseudo-random-number-generator seed (default: None)    
    
    Returns
    -------
    (timings,)
        a sequence of timings the same length as thread_cnts 
    """
    shape = (side_length, side_length)
    if prng_seed:
        np.random.seed(prng_seed)
    x = np.random.random_sample(shape)
    y = np.random.random_sample(shape)

    timings = []
    for i in thread_cnts:
        # Set the maximum number of threads that can be used by the MKL
        mkl.set_num_threads(i)
        print("Thread count: {}".format(i), end=" -> ")

        # Execute matrix multiplication repeatedly for at least 0.2s whilst timing
        # NB in order to generate a timeit Timer that can see the references to x and y 
        # defined in this func we must use a lambda function (https://stackoverflow.com/a/31572755)
        n_itrs, t = timeit.Timer(lambda: x @ y).autorange()
        print("{:.3f} ms per iteration on average ({} iterations took {:.3f} s)".format(
            t * 1000 / n_itrs, n_itrs, t))
        timings.append(t * 1000  / n_itrs)
    
    if plot_results:
        fig, ax = plt.subplots()
        ax.scatter(thread_cnts, timings)
        ax.set_xticks(thread_cnts)
        ax.set_xticklabels(thread_cnts)
        ax.set_xlabel('# cores')
        ax.set_ylabel('Runtime [ms]');
        ax.set_ylim(0, max(timings) * 1.1);
        
    return timings

timings = sq_matr_mult_timings(5000, range(1, 5), plot_results=True)

timings = sq_matr_mult_timings(50, range(1, 5), plot_results=True)

# Create an empty temporary file
tmp = os.path.join(os.sep, 
                   os.environ['TMPDIR'] if 'TMPDIR' in os.environ else 'tmp',
                   str(os.getpid()))

# Run the lstopo command-line utility to capture information about available hardware resources,
# generate a SVG image from that data and save the SVG to our temporary file
subprocess.check_output(['lstopo', '--no-io', 
                                   '--no-bridges', 
                                   '--output-format', 'svg', 
                                   '-f', tmp])

# Display the SVG in this Notebook
with open(tmp, 'r') as svg_file:
    svg = svg_file.read()
SVG(svg)

z = np.random.random_sample((1500, 500))

def svd_duration_ms(matr, n_threads):
    """Time a singular value decomposition (SVD)

    Parameters
    ----------
    matr : 
        numpy matrix to perform SVD on 
    n_threads : 
        number of threads to use with MKL    

    Returns
    -------
    timing - duration in milliseconds
    """
    print("Thread count: {}".format(n_threads), end=" -> ")
    mkl.set_num_threads(n_threads)

    n_itrs, t = timeit.Timer(lambda: scipy.linalg.svd(matr)).autorange()                                                                                                             
    print("{:.3f} ms per iteration on average ({} iterations took {:.3f} s)".format(                                                                                                 
          t * 1000 / n_itrs, n_itrs, t))                                                                                                                                           
    return t * 1000  / n_itrs                                                                                                                                                        

timings = [svd_duration_ms(z, n) for n in range(1, 5)]   

timings = [svd_duration_ms(z, n) for n in range(5, 9)]

# Size of 3D FFT input
size = (1024, 1024, 256)

# Preallocate a 2GB array of complex values for storing randomly generated input data.
# This preallocation allows the array to be 'aligned' in memory in a way that offers 
# best performance on hardware with SIMD support (e.g. with AVX support; see mention of SIMD in the next section)
a = pyfftw.empty_aligned(size, dtype='complex128')
out = pyfftw.empty_aligned(size, dtype='complex128')

# Randomly generate the real and imaginary components of our 3D FFT input
a[:] = np.random.randn(*size) + (1j * np.random.randn(*size))

# To store timings for the FFT calculations
timings = []

# Range of thread counts for which to measure performance
thread_cnts = np.arange(1, 5)

for i in thread_cnts:
    # Constructs a plan for how to calculate the FFT efficiently given the input data type and size 
    # Note that we're not timing this setup function
    fft_plan = pyfftw.FFTW(a, out, threads=i) 
    
    # Time the FFT calculation
    n_itrs, t = timeit.Timer(lambda: fft_plan.execute()).autorange()                                                                                                             
    print("{:.3f} ms per iteration on average ({} iterations took {:.3f} s)".format(                                                                                                 
          t * 1000 / n_itrs, n_itrs, t))   
    timings.append(t * 1000  / n_itrs)
    
# Plot the results
fig, ax = plt.subplots()
ax.scatter(thread_cnts, timings)
ax.set_xticks(thread_cnts)
ax.set_xticklabels(thread_cnts)
ax.set_xlabel('# cores')
ax.set_ylabel('Runtime [ms]');
ax.set_ylim(ymin=0);

# Free memory
del a, out

cpu_features = cpuinfo.get_cpu_info()['flags']
# See whether particular CPU features of interest are in the set of all available features
set(cpu_features).intersection(set(('fma', 'avx', 'avx2', 'avx512')))

get_ipython().system("grep -oE '\\W(fma|avx|avx2)\\W' /proc/cpuinfo | sort | uniq")

