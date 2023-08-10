import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

random_state = np.random.RandomState(2014)

def make_random_cluster_points(n_samples, random_state=random_state):
    mu_options = np.array([(-1, -1), (1, 1), (1, -1), (-1, 1)])
    sigma = 0.2
    mu_choices = random_state.randint(0, len(mu_options), size=n_samples)
    means = mu_options[mu_choices]
    return means + np.random.randn(n_samples, 2) * sigma, mu_choices

def plot_clusters(data, clusters, name):
    plt.figure()
    colors = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"]
    for i in np.unique(clusters):
        plt.scatter(data[clusters==i, 0], data[clusters==i, 1], color=colors[i])
    plt.axis('off')
    plt.title('Plot from %s' % name)

data, clusters = make_random_cluster_points(10000)
plot_clusters(data, clusters, "data in memory")

# 1: import PyTables
import tables

# 2: create some data
sample_data, sample_clusters = make_random_cluster_points(10000)

# 3: create a new HDF5 file
hdf5_path = "my_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    # 4: create 2 PyTables arrays to store the data
    data_storage = hdf5_file.create_array(hdf5_file.root, 'data', sample_data)
    clusters_storage = hdf5_file.create_array(hdf5_file.root, 'clusters', sample_clusters)

hdf5_path = "my_data.hdf5"
with tables.open_file(hdf5_path, mode='r') as read_hdf5_file:
    # Here we slice [:] all the data back into memory, then operate on it
    hdf5_data = read_hdf5_file.root.data[:]
    hdf5_clusters = read_hdf5_file.root.clusters[:]

plot_clusters(hdf5_data, hdf5_clusters, "PyTables Array")

sample_data, sample_clusters = make_random_cluster_points(10000)

hdf5_path = "my_compressed_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_carray(hdf5_file.root, 'data',
                                          tables.Atom.from_dtype(sample_data.dtype),
                                          shape=sample_data.shape,
                                          filters=filters)
    clusters_storage = hdf5_file.create_carray(hdf5_file.root, 'clusters',
                                              tables.Atom.from_dtype(sample_clusters.dtype),
                                              shape=sample_clusters.shape,
                                              filters=filters)
    data_storage[:] = sample_data
    clusters_storage[:] = sample_clusters

hdf5_path = "my_compressed_data.hdf5"
with tables.open_file(hdf5_path, mode='r') as compressed_hdf5_file:
    # Here we slice [:] all the data back into memory, then operate on it
    uncompressed_hdf5_data = compressed_hdf5_file.root.data[:]
    uncompressed_hdf5_clusters = compressed_hdf5_file.root.clusters[:]

plot_clusters(uncompressed_hdf5_data, uncompressed_hdf5_clusters, "CArray")

sample_data[0]

get_ipython().magic('pinfo tables.File.create_earray')

hdf5_path = "my_extendable_compressed_data.hdf5"

# create a file as before
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    # The compression filter is the same
    filters = tables.Filters(complevel=5, complib='blosc')
    
    # Create the data EArray
    data_storage = hdf5_file.create_earray(
        # Location of the array
        where=hdf5_file.root,
        # Array name
        name='data',
        # the data type atom. We specify it since we are creating an empty array
        atom=tables.Atom.from_dtype(sample_data.dtype),
        # Array shape. Note that the dimension with shape 0 is resizable
        shape=(0, sample_data.shape[-1]),
        # Compression filters
        filters=filters,
        # The expected number of rows that will be added. Used to optimise HDF5 storage.
        expectedrows=len(sample_data))
    
    clusters_storage = hdf5_file.create_earray(
        where=hdf5_file.root,
        name='clusters',
        atom=tables.Atom.from_dtype(sample_clusters.dtype),
        shape=(0,),
        filters=filters,
        expectedrows=len(sample_clusters))
    
    # append the data into the two earrays, one row at a time
    for n, (d, c) in enumerate(zip(sample_data, sample_clusters)):
        data_storage.append(sample_data[n][None])
        clusters_storage.append(sample_clusters[n][None])
        
    # Note that the data is appended a row at a time for illustration only.
    # Ordinarily, you would just pass the entire sequence of data in one call to append:
    #data_storage.append(sample_data)
    #clusters_storage.append(sample_clusters)

hdf5_path = "my_extendable_compressed_data.hdf5"

with tables.open_file(hdf5_path, mode='r') as extendable_hdf5_file:
    extendable_hdf5_data = extendable_hdf5_file.root.data[:]
    extendable_hdf5_clusters = extendable_hdf5_file.root.clusters[:]
    plot_clusters(extendable_hdf5_file.root.data[10:100], extendable_hdf5_file.root.clusters[10:100], "EArray subset")
    plot_clusters(extendable_hdf5_data, extendable_hdf5_clusters, "full EArray")

hdf5_path = "my_extendable_compressed_data.hdf5"
with tables.open_file(hdf5_path, mode='a') as extendable_hdf5_file:
    extendable_hdf5_data = extendable_hdf5_file.root.data
    extendable_hdf5_clusters = extendable_hdf5_file.root.clusters
    print("Length of current data: %i" % len(extendable_hdf5_data))
    print("Length of current cluster labels: %i" % len(extendable_hdf5_clusters))
    n_added = 5
    print("Now adding %i elements to each" % n_added)
    for n, (d, c) in enumerate(zip(sample_data[:n_added], sample_clusters[:n_added])):
        extendable_hdf5_data.append(d[None])
        extendable_hdf5_clusters.append(c[None])

with tables.open_file(hdf5_path, mode='r') as extendable_hdf5_file:
    print("Length of current data: %i" % len(extendable_hdf5_file.root.data))
    print("Length of current cluster labels: %i" % len(extendable_hdf5_file.root.clusters))

hdf5_path = "my_variable_length_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', atom=tables.Float32Atom(shape=()))
    clusters_storage = hdf5_file.create_vlarray(hdf5_file.root, 'clusters', atom=tables.Int32Atom(shape=()))
    random_state = np.random.RandomState()
    for n in range(1000):
        length = int(100 * random_state.randn() ** 2)
        data_storage.append(random_state.randn(length,))
        clusters_storage.append([length % random_state.randint(1, 5)])

hdf5_path = "my_variable_length_data.hdf5"
with tables.open_file(hdf5_path, mode='r') as variable_hdf5_file:
    # read the data
    variable_hdf5_data = variable_hdf5_file.root.data
    
    # create a histogram of the row lengths
    all_lengths = np.array([len(d) for d in variable_hdf5_data])
    plt.hist(all_lengths, color="steelblue")
    plt.title("Lengths of fake variable length data")
    plt.figure()
    
    # create a scatter plot of length vs cluster
    clusters = variable_hdf5_file.root.clusters
    colors = ["#9b59b6", "#3498db", "#e74c3c", "#2ecc71"]
    for i in np.unique(clusters):
        plt.scatter([len(d) for d in variable_hdf5_data[(clusters==i).ravel()]],
                    clusters[(clusters==i).ravel()], color=colors[i])
    plt.title("Length vs. Class")
    plt.show()

get_ipython().system('ls -lh *.hdf5')

hdf5_path = "my_nd_variable_length_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    data_storage = hdf5_file.create_vlarray(hdf5_file.root, 'data', tables.Float32Atom(shape=()))
    data_shapes_storage = hdf5_file.create_earray(hdf5_file.root, 'data_shape', tables.Int32Atom(), shape=(0, 2), expectedrows=1000)
    random_state = np.random.RandomState(1999)
    for n in range(1000):
        shape = (int(100 * random_state.randn() ** 2), random_state.randint(2, 10))
        data_storage.append(random_state.randn(*shape).ravel())
        data_shapes_storage.append(np.array(shape)[None])

hdf5_path = "my_nd_variable_length_data.hdf5"

# Note we don't use a context manager to open the file, as this file is used across multiple code cells
nd_variable_hdf5_file = tables.open_file(hdf5_path, mode='r')
nd_variable_hdf5_data = nd_variable_hdf5_file.root.data
nd_variable_hdf5_shape = nd_variable_hdf5_file.root.data_shape

old_data_getter = nd_variable_hdf5_file.root.data.__getitem__
shape_getter = nd_variable_hdf5_file.root.data_shape.__getitem__

import numbers

def getter(self, key):
    if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
        start, stop, step = self._process_range(key, key, 1)
        if key < 0:
            key = start
        return old_data_getter(key).reshape(shape_getter(key))
    elif isinstance(key, slice):
        start, stop, step = self._process_range(key.start, key.stop, key.step)
        raise ValueError("Variable length - what should we do?")

print(getter(nd_variable_hdf5_data, -1).shape)
print(nd_variable_hdf5_data[-1].shape)
print(nd_variable_hdf5_shape[-1])

def getter(self, key):
    if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
        start, stop, step = self._process_range(key, key, 1)
        if key < 0:
            key = start
        return old_data_getter(key).reshape(shape_getter(key))
    elif isinstance(key, slice):
        start, stop, step = self._process_range(key.start, key.stop, key.step)
        return [old_data_getter(k).reshape(shape_getter(k)) for k in range(start, stop, step)]

list_of_ragged_arrays = getter(nd_variable_hdf5_data, slice(-5, -1, 1))
print([d.shape for d in list_of_ragged_arrays])

class _my_VLArray_subclass(tables.VLArray):
    pass

nd_variable_hdf5_file.root.data.__class__ = _my_VLArray_subclass
_my_VLArray_subclass.__getitem__ = getter

print(nd_variable_hdf5_data[-1].shape)
list_of_ragged_arrays = nd_variable_hdf5_data[-5:-1]
print([d.shape for d in list_of_ragged_arrays])

hdf5_file.close()

hdf5_path = "my_earray_variable_length_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data',
                                          tables.Int32Atom(),
                                          shape=(0,),
                                          filters=filters,
                                          # guess that there will mean of 50 numbers per sample
                                          expectedrows=50 * 1000)
    data_start_and_stop_storage = hdf5_file.create_earray(hdf5_file.root,
                                                         'data_start_and_stop',
                                                         tables.Int32Atom(),
                                                         shape=(0, 2),
                                                         filters=filters)
    data_shape_storage = hdf5_file.create_earray(hdf5_file.root,
                                                'data_shape',
                                                tables.Int32Atom(),
                                                shape=(0, 2),
                                                filters=filters)
    random_state = np.random.RandomState(1999)
    start = 0
    stop = 0
    for n in range(1000):
        shape = random_state.randint(2, 10, 2)
        length = shape[0] * shape[1]
        # fake 2D array of ints (pseudo images) 
        fake_image = random_state.randint(0, 256, length)
        for i in range(len(fake_image)):
            data_storage.append(fake_image[i][None])
        stop = start + length  # Not inclusive!
        data_start_and_stop_storage.append(np.array((start, stop))[None])
        data_shape_storage.append(np.array((shape))[None])
        start = stop

hdf5_path = "my_earray_variable_length_data.hdf5"
earray_variable_hdf5_file = tables.open_file(hdf5_path, mode='r')
earray_variable_hdf5_data = earray_variable_hdf5_file.root.data
earray_variable_hdf5_start_and_stop = earray_variable_hdf5_file.root.data_start_and_stop
earray_variable_hdf5_shape = earray_variable_hdf5_file.root.data_shape

print("Length of flattened data (*not* n_samples!) %i" % len(earray_variable_hdf5_data))
print("Length of shapes (actual n_samples) %i" % len(earray_variable_hdf5_shape))
print("Length of start and stop (actual n_samples) %i" % len(earray_variable_hdf5_shape))

old_data_getter = earray_variable_hdf5_file.root.data.__getitem__
shape_getter = earray_variable_hdf5_file.root.data_shape.__getitem__
start_and_stop_getter = earray_variable_hdf5_file.root.data_start_and_stop.__getitem__
def getter(self, key):
    if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
        if key < 0:
            key = len(earray_variable_hdf5_shape) + key
        data_start, data_stop = start_and_stop_getter(key)
        shape = shape_getter(key)
        return old_data_getter(slice(data_start, data_stop, 1)).reshape(shape)
    elif isinstance(key, slice):
        start = key.start if key.start is not None else 0
        stop = key.stop if key.stop is not None else len(earray_variable_hdf5_shape)
        step = key.step if key.step is not None else 1
        pos_keys = [len(earray_variable_hdf5_shape) + k if k < 0 else k
                    for k in range(start, stop, step)]
        starts_and_stops = [start_and_stop_getter(k) for k in pos_keys]
        shapes = [shape_getter(k) for k in pos_keys]
        return [old_data_getter(slice(dstrt, dstp, 1)).reshape(shp)
                for ((dstrt, dstp), shp) in zip(starts_and_stops, shapes)]
print("Testing single key %s" % str(getter(earray_variable_hdf5_data, -1).shape))
print("Testing slice %i" % len(getter(earray_variable_hdf5_data, slice(-20, -1, 1))))
                                                                
class _my_EArray_subclass(tables.EArray):
    pass
earray_variable_hdf5_file.root.data.__class__ = _my_EArray_subclass
_my_EArray_subclass.__getitem__ = getter
_my_EArray_subclass.__len__ = earray_variable_hdf5_shape.__len__
print(earray_variable_hdf5_data[-2].shape)
print(len(earray_variable_hdf5_data))
print(len(earray_variable_hdf5_data[-20:-1]))

all_points = sum([shp[0] * shp[1] for shp in earray_variable_hdf5_shape])
print("Total number of datapoints %i" % all_points)
print("kB for raw storage %f" % ((all_points * 4.) / float(1E3)))
get_ipython().system('ls -lh my_earray_variable*.hdf5')

earray_variable_hdf5_file.close()

# create sample data
sample_data, sample_clusters = make_random_cluster_points(10000)
hdf5_path = "my_inmemory_data.hdf5"

# write the file
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    data_storage = hdf5_file.create_array(hdf5_file.root, 'data', sample_data)
    clusters_storage = hdf5_file.create_array(hdf5_file.root, 'clusters', sample_clusters)
    
# Open as an in-memory file using the H5FD_CORE driver
with tables.open_file(hdf5_path, mode='r', driver='H5FD_CORE') as read_hdf5_file:
    hdf5_data = read_hdf5_file.root.data[:]
    hdf5_clusters = read_hdf5_file.root.clusters[:]

plot_clusters(hdf5_data, hdf5_clusters, "in memory PyTables array")

def add_memory_swapper(earray, mem_size):
    class _cEArray(tables.EArray):
        pass

    # Filthy hack to override getter which is a cextension...
    earray.__class__ = _cEArray

    earray._in_mem_size = int(float(mem_size))
    assert earray._in_mem_size >= 1E6 # anything smaller than 1MB is pretty worthless
    earray._in_mem_slice = np.empty([1] * len(earray.shape)).astype("float32")
    earray._in_mem_limits = [np.inf, -np.inf]

    old_getter = earray.__getitem__

    def _check_in_mem(earray, start, stop):
        lower = earray._in_mem_limits[0]
        upper = earray._in_mem_limits[1]
        if start < lower or stop > upper:
            return False
        else:
            return True

    def _load_in_mem(earray, start, stop):
        # start and stop are slice indices desired - we calculate different
        # sizes to put in memory
        n_bytes_per_entry = earray._in_mem_slice.dtype.itemsize
        n_entries = earray._in_mem_size / float(n_bytes_per_entry)
        n_samples = earray.shape[0]
        n_other = earray.shape[1:]
        n_samples_that_fit = int(n_entries / np.prod(n_other))
        assert n_samples_that_fit > 0
        # handle - index case later
        assert start >= 0
        assert stop >= 0
        assert stop >= start
        slice_size = stop - start
        if slice_size > n_samples_that_fit:
            err_str = "Slice from [%i:%i] (size %i) too large! " % (start, stop, slice_size)
            err_str += "Max slice size %i" % n_samples_that_fit
            raise ValueError(err_str)
        slice_limit = [start, stop]
        earray._in_mem_limits = slice_limit
        if earray._in_mem_slice.shape[0] == 1:
            # allocate memory
            print("Allocating %i bytes of memory for EArray swap buffer" % earray._in_mem_size)
            earray._in_mem_slice = np.empty((n_samples_that_fit,) + n_other, dtype=earray.dtype)
        # handle edge case when last chunk is smaller than what slice will
        # return
        limit = min([slice_limit[1] - slice_limit[0], n_samples - slice_limit[0]])
        earray._in_mem_slice[:limit] = old_getter(
            slice(slice_limit[0], slice_limit[1], 1))

    def getter(self, key):
        if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
            start, stop, step = self._process_range(key, key, 1)
            if key < 0:
                key = start
            if _check_in_mem(self, key, key):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, key, key)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[key - lower]
        elif isinstance(key, slice):
            start, stop, step = self._process_range(key.start, key.stop, key.step)
            if _check_in_mem(self, start, stop):
                lower = self._in_mem_limits[0]
            else:
                # slice into memory...
                _load_in_mem(self, start, stop)
                lower = self._in_mem_limits[0]
            return self._in_mem_slice[start - lower:stop - lower:step]
    # This line is critical...
    _cEArray.__getitem__ = getter
    return earray

hdf5_path = "my_memory_extendable_compressed_data.hdf5"
with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data',
                                          tables.Atom.from_dtype(sample_data.dtype),
                                          shape=(0, sample_data.shape[-1]),
                                          filters=filters,
                                          expectedrows=len(sample_data))
    clusters_storage = hdf5_file.create_earray(hdf5_file.root, 'clusters',
                                              tables.Atom.from_dtype(sample_clusters.dtype),
                                              shape=(0,),
                                              filters=filters,
                                              expectedrows=len(sample_clusters))
    for n, (d, c) in enumerate(zip(sample_data, sample_clusters)):
        data_storage.append(sample_data[n][None])
        clusters_storage.append(sample_clusters[n][None])

hdf5_path = "my_memory_extendable_compressed_data.hdf5"
memory_extendable_hdf5_file = tables.open_file(hdf5_path, mode='r')
memory_extendable_hdf5_data = add_memory_swapper(memory_extendable_hdf5_file.root.data, 10E6)
memory_extendable_hdf5_clusters = memory_extendable_hdf5_file.root.clusters
plot_clusters(memory_extendable_hdf5_file.root.data[10:100],
              memory_extendable_hdf5_file.root.clusters[10:100], "EArray subset")

print("Current memory limits %s" % str(memory_extendable_hdf5_data._in_mem_limits))
memory_extendable_hdf5_data[-100:]
print("Moved memory limits %s" % str(memory_extendable_hdf5_data._in_mem_limits))
memory_extendable_hdf5_data[-5:]
print("Unchanged memory limits %s" % str(memory_extendable_hdf5_data._in_mem_limits))
memory_extendable_hdf5_file.close()

