from itertools import product
from functools import wraps

import numpy as np
from lll import lll
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
from ipywidgets import interact

init_notebook_mode()

def store_last_correct(func):
    last_correct = None
    
    @wraps(func)
    def wrapped(*args, **kwargs):
        nonlocal last_correct
        try:
            last_correct = func(*args, **kwargs)
            return last_correct
        except:
            return last_correct
    
    return wrapped

def unit_cell(a, b, c, atom_pos, Nx, Ny, Nz):
    """Make arrays of x-, y- and z-positions of a lattice from the
    lattice vectors, the atom positions and the number of unit cells.
    
    Parameters:
    -----------
    a : list
        First lattice vector
    b : list
        Second lattice vector
    c : list
        Third lattice vector
    atom_pos : list
        Positions of atoms in the unit cells in terms of a, b and c
    Nx : int
        number of unit cells in the x-direction to be plotted
    Ny : int
        number of unit cells in the y-direction to be plotted
    Nz : int
        number of unit cells in the z-direction to be plotted
        
    Returns:
    --------
    latt_coord_x : numpy.ndarray
        Array containing the x-coordinates of all atoms to be plotted
    latt_coord_y : numpy.ndarray
        Array containing the y-coordinates of all atoms to be plotted
    latt_coord_z : numpy.ndarray
        Array containing the z-coordinates of all atoms to be plotted
    """
    latt_coord_x = []
    latt_coord_y = []
    latt_coord_z = []
    for atom in atom_pos:
        xpos = atom[0]*a[0] + atom[1]*b[0] + atom[2]*c[0]
        ypos = atom[0]*a[1] + atom[1]*b[1] + atom[2]*c[1]
        zpos = atom[0]*a[2] + atom[1]*b[2] + atom[2]*c[2]
        xpos_all = [xpos + n*a[0] + m*b[0] + k*c[0] for n, m, k in
                     product(range(Nx), range(Ny), range(Nz))]
        ypos_all = [ypos + n*a[1] + m*b[1] + k*c[1] for n, m, k in
                     product(range(Nx), range(Ny), range(Nz))]
        zpos_all = [zpos + n*a[2] + m*b[2] + k*c[2] for n, m, k in
                     product(range(Nx), range(Ny), range(Nz))]
        latt_coord_x.append(xpos_all)
        latt_coord_y.append(ypos_all)
        latt_coord_z.append(zpos_all)
    latt_coord_x = np.array(latt_coord_x).flatten()
    latt_coord_y = np.array(latt_coord_y).flatten()
    latt_coord_z = np.array(latt_coord_z).flatten()
    return latt_coord_x, latt_coord_y, latt_coord_z


def reciprocal_lattice_vectors(*vectors):
    """Calculate reciprocal lattice vectors given lattice vectors.
    
    Using formula from
    https://en.wikipedia.org/wiki/Reciprocal_lattice#Generalization_of_a_dual_lattice
    """
    vectors = np.array(vectors).T
    reciprocal_vectors = vectors.dot(np.linalg.inv(vectors.T.dot(vectors)))
    return reciprocal_vectors


def laue_conditions(coords, k, dk, elastic=True):
    """Calculate which coordinates on reciprocal lattice satisfy Laue conditions.
    
    To not get nothing almost always, we're assuming that the wave vector `k`
    has a small spread dk.
    """
    k = np.array(k)[..., :]
    coords = np.asarray(coords)
    dk /= np.linalg.norm(k)
    where = ((np.linalg.norm(coords + k * (1 + elastic * dk/2), axis=1) < 
              np.linalg.norm(k) * (1 + dk/2)) &
             (np.linalg.norm(coords + k * (1 - elastic *dk/2), axis=1) > 
              np.linalg.norm(k) * (1 - dk/2)))
    return where

def rotation_matrix(k):
    if len(k) == 3:
        if np.allclose(k/np.linalg.norm(k), (1., 0, 0)):
            return np.identity(3)
        rotation_axis = np.cross(k, (1., 0, 0))
        rotation_axis /= np.linalg.norm(rotation_axis)
        cross_product_matrix = np.array([[0., -rotation_axis[2], rotation_axis[1]],
                                         [0, 0, -rotation_axis[0]], [0, 0, 0]])
        cross_product_matrix = cross_product_matrix - cross_product_matrix.T
        # rotation matrix
        R = (k[0] * np.identity(3) + np.linalg.norm(k[1:]) * cross_product_matrix +
             (np.linalg.norm(k) - k[0]) * np.outer(rotation_axis, rotation_axis))
        R /= np.linalg.norm(k)
    elif len(k) == 2:
        return np.array([[k[0], k[1]], [-k[1], k[0]]]) / np.linalg.norm(k)
    else:
        raise ValueError('Only 2D or 3D are supported.')
    return R

def k_image(coordinates, k, spherical_aberration=False):
    coordinates = coordinates.T + k[:, None]
    coordinates = rotation_matrix(k).dot(coordinates)
    small_angles = (2 * coordinates[0] > np.linalg.norm(coordinates, axis=0))
    if spherical_aberration:
        coordinates[1:] /= coordinates[0]
    return coordinates.T[small_angles, 1:], small_angles

def laue_visualizaton(lattice, k, dk, basis, form_factors, elastic=False):
    if np.array(lattice).shape != (2, 2):
        raise ValueError("Only 2D lattices are supported")
    k = np.array(k)
    reciprocal = lll(reciprocal_lattice_vectors(*lattice))[0]
    center = np.round(np.linalg.solve(reciprocal, k))
    r = np.linalg.norm(k) + dk/2
    r *= np.sqrt(np.max(np.diag(np.linalg.inv(reciprocal.T.dot(reciprocal)))))
    r = np.round(3 * r)
    points = np.mgrid[tuple(slice(i - r, i + r) for i in center)]
    points = points.reshape(len(center), -1).T
    point_coords = np.dot(points, reciprocal)
    where = laue_conditions(point_coords, k, dk, elastic)
    points, point_coords = np.array(points[where], dtype=int), point_coords[where]
    color = brightness(point_coords, basis, form_factors)

    data = [go.Scatter(x=point_coords[:, 0], y=point_coords[:, 1], 
                       text=[str(i) for i in points],
                       mode='markers', hoverinfo='text', 
                       marker=dict(size=5, color=color, cmin=0, cmax=max(color),
                                   colorscale="Greys",
                                   showscale=True, reversescale=True))]


    center1 = -k * (1 + elastic * dk / (2 * np.linalg.norm(k)))
    center2 = -k * (1 - elastic * dk / (2 * np.linalg.norm(k)))
    radius1 =  np.linalg.norm(k) + dk/2
    radius2 =  np.linalg.norm(k) - dk/2
    layout = go.Layout(
                    autosize=False,
                    width=600,
                    height=600,
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=True),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=True),
                    hovermode='closest',
                    shapes = [
                        {
                            'type': 'circle', 'xref': 'x', 'yref': 'y',
                            'x0': center1[0] - radius1,
                            'y0': center1[1] - radius1,
                            'x1': center1[0] + radius1,
                            'y1': center1[1] + radius1,
                            'line': {'color': 'blue'}},
                        {
                            'type': 'circle', 'xref': 'x', 'yref': 'y',
                            'x0': center2[0] - radius2,
                            'y0': center2[1] - radius2,
                            'x1': center2[0] + radius2,
                            'y1': center2[1] + radius2,
                            'line': {'color': 'blue'}},
                        {
                            'type': 'line',
                            'x0': 0,
                            'y0': 0,
                            'x1': -k[0],
                            'y1': -k[1],
                            'line': {
                                'color': 'blue',
                                'width': 4}}])

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)
    

def diffraction_pattern(lattice, k, dk, basis, form_factors, spherical_aberration=False, elastic=True):
    """Calculate diffraction pattern of a given lattice.
    
    Parameters:
    -----------
    lattice : array
        3x3 array of lattice vectors.
    k : array
        A vector of X-ray momentum.
    dk : float
        Spread in X-ray momentum (should be much smaller than k).
        
    Returns:
    --------
    data : array
        A 2D array with the coordinates of the diffraction pattern.
    """
    k = np.array(k)
    reciprocal = lll(reciprocal_lattice_vectors(*lattice))[0]
    center = np.round(np.linalg.solve(reciprocal, k))
    r = np.linalg.norm(k) + dk/2
    r /= np.sqrt(np.min(np.diag(np.linalg.inv(reciprocal.T.dot(reciprocal)))))
    r = np.round(3 * r)
    points = np.mgrid[tuple(slice(i - r, i + r) for i in center)]
    points = points.reshape(len(center), -1).T
    point_coords = np.dot(points, reciprocal)
    where = laue_conditions(point_coords, k, dk, elastic)
    points, point_coords = points[where], point_coords[where]
    image, indices = k_image(point_coords, k, spherical_aberration)
    projections, labels = image, points[indices]

    labels = np.array(labels, dtype=int)
    color = brightness(labels, basis, form_factors)
    data = [go.Scatter(x=projections.T[0], y=projections.T[1], text=[str(i) for i in labels],
                       mode='markers', hoverinfo='text', marker=dict(size=5, color=color,
                                                                     cmin=0, cmax=max(color),
                                                                     colorscale="Greys",
                                                                     showscale=True,
                                                                     reversescale=True))]
    layout = go.Layout(
            autosize=False,
            width=600,
            height=600,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True),
            hovermode='closest')
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, show_link=False)

def brightness(k, basis, form_factors):
    """Calculate the brightness of a point, based on the basis and form factors."""
    return np.abs(np.exp(2j * np.pi * k.dot(basis.T)).dot(form_factors))    

@store_last_correct
def validate_input2(lattice, k, basis, form_factors):
    lattice = np.array(eval(lattice))
    k = np.array(eval(k))
    basis = np.array(eval(basis))
    form_factors = np.array(eval(form_factors))
    assert lattice.shape == (2, 2)
    assert k.shape == (2,)
    assert len(basis.shape) == 2 and basis.shape[1] == 2
    assert form_factors.shape == (basis.shape[0],)
    return lattice, k, basis, form_factors


@interact(lattice='[[1, 0], [0, 1]]', k='(8, 8)', dk=(0., 4.), elastic=False,
          basis='[[0, 0], [0.5, 0.5]]', form_factors='[1, .5]')
def plot_laue(lattice, k, dk, elastic, basis, form_factors):
    lattice, k, basis, form_factors = validate_input2(lattice, k, basis, form_factors)

    laue_visualizaton(lattice, k, dk, basis, form_factors, elastic)

@store_last_correct
def validate_input(lattice, k, basis, form_factors):
    lattice = np.array(eval(lattice))
    k = np.array(eval(k))
    basis = np.array(eval(basis))
    form_factors = np.array(eval(form_factors))
    assert lattice.shape == (3, 3)
    assert k.shape == (3,)
    assert len(basis.shape) == 2 and basis.shape[1] == 3
    assert form_factors.shape == (basis.shape[0],)
    return lattice, k, basis, form_factors


@interact(lattice='[[1, 0, 0], [0, 1, 0], [0, 0, 1]]', 
          k='(4, 4, 4)', basis='[[0, 0, 0], [0.5, 0.5, 0.5]]', form_factors='[1, .5]',
          dk=(0., 2.), spherical_aberration=True, elastic=False)
def calculate_and_plot_diffraction(lattice, k, dk, basis, form_factors,
                                   spherical_aberration, elastic):
    lattice, k, basis, form_factors = validate_input(lattice, k, basis, form_factors)
    
    diffraction_pattern(lattice, k, dk, basis, form_factors,
                        spherical_aberration, elastic)



