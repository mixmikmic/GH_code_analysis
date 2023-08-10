import mxnet.ndarray as nd

X = nd.arange(24).reshape((3, 4, 2))

X

X[:, :, 0]

X[:, :, 1]

def unfold(tensor, mode):
    """Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.
    
    Parameters
    ----------
    tensor : ndarray
    mode : int, default is 0
           indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``
    
    Returns
    -------
    ndarray
        unfolded_tensor of shape ``(tensor.shape[mode], -1)``
    """
    return nd.reshape(nd.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

unfold(X, mode=0)

unfold(X, mode=1)

unfold(X, mode=2)

def fold(unfolded_tensor, mode, shape):
    """Refolds the mode-`mode` unfolding into a tensor of shape `shape`
    
    Parameters
    ----------
    unfolded_tensor : ndarray
        unfolded tensor of shape ``(shape[mode], -1)``
    mode : int
        the mode of the unfolding
    shape : tuple
        shape of the original tensor before unfolding
    
    Returns
    -------
    ndarray
        folded_tensor of shape `shape`
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return nd.moveaxis(nd.reshape(unfolded_tensor, full_shape), 0, mode)

unfolding = unfold(X, 1)
original_shape = X.shape
fold(unfolding, mode=1, shape=original_shape)

def mode_dot(tensor, matrix_or_vector, mode):
    """n-mode product of a tensor by a matrix at the specified mode.

    Parameters
    ----------
    tensor : ndarray
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector : ndarray
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode : int

    Returns
    -------
    ndarray
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector
    """
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    # tensor times vector case: make sure the sizes are correct 
    # (we are contracting over one dimension which then disappearas)
    if matrix_or_vector.ndim == 1: 
        if len(new_shape) > 1:
            new_shape.pop(mode)
            fold_mode -= 1
        else:
            new_shape = [1]

    # This is the actual operation: we use the equivalent formulation of the n-mode-product using the unfolding
    res = nd.dot(matrix_or_vector, unfold(tensor, mode))

    # refold the result into a tensor and return it 
    return fold(res, fold_mode, new_shape)

M = nd.arange(4*5).reshape((5, 4))
print(M.shape)

res = mode_dot(X, M, mode=1)

res.shape

v = nd.arange(4)
print(v.shape)

res = mode_dot(X, v, mode=1)

res.shape

