import cupy as cp

def get3DRotationMatrix(d=None, dim=None):
    
    # if the variables inside the method are empty (do not exist), we should get an error
    if d is None or dim is None:
        raise ValueError("get3DRotationMatrix:IsEmpty - dimension or angle has no value !!")
    
    # the variables should not be arrays, so an error will be received
    # (cp.size() requires an actual cupy.ndarray -- unlike numpy's np.size(),
    # it doesn't accept plain Python scalars, so only call it on array inputs)
    if (isinstance(dim, cp.ndarray) and cp.size(dim) > 1) or (isinstance(d, cp.ndarray) and cp.size(d) > 1):
        raise ValueError("get3DRotationMatrix:InputOutOfBounds - Enter a Number!")
    
    # if variables are not given as numbers, we should get an error
    if not isinstance(dim, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")
    
    if not isinstance(d, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")
    
    # the rotation matrix method has two entries, angle(d) and dimension(dim = must be between 1-3)
    # that will get an error if they have wrong entries
    #
    # built by assigning individual entries into cp.eye(4) rather than
    # constructing from a nested Python list of cp.cos(d)/cp.sin(d) results:
    # cupy refuses to build an array from a list containing device arrays
    # ("Implicit conversion to a NumPy array is not allowed"), unlike numpy.
    cosD = cp.cos(d)
    sinD = cp.sin(d)
    rotationmatrix = cp.eye(4)

    if dim == 3:
        rotationmatrix[0, 0] = cosD
        rotationmatrix[0, 1] = -sinD
        rotationmatrix[1, 0] = sinD
        rotationmatrix[1, 1] = cosD
    elif dim == 2:
        rotationmatrix[0, 0] = cosD
        rotationmatrix[0, 2] = sinD
        rotationmatrix[2, 0] = -sinD
        rotationmatrix[2, 2] = cosD
    elif dim == 1:
        rotationmatrix[1, 1] = cosD
        rotationmatrix[1, 2] = -sinD
        rotationmatrix[2, 1] = sinD
        rotationmatrix[2, 2] = cosD
    else:
        raise ValueError(f"get3DRotationMatrix:InputOutOfBounds - Input Must Be (1, 2 or 3): {dim}")

    return rotationmatrix