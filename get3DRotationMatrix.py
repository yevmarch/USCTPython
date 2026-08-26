import cupy as cp

def get3DRotationMatrix(d=None, dim=None):
    
    # if the variables inside the method are empty (do not exist), we should get an error
    if d is None or dim is None:
        raise ValueError("get3DRotationMatrix:IsEmpty - dimension or angle has no value !!")
    
    if (isinstance(dim, cp.ndarray) and cp.size(dim) > 1) or (isinstance(d, cp.ndarray) and cp.size(d) > 1):
        raise ValueError("get3DRotationMatrix:InputOutOfBounds - Enter a Number!")
    
    # if variables are not given as numbers, we should get an error
    if not isinstance(dim, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")
    
    if not isinstance(d, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")

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
