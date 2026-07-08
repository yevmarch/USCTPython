import cupy as cp

def get3DRotationMatrix(d=None, dim=None):
    
    # if the variables inside the method are empty (do not exist), we should get an error
    if d is None or dim is None:
        raise ValueError("get3DRotationMatrix:IsEmpty - dimension or angle has no value !!")
    
    # the variables should not be arrays, so an error will be received
    if cp.size(dim) > 1 or cp.size(d) > 1:
        raise ValueError("get3DRotationMatrix:InputOutOfBounds - Enter a Number!")
    
    # if variables are not given as numbers, we should get an error
    if not isinstance(dim, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")
    
    if not isinstance(d, (int, float, cp.integer, cp.floating)):
        raise ValueError("get3DRotationMatrix:InputMustBeNumeric - Enter a Number!")
    
    # the rotation matrix method has two entries, angle(d) and dimension(dim = must be between 1-3)
    # that will get an error if they have wrong entries
    if dim == 3:
        rotationmatrix = cp.array([
            [cp.cos(d), -cp.sin(d), 0, 0],
            [cp.sin(d),  cp.cos(d), 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1]
        ])
    elif dim == 2:
        rotationmatrix = cp.array([
            [cp.cos(d),  0, cp.sin(d), 0],
            [0,          1, 0,         0],
            [-cp.sin(d), 0, cp.cos(d), 0],
            [0,          0, 0,         1]
        ])
    elif dim == 1:
        rotationmatrix = cp.array([
            [1, 0,          0,           0],
            [0, cp.cos(d), -cp.sin(d),   0],
            [0, cp.sin(d),  cp.cos(d),   0],
            [0, 0,          0,           1]
        ])
    else:
        raise ValueError(f"get3DRotationMatrix:InputOutOfBounds - Input Must Be (1, 2 or 3): {dim}")
    
    return rotationmatrix