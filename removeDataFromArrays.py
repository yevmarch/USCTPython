import cupy as cp

def removeDataFromArrays(usedData, *varargin):

    if not isinstance(usedData, cp.ndarray) or usedData.ndim != 1:
        raise ValueError("'usedData' must be a numeric or boolean vector")

    varargout = []

    for arr in varargin:
        varargout.append(arr[:, usedData])

    return tuple(varargout)