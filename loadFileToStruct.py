import os
import scipy.io
import numpy as np
import cupy as cp
from types import SimpleNamespace


def _normalize(value):
    # scipy.io.loadmat returns plain numpy arrays for every MATLAB variable,
    # including scalars (as 1x1 arrays) and char arrays -- but the rest of
    # this codebase expects native Python scalars for MATLAB scalars (many
    # functions validate inputs with isinstance(x, (int, float, ...))), plain
    # str for MATLAB char arrays, and cupy arrays for everything else (e.g.
    # loadRotationMatrix checks isinstance(result, cp.ndarray)). Normalize
    # here, once, for every caller of loadFileToStruct.
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.kind == 'U':
        return str(value.reshape(-1)[0]) if value.size == 1 else value
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return cp.asarray(value)


def loadFileToStruct(filePath, *varargin):
    # input check -- most callers build filePath as `Path(path) / name`, not a
    # plain str, so accept any os.PathLike (pathlib.Path included) as well
    if not isinstance(filePath, (str, os.PathLike)) or not str(filePath):
        raise ValueError("'filePath' must be a non-empty string")
    filePath = str(filePath)

    # load list of variables into structure array
    data = scipy.io.loadmat(filePath)
    out = SimpleNamespace()

    if varargin:
        for var in varargin:
            if var in data:
                setattr(out, var, _normalize(data[var]))
    else:
        # load all variables (excluding scipy metadata keys)
        for key, value in data.items():
            if not key.startswith('__'):
                setattr(out, key, _normalize(value))

    return out