import scipy.io
from types import SimpleNamespace

def loadFileToStruct(filePath, *varargin):
    # input check
    if not isinstance(filePath, str) or not filePath:
        raise ValueError("'filePath' must be a non-empty string")
    
    # load list of variables into structure array
    data = scipy.io.loadmat(filePath)
    out = SimpleNamespace()
    
    if varargin:
        for var in varargin:
            if var in data:
                setattr(out, var, data[var])
    else:
        # load all variables (excluding scipy metadata keys)
        for key, value in data.items():
            if not key.startswith('__'):
                setattr(out, key, value)
    
    return out