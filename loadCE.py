from loadFileToStruct import loadFileToStruct
from types import SimpleNamespace
from pathlib import Path
import cupy as cp

def loadCE(path, name):
    
    # predefine variables
    loadVariables = ['CE', 'CE_SF', 'CEOffset']

    # input check
    if name is None or name == '':
        name = "CE.mat"

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    # load
    ce = loadFileToStruct(Path(path) / name, *loadVariables)

    if ce is None or not all(hasattr(ce, var) for var in loadVariables):
        raise ValueError('CE file does not contain required variables!')
    

    # fix for legacy data
    if ce.CE.ndim == 1 or ce.CE.shape[0] == 1:
        ce.CE = ce.CE.reshape(-1, 1)


    if not (isinstance(ce.CE, cp.ndarray) and ce.CE.size > 0 and ce.CE.shape[1] == 1):
        raise ValueError("CE must be a non-empty numeric column vector")
    
    if not (isinstance(ce.CE_SF, cp.ndarray) and ce.CE_SF.size > 0 and cp.asarray(ce.CE_SF).size == 1):
        raise ValueError("CE_SF must be a non-empty numeric column scalar")
    
    if not (isinstance(ce.CEOffset, cp.ndarray) and ce.CEOffset.size > 0 and cp.asarray(ce.CEOffset).size == 1):
        raise ValueError("CEOffset must be a non-empty numeric column scalar")
    
    return ce
