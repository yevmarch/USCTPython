import cupy as cp
from pathlib import Path
from types import SimpleNamespace
from loadFileToStruct import loadFileToStruct


def loadCEMeasured(path, name, measInfo):
    # predefine variables
    if measInfo.Hardware == 'USCT3Dv2':
        loadVariables = ['CEMeasured', 'CE_SF', 'CEOffset']
    else:  # change for CEMeasured format for USCT III
        loadVariables = ['CEMeasured', 'CE_SF', 'CEOffset', 'TASIndices', 'receiverIndices']

    # input checks
    if name is None or name == '':
        name = 'CEMeasured.mat'

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    # load
    ce = loadFileToStruct(Path(path) / name, *loadVariables)

    if ce is None or not all(hasattr(ce, var) for var in loadVariables):
        raise ValueError(
            "loadCE:variablesNotFound - CE measured file does not contain required variables!"
        )

    if not isinstance(ce.CEMeasured, cp.ndarray) or ce.CEMeasured.size == 0 or ce.CEMeasured.ndim != 2:
        raise ValueError("'ce.CEMeasured' must be a non-empty 2D numeric array")

    if not isinstance(ce.CE_SF, (int, float, cp.ndarray)) or cp.size(ce.CE_SF) != 1:
        raise ValueError("'ce.CE_SF' must be a non-empty numeric scalar")

    if not isinstance(ce.CEOffset, (int, float, cp.ndarray)) or cp.size(ce.CEOffset) != 1:
        raise ValueError("'ce.CEOffset' must be a non-empty numeric scalar")
    
    return ce