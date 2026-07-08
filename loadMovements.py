from pathlib import Path
from loadFileToStruct import loadFileToStruct
import cupy as cp


def loadMovements(path, name):
    loadVariable = 'MovementsListreal'

    if name is None or name == '':
        name = 'Movements.mat'

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    out = loadFileToStruct(Path(path) / name, loadVariable)

    if out is None or not hasattr(out, loadVariable):
        raise ValueError(f"{loadFileToStruct.__module__}:variablesNotFound - Movement file does not contain required variable!")
    
    movements = getattr(out, loadVariable)

    if not isinstance(movements, cp.ndarray) or movements.ndim != 2:
        raise ValueError(f"'{loadVariable}' must be a 2D numeric array")
    
    return movements