import cupy as cp
from pathlib import Path
from loadFileToStruct import loadFileToStruct

def loadRotationMatrix(path, rotationNumber):
    loadVariable = 'rotationMatrix'

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")

    if not isinstance(rotationNumber, (int, float, cp.integer, cp.floating)) or not cp.isfinite(rotationNumber):
        raise ValueError("'rotationNumber' must be a finite scalar number")
    

    # load requested variable including the rotation matrix from file
    fileName = f'measurementRotation{rotationNumber:02d}.mat'
    out = loadFileToStruct(Path(path) / fileName)

    # value check & formatting
    rotationMatrix = getattr(out, loadVariable)

    if not isinstance(rotationMatrix, cp.ndarray) or rotationMatrix.shape != (4, 4) or not cp.all(cp.isfinite(rotationMatrix)):
        raise ValueError(f"'{loadVariable}' must be a finite 4x4 numeric array")
    
    return rotationMatrix
