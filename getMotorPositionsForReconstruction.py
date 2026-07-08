import cupy as cp
from getAvailableMotorPositions import getAvailableMotorPositions

def getMotorPositionsForReconstruction(path, motorPos):
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")
    motorPos = cp.atleast_1d(motorPos)
    if motorPos.ndim != 1 or not cp.issubdtype(motorPos.dtype, cp.integer):
        raise ValueError("'motorPos' must be a row vector of integers")

    availableMotorPos = getAvailableMotorPositions(path)

    # take only motor positions really available
    motorPos = cp.intersect1d(motorPos, availableMotorPos).astype(cp.uint16)

    return motorPos