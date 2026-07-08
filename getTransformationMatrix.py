import cupy as cp
from writeReconstructionLog import writeReconstructionLog
from loadMovements import loadMovements
from getMotorPositionsForReconstruction import getMotorPositionsForReconstruction
from get3DRotationMatrix import get3DRotationMatrix
from loadRotationMatrix import loadRotationMatrix

def getTransformationMatrix(pathToMeasurement, files, motorPosList):
    
    # validate inputs
    if not isinstance(pathToMeasurement, str):
        raise ValueError("'pathToMeasurement' must be a string")
    if not hasattr(files, '__dict__') and not isinstance(files, dict):
        raise ValueError("'files' must be a struct-like object")
    motorPosList = cp.atleast_1d(motorPosList)
    if motorPosList.ndim != 1:
        raise ValueError("'motorPosList' must be a row vector")
    
    # getting transformation matrices of USCT aperture positions
    transformationMatrixList = cp.full((4, 4, int(cp.max(motorPosList))), cp.nan)
    
    # try to load movements from measurement
    try:
        movementRealAvailable = True
        movementsListReal = loadMovements(pathToMeasurement, files.movements)
        motorPosListAvailable = cp.arange(1, movementsListReal.shape[0] + 1)
        # take only motor positions really available
        motorPosList = cp.intersect1d(motorPosList, motorPosListAvailable).astype(cp.uint16)
        
        if cp.all(cp.isnan(movementsListReal)):
            # if only NaNs are stored -> fallback method via measurementRotation.mat
            movementRealAvailable = False
    
    except Exception as ME:
        writeReconstructionLog(f'Loading measured movements failed. {ME}', 3)
        movementRealAvailable = False
    
    if not movementRealAvailable:
        # if not definable by real movements, e.g. missing file, try to extract them
        motorPosList = getMotorPositionsForReconstruction(pathToMeasurement, motorPosList)
    
    motorPosListIntersect = motorPosList.copy()
    i = 0  # 0-based counter (was i = 1 in MATLAB)
    
    for mp in motorPosList:
        mp_idx = mp - 1  # convert to 0-based index for Python arrays
        
        # new: take the movementsListReal for calculation of rotations
        if movementRealAvailable:
            # extract movements & calculate rotation matrix
            if not cp.isnan(movementsListReal[mp_idx, 0]):
                transformationMatrixList[:, :, mp_idx] = get3DRotationMatrix(
                    movementsListReal[mp_idx, 0] / 180 * cp.pi, 3
                )
            else:
                # fallback if not recorded: identity
                transformationMatrixList[:, :, mp_idx] = cp.eye(4)
                # TODO: issue warning!
            
            if not cp.isnan(movementsListReal[mp_idx, 1]):
                transformationMatrixList[2, 3, mp_idx] = movementsListReal[mp_idx, 1]
            else:
                transformationMatrixList[2, 3, mp_idx] = 0
                # TODO: issue warning!
        
        else:
            # fallback: use old method, check if measurementRotation files are there
            try:
                transformationMatrixList[:, :, mp_idx] = loadRotationMatrix(pathToMeasurement, mp)
                i += 1
            except Exception:
                # if also this is not available: use identity
                transformationMatrixList[:, :, mp_idx] = cp.eye(4)
                # OLD:
                # If not possible reduce list
                # motorPosListIntersect = np.delete(motorPosListIntersect, i)
    
    # squeeze to max available mp
    if motorPosListIntersect.size > 0:
        transformationMatrixList = transformationMatrixList[:, :, :int(motorPosListIntersect[-1])]
    else:
        transformationMatrixList = cp.array([])
    
    return transformationMatrixList, motorPosListIntersect