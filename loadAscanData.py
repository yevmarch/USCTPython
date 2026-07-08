import cupy as cp
from pathlib import Path
from types import SimpleNamespace
from loadFileToStruct import loadFileToStruct
from compareUniqueIDs import compareUniqueIDs
from writeReconstructionLog import writeReconstructionLog

def loadAscanData(path, sl, sn, mp, hardware, rootUniqueID):

    # input validation
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")

    if not isinstance(sl, (int, float, cp.integer)) or sl != int(sl):
        raise ValueError("'sl' must be an integer scalar")

    if not isinstance(sn, (int, float, cp.integer)) or sn != int(sn):
        raise ValueError("'sn' must be an integer scalar")

    if not isinstance(mp, (int, float, cp.integer)) or mp != int(mp):
        raise ValueError("'mp' must be an integer scalar")

    if not isinstance(hardware, str) or not hardware:
        raise ValueError("'hardware' must be a non-empty string")

    if not isinstance(rootUniqueID, str) or not rootUniqueID:
        raise ValueError("'rootUniqueID' must be a non-empty string")
    
    # load requested variable from file
    filename = Path(path) / f'TAS{sl:03d}' / f'TASRotation{mp:02d}' / f'Emitter{sn:02d}.mat'

    if hardware.lower() == "usct3dv3":
        loadVariables = ['TASIndices', 'receiverIndices', 'AScans', 'Amplification', 'MeasurementID']
    else:
        loadVariables = ['TASIndices', 'receiverIndices', 'AScans', 'Amplification']


    # load data from file
    dataBlock = loadFileToStruct(filename, *loadVariables)

    # check variables
    if not all(hasattr(dataBlock, f) for f in loadVariables[:3]):
        raise ValueError(f"loadAScanFile:variablesNotFound - AScan data file {filename} does not contain required variables!")
    
    
    # special case for a single scan
    if dataBlock.AScans.ndim == 2 and dataBlock.AScans.shape[1] == 1:
        dataBlock.AScans = dataBlock.AScans.T
    expectedNumOfData = dataBlock.AScans.shape[1]

    if hardware.lower() == "usct3dv3":
        compareUniqueIDs(rootUniqueID, dataBlock.MeasurementID)

    
    if not isinstance(dataBlock.TASIndices, cp.ndarray) or dataBlock.TASIndices.shape != (expectedNumOfData, 1):
        raise ValueError(f"'TASIndices' must be size ({expectedNumOfData}, 1)")

    if not isinstance(dataBlock.receiverIndices, cp.ndarray) or dataBlock.receiverIndices.shape != (expectedNumOfData, 1):
        raise ValueError(f"'receiverIndices' must be size ({expectedNumOfData}, 1)")

    if not isinstance(dataBlock.AScans, cp.ndarray) or dataBlock.AScans.ndim != 2 or dataBlock.AScans.shape[1] != expectedNumOfData:
        raise ValueError(f"'AScans' must be a 2D array with {expectedNumOfData} columns")
    
    try:
        # Amplification: 1 value per Receiver, otherwise something is wrong....
        if not isinstance(dataBlock.Amplification, cp.ndarray) or dataBlock.Amplification.shape != (expectedNumOfData, 1):
            raise ValueError("'Amplification' must be size (expectedNumOfData, 1)")

    except Exception as ME:
        dataBlock.Amplification = cp.ones(dataBlock.TASIndices.shape)
        writeReconstructionLog(f'Amplification not found as expected in input file {filename} ({ME}). Using ones!', 3)


    return dataBlock
