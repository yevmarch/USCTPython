from loadFileToStruct import loadFileToStruct
import cupy as cp

def loadChannelInfos(name):

    # predefine a variable
    loadVariable = 'HeadTable'

    if name is None or name == '':
        name = 'HeadTable-2011-01-30.mat'

    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    # load requested variable including the rotation matrix from file
    headTable = loadFileToStruct(name, loadVariable)

    # value check & formatting
    if headTable is None or not hasattr(headTable, loadVariable):
        raise ValueError(
            "loadHeadTable:variablesNotFound - "
            "Channel infos not contained in requested variable and file!")
    
    headTable = getattr(headTable, loadVariable)

    if not isinstance(headTable, cp.ndarray) or headTable.ndim != 2:
        raise ValueError(f"'{loadVariable}' must be a 2D numeric array")
    
    return headTable