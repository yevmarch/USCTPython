from pathlib import Path
from types import SimpleNamespace
from loadFileToStruct import loadFileToStruct

def loadTASTemperatures(path, name):

    # predefine variables
    loadVariable = 'TASTemperature'

    # input validation
    if name is None or name == '':
        name = 'TASTemp.mat'

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    # load requested variable from file (TAS Temperature informations)
    temp = loadFileToStruct(Path(path) / name, *loadVariable)

    if temp is None or not hasattr(temp, loadVariable):
        raise ValueError(
            "loadTASTemperatures:variablesNotFound - "
            "TAS Temperature informations does not contain required variables!"
        )
    
    return temp