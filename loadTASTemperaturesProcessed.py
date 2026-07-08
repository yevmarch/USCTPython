from pathlib import Path
from types import SimpleNamespace
from loadFileToStruct import loadFileToStruct

def loadTASTemperaturesProcessed(path, name):

    # predefine variables
    loadVariables = ['TemperatureModel4D', 'TASTemperature']

    # input validation
    if name is None or name == '':
        name = 'TASTempComp.mat'

    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")

    # load requested variable from file (TAS Temperature informations)
    temp = loadFileToStruct(Path(path) / name, *loadVariables)

    if not isinstance(temp, SimpleNamespace):
        raise ValueError("'Temperature' must be a struct-like object")
    
    return temp