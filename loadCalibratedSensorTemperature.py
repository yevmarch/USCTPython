from loadFileToStruct import loadFileToStruct
from compareUniqueIDs import compareUniqueIDs
from types import SimpleNamespace
import cupy as cp
from pathlib import Path

def loadCalibratedSensorTemperature(path, name, rootUniqueID, hardware):
    fullPath = Path(path) / name

    if hardware.lower() == "usct3dv3":
        # definition of variables
        loadVariables = ['JumoTemperature1', 'JumoTemperature2', 'JumoTemperature3', 'JumoTemperature4', 'MeasurementID']
        # load to struct
        temp = loadFileToStruct(fullPath, *loadVariables)
        compareUniqueIDs(rootUniqueID, temp.MeasurementID)
    else:
        # definition of variables
        loadVariables = ['JumoTemperature1', 'JumoTemperature2', 'TimeStamps']
        # load to struct
        temp = loadFileToStruct(fullPath, *loadVariables)

    if temp is None or not all(hasattr(temp, var) for var in loadVariables):
        raise ValueError(
            f"loadCalibratedSensorTemperature:variablesNotFound - "
            f"Jumo temperature values in file {fullPath} not in correct format!"
        )

    # loadFileToStruct normalizes a MATLAB struct array into a SimpleNamespace
    # (single element) or a list of SimpleNamespace (multiple elements) --
    # these helpers treat a bare SimpleNamespace as a length-1 sequence so
    # both cases can be handled uniformly below.
    def hasField(structVal, fieldname):
        elem = structVal[0] if isinstance(structVal, list) else structVal
        return isinstance(elem, SimpleNamespace) and hasattr(elem, fieldname)

    def structLen(structVal):
        return len(structVal) if isinstance(structVal, list) else 1

    def structElem(structVal, idx):
        return structVal[idx] if isinstance(structVal, list) else structVal

    def hasValue(x):
        if isinstance(x, (int, float)):
            return True
        return isinstance(x, cp.ndarray) and x.size > 0

    def scalarValue(x):
        return x if isinstance(x, (int, float)) else cp.asarray(x).item()

    if hardware.lower() == "usct3dv3":
        # resolve new data format with struct arrays

        # find out how many temperature values have been captured
        numTemps = 0
        if hasField(temp.JumoTemperature1, 'Temperature'):
            numTemps = structLen(temp.JumoTemperature1)
        elif hasField(temp.JumoTemperature2, 'Temperature'):
            numTemps = structLen(temp.JumoTemperature2)
        elif hasField(temp.JumoTemperature3, 'Temperature'):
            numTemps = structLen(temp.JumoTemperature3)
        elif hasField(temp.JumoTemperature4, 'Temperature'):
            numTemps = structLen(temp.JumoTemperature4)

        t = cp.full((4, numTemps), cp.nan)
        tStamps = cp.full((4, numTemps), cp.nan)

        def fillArray(out, row, structVal, fieldname):
            if hasField(structVal, fieldname):
                for idx in range(structLen(structVal)):
                    val = getattr(structElem(structVal, idx), fieldname)
                    out[row, idx] = scalarValue(val) if hasValue(val) else cp.nan

        # temperatures
        fillArray(t, 0, temp.JumoTemperature1, 'Temperature')
        fillArray(t, 1, temp.JumoTemperature2, 'Temperature')
        fillArray(t, 2, temp.JumoTemperature3, 'Temperature')
        fillArray(t, 3, temp.JumoTemperature4, 'Temperature')

        # same for time stamps
        fillArray(tStamps, 0, temp.JumoTemperature1, 'TimeStamp')
        fillArray(tStamps, 1, temp.JumoTemperature2, 'TimeStamp')
        fillArray(tStamps, 2, temp.JumoTemperature3, 'TimeStamp')
        fillArray(tStamps, 3, temp.JumoTemperature4, 'TimeStamp')

    else:
        t = cp.vstack([temp.JumoTemperature1, temp.JumoTemperature2])
        tStamps = temp.TimeStamps

    return t, tStamps
