from loadFileToStruct import loadFileToStruct
from compareUniqueIDs import compareUniqueIDs
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

    def hasField(structArr, fieldname):
        return (hasattr(structArr, 'dtype')
                and structArr.dtype.names is not None
                and fieldname in structArr.dtype.names)

    if hardware.lower() == "usct3dv3":
        # resolve new data format with struct arrays

        # find out how many temperature values have been captured
        numTemps = 0
        if hasField(temp.JumoTemperature1, 'Temperature'):
            numTemps = temp.JumoTemperature1.shape[0]
        elif hasField(temp.JumoTemperature2, 'Temperature'):
            numTemps = temp.JumoTemperature2.shape[0]
        elif hasField(temp.JumoTemperature3, 'Temperature'):
            numTemps = temp.JumoTemperature3.shape[0]
        elif hasField(temp.JumoTemperature4, 'Temperature'):
            numTemps = temp.JumoTemperature4.shape[0]

        t = cp.full((4, numTemps), cp.nan)
        tStamps = cp.full((4, numTemps), cp.nan)

        def fillArray(out, row, structArr, fieldname):
            if hasField(structArr, fieldname):
                for idx in range(structArr.shape[0]):
                    val = structArr[fieldname][idx]
                    if cp.size(val) > 0:
                        out[row, idx] = cp.asarray(val).item()
                    else:
                        out[row, idx] = cp.nan

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