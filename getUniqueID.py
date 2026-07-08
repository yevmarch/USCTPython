import scipy.io
import os
import warnings

def getUniqueID(filePath, fileName):
    # validate inputs
    if not isinstance(filePath, str) or not filePath:
        raise ValueError("filePath must be a non-empty string")
    if not isinstance(fileName, str) or not fileName:
        raise ValueError("fileName must be a non-empty string")

    try:
        fullPath = os.path.join(filePath, fileName)
        data = scipy.io.loadmat(fullPath)

        hardware = data.get('Hardware', None)
        # squeeze to scalar string if needed
        if hardware is not None:
            hardware = str(hardware).strip()

        if hardware is None or hardware.lower() != 'usct3dv2':
            metaData = data.get('MetaData', None)
            if metaData is None:
                raise ValueError("MetaData not found in file")
            uniqueID = str(metaData['MeasurementID'][0][0]).strip()
        else:
            uniqueID = 'none'

    except Exception as e:
        uniqueID = 'none'
        warnings.warn(f"Fetching unique ID not possible: {e}")

    return uniqueID