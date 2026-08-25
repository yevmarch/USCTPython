import scipy.io
import os
import warnings

from lenientMatStruct import load_struct

def getUniqueID(filePath, fileName):
    # validate inputs
    if not isinstance(filePath, str) or not filePath:
        raise ValueError("filePath must be a non-empty string")
    if not isinstance(fileName, str) or not fileName:
        raise ValueError("fileName must be a non-empty string")

    try:
        fullPath = os.path.join(filePath, fileName)
        # only read the cheap top-level Hardware field via scipy; MetaData is
        # read separately (and only if needed) via the lenient parser, since
        # scipy's mio5 reader fails on the MetaData struct in some info.mat
        # files (e.g. written by Octave) even when the file isn't corrupted.
        data = scipy.io.loadmat(fullPath, variable_names=['Hardware'])

        hardware = data.get('Hardware', None)
        # squeeze to scalar string if needed
        if hasattr(hardware, 'reshape'):
            hardware = hardware.reshape(-1)[0]
        if hardware is not None:
            hardware = str(hardware).strip()

        if hardware is None or hardware.lower() != 'usct3dv2':
            metaData = load_struct(fullPath, variable_names=['MetaData']).get('MetaData', None)
            if metaData is None:
                raise ValueError("MetaData not found in file")
            uniqueID = str(metaData.MeasurementID).strip()
        else:
            uniqueID = 'none'

    except Exception as e:
        uniqueID = 'none'
        warnings.warn(f"Fetching unique ID not possible: {e}")

    return uniqueID