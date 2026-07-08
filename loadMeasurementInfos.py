import os
import scipy.io
from types import SimpleNamespace

from compareUniqueIDs import compareUniqueIDs

def loadMeasurementInfos(path, name=None, loadVariables=None, rootUniqueID=None):
    
    # default filename
    if name is None or name == '':
        name = 'info.mat'
    
    # validate inputs
    if not isinstance(path, str) or not path:
        raise ValueError("'path' must be a non-empty string")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")
    
    # default loadVariables
    if loadVariables is None or len(loadVariables) == 0:
        loadVariables = [
            'NumberSamples', 'Bandpassundersampling', 'SampleRate',
            'AScanDatatype', 'BeginDate', 'Hardware', 'HardwareVersion'
        ]
    else:
        if not isinstance(loadVariables, list):
            raise ValueError("'loadVariables' must be a list")
    
    fullPath = os.path.join(path, name)
    if not os.path.exists(fullPath):
        raise FileNotFoundError(f"File not found: {fullPath}")
    
    data = scipy.io.loadmat(fullPath)
    measInfo = SimpleNamespace()
    for var in loadVariables:
        if var in data:
            setattr(measInfo, var, data[var])
    
    if hasattr(measInfo, 'Hardware') and str(measInfo.Hardware).strip().lower() == 'usct3dv3':
        measurementID = str(data['MetaData']['MeasurementID'][0][0]).strip()
        compareUniqueIDs(rootUniqueID, measurementID)
        measInfo.MetaData = data['MetaData']
    
    # default AScanDatatype if missing
    if 'AScanDatatype' in loadVariables and not hasattr(measInfo, 'AScanDatatype'):
        measInfo.AScanDatatype = None
    
    # default EOffset if missing
    if not hasattr(measInfo, 'EOffset'):
        measInfo.EOffset = 0
    
    # default Wavelength if missing
    if not hasattr(measInfo, 'Wavelength'):
        measInfo.Wavelength = 3
    
    return measInfo