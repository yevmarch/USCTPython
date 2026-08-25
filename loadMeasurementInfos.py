import os
import scipy.io
from types import SimpleNamespace

from compareUniqueIDs import compareUniqueIDs
from lenientMatStruct import load_struct

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
    
    # loaded without the (large, nested) MetaData variable: some info.mat files
    # (e.g. written by Octave) contain a MetaData struct that scipy's mio5
    # reader cannot parse ("buffer is too small for requested array") even
    # though the file itself is intact -- see lenientMatStruct.py. MetaData is
    # loaded separately below, only when actually needed.
    data = scipy.io.loadmat(fullPath, variable_names=loadVariables)
    measInfo = SimpleNamespace()
    for var in loadVariables:
        if var in data:
            setattr(measInfo, var, data[var])

    hardware = getattr(measInfo, 'Hardware', None)
    if hasattr(hardware, 'reshape'):
        hardware = hardware.reshape(-1)[0]

    if hardware is not None and str(hardware).strip().lower() == 'usct3dv3':
        # MetaData is read via the lenient parser (not scipy.io.loadmat) so it
        # comes back as dotted-attribute SimpleNamespace objects, which is the
        # access style downstream code (estimateOffset, getCEInfo) expects
        # (e.g. measInfo.MetaData.generateCE.DACDelay) -- and it sidesteps
        # scipy's read failure on this struct.
        metaData = load_struct(fullPath, variable_names=['MetaData'])['MetaData']
        measurementID = str(metaData.MeasurementID).strip()
        compareUniqueIDs(rootUniqueID, measurementID)
        measInfo.MetaData = metaData
    
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