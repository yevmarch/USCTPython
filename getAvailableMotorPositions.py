import re
from pathlib import Path
import warnings
from writeReconstructionLog import writeReconstructionLog

def getAvailableMotorPositions(path):
    
    # validate input
    if not isinstance(path, str):
        raise ValueError("'path' must be a string")
    
    # looking for possible rotation folders in first TAS folder
    path = Path(path) / 'TAS001'
    if not path.is_dir():
        raise ValueError(f"{getAvailableMotorPositions.__module__}:wrongPath - Required folder 'TAS001' does not exist {path}.")
    
    # list directory entries that are folders
    entries = [item.name for item in path.iterdir() if item.is_dir()]
    
    pos = []
    # extract numbers
    for elem in entries:
        nn = re.findall(r'(?<=^TASRotation)\d+$', elem)
        if len(nn) == 1:
            pos.append(float(nn[0]))
    
    # sort ascending
    pos.sort()
    
    # check if something valid was found
    if not pos:
        writeReconstructionLog('No motor positions found for this measurement.', 3)
        warnings.warn(f"NoPositions: No motor positions found under {path}.")
    
    return pos