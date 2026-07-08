from writeReconstructionLog import writeReconstructionLog

from types import SimpleNamespace
from pathlib import Path
import re
import scipy.io


def detectAScanLength(path, info, params):
    ascanLength = None

    if hasattr(info, 'NumberSamples'):
        ascanLength = info.NumberSamples
    else:
        path = Path(path)
        list_ = [item.name for item in path.iterdir() if item.is_dir()]

        tasRefFolder = None
        for item in list_:
            match = re.match(r'^(TAS\d+)', item)
            if match:
                tasRefFolder = match.group(1)
                break
        
        if tasRefFolder is None:
            writeReconstructionLog(f'Not able to detect Ascan length from data: folder starting with TAS not found under given path {path}.', 3)
            return

        path = Path(path) / tasRefFolder


        path = Path(path)
        list_ = [item.name for item in path.iterdir() if item.is_dir()]

        tasRotationRefFolder = None
        for item in list_:
            match = re.match(r'^(TASRotation\d+)', item)
            if match:
                tasRotationRefFolder = match.group(1)
                break

        if tasRotationRefFolder is None:
            writeReconstructionLog(f'Not able to detect Ascan length from data: folder starting with TASRotation not found under given path {path}.', 3)

        path = Path(path) / tasRotationRefFolder


        names = [item.name for item in Path(path).iterdir()]

        emitterFile = None
        for name in names:
            match = re.match(r'^(Emitter\d+)', name)
            if match:
                emitterFile = match.group(1)
                break

        if emitterFile is None:
            writeReconstructionLog(f'Not able to detect Ascan length from data: file starting with Emitter not found under given path {path}.', 3)

        path = Path(path) / emitterFile


        fileContent = scipy.io.whosmat(path)

        for name, shape, dtype in fileContent:
            if name.lower() == 'ascans':
                ascanLength = shape[0]
                break


    if(info.Bandpassundersampling):
        downsamplingFactor = params.aScanReconstructionFrequency / info.SampleRate
        ascanLength = ascanLength + downsamplingFactor
    

    return ascanLength