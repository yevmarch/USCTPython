from types import SimpleNamespace
import warnings
from loadCEMeasured import loadCEMeasured
from preprocessMeasuredCE import preprocessMeasuredCE
from phaseshift import phaseshift
from writeReconstructionLog import writeReconstructionLog
from loadCE import loadCE
from preprocessCE import preprocessCE

def getCEInfo(pathToData, files, measInfo, paramsPreprocessing):

    # validate inputs
    if not isinstance(pathToData, str):
        raise ValueError("'pathToData' must be a string")

    if not isinstance(files, SimpleNamespace):
        raise ValueError("'files' must be a struct-like object")

    if not isinstance(measInfo, SimpleNamespace):
        raise ValueError("'measInfo' must be a struct-like object")

    if not isinstance(paramsPreprocessing, SimpleNamespace):
        raise ValueError("'paramsPreprocessing' must be a struct-like object")
    
    ce = SimpleNamespace

    measuredCEAvailable = True

    try:
        # load and preprocess measured ce
        ceLoaded = loadCEMeasured(pathToData, files.ceMeasured, measInfo)
        ce.CE = preprocessMeasuredCE(ceLoaded, measInfo, paramsPreprocessing.aScanReconstructionFrequency, paramsPreprocessing.removeDCOffset)
        ce.CEOffset = ceLoaded.CEOffset
        ce.CE_SF = ceLoaded.CE_SF

        # new addition May 2023: compensate individual DACDelay for CEMeasured and A-Scans
        if hasattr(measInfo.MetaData.generateCE, 'DACDelay'):
            for idx in range(ce.CE.shape[1]):
                ce.CE[:, idx] = phaseshift(ce.CE[:, idx], -(measInfo.MetaData.generateCE.DACDelay - measInfo.MetaData.DACDelay) / measInfo.SampleRate, measInfo.SampleRate)

        if measInfo.Hardware.lower() == 'usct3dv3':
            ce.TASIndices = ceLoaded.TASIndices
            ce.receiverIndices = ceLoaded.receiverIndices

    except Exception as ME:
        warnings.warn(f"{type(ME).__name__}: Measured CE information not complete. {ME}.")
        writeReconstructionLog('Measured CE not available', 3)
        measuredCEAvailable = False

    
    ceAvailable = True
    try:
        # load ce
        ceLoaded = loadCE(pathToData, files.ce)
        ceLoaded.CE = preprocessCE(ceLoaded.CE, ceLoaded.CE_SF, paramsPreprocessing.aScanReconstructionFrequency, measInfo.expectedAScanLength)

        # write to output
        ce.CERef = ceLoaded.CE

        # new addition May 2023: compensate individual DACDelay for CE and A-Scans
        if hasattr(measInfo.MetaData.generateCE, 'DACDelay'):
            for idx in range(ce.CERef.shape[1]):
                ce.CERef[:, idx] = phaseshift(ce.CERef[:, idx], -(measInfo.MetaData.generateCE.DACDelay - measInfo.MetaData.DACDelay) / measInfo.SampleRate, measInfo.SampleRate)

        ce.CERefOffset = ceLoaded.CEOffset
        ce.CERef_SF = ceLoaded.CE_SF

    except Exception as ME:
        warnings.warn(f"{type(ME).__name__}: Measured CE information not complete. {ME}.")
        ceAvailable = False
        writeReconstructionLog('CE not available', 3)

    ce.measuredCEused = files.useCEMeasured.astype(bool)
    ce.measuredCEusedForCEcompensation = files.useCEMeasuredForCECompensation.astype(bool)
    ce.measuredCEavailable = measuredCEAvailable
    ce.ceAvailable = ceAvailable

    # attaching calibrated offsets from config file
    ce.offsetFilterEnabled = files.offsetFilterEnabled
    ce.offsetFilterDisabled = files.offsetFilterDisabled

    if ce.measuredCEused and not ce.measuredCEavailable:
        writeReconstructionLog('Selected to use CEMeasured, but CEMeasured is not available. Trying to use CE instead', 3)
        if ce.ceAvailable:
            ce.measuredCEused = False
        else:
            writeReconstructionLog('Neither CE nor CEMeasured available. Not able to continue.', 4)
            raise RuntimeError('Neither CE nor CEMeasured available. Not able to continue.')

    if not ce.ceAvailable and not ce.measuredCEused:
        writeReconstructionLog('Selected to use CE, but CE is not available. Trying to use CEMeasured instead', 3)
        if ce.measuredCEavailable:
            ce.measuredCEused = True
        else:
            writeReconstructionLog('Neither CE nor CEMeasured available. Not able to continue.', 4)
            raise RuntimeError('Neither CE nor CEMeasured available. Not able to continue.')
        

    return ce