import cupy as cp
import warnings

def determineOptimalPulse(imgResolution, optPulseFactor, timeInterval, expectedAScanLength):
    minOptPulse = cp.ceil(8 * imgResolution / timeInterval / 1500)

    if optPulseFactor == -1:
        optPulseFactor = minOptPulse
        print(f'Optimal pulse automatically set to {minOptPulse}')
    else:
        if optPulseFactor < minOptPulse:
            warnings.warn('Optimal pulse too small for resolution')

    # convolute the result with optimal pulse of certain width
    desSamplingFreq = 0.5 / timeInterval
    sincFact = optPulseFactor
    sincLength = cp.round(11 * sincFact / 16)
    sincLength = 3 * sincLength

    sincT = timeInterval * cp.arange((-sincLength + timeInterval) / 2, sincLength / 2 + 1)
    sincFact = sincFact / 2
    sincPeak = (2 * cp.sqrt(cp.pi) * (desSamplingFreq / sincFact)**3 * (1 - 2 * (cp.pi * desSamplingFreq / sincFact * sincT)**2) *cp.exp(-(cp.pi * desSamplingFreq / sincFact * sincT)**2))

    sincPeak = sincPeak / cp.max(sincPeak)
    sincPeak_len = len(sincPeak)

    # for fourier-based convolution shift
    # sincPeak (a short kernel) needs to be zero-padded out to
    # expectedAScanLength before the FFT below, since sincPeak_ft is later
    # multiplied elementwise against the full-length A-scan spectrum --
    # MATLAB's sincPeak(expectedAScanLength) = 0 grows the array
    # automatically; Python doesn't, so pad explicitly. -1 for 0-based indexing.
    if len(sincPeak) < expectedAScanLength:
        sincPeak = cp.concatenate([sincPeak, cp.zeros(expectedAScanLength - len(sincPeak))])
    sincPeak[expectedAScanLength - 1] = 0
    sincPeak = cp.roll(sincPeak, -int(cp.floor(sincPeak_len / 2)))
    sincPeak_ft = cp.fft.fft(sincPeak)

    return sincPeak_ft, optPulseFactor