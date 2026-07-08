import cupy as cp
import warnings

def temperatureToSoundSpeed(temperature, method='MARCZAK'):

    if temperature is None or cp.size(temperature) == 0:
        return cp.array([])

    # if temperature are not given as numbers, we should get an error
    try:
        temperature = cp.asarray(temperature, dtype=cp.float64)
    except (ValueError, TypeError):
        raise ValueError("temperatureToSoundSpeed:InputMustBeNumeric - Temperature values should be numeric.")
    

    # select method, calculate values
    method_lower = method.lower()

    if method_lower == 'mader':
        # Bilaniuk and Wong/Mader 1972 - 36 point equation
        k = [3.14643e-9, -1.478e-6, 0.000334199, -0.0580852, 5.03711, 1402.39]
        speed = cp.polyval(cp.array(k), temperature)
        if cp.any(temperature > 100) or cp.any(temperature < 0):
            warnings.warn("temperatureToSoundSpeed:Mader_InputOutOfBounds - Number OutOfBounds!!")

    elif method_lower == 'marczak':
        k = [2.78786e-9, -1.398845e-6, 3.287156e-4, -5.799136e-2, 5.038813, 1.402385e3]
        speed = cp.polyval(cp.array(k), temperature)
        if cp.any(temperature > 95) or cp.any(temperature < 0):
            warnings.warn("temperatureToSoundSpeed:Marczak_InputOutOfBounds - Number OutOfBounds!!")

    elif method_lower == 'jan':  # schiffer
        speed = 1557 - 0.0245 * (74 - temperature)**2
        if cp.any(temperature > 45) or cp.any(temperature < 15):
            warnings.warn("temperatureToSoundSpeed:Jan_InputOutOfBounds - Number OutOfBounds!!")

    else:
        raise ValueError(f"temperatureToSoundSpeed:UnknownMethod - Unknown method {method} to compute sound speed.")

    return speed
    
