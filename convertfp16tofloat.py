import cupy as cp

def convertfp16tofloat(input):
    inputSize = input.shape
    input_flat = input.ravel(order='F')
    input_u16 = input_flat.view(cp.uint16)
    
    exponent = (input_u16 >> cp.uint16(11)) & cp.uint16(15)
    
    input_i16 = input_u16.view(cp.int16)
    sign_bit = (input_i16 < 0).astype(cp.float64)
    
    fraction3 = (input_u16 & cp.uint16(2047)).astype(cp.uint32)
    
    cond = ((sign_bit == 1) & (exponent == 0)) | ((sign_bit == 0) & (exponent != 0))
    hidden_bit = cond.astype(cp.uint32) * cp.uint32(2048)
    
    offset_const = cp.uint32(2**32 - 1 - sum(2**i for i in range(12)))
    offset_term = offset_const * sign_bit.astype(cp.uint32)
    
    combined = (fraction3 + hidden_bit + offset_term).astype(cp.uint32)
    
    shift_amount = exponent.astype(cp.int64) - 1
    output_u32 = bitshift_uint32(combined, shift_amount)
    
    output_i32 = output_u32.view(cp.int32)
    output = output_i32.astype(cp.float64).reshape(inputSize, order='F')
    return output


def bitshift_uint32(x, k):
    #Mimics MATLAB's bitshift(x, k): k>0 -> left shift, k<0 -> right shift, 32-bit wraparound
    x64 = x.astype(cp.uint64)
    k_left = cp.where(k >= 0, k, 0).astype(cp.uint64)
    k_right = cp.where(k < 0, -k, 0).astype(cp.uint64)
    
    left = (x64 << k_left) & cp.uint64(0xFFFFFFFF)
    right = x64 >> k_right
    
    result = cp.where(k >= 0, left, right)
    return result.astype(cp.uint32)