import cupy as cp


def complex_to_float(x_complex):

    x_float = cp.zeros(x_complex.shape + (2,))
    x_float[... , 0] = cp.real(x_complex)
    x_float[... , 1] = cp.imag(x_complex)

    return x_float.astype(cp.float32)
