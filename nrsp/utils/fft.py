import numpy as np
from numpy.typing import NDArray
from typing import Union, Literal, List, Sequence
from scipy.sparse import csr_matrix
import nrsp.utils.log

def digit_reverse_indices_radix2(length: int) -> NDArray[int]:
    """Compute radix-2 (bit) reversed indices for length = 2^n."""
    n_bits = int(np.log2(length))
    indices = np.arange(length)
    reversed_indices = np.zeros(length, dtype=int)
    for i in range(n_bits):
        reversed_indices |= ((indices >> i) & 1) << (n_bits - 1 - i)
    return reversed_indices

def digit_reverse_indices_radix4(length: int) -> NDArray[int]:
    """Compute radix-4 digit-reversed indices for length = 4^n."""
    n_quads = int(np.log(length) / np.log(4))
    indices = np.arange(length)
    reversed_indices = np.zeros(length, dtype=int)
    for layer in range(n_quads):
        # Extract 2-bit digit at position layer
        digit = (indices >> (2 * layer)) & 0b11
        # Place digit reversed in position from the end
        reversed_indices |= digit << (2 * (n_quads - 1 - layer))
    return reversed_indices

def digit_reverse(
    data: NDArray[np.complex128],
    base: int,
    axis: Union[int, Sequence[int], None] = None
) -> NDArray[np.complex128]:
    """
    Perform digit reversal along one or more axes of a complex array.

    Args:
        data: Complex input array.
        base: Base of the digit reversal (only 2 or 4 supported).
        axis: 
            - int: Apply digit reversal along this axis.
            - sequence of ints: Apply digit reversal along these axes.
            - None: Apply digit reversal over all axes.

    Returns:
        Array with digit-reversed elements along specified axes.
    """
    if base not in (2, 4):
        raise ValueError("Only radix-2 or radix-4 supported.")

    data = np.asarray(data)

    if axis is None:
        axes = range(data.ndim)
    elif isinstance(axis, int):
        axes = [axis]
    else:
        axes = list(axis)

    for ax in axes:
        ax = np.core.numeric.normalize_axis_index(ax, data.ndim)
        length = data.shape[ax]
        if base == 2:
            indices = digit_reverse_indices_radix2(length)
        else:
            indices = digit_reverse_indices_radix4(length)
        data = np.take(data, indices, axis=ax)

    return data

#def digit_reverse(data: NDArray[np.complex128], base: int, axis:int=None) -> NDArray[np.complex128]:
#    """Digit reverse 1D or 2D complex array with radix-2 or radix-4."""
#    if base not in (2, 4):
#        raise ValueError("Only radix-2 or radix-4 supported.")
#    if data.ndim == 1 or axis is not None:
#        if axis is not None:
#            length = data.shape[axis]
#        else:
#            length = data.shape[0]
#        if base == 2:
#            indices = digit_reverse_indices_radix2(length)
#        else:
#            indices = digit_reverse_indices_radix4(length)
#        return data[indices]
#    elif data.ndim == 2:
#        n0, n1 = data.shape
#        if base == 2:
#            indices_0 = digit_reverse_indices_radix2(n0)
#            indices_1 = digit_reverse_indices_radix2(n1)
#        else:
#            indices_0 = digit_reverse_indices_radix4(n0)
#            indices_1 = digit_reverse_indices_radix4(n1)
#
#        data_reordered = data[:, indices_1]
#        data_reordered = data_reordered[indices_0, :]
#        return data_reordered
#        #return data[np.ix_(indices_0, indices_1)]
#    else:
#        raise ValueError("digit_reverse only supports 1D or 2D arrays.")
    
#def _tf_vector(layer: int, n_samples: int, radix: int = 2) -> NDArray[np.complex128]:
#    n_layers = int(np.log2(n_samples) / np.log2(radix))
#    group_size = radix ** (layer + 1)
#    n_groups = n_samples // group_size
#
#    tf = np.ones(n_samples, dtype=np.complex128)
#
#    for r in range(1, radix):  # skip r=0
#        for i in range(radix ** layer):
#            for j in range(n_groups):
#                idx = j * group_size + r * (radix ** layer) + i
#                k = r * i * n_groups
#                tf[idx] *= twiddle_factor(k, n_samples)
#
#    return tf

def twiddle_factor(
    k: Union[int, NDArray[np.int_]], N: int, type: Literal['cos', 'sin', 'exp'] = 'exp'
) -> NDArray[np.complex128]:
    """Generate twiddle factors of given type."""
    if type == 'cos':
        return np.cos(2 * np.pi * k / N)
    elif type == 'sin':
        return np.sin(2 * np.pi * k / N)
    elif type == 'exp':
        # Standard FFT twiddle factor: W_N^k = exp(-2*pi*i*k / N)
        return np.exp(-2j * np.pi * k / N)
    else:
        raise ValueError("Invalid twiddle factor type.")



def tf_vector_radix_dif(layer: int, n_samples: int, radix: int = 2):
    """
    Generate the twiddle factor vector for a given layer in a Radix-2 or Radix-4 FFT.

    This function computes the complex exponential "twiddle factors" used in FFT algorithms
    (Fast Fourier Transform) for a specific layer of the transform. These are the coefficients
    applied to FFT outputs (in DIF) or inputs (in DIT) to mix frequency components.

    The generated vector has length `n_samples`, with twiddle values arranged according to
    the required index pattern for a decimation-in-frequency (DIF) radix-based FFT.

    Parameters:
    ----------
    layer : int
        The index of the FFT layer (0-based). Layer 0 is the first butterfly layer applied.
    n_samples : int
        Total number of FFT input samples. Must be an integer power of `radix`.
    radix : int, optional
        The radix of the FFT algorithm. Supported values are 2 and 4. Default is 2.

    Returns:
    -------
    np.ndarray
        A 1D NumPy array of complex128 twiddle factors of shape (n_samples,).

    Raises:
    ------
    ValueError
        If `n_samples` is not a power of `radix`, or if the `radix` is not supported.

    Notes:
    -----
    - In radix-2, the twiddle factors are computed as:
          W_N^k = exp(-2πj * k / N)
      where `k` is determined by the butterfly block and position within that block.
    - The index pattern `(n * c)` used for `k` ensures correct grouping across layers.
    - For DIT usage, the output may need to be reshaped or reordered depending on how 
      twiddles are applied relative to the butterfly structure.

    Examples:
    --------
    >>> tf_vector_radix(layer=0, n_samples=8, radix=2)
    array([ 1.000+0.j      ,  1.000+0.j      ,  1.000+0.j      ,  1.000+0.j      ,
            1.000+0.j      ,  1.000+0.j      ,  1.000+0.j      ,  1.000+0.j      ])

    >>> tf_vector_radix(layer=1, n_samples=8, radix=2)
    array([ 1.000+0.j      ,  0.707-0.707j  ,  1.000+0.j      ,  0.707-0.707j  ,
            1.000+0.j      ,  0.707-0.707j  ,  1.000+0.j      ,  0.707-0.707j  ])
    """
    # Validate radix and sample size
    n_layers = int(np.log(n_samples) / np.log(radix))
    if not np.isclose(radix ** n_layers, n_samples):
        raise ValueError(f"n_samples={n_samples} must be a power of radix={radix}.")

    if radix == 2:
        n = np.tile(np.arange(radix**(n_layers - layer - 1)), radix**(layer + 1))
        c = np.tile(
            np.repeat(np.arange(0, radix**(layer + 1), radix**layer), radix**(n_layers - layer - 1)),
            radix**layer,
        )
    elif radix == 4:
        n = np.tile(np.arange(radix**(n_layers - layer - 1)), radix**(layer + 1))
        c = np.tile(
            np.repeat(np.arange(0, radix**(layer + 1), radix**layer), radix**(n_layers - layer - 1)),
            radix**layer,
        )
    else:
        raise ValueError(f"Radix {radix} not supported. Use radix=2 or radix=4.")

    # Compute twiddle factors
    k = n * c
    tf = np.exp(-2j * np.pi * k / n_samples)

    # Sanity check
    assert len(tf) == n_samples, f"Twiddle vector length {len(tf)} != n_samples {n_samples}"

    return tf

import numpy as np

def tf_vector_radix_dit(layer: int, n_samples: int, radix: int = 2) -> np.ndarray:
    """
    Generate a layer-wise twiddle factor vector for Radix-2 or Radix-4 DIT FFT
    constructed from first principles (no DIF dependency).

    Parameters
    ----------
    layer : int
        Layer index (0-based).
    n_samples : int
        FFT size (power of radix).
    radix : int, optional
        Radix of FFT, default 2.

    Returns
    -------
    np.ndarray
        1D complex twiddle factor vector of length n_samples, for DIT application.
    """
    # Validate n_samples is power of radix
    n_layers = int(np.log(n_samples) / np.log(radix))
    if not np.isclose(radix ** n_layers, n_samples):
        raise ValueError(f"n_samples={n_samples} must be a power of radix={radix}.")

    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Layer index must be in [0, {n_layers - 1}].")

    # Number of butterflies in this layer
    butterflies_per_group = radix
    groups_per_layer = n_samples // (radix ** (layer + 1))

    # Stride between twiddle factor indices in the complex exponential
    stride = n_samples // (radix ** (layer + 1))

    # Initialize twiddle vector
    twiddle_vector = np.empty(n_samples, dtype=np.complex128)

    # Fill twiddle factors per butterfly group and position
    block_size = radix ** (layer + 1)
    n_blocks = n_samples // block_size

    for block in range(n_blocks):
        block_start = block * block_size
        # For each butterfly input in this block
        for pos in range(block_size):
            # Calculate twiddle exponent index k:
            # The twiddle index resets every butterfly (radix)
            k = (pos % radix) * stride
            twiddle_vector[block_start + pos] = np.exp(2j * np.pi * k / n_samples)

    return twiddle_vector



def create_1d_radix2_dif_layer(
    layer: int, n_samples: int
) -> NDArray[np.complex128]:
    """
    Construct the transformation matrix for a single layer of the 
    1D Radix-2 Decimation-In-Frequency (DIF) FFT algorithm.

    Each layer of a Radix-2 DIF FFT performs:
      1. Butterfly operations (2x2 units that combine pairs of elements),
      2. Followed by multiplication with layer-specific twiddle factors.

    The full FFT is composed by chaining these sparse linear transformations.

    Args:
        layer (int): Index of the FFT layer (0 is the first FFT stage).
        n_samples (int): Total number of input samples (must be a power of 2).

    Returns:
        np.ndarray: A dense complex-valued matrix of shape (n_samples, n_samples)
                    representing the linear transformation for this FFT stage.
    
    Matrix construction detail:
        - The full layer matrix is built as: M = G ⊗ butterfly ⊗ B
          where:
            G: Identity of size 2^layer (determines how many blocks),
            B: Identity of size N / 2^(layer+1) (size of each block),
            butterfly: 2x2 matrix for pairwise sum/difference.

        - Twiddle factors (complex phase rotations) are applied to each row
          after the butterfly step.

    Raises:
        AssertionError: If `layer` is out of valid range.
    """

    radix = 2
    n_layers = int(np.log2(n_samples))
    assert 0 <= layer < n_layers, "Layer index out of range"

    # Butterfly matrix for radix-2
    butterfly = np.array([[1, 1], [1, -1]], dtype=np.complex128)

    # Identity matrices for Kronecker product
    G = np.eye(radix ** layer, dtype=np.complex128)
    B = np.eye(n_samples // (radix ** (layer + 1)), dtype=np.complex128)

    # Construct layer matrix: G ⊗ butterfly ⊗ B
    M = np.kron(G, np.kron(butterfly, B))

    # Compute twiddle factor vector for this layer
    tf = tf_vector_radix_dif(layer, n_samples, radix=2)

    # Multiply each row by the twiddle factor (twiddle factor matrix is diagonal)
    M = tf[:, np.newaxis] * M

    return M

def create_1d_radix2_dit_layer(layer: int, n_samples: int) -> NDArray[np.complex128]:
    """
    Construct the matrix for one Radix-2 DIT FFT layer.

    Args:
        layer: The FFT layer index (0-based from input side).
        n_samples: Total number of input samples (must be power of 2).

    Returns:
        A dense ndarray representing the layer transformation matrix.
    """

    radix = 2
    n_layers = int(np.log2(n_samples))
    assert 0 <= layer < n_layers, "Layer index out of range"

    # Radix-2 butterfly operation (before twiddle factors)
    butterfly = np.array([[1, 1], [1, -1]], dtype=np.complex128)

    # Outer structure of the Kronecker product
    G = np.eye(n_samples // (radix ** (layer + 1)), dtype=np.complex128)
    B = np.eye(radix ** layer, dtype=np.complex128)

    # Apply butterflies first: G ⊗ butterfly ⊗ B
    M = np.kron(G, np.kron(butterfly, B))

    # Twiddle factors are applied AFTER the butterfly in DIT
    tf = tf_vector_radix_dit(layer, n_samples, radix=2)

    # Full layer matrix = butterfly_layer × T (twiddles first, then butterflies)
    M = M * tf[np.newaxis, :]  # Elementwise multiply each column by twiddle

    return M


def create_2d_radix2_dif_layer(
    layer: int, n_samples_0: int, n_samples_1: int
) -> NDArray[np.complex128]:
    """
    Construct a single layer of the 2D Radix-2 Decimation-In-Frequency (DIF) FFT 
    as a Kronecker product of 1D FFT layer matrices.

    This function builds a sparse matrix representing one stage of the 2D FFT.
    Each 2D layer is the Kronecker product of the current layer along each axis.
    If the layer exceeds the number of stages along one axis, identity is used 
    for that axis (i.e., no transform applied in that direction at this stage).

    Args:
        layer (int): Index of the current FFT layer (0-based).
        n_samples_0 (int): Number of samples along the first (row) dimension.
        n_samples_1 (int): Number of samples along the second (column) dimension.

    Returns:
        NDArray[np.complex128]: A (n_samples_0 * n_samples_1) x 
        (n_samples_0 * n_samples_1) complex-valued matrix representing the 
        transformation for the given layer in the 2D Radix-2 DIF FFT.
    """
    n_layers_0 = int(np.log2(n_samples_0))
    n_layers_1 = int(np.log2(n_samples_1))

    if layer < n_layers_0:
        M_0 = create_1d_radix2_dif_layer(layer, n_samples_0)
    else:
        M_0 = np.eye(n_samples_0, dtype=np.complex128)

    if layer < n_layers_1:
        M_1 = create_1d_radix2_dif_layer(layer, n_samples_1)
    else:
        M_1 = np.eye(n_samples_1, dtype=np.complex128)

    # 2D layer is kron(M_1, M_0)
    W_0 = np.kron(M_0, M_1)

    return W_0

def create_2d_radix2_dit_layer(
    layer: int, n_samples_0: int, n_samples_1: int
) -> NDArray[np.complex128]:
    """
    Construct a 2D FFT layer matrix for Radix-2 DIT by Kronecker product of 1D DIT layers.

    In DIT (Decimation-In-Time), the twiddle factors are applied before the butterfly.

    Args:
        layer (int): The current FFT layer index (0-based).
        n_samples_0 (int): Number of samples along the first (row) dimension.
        n_samples_1 (int): Number of samples along the second (column) dimension.

    Returns:
        np.ndarray: A complex-valued matrix of shape (n0*n1, n0*n1) representing
                    the layer-wise transformation of the 2D FFT.
    """
    n_layers_0 = int(np.log2(n_samples_0))
    n_layers_1 = int(np.log2(n_samples_1))

    # Row-wise transformation
    if layer < n_layers_0:
        M_0 = create_1d_radix2_dit_layer(layer, n_samples_0)
    else:
        M_0 = np.eye(n_samples_0, dtype=np.complex128)

    # Column-wise transformation
    if layer < n_layers_1:
        M_1 = create_1d_radix2_dit_layer(layer, n_samples_1)
    else:
        M_1 = np.eye(n_samples_1, dtype=np.complex128)

    # 2D layer is computed as Kronecker product of row and column transforms
    W_0 = np.kron(M_1, M_0)

    return W_0

def create_1d_radix4_dif_layer(layer: int, n_samples: int) -> NDArray[np.complex128]:
    """Construct a sparse matrix for one layer of the Radix-4 DIF FFT."""
    radix = 4
    n_layers = int(np.log(n_samples) / np.log(radix))
    W4 = np.array([
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ], dtype=np.complex128)
    G = np.eye(radix ** layer)
    B = np.eye(n_samples // radix ** (layer + 1))
    M = np.kron(G, np.kron(W4, B))

    # Apply twiddle factors after butterfly
    #if layer < n_layers - 1:
    #    tf = tf_vector_radix(layer, n_samples, radix=4)
    #    M = np.diag(tf) @ M
    tf = tf_vector_radix_dif(layer, n_samples, radix=4)

    M = tf[:, np.newaxis] * M

    return M

def create_1d_radix4_dit_layer(layer: int, n_samples: int) -> NDArray[np.complex128]:
    """
    Construct the sparse matrix for one layer of a Radix-4 DIT FFT.

    In DIT (Decimation-In-Time), the twiddle factors are applied before the butterfly,
    and the layer matrix is structured accordingly.

    Parameters
    ----------
    layer : int
        Index of the FFT layer (0-based).
    n_samples : int
        Total number of samples (must be a power of 4).

    Returns
    -------
    NDArray[np.complex128]
        Layer transformation matrix for one radix-4 DIT stage.
    """
    radix = 4
    n_layers = int(np.log(n_samples) / np.log(radix))
    assert 0 <= layer < n_layers, "Layer index out of range"

    # 4-point DIT butterfly matrix (same as DIF)
    W4 = np.array([
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ], dtype=np.complex128)

    # Apply twiddle factors BEFORE the butterfly in DIT
    tf = tf_vector_radix_dit(layer, n_samples, radix=4)
    T = np.diag(tf)  # Twiddle factor diagonal matrix

    # Build Kronecker structure: A = (G ⊗ I), B = (I ⊗ B)
    G = np.eye(n_samples // (radix ** (layer + 1)), dtype=np.complex128)
    B = np.eye(radix ** layer, dtype=np.complex128)

    # Twiddle factor placement matrix: G ⊗ I
    T_mat = np.kron(G, np.kron(T, B))

    # Butterfly application matrix: G ⊗ W4 ⊗ B
    M = np.kron(G, np.kron(W4, B))

    # Final DIT layer: apply twiddles first, then butterfly
    M = M @ T_mat

    return M


def create_2d_radix4_dif_layer(layer: int, n_samples_0: int, n_samples_1: int) -> NDArray[np.complex128]:
    """Construct a 2D sparse matrix for one layer of the 2D Radix-4 DIF FFT."""
    n_layers_0 = int(np.log(n_samples_0) / np.log(4))
    n_layers_1 = int(np.log(n_samples_1) / np.log(4))
    if layer < n_layers_0:
        M_0 = create_1d_radix4_dif_layer(layer, n_samples_0)
    else:
        M_0 = np.eye(n_samples_0, dtype=np.complex128)
    if layer < n_layers_1:
        M_1 = create_1d_radix4_dif_layer(layer, n_samples_1)
    else:
        M_1 = np.eye(n_samples_1, dtype=np.complex128)
    return np.kron(M_1, M_0)


def create_2d_radix4_dit_layer(layer: int, n_samples_0: int, n_samples_1: int) -> NDArray[np.complex128]:
    """
    Construct a 2D sparse matrix for one layer of the 2D Radix-4 DIT FFT.

    In DIT, twiddle factors are applied before the butterfly. This function composes
    the 2D layer as a Kronecker product of two 1D DIT layer matrices.

    Parameters
    ----------
    layer : int
        The index of the FFT layer (0-based).
    n_samples_0 : int
        Number of rows in the 2D FFT (must be power of 4).
    n_samples_1 : int
        Number of columns in the 2D FFT (must be power of 4).

    Returns
    -------
    NDArray[np.complex128]
        A sparse matrix representing a single 2D DIT FFT layer.
    """
    radix = 4
    n_layers_0 = int(np.log(n_samples_0) / np.log(radix))
    n_layers_1 = int(np.log(n_samples_1) / np.log(radix))

    if layer < n_layers_0:
        M_0 = create_1d_radix4_dit_layer(layer, n_samples_0)
    else:
        M_0 = np.eye(n_samples_0, dtype=np.complex128)

    if layer < n_layers_1:
        M_1 = create_1d_radix4_dit_layer(layer, n_samples_1)
    else:
        M_1 = np.eye(n_samples_1, dtype=np.complex128)

    # The 2D DIT layer is the Kronecker product of the 1D row and column transforms
    return np.kron(M_1, M_0)


def create_fft_layer(layer, n_samples: list, radix: int, mode: str = "dif"):
    """
    Create an FFT layer matrix for given radix and dimension(s), with mode selection.

    Parameters
    ----------
    layer : int
        Layer index (0-based).
    n_samples : list[int]
        List of sample sizes for each dimension (1D or 2D).
    radix : int
        Radix of FFT (2 or 4).
    mode : str, optional
        FFT style, either 'dif' (Decimation in Frequency) or 'dit' (Decimation in Time).
        Default is 'dif'.

    Returns
    -------
    np.ndarray
        FFT layer matrix.
    """
    assert mode in ("dif", "dit"), "mode must be 'dif' or 'dit'"

    if radix == 2:
        if len(n_samples) == 1:
            if mode == "dif":
                M = create_1d_radix2_dif_layer(layer=layer, n_samples=n_samples[0])
            else:
                M = create_1d_radix2_dit_layer(layer=layer, n_samples=n_samples[0])

        elif len(n_samples) == 2:
            if mode == "dif":
                M = create_2d_radix2_dif_layer(layer=layer, n_samples_0=n_samples[0], n_samples_1=n_samples[1])
            else:
                M = create_2d_radix2_dit_layer(layer=layer, n_samples_0=n_samples[0], n_samples_1=n_samples[1])

    elif radix == 4:
        if len(n_samples) == 1:
            if mode == "dif":
                M = create_1d_radix4_dif_layer(layer=layer, n_samples=n_samples[0])
            else:
                M = create_1d_radix4_dit_layer(layer=layer, n_samples=n_samples[0])

        elif len(n_samples) == 2:
            if mode == "dif":
                M = create_2d_radix4_dif_layer(layer=layer, n_samples_0=n_samples[0], n_samples_1=n_samples[1])
            else:
                M = create_2d_radix4_dit_layer(layer=layer, n_samples_0=n_samples[0], n_samples_1=n_samples[1])

    else:
        raise ValueError(f"Unsupported radix {radix}, only 2 or 4 supported.")
    
    return M

