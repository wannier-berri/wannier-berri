
import numpy as np


def rotate_block_matrix(Z, lblocks, lindices, rblocks, rindices,
                        # inv_left, inv_right,
                        result=None):
    """
    Rotates the matrix Z using the block-diagonal rotation matrices

    Parameters
    ----------
    Z : np.array(complex, shape=(M,N))
        the matrix to be rotated
    lblocks : list(np.array(complex, shape=(m,m)))
        the blocks of hte left matrix. sum(m) = M
    lindices : list(tuple(int))
        the indices of the blocks of the left matrix
    rblocks : list(np.array(complex, shape=(n,n)))
        the blocks of hte right matrix. sum(n) = N
    rindices : list(tuple(int))
        the indices of the blocks of the right matrix

    Returns
    -------
    np.array(complex, shape=(M,N))
        the rotated matrix
    """
    if result is None:
        result = np.zeros(Z.shape, dtype=Z.dtype)
    for (start, end), block in zip(lindices, lblocks):
        result[start:end, :] = block @ Z[start:end, :]

    for (start, end), block in zip(rindices, rblocks):
        result[:, start:end] = result[:, start:end] @ block

    return result


def block_matrix_to_diagonals(blocks, indices=None):
    """
    Extracts the blocks of a block-diagonal matrix

    Parameters
    ----------
    blocks : list(np.array(complex, shape=(m,m)))
        the blocks of hte left matrix. sum(m) = M
    indices : list(tuple(int))
        the indices of the blocks of the left matrix

    Returns
    -------
    list(np.array(complex, shape=(m,m)))
        the blocks of the matrix
    """
    if indices is None:
        indices = []
        start = 0
        for block in blocks:
            indices.append((start, start + block.shape[0]))
            start += block.shape[0]

    size = indices[-1][1] - indices[0][0]
    tmp = np.zeros((size, size), dtype=blocks[0].dtype)
    max_block_size = max(block.shape[0] for block in blocks)
    for (start, end), block in zip(indices, blocks):
        tmp[start:end, start:end] = block
    result = np.zeros((2 * max_block_size - 1, size), dtype=blocks[0].dtype)
    rng = np.arange(size)
    result[0] = tmp[rng, rng]
    for l in range(1, max_block_size):
        result[l, :-l] = tmp[rng[:-l], rng[l:]]
        result[-l, :-l] = tmp[rng[l:], rng[:-l]]
    return result


def get_inverse_block(D):
    """
    Get the inverse of a block-diagonal matrix (given as a nested list of numpy arrays)
    """
    if isinstance(D, list):
        return [get_inverse_block(d) for d in D]
    elif isinstance(D, np.ndarray):
        return np.linalg.inv(D)
    else:
        raise ValueError(f"Unknown type {type(D)}")
