from numpy import empty, dot
from numba import njit


@njit
def axis0_pos_neg_mask(arr, mask, mask_length):
    """Safely slice a 2D array along the first axis using a boolean mask."""
    assert arr.ndim == 2
    assert mask.ndim == 1 and mask.shape[0] == arr.shape[0]
    assert 0 <= mask_length <= mask.sum()

    if mask_length == 0:
        return empty((0, arr.shape[1]), dtype=arr.dtype), arr
    if mask_length == arr.shape[0]:
        return arr, empty((0, arr.shape[1]), dtype=arr.dtype)

    out_pos = empty((mask_length, arr.shape[1]), dtype=arr.dtype)
    out_neg = empty((mask.shape[0] - mask_length, arr.shape[1]), dtype=arr.dtype)
    j_pos = j_neg = 0
    for i in range(mask.shape[0]):
        if mask[i]:
            out_pos[j_pos] = arr[i]
            j_pos += 1
        else:
            out_neg[j_neg] = arr[i]
            j_neg += 1

    return out_pos, out_neg


@njit
def safe_sparse_dot(a, b):
    if a.ndim > 2 or b.ndim > 2:
        ret = dot(a, b)
    else:
        ret = a @ b
    return ret
