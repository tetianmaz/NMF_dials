import numpy as np
import math

def normalize_by_area(series):
    """
    Normalize a series by the total sum (area under the curve).
    If the total sum is zero, returns the series unchanged.
    """
    total_sum = np.sum(series)  
    if total_sum != 0:
        return np.round(series / total_sum, decimals=15)  
    else:
        return series

def contains_only_zeros_nan_or_inf(lst):
    """
    Check if a list contains only zeros, NaN, or infinity.
    """
    return all(x == 0 for x in lst) or any(math.isnan(x) for x in lst) or any(math.isinf(x) for x in lst)

def filter_and_normalize(lst):
    """
    Normalize a list by area which does not contain only zeros, NaN, or infinity.
    If the list contains only invalid values, return NaN.
    """
    if not contains_only_zeros_nan_or_inf(lst):
        return normalize_by_area(np.array(lst))
    else:
        return np.nan
