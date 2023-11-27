import numpy as np
import warnings

def fmt_cnumber(cnumber):
    """generate a formated string representing a 8 precision complex number

    Args:
        cnumber (complx): a complex number

    Returns:
        str: a string representing the complex number, which is rounded
    """
    return np.format_float_scientific(cnumber.real, precision=8) + " + 1.j * " + np.format_float_scientific(cnumber.imag, precision=8)

def numpy_float_to_set(a: np.ndarray):
    """tansform a complex valued numpy array into a set of strings with fixed (8) precision.

    Args:
        a (np.ndarray): a complex 1-d numpy array

    Returns:
        set: A set of strings representing the numpy array
    """

    return set(map(fmt_cnumber, a))

def remove_from_set(s, element):
    """remove a element from the set storing complex values

    Args:
        s (set): the set created by numpy_float_to_set 
        element (complex): the complex number that needs to be removed
    """
    s.remove(fmt_cnumber(element))

def extract_conjugate_pairs(expn):
    """extract the complex pairs and the non-complex conjugate pairs from the exponent array.

    This function sorts out the the expn array generated from the prony fitting program into two separate arrays:
    - expn_conjugate_pairs: the exponential results that forms a complex conjugate pair. lenth of this array is even.
    - expn_non_conjugate_pairs: any exponent that does not have an complex conjugate in the expn list.

    Args:
        expn (np.array): The expn array obtained from sum over poles scheme, or time domain fitting.

    Raises:
        ValueError: It is strange to have duplicated exponentials after you decompose the spectral function. Though it might be very possilbe, but generally you won't have them. So, cautiously throw this this ValueError won't hurt!

    Returns:
        tuple: `expn_conjugate_pairs` and `expn_non_conjugate_pairs`. Containing values and indicies.
    """
    expn_set = numpy_float_to_set(expn)
    if len(expn) > len(expn_set):
        raise ValueError("There is duplicated items in your exponent list. This does not fit in the context of spectral function decomposition. Double check your spectral decomposition data!!!")

    expn_conjugate_pairs = []
    expn_non_conjugate_pairs = []

    for ii, e in enumerate(expn):
        conjugate = np.conj(e)
        if np.isreal(e):
            expn_non_conjugate_pairs.append((ii, e))
            remove_from_set(expn_set, e)
        elif fmt_cnumber(conjugate) in expn_set:
            expn_conjugate_pairs.append((ii, e))
            jj = np.where(np.abs((expn-conjugate)/conjugate) < 1e-8)[0][0]
            expn_conjugate_pairs.append((jj, conjugate))
            remove_from_set(expn_set, e)
            remove_from_set(expn_set, conjugate)
        elif fmt_cnumber(e) in expn_set:
            expn_non_conjugate_pairs.append((ii, e))
            remove_from_set(expn_set, e)
        else:
            pass
    return expn_conjugate_pairs, expn_non_conjugate_pairs

def get_symmetrized_deom_inputs(etal: np.ndarray, expn: np.ndarray):
    """Generate a the correct eta and exponential data to be the inputs of the DEOM c++ library. 
    In the current version of c++ MOSCAL library, the bath mode spectral function decomposition is treated as inputs, passed to the c++ code python `json`. Thus, the user shall provide the 3 eta vectors and one exponents vector to the c++ library via json input files.
    This python function transforms the `etal` and `expn` vector into the `etal`, `etar`, `etaa`, and `expn` vectors, directly usable by the DEOM program. The underlying idea is, when ever there is a complex conjugate pair in `expn`, MOSCAL2.0 needs corresponding `etal` and `etar` as inputs. Precisely, the pair (2-sized array) needs to be symmetrized as such.
        `etar = np.flip(np.conj(etal))` 
    Wheras for the exponemts that don't for a pair, things are easier.
        `etar = np.flip(etal)`.

    Warning:
    This function is a personal implementation of this symmetrization process. The author donnot have direct affliation with the original developer of the MOSCAL2.0. Please use with care.

    Args:
        etal (np.ndarray): _description_
        expn (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    warnings.warn("Function `get_symmetrized_deom_inputs` is a personal implementation, which is not an official implementation. Please use with extreme care.")
    expn_conjugate_pairs, expn_non_conjugate_pairs = extract_conjugate_pairs(expn)

    if expn_conjugate_pairs:
        arg_pair, expn_pair = list(map(np.array, zip(*expn_conjugate_pairs)))
        etal_pair = etal[arg_pair]
        etar_pair = np.flip(np.conj(etal_pair.reshape(2, -1))).flatten()
    else:
        expn_pair = np.array([])
        etal_pair = np.array([])
        etar_pair = np.array([])

    if expn_non_conjugate_pairs:
        arg_non_pair, expn_non_pair = list(map(np.array, zip(*expn_non_conjugate_pairs)))
        etal_non_pair = etal[arg_non_pair]
        etar_non_pair = np.conj(etal_non_pair)
    else:
        expn_non_pair = np.array([])
        etal_non_pair = np.array([])
        etar_non_pair = np.array([])

    expn = np.append(expn_pair, expn_non_pair)
    etal = np.append(etal_pair, etal_non_pair)
    etar = np.append(etar_pair, etar_non_pair)
    etaa = np.abs(etal)

    return etal, etar, etaa, expn