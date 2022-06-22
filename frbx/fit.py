import warnings
import numpy as np
from scipy.optimize import minimize


def sanity_check_cl(l, cl, cov_inv):
    """
    This helper function asserts the type and shape of a 1-d array of angular wavenumbers 'l',
    2-d array of angular power spectra 'cl', and 2-d array of inverted covariance 'cov_inv'.
    """

    assert isinstance(l, np.ndarray), 'l must be a numpy array.'
    assert isinstance(cl, np.ndarray), 'cl must be a numpy array.'
    assert l.ndim == 1, 'l must be 1-d.'
    assert cl.shape[-1] == l.size, 'cl must be 1-d with axis (l) or 2-d with axes (realizations, l).'
    assert l.size != 0, 'l must not be empty.'
    assert cl.size != 0, 'cl must not be empty.'
    assert isinstance(cov_inv, np.ndarray), 'cov_inv must be a numpy array.'
    assert cov_inv.shape == (l.size, l.size), 'cov_inv shape must be consistent with input l.'
    assert np.isfinite(cov_inv).all(), 'cov_inv elements must be finite.'


def chi2(p, m, l, cl, cov_inv, reduced=False):
    """
    This function computes the chi-square residuals of an angular power spectrum
    summed over angular wavenumbers.

    Args:

        p: (list or 1-d array) model parameters.
        m: (function) model.
        l: (1-d array) angular wavenumbers.
        cl: (1-d array) angular power spectrum.
        cov_inv: (2-d array) inverse of the covariance matrix of 'cl'.
        reduced: (bool) whether to return the reduced chi-square.

    Returns:

        float, (reduced) chi-square residuals summed over 'l'.

    Raises:

        AssertionError: invalid input args.
    """

    assert isinstance(p, (list, np.ndarray))
    assert callable(m)
    sanity_check_cl(l, cl, cov_inv)
    assert isinstance(reduced, bool)

    r = m(l, p) - cl
    ret = r.T.dot(cov_inv).dot(r)

    if not reduced:
        return ret
    else:
        return ret / (l.size - len(p))


def jcov(jac, mode='naive'):
    """
    This function converts a jacobian matrix of parameters to their covariance.

    Args:

        jac: (2-d array) jacobian matrix of parameters.
        mode: (str) method of inversion:
              'naive' (atol=1.0e-16, rtol=1.0e-14)
              'svd'   (atol=1.0e-16, rtol=1.0e-2).

    Returns:

        2-d array, covariance matrix of parameters.

    Raises:

        AssertionError: invalid input args.
        RuntimeError: invalid internal mode.
        RuntimeWarning: np.linalg.linalg.LinAlgError.
    """

    assert isinstance(jac, np.ndarray)
    assert jac.dtype == np.float64
    assert jac.ndim == 2
    assert mode in ('naive', 'svd')

    try:
        if mode == 'naive':
            # Gauss-Newton approximation of the Hessian of the cost function
            h = jac.T.dot(jac)

            # prone to failure if h is ill-conditioned
            cov = np.linalg.inv(h)
        elif mode == 'svd':
            # full_matrices=False -> u and v have an axis with a size of min(jac.shape)
            u, s, v = np.linalg.svd(jac, full_matrices=False)

            # The following criterion is from scipy.optimize.curve_fit routine.
            eps = np.finfo(np.float64).eps * max(jac.shape) * s[0]
            s = s[s > eps]
            v = v[:s.size]

            # Moore-Penrose pseudo-inverse of the Hessian of the cost function
            # h = v.T * s^2 @ v -> cov = v.T / s^2 @ v
            cov = np.dot(v.T / s**2.0, v)
        else:
            raise RuntimeError('jcov: invalid mode!')
    except np.linalg.LinAlgError:
        warnings.warn('jcov: inversion failed! cov[:] = np.inf', RuntimeWarning)
        cov = np.full_like(jac, np.inf)

    return cov


def fit_cl(m, l, cl, cov_inv, bounds, mask=None, verbose=False, **kwargs):
    """
    This function fits an array of angular power spectrum realizations to a model.

    Args:

        m: (function) model.
        l: (1-d array) angular wavenumbers.
        cl: (2-d array) angular power spectrum with the following axes: [realizations, l].
        cov_inv: (2-d array) inverse of the ensemble-averaged covariance matrix of cl.
        bounds: (tuple) lists of lower and upper bounds for constraining model parameters:
                ([lower bounds], [upper bounds]).
        mask: (1-d array) if not None, specifies a boolean mask for 'cl' along its last axis.
        verbose: (bool) enables a verbose mode.
        **kwargs: optional 'minimize' parameters.

    Returns:

        tuple (p, c): n-d array of the best-fit parameters 'p', along with an
                      n-d array of covariance 'c' for the set of realizations.

    Raises:

        AssertionError: invalid input args.
        RuntimeWarning: power spectra are all zeros.
    """

    assert callable(m)
    sanity_check_cl(l, cl, cov_inv)

    for (i, j) in enumerate(bounds[0]):
        assert j < bounds[1][i]

    if mask is not None:
        assert mask.size == l.size
        l, cl = l[mask], cl[:,mask]

    if np.all(cl[:, mask] == 0.0):
        warnings.warn('fit_cl: cl[:,mask] are all 0.', RuntimeWarning)

    p0 = []
    for (i, j) in enumerate(bounds[0]):
        _p0_low = j
        _p0_high = bounds[1][i]

        _p0_log_spaced = False
        if (_p0_low > 0.0) and (_p0_high > 0.0):
            _p0_low = np.log10(_p0_low)
            _p0_high = np.log10(_p0_high)
            _p0_log_spaced = True

        _mean = np.mean((_p0_low, _p0_high))

        if _p0_log_spaced:
            _mean = 10.0**_mean

        p0.append(_mean)

    (p, c) = ([], [])
    for r in range(cl.shape[0]):
        try:
            ls = minimize(chi2, np.asarray(p0), args=(m, l, cl[r,:], cov_inv), **kwargs)

            if verbose:
                print(f'fit_cl: minimize status -> {ls.message}, {ls.status}')

            px = ls.x

            if hasattr(ls, 'jac'):
                covx = jcov(ls.jac)
            else:
                covx = np.full((px.size, px.size), np.inf)
        except (TypeError, ValueError) as err:
            ls = None
            print(err)
            px = np.full(len(p0), np.nan)
            covx = np.full((px.size, px.size), np.inf)

        if np.all(covx == np.inf) and hasattr(ls, 'jac'):
            if verbose:
                print('fit_cl: jcov failed in the naive mode. Trying the SVD decomposition.')

            covx = jcov(ls.jac, mode='svd')

        if np.all(covx == np.inf) and verbose:
            print('fit_cl: jcov failed in the svd mode!')

        assert px.size == covx.shape[0] == covx.shape[1]

        p.append(px)
        c.append(covx)

    ret = (np.asarray(p), np.asarray(c))

    return ret


def max_likelihood(f, x, p0, verbose=True):
    """
    This function computes a set of parameters 't' which maximize the distribution
    f(x|t) within bounds 'p0'.  It assumes that 'f' is vectorized.
    """

    assert callable(f)
    p0 = np.asarray(p0)

    def logf(p):
        ret = f(x, *p)
        return -np.sum(np.log(ret))

    ls = minimize(logf, p0, method='Nelder-Mead')

    if verbose:
        print(f'max_likelihood: minimize status -> {ls.message}, {ls.status}')

    return ls.x
