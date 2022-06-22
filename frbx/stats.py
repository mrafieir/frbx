import numpy as np
import frbx as fx


class stats:
    """
    This class is a toolbox for carrying out statistical analyses on power spectra.  The constructor
    requires a 'fx.analysis.*_analysis' object containing power spectra of data and mocks.  The 'lmin'
    argument specifies a multipole below which estimators are invalid.

    It currently supports 'fx.clfg_analysis' and 'fx.multi_clfg_analysis' objects.

    Members:

        self.analysis            (object) instance of 'fx.analysis.*_analysis'
        self.run_sig_loc         (method) runs 'self.sig_loc' using 'self.alpha_exp'
        self.alpha_exp           (method) optimal estimator based on 'fx.model.cl_models.exp'
        self.sig_loc             (method) local significance
        clfg_analysis.sig_glo    (static method) global significance
    """

    def __init__(self, analysis, lmin=20.0):
        assert isinstance(analysis, (fx.clfg_analysis, fx.multi_clfg_analysis))
        assert isinstance(lmin, (int, float)) and (lmin > 0)

        self.analysis = analysis
        self.lmin = float(lmin)

    def run_sig_loc(self, l):
        """
        Adopts the estimator 'self.alpha_exp' and runs 'self.sig_loc' for data and mocks.  The 'l' argument
        specifies a 1-d array containing trial L-values.  It returns a 2-d array containing local significance
        values for data binned in redshift and trial L-values, along with a 3-d array containing the same
        statistics for all mocks (along axis 0).
        """

        # Number of redshift bins in 'self.analysis'.
        nz = max(self.analysis.ng, 1) if hasattr(self.analysis, 'ng') else 1

        # 2-d array of shape (self.nz, l.size) that will contain local significance values
        # over z bins and trial L-values for data.
        sig_loc_data = np.zeros((nz, l.size))

        # 3-d array of shape (self.analysis.nmc, nz, l.size) that will contain local significance
        # values over Monte Carlo realizations, z bins and trial L-values for all mocks.
        sig_loc_mocks = np.zeros((self.analysis.nmc, nz, l.size))

        # Trial L-values.
        for i, l_char in enumerate(l):
            alpha_data, alpha_mocks = self.alpha_exp(l_char)
            sig_loc_data[:,i] = self.sig_loc(alpha_data, alpha_mocks)

            # Monte Carlo realizations.
            for j in range(self.analysis.nmc):
                sig_loc_mocks[j,:,i] = self.sig_loc(alpha_mocks[j,:], alpha_mocks)

        return sig_loc_data, sig_loc_mocks

    def alpha_exp(self, l_char, lmin=None):
        """
        Computes an optimal estimator for the template 'fx.model.cl_models.exp'.  The 'l_char' argument
        specifies the characteristic multipole at which the baseline of angular power spectrum falls
        to 1/e of its extremum (l -> 0); it also specifies the multipole at which a significant deviation
        is observed from the baseline.  The 'lmin' argument specifies a multipole below which the
        estimator is invalid.  The estimator is computed independently for data and mocks.

        It assumes that self.analysis.clfg_mocks has a shape of (self.analysis.nmc, self.analysis.ng, l),
        where the last axis corresponds to multipoles.

        Hence, this method returns a 1-d array of shape (self.analysis.ng) and a 2-d array of shape
        (nmc, self.analysis.ng).
        """

        assert isinstance(l_char, (int, float))
        assert l_char > 0

        lmin = self.lmin if (lmin is None) else lmin

        ell = np.arange(self.analysis.lmax+1, dtype=np.float)

        # Sanity check.
        assert self.analysis.clfg_mocks.shape[0] == self.analysis.nmc
        assert self.analysis.clfg_data.shape[-1] == self.analysis.clfg_mocks.shape[-1] == ell.size

        # Template.
        clm = fx.model.cl_models(l_char)
        _par = [1.0, l_char]
        m = clm.exp(ell, _par)      # 1-d array of length (ell.size).

        if hasattr(self.analysis, 'clgg_list'):
            # 2-d arrays of shape (self.analysis.ng, ell.size).
            _clgg = np.asarray(self.analysis.clgg_list)
            _clfg_data = self.analysis.clfg_data
        else:
            # 2-d arrays of shape (1, ell.size).
            _clgg = np.asarray([self.analysis.deltag.clgg])
            _clfg_data = np.asarray([self.analysis.clfg_data])

        assert _clgg.shape[0] == _clfg_data.shape[0]
        assert _clgg.shape[-1] == _clfg_data.shape[-1] == ell.size

        # True -> valid.
        mask = (ell >= lmin)

        _ret = _clgg[:,mask]**(-1) * (2*ell[mask] + 1) * m[mask]
        _norm = np.sum(_ret * m[mask], axis=-1)

        ret_data = _clfg_data[:,mask] * _ret
        ret_data = np.sum(ret_data, axis=-1)
        ret_data /= _norm

        ret_mocks = self.analysis.clfg_mocks[...,mask] * _ret
        ret_mocks = np.sum(ret_mocks, axis=-1)
        ret_mocks /= _norm

        return ret_data, ret_mocks

    def sig_loc(self, xdata, xmocks):
        """
        Computes the local (i.e. not accounting for look-elsewhere effect) significance for expected
        value of the estimator 'xdata' of length (self.analysis.ng) using the variance of the same
        estimator over mocks, which is passed by 'xmocks' of shape (self.analysis.nmc, self.analysis.ng).
        """

        assert isinstance(xdata, (float, np.ndarray))
        assert isinstance(xmocks, np.ndarray)
        assert xmocks.shape[0] == self.analysis.nmc

        # np.mean(xmocks) -> 0.
        var = np.mean(xmocks**2, axis=0)
        assert np.all(var > 0)

        ret = xdata / var**0.5

        return ret

    @staticmethod
    def sig_glo(s_loc, axis=None):
        """
        Computes (min,max) global significance along the 'axis' of 's_loc', which is an array-like object
        corresponding to the local significance of an estimator. NaN's are ignored.
        """

        assert isinstance(s_loc, (float, list, np.ndarray))

        return np.nanmin(s_loc, axis=axis), np.nanmax(s_loc, axis=axis)
