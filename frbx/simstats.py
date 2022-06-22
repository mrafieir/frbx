import glob
import numpy as np
import numpy.random as rand
from h5py import File as FileH5
import frbx as fx


class simstats:
    """
    Contains methods for collating angular power spectra from simulations.

    Members:

        self.file_path: (list) paths to h5 files.
        self.dataset: (list) dataset to be processed.
        self.m: (dict) all data organized into one object.
        self.read_file: (method) reads and concatenates dataset.
        simstats.snr: (static method) computes the total SNR using the full covariance matrix.
        simstats.snr_gaussian: (static method) computes the Gaussian SNR.
        simstats.nl_gaussian: (static method) computes the bandpower noise N_l^{xy} using the Gaussian approximation.
        simstats.covariance: (static method) computes the covariance matrix.
        simstats.make_mask: (static method) makes a boolean mask for angular power spectra.
        self._simstats__test_*.
    """

    def __init__(self, file_path, dataset):
        """
        Constructor arguments:

            file_path: (str) path to h5 files containing simulation results.
            dataset: (list) name of datasets to be processed.
        """

        assert isinstance(file_path, str)
        assert isinstance(dataset, list)

        self.file_path = sorted(glob.glob(file_path))
        if len(self.file_path) == 0:
            raise RuntimeError('simstats.__init__: verify the file_path!')

        self.dataset = dataset

        self.m = {}
        for i in dataset:
            assert i in ('ff', 'gg', 'fg')

            try:
                r = self.read_file(i)
            except MemoryError as err:
                raise RuntimeError(f'{err}\nsimstats is not able to concatenate a very large stack of dataset.')
            except IOError:
                raise RuntimeError('simstats is not finding stats in self.file_path.')

            r.update({'E': np.mean(r['D'], axis=0)})
            self.m.update({i: r})

    def read_file(self, dataset):
        """
        This method reads a multi-dimensional dataset of angular power spectra from the list in self.file_path.
        Angular power spectra are appended to the zeroth axis, which (by convention) corresponds to randomly
        different simulation runs.

        Args:

            dataset: (str) name of a dataset.

        Returns:

            dictionary of the following arrays:
                'ell': angular wavenumbers.
                'D': angular power spectra.
                'axes': axis labels for 'D'.
        """

        assert isinstance(dataset, str)

        d0 = FileH5(self.file_path[0], 'r')
        _d0 = d0[dataset]

        try:
            ell = _d0.attrs['ell']
        except KeyError:
            ell = _d0.attrs['l']

        axes = _d0.attrs['axes']

        d0.close()

        d = None
        for (i, path) in enumerate(self.file_path):
            dn = fx.read_h5(path, dataset)
            if not i:
                d = dn.copy()
            else:
                assert np.all(d.shape[1:] == dn.shape[1:])
                d = np.append(d, dn, axis=0)

        return {'ell': ell, 'axes': axes, 'D': d}

    @staticmethod
    def snr(cl, cov, cumulative=False):
        """
        This static method computes the total signal-to-noise using the full covariance matrix.

        Args:

            cl: (1-d array) power spectrum given a correlated hypothesis.
            cov: (2-d array) covariance matrix given a null hypothesis.
            cumulative: (bool) whether to return cumulative statistics.

        Returns:

            float (cumulative=False) or 1-d array (cumulative=True).
        """

        assert isinstance(cl, np.ndarray) and (cl.ndim == 1)
        assert isinstance(cov, np.ndarray)
        assert isinstance(cumulative, bool)

        if cl.size == cov.size == 1:
            if not cov:
                return cl**2.0 / cov
            else:
                return 0.0

        assert cl.shape[0] == cov.shape[0] == cov.shape[1]

        try:
            cov_inv = np.linalg.inv(cov)

            if not cumulative:
                ret = np.dot(cl.T.dot(cov_inv), cl)

                if not (ret >= 0.0):
                    raise RuntimeError('simstats.snr: negative snr2 encountered!')
            else:
                ret = np.zeros_like(cl)

                for i in range(cl.size):
                    j = i + 1
                    ret[i] = cl[:j].T.dot(cov_inv[:j,:j]).dot(cl[:j])

                mask = (ret >= 0.0)
                ret[~mask] = 0.0
        except np.linalg.linalg.LinAlgError:
            ret = np.zeros_like(cl) if cumulative else 0.0

        return np.sqrt(ret)

    @staticmethod
    def snr_gaussian(ell, cl_xy, cl_xx, cl_yy, f_sky, corr_like=False, cumulative=False, interp=False):
        """
        This static method computes the Gaussian signal-to-noise ratio for the cross power of two fields.  It assumes
        that l bins are spaced linearly.

        Args:

            ell: (1-d array) angular wavenumbers.
            cl_xy: (1-d array) averaged cross power (corr hypothesis).
            cl_xx: (1-d array) averaged auto-power for a source type (null hypothesis).
            cl_yy: (1-d array) averaged auto-power for another source type (null hypothesis).
            f_sky: (float) fraction of the sky covered by sources.
            corr_like: (bool) whether to return a correlation-like factor, (cl_xy^2 / (Cl_xx * cl_yy))^0.5, not the SNR.
            cumulative: (bool) whether to return cumulative statistics.
            interp: (bool) whether to interpolate over discrete variables.

        Returns:

            float if cumulative is False.
            tuple containing two 1-d arrays (l, SNR) if cumulative is True.
        """

        assert isinstance(ell, np.ndarray)
        assert np.isfinite(ell).all()
        assert np.all(ell >= 0.0)
        assert np.all(ell == np.sort(ell))
        assert isinstance(cl_xy, np.ndarray) and np.isfinite(cl_xy).all()
        assert isinstance(cl_xx, np.ndarray) and np.isfinite(cl_xx).all()
        assert isinstance(cl_yy, np.ndarray) and np.isfinite(cl_yy).all()
        assert ell.shape == cl_xy.shape == cl_xx.shape == cl_yy.shape
        assert isinstance(f_sky, float) and (0.0 < f_sky <= 1.0)
        assert isinstance(corr_like, bool)
        assert isinstance(cumulative, bool)
        assert isinstance(interp, bool)

        snr2 = np.zeros_like(cl_xy)
        num = cl_xy ** 2

        if (not interp) and (not corr_like):
            den = fx.simstats.nl_gaussian(ell, cl_xx, cl_yy, f_sky)
            den **= 2
        else:
            den = cl_xx * cl_yy

        _mask = (den != 0)
        snr2[_mask] = num[_mask] / den[_mask]

        if not interp:
            snr2 = np.cumsum(snr2)
            snr = np.sqrt(snr2)

            if cumulative:
                return ell, snr
            else:
                return snr[-1]
        else:
            s = 2 * f_sky * ell * snr2
            _sp = fx.spline(ell, s, ext=3)      # Out of bounds -> boundary values.

            l_low, l_high = fx.utils.bin_edges(ell)

            l_min = max(0.0, l_low[0])
            if cumulative:
                snr = np.sqrt([_sp.integral(l_min, i) for i in l_high])
                return ell, snr
            else:
                snr = np.sqrt(_sp.integral(l_min, l_high[-1]))
                return snr

    @staticmethod
    def nl_gaussian(ell, cl_xx, cl_yy, f_sky):
        """
        This static method computes the bandpower noise N_l^{xy} in the bandpower C_l^{xy} using the Gaussian
        approximation, which includes a mode-counting term.  To this end, the bandpower SNR is given simply by
        C_l^{xy} / N_l^{xy}.  It assumes that l bins are spaced linearly.

        Args:

            ell: (1-d array) angular wavenumbers.
            cl_xx: (1-d array) angular auto-power spectrum of source x at l.
            cl_yy: (1-d array) angular auto-power spectrum of source y at l.
            f_sky: (float) fraction of the sky covered by the simulations.

        Returns:

            array, the noise.
        """

        assert isinstance(ell, np.ndarray) and np.isfinite(ell).all() and np.all(ell >= 0.0)
        assert np.all(ell == np.sort(ell))
        assert isinstance(cl_xx, np.ndarray) and np.isfinite(cl_xx).all()
        assert isinstance(cl_yy, np.ndarray) and np.isfinite(cl_yy).all()
        assert ell.shape == cl_xx.shape == cl_yy.shape
        assert 0.0 < f_sky <= 1.0

        l_low, l_high = fx.utils.bin_edges(ell)

        b = f_sky * (l_high**2.0 - l_low**2.0)
        assert np.all(b > 0.0)

        ret = np.sqrt(cl_xx * cl_yy / b)

        return ret

    @staticmethod
    def covariance(ell, cl, nl=None, cl_aux=None):
        """
        This static method computes the covariance matrix for a set of angular power spectra.

        Args:

            ell: (array) angular wavenumbers.
            cl: (array) angular power spectra for computing the covariance. (*)
            nl: (int) if not None, number of ell bins along each axis of the covariance matrix.
            cl_aux: (array) if not None, auxiliary angular power spectra which will go under the same set of operations
                    (e.g. downsampling, but excluding the computation of covariance) as the original Cl. (*)
            (*) zeroth axes must refer to Monte Carlo realizations.

        Returns:

            dictionary of arrays, corresponding to (downsampled) angular wavenumbers, angular power spectra,
            covariance and correlation matrices.
        """

        assert isinstance(ell, np.ndarray) and np.isfinite(ell).all() and np.all(ell >= 0.0)
        assert np.all(ell == np.sort(ell))
        assert isinstance(cl, np.ndarray)
        assert ell.size and cl.size
        assert (ell.ndim == 1) and (cl.ndim == 2)

        if ell.shape[0] != cl.shape[-1]:
            raise RuntimeError('ell and cl do not match in shape!')

        assert (isinstance(nl, int) and (nl >= 2)) or (nl is None)

        if (cl_aux is not None) and (cl.shape != cl_aux.shape):
            raise RuntimeError('cl_aux and cl do not match in shape!')

        if nl is not None:
            nl = max(ell.size//nl, 1)
        else:
            nl = 1

        # Downsampling the power spectra.
        _lx, _clx = fx.downsample_cl(ell, cl, nl)

        cov = np.cov(_clx, rowvar=False)

        if cov.size > 1:
            s = np.sqrt(np.diag(cov))
            corr = cov / s[:,None] / s[None,:]      # NaN if div by zero.
        else:
            corr = 1.0

        ret = {'ell': _lx,
               'cl': np.mean(_clx, axis=0),         # Averaging MC realizations.
               'cov': cov,
               'corr': corr}

        if cl_aux is not None:
            _lx_aux, _clx_aux = fx.downsample_cl(ell, cl_aux, nl)

            assert np.all(_lx_aux == _lx)

            ret.update({'cl_aux': np.mean(_clx_aux, axis=0)})

        return ret

    @staticmethod
    def make_mask(ell, cl=None, l_max=None, cl_min=None):
        """
        This static method makes a boolean mask for angular power spectra.

        Args:

            ell: (1-d array) angular wavenumbers.
            cl: (1-d array) if not None, power spectrum.
            l_max: (int or float) if not None, maximum angular wavenumber beyond which the power spectrum is ignored.
                   It supersedes Cl_min.
            cl_min: (float) between 0 and 1, if not None, minimum fraction of maximum power below which the power
                    spectrum is ignored.  It assumes that Cl is a decreasing function of l.

        Returns:

            1-d array of boolean mask.
        """

        assert isinstance(ell, np.ndarray) and np.isfinite(ell).all() and np.all(ell >= 0.0)
        assert np.all(ell == np.sort(ell))

        nv = 0
        if l_max is not None:
            assert isinstance(l_max, float)
            nv += 1
        if cl_min is not None:
            assert isinstance(cl_min, float)
            nv += 1

        assert nv <= 1, 'Either one of l_max or cl_min may be specified, not both!'

        if (cl_min is not None) and ((not (0.0 <= cl_min <= 1.0)) or (cl is None)):
            raise RuntimeError('Invalid cl_min or cl!')
        if (l_max is not None) and (l_max <= 0.0):
            raise RuntimeError('Invalid l_max!')

        if l_max is not None:
            return ell <= l_max
        elif cl_min is not None:
            _cl_min = cl_min * np.max(cl)

            mask = (cl >= _cl_min)

            mask_i = [i for (i,j) in enumerate(mask) if j]
            mask[:np.max(mask_i)] = True

            return mask
        else:
            mask = (np.ones_like(ell) == 1.0)
            return mask

    def __test_all(self):
        self.__test_read_file()
        self.__test_snr()
        self.__test_snr_gaussian()
        self.__test_nl_gaussian()
        self.__test_covariance()
        self.__test_make_mask()

    @staticmethod
    def __test_read_file(file_path=fx.data_path('test_simstats.h5',envar='FRBXDATA'), dataset=None):
        if dataset is None:
            dataset = ['ff', 'gg', 'fg']

        stats = fx.simstats(file_path, dataset)

        for i, d in enumerate(dataset):
            assert hasattr(stats, 'm')
            assert isinstance(stats.m, dict)
            assert np.all(d == stats.dataset[i])

            _a = stats.read_file(d)
            for k, v in _a.items():
                s = 0
                assert k in stats.m[d]
                if k != 'axes':
                    if not s:
                        s += v.size
                    else:
                        assert s == v.shape[-1]
                        assert s == stats.m[d][k].shape[-1]

    def __test_snr(self, niter=10):
        cl = np.linspace(0.0, 9.0, 10)
        for _ in range(niter):
            cov = rand.ranf((10,10))
            cov = fx.fit.jcov(cov, mode='svd')

            for c in [False, True]:
                ret = self.snr(cl, cov, c)
                assert np.isfinite(ret).all(), ret
                assert np.all(ret >= 0.0)
                if c:
                    assert ret.size == 10

    def __test_snr_gaussian(self, nl=100):
        lvec = np.logspace(0, 5, nl)
        f_sky = rand.ranf()
        snr_expected = np.sqrt(f_sky * (lvec[-1]**2 - lvec[0]**2))

        cl_fg = np.ones(nl)
        cl_ff = np.ones(nl)
        cl_gg = np.ones(nl)
        for c in [True, False]:
            snr = self.snr_gaussian(lvec, cl_fg, cl_ff, cl_gg, f_sky, cumulative=c)
            snr_interp = self.snr_gaussian(lvec, cl_fg, cl_ff, cl_gg, f_sky, cumulative=c, interp=True)

            if c:
                assert np.isclose(snr[1][-1], snr_expected, rtol=10.0/nl, atol=1.0e-16)
                assert np.isclose(snr_interp[1][-1], snr_expected, rtol=1.0e-6/nl, atol=1.0e-16)
            else:
                assert np.isclose(snr, snr_expected, rtol=10.0/nl, atol=1.0e-16)
                assert np.isclose(snr_interp, snr_expected, rtol=1.0e-6/nl, atol=1.0e-16)

        cl_fg = rand.uniform(1.e-10, 1.0e-6, nl)
        cl_ff = rand.uniform(1.e-8, 1.0e-3, nl)
        cl_gg = rand.uniform(1.e-8, 1.0e-3, nl)

        cl_ff[rand.randint(nl, size=nl//20)] = 0.0
        cl_gg[rand.randint(nl, size=nl//20)] = 0.0

        for c in [True, False]:
            snr = self.snr_gaussian(lvec, cl_fg, cl_ff, cl_gg, f_sky, cumulative=c)
            snr_interp = self.snr_gaussian(lvec, cl_fg, cl_ff, cl_gg, f_sky, cumulative=c, interp=True)

            if c:
                assert np.isclose(snr[1][-1], snr_interp[1][-1], rtol=0.5, atol=1.0e-16)
            else:
                assert np.isclose(snr, snr_interp, rtol=0.5, atol=1.0e-16)

    def __test_nl_gaussian(self, nl=100):
        lvec = np.arange(1, nl+1, dtype=np.float64)

        l_high = lvec + 0.5
        l_low = lvec - 0.5
        b = l_high**2 - l_low**2

        x = rand.ranf(nl)
        y = rand.ranf(nl)
        f_sky = rand.ranf()

        a = self.nl_gaussian(lvec, x, y, f_sky)
        e = np.sqrt(x * y / f_sky / b)

        for i in range(nl):
            assert np.isclose(a[i], e[i], rtol=1.0e-10, atol=0.0)

    def __test_covariance(self, nl=10):
        lvec = np.ones(nl)
        cl = np.ones((10000, nl))
        cl[:,1:] = 0.0

        ret = self.covariance(ell=lvec, cl=cl)

        assert np.all(ret['ell'] == 1.0)
        assert (ret['cl'][0] == 1.0) and np.all(ret['cl'][1:] == 0.0)
        assert np.all(ret['cov'] == 0.0)
        assert not np.isfinite(ret['corr']).all()

        for i in range(10):
            cl[:,i] = rand.normal(loc=i, scale=i, size=10000)

        cl_aux = np.ones_like(cl)
        ret = self.covariance(ell=lvec, cl=cl, cl_aux=cl_aux)

        e = np.linspace(0.0, 9.0, nl)
        var = np.diag(ret['cov'])**0.5

        assert np.isclose(ret['cl'], e, rtol=0.1, atol=0.0).all()
        assert np.isclose(var, e, rtol=0.1, atol=0.0).all()

    def __test_make_mask(self, lmin=0.0, lmax=99.0, nl=100):
        ell = np.linspace(lmin, lmax, nl)
        cl = ell.copy()

        e = np.zeros_like(ell, dtype=bool)
        e[:71] = True

        a1 = self.make_mask(ell=ell, cl=cl, cl_min=70./max(cl) + 1.0e-6)
        a2 = self.make_mask(ell=ell, cl=cl, l_max=70.)
        assert np.all((e == (~a1)) == (~(e * a2)))
