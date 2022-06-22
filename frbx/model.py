import numpy as np
import scipy.interpolate
import frbx as fx

try:
    import pygedm
except ImportError:
    pass


class cl_models:
    """
    This class contains various models of angular power spectrum.

    Members:

        self.lchar: (int or float) specifies a fixed lchar for self.exp.
        self.exp: (method) C_l = alpha * exp(-l^2 / lchar^2).
        cl_models.l2: (static method) C_l = a / (1 + b*l^2).
    """

    def __init__(self, lchar=None):
        """
        Constructor arguments:

            lchar: (int or float) if not None, a fixed lchar which supersedes p[1] in self.exp.

        Raises:

            AssertionError: invalid input arg.
        """

        assert (lchar is None) or isinstance(lchar, (int, float))
        self.lchar = lchar

    def exp(self, l, p):
        """
        This method computes: C_l = alpha * exp(-l^2 / lchar^2).

        Args:

            l: (int, float or n-d array) angular wavenumbers.
            p: (list or 1-d array) model parameters [alpha, lchar].

        Returns:

            float or n-d array of floats, depending on the input 'l'.

        Raises:

            AssertionError: invalid input args.
        """

        assert isinstance(l, (int, float, np.ndarray))
        assert isinstance(p, (list, np.ndarray))
        assert len(p) == 2

        lchar = p[1] if self.lchar is None else self.lchar

        x = (l / lchar)**2.0
        ret = p[0] * np.exp(-x)

        return ret

    @staticmethod
    def l2(l, p):
        """
        This static method computes: C_l = a / (1 + b*l^2).

        Args:

            l: (int, float or n-d array) angular wavenumbers.
            p: (list or 1-d array) model parameters [a, b].

        Returns:

            1-d array of floats.

        Raises:

            AssertionError: invalid input args.
        """

        assert isinstance(l, (float, np.ndarray))
        assert isinstance(p, (list, np.ndarray))
        assert len(p) == 2

        ret = p[0] / (1.0 + p[1]*l**2.0)

        return ret


class gal_dm:
    """
    Computes the max galactic DMs for arrays of equatorial coordinates.
    TODO: Spherical interpolation; the current version interpolates values over a flat grid which is not quite accurate.
    """

    def __init__(self):
        # (n, (ra, dec, dm, dm_err))
        ymw16 = np.load(fx.data_path('archive/maps/YMW16_map.npy'))
        ne01 = np.load(fx.data_path('archive/maps/NE2001_map.npy'))

        # (ra, dec)
        self.ymw16 = scipy.interpolate.LinearNDInterpolator(ymw16[:,:2], ymw16[:,2], fill_value=-99)
        self.ne01 = scipy.interpolate.LinearNDInterpolator(ne01[:,:2], ne01[:,2], fill_value=-99)

    def __call__(self, ra, dec, mode='ymw16'):
        ra, dec = np.asarray(ra), np.asarray(dec)

        valid = np.logical_and((ra != -99), (dec != -99))
        _ra, _dec = ra[valid], dec[valid]
        try:
            fx.utils.sanity_check_lon_lat_arrays(_ra, _dec)
        except AssertionError as err:
            raise RuntimeError(f'gal_dm: {err}\nra={_ra}\ndec={_dec}')

        d1 = self.ymw16(ra, dec)
        d2 = self.ne01(ra, dec)

        if mode == 'ymw16':
            return d1 if (d1.size > 1) else float(d1)
        elif mode == 'ne01':
            return d2 if (d2.size > 1) else float(d2)
        elif 'pygedm' in mode:
            dist = 5.5e4        # pc
            method = mode.split('_')[1]

            l, b = fx.utils.convert_ra_dec_to_l_b(ra, dec)

            if isinstance(l, float):
                if method == 'yt2020':
                    # both -> (disk + spherical) halo
                    return pygedm.yt2020.calculate_halo_dm(l, b, 'both').value
                else:
                    return pygedm.dist_to_dm(l, b, dist, method=method)[0].value
            else:
                ret = np.full_like(l, -99)
                for i, v in enumerate(zip(l,b)):
                    if method == 'yt2020':
                        ret[i] = pygedm.yt2020.calculate_halo_dm(l, b, 'both')
                    else:
                        ret[i] = pygedm.dist_to_dm(v[0], v[1], dist, method=method)[0].value
                return ret
        else:
            return np.maximum(d1, d2)
