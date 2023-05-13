import numpy as np
import healpy
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
    """Computes the max galactic DMs for arrays of equatorial coordinates."""

    def __init__(self, nside=4):
        self.nside = nside

        # (n, (ra, dec, dm, dm_err))
        ymw16 = np.load(fx.data_path('archive/maps/YMW16_map.npy'))
        # (n, (ra, dec, dm))
        ne01 = np.load(fx.data_path('archive/maps/NE2001_map.npy'))

        # (ra, dec)
        self.ymw16 = scipy.interpolate.LinearNDInterpolator(ymw16[:,:2], ymw16[:,2], fill_value=-99)
        self.ne01 = scipy.interpolate.LinearNDInterpolator(ne01[:,:2], ne01[:,2], fill_value=-99)

        l_deg, b_deg = fx.utils.convert_ra_dec_to_l_b(ymw16[:,0], ymw16[:,1])
        self.ymw16_hp = fx.utils.make_healpix_map_from_catalog(
                      self.nside, l_deg, b_deg, weight=ymw16[:,2], interpolate=False, invar=1.0/ymw16[:,3]**2)

        l_deg, b_deg = fx.utils.convert_ra_dec_to_l_b(ne01[:,0], ne01[:,1])
        self.ne01_hp = fx.utils.make_healpix_map_from_catalog(
                     self.nside, l_deg, b_deg, weight=ne01[:,2], interpolate=False, invar=1.0)

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

        elif mode == 'ymw16_hp':
            l, b = fx.utils.convert_ra_dec_to_l_b(ra, dec)
            d1_hp = healpy.pixelfunc.get_interp_val(self.ymw16_hp, l, b, lonlat=True)
            return d1_hp if (d1_hp.size > 1) else float(d1_hp)

        elif mode == 'ne01_hp':
            l, b = fx.utils.convert_ra_dec_to_l_b(ra, dec)
            d2_hp = healpy.pixelfunc.get_interp_val(self.ne01_hp, l, b, lonlat=True)
            return d2_hp if (d2_hp.size > 1) else float(d2_hp)

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
