import numpy as np
import scipy.optimize
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
import astropy.units as u
import frbx as fx


class cosmology:
    """
    Constructs a flat Lambda-CDM cosmology.

    Implicit Units:

        Mass:                   M_sun/h
        Distance:               Mpc/h (comoving)
        Spatial wavenumber:     h/Mpc (comoving)
        Dispersion Measure:     pc/cm^3 (comoving)

    Members:

        self.base: (obj) base cosmology.
        self.ne0: (float) in cm^(-3), physical number density of free electrons at z=0.
        self.zmax: (float) global max redshift.
        self.dm_igm_interp: (interp obj) interpolated version of self.dm_igm.
        self.dm_igm: (method) integrated DM due to IGM.
        self.z_at_d: (method) returns the redshift at a given DM.
        self._diff_dm_igm: (helper method) differential DM due to IGM.
    """

    def __init__(self, ne0=2.13e-7, H0=67, Om0=0.315, Ob0=0.048, Tcmb0=2.726*u.K,
                 Neff=3.046, m_nu=0.02*u.eV, zmax=20.0, **kwargs):
        """
        Constructor arguments:

            ne0: (float) in cm^(-3), physical number density of free electrons at z=0.
            H0: (float or astropy Quantity) Hubble constant at z=0.
            Om0: (float) Omega matter at z=0.
            Ob0: (float) Omega baryons at z=0.
            Tcmb0: (astropy Quantity) CMB temperature at z=0.
            Neff: (float) effective number of neutrinos.
            m_nu: (astropy Quantity) mass of a neutrino.
            zmax: (float) global max redshift.
            **kwargs: optional astropy cosmology parameters.
        """

        assert isinstance(ne0, float) and (0.0 < ne0 < 1.0)
        assert isinstance(zmax, (float, int)) and (3.0 <= zmax <= 20.0)

        self.base = FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0, Neff=Neff, m_nu=m_nu, **kwargs)
        self.ne0 = ne0
        self.zmax = zmax

        try:
            self.dm_igm_interp = fx.read_pickle(fx.data_path('archive/pkls/dm_igm_interp.pkl'))
        except FileNotFoundError:
            _z = np.linspace(0.0, self.zmax, 1000)
            _dm = np.asarray([self.dm_igm(i) for i in _z])

            _dm_igm_interp = fx.spline(_z, _dm)

            fx.write_pickle(fx.data_path('archive/pkls/dm_igm_interp.pkl', mode='w'), _dm_igm_interp)
            self.dm_igm_interp = _dm_igm_interp

    def dm_igm(self, z):
        """Returns the integrated DM (due to IGM) for a source at redshift 'z'."""

        assert isinstance(z, (int, float)) and (z >= 0.0)

        ret = fx.quad(lambda x: self._diff_dm_igm(x), 0.0, z)
        assert ret >= 0.0

        return ret

    def z_at_d(self, d):
        """Returns the redshift at a given DM 'd' (float)."""

        assert isinstance(d, (int, float)) and (d >= 0.0)

        try:
            ret = scipy.optimize.brentq(lambda z: self.dm_igm(z) - d, 0.0, self.zmax)
        except ValueError as err:
            raise RuntimeError(f'{err}\ncosmology.z_at_d: d={d}')

        assert np.isfinite(ret)
        return ret

    def _diff_dm_igm(self, z):
        """Returns the differential DM (due to IGM) for a source at redshift 'z'."""

        assert isinstance(z, (int, float)) and (z >= 0.0)

        ret = const.c * (self.ne0/u.cm**3) * (1+z) / self.base.H(z)
        ret = ret.to(u.pc/u.cm**3).value

        return ret
