import warnings
import numpy as np
import numpy.random as rand
import astropy.units as u
from astropy import constants as const
import scipy.optimize
from scipy.special import sici
import camb
from astropy.cosmology import FlatLambdaCDM, z_at_value
import matplotlib.pyplot as plt
import frbx as fx
from frbx.configs import tiny, eps


class cosmology_base:
    """
    Constructs a flat Lambda-CDM cosmology.

    Implicit Units:

        Mass:                   M_sun/h
        Distance:               Mpc/h (comoving)
        Spatial wavenumber:     h/Mpc (comoving)
        Dispersion Measure:     pc/cm^3 (comoving)

    Members:

        self.base: (obj) base cosmology from astropy.
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
            raise RuntimeError(f'{err}\ncosmology_base.z_at_d: d={d}')

        assert np.isfinite(ret)
        return ret

    def _diff_dm_igm(self, z):
        """Returns the differential DM (due to IGM) for a source at redshift 'z'."""

        assert isinstance(z, (int, float)) and (z >= 0.0)

        ret = const.c * (self.ne0/u.cm**3) * (1+z) / self.base.H(z)
        ret = ret.to(u.pc/u.cm**3).value

        return ret


class cosmology(cosmology_base):
    """
    Constructs a flat Lambda-CDM cosmology containing FRB's (f), galaxies (g) and electrons (e).

    Implicit units:

        Mass:                   M_sun/h.
        Distance:               Mpc/h (comoving).
        Spatial wavenumber:     h/Mpc (comoving).
        Dispersion Measure:     pc/cm^3 (comoving).

    Members:

        self.config: (obj) base configs.
        self.pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
        self.survey_frb: (obj) FRB survey configs.
        self.survey_galaxy: (obj) galaxy survey configs.
        self.frb_par: (list) lists of FRB model parameters.
        self.interp_zmin: (float) local min redshift bound for interp obj that involve angular wavenumbers.
        self.interp_zmax: (float) local max redshift bound for interp obj that involve angular
                          wavenumbers and FRB quantities.
        self.z_to_chi: (function) interpolator for converting redshift to comoving distance.
        self.chi_to_z: (function) interpolator for converting comoving distance to redshift.
        self.dvoz: (function) interpolated differential volume factor dV(z)/dOmega/dz.
        self.kmin: (float) min k above which self.p_k is defined without extrapolation.
        self._p_k: (function) camb interpolator (k,z) for linear matter power spectrum.
        self.hmf: (obj) instance of fx.halo_mass_function.
        self.interp_ng_3d: (function) interpolated 3-d comoving number density of galaxies (z). (*)
        self.dndz_galaxy: (lambda) interpolated differential number of galaxies per unit steradian. (*)
        self.zbin_delim_galaxy: (1-d array) redshift bin delimiters for the current galaxy survey. (*)
        self.n2d_galaxy: (1-d array) 2-d number counts between self.zbin_delim_galaxy delimiters. (*)
        self.dn2d_galaxy: (1-d array) errors in self.n2d_galaxy. (*)
        self.m_g: (function) interpolator of redshift-dependant min halo mass for hosting galaxies.
        self.dndz_frb: (list) interpolated differential number of FRB's per unit steradian in self.frb_par models.
        self.eta: (list) function (z) for normalizing FRB counts in self.frb_par models.
        self.psg: (list) probability for single galaxies to host FRB's.
        self.kk: (1-d array) log-spaced spatial wavenumbers.
        self.ll: (1-d array) log-spaced angular wavenumbers over which radial weight functions are interpolated.
        self.interp_ngg_3d: (lambda) interpolated 3-d comoving number density of (g,g) pairs in the same halo (z).
        self.interp_nge_3d: (lambda) interpolated 3-d comoving number density of (g,e) pairs in the same halo (z).
        self.interp_bias_g: (interp obj) galaxy bias (k,z).
        self.interp_bias_e: (interp obj) electron bias (k,z).
        self.interp_p1_ge: (lambda) interpolated 1h term of the 3-d galaxy-electron cross power spectrum (k,z).
        self.interp_p2_ge: (lambda) interpolated 2h term of the 3-d galaxy-electron cross power spectrum (k,z).
        self.interp_w_g: (interp obj) interpolated radial weight function (two-halo term) for galaxies (z,l).
        self.init_camb: (method) initializes (self.kmin, self._p_k).
        self.p_k: (method) returns the linear matter power spectrum (k,z).
        self.bin: (method) bins (DM,z) space.
        self.delta_z: (method) uncertainty in redshift due to an uncertainty in DM.
        self.rho_nfw: (method) Fourier transform of the NFW halo profile.
        self.l_to_k: (method) converts (l,z) to k.
        self.n_g: (method) expected number of galaxies in a halo.
        self.gamma_e: (method) quantity gamma for electrons (z).
        self.w_x: (method) radial weight function.
        cosmology.n_x: (static method) wraps a callable HOD f(z,m,xmin,xmax) given the constraint (xmin,xmax).
        cosmology.dm_h: (static method) random host DM from a log-normal distribution.
        cosmology.frb_dndz: (static method) redshift distribution of FRB's.
        self._frb_dndz_interp: (helper method) interpolated normalized redshift distribution of FRB's.
        self._vol_s: (helper method) comoving volume in a redshift shell.
        self._dvoz: (helper method) returns the differential volume factor dV(z)/dOmega/dz.
        self._chi_to_z: (helper method) interpolates line-of-sight comoving distance over a range of redshifts.
        self._cosmology__init_galaxy: (special method) initializes members which are marked by (*).
        self._cosmology__m_g: (special method) returns self.m_g.
        self._cosmology__eta: (special method) returns self.eta.
        self._cosmology__psg: (special method) returns self.psg.
        self._cosmology__ng_3d: (special method) returns self.interp_ng_3d.
        self._cosmology__ngg_3d: (special method) returns self.interp_ngg_3d.
        self._cosmology__nge_3d: (special method) returns self.interp_nge_3d.
        self._cosmology__bias_g: (special method) returns self.interp_bias_g.
        self._cosmology__bias_e: (special method) returns self.interp_bias_e.
        self._cosmology__p1_ge: (special method) returns self.interp_p1_ge.
        self._cosmology__p2_ge: (special method) returns self.interp_p2_ge.
        self._cosmology__test_*.
    """

    def __init__(self, config, pkl_dir, construct_ge_obj=True, **kwargs):
        """
        Constructor arguments:

            config: (obj) instance of fx.configs.
            pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
            construct_ge_obj: (bool) whether to construct galaxy-electron objects (computationally expensive!).
            **kwargs: optional astropy cosmology parameters.
        """

        assert isinstance(config, fx.configs)
        assert isinstance(pkl_dir, str) and pkl_dir.endswith('/')

        self.config = config
        self.pkl_dir = pkl_dir

        super(cosmology, self).__init__(zmax=self.config.interp_zmax, **kwargs)

        self.survey_frb = self.config.survey_frb
        self.survey_galaxy = self.config.survey_galaxy
        self.frb_par = self.config.frb_par

        self.interp_zmin = max(self.config.interp_zmin*10, 1.0e-4)
        assert 0.0 <= self.config.interp_zmin <= self.interp_zmin

        if self.survey_frb.dmax is not None:
            self.interp_zmax = max(self.config.md * self.config.zmax, self.z_at_d(self.survey_frb.dmax))
        else:
            self.interp_zmax = self.config.md * self.config.zmax

        self.interp_zmax += self.config.zpad_interp_zmax
        assert self.interp_zmin < self.interp_zmax < (self.config.interp_zmax-self.config.zpad_eps)
        assert self.config.fn_zmax <= (self.interp_zmax - self.config.zpad_interp_zmax)
        assert self.survey_galaxy.zmax < self.interp_zmax
        assert self.interp_zmin < self.config.zmin < self.config.zmax < self.interp_zmax

        _path = self.pkl_dir + 'chi.pkl'
        try:
            z_to_chi, chi_to_z = fx.read_pickle(_path)
        except OSError as err:
            print(err)
            z_to_chi = self._chi_to_z(zmin=0.0, zmax=self.config.interp_zmax, inverse=True)
            chi_to_z = self._chi_to_z(zmin=0.0, zmax=self.config.interp_zmax)
            fx.write_pickle(_path, [z_to_chi, chi_to_z])

        def f0(f):
            def _f0(z, _f=f):
                z = np.asarray(z)
                mask = (z > 0.0)
                _ret = np.zeros_like(z)
                _ret[mask] += _f(z)[mask]
                return _ret
            return _f0

        self.z_to_chi = f0(z_to_chi)
        self.chi_to_z = f0(chi_to_z)

        _zz = fx.lspace(0.0, self.config.interp_zmax, 10000, log=True)
        _dv = [self._dvoz(z) for z in _zz]
        dvoz = fx.spline(_zz, _dv)
        self.dvoz = f0(dvoz)

        self._kmin = None
        self._interp_p_k = None
        self.kmin, self._p_k = self.init_camb

        self.hmf = fx.halo_mass_function(
                 zmin=0.0, zmax=self.config.interp_zmax-self.config.zpad_eps,
                 m_min=self.config.m_min/100., m_max=self.config.m_max*100., pkl_dir=self.pkl_dir,
                 cosmo=self.base, p_k=self.p_k, kmin=self.kmin/10., kmax=self.config.kmax_extrap)

        self.__init_galaxy()
        self.m_g = self.__m_g()

        self.dndz_frb = []
        for par in self.frb_par:
            dndz_frb = self._frb_dndz_interp(zmin=0.0, zmax=self.interp_zmax, p=par[1], alpha=par[2],
                                             n_frb=par[0]/(4*np.pi*self.survey_frb.f_sky), z_log=par[6])
            self.dndz_frb.append(dndz_frb)

        _path = self.pkl_dir + 'eta.pkl'
        try:
            eta_list = fx.read_pickle(_path)
        except OSError as err:
            print(err)
            eta_list = self.__eta()
            fx.write_pickle(_path, eta_list)

        def f1(f):
            def eta(z, _f=f):
                return max(0.0, _f(z))
            return eta

        self.eta = []
        for _eta in eta_list:
            self.eta.append(f1(_eta))

        self.psg = self.__psg()

        self.kk = fx.lspace(self.kmin/10., self.config.kmax_extrap-self.config.kpad_eps,
                            self.config.interp_nstep_kk, log=True)

        _ll_min = max(self.kk[0] * self.z_to_chi(self.interp_zmin), self.config.ll_min)
        _ll_max = min(self.kk[-1] * self.z_to_chi(self.interp_zmax), self.config.ll_max)

        self.ll = fx.lspace(_ll_min, _ll_max, self.config.interp_nstep_ll, log=True)

        kmin, kmax = self.kk[0], self.kk[-1]
        for _ll in self.ll:
            kzmin = self.l_to_k(_ll, self.interp_zmin)
            kzmax = self.l_to_k(_ll, self.interp_zmax)

            assert kmin <= kzmin <= kmax, f'{_ll}: {kmin} <= {kzmin} <= {kmax}'
            assert kmin <= kzmax <= kmax, f'{_ll}: {kmin} <= {kzmax} <= {kmax}'

        self.interp_ngg_3d = None
        self.interp_nge_3d = None
        self.interp_bias_g = None
        self.interp_bias_e = None
        self.interp_p1_ge = None
        self.interp_p2_ge = None
        self.interp_w_g = None

        if construct_ge_obj:
            ###################################################################################################
            #
            # (g,e) objects.

            _path = self.pkl_dir + 'ge_objs.pkl'
            try:
                ge_obj = fx.read_pickle(_path)
            except OSError as err:
                print(err)
                ### self.interp_ngg_3d
                ngg_3d = np.asarray([self.__ngg_3d(i) for i in self.config.zz_g])
                ngg_3d[ngg_3d < tiny] = tiny
                log_ngg_3d = np.log(ngg_3d)

                _interp_log_ngg_3d = fx.spline(self.config.zz_g, log_ngg_3d)
                print('self.interp_ngg_3d done.')

                ### self.interp_nge_3d
                nge_3d = np.asarray([self.__nge_3d(i) for i in self.config.zz_g])
                nge_3d[nge_3d < tiny] = tiny
                log_nge_3d = np.log(nge_3d)

                _interp_log_nge_3d = fx.spline(self.config.zz_g, log_nge_3d)
                print('self.interp_nge_3d done.')

                ### self.interp_bias_g
                _bias_g = np.zeros((self.config.interp_nstep_kk, self.config.interp_nstep_zz_g))
                for i, kk in enumerate(self.kk):
                    for j, z in enumerate(self.config.zz_g):
                        print(f'(i={i},j={j})')
                        try:
                            ret = self.__bias_g(kk, z)
                        except RuntimeError as err:
                            print(err)
                            print(f'self.interp_bias_g ({kk},{z}): default mc failed. Trying mc=2.')
                            try:
                                ret = self.__bias_g(kk, z, 2)
                            except RuntimeError as err:
                                print(err)
                                print(f'self.interp_bias_g ({kk},{z}) -> 1.0')
                                ret = 1.0

                        _bias_g[i,j] = ret

                self.interp_bias_g = fx.spline(self.kk, self.config.zz_g, _bias_g)
                print('self.interp_bias_g done.')

                ### self.interp_bias_e
                _bias_e = np.zeros((self.config.interp_nstep_kk, self.config.interp_nstep_zz_e))
                for i, kk in enumerate(self.kk):
                    for j, z in enumerate(self.config.zz_e):
                        print(f'(i={i},j={j})')
                        try:
                            ret = self.__bias_e(kk, z)
                        except RuntimeError as err:
                            print(err)
                            print(f'self.interp_bias_e ({kk},{z}): default mc failed. Trying mc=2.')
                            try:
                                ret = self.__bias_e(kk, z, 2)
                            except RuntimeError as err:
                                print(err)
                                print(f'self.interp_bias_e ({kk},{z}) -> 1.0')
                                ret = 1.0

                        _bias_e[i,j] = ret

                self.interp_bias_e = fx.spline(self.kk, self.config.zz_e, _bias_e)
                print('self.interp_bias_e done.')

                ### self.interp_p1_ge
                _p1_ge = np.zeros((self.config.interp_nstep_kk, self.config.interp_nstep_zz_g))
                for i, kk in enumerate(self.kk):
                    for j, z in enumerate(self.config.zz_g):
                        print(f'(i={i},j={j})')
                        try:
                            ret = self.__p1_ge(kk, z)
                        except RuntimeError:
                            ret = 0.0

                        _p1_ge[i,j] = ret

                _p1_ge[_p1_ge < tiny] = tiny
                log_p1_ge = np.log(_p1_ge)

                _interp_log_p1_ge = fx.spline(self.kk, self.config.zz_g, log_p1_ge)
                print('self.interp_p1_ge done')

                ### self.interp_p2_ge
                _p2_ge = np.zeros((self.config.interp_nstep_kk, self.config.interp_nstep_zz_g))
                for i, kk in enumerate(self.kk):
                    for j, z in enumerate(self.config.zz_g):
                        print(f'(i={i},j={j})')
                        try:
                            ret = self.__p2_ge(kk, z) * fx.halo_mass_function.w_cutoff(kk / self.config.kmax_extrap)
                        except RuntimeError:
                            ret = 0.0

                        _p2_ge[i,j] = ret

                _p2_ge[_p2_ge < tiny] = tiny
                log_p2_ge = np.log(_p2_ge)

                _interp_log_p2_ge = fx.spline(self.kk, self.config.zz_g, log_p2_ge)
                print('self.interp_p2_ge done.')

                ### self.interp_w_g
                zz_g = fx.lspace(max(self.interp_zmin, self.config.survey_galaxy.zmin),
                                 self.survey_galaxy.zmax, self.config.interp_nstep_zz_g, True)

                _w_g = np.zeros((self.config.interp_nstep_zz_g, self.config.interp_nstep_ll))
                _w_y = self.w_x(n_x=self.n_g, m_min=self.m_g)

                for i, z in enumerate(zz_g):
                    for j, l in enumerate(self.ll):
                        print(f'(i={i},j={j})')
                        _w_g[i,j] = _w_y(z, l)

                interp_w_g = fx.spline(zz_g, self.ll, _w_g)
                print('self.interp_w_g done.')

                ge_obj = [_interp_log_ngg_3d,
                          _interp_log_nge_3d,
                          self.interp_bias_g,
                          self.interp_bias_e,
                          _interp_log_p1_ge,
                          _interp_log_p2_ge,
                          interp_w_g]

                fx.write_pickle(_path, ge_obj)

            self.interp_ngg_3d = lambda redshift: max(0.0, np.exp(ge_obj[0](redshift)))
            self.interp_nge_3d = lambda redshift: max(0.0, np.exp(ge_obj[1](redshift)))
            self.interp_bias_g = ge_obj[2]
            self.interp_bias_e = ge_obj[3]
            self.interp_p1_ge = lambda wavenumber, redshift: np.exp(ge_obj[4](wavenumber, redshift))
            self.interp_p2_ge = lambda wavenumber, redshift: np.exp(ge_obj[5](wavenumber, redshift))
            self.interp_w_g = ge_obj[6]

    def __getstate__(self):
        """
        Specifies members that can be pickled.
        FIXME: Some nested callable obj cannot be pickled -> Currently pickling locally!
        """

        flag = ('_kmin', '_interp_p_k')
        return dict((k,v) for (k,v) in self.__dict__.items() if k not in flag)

    @property
    def init_camb(self):
        """Initializes (self.kmin, self._p_k)."""

        if not (hasattr(self, '_p_k') and hasattr(self, 'kmin')):
            camb_par = camb.CAMBparams()
            camb_par.WantTransfer = True
            camb_par.NonLinear = 'NonLinear_none'
            camb_par.set_cosmology(H0=self.base.H0.to(u.km / u.s / u.Mpc).value, tau=self.config.tau,
                                   omch2=(self.base.Om0 - self.base.Ob0) * self.base.h**2,
                                   ombh2=self.base.Ob0 * self.base.h**2, TCMB=self.base.Tcmb0.to(u.K).value,
                                   standard_neutrino_neff=self.base.Neff,
                                   mnu=self.base.m_nu.to(u.eV).value[0] * np.floor(self.base.Neff))

            camb_par.set_dark_energy()
            camb_par.InitPower.set_params(ns=self.config.ns, As=self.config.As, r=self.config.r)

            # hubble_units = True -> self.p_k outputs in (Mpc/h)^3.
            # k_hunit = True      -> self.p_k = P(k), where k is assumed to be in (h/Mpc).
            _p_k = camb.get_matter_power_interpolator(
                 camb_par, zmin=0.0, zmax=self.config.interp_zmax, nz_step=150,
                 kmax=self.config.kmax, nonlinear=False, extrap_kmax=self.config.kmax_extrap)

            # Using the default kmin.
            self._kmin = _p_k.kmin
            self._interp_p_k = _p_k.P

            return self._kmin, self._interp_p_k
        else:
            return self.kmin, self._p_k

    def p_k(self, k, z):
        """Returns the linear matter power spectrum.  Originally written by Kendrick Smith."""

        assert isinstance(k, float) and (0.0 <= k <= self.config.kmax_extrap)
        assert isinstance(z, float) and (0.0 <= z <= self.config.interp_zmax)

        if k >= 2*self.kmin:
            return self._p_k(kh=k, z=z)

        pk_extrap = (k / self.kmin) * self._p_k(kh=self.kmin, z=z)

        if k <= self.kmin:
            return pk_extrap

        pk = self._p_k(kh=k, z=z)

        # The weight 'w' is 1 for k <= self.kmin, and 0 for k >= 2*self.kmin.
        w = fx.halo_mass_function.w_cutoff(k / (2 * self.kmin))
        return w*pk_extrap + (1-w)*pk

    def bin(self, nz, nd, cutoff=True, edge=False, zmax=None):
        """
        Bins (DM,z) space.

        Args:

            nz: (int) total number of redshift bins.
            nd: (int) total number of extragalactic DM bins.
            cutoff: (bool) If True, then 'dd_z' bins (see below) are constrained by self.config.fn_zmax.
            edge: (bool) If True, upper bounds of FRB DM bins are aligned with edges of galaxy
                  redshift bins.
            zmax: (float) if not None, maximum redshift which supersedes all internal upper bounds.

        Returns:

            tuple (zz, dd, dd_z), lists of [min, max] bounds for (redshift, DM, DM converted to redshift) bins.
        """

        config_zmax = self.config.zmax if zmax is None else zmax

        zz = fx.slicer(self.config.zmin, config_zmax, nz)
        if (np.diff(zz) <= self.survey_galaxy.zerr).any():
            warnings.warn('cosmology.bin: redshift bins are smaller than the redshift error.')

        if zmax is None:
            if cutoff:
                fn_zmax = self.config.fn_zmax + self.config.zpad_eps
            else:
                fn_zmax = self.z_at_d(self.config.dmax)
        else:
            fn_zmax = zmax

        fn_zmin = max(self.z_at_d(self.config.dmin), self.config.zmin)

        if nd == 1:
            dd_z = [np.asarray([fn_zmin]), np.asarray([fn_zmax])]
            dd = self.dm_igm_interp(dd_z)
            if cutoff:
                dd[1][0] = self.config.dmax
        else:
            if edge:
                dd = self.dm_igm_interp(zz)
                dd[0][0] = self.config.dmin

                ndiff = nd - nz
                if ndiff > 0:
                    _dmax_md = float(self.dm_igm_interp(self.config.md * config_zmax))
                    _dd_md = fx.slicer(dd[1][-1], _dmax_md, ndiff)
                    dd = np.append(dd, _dd_md, axis=1)
                elif nd != 1:
                    warnings.warn('cosmology.bin: nd <= nz.')

                dd[1][-1] = self.config.dmax
            else:
                dd = fx.slicer(self.config.dmin, self.config.dmax, nd)

            t0 = float(self.dm_igm_interp(self.config.zmin))
            dd_z = []
            for i in (0, 1):
                _dd_z = []
                for j, k in enumerate(dd[i]):
                    t = max(self.z_at_d(k), self.config.zmin)
                    t = min(t, fn_zmax)
                    if t == self.config.zmin:
                        dd[i][j] = t0
                    _dd_z.append(t)
                dd_z.append(np.asarray(_dd_z))

        return zz, list(dd), dd_z

    def delta_z(self, z, delta_DM):
        """
        This method returns the uncertainty in redshift due to an uncertainty in DM.

        Args:

            z: (float) redshift.
            delta_DM: (array or float) uncertainty in DM at the given redshift.

        Returns:

            array or float, depending on the type of delta_DM.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(delta_DM, (float, np.ndarray))
        assert np.all(delta_DM >= 0.0)

        den = self._diff_dm_igm(z)

        assert den > 0.0
        ret = delta_DM / den

        return ret

    def rho_nfw(self, k, z, m, normalized=False):
        """
        This method computes the Fourier transform of the NFW halo profile.

        Args:

            k: (float) spatial wavenumber.
            z: (float) halo redshift.
            m: (float) halo mass.
            normalized: (bool) whether to normalize by the halo mass.

        Returns:

            scalar corresponding to the Fourier transform of the profile density at k.
        """

        assert isinstance(k, float) and (k >= 0.0)
        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m >= 0.0), f'm ({type(m)}) = {m}'
        assert isinstance(normalized, bool)

        if k == 0.0:
            ret = m
        else:
            rho_vir = float(self.hmf.rho_vir(z))
            log_m = np.log(m)

            r = float(self.hmf.interp_r_vir(z, log_m))

            conc = float(self.hmf.interp_conc(z, log_m))
            r_s = r / conc

            _k = k * r_s

            c1 = 1 + conc
            _kc1 = _k * c1

            a = rho_vir * conc**3 / 3 / (np.log(c1) - conc/c1)

            # Factor in front of the main expression below.
            ret = 4 * np.pi * a * r_s**3

            _sici = np.asarray(sici(_kc1)) - np.asarray(sici(_k))
            ret *= ((-np.sin(_k*conc)/_kc1) + (np.cos(_k)*_sici[1]) + (np.sin(_k)*_sici[0]))

        if normalized:
            ret /= m

        return np.float64(ret)

    def l_to_k(self, ell, z):
        """
        This method converts an angular wavenumber to a spatial wavenumber at a given redshift.

        Args:

            ell: (float) angular wavenumber.
            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(ell, float) and (ell >= 0.0)
        assert isinstance(z, float) and (self.interp_zmin <= z <= self.interp_zmax)

        return (ell / self.z_to_chi(z)) if (ell > 0) else 0.0

    def n_g(self, z, m, zmin=None, zmax=None):
        """
        This method returns the expected number of galaxies in a halo of mass m, located at redshift z
        in the redshift shell (zmin, zmax).

        Args:

            z: (float) halo redshift.
            m: (float) halo mass.
            zmin: (float) if not None, min redshift value in the shell.
            zmax: (float) if not None, max redshift value in the shell.

        Returns:

            float, expected number of galaxies in the specified halo.
        """

        assert isinstance(m, float) and (m >= 1.0e4)

        constrain_z = (zmin is not None) and (zmax is not None)

        if constrain_z:
            assert isinstance(z, float)
            assert self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax
            assert isinstance(zmin, float) and isinstance(zmax, float)
            assert self.survey_galaxy.zmin <= zmin < zmax <= self.survey_galaxy.zmax

            if not (zmin <= z < zmax):
                return np.float64(0.0)

        m_g = self.m_g(z)
        assert m_g > 0.0

        ret = (m / m_g) if (m >= m_g) else 0.0
        assert ret >= 0.0

        return np.float64(ret)

    def gamma_e(self, z):
        """
        This method returns the quantity gamma for the electron field.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        n_x = self.interp_ng_3d(z) / self.hmf.rho_m
        assert np.isfinite(n_x)

        n_xy = self.interp_nge_3d(z) / self.interp_ngg_3d(z)
        assert np.isfinite(n_xy)

        ret = n_x * n_xy

        return ret

    def w_x(self, n_x, m_min):
        """
        This method returns the radial weight function, excluding nbar_x in its denominator, for X.

        Args:

            n_x: (callable) given (z,m), returns the expected number of X's.
            m_min: (lambda, interp obj, or float) min halo mass as a function of redshift.

        Returns:

            function (z,l).
        """

        assert callable(n_x)
        lambda_m_min = fx.utils.mz(m_min)

        def _int(z, l):
            _ret = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(z, m)
                           * self.hmf.interp_bias(z, m)
                           * n_x(z, np.exp(m))
                           * self.rho_nfw(self.l_to_k(l,z), z, np.exp(m), normalized=True),
                           lambda_m_min(z), np.log(self.config.m_max))

            return _ret

        return _int

    @staticmethod
    def n_x(f, xmin, xmax):
        """Wraps a callable HOD f(z,m,xmin,xmax) given the constraint (xmin,xmax)."""

        def ret(z, m):
            return f(z, m, xmin, xmax)

        return ret

    @staticmethod
    def dm_h(n, mu, sigma):
        """
        This static method returns N random samples from a standard log-normal distribution of host DMs.

        Args:

            n: (int) number of samples to be drawn.
            mu: (float) mean of the underlying normal distribution.
            sigma: (float) standard deviation of the underlying normal distribution.

        Returns:

            1-d array of N random floats.
        """

        assert (n == int(n)) and (n >= 0)
        assert isinstance(mu, float) and (mu >= 0.0)
        assert isinstance(sigma, float) and (sigma >= 0.0)

        return rand.lognormal(mean=mu, sigma=sigma, size=n)

    @staticmethod
    def frb_dndz(z, p, alpha):
        """
        This static method models the differential number density of FRB's as a function of redshift:

        dn/dz = z^p * exp(-alpha * z), where dn/dz is not normalized.

        Args:

            z: (scalar or array) redshift.
            p: (float) FRB model parameter.
            alpha: (float) FRB model parameter.

        Returns:

            scalar or array, depending on the type of input for redshift.
        """

        assert isinstance(p, float) and (p <= 10.0)
        assert isinstance(alpha, float) and (alpha >= 0.0)

        z = np.asarray(z)
        assert np.all(z >= 0.0)

        ret = z**p * np.exp(-alpha * z)
        assert np.isfinite(ret).all()

        return ret

    def _frb_dndz_interp(self, n_frb, p, alpha, zmin, zmax, z_log, interp_nstep=10000):
        """
        This helper method returns the interpolated differential number density
        of FRB's, normalized as a function of redshift.

        Args:

            n_frb: (float) 2-d angular number density of FRB's on the sky. (per unit steradian)
            p: (float) an FRB model parameter.
            alpha: (float) an FRB model parameter.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            z_log: (bool) If True, redshift is log-spaced.
            interp_nstep: (int) number of interpolation steps.

        Returns:

            function (z), number of FRB's per unit redshift per unit steradian.
        """

        assert isinstance(n_frb, float) and (n_frb >= 0.0)
        assert isinstance(p, float) and isinstance(alpha, float)
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert 0.0 <= zmin < self.config.zmin < self.config.zmax < zmax
        assert isinstance(z_log, bool)
        assert (interp_nstep == int(interp_nstep)) and (interp_nstep >= 10000)

        z_i = fx.lspace(zmin, zmax, interp_nstep, z_log)

        _dndz_i = self.frb_dndz(z_i, p, alpha)
        _dndz = fx.spline(z_i, _dndz_i)

        # Normalizing factor based on the parametrization of dndz.
        norm = fx.quad(lambda x: _dndz(x), self.config.zmin, self.config.fn_zmax)

        assert norm > 0.0
        _ret = [(n_frb/norm)*_dndz(z) for z in z_i]
        _ret = fx.spline(z_i, _ret)

        def ret(x, cutoff=True):
            """
            Returns the differential number density of FRB's at a given redshift 'x'.
            If the 'cutoff' (bool) argument is True, then the function always returns
            zero for input redshifts greater than 'self.config.fn_zmax'.
            """
            if cutoff and (x > self.config.fn_zmax):
                return 0.0
            else:
                return max(0.0, _ret(x))

        return ret

    def _vol_s(self, zmin, zmax):
        """
        This helper method computes the comoving volume in a redshift shell.

        Args:

            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.

        Returns:

            float.
        """

        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert 0.0 <= zmin <= zmax

        dv = self.base.comoving_volume(zmax) - self.base.comoving_volume(zmin)
        dv = dv.to(u.Mpc**3).value * self.base.h**3.

        return dv * self.config.f_sky

    def _dvoz(self, z):
        """
        This helper method computes the differential volume factor dV(z)/dOmega/dz.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (z >= 0.0)

        if z == 0.0:
            return 0.0
        else:
            chi = self.z_to_chi(z) * u.Mpc / self.base.h

            ret = const.c * chi**2 / self.base.H(z)
            ret = ret.to(u.Mpc**3).value * self.base.h**3.

            return ret

    def _chi_to_z(self, r=None, zmin=None, zmax=None, z_bound=1.0e-3, z_step=10000, inverse=False):
        """
        This helper method returns an interpolator which converts an array of line-of-sight
        comoving distance to redshift values, and vice versa.

        Args:

            r: optional (1-d array) of line-of-sight comoving distance in Mpc/h (if inverse is False)
               or redshifts (if inverse is True).
            zmin: (float) minimum redshift for the interpolation. If None, the value is derived from min 'r'.
            zmax: (float) maximum redshift for the interpolation. If None, the value is derived from max 'r'.
            z_bound: (float) is subtracted/added from/to the min/max redshifts for the interpolation,
                     only if 'zmin' and/or 'zmax' are None.
            z_step: (int) number of logarithmic steps (base 10) for the interpolation.
            inverse: (bool) whether to return an interpolator for converting z to Mpc/h.

        Returns:

            interpolator object.
        """

        if (r is None) and ((zmin is None) or (zmax is None)):
            raise RuntimeError('insufficient input: either r or (zmin and zmax) should be supplied.')

        assert isinstance(z_bound, float) and (z_bound <= 1e-3)
        assert (z_step == int(z_step)) and (z_step >= 100)
        assert isinstance(inverse, bool)

        if zmin is None:
            if not inverse:
                # self.base.comoving_distance is in Mpc, but r is in Mpc/h.
                zmin = z_at_value(self.base.comoving_distance, np.min(r) * u.Mpc / self.base.h)
            else:
                zmin = np.min(r)

            zmin -= z_bound

        if zmax is None:
            if not inverse:
                # self.base.comoving_distance is in Mpc, but r is in Mpc/h.
                zmax = z_at_value(self.base.comoving_distance, np.max(r) * u.Mpc / self.base.h)
            else:
                zmax = np.max(r)

            zmax += z_bound

        zmin = max(zmin, 0.0)
        assert zmin < zmax

        zgrid = np.linspace(zmin, zmax, z_step)
        rgrid = self.base.comoving_distance(zgrid) * self.base.h

        if not inverse:
            ret = fx.spline(rgrid.value, zgrid)
        else:
            ret = fx.spline(zgrid, rgrid.value)

        return ret

    def __init_galaxy(self):
        """Initializes: self.interp_ng_3d, self.dndz_galaxy."""

        nz_data = np.genfromtxt(self.survey_galaxy.dndz_data)

        if self.survey_galaxy.zbin_fmt == 'edge':
            zmin, zmax, n2d = nz_data[:,0:3].T
            dn2d = nz_data[:,3]

            assert zmin[0] == self.survey_galaxy.zmin
            assert zmax[-1] == self.survey_galaxy.zmax

        elif self.survey_galaxy.zbin_fmt == 'mid':
            zmid, n2d = nz_data[:,0:2].T
            dn2d = nz_data[:,2]

            zmin, zmax = fx.utils.bin_edges(zmid)
            zmin[0] = self.survey_galaxy.zmin
            zmax[-1] = self.survey_galaxy.zmax

        else:
            raise RuntimeError(f'cosmology.__init_galaxy: invalid format: {self.survey_galaxy.zbin_fmt}')

        assert np.all(zmax[:-1] == zmin[1:])

        zbin_delim = np.append(zmin, zmax[-1])
        assert (self.survey_galaxy.nspl < n2d.size)

        n_tot = self.survey_galaxy.n_total / (4*np.pi*self.survey_galaxy.f_sky)     # per sr.
        t = n_tot / np.sum(n2d)
        n2d *= t
        dn2d *= t
        dlogn2d = dn2d / n2d        # dn2d -> dlog(n2d).

        _path = self.pkl_dir + 'ng_3d.pkl'
        try:
            interp_log_ng_3d = fx.read_pickle(_path)
        except OSError as err:
            print(err)

            # The following fitting routine was originally written by Kendrick Smith.
            def chi2(log_n3d):
                s = fx.spline(self.config.zz_gi, log_n3d)

                _ret = 0.0
                for (_zmin, _zmax, n, _sigma) in zip(zmin, zmax, n2d, dlogn2d):

                    z_trap = np.linspace(_zmin, _zmax, self.survey_galaxy.ntrap)
                    dndz_trap = np.array([self.dvoz(_z) * np.exp(s(_z)) for _z in z_trap])
                    n2d_model = np.trapz(dndz_trap, z_trap)

                    _ret += (np.log(n2d_model) - np.log(n))**2 / _sigma**2

                return _ret

            iguess = np.log(0.1) - self.config.zz_gi * np.log(1000.)
            for (i,z) in enumerate(self.config.zz_gi):
                b = np.searchsorted(zbin_delim,z) - 1
                b = min(b, zmin.size)
                b = max(b, 0)

                chi0 = self.z_to_chi(zbin_delim[b])
                chi1 = self.z_to_chi(zbin_delim[b+1])

                _n2d = n2d[b]
                _n3d = 3. * _n2d / (chi1**3 - chi0**3)
                iguess[i] = np.log(_n3d)

            ret = scipy.optimize.minimize(chi2, iguess, method=self.survey_galaxy.solver)

            if not ret.success:
                raise RuntimeError('cosmology.__init_galaxy: model fitting failed!')

            log_n3d_vec = ret.x
            interp_log_ng_3d = fx.spline(self.config.zz_gi, log_n3d_vec)

            ng_2d_tot = fx.quad(lambda x: self.dvoz(x) * np.exp(interp_log_ng_3d(x)),
                                self.survey_galaxy.zmin, self.survey_galaxy.zmax)

            assert ng_2d_tot > 0
            log_n3d_vec += np.log(n_tot / ng_2d_tot)

            interp_log_ng_3d = fx.spline(self.config.zz_gi, log_n3d_vec)
            fx.write_pickle(_path, interp_log_ng_3d)

        def interp_ng_3d(x):
            return np.exp(interp_log_ng_3d(x))

        self.interp_ng_3d = interp_ng_3d
        self.dndz_galaxy = lambda redshift: self.dvoz(redshift) * self.interp_ng_3d(redshift)

        self.zbin_delim_galaxy = zbin_delim
        self.n2d_galaxy = n2d
        self.dn2d_galaxy = dn2d

    def __m_g(self, interp_nstep_z=None):
        """
        This special method computes the minimum mass m_g that a halo needs to have for hosting galaxies.

        Args:

            interp_nstep_z: (int) if not None, number of interpolation steps over redshift.

        Returns:

            function of redshift.
        """

        _path = self.pkl_dir + 'm_g.pkl'
        try:
            _ret = fx.read_pickle(_path)
        except OSError as err:
            print(err)
            interp_nstep_z = self.config.interp_nstep_m_g if interp_nstep_z is None else interp_nstep_z
            assert (interp_nstep_z == int(interp_nstep_z))

            zz = np.linspace(self.survey_galaxy.zmin, self.survey_galaxy.zmax, interp_nstep_z)

            m_g = []
            for z in zz:
                _ng = self.interp_ng_3d(z)
                _m_g = scipy.optimize.brentq(lambda Mg: _ng - (self.hmf.rho_m/Mg * self.hmf.f_coll(z,Mg)),
                                             self.config.m_min, self.config.m_max)
                m_g.append(_m_g)

            _ret = fx.spline(zz, np.log(m_g))
            fx.write_pickle(_path, _ret)

        def ret(x):
            return max(0.0, np.exp(_ret(x)))

        return ret

    def __eta(self, interp_nstep_z=None):
        """
        This special method computes the normalization constant 'eta' as a function of redshift.

        Args:

            interp_nstep_z: (int) if not None, number of interpolation steps over redshift.

        Returns:

            list, interp obj (z) for self.frb_par models.
        """

        interp_nstep_z = self.config.interp_nstep_eta if interp_nstep_z is None else interp_nstep_z

        assert (interp_nstep_z == int(interp_nstep_z))
        assert self.config.m_max <= self.hmf.m_max / 100.

        def _in(x, m_f):
            assert self.hmf.zmin <= x <= self.hmf.zmax, f'({self.hmf.zmin}, {x}, {self.hmf.zmax})'

            y = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(x, m)
                        * np.exp(m) / m_f,
                        np.log(m_f), np.log(self.config.m_max))

            y *= self.dvoz(x)
            assert np.isfinite(y) and (y >= 0.0)
            return y

        ret = []
        for par_index, par in enumerate(self.frb_par):
            zz = fx.lspace(self.interp_zmin, self.interp_zmax, interp_nstep_z, par[6])
            dndz_frb = self.dndz_frb[par_index]

            _eta = np.asarray([dndz_frb(z,cutoff=False) for z in zz])
            _in_zz = np.asarray([_in(z, par[7]) for z in zz])
            _eta /= _in_zz

            eta = fx.spline(zz, _eta)
            ret.append(eta)

        return ret

    def __psg(self):
        """
        This special method computes the probability for single galaxies to host FRB's.

        Returns:

            list, functions (z,m) for self.frb_par models.
        """

        def f(x, eta, m_f):
            def _f(z, m, _eta=eta, _x=x):
                if not (max(self.interp_zmin,self.survey_galaxy.zmin) <= z <= self.survey_galaxy.zmax):
                    return 0.0
                else:
                    if m >= self.m_g(z):
                        return _x * _eta(z) * self.m_g(z) / m_f
                    else:
                        return 0.0
            return _f

        ret = []
        for i, par in enumerate(self.frb_par):
            _ret = f(par[5], self.eta[i], par[7])
            ret.append(_ret)

        return ret

    def __ng_3d(self, z):
        """
        This special method computes the 3-d comoving number density of galaxies.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        ret = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(z, m)
                      * self.n_g(z, np.exp(m)),
                      np.log(self.m_g(z)), np.log(self.config.m_max))

        assert np.isfinite(ret) and (ret >= 0.0)
        return ret

    def __ngg_3d(self, z):
        """
        This special method computes the 3-d comoving number density of (galaxy, galaxy) pairs in the same halo.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        ret = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(z, m)
                      * (self.n_g(z, np.exp(m))**2 + self.n_g(z, np.exp(m))),
                      np.log(self.m_g(z)), np.log(self.config.m_max))

        assert np.isfinite(ret) and (ret >= 0.0)
        return ret

    def __nge_3d(self, z):
        """
        This special method computes the 3-d comoving number density of (galaxy, electron) pairs in the same halo.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        ret = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(z, m)
                      * np.exp(m) * self.n_g(z, np.exp(m)),
                      np.log(self.m_g(z)), np.log(self.config.m_max))

        assert np.isfinite(ret) and (ret >= 0.0)
        return ret

    def __bias_g(self, k, z, mc=10):
        """
        This special method computes the galaxy bias.

        Args:

            k: (float) spatial wavenumber.
            z: (float) redshift.
            mc: (int) coefficient for partitioning the integral over mass, e.g.
                m=10 -> _int(x_min, x_max) = _int(x_min, x_min*10) + _int(x_min*10, x_max).

        Returns:

            float.
        """

        assert isinstance(k, float) and (self.hmf.kmin <= k <= self.hmf.kmax)
        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)
        assert (mc == int(mc)) and (mc >= 2)

        def _int(m_min, m_max):
            assert m_min < m_max, "cosmology.__bias_g: _int expects m_min < m_max. 'mc' might be too large!"

            m_min = np.log(m_min)
            m_max = np.log(m_max)

            _ret = fx.quad(lambda m: self.hmf.interp_bias(z, m)
                           * self.hmf.interp_dn_dlog_m(z, m)
                           * self.n_g(z, np.exp(m))
                           * self.rho_nfw(k, z, np.exp(m), normalized=True),
                           m_min, m_max)

            return _ret

        ret = _int(self.m_g(z), self.m_g(z)*mc) + _int(self.m_g(z)*mc, self.config.m_max)
        ret /= self.interp_ng_3d(z)

        assert np.isfinite(ret), f'cosmology.__bias_g: ret(k={k},z={z}) = {ret}'
        return ret

    def __bias_e(self, k, z, mc=10):
        """
        This special method computes the electron bias.  The current version assumes that electrons
        follow the matter distribution.

        Args:

            k: (float) spatial wavenumber.
            z: (float) redshift.
            mc: (int) coefficient for partitioning the integral over mass, e.g.
                m=100 -> _int(x_min, x_max) = _int(x_min, x_min*100) + _int(x_min*100, x_max).

        Returns:

            float.
        """

        assert isinstance(k, float) and (self.hmf.kmin <= k <= self.hmf.kmax)
        assert isinstance(z, float) and (self.hmf.zmin <= z <= self.hmf.zmax)
        assert (mc == int(mc)) and (mc >= 2)

        def _int(m_min, m_max):
            assert m_min < m_max, "cosmology.__bias_e: _int expects m_min < m_max. 'mc' might be too large!"

            m_min = np.log(m_min)
            m_max = np.log(m_max)

            _ret = fx.quad(lambda m: self.hmf.interp_bias(z, m)
                           * self.hmf.interp_dn_dlog_m(z, m)
                           * np.exp(m)
                           * (self.rho_nfw(k, z, np.exp(m), normalized=True)-1.0),
                           m_min, m_max)

            return _ret

        ret = _int(self.hmf.m_min, self.hmf.m_min*mc) + _int(self.hmf.m_min*mc, self.config.m_max)
        ret = 1.0 + ret / self.hmf.rho_m

        assert np.isfinite(ret), f'cosmology.__bias_e: ret(k={k},z={z}) = {ret}'
        return ret

    def __p1_ge(self, k, z):
        """
        This special method computes the 1-halo term of the 3-d spatial galaxy-electron cross power spectrum.

        Args:

            k: (float) spatial wavenumber.
            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(k, float) and (self.hmf.kmin <= k <= self.hmf.kmax)
        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        _int = fx.quad(lambda m: self.hmf.interp_dn_dlog_m(z, m)
                       * np.exp(m)**2
                       / self.m_g(z)
                       * self.rho_nfw(k, z, np.exp(m), normalized=True)**2,
                       np.log(self.m_g(z)), np.log(self.config.m_max))

        ret = _int / self.hmf.rho_m / self.interp_ng_3d(z)

        assert np.isfinite(ret), f'cosmology.__p1_ge: ret(k={k},z={z}) = {ret}'
        return ret

    def __p2_ge(self, k, z):
        """
        This special method computes the 2-halo term of the 3-d spatial galaxy-electron cross power spectrum.

        Args:

            k: (float) spatial wavenumber.
            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(k, float) and (self.hmf.kmin <= k <= self.hmf.kmax)
        assert isinstance(z, float) and (self.survey_galaxy.zmin <= z <= self.survey_galaxy.zmax)

        ret = self.interp_bias_g(k,z) * self.interp_bias_e(k,z) * self.hmf.p_k(k=k, z=z)

        assert np.isfinite(ret), f'cosmology.__p_ge (2): ret(k={k},z={z}) = {ret}'

        return ret

    def __test_all(self):
        self.__test_p_k()
        self.__test_dvoz()
        self.__test_m_g()
        self.__test_eta()
        self.__test_psg()
        self.__test_interp_w_g()
        self.__test_bin()
        self.__test_delta_z()
        self.__test_rho_nfw()
        self.__test_l_to_k()
        self.__test_n_g()
        self.__test_gamma_e()
        self.__test_w_x()
        self.__test_n_x()
        self.__test_dm_h()
        self.__test_frb_dndz()
        self.__test__frb_dndz_interp()
        self.__test__vol_s()
        self.__test__dvoz()
        self.__test__chi_to_z()
        self.__test__cosmology__init_galaxy()
        self.__test__cosmology__ng_3d()
        self.__test__cosmology__ngg_3d()
        self.__test__cosmology__nge_3d()
        self.__test__cosmology__bias_g()
        self.__test__cosmology__bias_e()
        self.__test__cosmology__p1_ge()
        self.__test__cosmology__p2_ge()

    def __test_p_k(self):
        pass

    def __test_dvoz(self, niter=100, rtol=1.0e-4, atol=0.0):
        zz = rand.uniform(0.0, self.config.interp_zmax, niter)
        zz = np.append(zz, [0.0, self.config.interp_zmax])

        for z in zz:
            a, e = self.dvoz(z), self._dvoz(z)
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test_m_g(self, niter=1000):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append(zvec, [self.survey_galaxy.zmin, self.survey_galaxy.zmax])

        for z in zvec:
            a = self.m_g(z)
            assert np.isfinite(a) and (a >= 0.0)

        plt.semilogy(self.config.zz_g, [self.m_g(z) for z in self.config.zz_g])
        plt.xlabel(r'$z$')
        plt.ylabel(r'$M_g(z)$')
        plt.savefig('m_g.pdf')
        plt.clf()

    def __test_eta(self, niter=1000):
        zvec = rand.uniform(self.interp_zmin, self.interp_zmax, niter)
        zvec = np.append(zvec, [self.interp_zmin, self.interp_zmax])
        zvec_plt = np.linspace(self.interp_zmin, 3.0, 10000)

        for i in range(len(self.eta)):
            eta = self.eta[i]
            for z in zvec:
                _eta = eta(z)
                assert np.isfinite(_eta) and (_eta >= 0.0)

            plt.semilogy(zvec_plt, [eta(z) for z in zvec_plt], label=r'${\rm FRB~par~index}~%d$'%i)

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\eta(z)$')
        plt.ylim(1.0e-15, 1.0e-5)
        plt.savefig('eta.pdf')
        plt.clf()

    def __test_psg(self, niter=100):
        _zmin = max(self.interp_zmin, self.config.survey_galaxy.zmin)
        _zmax = min(self.interp_zmax, self.config.survey_galaxy.zmax)
        for z in rand.uniform(_zmin, _zmax, niter):
            for m in rand.uniform(self.hmf.m_min, self.hmf.m_max, niter):
                for i in range(len(self.frb_par)):
                    _psg = self.psg[i]
                    psg = _psg(z, m)
                    assert np.isfinite(psg) and (0.0 <= psg <= 1.0)

    def __test_interp_w_g(self, niter=20, rtol=5.0e-3, atol=1.0e-10):
        w_x = self.w_x(n_x=self.n_g, m_min=self.m_g)

        _zmin = max(self.interp_zmin, self.config.survey_galaxy.zmin)
        zvec = rand.uniform(_zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([_zmin, self.survey_galaxy.zmax], zvec)

        lvec = rand.choice(self.ll, size=niter, replace=False)
        lvec = np.append(lvec, [self.ll[0], self.ll[-1]])

        for z in zvec:
            n3d = self.interp_ng_3d(z)

            for l in lvec:
                e = w_x(z, l)
                a = self.interp_w_g(z, l)

                print(f'z={z}, l={l}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

                k = self.l_to_k(l, z)
                b = self.interp_bias_g(k, z) * n3d

                print(f'z={z}, l={l}, a={a}, b={b}')
                assert np.isclose(a, b, rtol, atol)

    def __test_bin(self):
        for nz in range(1, 5):
            for nd in range(1, 5):
                zz, dd, dd_z = self.bin(nz, nd)

                zz = np.asarray(zz)
                dd = np.asarray(dd)
                dd_z = np.asarray(dd_z)

                assert np.isfinite(zz).all()
                assert np.logical_and((self.config.zmin <= zz), (zz <= self.config.zmax)).all()

                assert np.isfinite(dd).all()
                assert np.logical_and((self.config.dmin <= dd),
                                      (dd <= self.config.dmax)).all()

                assert np.isfinite(dd_z).all()
                assert np.logical_and((0.0 <= zz), (zz <= self.interp_zmax)).all()

    def __test_delta_z(self):
        assert self.delta_z(0.0, 0.0) == 0.0

        for z in (0.0, 1.0, 1.0e3):
            ret = self.delta_z(z, rand.uniform(0, 1e4, 1000))
            assert np.all(ret >= 0.0)

    def __test_rho_nfw(self, niter=10, z=1.0):
        for i in range(niter):
            m = rand.uniform(0, 1e17)
            rho = self.rho_nfw(k=0.0, z=z, m=m)
            assert rho == m

        mvec = np.logspace(9, 20, 11)
        lvec = np.logspace(0, 7, 100)
        for m in mvec:
            k = np.zeros_like(lvec)
            rho = np.zeros_like(k)

            for i, j in enumerate(lvec):
                k[i] = self.l_to_k(j, z)
                rho[i] = self.rho_nfw(k[i], z, m, normalized=True)

            plt.xscale('log', nonpositive='clip')
            plt.plot(lvec, rho, label=f'$\log(m)={round(np.log10(m))}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$l$')
        plt.ylabel(r'$\rho_{NFW}(l,z,m)$')
        plt.savefig('test_rho_nfw.pdf')
        plt.clf()

    def __test_l_to_k(self, niter=1000, rtol=1.0e-4, atol=0.0):
        for i in range(niter):
            z = rand.uniform(self.interp_zmin, self.interp_zmax)
            l = 10**rand.uniform(high=6)

            k = self.l_to_k(l, z)
            assert isinstance(k, float) and (k >= 0.0)

            chi = self.base.comoving_distance(z) * self.base.h
            ke = l / chi.value
            assert np.isclose(k, ke, rtol, atol)

    def __test_n_g(self):
        pass

    def __test_gamma_e(self, niter=1000):
        for i in range(niter):
            z = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax)
            ret = self.gamma_e(z)
            assert np.isfinite(ret) and (0.0 < ret < 1.0)

    def __test_w_x(self):
        pass

    def __test_n_x(self):
        def f(z, m, xmin, xmax):
            return z**m + (xmin*xmax)

        ret = self.n_x(f, 1.0, 2.0)
        assert ret(2, 3) == float(10)

        ret = self.n_x(lambda z, m, xmin, xmax: z**m + (xmin*xmax), 4, 1)
        assert ret(10, 2) == int(104)

    def __test_dm_h(self, niter=1000):
        assert np.all(self.dm_h(niter, 0.0, 0.0) == 1.0)
        assert np.all(self.dm_h(niter, rand.ranf()*1.0e6, rand.ranf()*1.0e6) >= 0.0)

    def __test_frb_dndz(self, niter=1000):
        for z in rand.uniform(0.0, 1.0e12, size=niter):
            assert self.frb_dndz(z, 0.0, 0.0) == 1.0
            assert self.frb_dndz(z, 0.0, 1.0) == np.exp(-z)

            for p in rand.uniform(10.0, size=niter):
                assert self.frb_dndz(z, p, 0.0) == z**p
                assert self.frb_dndz(z, p, np.inf) == 0.0

    def __test__frb_dndz_interp(self, niter=10, rtol=5.0e-3, atol=0.0):
        n_e = rand.ranf() * 1.0e3
        for p in rand.uniform(10.0, size=niter):
            for alpha in rand.uniform(100.0, size=niter):
                spl = self._frb_dndz_interp(n_e, p, alpha, 0.0, self.interp_zmax, bool(rand.randint(2)))

                n_a = fx.quad(lambda z: spl(z), self.config.zmin, (self.interp_zmax-self.config.zpad_interp_zmax))
                assert np.isclose(n_a, n_e, rtol, atol), f'({n_a}, {n_e})'

    def __test__vol_s(self, niter=10, rtol=1.0e-4, atol=0.0):
        zvec = np.linspace(0.0, self.config.interp_zmax, niter)
        omega = self.config.f_sky * 4 * np.pi

        for (zmin,zmax) in zip(zvec[0::2][:-1], zvec[0::2][1:]):
            e = fx.quad(lambda x: self.base.differential_comoving_volume(x).value, zmin, zmax)
            e *= omega * self.base.h**3
            a = self._vol_s(zmin, zmax)
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test__dvoz(self, niter=100, rtol=1.0e-6, atol=0.0):
        for _ in range(niter):
            z = rand.uniform(0.0, self.config.interp_zmax)

            e = self.base.differential_comoving_volume(z) * u.sr
            e = e.value * self.base.h**3
            a = self._dvoz(z)
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test__chi_to_z(self, niter=1000, rtol=1.0e-4, atol=0.0):
        zvec = rand.uniform(0.0, self.config.interp_zmax, size=niter)
        zvec = np.append(zvec, [0.0, self.config.interp_zmax])

        for z in zvec:
            e = self.base.comoving_distance(z) * self.base.h
            a = self.z_to_chi(z)
            assert np.isclose(a, e.value, rtol, atol), f'({a}, {e})'
            assert np.isclose(self.chi_to_z(a), z, rtol, atol)

    def __test__cosmology__init_galaxy(self, rtol=1.0e-4, atol=0.0):
        a = fx.quad(lambda z: self.dndz_galaxy(z), self.survey_galaxy.zmin, self.survey_galaxy.zmax)
        e = self.survey_galaxy.n_total / (4 * np.pi * self.survey_galaxy.f_sky)
        assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

        plt.semilogy(self.config.zz_g, [self.dndz_galaxy(z) for z in self.config.zz_g])
        plt.ylim(10, 1.0e8)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$dn^{2d}_g/dz$')
        plt.savefig('test__cosmology__init_galaxy.pdf')
        plt.clf()

    def __test__cosmology__ng_3d(self, niter=100, rtol=1.0e-2, atol=eps):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        for z in zvec:
            e = self.interp_ng_3d(z)
            assert np.isfinite(e) and (e >= 0.0)
            a = self.__ng_3d(z)
            assert np.isfinite(a) and (a >= 0.0)
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

            a2d = self.dvoz(z) * a
            e2d = self.dndz_galaxy(z)
            assert np.isclose(a2d, e2d, rtol, atol)

    def __test__cosmology__ngg_3d(self, niter=100, rtol=5.0e-3, atol=eps):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        for z in zvec:
            a = self.interp_ngg_3d(z)
            assert np.isfinite(a) and (a >= 0.0)
            e = self.__ngg_3d(z)
            assert np.isfinite(e) and (e >= 0.0)

            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test__cosmology__nge_3d(self, niter=100, rtol=5.0e-3, atol=eps):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        for z in zvec:
            a = self.interp_nge_3d(z)
            assert np.isfinite(a) and (a >= 0.0)
            e = self.__nge_3d(z)
            assert np.isfinite(e) and (e >= 0.0)

            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test__cosmology__bias_g(self, niter=10, rtol=1.0e-3, atol=1.0e-7):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        kvec = rand.choice(self.kk, size=niter, replace=False)
        kvec = np.append([self.kk[0], self.kk[-1]], kvec)

        for z in zvec:
            for k in kvec:
                a = self.interp_bias_g(k, z)
                assert np.isfinite(a) and (a >= 0.0)
                e = self.__bias_g(k, z)
                assert np.isfinite(e) and (e >= 0.0)

                print(f'z={z}, k={k}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

        for z in (0.1, 0.4, 0.7, 1.0):
            _bias_g = [self.interp_bias_g(self.l_to_k(l,z), z).flatten() for l in self.ll]
            plt.plot(self.ll, _bias_g, label=f'$z = {z}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$l$')
        plt.ylabel(r'$b_g$')
        plt.savefig('b_g.pdf')
        plt.clf()

    def __test__cosmology__bias_e(self, niter=10, rtol=1.0e-3, atol=1.0e-7):
        zvec = rand.uniform(self.config.zz_e[0], self.config.zz_e[-1], niter)
        zvec = np.append([self.config.zz_e[0], self.config.zz_e[-1]], zvec)

        kvec = rand.choice(self.kk, size=niter, replace=False)
        kvec = np.append([self.kk[0], self.kk[-1]], kvec)

        for z in zvec:
            for k in kvec:
                a = self.interp_bias_e(k, z)
                assert np.isfinite(a) and (a >= 0.0)
                try:
                    e = self.__bias_e(k, z)
                    assert np.isfinite(e) and (e >= 0.0)
                except RuntimeError:
                    e = 1.0

                print(f'z={z}, k={k}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

        for z in (0.1, 0.4, 0.7, 1.0):
            _bias_e = [self.interp_bias_e(self.l_to_k(l,z), z).flatten() for l in self.ll]
            plt.plot(self.ll, _bias_e, label=f'$z = {z}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$l$')
        plt.ylabel(r'$b_e$')
        plt.savefig('b_e.pdf')
        plt.clf()

    def __test__cosmology__p1_ge(self, niter=20, rtol=1.0e-2, atol=1.0e-10):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        kvec = rand.choice(self.kk, size=niter, replace=False)
        kvec = np.append([self.kk[0], self.kk[-1]], kvec)

        for z in zvec:
            for k in kvec:
                a = self.interp_p1_ge(k, z)
                assert np.isfinite(a) and (a >= 0.0)
                e = self.__p1_ge(k, z)
                assert np.isfinite(e) and (e >= 0.0)

                print(f'z={z}, k={k}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

        for z in (0.0, 0.5, 1.0):
            _p1ge = [self.interp_p1_ge(k,z).flatten() for k in self.kk]
            plt.loglog(self.kk, _p1ge, label=f'$z = {z}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P_{ge}^{1h}$')
        plt.savefig('pge1.pdf')
        plt.clf()

    def __test__cosmology__p2_ge(self, niter=100, rtol=5.0e-3, atol=1.0e-10):
        zvec = rand.uniform(self.survey_galaxy.zmin, self.survey_galaxy.zmax, niter)
        zvec = np.append([self.survey_galaxy.zmin, self.survey_galaxy.zmax], zvec)

        kvec = rand.choice(self.kk, size=niter, replace=False)
        kvec = np.append([self.kk[0], self.kk[-1]], kvec)

        for z in zvec:
            for k in kvec:
                a = self.interp_p2_ge(k, z)
                assert np.isfinite(a) and (a >= 0.0)
                e = self.__p2_ge(k, z)
                assert np.isfinite(e) and (e >= 0.0)

                print(f'z={z}, k={k}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

        for z in (0.0, 0.5, 1.0):
            _p2ge = [self.interp_p2_ge(k,z).flatten() for k in self.kk]
            plt.loglog(self.kk, _p2ge, label=f'$z = {z}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P_{ge}^{2h}$')
        plt.savefig('pge2.pdf')
        plt.clf()
