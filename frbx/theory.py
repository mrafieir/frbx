import numpy as np
import numpy.random as rand
import astropy.units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import frbx as fx
from frbx.configs import tiny, eps


class clfg_theory:
    """
    (FRB, galaxy) cross power spectrum C_l^{fg} and observables for a single set of FRB parameters.

    Implicit units:

        Mass:                   M_sun/h.
        Distance:               Mpc/h (comoving).
        Spatial wavenumber:     h/Mpc (comoving).
        Dispersion Measure:     pc/cm^3 (comoving).

    Members:

        self.ftc: (obj) instance of fx.cosmology, which is recommended to be pickled in advance.
        self.pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
        self.config: (obj) points to self.ftc.config, which is an instance of fx.configs.
        self.frb_par_index: (int) specifies a set of FRB model parameters.
        self.fpar: (list) the set of FRB model parameters.
        self.nz: (int) if not None, total number of redshift bins.
        self.nd: (int) if not None, total number of extragalactic DM bins.
        self.pdf_dh: (lambda) pdf of the log-normal distribution for host DMs.
        self.cdf_dh: (lambda) cdf of the log-normal distribution for host DMs.
        self.m_f: (float) min halo mass for hosting FRBs.
        self.eta: (function) quantity eta (interpolated over z) for normalizing FRB counts.
        self.dndz_f: (function) interpolated differential number of FRBs per unit steradian (z).
        self.interp_nf_3d: (lambda) interpolated 3-d comoving number density of FRBs (z).
        self.interp_nfg_3d: (lambda) interpolated 3-d comoving number density of (f,g) pairs in the same halo (z).
        self.interp_bias_f: (interp obj) FRB bias (k,z).
        self.interp_w_f: (interp obj) interpolated radial weight function (two-halo term) for FRBs (z,l).
        self.zz: (list) [min, max] bounds of redshift bins. (*)
        self.dd: (list) [min, max] bounds of extragalactic DM bins. (*)
        self.dd_z: (list) [min, max] bounds of extragalactic DM bins, converted to redshift values. (*)
        self.zz_y: (list) 1-d arrays of sliced redshift for interpolating objects in redshift bins. (*)
        self.zz_x: (list) 1-d arrays of sliced redshift for interpolating obj in DM bins. (*)
        self.zz_x_fine: (list) 1-d arrays of finely sliced redshift for interpolating high-precision obj in DM bins. (*)
        self.nbar_f: (list) angular number density of FRBs in DM bins: [all incl. single-galaxy, single-galaxy]. (*)
        self.nbar_g: (list) angular number density of galaxies in redshift bins. (*)
        self.omega_f: (list) first-order terms (z) in the expression for the perturbed number density of FRBs. (*)
        self.w_ls: (list) radial weight function (z) for line-of-sight effects in extragalactic DM bins. (*)
        self.nbar_x: (method) 2-d angular number density.
        self.c1_xy: (method) angular noise cross power, i.e. one-halo term plus Poisson terms (if any).
        self.c2_xy: (method) two-halo term of the angular cross power.
        self.c_ls_fg: (method) line-of-sight term of the angular cross power.
        self.prob_d_z: (method) computes the probability distribution for FRBs to be in a DM bin.
        self.gamma_f: (method) quantity gamma for FRBs.
        self.n_f: (method) expected number of FRBs in a halo.
        self.convert_w_ls: (method) converts w_ls (old def) -> w_ls (new def).
        clfg_theory.beam: (static method) returns a Fourier beam model (l).
        clfg_theory.csg_xy: (static method) single-galaxy term of the angular cross power.
        self._omega_f: (helper method) returns self.omega_f.
        self._w_ls: (helper method) radial weight function for perturbed distribution of FRBs along the line of sight.
        self._clfg_theory__nf_3d: (special method) returns self.interp_nf_3d.
        self._clfg_theory__nfg_3d: (special method) returns self.interp_nfg_3d.
        self._clfg_theory__bias_f: (special method) returns self.interp_bias_f.
        self._clfg_theory__zz_y: (special method) returns self.zz_y.
        self._clfg_theory__zz_x: (special method) returns (self.zz_x, self.zz_x_fine).
        self._clfg_theory__bin_xy: (special method) initializes members which are marked by (*).
        self._clfg_theory__test_*.
    """

    def __init__(self, ftc, pkl_dir, frb_par_index=0, nz=None, nd=None):
        """
        Constructor arguments:

            ftc: (obj) instance of fx.cosmology.
            pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
            frb_par_index: (int) index specifying a single set of frb parameters.
            nz: (int) if not None, total number of redshift bins.
            nd: (int) if not None, total number of extragalactic DM bins.
        """

        assert isinstance(ftc, fx.cosmology)
        assert isinstance(pkl_dir, str) and pkl_dir.endswith('/')

        self.ftc = ftc
        self.pkl_dir = pkl_dir

        assert hasattr(self.ftc, 'config')
        self.config = self.ftc.config

        assert isinstance(frb_par_index, int) and (0 <= frb_par_index <= (len(self.ftc.frb_par)-1))
        self.frb_par_index = frb_par_index
        self.fpar = self.ftc.frb_par[self.frb_par_index]

        if (nz is not None) or (nd is not None):
            assert isinstance(nz, int) and (nz >= 1)
            assert isinstance(nd, int) and (nd >= 1)

        self.nz = nz
        self.nd = nd

        self.pdf_dh = fx.utils.lognorm(mu=self.fpar[3], sigma=self.fpar[4])
        self.cdf_dh = fx.utils.lognorm(mu=self.fpar[3], sigma=self.fpar[4], cdf=True)
        self.m_f = self.fpar[7]
        self.eta = self.ftc.eta[self.frb_par_index]
        self.dndz_f = self.ftc.dndz_frb[self.frb_par_index]

        ###################################################################################################
        #
        # (f,g) objects.

        _path = self.pkl_dir + f'fg_obj_{self.frb_par_index}.pkl'
        try:
            fg_obj = fx.read_pickle(_path)
        except OSError as err:
            print(err)

            # Min/Max redshift bounds for (f) and (f,g) objects in this constructor.
            fgn_zmin = max(self.ftc.interp_zmin,  self.config.survey_galaxy.zmin)
            fn_zmax = self.config.fn_zmax + self.config.zpad_eps
            fgn_zmax = min(fn_zmax, self.config.survey_galaxy.zmax)

            ### self.interp_nf_3d
            zz_f = fx.lspace(self.ftc.interp_zmin, fn_zmax, self.config.interp_nstep_zz_f, True)
            nf_3d = np.asarray([self.__nf_3d(i) for i in zz_f])
            _interp_nf_3d = fx.spline(zz_f, nf_3d)
            self.interp_nf_3d = _interp_nf_3d
            print('self.interp_nf_3d done.')

            ### self.interp_nfg_3d
            zz_fg = fx.lspace(fgn_zmin, fgn_zmax, self.config.interp_nstep_zz_f, True)

            nfg_3d = np.asarray([self.__nfg_3d(i) for i in zz_fg])
            _interp_nfg_3d = fx.spline(zz_fg, nfg_3d)
            print('self.interp_nfg_3d done.')

            ### self.interp_bias_f
            zz_f = fx.lspace(self.ftc.interp_zmin, fn_zmax, self.config.interp_nstep_zz_f, self.fpar[6])
            _bias_f = np.zeros((self.config.interp_nstep_kk, self.config.interp_nstep_zz_f))

            for i, k in enumerate(self.ftc.kk):
                for j, z in enumerate(zz_f):
                    print(f'(i={i},j={j})')
                    try:
                        ret = self.__bias_f(k, z)
                    except RuntimeError as err:
                        print(err)
                        print(f'self.interp_bias_f ({k},{z}): default mc failed. Trying mc=2.')
                        try:
                            ret = self.__bias_f(k, z, 2)
                        except RuntimeError as err:
                            print(err)
                            print(f'self.interp_bias_f ({k},{z}) -> 1.0')
                            ret = 1.0

                    _bias_f[i,j] = ret

            self.interp_bias_f = fx.spline(self.ftc.kk, zz_f, _bias_f)
            print('self.interp_bias_f done.')

            ### self.interp_w_f
            zz_f = fx.lspace(self.ftc.interp_zmin, fn_zmax, self.config.interp_nstep_zz_f, True)
            _w_f = np.zeros((self.config.interp_nstep_zz_f, self.config.interp_nstep_ll))
            _w_x = self.ftc.w_x(n_x=self.n_f, m_min=self.m_f)

            for i, z in enumerate(zz_f):
                for j, l in enumerate(self.ftc.ll):
                    print(f'(i={i},j={j})')
                    _w_f[i,j] = _w_x(z,l)

            self.interp_w_f = fx.spline(zz_f, self.ftc.ll, _w_f)
            print('self.interp_w_f done.')

            fg_obj = [_interp_nf_3d,
                      _interp_nfg_3d,
                      self.interp_bias_f,
                      self.interp_w_f]

            fx.write_pickle(_path, fg_obj)

        self.interp_nf_3d = fg_obj[0]
        self.interp_nfg_3d = fg_obj[1]
        self.interp_bias_f = fg_obj[2]
        self.interp_w_f = fg_obj[3]

        ###################################################################################################
        #
        # bin_xy objects.

        if (self.nz is not None) and (self.nd is not None):
            _path = self.pkl_dir + f'bin_xy_objs_{self.frb_par_index}_{self.nz}z_{self.nd}_d.pkl'
            try:
                bin_xy_obj = fx.read_pickle(_path)
                self.zz = bin_xy_obj[0]
                self.dd = bin_xy_obj[1]
                self.dd_z = bin_xy_obj[2]
                self.zz_y = bin_xy_obj[3]
                self.zz_x = bin_xy_obj[4]
                self.zz_x_fine = bin_xy_obj[5]
                self.nbar_f = bin_xy_obj[6]
                self.nbar_g = bin_xy_obj[7]
                self.omega_f = bin_xy_obj[8]
                self.w_ls = bin_xy_obj[9]
            except OSError as err:
                print(err)
                self.__bin_xy()
                bin_xy_obj = [self.zz, self.dd, self.dd_z, self.zz_y, self.zz_x, self.zz_x_fine,
                              self.nbar_f, self.nbar_g, self.omega_f, self.w_ls]
                fx.write_pickle(_path, bin_xy_obj)

    def nbar_x(self, n_x, m_min, zmin, zmax, epsrel=1.0e-4):
        """
        This method computes the 2-d angular number density of X, which corresponds to e.g.
        FRBs in a DM bin or galaxies in a redshift shell.

        Args:

            n_x: (callable) given (z,m), returns the expected number of X's.
            m_min: (lambda, interp obj, or float) min halo mass as a function of redshift.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            epsrel: (float) relative error while integrating with epsabs=0.

        Returns:

            float, 2-d angular number density.
        """

        assert callable(n_x)
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert self.ftc.interp_zmin < zmin < zmax < self.ftc.interp_zmax

        def lambda_m_max(z):
            assert z >= 0.0
            return np.log(self.config.m_max)

        lambda_m_min = fx.utils.mz(m_min)

        def _int(m, z):
            return self.ftc.dvoz(z) * self.ftc.hmf.interp_dn_dlog_m(z, m) * n_x(z, np.exp(m))

        ret = fx.dblquad(_int, zmin, zmax, lambda_m_min, lambda_m_max, epsrel=epsrel)

        assert np.isfinite(ret) and (ret >= 0.0)

        return ret

    def c1_xy(self, ell, nbar_x, nbar_y, n_x, n_y, m_min, zmin, zmax, auto=False, nxy=None, epsrel=1.0e-4):
        """
        This method computes the angular noise cross power at a given l.  Depending on the input args, it may
        (not) include the shot-noise and single-galaxy terms.

        Args:

            ell: (float) angular wavenumber.
            nbar_x: (float) angular number density (per unit steradian) of X's.
            nbar_y: (float) angular number density (per unit steradian) of Y's.
            n_x: (callable) given (z,m), returns the expected number of X's.
            n_y: (callable) given (z,m), returns the expected number of Y's.
            m_min: (lambda, interp obj, or float) min halo mass as a function of redshift.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            auto: (bool) whether this is an auto-power, i.e., X = Y.
            nxy: (float) if not None, the angular number density (per unit steradian) of X's that are
                 spatially overlapping with Y's. This can e.g. be the single-galaxy term.
            epsrel: (float) relative error while integrating with epsabs=0.

        Returns:

            float, angular noise cross power at the given l.
        """

        assert isinstance(ell, float) and np.isfinite(ell) and (ell >= 0.0)
        assert isinstance(nbar_x, float) and (nbar_x >= 0.0)
        assert isinstance(nbar_y, float) and (nbar_y >= 0.0)
        assert callable(n_x) and callable(n_y)
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert self.ftc.interp_zmin < zmin < zmax < self.ftc.interp_zmax
        assert isinstance(auto, bool)
        assert (nxy is None) or isinstance(nxy, float)

        if auto and (nxy is not None):
            raise AssertionError('fx.clfg_theory.c1_xy: nxy term assumes X != Y.')

        _nbar_xy = nbar_x * nbar_y

        if _nbar_xy == 0.0:
            return 0.0
        else:
            def lambda_m_max(z):
                assert z >= 0.0
                return np.log(self.config.m_max)

            lambda_m_min = fx.utils.mz(m_min)

            def _int(m, z):
                rho_nfw = self.ftc.rho_nfw(self.ftc.l_to_k(ell,z), z, np.exp(m), normalized=True)

                return (self.ftc.dvoz(z) * self.ftc.hmf.interp_dn_dlog_m(z,m)
                        * n_x(z, np.exp(m)) * n_y(z, np.exp(m)) * rho_nfw**2)

            ret = fx.dblquad(_int, zmin, zmax, lambda_m_min, lambda_m_max, epsrel=epsrel)

            assert np.isfinite(ret)

            ret /= _nbar_xy

            if auto:
                ret += (1 / nbar_x)
            elif nxy is not None:
                ret += clfg_theory.csg_xy(nbar_x, nbar_y, nxy)
            else:
                pass

            return ret

    def c2_xy(self, ell, nbar_x, nbar_y, w_x, w_y, zmin, zmax, epsrel=1.0e-4):
        """
        This method computes the two-halo term of the angular cross power at a given l.

        Args:

            ell: (float) angular wavenumber.
            nbar_x: (float) angular number density (per unit steradian) of X's.
            nbar_y: (float) angular number density (per unit steradian) of Y's.
            w_x: (callable) given (z,l), evaluates a radial weight function for X.
            w_y: (callable) given (z,l), evaluates a radial weight function for Y.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            epsrel: (float) relative error while integrating with epsabs=0.

        Returns:

            float, angular cross power at the given l.
        """

        assert isinstance(ell, float) and (self.ftc.ll[0] < ell < self.ftc.ll[-1])
        assert isinstance(nbar_x, float) and (nbar_x >= 0.0)
        assert isinstance(nbar_y, float) and (nbar_y >= 0.0)
        assert callable(w_x) and callable(w_y)
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert self.ftc.interp_zmin < zmin < zmax < self.ftc.interp_zmax

        _nbar_xy = nbar_x * nbar_y

        if _nbar_xy == 0.0:
            return 0.0
        else:
            def _int(z):
                return (self.ftc.dvoz(z)
                        * w_x(z, ell) * w_y(z, ell)
                        * self.ftc.p_k(self.ftc.l_to_k(ell,z), z))

            ret = fx.quad(_int, zmin, zmax, epsrel=epsrel, limit=10240)
            ret /= _nbar_xy

            return ret

    def c_ls_fg(self, ell, nbar_g, w_ls, zmin, zmax, mode=1, epsrel=1.0e-4):
        """
        This method computes the line-of-sight terms of (f,g) angular cross power.
        It assumes the old definition of w_ls.

        Args:

            ell: (float) angular wavenumber.
            nbar_g: (float) angular number density (per unit steradian) of galaxies.
            w_ls: (callable) given (z), evaluates a radial weight function for FRBs.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            mode: (int) either 1 (1h term) or 2 (2h term).
            epsrel: (float) relative error while integrating with epsabs=0.

        Returns:

            float, angular cross power at the given l.
        """

        assert isinstance(ell, float) and (self.ftc.ll[0] < ell < self.ftc.ll[-1])
        assert isinstance(nbar_g, float) and (nbar_g >= 0.0)
        assert callable(w_ls)
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert self.ftc.interp_zmin < zmin < zmax < self.ftc.interp_zmax
        assert mode in (1, 2)

        if nbar_g == 0.0:
            return 0.0
        else:
            _p_ge = self.ftc.interp_p1_ge if (mode == 1) else self.ftc.interp_p2_ge

            def _int(z):
                return (w_ls(z)
                        * self.ftc.dndz_galaxy(z)
                        * _p_ge(self.ftc.l_to_k(ell,z), z)
                        / self.ftc.z_to_chi(z)**2.)

            ret = fx.quad(_int, zmin, zmax, epsrel=epsrel, limit=10240)
            ret /= nbar_g

            return ret

    def prob_d_z(self, z, dmin, dmax):
        """
        This method computes the probability distribution, as a function of redshift, for FRBs to be in a DM bin.

        Args:

            z: (float) redshift.
            dmin: (float) minimum extragalactic DM.
            dmax: (float) maximum extragalactic DM.

        Returns:

            float, specifying the probability.
        """

        assert isinstance(z, float)
        assert 0.0 <= z < self.ftc.interp_zmax
        assert isinstance(dmin, float) and isinstance(dmax, float)
        assert 0.0 <= dmin < dmax

        di = max(float(self.ftc.dm_igm_interp(z)), 0.0)
        ret = self.cdf_dh(dmax - di) - self.cdf_dh(dmin - di)

        assert 0.0 <= ret <= 1.0
        return np.float64(ret)

    def gamma_f(self, z):
        """
        This method returns the quantity gamma for the FRB field.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.ftc.interp_zmin <= z <= self.ftc.survey_galaxy.zmax)

        n_x = self.ftc.interp_ng_3d(z) / self.interp_nf_3d(z)
        assert np.isfinite(n_x)

        n_xy = self.interp_nfg_3d(z) / self.ftc.interp_ngg_3d(z)
        assert np.isfinite(n_xy)

        ret = n_x * n_xy
        return ret

    def n_f(self, z, m, dmin=None, dmax=None):
        """
        This method returns the expected number of FRBs in a halo of mass m,
        located at redshift z in the DM bin (dmin, dmax).

        Args:

            z: (float) halo redshift.
            m: (float) halo mass.
            dmin: (float) if not None, minimum DM value in the bin.
            dmax: (float) if not None, maximum DM value in the bin.

        Returns:

            float, corresponding to the expected number of FRBs in the specified halo bin.
        """

        assert isinstance(z, float) and (self.ftc.interp_zmin <= z <= self.ftc.interp_zmax)
        assert isinstance(m, float) and (m >= 1.0e4), m

        if (dmin is not None) or (dmax is not None):
            assert isinstance(dmin, float) and isinstance(dmax, float)
            assert 0.0 <= dmin < dmax
            dv = True
        else:
            dv = False

        ret = max(0.0, self.eta(z)) * m / self.m_f

        if dv:
            ret *= self.prob_d_z(z, dmin, dmax)

        assert ret >= 0.0
        return ret

    def convert_w_ls(self, w):
        """
        Converts a callable line-of-sight radial weight function 'w' to a function
        with the new definition: w_ls^{new}(z) = w_ls^{old}(z) / H(z).
        """

        assert callable(w)

        def w_new(z):
            return w(z) / (self.ftc.base.H(z) / const.c).to(1/u.Mpc).value * self.ftc.base.h

        return w_new

    @staticmethod
    def beam(theta):
        """Returns the Fourier beam (l) of an instrument with a real FWHM resolution of 'theta' (astropy Quantity)."""

        theta = theta.to(u.rad).value

        if theta > 0.0:
            s = np.sqrt(8 * np.log(2)) / theta
        elif theta == 0.0:
            s = np.inf
        else:
            raise RuntimeError('fx.clfg_theory.beam: invalid beam size!')

        return fx.utils.gaussian(mu=0.0, sigma=s, a=1.0)

    @staticmethod
    def csg_xy(nbar_x, nbar_y, nxy):
        """
        This static method computes the single-galaxy term of the angular cross power.

        Args:

            nbar_x: (float) angular number density (per unit steradian) of X's.
            nbar_y: (float) angular number density (per unit steradian) of Y's.
            nxy: (float) if not None, the angular number density (per unit steradian) of X's that are
                 spatially overlapping with Y's. This can e.g. be the single-galaxy term.

        Returns:

            float.
        """

        assert isinstance(nbar_x, float) and (nbar_x >= 0.0)
        assert isinstance(nbar_y, float) and (nbar_y >= 0.0)
        assert isinstance(nxy, float) and (nxy >= 0.0)

        _nbar_xy = nbar_x * nbar_y

        if _nbar_xy == 0.0:
            return 0.0
        else:
            return nxy / _nbar_xy

    def _omega_f(self, dmin=None, dmax=None, mode='dm-shifting'):
        """
        This helper method returns the integrated first-order term in the expression for the perturbed
        number density of FRBs in a DM bin.

        Args:

            dmin: (float) if not None, minimum DM value in the bin.
            dmax: (float) if not None, maximum DM value in the bin.
            mode: (str) specifies the term: 'dm-shifting' or 'completeness'.

        Returns:

            function (z).
        """

        if (dmin is None) or (dmax is None):
            dmin = 0.0
            dmax = np.inf
        else:
            assert isinstance(dmin, float) and isinstance(dmax, float)
            assert 0.0 <= dmin < dmax

        assert mode in ('dm-shifting', 'completeness')

        if mode == 'dm-shifting':
            def ret(z):
                return self.dndz_f(z,cutoff=False) * (self.pdf_dh(dmin - self.ftc.dm_igm_interp(z))
                                                      - self.pdf_dh(dmax - self.ftc.dm_igm_interp(z)))
        elif mode == 'completeness':
            def ret(z):
                return fx.quad(lambda x: self.dndz_f(z,cutoff=False) * self.pdf_dh(x-self.ftc.dm_igm_interp(z))
                               * self.config.t_x(x), dmin, dmax, epsrel=1.0e-6, limit=10240)
        else:
            raise RuntimeError(f'clfg_theory._omega_f: {mode} is an invalid mode!')

        return ret

    def _w_ls(self, nbar_f, omega_f, zmax):
        """
        This helper method returns the radial weight function for the distribution of FRBs which are located
        behind electron density fluctuations (keeping only the first-order term) of massive halos along the
        line of sight.  It assumes the old definition of w_ls.

        Args:

            nbar_f: (float) angular number density (per unit steradian) of FRBs.
            omega_f: (callable) given (z), returns the integrated first-order term
                     in the expression for the perturbed number density of FRBs.
            zmax: (float) specifies the max redshift to be integrated.

        Returns:

            function (z).
        """

        assert isinstance(nbar_f, float) and (nbar_f >= 0.0)
        assert callable(omega_f)
        assert isinstance(zmax, float) and (zmax < self.ftc.interp_zmax)

        if nbar_f == 0.0:
            def ret(z):
                assert isinstance(z, float) and (z >= 0)
                return 0.0
        else:
            def ret(z):
                assert isinstance(z, float) and (z >= 0)

                _c = 1.0e6 / self.ftc.base.h       # Mpc/h -> pc.
                _ret = _c * self.ftc.ne0 * (1+z) / nbar_f
                _ret *= fx.quad(lambda x: omega_f(x), z, zmax, epsrel=1.0e-6, limit=10240)
                return _ret

        return ret

    def __nf_3d(self, z):
        """
        This special method computes the 3-d comoving number density of FRBs.

        Args:

            z: (float) redshift.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.ftc.interp_zmin <= z <= self.ftc.interp_zmax)

        ret = fx.quad(lambda m: self.ftc.hmf.interp_dn_dlog_m(z, m)
                      * self.n_f(z, np.exp(m)),
                      np.log(self.m_f), np.log(self.config.m_max))

        assert np.isfinite(ret) and (ret >= 0.0)
        return ret

    def __nfg_3d(self, z):
        """
        This special method computes the 3-d comoving number density of (FRB, galaxy) pairs in the same halo.

        Args:

            z: (float) redshxft.

        Returns:

            float.
        """

        _zmin = max(self.ftc.survey_galaxy.zmin, self.ftc.interp_zmin)
        _zmax = min(self.ftc.survey_galaxy.zmax, self.ftc.interp_zmax)
        assert isinstance(z, float) and (_zmin <= z <= _zmax)

        _m_min = max(self.m_f, self.ftc.m_g(z))
        assert self.ftc.hmf.m_min <= _m_min

        ret = fx.quad(lambda m: self.ftc.hmf.interp_dn_dlog_m(z, m)
                      * self.n_f(z, np.exp(m)) * self.ftc.n_g(z, np.exp(m)),
                      np.log(_m_min), np.log(self.config.m_max))

        assert np.isfinite(ret) and (ret >= 0.0)
        return ret

    def __bias_f(self, k, z, mc=10):
        """
        This special method computes the FRB bias.

        Args:

            k: (float) spatial wavenumber.
            z: (float) redshift.
            mc: (int) coefficient for partitioning the integration over mass,
                E.g., m=10 -> _int(x_min, x_max) = _int(x_min, x_min*10) + _int(x_min*10, x_max).

        Returns:

            float.
        """

        assert isinstance(k, float) and (self.ftc.hmf.kmin <= k <= self.ftc.hmf.kmax)
        assert isinstance(z, float) and (self.ftc.interp_zmin <= z <= self.ftc.interp_zmax)
        assert isinstance(mc, int) and (mc >= 2)

        def _int(m_min, m_max):
            assert m_min < m_max, "clfg_theory__bias_f: _int expects m_min < m_max. 'mc' might be too large!"

            m_min = np.log(m_min)
            m_max = np.log(m_max)

            _ret = fx.quad(lambda m: self.ftc.hmf.interp_bias(z, m)
                           * self.ftc.hmf.interp_dn_dlog_m(z, m)
                           * self.n_f(z, np.exp(m))
                           * self.ftc.rho_nfw(k, z, np.exp(m), normalized=True),
                           m_min, m_max)

            return _ret

        ret = _int(self.m_f, self.m_f*mc) + _int(self.m_f*mc, self.config.m_max)

        _nf_3d = self.interp_nf_3d(z)
        if _nf_3d > 0.0:
            assert np.isfinite(ret), f'clfg_theory.__bias_f: ret(k={k},z={z}) = {ret} / {_nf_3d}'
            ret = ret / _nf_3d
        else:
            ret = 0.0

        return ret

    def __zz_y(self, zz):
        """
        This special method slices redshift bins in redshift space.

        Args:

            zz: (1-d array) [min, max] bounds of redshift bins.

        Returns:

            list, containing 1-d arrays of sliced redshift ranges for interpolating objects in redshift bins.
        """

        zz_y = []
        for (i,j) in zip(*zz):
            i -= self.config.zpad_eps
            j += self.config.zpad_eps

            zmin = max(max(self.ftc.survey_galaxy.zmin, self.ftc.interp_zmin), i)
            zmax = min(self.ftc.survey_galaxy.zmax, j)

            zz_y.append(fx.lspace(zmin, zmax, self.config.interp_nstep_zz_g, log=False))

        return zz_y

    def __zz_x(self, dd_z, log):
        """
        This special method slices extragalactic DM bins in redshift space.

        Args:

            dd_z: (1-d array) [min, max] bounds of extragalactic DM bins, converted to redshift values.
            log: (bool) if True, then redshift slices are log-spaced.

        Returns:

            tuple (zz_x, zz_x_fine), containing lists of 1-d arrays, containing (finely) sliced redshift ranges for
            interpolating objects in DM bins.
        """

        zmin = self.ftc.interp_zmin

        ext = 0
        zz_x = []
        zz_x_fine = []
        for (i,j) in zip(*dd_z):
            if j > i:
                ext += 1

            j += self.config.zpad_eps

            assert i >= zmin
            assert j <= (self.ftc.interp_zmax - self.config.zpad_interp_zmax/5)

            def _bin(n):
                n1 = int(round((i-zmin) / (j-zmin) * n))
                n2 = n - n1 + 1
                if (n1 == 0) or (n2 < 2):
                    return fx.lspace(zmin, j, n, log)
                else:
                    return np.append(fx.lspace(zmin, i, n1, log),
                                     fx.lspace(i, j, n2, log)[1:])

            zz_x.append(_bin(self.config.interp_nstep_zz_f + self.config.interp_nstep_zz_x * ext))
            zz_x_fine.append(_bin(self.config.interp_nstep_zz_f_fine + self.config.interp_nstep_zz_x * ext))

        return zz_x, zz_x_fine

    def __bin_xy(self):
        """
        Initializes binned objects for computing (FRB, galaxy) cross power spectrum C_l^{fg} and observables
        for (an FRB model, a galaxy survey) in a (DM,z) bin.

        Initializes:

            self.zz, self.dd, self.dd_z,
            self.zz_x, self.zz_x_fine, self.zz_y,
            self.nbar_f, self.nbar_g, self.omega_f, self.w_ls
        """

        self.zz, self.dd, self.dd_z = self.ftc.bin(self.nz, self.nd)
        self.zz_x, self.zz_x_fine = self.__zz_x(self.dd_z, self.fpar[6])
        self.zz_y = self.__zz_y(self.zz)

        ###################################################################################################
        #
        # self.nbar_f

        nbar_x = []         # Angular number density of all FRBs binned in DM.
        nbar_sg = []        # Angular number density of single-galaxy FRBs binned in DM.

        if (self.nd > 1) or self.config.sim.force_bin_d:
            for dm_i in range(self.nd):
                (dmin, dmax) = (self.dd[0][dm_i], self.dd[1][dm_i])
                zmax_x = self.dd_z[1][dm_i] + self.config.zpad_eps
                assert zmax_x <= self.ftc.interp_zmax

                # n_x = self.ftc.n_x(self.n_f, dmin, dmax)
                # nbar_x = self.nbar_x(n_x=n_x, m_min=self.m_f, zmin=self.config.zmin, zmax=zmax_x)
                _nbar_x = fx.quad(lambda x: self.dndz_f(x) * self.prob_d_z(x, dmin, dmax), self.config.zmin, zmax_x)

                nbar_x.append(_nbar_x)

                # Assuming psg is not a function of halo mass <- (M = Mg+1).
                _nbar_sg = fx.quad(lambda x: self.ftc.dndz_galaxy(min(x, self.ftc.survey_galaxy.zmax)) *
                                   self.ftc.psg[self.frb_par_index](x, self.ftc.m_g(min(x,self.ftc.survey_galaxy.zmax))
                                   + 1.0) * self.prob_d_z(x, dmin, dmax), self.config.zmin, zmax_x)

                psg = _nbar_sg / _nbar_x
                assert 0.0 <= psg <= 1.0
                nbar_sg.append(_nbar_sg)

                print(f'dm_i={dm_i}, psg={psg}')
        else:
            _nbar_x = fx.quad(lambda x: self.dndz_f(x), self.config.zmin, self.config.fn_zmax)

            nbar_x.append(_nbar_x)

            _nbar_sg = fx.quad(lambda x: self.ftc.dndz_galaxy(min(x,self.ftc.survey_galaxy.zmax)) *
                               self.ftc.psg[self.frb_par_index](x, self.ftc.m_g(min(x, self.ftc.survey_galaxy.zmax))
                               + 1.0), self.config.zmin, self.config.fn_zmax)

            psg = _nbar_sg / _nbar_x
            assert 0.0 <= psg <= 1.0
            nbar_sg.append(_nbar_sg)

            print(f'dm_i=None, psg={psg}')

        print(f'sum(nbar_x) over DM bins: {np.sum(nbar_x):.1f}')
        print(f'sum(nbar_sg) over DM bins: {np.sum(nbar_sg):.1f}')

        nbar_tot_e = self.fpar[0] / (4.0 * np.pi * self.config.survey_frb.f_sky)
        nbar_tot_a = fx.quad(lambda x: self.dndz_f(x), self.config.zmin, self.config.zmax)
        print(f'nbar_tot_e = {nbar_tot_e:.1f},\nnbar_tot_a = {nbar_tot_a:.1f}')

        self.nbar_f = [nbar_x, nbar_sg]
        assert np.isfinite(self.nbar_f).all() and np.all(0.0 <= np.asarray(self.nbar_f))

        ###################################################################################################
        #
        # self.nbar_g

        self.nbar_g = []        # Angular number density of galaxies binned in redshift.
        for z_i in range(self.nz):
            (zmin, zmax) = (self.zz[0][z_i], self.zz[1][z_i])
            n_x = self.ftc.n_x(self.ftc.n_g, zmin, zmax)
            _nbar_g = self.nbar_x(n_x=n_x, m_min=self.ftc.m_g, zmin=zmin, zmax=zmax)
            self.nbar_g.append(_nbar_g)

        assert np.isfinite(self.nbar_g).all() and np.all(0.0 <= np.asarray(self.nbar_g))

        ###################################################################################################
        #
        # self.omega_f

        ext = 0
        self.omega_f = []
        for dm_i in range(self.nd):
            if (self.nd > 1) or self.config.sim.force_bin_d:
                (dmin, dmax) = (self.dd[0][dm_i], self.dd[1][dm_i])
            else:
                dmin = dmax = None

            zz_x_fine = self.zz_x_fine[dm_i]

            if self.dd_z[1][dm_i] > self.dd_z[0][dm_i]:
                ext += 1

            omega1 = self._omega_f(dmin=dmin, dmax=dmax, mode='dm-shifting')
            omega2 = self._omega_f(dmin=dmin, dmax=dmax, mode='completeness')

            v1 = np.zeros(self.config.interp_nstep_zz_f_fine + self.config.interp_nstep_zz_x * ext)
            v2 = np.zeros(self.config.interp_nstep_zz_f_fine + self.config.interp_nstep_zz_x * ext)
            for z_i, z in enumerate(zz_x_fine):
                print(f'(dm_i={dm_i},z_i={z_i})')
                v1[z_i] = omega1(z)
                v2[z_i] = omega2(z)

            _interp_omega1 = fx.spline(zz_x_fine, v1)
            _interp_omega2 = fx.spline(zz_x_fine, v2)
            self.omega_f.append([_interp_omega1, _interp_omega2])

        ###################################################################################################
        #
        # self.w_ls

        ext = 0
        self.w_ls = []
        for dm_i in range(self.nd):
            zz_x_fine = self.zz_x_fine[dm_i]

            if self.dd_z[1][dm_i] > self.dd_z[0][dm_i]:
                ext += 1

            w_ls1 = self._w_ls(nbar_f=self.nbar_f[0][dm_i], omega_f=self.omega_f[dm_i][0], zmax=self.dd_z[1][dm_i])
            w_ls2 = self._w_ls(nbar_f=self.nbar_f[0][dm_i], omega_f=self.omega_f[dm_i][1], zmax=self.dd_z[1][dm_i])

            v1 = np.zeros(self.config.interp_nstep_zz_f_fine + self.config.interp_nstep_zz_x * ext)
            v2 = np.zeros(self.config.interp_nstep_zz_f_fine + self.config.interp_nstep_zz_x * ext)
            for z_i, z in enumerate(zz_x_fine):
                print(f'(dm_i={dm_i},z_i={z_i})')
                v1[z_i] = w_ls1(z)
                v2[z_i] = w_ls2(z)

            _interp_w_ls1 = fx.spline(zz_x_fine, v1)
            _interp_w_ls2 = fx.spline(zz_x_fine, v2)
            self.w_ls.append([_interp_w_ls1, _interp_w_ls2])

    def __test_all(self):
        self.__test_pdf_dh()
        self.__test_cdf_dh()
        self.__test_interp_nf_3d()
        self.__test_interp_nfg_3d()
        self.__test_interp_bias_f()
        self.__test_interp_w_f()
        if (self.nz is not None) and (self.nd is not None):
            self.__test_nbar_x()
            self.__test_omega_f()
            self.__test_w_ls()
            self.__test_prob_d_z()
        self.__test_gamma_f()
        self.__test_beam()
        self.__test_csg_xy()

    def __test_pdf_dh(self, niter=1000, rtol=1.0e-6, atol=0.0):
        assert self.pdf_dh(-100.0) == 0.0
        assert self.pdf_dh(0.0) == 0.0
        assert self.pdf_dh(np.inf) == 0.0

        for d in rand.uniform(-1.0e4, 1.0e4, niter):
            p = self.pdf_dh(d)
            assert np.isfinite(p) and (0.0 <= p <= 1.0)

        _int = fx.quad(lambda _x: self.pdf_dh(_x), 0.0, np.inf)
        assert np.isclose(_int, 1.0, rtol, atol)

    def __test_cdf_dh(self, niter=1000):
        assert self.cdf_dh(-100.0) == 0.0
        assert self.cdf_dh(0.0) == 0.0
        assert self.cdf_dh(np.inf) == 1.0

        for d in rand.uniform(-1.0e4, 1.0e4, niter):
            p = self.cdf_dh(d)
            assert np.isfinite(p) and (0.0 <= p <= 1.0)

    def __test_interp_nf_3d(self, niter=100, rtol=1.0e-3, atol=1.0e-7):
        zvec = rand.uniform(self.ftc.interp_zmin, self.config.fn_zmax, niter)
        zvec = np.append([self.ftc.interp_zmin, self.config.fn_zmax], zvec)

        for z in zvec:
            a = self.interp_nf_3d(z)
            e = self.__nf_3d(z)

            assert np.isclose(a, e, rtol, atol), f'{z}, {a}, {e}'

        zvec = rand.uniform(self.config.zmin, self.config.fn_zmax, niter)

        for z in zvec:
            a_2d = self.ftc.dvoz(z) * self.interp_nf_3d(z)
            e_2d = self.dndz_f(z)

            assert np.isclose(a_2d, e_2d, rtol, atol), f'{z}, {a_2d}, {e_2d}'

    def __test_interp_nfg_3d(self, niter=100, rtol=1.0e-3, atol=1.0e-7):
        fgn_zmax = min(self.config.fn_zmax, self.ftc.survey_galaxy.zmax)
        zvec = rand.uniform(self.ftc.interp_zmin, fgn_zmax, niter)
        zvec = np.append([self.ftc.interp_zmin, fgn_zmax], zvec)

        for z in zvec:
            a = self.interp_nfg_3d(z)
            e = self.__nfg_3d(z)

            assert np.isclose(a, e, rtol, atol)

    def __test_interp_bias_f(self, niter=20, rtol=5.0e-3, atol=1.0e-7):
        zvec = rand.uniform(self.ftc.interp_zmin, self.config.fn_zmax, niter)
        zvec = np.append([self.ftc.interp_zmin, self.config.fn_zmax], zvec)

        kvec = rand.choice(self.ftc.kk, size=niter, replace=False)
        kvec = np.append([self.ftc.kk[0], self.ftc.kk[-1]], kvec)

        for z in zvec:
            for k in kvec:
                nf3d = self.interp_nf_3d(z)
                a = self.interp_bias_f(k, z) * nf3d
                e = self.__bias_f(k, z) * nf3d

                print(f'z={z}, k={k}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

    def __test_interp_w_f(self, niter=10, rtol=5.0e-3, atol=1.0e-10):
        w_x = self.ftc.w_x(n_x=self.n_f, m_min=self.m_f)

        zvec = rand.uniform(self.ftc.interp_zmin, self.config.fn_zmax, niter)
        zvec = np.append([self.ftc.interp_zmin, self.config.fn_zmax], zvec)

        lvec = rand.choice(self.ftc.ll, size=niter, replace=False)
        lvec = np.append(lvec, [self.ftc.ll[0], self.ftc.ll[-1]])

        for z in zvec:
            n3d = self.interp_nf_3d(z)

            for l in lvec:
                e = w_x(z, l)
                a = self.interp_w_f(z, l)

                print(f'z={z}, l={l}, a={a}, e={e}')
                assert np.isclose(a, e, rtol, atol)

                k = self.ftc.l_to_k(l, z)
                b = self.interp_bias_f(k, z) * n3d

                print(f'z={z}, l={l}, a={a}, b={b}')
                assert np.isclose(a, b, rtol, atol)

    def __test_nbar_x(self, rtol=1.0e-3, atol=0.0):
        assert isinstance(self.nbar_f, list) and (len(self.nbar_f) == 2)
        assert np.isfinite(self.nbar_f).all() and np.all(np.asarray(self.nbar_f) >= 0.0)

        for i in range(self.nd):
            print(f'{i}/{self.nd}')
            (dmin, dmax) = (self.dd[0][i], self.dd[1][i])
            n_x = self.ftc.n_x(self.n_f, dmin, dmax)
            zmax = self.dd_z[1][i] + self.config.zpad_eps
            a = self.nbar_x(n_x=n_x, m_min=self.m_f, zmin=self.config.zmin, zmax=zmax)
            e = self.nbar_f[0][i]
            assert np.isclose(a, e, rtol, atol)

        assert isinstance(self.nbar_g, list) and (len(self.nbar_g) == self.nz)
        assert np.isfinite(self.nbar_g).all() and np.all(np.asarray(self.nbar_g) >= 0.0)

        for j in range(self.nz):
            print(f'{j}/{self.nz}')
            (zmin, zmax) = (self.zz[0][j], self.zz[1][j])
            a = self.nbar_g[j]
            e = fx.quad(lambda x: self.ftc.dndz_galaxy(x), zmin, zmax)
            assert np.isclose(a, e, rtol, atol)

    def __test_omega_f(self, niter=20, rtol=5.0e-3, atol=1.0e-7):
        for i in range(self.nd):
            print(f"nd={i} {80*'-'}")
            (dmin, dmax) = (self.dd[0][i], self.dd[1][i])
            (zmin, zmax) = (self.zz_x_fine[i][0], self.zz_x_fine[i][-1])

            omega1 = self._omega_f(dmin=dmin, dmax=dmax, mode='dm-shifting')
            omega2 = self._omega_f(dmin=dmin, dmax=dmax, mode='completeness')

            for z in rand.uniform(zmin, zmax, niter):
                a1, a2 = float(self.omega_f[i][0](z)), float(self.omega_f[i][1](z))
                e1, e2 = float(omega1(z)), float(omega2(z))

                print(f'z={z}, a1={a1}, e1={e1}, a2={a2}, e2={e2}')
                assert np.isclose(a1, e1, rtol, atol)
                assert np.isclose(a2, e2, rtol, atol)

            x = self.zz_x[i]
            plt.plot(x, self.omega_f[i][0](x), label=r'${\rm d}%d$'%i)
            plt.plot(x, self.omega_f[i][1](x), label=r'${\rm c}%d$'%i, linestyle='dotted')

        plt.legend(loc='upper right', prop={'size': 3}, frameon=False)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\Omega_f$')
        if self.fpar[6]:
            plt.xscale('log')
        plt.xlim(self.zz_x[0][0], self.dd_z[0][-1]*1.1)
        plt.savefig(f'omega_f_{self.frb_par_index}.pdf')
        plt.clf()

    def __test_w_ls(self, niter=100, rtol=5.0e-3, atol=1.0e-10):
        for i in range(self.nd):
            print(f"nd={i} {80*'-'}")
            (zmin, zmax) = (self.zz_x_fine[i][0], self.zz_x_fine[i][-1])

            w_ls1 = self._w_ls(nbar_f=self.nbar_f[0][i], omega_f=self.omega_f[i][0], zmax=self.dd_z[1][i])
            w_ls2 = self._w_ls(nbar_f=self.nbar_f[0][i], omega_f=self.omega_f[i][1], zmax=self.dd_z[1][i])

            for z in rand.uniform(zmin, zmax, niter):
                a1, a2 = self.w_ls[i][0](z), self.w_ls[i][1](z)
                e1, e2 = w_ls1(z), w_ls2(z)

                print(f'z={z}, a1={a1}, e1={e1}, a2={a2}, e2={e2}')
                assert np.isclose(a1, e1, rtol, atol)
                assert np.isclose(a2, e2, rtol, atol)

            x = self.zz_x[i]
            plt.plot(x, self.w_ls[i][0](x), label=r'${\rm d}%d$'%i)
            plt.plot(x, self.w_ls[i][1](x), label=r'${\rm c}%d$'%i, linestyle='dotted')

        plt.legend(loc='upper right', prop={'size': 3}, frameon=False)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$W_{\rm prop}$')
        if self.fpar[6]:
            plt.xscale('log')
        plt.xlim(self.zz_x[0][0], self.dd_z[0][-1]*1.1)
        plt.savefig(f'w_prop_{self.frb_par_index}.pdf')
        plt.clf()

    def __test_prob_d_z(self, nz=10000):
        zvec = fx.lspace(self.dd_z[0][0], self.dd_z[0][-1]*1.1, nz, log=True)
        dvec = self.ftc.dm_igm_interp(zvec)
        p = np.zeros((self.nd, nz))

        for i in range(self.nd):
            (dmin, dmax) = (self.dd[0][i], self.dd[1][i])
            for j, z in enumerate(zvec):
                p[i,j] = self.prob_d_z(z, dmin, dmax)

            plt.semilogx(dvec, p[i,:], label=f'${i}$')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.xlabel(r'$D$')
        plt.ylabel(r'$\Delta {\rm CDF}$')
        plt.savefig(f'prob_d_z_fpar{self.frb_par_index}.pdf')
        plt.clf()

        print(f'Sum over D bins = {np.sum(p,axis=0)}')
        print(f'Sum over z bins = {np.sum(p,axis=1)}')

    def __test_gamma_f(self, niter=1000):
        fgn_zmax = min(self.config.fn_zmax, self.ftc.survey_galaxy.zmax)
        zvec = rand.uniform(self.ftc.interp_zmin, fgn_zmax, niter)
        for z in zvec:
            _g_f = self.gamma_f(z)
            assert np.isfinite(_g_f)

    def __test_beam(self, niter=1000, rtol=1.0e-10, atol=0.0):
        for l in rand.uniform(-1e4, 1e4, niter):
            beam = self.beam(0.0*u.arcsec)
            assert np.isclose(beam(l), 1.0, rtol, atol)

        beam = self.beam(rand.uniform(high=100, size=1)[0] * u.arcmin)
        for l in rand.uniform(-1.0e6, 1.0e6, niter):
            assert 0.0 <= beam(l) <= 1.0
            assert beam(-l) == beam(l)

    def __test_csg_xy(self, niter=1000):
        r = rand.normal(size=(niter,3))
        nv = 0
        for i, x in enumerate(r):
            try:
                ret = self.csg_xy(x[0], x[1], x[2])
            except AssertionError:
                continue
            assert isinstance(ret, float) and np.isfinite(ret) and (ret >= 0.0)
            nv += 1
        assert nv > 0
