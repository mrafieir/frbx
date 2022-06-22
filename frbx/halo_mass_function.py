import numpy as np
import numpy.random as rand
import scipy.special as ssp
import astropy.units as u
from astropy.cosmology import FLRW
import matplotlib.pyplot as plt
import frbx as fx
from frbx.configs import tiny, eps


class halo_mass_function:
    """
    Sheth-Tormen mass function and halo-related quantities.

    Implicit units:

        Mass:                   M_sun/h.
        Distance:               Mpc/h (comoving).
        Spatial wavenumber:     h/Mpc (comoving).

    Members:

        self.cosmo: (obj) instance of astropy cosmology.
        self.pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
        self.rho_m: (float) matter density (comoving).
        self.zmin: (float) min redshift bound.
        self.zmax: (float) max redshift bound.
        self.interp_nstep_z: (int) number of interpolation steps along the z axis.
        self.m_min: (float) min halo mass.
        self.m_max: (float) max halo mass.
        self.interp_nstep_m: (int) number of interpolation steps along the m axis.
        self.kmin: (float) min k above which self.p_k is defined with extrapolation.
        self.kmax: (float) max k up to which self.p_k is defined with extrapolation.
        self.p_k: (function) camb interpolator for the matter power spectrum.
        self.fpar: (list) model parameters for the Sheth-Tormen mass function.
        self.z: (1-d array) linearly-spaced redshift shells mainly used for interpolations.
        self.m: (1-d array) log-spaced mass bins.
        self.rho_vir: (interp obj) virial matter density (comoving) as a function of redshift.
        self.interp_dn_dlog_m: (lambda) interpolated Sheth-Tormen mass function (z,log_m).
        self.interp_bias: (lambda) interpolated Sheth-Tormen halo bias (z,log_m).
        self.interp_conc: (lambda) interopolated NFW concentration parameter (z,log_m).
        self.interp_r_vir: (lambda) interpolated virial radius (z,log_m).
        self.r: (method) comoving radius of a spherical halo.
        self.f_coll: (method) fraction of matter which is collapsed into halos of mass >= a min mass.
        self._f_coll_sigma: (helper method) fraction of matter which is collapsed into halos <= a max sigma.
        self._w: (helper method) Fourier transform of a spherical tophat (incl. derivatives).
        halo_mass_function.w_cutoff: (static method) smooth cutoff function.
        self._sigma2: (helper method) square of the RMS amplitude of the linear density field (incl. derivatives).
        self._f: (helper method) collapse fraction f(RMS_amplitude) which appears in the Sheth-Tormen mass function.
        self._dlog_dm: (helper method) derivative of log(1/RMS_amplitude) with respect to mass.
        self._dn_dm: (helper method) Sheth-Tormen mass function.
        self._bias: (helper method) Sheth-Tormen halo bias.
        self.conc: (method) NFW concentration parameter.
        self._halo_mass_function__test_*.
    """

    def __init__(self, cosmo, pkl_dir, zmin, zmax, p_k, kmin, kmax, interp_nstep_z=512, m_min=1e2, m_max=1e19,
                 interp_nstep_m=512, fpar=None):
        """
        Constructor arguments:

            cosmo: (obj) instance of astropy cosmology.
            pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
            zmin: (float) min redshift bound.
            zmax: (float) max redshift bound.
            p_k: (function) camb interpolator for the matter power spectrum. (*)
            kmin: (float) min k above which p_k is defined with extrapolation. (*)
            kmax: (float) max k up to which p_k is defined with extrapolation. (*)
            interp_nstep_z: (int) number of interpolation steps along the z axis.
            m_min: (float) min halo mass.
            m_max: (float) max halo mass.
            interp_nstep_m: (int) number of interpolation steps along the m axis.
            fpar: (list) if not None, model parameters for the Sheth-Tormen mass function: [A, a, delta_c, p]

            (*)  requires the following args for constructing a p_k with 'ft' conventions:
                    hubble_units = True -> self.p_k outputs in (Mpc/h)^3.
                    k_hunit = True      -> self.p_k = P(k), where k is assumed to be in (h/Mpc).
                    extrap_kmax = kmax, where e.g. kmax=1.0e5 here by default.
                    p_k(k,z) is extrapolated smoothly to (k/kmin) * p_k(kmin,z) for all (k<kmin, z).
        """

        assert isinstance(cosmo, FLRW)
        assert isinstance(pkl_dir, str) and pkl_dir.endswith('/')
        assert isinstance(zmin, float) and isinstance(zmax, float)
        assert 0.0 <= zmin < zmax
        assert isinstance(interp_nstep_z, int) and (10 <= interp_nstep_z)
        assert isinstance(m_min, float) and isinstance(m_max, float)
        assert 1.0 <= m_min < m_max
        assert isinstance(interp_nstep_m, int) and (10 <= interp_nstep_m)

        self.cosmo = cosmo
        self.pkl_dir = pkl_dir

        rho_m = self.cosmo.critical_density(0.) * self.cosmo.Om(0.)
        self.rho_m = rho_m.to(u.M_sun/u.Mpc**3).value / self.cosmo.h**2

        self.zmin = zmin
        self.zmax = zmax
        self.interp_nstep_z = interp_nstep_z
        self.m_min = m_min
        self.m_max = m_max
        self.interp_nstep_m = interp_nstep_m
        self.kmin = kmin
        self.kmax = kmax
        self.p_k = p_k

        self.fpar = [0.3222, 0.707, 1.686, 0.3] if fpar is None else fpar
        assert isinstance(self.fpar, list)
        assert len(self.fpar) == 4

        self.z = np.linspace(self.zmin, self.zmax, self.interp_nstep_z, dtype=np.float64)
        self.m = fx.logspace(self.m_min/10., self.m_max*10., self.interp_nstep_m)

        def _rv(x):
            ret = 178 * self.cosmo.Om(x)**0.45 * self.cosmo.critical_density(x) / (1+x)**3
            return ret.to(u.M_sun / u.Mpc**3).value / self.cosmo.h**2

        _rho_vir = np.asarray([_rv(i) for i in self.z])
        assert np.isfinite(_rho_vir).all() and (_rho_vir >= 0.0).all()

        self.rho_vir = fx.spline(self.z, _rho_vir)

        _path = pkl_dir + 'hmf_objs.pkl'
        try:
            hmf = fx.read_pickle(_path)
        except OSError as err:
            print(err)
            _dn_dlog_m = np.zeros((self.z.size, self.m.size))
            _bias = np.zeros((self.z.size, self.m.size))
            _conc = np.zeros((self.z.size, self.m.size))
            _r_vir = np.zeros((self.z.size, self.m.size))

            for i, z in enumerate(self.z):
                for j, m in enumerate(self.m):
                    print(f'(i={i},j={j})')

                    _dn_dlog_m[i,j] = self._dn_dm(z, m) * m
                    _bias[i,j] = self._bias(z, m)
                    _conc[i,j] = self.conc(z, m)
                    _r_vir[i,j] = self.r(z, m, vir=True)

            _log_m = np.log(self.m)

            _dn_dlog_m[_dn_dlog_m < tiny] = tiny
            log_dn_dlog_m = np.log(_dn_dlog_m)

            _bias[_bias < tiny] = tiny
            log_bias = np.log(_bias)

            interp_log_dn_dlog_m = fx.spline(self.z, _log_m, log_dn_dlog_m)
            interp_log_bias = fx.spline(self.z, _log_m, log_bias)
            interp_conc = fx.spline(self.z, _log_m, _conc)
            interp_r_vir = fx.spline(self.z, _log_m, _r_vir)

            hmf = [interp_log_dn_dlog_m, interp_log_bias, interp_conc, interp_r_vir]
            fx.write_pickle(_path, hmf)

        self.interp_dn_dlog_m = lambda redshift, log_mass: np.exp(hmf[0](redshift, log_mass))
        self.interp_bias = lambda redshift, log_mass: np.exp(hmf[1](redshift, log_mass))
        self.interp_conc = lambda redshift, log_mass: hmf[2](redshift, log_mass)
        self.interp_r_vir = lambda redshift, log_mass: hmf[3](redshift, log_mass)

    def r(self, z, m, ddm=False, vir=False):
        """
        This method computes the (comoving) radius of a sphere with mass m at redshift z.

        Args:

            z: (float or array) halo redshift.
            m: (float or array) halo mass.
            ddm: (bool) whether to return the function or its derivative with respect to mass.
            vir: (bool) whether to assume a virialized mass (astro-ph/9708070).

        Returns:

            (float) radius of a spherical halo.
        """

        assert isinstance(m, (float, np.ndarray))

        if isinstance(z, np.ndarray):
            assert isinstance(m, np.ndarray)
            assert z.shape == m.shape
        elif isinstance(z, float):
            pass
        else:
            raise RuntimeError('halo_mass_function.r: invalid type encountered! Check the first arg (z).')

        assert np.all(z >= 0.0), '%s' % z
        assert np.all(m >= 0.0), '%s' % m

        assert isinstance(ddm, bool)
        assert isinstance(vir, bool)

        num = 3.0 * m               # M_sun / h.

        den = 4 * np.pi
        if not vir:
            den *= self.rho_m       # M_sun * h^2 / Mpc^3.
        else:
            den *= self.rho_vir(z)

        if not ddm:
            r = ((num/den)**(1.0/3))
        else:
            r = ((num/den)**(-2.0/3) / den)

        return r

    def f_coll(self, z, m):
        """
        This method returns the fraction of matter which is collapsed into halos of mass >= m.

        Args:

            z: (float) halo redshift.
            m: (float) halo mass.

        Returns:

            float.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m >= 0.0)

        sigma = np.sqrt(self._sigma2(z, m))

        return self._f_coll_sigma(sigma)

    def _f_coll_sigma(self, sigma_max):
        """
        This helper function returns the fraction of matter which is collapsed into
        halos of sigma <= sigma_max.

        Args:

            sigma_max: (float) maximum sigma value.

        Returns:

            float.
        """

        assert isinstance(sigma_max, float) and (1.0e-16 <= sigma_max <= 1.0e10)

        ret = fx.quad(lambda sigma: self._f(np.exp(sigma)), np.log(1.0e-16), np.log(sigma_max))
        assert np.isfinite(ret)

        return ret

    def _w(self, m, k, ddm=False):
        """
        This helper method computes the (derivative of the) Fourier transform of a spherical
        tophat with mass m(r).

        Args:

            m: (float) halo mass.
            k: (float) k parameter (in h/Mpc) specifying the Fourier domain.
            ddm: (bool) whether to return the function or its derivative with respect to mass.

        Returns:

            (float) Fourier transform of the spherical tophat evaluated at k.
        """

        assert isinstance(m, float) and (m > 0.0)
        assert isinstance(k, float) and (k >= 0.0)
        assert isinstance(ddm, bool)

        x = k * self.r(0.0,m)

        if not ddm:
            return 3.0 * ssp.spherical_jn(1,x) / x
        else:
            return 3.0 * k * self.r(0.0, m, ddm=ddm) * (np.sin(x) * (x**2.0 - 3) + (3.0 * x) * np.cos(x)) / x**4.0

    @staticmethod
    def w_cutoff(x):
        """
        This static method computes a smooth cutoff function W(x) satisfying
    
        W(x) = 1     for x < 0.5
        W(x) = 0     for x > 1
    
        Changing variables to t = 2-2x, this becomes
    
        W(t) = 0           for t <= 0
        W(t) = t^2 (3-2t)  for 0 < t < 1
        W(t) = 1           for t > 1

        Originally written by Kendrick Smith.

        Args:

            x: (float) independent parameter.

        Returns:

            float.
        """

        assert isinstance(x, float)

        t = 2 - 2*x
        t = min(t, 1.0)
        t = max(t, 0.0)

        return t**2 * (3-2*t)

    def _sigma2(self, z, m, ddm=False):
        """
        This helper method computes the (derivative of the) square of the RMS amplitude
        of the linear density field.

        Args:

            z: (float) redshift of the halo.
            m: (float) halo mass.
            ddm: (bool) whether to return the function or its derivative with respect to mass.

        Returns:

            (float) square of the RMS amplitude or its derivative with respect to mass.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m > 0.0)
        assert isinstance(ddm, bool)

        # k values are assumed to be in h/Mpc.
        kmin = 1.0001 * self.kmin
        kmax = min(100.0 / self.r(0.0, m), 0.9999 * self.kmax)

        if not ddm:
            return fx.quad(lambda k: k**2.0 / (2*np.pi**2) * self.p_k(k=k,z=z)
                           * self._w(m,k)**2.0 * self.w_cutoff(k/kmax), kmin, kmax)
        else:
            return fx.quad(lambda k: k**2.0 / np.pi**2 * self.p_k(k=k,z=z)
                           * self._w(m,k) * self._w(m,k,ddm) * self.w_cutoff(k/kmax), kmin, kmax)

    def _f(self, sigma):
        """
        This helper method computes the collapse fraction f(sigma) which appears in
        the definition of Sheth-Tormen mass function.

        Args:

            sigma: (float) independent parameter, sigma.

        Returns:

            float.
        """

        assert isinstance(sigma, float) and (sigma > 0.0)

        r1 = self.fpar[0] * self.fpar[2] / sigma * np.sqrt(2 * self.fpar[1]/np.pi)
        r2 = 1 + (sigma**2.0 / self.fpar[1] / self.fpar[2]**2.0)**self.fpar[3]
        r3 = np.exp(-self.fpar[1] * self.fpar[2]**2.0 / 2 / sigma**2.0)

        ret = r1 * r2 * r3

        assert np.isfinite(ret)

        return ret

    def _dlog_dm(self, z, m):
        """
        This helper method returns the mass derivative of log(1/sigma) evaluated at z.

        Args:

            z: (float) redshift of the halo.
            m: (float) halo mass.

        Returns:

            (float) derivative of log(1/sigma) with respect to mass.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m > 0.0)

        _s = self._sigma2(z, m)

        if _s == 0.0:
            return 0.0
        else:
            return self._sigma2(z,m,ddm=True) / _s / (-2.)

    def _dn_dm(self, z, m):
        """
        This helper method returns the Sheth-Tormen mass function for a given mass m at redshift z.

        Args:

            z: (float) redshift of the halo.
            m: (float) halo mass.

        Returns:

            (float) number of halos per unit mass per unit comoving volume.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m > 0.0)

        sigma = np.sqrt(self._sigma2(z, m))

        # self.rho_m evaluated at z=0 -> comoving density.
        ret = (self.rho_m / m * self._dlog_dm(z,m) * self._f(sigma))

        return ret

    def _bias(self, z, m):
        """
        This helper method returns the Sheth-Tormen halo bias.

        Args:

            z: (float) redshift of the halo.
            m: (float) halo mass.

        Returns:

            (float) linear halo bias.
        """

        assert isinstance(z, float) and (z >= 0.0)
        assert isinstance(m, float) and (m > 0.0)

        sigma2 = self._sigma2(z, m)
        v = self.fpar[1] * self.fpar[2]**2.0 / sigma2

        r1 = v - 1.0
        r2 = 2.0 * self.fpar[3] / (1.0 + v**self.fpar[3])

        ret = 1.0 + (r1+r2) / self.fpar[2]

        assert 0.0 < ret

        return ret

    def conc(self, z, m):
        """
        This method computes the NFW concentration parameter using a
        fitting model (1402.7073) for halos with mass m at redshift z.

        Args:

            z: (float) halo redshift.
            m: (float or array) halo mass.

        Returns:

            float.
        """

        assert isinstance(z, float) and (self.zmin <= z <= self.zmax)
        assert isinstance(m, (float, np.ndarray))

        alpha = 0.537 + 0.488 * np.exp(-0.718 * z**1.08)
        beta = -0.097 + 0.024 * z

        _log_m = np.log10(m / 1.0e12)

        ret = 10**(alpha + (beta*_log_m))

        assert np.isfinite(ret).all()
        assert np.all(0.0 < ret)

        return ret
    
    def __test_all(self):
        self.__test_interp_dn_dlog_m()
        self.__test_interp_bias()
        self.__test_interp_conc()
        self.__test_interp_r_vir()
        self.__test_r()
        self.__test_f_coll()
        self.__test__f_coll_sigma()
        self.__test__w()
        self.__test_w_cutoff()
        self.__test__sigma2()
        self.__test__f()
        self.__test__dlog_dm()

    def __test_interp_dn_dlog_m(self, niter=100, rtol=1.0e-2, atol=eps):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            e = self._dn_dm(z, m)
            assert isinstance(e, float) and np.isfinite(e) and (e >= 0.0)

            a = float(self.interp_dn_dlog_m(z, np.log(m)))
            a /= m
            assert a >= 0.0

            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

        mvec = np.logspace(np.log10(self.m_min), np.log10(self.m_max), niter)

        for z in (self.zmin, 1.0, 2.0, 3.0, self.zmax):
            n = [m * self.interp_dn_dlog_m(z, np.log(m)).flatten() for m in mvec]
            plt.loglog(mvec, n, label=f'$z = {z}$')

        plt.xlabel(r'$M$')
        plt.ylabel(r'$M^2n_h(M,z)$')
        plt.ylim(1.0e7, 1.0e10)
        plt.legend(loc='lower left', prop={'size': 8}, frameon=False)
        plt.savefig('test_interp_dn_dlog_m.pdf')
        plt.clf()

    def __test_interp_bias(self, niter=100, rtol=1.0e-3, atol=eps):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        (zvec[0], zvec[-1]) = (self.zmin, self.zmax)

        mvec = rand.uniform(self.m_min, self.m_max, niter)
        (mvec[0], mvec[-1]) = (self.m_min, self.m_max)

        for (z,m) in zip(zvec, mvec):
            e = self._bias(z, m)
            assert isinstance(e, float) and np.isfinite(e) and (e > 0.0)

            a = self.interp_bias(z, np.log(m))
            assert a > 0.0

            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

        mvec = np.logspace(np.log10(self.m_min), np.log10(self.m_max), 100)
        rtol = 0.2  # Relaxing tolerance is OK here since we're cross-checking def's below.

        for z in zvec:
            sigma = np.sqrt([self._sigma2(z, m) for m in mvec])
            sigma[sigma == 0.0] = tiny

            f = np.asarray([self._f(s) for s in sigma])

            _sp = fx.spline(np.log(sigma)[::-1], f[::-1])
            _dsp = _sp.derivative(n=1)

            def e_bias(x):
                return 1.0 + _dsp(np.log(sigma[x])) / self.fpar[2] / _sp(np.log(sigma[x]))

            for i in range(0, int(np.sum(mvec < 1.0e12)), 10):      # _dsp fails beyond this mass limit.
                e = e_bias(i)
                a = self.interp_bias(z, np.log(mvec[i]))
                assert np.isclose(a, e, rtol, atol)

        for z in (self.zmin, 1.0, 2.0, 3.0, self.zmax):
            bias = [self.interp_bias(z, np.log(m)).flatten() for m in mvec]
            plt.loglog(mvec, bias, label=f'$z = {z}$')

        plt.xlabel(r'$M$')
        plt.ylabel(r'$b_h(M,z)$')
        plt.legend(loc='upper left', prop={'size': 8}, frameon=False)
        plt.savefig('test_interp_bias.pdf')
        plt.clf()

    def __test_interp_conc(self, niter=100, rtol=1.0e-4, atol=0.0):
        zvec = np.linspace(self.zmin, self.zmax, niter)
        mvec = np.logspace(np.log10(self.m_min), np.log10(self.m_max), niter)

        for (z,m) in zip(zvec, mvec):
            e = self.conc(z, m)
            assert np.isfinite(e) and (e >= 0.0)

            a = self.interp_conc(z, np.log(m))
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

        for z in (self.zmin, 1.0, 2.0, 3.0, self.zmax):
            conc = [self.interp_conc(z, np.log(m)).flatten() for m in mvec]
            plt.loglog(mvec, conc, label=f'$z = {z}$')

        plt.xlabel(r'$M$')
        plt.ylabel(r'$c(M,z)$')
        plt.legend(loc='lower right', prop={'size': 8}, frameon=False)
        plt.savefig('test_interp_conc.pdf')
        plt.clf()

    def __test_interp_r_vir(self, niter=100, rtol=1.0e-7, atol=0.0):
        zvec = np.linspace(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            e = self.r(z, m, vir=True)
            assert np.isfinite(e) and (e >= 0.0)

            a = self.interp_r_vir(z, np.log(m))
            assert np.isclose(a, e, rtol, atol), f'({a}, {e})'

    def __test_r(self, niter=100):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            ddm = [False, True][rand.randint(2)]

            r1 = self.r(z, m, ddm=ddm, vir=False)
            assert isinstance(r1, float) and np.isfinite(r1) and (r1 > 0.0)

            r2 = self.r(z, m, ddm=ddm, vir=True)
            assert isinstance(r2, float) and np.isfinite(r2) and (r2 > 0.0)

            assert r1 > r2

    def __test_f_coll(self, niter=100):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            f = self.f_coll(z, m)
            assert isinstance(f, float) and np.isfinite(f) and (f >= 0.0)

    def __test__f_coll_sigma(self, rtol=1.0e-4, atol=0.0):
        assert np.isclose(self._f_coll_sigma(1.0e10), 1.0, rtol, atol)

    def __test__w(self, niter=10, nk=100):
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))
        kvec = rand.uniform(self.kmin, self.kmax, niter)

        _k = np.linspace(self.kmin, self.kmax, nk)
        for (m,k) in zip(mvec, kvec):
            ddm = [False, True][rand.randint(2)]
            w = self._w(m, k, ddm=ddm)
            assert isinstance(w, float) and np.isfinite(w)

            _w = [self._w(m, _, ddm=False) for _ in _k]
            plt.plot(_k, _w, label=f'log(m) = {np.log(m):.2f}')

        plt.legend(loc='upper right', prop={'size': 8}, frameon=False)
        plt.savefig('test_w.pdf')
        plt.clf()

    def __test_w_cutoff(self, niter=10000):
        for i in rand.uniform(-1e3, 1e3, niter):
            _w = self.w_cutoff(i)
            assert isinstance(_w, float) and np.isfinite(_w)

            i = 2.0 - 2.0 * i
            if i <= 0.0:
                assert (_w == 0.0), _w
            elif 0.0 < i < 1.0:
                assert (_w == i**2.0 * (3.0 - 2.0*i)), _w
            else:
                assert (_w == 1.0), _w

    def __test__sigma2(self, niter=100):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            ddm = [False, True][rand.randint(2)]
            _s = self._sigma2(z, m, ddm=ddm)
            assert isinstance(_s, float) and np.isfinite(_s)

        mvec = np.logspace(np.log10(self.m_min), np.log10(self.m_max), niter)

        for z in (self.zmin, 1.0, 2.0, 3.0, self.zmax):
            s = [self._sigma2(z, m, ddm=False)**0.5 for m in mvec]
            plt.loglog(mvec, s, label=f'$z = {z}$')

        plt.xlabel(r'$M$')
        plt.ylabel(r'$\sigma(M,z)$')
        plt.legend(loc='lower left', prop={'size': 8}, frameon=False)
        plt.savefig('test_sigma2.pdf')
        plt.clf()

    def __test__f(self, niter=1000):
        for sigma in rand.uniform(1e-3, 1e3, niter):
            _f = self._f(sigma)
            assert isinstance(_f, float) and np.isfinite(_f) and (_f >= 0.0)

    def __test__dlog_dm(self, niter=100):
        zvec = rand.uniform(self.zmin, self.zmax, niter)
        mvec = np.exp(rand.uniform(np.log(self.m_min), np.log(self.m_max), niter))

        for (z,m) in zip(zvec, mvec):
            ret = self._dlog_dm(z, m)
            assert isinstance(ret, float) and np.isfinite(ret)
