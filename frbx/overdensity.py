import time
import numpy as np
import matplotlib.pyplot as plt
import healpy
import frbx as fx


class galaxy_overdensity:
    """
    A 'galaxy_overdensity' object is constructed from a 'galaxy_catalog' object.  The galaxy_catalog
    represents a list of objects, whereas the galaxy_overdensity represents the 2-d field delta_g,
    with a model for the (masked) selection function subtracted.

    Constructor arguments:

      - galaxy_cat: An instance of class galaxy_catalog.

      - nside: Healpix resolution parameter.

      - lmax: Maximum multipole used when computing a_{lm} values (or C_l^{gg}, C_l^{fg}).

      - healpix_mask: A 1-d array of zeros (representing masked pixels) or ones (representing
        unmasked pixels).

      - randcat: An instance of class galaxy_catalog, representing random objects.

      - randmap: A 1-d array representing the overdensity field of random objects.

      - map2alm_iter: Iteration count in healpy.sphtfunc.map2alm().  I haven't tried playing
        with this, to see whether it makes any difference in the final power spectra.

      - interpolate: If True, then a CIC-like weighting scheme is assumed throughout.
    """

    def __init__(self, galaxy_cat, nside, lmax, healpix_mask, randcat=None, randmap=None, map2alm_iter=3, interpolate=False):
        assert isinstance(galaxy_cat, fx.galaxy_catalog)
        assert healpix_mask.ndim == 1
        assert np.all(np.logical_or(healpix_mask == 0, healpix_mask == 1))
        assert np.any(healpix_mask == 1)

        self.catalog = galaxy_cat
        self.nside = nside
        self.lmax = lmax
        self.randcat = randcat
        self.randmap = randmap

        self.healpix_mask = healpix_mask
        fsky = np.mean(self.healpix_mask)

        _npix = len(self.healpix_mask)
        _nside = healpy.pixelfunc.npix2nside(_npix)

        if _nside != self.nside:
            self.healpix_mask = healpy.pixelfunc.ud_grade(self.healpix_mask, self.nside)
            self.healpix_mask[self.healpix_mask != 0.0] = 1.0
            assert np.mean(self.healpix_mask) == fsky

        self.interpolate = interpolate
        self.map2alm_iter = map2alm_iter

        npix = healpy.pixelfunc.nside2npix(self.nside)
        self.pixarea = (4*np.pi) / npix      # pixel area in steradians

        if (self.randcat is not None) and (self.randmap is not None):
            raise RuntimeError('galaxy_overdensity: only one of randcat or randmap may be provided, not both!')

        if self.randcat is not None:
            assert isinstance(self.randcat, fx.galaxy_catalog)
            r = fx.utils.make_healpix_map_from_catalog(self.nside, self.randcat.l_deg, self.randcat.b_deg,
                                                       weight=(1.0/self.pixarea), interpolate=self.interpolate)
        elif self.randmap is not None:
            assert isinstance(self.randmap, np.ndarray)
            assert self.randmap.size == npix
            r = self.randmap
        else:
            r = 1.0

        if (self.randcat is not None) or (self.randmap is not None):
            self.healpix_mask *= r.astype(bool).astype(float)
            self.fsky = np.mean(self.healpix_mask)
            self.nr_2d = np.mean(self.healpix_mask * r) / self.fsky
        else:
            self.fsky = fsky
            self.nr_2d = 1.0

        # We weight each pixel count with a factor (1/Omega_pix), as appropriate when discretizing
        # a sum of delta functions (see frbx_pipelines_notes.tex).
        g = fx.utils.make_healpix_map_from_catalog(self.nside, self.catalog.l_deg, self.catalog.b_deg,
                                                   weight=(1.0/self.pixarea), interpolate=self.interpolate)

        # The 2d galaxy density has units sr^{-1}, and is computed using only unmasked pixels.
        self.ng_2d = np.mean(self.healpix_mask * g) / self.fsky

        # Galaxy overdensity delta_g(x) = (g(x)/gbar - r(x)/rbar).
        # Code checks: this map should have zero mean, and should be zero in masked pixels.
        self.deltag_map = self.healpix_mask * (g/self.ng_2d - r/self.nr_2d)

        # Spherical transform delta_g(x) -> a_{lm}^g
        self.deltag_alm = healpy.sphtfunc.map2alm(self.deltag_map, lmax, iter=map2alm_iter, pol=False)

        # We estimate C_l^{gg} from a_{lm}^g, by pretending that delta_g is all-sky, then applying
        # the debiasing factor 1/fsky.
        self.clgg = healpy.sphtfunc.alm2cl(self.deltag_alm) / self.fsky

    def plot_deltag(self, doc=None, nside=None):
        m = healpy.pixelfunc.ud_grade(self.deltag_map, nside) if (nside is not None) else self.deltag_map
        fx.utils.show_healpix_map(m, doc)

    def plot_clgg(self, doc=None, aux=None, plt_args=None):
        if plt_args is None:
            plt_args = {}
        b = fx.simple_l_binning(self.lmax)

        plt.loglog(b.l_vals, b.bin_average(self.clgg))
        if aux is not None:
            plt.loglog(*aux)

        plt.axhline(1.0/self.ng_2d, color='red', ls='--', label=r'$1/n_g$')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{gg}$')
        plt.legend(loc='upper right').draw_frame(False)
        fx.showfig(doc, plt_args)


class frb_overdensity:
    """
    An 'frb_overdensity' object is constructed from a 'frb_catalog' object.  The frb_catalog
    represents a list of objects, whereas the frb_overdensity represents the 2-d field delta_f,
    with a model for the selection function subtracted.

    In this simple version, we model the selection function by subtracting a random catalog 
    from the FRB catalog.  The random catalog is constructed by taking real declinations from
    the FRB catalog, and randomly assigning RA's.

    Constructor arguments:

      - frb_cat: An instance of class frb_catalog.

      - nside: Healpix resolution parameter.

      - lmax: Maximum multipole used when computing a_{lm} values (or C_l^{gg}, C_l^{fg}).

      - dec_min, dec_max: If specified, then a simple sky mask will be imposed, by restricting
        FRBs to declination range dec_min <= dec <= dec_max.

      - rmult: This parameter determines the size of FRB random catalogs.  Random catalogs
        will be larger than "data" catalogs (or mocks) by a factor rmult.

      - map2alm_iter: Iteration count in healpy.sphtfunc.map2alm().  I haven't tried playing
        with this, to see whether it makes any difference in the final power spectra.

      - interpolate: If True, then a CIC-like weighting scheme is assumed throughout.

      - nmc: Number of mocks

    In plot_*() methods, the 'doc' argument should either be None (to show a plot interactively), 
    or a handout.Handout instance (to show a plot in a web-browsable output directory).
    """

    def __init__(self, frb_cat, nside, lmax, dec_min=None, dec_max=None, rmult=1000, map2alm_iter=3,
                 interpolate=False, nmc=0):

        assert isinstance(frb_cat, fx.frb_catalog)

        self.nside = nside
        self.lmax = lmax

        _dec_min = np.min(frb_cat.dec_deg)
        self.dec_min = _dec_min if (dec_min is None) else max(_dec_min, dec_min)
        self.dec_max = 90.0 if (dec_max is None) else dec_max

        self.rmult = rmult
        self.map2alm_iter = map2alm_iter
        self.interpolate = interpolate

        mask = np.logical_and(frb_cat.dec_deg >= self.dec_min, frb_cat.dec_deg <= self.dec_max)
        assert np.sum(mask) > 0

        self.ra_deg = frb_cat.ra_deg[mask]
        self.dec_deg = frb_cat.dec_deg[mask]

        if frb_cat.mocks is None:
            self.mocks = None
        elif isinstance(frb_cat.mocks, np.ndarray):
            self.mocks = frb_cat.mocks[mask,...]
        else:
            assert np.sum(mask) == frb_cat.size,\
                'frb_overdensity.__init__: jackknifed mocks are not compatible with masked catalogs'
            self.mocks = frb_cat.mocks

        self.size = len(self.ra_deg)

        self.pixarea = (4*np.pi) / healpy.pixelfunc.nside2npix(self.nside)       # pixel area in steradians
        self.fsky = (np.sin(self.dec_max*np.pi/180.) - np.sin(self.dec_min*np.pi/180.)) / 2.0

        # 2d FRB number density (units sr^{-1})
        self.nf_2d = self.size / (self.fsky * 4 * np.pi)

        # The helper function _make_overdensity_map() makes a Healpix overdensity map delta_f
        # from the FRB catalog (see below).
        self.deltaf_map = self._make_overdensity_map(self.ra_deg, self.dec_deg)

        # Spherical transform delta_f(x) -> a_{lm}^f
        self.deltaf_alm = healpy.sphtfunc.map2alm(self.deltaf_map, self.lmax, iter=map2alm_iter, pol=False)

        # We estimate C_l^{ff} from a_{lm}^f, by pretending that delta_f is all-sky, then applying
        # the debiasing factor 1/fsky.
        self.clff = healpy.sphtfunc.alm2cl(self.deltaf_alm) / self.fsky

        self.update_mocks(nmc)

    def update_mocks(self, nmc):
        """
        This method is called by the constructor in order to generate mock-based auto power spectra.
        It may also be called by an instance of the frb_overdensity class.  It updates the following
        members: self.nmc, self.clff_mocks
        """

        assert isinstance(nmc, int)

        if nmc and isinstance(self.mocks, np.ndarray):
            assert self.mocks.shape[-1] >= nmc, "frb_overdensity.update_mocks: input nmc and mocks don't match!"

        self.nmc = nmc

        if not self.nmc:
            self.clff_mocks = None
        else:
            self.clff_mocks = np.zeros((self.nmc, self.lmax+1))

            t0 = time.time()
            for i in range(self.nmc):
                if self.mocks is None:
                    mockcat = None
                elif isinstance(self.mocks, np.ndarray):
                    mockcat = self.mocks[...,i]
                else:
                    m = self.mocks[1][:,i]
                    mockcat = self.mocks[0][m,:]

                mock_deltaf_alm = self.get_mock_alm(mockcat)

                self.clff_mocks[i,:] = healpy.sphtfunc.alm2cl(mock_deltaf_alm)
                print(f'frb_overdensity.update_mocks: mock {i+1}/{self.nmc} [{time.time()-t0} sec]')

            self.clff_mocks /= self.fsky

    def _make_overdensity_map(self, ra, dec):
        """
        This helper function makes an FRB overdensity map 'delta_f' from an FRB catalog (ra,dec).
        We subtract a random catalog, rotate to galactic coordinates, and pixelize to Healpix.
        (Reminder: throughout the pipeline, Healpix maps are always in galactic coordinates).

        The (ra, dec) arguments are 1-d arrays of length N_{frb}, representing the FRB catalog.
        When called by the constructor, the true FRB locations (self.ra_deg, self.dec_deg) will
        be specified.  When called by get_mock_alm(), a mock FRB catalog will be specified.

        Returns a Healpix map representing the overdensity field delta_f.
        """

        n = len(ra)
        nf_2d = n / (self.fsky * 4 * np.pi)
        r = self.rmult

        assert ra.shape == dec.shape == (n,)
        l, b = fx.utils.convert_ra_dec_to_l_b(ra, dec)

        # The input catalog (ra,dec) has size n.  We simulate a random catalog of size (n*r),
        # by repeating each input declination 'r' times, and simulating RA's at random.
        randcat_ra = np.random.uniform(0.0, 360.0, size=n*r)
        randcat_dec = np.zeros((n,r))
        randcat_dec[:,:] = dec[:,np.newaxis]
        randcat_dec = randcat_dec.reshape((n*r,))
        lrand, brand = fx.utils.convert_ra_dec_to_l_b(randcat_ra, randcat_dec)

        # We weight each pixel count with a factor (1/Omega_pix), as appropriate when discretizing
        # a sum of delta functions (see frbx_pipelines_notes.tex).  We also include the 1/n_f^{2d}
        # factor here.
        w = 1.0 / self.pixarea / nf_2d
        m = fx.utils.make_healpix_map_from_catalog(self.nside, l, b, weight=w, interpolate=self.interpolate)

        # When the random catalog is subtracted, the weighting is 'r' times smaller, since
        # the random catalog is larger by a factor r.
        m -= fx.utils.make_healpix_map_from_catalog(self.nside, lrand, brand, weight=w/r, interpolate=self.interpolate)

        return m

    def get_mock_alm(self, mockcat):
        """
        This helper function makes a mock FRB catalog, converts it to an overdensity field delta_f,
        and returns its spherical transform a_{lm}^f.

        If mockcat is None, then we make mock catalogs by randomizing RA's in the data, leaving
        declinations unchanged.  That is, we make mocks the same way as randoms, except that the
        number of objects is different.  If mockcat is not None, then we expect RA's (1-d array)
        or (RA's, DEC's) (2-d array) to be provided as input.

        Called by clfg_analysis.___init__(), to assign error bars to C_l^{fg}.
        """

        if mockcat is None:
            # Randomize RA's (leaving dec's unchanged).
            mockcat_ra = np.random.uniform(0.0, 360.0, size=self.size)
            mockcat_dec = self.dec_deg
        else:
            if mockcat.ndim == 1:
                mockcat_ra = mockcat
                mockcat_dec = self.dec_deg
            elif mockcat.ndim == 2:
                assert mockcat.shape[-1] == 2
                mockcat_ra = mockcat[:,0]
                mockcat_dec = mockcat[:,1]
            else:
                raise RuntimeError(f'frb_overdensity.get_mock_alm: mockcat.ndim = {mockcat.ndim}.')

        mock_deltaf_map = self._make_overdensity_map(mockcat_ra, mockcat_dec)

        # Map -> a_{lm}
        return healpy.sphtfunc.map2alm(mock_deltaf_map, self.lmax, iter=self.map2alm_iter, pol=False)

    def plot_deltaf(self, doc=None, nside=None):
        m = healpy.pixelfunc.ud_grade(self.deltaf_map, nside) if (nside is not None) else self.deltaf_map
        fx.utils.show_healpix_map(m, doc)

    def plot_clff(self, doc=None, plt_args=None):
        if plt_args is None:
            plt_args = {}
        b = fx.simple_l_binning(self.lmax)

        if self.clff_mocks is not None:
            for i in range(self.nmc):
                plt.semilogx(b.l_vals, b.bin_average(self.clff_mocks[i,:]), color='beige')

        plt.semilogx(b.l_vals, b.bin_average(self.clff))
        plt.axhline(1.0/self.nf_2d, color='red', ls='--', label=r'$1/n_f$')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{ff}$')
        plt.legend(loc='upper right').draw_frame(False)
        fx.showfig(doc, plt_args)
