import numpy as np
import matplotlib.pyplot as plt
import time
import healpy
import frbx as fx


class clfg_analysis:
    """
    The 'clfg_analysis' object is constructed from a pair (frb_overdensity, galaxy_overdensity).
    It estimates C_l^{fg}, and assigns Monte Carlo error bars by simulating mocks.

    The constructor can take a long time to run!  (The mock catalog generation is done in the
    constructor.)  Therefore, it may be useful to pickle the clfg_analysis object to disk for
    postprocessing.  The pickled clfg_analysis object contains C_l^{fg} spectra (with no l-binning)
    for each mock.

    Members:
    
       self.nmc                 number of mocks
       self.seed                RNG seed
       self.lmax                max multipole l
       self.deltaf              frb_overdensity object
       self.deltag              galaxy_overdensity object
       self.clfg_data           1-d array of length (lmax+1)
       self.clfg_mocks          2-d array of shape (nmc,lmax+1)
       self.plot_clfg           (method) plots C_l^{fg} with error bars
    """

    def __init__(self, deltaf, deltag, nmc, seed=None):
        assert isinstance(deltaf, fx.frb_overdensity)
        assert isinstance(deltag, fx.galaxy_overdensity)
        assert deltaf.nside == deltag.nside
        assert deltaf.lmax == deltag.lmax
        assert nmc >= 4
        assert (seed is None) or isinstance(seed, int)

        if isinstance(deltaf.mocks, np.ndarray):
            assert deltaf.mocks.shape[-1] >= nmc, "clfg_analysis: input nmc and mocks don't match!"

        self.lmax = deltaf.lmax
        self.deltaf = deltaf
        self.deltag = deltag
        self.nmc = nmc

        self.seed = seed
        np.random.seed(self.seed)

        # Determine fsky, by intersecting the FRB and galaxy masks.
        ra, dec = fx.utils.make_healpix_ra_dec_maps(self.deltag.nside)
        mask = self.deltag.healpix_mask
        mask = np.logical_and(mask, dec >= self.deltaf.dec_min)
        mask = np.logical_and(mask, dec <= self.deltaf.dec_max)
        self.fsky = np.mean(mask)

        # We estimate C_l^{fg} from the delta fields, by pretending that the deltas are all-sky, then
        # applying the debiasing factor 1/fsky.
        self.clfg_data = healpy.sphtfunc.alm2cl(self.deltaf.deltaf_alm, deltag.deltag_alm) / self.fsky
        self.clfg_mocks = np.zeros((nmc, self.lmax+1))

        t0 = time.time()

        # Monte Carlo loop over mocks.
        for i in range(nmc):
            if self.deltaf.mocks is None:
                mockcat = None
            elif isinstance(self.deltaf.mocks, np.ndarray):
                mockcat = self.deltaf.mocks[...,i]
            else:
                m = self.deltaf.mocks[1][:,i]
                mockcat = self.deltaf.mocks[0][m,:]

            mock_deltaf_alm = self.deltaf.get_mock_alm(mockcat)

            self.clfg_mocks[i,:] = healpy.sphtfunc.alm2cl(mock_deltaf_alm, deltag.deltag_alm) / self.fsky
            print(f'clfg_analysis: mock {i+1}/{nmc} [{time.time()-t0} sec]')

    def plot_clfg(self, doc=None, dlog=0.35, xlim=None, ylim=None, plt_args=None, yscale='lin'):
        """
        The 'doc' argument should either be None (to show a plot interactively), or a
        handout.Handout instance (to show a plot in a web-browsable output directory).
        'dlog' specifies the difference in log(l) between endpoints of each bin.
        """

        if plt_args is None:
            plt_args = {}
        ell = np.arange(self.lmax+1, dtype=np.float)
        b = fx.simple_l_binning(self.lmax, dlog)

        # Note that we plot (l C_l^{fg}), not for any deep reason, but because
        # empirically it makes the prettiest plot.

        binned_clfg_data = b.bin_average(ell * self.clfg_data)
        binned_clfg_mocks = np.array([ b.bin_average(ell*cl) for cl in self.clfg_mocks ])
        binned_clfg_errorbar = np.mean(binned_clfg_mocks**2,axis=0)**0.5
        binned_clgg = b.bin_average(ell * self.deltag.clgg)

        plt.semilogx(b.l_vals, binned_clfg_data, marker='', markersize=3.2, linestyle='-', color='tomato')
        plt.errorbar(b.l_vals, binned_clfg_data, yerr=binned_clfg_errorbar, ecolor='tomato', fmt='none', lw=1)
        plt.semilogx(b.l_vals, binned_clgg, marker='', markersize=3.2, linestyle='-', color='dimgrey')

        plt.axhline(0.0, color='grey', ls='dotted')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell C_\ell^{fg}$')
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xscale('log', nonposx='clip')

        if yscale == 'symlog':
            linthreshy = 1.0e-6
            linscaley = 1.0e-5
            plt.yscale('symlog', linthreshy=linthreshy, linscaley=linscaley, subsy=[2, 4, 6, 8])
            plt.yticks([-10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(-plt.ylim()[0])), 1)]
                       + [0]
                       + [+10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(+plt.ylim()[1])), 1)])
        elif yscale != 'lin':
            raise RuntimeError('plot_clfg: invalid yscale!')

        fx.showfig(doc, plt_args)


####################################################################################################


class multi_clfg_analysis:
    """
    The 'multi_clfg_analysis' object is roughly equivalent to an ensemble of N
    clfg_analysis objects, with the same underlying frb_overdensity object, but
    different galaxy_overdensity objects.  (However, the multi_clfg_analysis pipeline
    will run a lot faster than N clfg_analysis objects.)

    Usage:

        - call multi_clfg_analysis constructor to initialize delta_f and nmc
        - call multi_clfg_analysis.add_deltag() once for each galaxy field delta_g
        - call multi_clfg_analysis.run_pipeline()

    Example code:

        import frbx_pipelines as fx

        # FRB catalog
        f_cat = fx.frb_catalog_jun23()
        deltaf = fx.frb_overdensity(f_cat, nside=1024, lmax=1000, dec_min=10.0)

        # Galaxy catalog and mask
        g_cat = fx.galaxy_catalog_2mpz()
        mask = fx.utils.make_bthresh_mask(nside, bthresh)
      
        # Construct multi_clfg_analysis object.
        clfg = fx.multi_clfg_analysis(deltaf, nmc=1000)

        # Add 20 galaxy_overdensities, corresponding to different zmax values      
        for zmax in [ 0.01*i for i in range(1,21) ]:
            g_cat_zmax = g_cat.make_zbin_subcatalog(0.0, zmax)
            deltag = fx.galaxy_overdensity(g_cat_zmax, mask, lmax=1000)
            clfg.add_deltag(deltag, name=f'zmax={zmax}')

        # Run pipeline, and pickle to disk for postprocessing
        clfg.run_pipeline()
        fx.write_pickle('clfg.pkl', clfg)

    See handouts/explore_multi_clfg.py for a runnable version of this example.

    Note that we build up the galaxy_overdensity list incrementally through calls to add_deltag().
    An alternative design would have been to specify a list of galaxy_overdensity objects in the
    constructor.  The "incremental" design was chosen to reduce memory usage, since it avoids keeping
    all of the galaxy_overdensities in memory simultaneously.  (E.g. in the example code above, each
    galaxy subcatalog and galaxy_overdensity object is destroyed in the next iteration of the loop.)
    """

    def __init__(self, deltaf, nmc, mc_start=0, mc_end=-1, seed=None):
        assert isinstance(deltaf, fx.frb_overdensity)
        assert (seed is None) or isinstance(seed, int)

        if isinstance(deltaf.mocks, np.ndarray):
            assert deltaf.mocks.shape[-1] >= nmc, "multi_clfg_analysis: input nmc and mocks don't match!"

        self.lmax = deltaf.lmax
        self.deltaf = deltaf

        self.nmc = nmc
        self.mc_start = mc_start
        self.mc_end = mc_end

        if self.mc_start == 0 and self.mc_end == -1:
            self._nmc = self.nmc
        else:
            self._nmc = self.mc_end - self.mc_start + 1

        assert 4 <= self._nmc <= self.nmc

        self.seed = seed
        np.random.seed(self.seed)

        self.ng = 0
    
        self.deltag_alm_list = [ ]   # list of length ng
        self.clgg_list = [ ]         # list of length ng
        self.ng_2d_list = [ ]        # list of length ng
        self.fsky_list = [ ]         # list of length ng
        self.name_list = [ ]         # list of length ng
        self.finalized = False       # will be set to True when run_pipeline() is called

    def add_deltag(self, deltag, name):
        """
        For documentation on add_deltag(), see the multi_clfg_analysis class docstring.

        The 'deltag' argument should be a galaxy_overdensity object.
        The 'name' argument is a descriptive string used to identify the galaxy_overdensity in plots.
        """
        
        assert not self.finalized, "multi_clfg_analysis: add_deltag() called after run_pipeline()"
        assert isinstance(deltag, fx.galaxy_overdensity)
        assert isinstance(name, str)
        assert deltag.nside == self.deltaf.nside
        assert deltag.lmax == self.deltaf.lmax

        # Determine fsky, by intersecting the FRB and galaxy masks.
        ra, dec = fx.utils.make_healpix_ra_dec_maps(deltag.nside)
        mask = deltag.healpix_mask
        mask = np.logical_and(mask, dec >= self.deltaf.dec_min)
        mask = np.logical_and(mask, dec <= self.deltaf.dec_max)
        fsky = np.mean(mask)
        
        self.deltag_alm_list.append(deltag.deltag_alm)
        self.clgg_list.append(deltag.clgg)
        self.ng_2d_list.append(deltag.ng_2d)
        self.fsky_list.append(fsky)
        self.name_list.append(name)
        self.ng += 1

        print(f'multi_clfg_analysis: added {name}, ng={self.ng}')

    def _compute_clfg(self, deltaf_alm):
        """Helper method called by run_pipeline().  Returns 2-d array with shape (ng,lmax+1)"""
        
        # We estimate C_l^{fg} from the delta fields, by pretending that the deltas are all-sky, then
        # applying the debiasing factor 1/fsky.

        clfg = np.zeros((self.ng, self.lmax+1))

        for i in range(self.ng):
            fsky = self.fsky_list[i]
            deltag_alm = self.deltag_alm_list[i]
            clfg[i,:] = healpy.sphtfunc.alm2cl(deltaf_alm, deltag_alm) / fsky

        return clfg

    def run_pipeline(self):
        """For documentation on run_pipeline(), see the multi_clfg_analysis class docstring."""
        
        assert self.ng > 0, "multi_clfg_analysis: must call add_deltag() before calling run_pipeline()"
        assert not self.finalized, "multi_clfg_analysis: double call to run_pipeline()"

        # The following attributes are defined here, which is outside the constructor __init__() above.
        self.clfg_data = self._compute_clfg(self.deltaf.deltaf_alm)
        self.clfg_mocks = np.zeros((self.nmc, self.ng, self.lmax+1))

        t0 = time.time()

        # Monte Carlo loop over mocks.
        for i in range(self._nmc):
            j = self.mc_start + i

            if self.deltaf.mocks is None:
                mockcat = None
            elif isinstance(self.deltaf.mocks, np.ndarray):
                mockcat = self.deltaf.mocks[...,j]
            else:
                m = self.deltaf.mocks[1][:,j]
                mockcat = self.deltaf.mocks[0][m,:]

            deltaf_alm = self.deltaf.get_mock_alm(mockcat)

            self.clfg_mocks[j,:,:] = self._compute_clfg(deltaf_alm)
            print(f'multi_clfg_analysis: mock {j+1}/{self.nmc} [{time.time()-t0} sec]')

        self.finalized = True

    def plot_clfg(self, doc=None, dlog=0.35, xlim=None, ylim=None, plt_args=None, yscale='lin'):
        """
        The 'doc' argument should either be None (to show plots interactively), or a
        handout.Handout instance (to show plots in a web-browsable output directory).
        'dlog' specifies the difference in log(l) between endpoints of each bin.
        """

        if plt_args is None:
            plt_args = {}
        assert self.finalized, "multi_clfg_analysis: must call run_pipeline() before calling plot_clfg()"
        
        ell = np.arange(self.lmax+1, dtype=np.float)
        b = fx.simple_l_binning(self.lmax, dlog)

        # Note that we plot (l C_l^{fg}), not for any deep reason, but because
        # empirically it makes the prettiest plot.

        for ig in range(self.ng):
            binned_clfg_data = b.bin_average(ell * self.clfg_data[ig,:])
            binned_clfg_mocks = np.array([ b.bin_average(ell*cl) for cl in self.clfg_mocks[:,ig,:] ])
            binned_clfg_errorbar = np.mean(binned_clfg_mocks**2,axis=0)**0.5
            binned_clgg = b.bin_average(ell * self.clgg_list[ig][:])
            binned_clgg_p = b.bin_average(ell / self.ng_2d_list[ig])

            plt.semilogx(b.l_vals, binned_clfg_data, marker='', markersize=3.2, linestyle='-', color='tomato')
            plt.errorbar(b.l_vals, binned_clfg_data, yerr=binned_clfg_errorbar, ecolor='tomato', fmt='none', lw=1)
            plt.semilogx(b.l_vals, binned_clgg, marker='', markersize=3.2, linestyle='-', color='dimgrey')
            plt.semilogx(b.l_vals, binned_clgg_p, marker='', markersize=2.2, linestyle=':', color='dimgrey')

            plt.axhline(0.0, color='grey', ls='dotted')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\ell C_\ell^{fg}$')
            plt.ylim(ylim)
            plt.xlim(xlim)
            plt.xscale('log', nonposx='clip')

            if yscale == 'symlog':
                linthreshy = 1.0e-6
                linscaley = 1.0e-5
                plt.yscale('symlog', linthreshy=linthreshy, linscaley=linscaley, subsy=[2, 4, 6, 8])
                plt.yticks([-10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(-plt.ylim()[0])), 1)]
                           + [0]
                           + [+10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(+plt.ylim()[1])), 1)])
            elif yscale != 'lin':
                raise RuntimeError('plot_clfg: invalid yscale!')

            plt.title(self.name_list[ig])
            fx.showfig(doc, plt_args)


####################################################################################################


class clgg_analysis:
    """
    The 'clgg_analysis' object is constructed from a (galaxy_overdensity, galaxy_overdensity) pair.

    Members:

        self.lmax                max multipole l
        self.deltag1             first galaxy_overdensity object
        self.deltag2             second galaxy_overdensity object
        self.names               list of names for galaxy_overdensity objects
        self.mask                combined mask
        self.fsky                fraction of the sky covered with both objects
        self.clgg                angular cross power spectrum C_l^{gg'}
        self._bin_cl             (helper method) computes auto/cross-spectrum bandpowers
        self.plot_cl             (method) plots C_l^{gg'}
        self.plot_rl             (method) plots r_l = C_l^{gg'} / (C_l^{gg} * C_l^{g'g'})^0.5
    """

    def __init__(self, deltag1, deltag2, names):
        assert isinstance(deltag1, fx.galaxy_overdensity)
        assert isinstance(deltag2, fx.galaxy_overdensity)
        assert isinstance(names, list) and len(names) == 2
        assert deltag1.nside == deltag2.nside
        assert deltag1.lmax == deltag2.lmax

        self.lmax = deltag1.lmax
        self.deltag1 = deltag1
        self.deltag2 = deltag2

        self.names = names

        self.mask = self.deltag1.healpix_mask * self.deltag2.healpix_mask
        self.fsky = np.mean(self.mask)

        # We estimate C_l^{gg'} from the delta fields, by pretending that the deltas are all-sky, then
        # applying the debiasing factor 1/fsky.
        self.clgg = healpy.sphtfunc.alm2cl(self.deltag1.deltag_alm, self.deltag2.deltag_alm) / self.fsky

    def _bin_cl(self, dlog):
        """Returns binned cross and auto power spectra."""

        ell = np.arange(self.lmax+1, dtype=np.float)
        b = fx.simple_l_binning(self.lmax, dlog)

        clgg1 = b.bin_average(ell * self.deltag1.clgg[:])
        clgg_p1 = b.bin_average(ell / self.deltag1.ng_2d)

        clgg2 = b.bin_average(ell * self.deltag2.clgg[:])
        clgg_p2 = b.bin_average(ell / self.deltag2.ng_2d)

        clgg = b.bin_average(ell * self.clgg[:])

        return b.l_vals, clgg1, clgg_p1, clgg2, clgg_p2, clgg

    def plot_cl(self, doc=None, dlog=0.35, xlim=None, ylim=None, plt_args=None, yscale='lin'):
        """
        The 'doc' argument should either be None (to show plots interactively), or a
        handout.Handout instance (to show plots in a web-browsable output directory).
        'dlog' specifies the difference in log(l) between endpoints of each bin.
        """

        if plt_args is None:
            plt_args = {}

        l_vals, clgg1, clgg_p1, clgg2, clgg_p2, clgg = self._bin_cl(dlog)

        plt.semilogx(l_vals, clgg1, marker='', markersize=3.2, linestyle='-', color='blue', label=self.names[0] + r' $(gg)$')
        plt.semilogx(l_vals, clgg_p1, marker='', markersize=2.2, linestyle=':', color='blue', label=self.names[0] + r' $(gg,p)$')

        plt.semilogx(l_vals, clgg2, marker='', markersize=3.2, linestyle='-', color='red', label=self.names[1] + r' $(gg)$')
        plt.semilogx(l_vals, clgg_p2, marker='', markersize=2.2, linestyle=':', color='red', label=self.names[1] + r' $(gg,p)$')

        plt.semilogx(l_vals, clgg, marker='', markersize=3.2, linestyle='-', color='k', label=r"$gg'$")

        plt.axhline(0.0, color='grey', ls='dotted')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell C_\ell$')
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xscale('log', nonposx='clip')
        plt.legend(loc='upper left')

        if yscale == 'symlog':
            linthreshy = 1.0e-6
            linscaley = 1.0e-5
            plt.yscale('symlog', linthreshy=linthreshy, linscaley=linscaley, subsy=[2, 4, 6, 8])
            plt.yticks([-10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(-plt.ylim()[0])), 1)]
                       + [0]
                       + [+10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(+plt.ylim()[1])), 1)])
        elif yscale != 'lin':
            raise RuntimeError('plot_clgg: invalid yscale!')

        fx.showfig(doc, plt_args)

    def plot_rl(self, doc=None, dlog=0.35, xlim=None, ylim=None, plt_args=None, yscale='lin'):
        """
        The 'doc' argument should either be None (to show plots interactively), or a
        handout.Handout instance (to show plots in a web-browsable output directory).
        'dlog' specifies the difference in log(l) between endpoints of each bin.
        """

        if plt_args is None:
            plt_args = {}

        l_vals, clgg1, clgg_p1, clgg2, clgg_p2, clgg = self._bin_cl(dlog)

        r = clgg / (clgg1 * clgg2)**0.5

        plt.semilogx(l_vals, r, marker='', markersize=3.2, linestyle='-', color='red')

        plt.axhline(0.0, color='grey', ls='dotted')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$r_\ell$')
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xscale('log', nonposx='clip')

        if yscale == 'symlog':
            linthreshy = 1.0e-6
            linscaley = 1.0e-5
            plt.yscale('symlog', linthreshy=linthreshy, linscaley=linscaley, subsy=[2, 4, 6, 8])
            plt.yticks([-10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(-plt.ylim()[0])), 1)]
                       + [0]
                       + [+10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(+plt.ylim()[1])), 1)])
        elif yscale != 'lin':
            raise RuntimeError('plot_clgg: invalid yscale!')

        fx.showfig(doc, plt_args)
