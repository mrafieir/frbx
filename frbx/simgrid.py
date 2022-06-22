import os
import numpy as np
import numpy.random as rand
import astropy.units as u
from numpy.fft import fftshift
import pyfftw
import multiprocessing as mp
from h5py import File as FileH5
import matplotlib.pyplot as plt
import frbx as fx
from frbx.configs import sky_sr


class simgrid:
    """
    Contains methods for mapping catalogs of sky positions onto flat grids.

    Members:

        self.ftc: (obj) instance of fx.cosmology, which is recommended to be pickled in advance.
        self.config: (obj) points to self.ftc.config, which is an instance of fx.configs.
        self.nz: (int) number of redshift shells.
        self.nd: (int) number of DM bins.
        self.zz: (list) edges of redshift shells.
        self.dd: (list) edges of DM bins, set by self.survey_frb.
        self.dd_z: (list) redshift at self.dd.
        self.nCPU: (int) static attr specifying the total number of available CPU cores for multi-processing.
        self.run_grid: (method) maps catalogs onto grids, generates FFT arrays, computes and writes all statistics.
        simgrid.read_h5_catalog: (static method) reads and selects sources from a catalog.
        simgrid.add_gaussian_noise: (static method) adds Gaussian noise to FRB positions on the sky.
        simgrid.nonzero_overlap: (static method) looks for non-zero overlaps between two maps.
        self._catalog_to_grid: (helper method) maps a catalog onto a real grid.
        self._fft: (helper method) computes the FFT of a real grid.
        simgrid.power_spectrum: (static method) computes the cross-power for two Fourier grids.
        self.power_spectrum_to_cl: (method) takes an FFT grid (real parts), returns 1-d angular power spectrum.
        simgrid.radial_bincount: (static method) radially bins a 2-d grid, returns a 1-d histogram.
        self._simgrid__test_*.
    """

    def __init__(self, ftc, nz, nd):
        """
        Constructor arguments:

            ftc: (obj) instance of fx.cosmology.
            nz: (int) number of redshift shells.
            nd: (int) number of DM bins.
        """

        assert isinstance(ftc, fx.cosmology)
        assert isinstance(nz, int) and (1 <= nz)
        assert isinstance(nd, int) and (1 <= nd)

        self.ftc = ftc
        assert hasattr(self.ftc, 'config')
        self.config = self.ftc.config

        self.nz = nz
        self.nd = nd

        self.zz, self.dd, self.dd_z = self.ftc.bin(nz, nd)

        self.nCPU = mp.cpu_count()

    def run_grid(self, i_path, s_path, grid_shape, nsim, noise_frb=None, normalize=True, nl=None, up=1,
                 save_ff=True, save_gg=True, verbose=False):
        """
        This method reads saved catalogs, distributes sources in CIC-weighted grids, and finally computes
        angular power spectra.  Results are saved to disk.

        Args:

            i_path: (str) path, ending with '.h5', for reading catalogs.
            s_path: (str) path, ending with '.h5', for writing results to disk.
            grid_shape: (odd int) specifying the shape of a square grid.
            nsim: (int) total number of simulation runs.
            noise_frb: (Quantity) FWHM of Gaussian noise which will be added to the FRB catalogs.
            normalize: (bool) whether to normalize the grid by its sum.
            nl: (int) if not None, it specifies the number of l's for downsampling power spectra.
            up: (int) odd multiplicative factor for upsampling, hence increasing the
                resolution of the grid prior to radial binning, along the two axes.
            save_ff: (bool) whether to compute and save the angular cross-power (between DM bins) for FRBs.
            save_gg: (bool) whether to compute and save the angular auto-power (in redshift shells) for galaxies.
            verbose: (bool) enables prints for debugging catalogs.
        """

        assert isinstance(i_path, str) and isinstance(s_path, str)
        assert i_path.endswith('.h5') and s_path.endswith('.h5')
        assert (grid_shape == int(grid_shape)) and (grid_shape % 2 == 1), 'grid_shape must be an odd integer'
        assert (nsim == int(nsim)) and (nsim % 2 == 0) and (nsim > 0), 'nsim must be an even integer'
        assert isinstance(normalize, bool)
        if nl is not None:
            assert (nl == int(nl)) and (nl > 1)
        assert (up == int(up)) and (up > 0) and (up % 2 == 1)
        assert isinstance(save_ff, bool)
        assert isinstance(save_gg, bool)
        assert isinstance(verbose, bool)

        if noise_frb is not None:
            noise_frb = noise_frb.to(self.config.sim.unit).value

        h5_mode = 'w'

        _exf = np.zeros((len(self.ftc.frb_par), self.nd, 2))
        _exg = np.zeros(self.nz)

        for q in range(nsim):
            print(f'Iter {q} of {nsim-1}')

            map_ff_super = np.zeros((len(self.ftc.frb_par), self.nd, 2, grid_shape, grid_shape), dtype=np.complex128)

            for (qqq, frb_par) in enumerate(self.ftc.frb_par):
                for iD in range(self.nd):

                    if (self.nd > 1) or self.config.sim.force_bin_d:
                        (mask_col, s_min, s_max) = (3, self.dd[0][iD], self.dd[1][iD])
                    else:
                        zmax = self.ftc.interp_zmax - self.config.zpad_interp_zmax + self.config.zpad_eps
                        (mask_col, s_min, s_max) = (2, 0.0, zmax)

                    cat_f0 = simgrid.read_h5_catalog(
                           i_path, f'frb_{qqq}_{nsim-q-1}', mask_col, s_min, s_max, verbose=verbose)

                    cat_f1 = simgrid.read_h5_catalog(
                           i_path, f'frb_{qqq}_{q}', mask_col, s_min, s_max, verbose=verbose)

                    _exf[qqq, iD, 0] += cat_f0.shape[0]
                    _exf[qqq, iD, 1] += cat_f1.shape[0]

                    if noise_frb is not None:
                        if cat_f0.size:
                            simgrid.add_gaussian_noise(cat_f0, noise_frb)
                        if cat_f1.size:
                            simgrid.add_gaussian_noise(cat_f1, noise_frb)

                    map_ff_super[qqq, iD, 0, :, :] = self._fft(self._catalog_to_grid(grid_shape, cat_f0))
                    map_ff_super[qqq, iD, 1, :, :] = self._fft(self._catalog_to_grid(grid_shape, cat_f1))

            if save_ff:
                for (qqq, frb_par) in enumerate(self.ftc.frb_par):
                    for iD in range(self.nd):
                        for iDx in range(self.nd):
                            map_f1 = map_ff_super[qqq, iD, 1, :, :]
                            map_fy = map_ff_super[qqq, iDx, 1, :, :]

                            ffy = self.power_spectrum_to_cl(simgrid.power_spectrum(map_f1, map_fy), nl, up)

                            o = FileH5(s_path, mode=h5_mode)
                            ret = o.require_dataset(
                                'ff', shape=(nsim, len(self.ftc.frb_par), self.nd, self.nd, ffy['ell'].size),
                                dtype=np.float64, exact=True)

                            ret[q, qqq, iD, iDx, :] = ffy['cl'][:]

                            if h5_mode == 'w':
                                ret.attrs['axes'] = ['simulation_i', 'frb_par_i', 'dm_i', 'dm_j', 'cl']
                                ret.attrs['ell'] = ffy['ell']
                            o.close()
                            h5_mode = 'a'

            h5w_z = (q == 0)
            for iz in range(self.nz):
                mask_col = 2 if (self.zz[0].size > 1) else 0

                cat_g = simgrid.read_h5_catalog(
                      i_path, f'galaxy_{q}', mask_col, self.zz[0][iz], self.zz[1][iz], verbose=verbose)

                _exg[iz] += cat_g.shape[0]

                map_g = self._fft(self._catalog_to_grid(grid_shape, cat_g))

                if save_gg:
                    gg = self.power_spectrum_to_cl(simgrid.power_spectrum(map_g, map_g), nl, up)

                    o = FileH5(s_path, mode=h5_mode)
                    ret = o.require_dataset('gg', shape=(nsim, self.nz, gg['ell'].size), dtype=np.float64, exact=True)

                    ret[q, iz, :] = gg['cl'][:]

                    if h5w_z:
                        ret.attrs['axes'] = ['simulation_i', 'redshift_i', 'gg']
                        ret.attrs['ell'] = gg['ell']
                    o.close()
                    h5_mode = 'a'

                h5w_fg = (q == iz == 0)
                for (qqq, frb_par) in enumerate(self.ftc.frb_par):
                    for iD in range(self.nd):
                        map_f0 = map_ff_super[qqq, iD, 0, :, :]
                        map_f1 = map_ff_super[qqq, iD, 1, :, :]

                        fg0 = self.power_spectrum_to_cl(simgrid.power_spectrum(map_f0, map_g), nl, up)
                        fg1 = self.power_spectrum_to_cl(simgrid.power_spectrum(map_f1, map_g), nl, up)

                        o = FileH5(s_path, mode=h5_mode)
                        ret = o.require_dataset(
                            'fg', shape=(nsim, len(self.ftc.frb_par), self.nz, self.nd, 2, fg0['ell'].size),
                            dtype=np.float64, exact=True)

                        ret[q, qqq, iz, iD, 0, :] = fg0['cl'][:]
                        ret[q, qqq, iz, iD, 1, :] = fg1['cl'][:]

                        if h5w_fg:
                            ret.attrs['axes'] = ['simulation_i', 'frb_par_i', 'redshift_i', 'dm_i', '[fg0,fg1]', 'cl']
                            ret.attrs['ell'] = fg0['ell']
                        o.close()
                        h5_mode = 'a'
                        h5w_fg = False
                h5w_z = False

        print(f'{s_path} saved!')
        print(f'<Nf> = {_exf/nsim/self.config.f_sky/sky_sr}')
        print(f'<Ng> = {_exg/nsim/self.config.f_sky/sky_sr}')
        print('run_grid done!\n')

    @staticmethod
    def read_h5_catalog(i_path, name, mask_col, s_min, s_max, dim=2, verbose=False):
        """
        This static method reads and selects sources from a catalog.

        Args:

            i_path: (str) path, ending with '.h5', for reading catalogs.
            name: (str) name of dataset containing the catalog.
            mask_col: (int) column number used for masking and selecting sources.
            s_min: (float) the min value of the selection criterion.
            s_max: (float) the max value of the selection criterion.
            dim: (int) number of dimensions to be returned.
            verbose: (bool) enables prints for debugging catalogs, only if mask_col > 0.

        Returns:

            array of sources from the catalog.
        """

        assert isinstance(i_path, str)
        assert i_path.endswith('.h5')
        assert os.path.exists(i_path)
        assert isinstance(name, str)
        assert (mask_col == int(mask_col)) and (mask_col >= 0)
        assert isinstance(s_min, float) and isinstance(s_max, float)
        assert (dim == int(dim)) and (dim > 0)
        assert isinstance(verbose, bool)

        try:
            cat = fx.read_h5(i_path, name)

            assert mask_col <= max(cat.shape)

            if verbose and mask_col:
                _pc = cat[:, mask_col]
                print("\n120*'='\n")
                print(f'cat.shape -> {cat.shape}')
                print(f'mask_col = {mask_col} -> {_pc[:4]}')
                print(f'(min,max) = ({np.min(_pc)},{np.max(_pc)})')
                print(f'(frac of cat[:,mask_col] <= 0.0) = {float(np.sum(_pc <= 0.0))/_pc.size}')

            if mask_col:
                mask = np.logical_and((s_min <= cat[:,mask_col]), (cat[:,mask_col] < s_max))

                if verbose:
                    print(f"\nsimgrid.read_h5_catalog: masking out {s_min} > (-) > {s_max} {'>'*20}")
                    print(cat[np.logical_not(mask)][:10])
                    print('<'*20)

                cat = cat[mask][:,:dim]
            else:
                cat = cat[:,:dim]
        except IOError:
            cat = np.asarray([])

        if verbose and mask_col:
            v = cat[0] if cat.size else 0
            print(f'\ncat[0]={v}\n')

        return cat

    @staticmethod
    def add_gaussian_noise(cat, fwhm):
        """
        This static method adds (in place) Gaussian noise (mean=0) to a catalog of sky positions.

        Args:

            cat: (array) of (N,2) sky positions for N sources in self.config.sim.unit.
            fwhm: (float) FWHM of Gaussian noise profile in self.config.sim.unit.

        Returns:

            array of shape (N,2) containing new positions.
        """

        assert isinstance(cat, np.ndarray)
        assert isinstance(fwhm, float) and (fwhm > 0.0)

        n, dim = cat.shape
        assert dim == 2     # 2-d sky.

        # fwhm -> sigma.
        sigma = fwhm / np.sqrt(8.0*np.log(2))

        # Assuming cat and sigma have the same unit.
        cat += np.random.normal(0.0, sigma, (n,dim))

    @staticmethod
    def nonzero_overlap(map1, map2):
        """
        This static method computes the sum of non-zero overlaps between two maps.

        Args:

            map1: (array) first map.
            map2: (array) second map.
        
        Returns:

            two floats for map1 and map2, respectively.
        """

        assert isinstance(map1, np.ndarray) and isinstance(map2, np.ndarray)
        assert map1.shape == map2.shape != (0,0)

        mask = np.logical_and((map1 > 0.0), (map2 > 0.0))

        ov1 = np.sum(map1[mask])
        ov2 = np.sum(map2[mask])

        return ov1, ov2

    def _catalog_to_grid(self, grid_shape, catalog, normalize=True):
        """
        This method maps a catalog of 2-d positions onto a CIC-weighted grid with periodic boundary conditions.
        
        Args:

            grid_shape: (int) specifies the shape of a square grid.
            catalog: (array) of (n,2) sky positions. The unit is implicit.
            normalize: (bool) whether to divide the CIC-weighted grid by its sum; a convention in Fourier analysis.

        Returns:

            2-d array containing all sources in the catalog.
        """

        assert grid_shape == int(grid_shape)
        assert isinstance(normalize, bool)

        len_bin = self.config.xymax_cov.value / grid_shape
        assert len_bin > 0

        grid_shape = (grid_shape, grid_shape)
        catalog = np.asarray(catalog)

        if not catalog.size:
            return np.zeros(grid_shape)

        ij = np.floor_divide(catalog, len_bin).astype(int)
        xy = np.remainder(catalog, len_bin)

        (x, y) = (xy[:,0], xy[:,1])

        raw_index = [(ij[:,0], ij[:,1]),
                     (ij[:,0], ij[:,1]+1),
                     (ij[:,0]+1, ij[:,1]),
                     (ij[:,0]+1, ij[:,1]+1)]

        index = []
        for rx in raw_index:
            index.append(np.ravel_multi_index(rx, grid_shape, mode='wrap'))

        index = np.asarray(index).reshape(len(raw_index), ij.shape[0])

        grid_size = int(np.prod(grid_shape))

        ret = np.bincount(index[0,:], (1.-x)*(1.-y), grid_size)
        ret += np.bincount(index[1,:], (1.-x)*y, grid_size)
        ret += np.bincount(index[2,:], x*(1.-y), grid_size)
        ret += np.bincount(index[3,:], x*y, grid_size)

        ret = ret.reshape(grid_shape)

        s = float(np.sum(ret))
        assert len(catalog) == int(round(s)), '_catalog_to_grid: the total number of sources must be conserved!'

        if normalize and (s >= 1.0):
            ret /= s
                 
        return ret

    def _fft(self, grid, remove_dc=True):
        """
        This helper method computes the fast Fourier transform of a 2-d grid.  We note that numpy FFT operations are
        unscaled in the forward direction.

        Args:

            grid: (2-d array) of np.float64.
            remove_dc: (bool) whether to subtract the mean value (in place) prior to FFT.

        Returns:

            2-d array of complex numbers.
        """

        assert isinstance(grid, np.ndarray)
        assert isinstance(remove_dc, bool)

        if remove_dc:
            grid -= np.mean(grid)

        return pyfftw.interfaces.numpy_fft.fft2(grid, threads=self.nCPU)

    @staticmethod
    def power_spectrum(grid1, grid2):
        """
        This static method computes the cross-power for two Fourier grids.

        Args:

            grid1: (2-d array) of complex numbers.
            grid2: (2-d array) of complex numbers.

        Returns:

            2-d array of floats, shifted to the center.
        """

        assert isinstance(grid1, np.ndarray) and isinstance(grid2, np.ndarray)
        assert np.all(grid1.shape == grid2.shape)

        ret = fftshift(np.conj(grid1) * grid2)

        # Verifying whether the imaginary part is zero (within roundoff error).
        err = np.mean(np.imag(ret))
        if err > 1e-13:
            raise RuntimeError(f'power_spectrum: err = {err}')

        return np.real(ret)

    def power_spectrum_to_cl(self, arr, nl=None, up=1, l_min=None, l_max=None):
        """
        Given a 2-d power spectrum in the frequency space, returns the angular power spectrum.  The current version
        adopts a linear downsampling.

        Args:

            arr: (2-d array) of complex floats; the Fourier grid.
            nl: (int) if not None, specifies the number of l's for binning and downsampling.
            up: (int) odd multiplicative factor for upsampling, hence increasing the
                resolution of the input arr prior to radial binning, along the two axes.
            l_min: (scalar) minimum 'ell' for the output. If None, the lowest possible value is used.
            l_max: (scalar) maximum 'ell' for the output. If None, the highest possible value is used.

        Returns:

            dict containing two entries: {'ell': 1-d array of floats, 'cl': 1-d array of floats}.
        """

        assert isinstance(arr, np.ndarray)
        assert (isinstance(nl, int) and (nl > 1)) or (nl is None)
        assert (up == int(up)) and (up > 0) and (up % 2 == 1)
        assert isinstance(l_min, (float, int)) or (l_min is None)
        assert isinstance(l_max, (float, int)) or (l_max is None)

        _xymax = self.config.xymax_cov.to(u.rad).value

        box_size = arr.shape[0]
        len_bin = _xymax / box_size

        assert arr.shape[1] == box_size, 'power_spectrum_to_cl expects a square array!'
        assert len_bin > 0

        _l_min = 2*np.pi / _xymax
        _l_max = (np.pi/len_bin) if l_max is None else l_max

        r, cl = simgrid.radial_bincount(arr, up=up)
        ell = _l_min * r

        if (up > 1) and (nl is None):
            nl = 1

        if nl is not None:
            ell, cl = fx.downsample_cl(ell, cl, nl * up)

        mask = (ell <= _l_max)
        if l_min is not None:
            mask = np.logical_and(mask, (l_min <= ell))

        ell, cl = ell[mask], cl[mask]
        cl *= (4 * np.pi * self.config.f_sky)

        return {'ell': ell, 'cl': cl}

    @staticmethod
    def radial_bincount(arr, cxy=None, up=1):
        """
        This static method computes the mean value within concentric circular annuli over a 2-d array.
        
        Args:

            arr: (2-d array) must have odd number of elements along each axis.
            cxy: (list or tuple) contains two non-negative scalars which specify the
                 center of radial bins.  If None, then the center of array is used instead.
            up: (int) odd multiplicative factor for upsampling, hence increasing the
                resolution of the input arr prior to radial binning, along the two axes.

        Returns:

            rx: 1-d array of radial bin locations with respect to the center.
            ret: 1-d array of mean values at rx[:].
        """

        assert isinstance(arr, np.ndarray)

        ay, ax = arr.shape
        assert ay % 2 == 1
        assert ax % 2 == 1

        assert (up == int(up)) and (up > 0) and (up % 2 == 1)

        if (up > 1) and ((ax != ay) or (cxy is not None)):
            raise RuntimeError('simgrid.radial_bincount does not upsample shifted annuli or asymmetric arrays!')

        if cxy is not None:
            assert isinstance(cxy, (list, tuple))
            assert (0.0 <= cxy[0] <= (ax-1)) and (0.0 <= cxy[1] <= (ay-1))
        else:
            if up > 1:
                arr = fx.utils.upsample(arr, ay * up, ax * up)
                cxy = np.asarray(arr.shape)[::-1] / 2       # Centered with upsampling.
            else:
                cxy = np.asarray((ax,ay)) / 2               # Centered without upsampling.

        y, x = np.indices(arr.shape, dtype=np.float64)

        r = np.sqrt((x-cxy[0])**2 + (y-cxy[1])**2)
        r = np.round(r).astype(int).ravel()
        rx = np.arange(np.max(r) + 1, dtype=np.float64)

        # Scaling the radial coordinates back to the original scheme.
        if up > 1:
            rx /= float(up)

        n = np.bincount(r)
        mask = n > 0.0

        ret = np.asarray(np.bincount(r, arr.ravel())[mask] / n[mask])
        rx = np.asarray(rx[mask])

        assert np.all(rx.shape == ret.shape)

        return rx, ret

    def __test_all(self):
        self.__test_read_h5_catalog()
        self.__test_add_gaussian_noise()
        self.__test_nonzero_overlap()
        self.__test__catalog_to_power_spectrum()
        self.__test_radial_bincount()

    def __test_read_h5_catalog(self):
        # Total number of catalog entries.
        n = 100

        # The first 5 cols will be populated with data and the last 5 cols will be used for masking.
        toy_cat_shape = (n,10)

        # Picking a random col for masking.
        mask_col = rand.randint(5, 10)

        # Init a toy catalog.
        toy_cat = np.zeros(toy_cat_shape)

        # Random numbers, normally distributed with (mu,sigma) = (0,1).
        toy_cat[:,mask_col] = rand.normal(size=n)

        for i in range(5):
            toy_cat[:,i] = float(i)

        # Fraction of entries within 1 sigma, adopting an open (<,<) convention.
        frac_1sigma = np.sum(np.abs(toy_cat[:,mask_col]) < 1.0) / float(n)

        # Writing the catalog to disk.
        dir_path = fx.utils.data_path('archive', envar='FRBXDATA')
        file_path = os.path.join(dir_path, 'test-simgrid_toy_catalog.h5')
        name = 'toy_cat'

        temp_file = FileH5(file_path, 'w')
        temp_dat = temp_file.require_dataset(name, shape=toy_cat_shape, dtype=np.float64)
        temp_dat[:] = toy_cat[:]
        temp_file.close()

        # r_cat adopts a half-open (<=,<) convention.
        r_cat = self.read_h5_catalog(file_path, name, mask_col, -1.0, 1.0, 5)

        for i in range(1, 5, 1):
            c = np.sum(r_cat[:,i]) / i
            c /= n
            assert c == frac_1sigma

        os.remove(file_path)

    def __test_add_gaussian_noise(self, niter=10, n=100000):
        for i in range(1, niter, 1):
            # Init a new cat for each iter since the following function operates in place.
            cat = np.zeros((n,2))

            fwhm = float(i)
            self.add_gaussian_noise(cat, fwhm)

            assert np.abs(cat.std()*np.sqrt(8.0*np.log(2)) - fwhm) < (1.0e-2 * fwhm)

    def __test_nonzero_overlap(self, niter=1000):
        for i in range(1, niter, 1):
            ones = np.ones(rand.randint(1000, size=(2,)))
            ret11 = self.nonzero_overlap(ones, ones)

            assert float(ones.size) == ret11[0] == ret11[1]

            zeros = np.zeros(rand.randint(1000, size=(2,)))
            ret00 = self.nonzero_overlap(zeros, zeros)
            ret01 = self.nonzero_overlap(np.where(ones, 0, 1), ones)

            assert (0.0, 0.0) == ret00 == ret01

            r = rand.ranf()
            partial_ones = np.random.choice([0,1], size=ones.shape, p=[1.0-r,r])

            ret1p = self.nonzero_overlap(partial_ones, ones)[0]
            assert np.sum(partial_ones) == ret1p

            if partial_ones.size:
                assert (r-0.2) <= (float(ret1p)/partial_ones.size) <= (r+0.2)

    def __test__catalog_to_power_spectrum(self, niter=10, n=1000):
        def test_wrap(n_sources, grid_shape, remove_dc, tolerance):
            """
            In this test, we generate a random catalog of uniformly distributed sources.  We map all positions
            onto CIC-weighted grids of various sizes whose bins are dx-by-dy.  Then, we compute the auto-power
            while shifting all sources by n*dx and/or n*dy, where n is in [1, 2, 3, ..., 10].  We find that the
            shift in a source position does not have any effects on the computed power.  This test verifies
            several key steps in our pipeline, including the CIC-weighted griding scheme with periodic boundary
            conditions.

            Args:

                n_sources: (int) total number of sources in the catalog.
                grid_shape: (int) side of a square grid.
                remove_dc: (bool) whether to remove the mean prior to FFT.
                tolerance: (float) error in the assertion.
            """

            assert (n_sources == int(n_sources))
            assert (grid_shape == int(grid_shape)) and (grid_shape % 2 == 1)
            assert isinstance(remove_dc, bool)
            assert tolerance > 0

            ref_cat = rand.uniform(0, self.config.xymax_cov.value, size=(n_sources,2))

            # Map the reference catalog onto a grid.
            ref_g = self._catalog_to_grid(grid_shape, ref_cat)

            # Compute the power spectrum.
            _ref_g = self._fft(ref_g, remove_dc)
            ref_spec = self.power_spectrum(_ref_g, _ref_g)

            for ix in range(10):
                for iy in range(10):
                    len_bin = self.config.xymax_cov.value / grid_shape

                    # Shift in real space.
                    shift = len_bin * np.asarray([ix, iy], dtype=np.float64)
                    shifted_cat = ref_cat + shift

                    # Map the shifted catalog onto another grid.
                    g = self._catalog_to_grid(grid_shape, shifted_cat)

                    # Compute the power spectrum for the shifted grid.
                    _g = self._fft(g, remove_dc)
                    spec = self.power_spectrum(_g, _g)

                    # Assertion block.
                    diff = np.abs(ref_spec - spec)
                    mask = (diff < tolerance)

                    if np.all(mask):
                        pass
                    else:
                        mask = np.logical_not(mask)
                        print(f'\ngrid_shape : {grid_shape}')
                        print(f'Values : {diff[mask]}')
                        print(f'mask index : {np.where(mask)}\n')

        for i in range(niter):
            print(f'test_wrap: iter {i}/{niter-1}')

            r = rand.randint(100, 1000, 1)[0]
            if r % 2 == 0:
                r += 1

            test_wrap(n, grid_shape=r, remove_dc=True, tolerance=2e-7)

    def __test_radial_bincount(self, niter=10):
        for i in range(niter):
            print(f'__test_radial_bincount: iter {i}/{niter-1}')

            r = rand.randint(10, 200, 1)[0]

            if r % 2 == 0:
                r += 1

            for j in ((r, r), (r, r*3), (r*5, r*7)):
                arr0 = np.zeros(j)
                arr1 = np.ones(j)

                for cxy in ([r/4, r/2], (0, 0), [r-1, r-1]):
                    l0, cl0 = self.radial_bincount(arr0, cxy=cxy)
                    l1, cl1 = self.radial_bincount(arr1, cxy=cxy)

                    assert arr0.mean() == cl0.mean()
                    assert arr1.mean() == cl1.mean()

            for j in ((r, r), (r*3, r*3), (r*5, r*5)):
                arr0 = np.zeros(j)
                arr1 = np.ones(j)

                for up in (1, 3, 5, 7):
                    l0, cl0 = self.radial_bincount(arr0, up=up)
                    l1, cl1 = self.radial_bincount(arr1, up=up)

                    assert np.all(l0 == l1)

                    l_max_ref = (arr0.shape[0] / 2) * np.sqrt(2.0)
                    l_max_out = np.max(l0)

                    assert abs(l_max_out - l_max_ref) < 1.0
                    assert arr0.mean() == cl0.mean()
                    assert arr1.mean() == cl1.mean()

        # Creating an array with shape (101,101) whose elements correspond
        # to radial values from the center (e.g. r[50][50] = 0.0).
        x = np.linspace(0.0, 100, 101)
        x, y = np.meshgrid(x, x)
        r = np.hypot((x-50), (y-50))

        # Experimenting with a few different upsampling scales.
        ups = [1, 3, 9, 27, 81]
        for up in ups:
            l, cl = self.radial_bincount(r, up=up)
            l, cl = fx.downsample_cl(l, cl, nl=up)

            delta = np.abs(l - cl)
            plt.plot(l, delta, label=f'${up}$')

        plt.xlabel(r'$l$')
        plt.ylabel(r'$\Delta$')
        plt.legend(loc='best')
        plt.savefig('test_radial_bincount_0.pdf')
        plt.clf()

        # Testing the invariance under rescaling transformations.
        f0 = self.power_spectrum_to_cl(r)
        f1 = self.power_spectrum_to_cl(r, nl=10)

        plt.plot(f0['ell'], f0['cl'])
        plt.plot(f1['ell'], f1['cl'])
        plt.savefig('test_radial_bincount_1.pdf')
        plt.clf()

        f0 = self.power_spectrum_to_cl(r)
        f1 = self.power_spectrum_to_cl(r, up=9)

        plt.plot(f0['ell'], f0['cl'])
        plt.plot(f1['ell'], f1['cl'])
        plt.savefig('test_radial_bincount_2.pdf')
        plt.clf()

        f0 = self.power_spectrum_to_cl(r)
        f1 = self.power_spectrum_to_cl(r, nl=10, up=9)

        plt.plot(f0['ell'], f0['cl'])
        plt.plot(f1['ell'], f1['cl'])
        plt.savefig('test_radial_bincount_3.pdf')
        plt.clf()

        f0 = self.power_spectrum_to_cl(r, nl=10)
        f1 = self.power_spectrum_to_cl(r, up=9)

        plt.plot(f0['ell'], f0['cl'])
        plt.plot(f1['ell'], f1['cl'])
        plt.savefig('test_radial_bincount_4.pdf')
        plt.clf()
