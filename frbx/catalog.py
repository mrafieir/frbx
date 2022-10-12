import os
import re
import csv
import glob
import requests
import json
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from h5py import File as FileH5
import matplotlib.pyplot as plt
import frbx as fx


######################          G A L A X Y   C A T A L O G S          ######################


class galaxy_catalog:
    """
    Represents a galaxy catalog, i.e. a list of objects with redshifts and sky coordinates.
    See galaxy_catalog_2mpz() for a helper function which returns the 2MPZ catalog.

    Constructor arguments:

       z            1-d array containing redshifts
       l_deg        1-d array containing values of 'l', the longitude coordinate in galactic coordinates
       b_deg        1-d array containing values of 'b', the latitude coordinate in galactic coordinates
       aux          n-d array containing values of auxiliary data, e.g. magnitudes
       plt_args     dictionary containing optional keyword arguments for customizing plots.

    In plot_*() methods, the 'doc' argument should either be None (to show a plot interactively),
    or a handout.Handout instance (to show a plot in a web-browsable output directory).
    """

    def __init__(self, z, l_deg, b_deg, aux=None, plt_args=None):
        if plt_args is None:
            plt_args = {}
        assert z.size > 0
        assert z.ndim == 1
        assert l_deg.shape == z.shape
        assert b_deg.shape == z.shape

        self.size = z.size
        if aux is not None:
            assert aux.shape[0] == self.size

        fx.utils.sanity_check_lon_lat_arrays(l_deg, b_deg)

        self.z = z
        self.l_deg = l_deg
        self.b_deg = b_deg
        self.aux = aux
        self.plt_args = plt_args

    def __add__(self, gcat):
        z = np.append(self.z, gcat.z, axis=0)
        l_deg = np.append(self.l_deg, gcat.l_deg, axis=0)
        b_deg = np.append(self.b_deg, gcat.b_deg, axis=0)

        plt_args = {**gcat.plt_args, **self.plt_args}

        if self.aux is not None:
            aux = np.append(self.aux, gcat.aux, axis=0)
        else:
            aux = None

        return galaxy_catalog(z, l_deg, b_deg, aux, plt_args)

    def __radd__(self, gcat):
        if gcat == 0:
            return self
        else:
            return self.__add__(gcat)

    def make_subcatalog(self, mask):
        """
        The 'mask' argument is a boolean array of length self.size, indicating
        which galaxies are in the subcatalog (True), or not in the subcatalog (False).

        Returns an object of type galaxy_catalog.
        """

        assert mask.shape == (self.size,)
        assert mask.dtype == np.bool
        assert np.sum(mask) > 0

        aux = self.aux[mask] if self.aux is not None else None
        return galaxy_catalog(self.z[mask], self.l_deg[mask], self.b_deg[mask], aux)

    def make_zbin_subcatalog(self, zmin, zmax):
        """Returns subcatalog obtained by restricting to zmin <= z <= zmax."""

        mask = np.logical_and(self.z >= zmin, self.z <= zmax)
        return self.make_subcatalog(mask)

    def plot_z_histogram(self, doc=None):
        plt.hist(self.z, bins=50)
        plt.xlabel(r'$z$')
        fx.showfig(doc, self.plt_args)

    def plot_prob_histogram(self, n, doc=None):
        plt.hist(self.aux[:,n], bins=50)
        plt.xlabel(r'$p$')
        fx.showfig(doc, self.plt_args)

    def plot_histogram(self, arr, xlabel, bins=20, doc=None, label=None):
        plt.hist(arr, bins=bins, label=None)
        plt.xlabel(xlabel)
        if label is not None:
            plt.legend()
        fx.showfig(doc, self.plt_args)

    def plot_l_b(self, doc=None):
        plt.hexbin(self.l_deg, self.b_deg, extent=(0,360,-90,90), gridsize=512)
        plt.xlabel(r'$l~{\rm (deg)}$')
        plt.ylabel(r'$b~{\rm (deg)}$')
        plt.xlim(0, 360.)
        plt.ylim(-90.0, 90.0)
        fx.showfig(doc, self.plt_args)

    def plot_healpix_map(self, doc=None, nside=64):
        m = fx.utils.make_healpix_map_from_catalog(nside, self.l_deg, self.b_deg)
        fx.utils.show_healpix_map(m, doc)


def galaxy_catalog_sdss_dr14(dirpath=fx.data_path('archive/catalogs/sdss_dr14'), obj='LRG', stype='data'):
    """Returns SDSS-DR14 spectroscopic catalog."""

    assert obj_type in ('LRG', 'QSO')
    assert stype in ('data', 'random')

    filename = f'{stype}_DR14_{obj}*.fits'
    input_files = glob.glob(os.path.join(dirpath, filename))

    d = []
    for _file in input_files:
        with fits.open(_file) as _f:
            # Data is located in the second HDU.
            d.append(Table.read(_f[1]))

    d = vstack(d, join_type='outer', metadata_conflicts='silent')

    z = np.array(d['Z'])

    # Being unsure of the frame below.
    l_deg, b_deg = fx.utils.convert_ra_dec_to_l_b(np.array(d['RA']), np.array(d['DEC']), frame='icrs')

    return galaxy_catalog(z=z, l_deg=l_deg, b_deg=b_deg)


def galaxy_catalog_2mpz(filename=fx.data_path('archive/catalogs/2mpz/2mpz.fits')):
    """
    Returns 2MPZ, the 2MASS photometric galaxy catalog.

    (Not to be confused with 2MRS, the 2MASS spectroscopic galaxy catalog.)

    Catalog:    https://iopscience.iop.org/article/10.1088/0067-0049/210/1/9/pdf
                http://ssa.roe.ac.uk/sqlcookbook.html#2MPZ-matching
    Mask:       https://arxiv.org/pdf/1412.5151.pdf     (-> fx.utils.get_2mass_mask)
    Cl^{gg}:    https://arxiv.org/pdf/1711.04583.pdf
    """

    t = Table.read(filename, memmap=True)
    print(f'read {filename}')

    assert len(t) == 934175

    mask = t['KCORR'] < 13.9        # Completeness limit.
    assert sum(mask) == 933040

    z = np.array(t['ZPHOTO'][mask])
    l_deg = np.array(t['L'][mask])
    b_deg=np.array(t['B'][mask])

    return galaxy_catalog(z=z, l_deg=l_deg, b_deg=b_deg)


def fits_to_h5(fits_file, h5_file, dataset='catalog', chunk_size=16384, aux_keys=None,
               sift=None, frame='fk5', photoz_file=None):
    """
    Parses relevant columns of 'fits_file' (str) into a 'dataset' (str) in 'h5_file' (str).
    The 'chunk_size' argument specifies the number of rows to be parsed in each iteration while
    reading an entire memory map.  It assumes that the following keys exist in the input catalog:
    ['RA', 'DEC'].  In addition, a list of auxiliary keys ['aux_key1', 'aux_key2', ...], sourced
    by 'fits_file' (primary)  or 'photoz_file' (secondary), can be supplied.
    The 'sift' argument specifies a list of [key,value] lists which are matched against data;
    matched and unmatched values (for given keys) are set to 0 and 1, respectively.
    The 'photoz_file' argument specifies a file containing photometric redshift values
    (and/or 'aux_keys'). It returns the total number of objects.
    """

    assert fits_file.endswith('.fits')
    assert h5_file.endswith('.h5')
    assert isinstance(dataset, str)
    assert chunk_size == int(chunk_size)

    t = Table.read(fits_file, memmap=True)

    if photoz_file is not None:
        _n = fits_file.split('/')[-1].split('.')[0]
        _nz = photoz_file.split('/')[-1].split('.')[0]
        assert _n in _nz, f'fits_to_h5: {fits_file}, {photoz_file}'
        tz = Table.read(photoz_file, memmap=True)
    else:
        tz = None

    try:
        ra = t['RA']
    except KeyError:
        ra = t['ra']

    try:
        dec = t['DEC']
    except KeyError:
        dec = t['dec']

    n = ra.size

    w = 3
    if aux_keys is not None:
        w += len(aux_keys)
    if sift is not None:
        w += len(sift)

    try:
        if photoz_file is not None:
            z = tz['z_phot_mean']
        else:
            z = t['photo_z']
    except KeyError:
        # Uniformly sampling redshifts -> easier selection cuts and inverse transform sampling.
        z = np.random.uniform(1.0e-7, 1.0, size=n)

    f = FileH5(h5_file, mode='w')
    d = f.create_dataset(dataset, shape=(n,w))

    for i in range(0, n, chunk_size):
        print(f'fits_to_h5: {i/n*100:.1f} %')

        j = i + chunk_size
        l_deg, b_deg = fx.utils.convert_ra_dec_to_l_b(ra[i:j], dec[i:j], frame=frame)

        d[i:j,0] = l_deg[:]
        d[i:j,1] = b_deg[:]
        d[i:j,2] = z[i:j]

        if aux_keys is not None:
            for k, key in enumerate(aux_keys, 3):
                try:
                    _d = t[key]
                except KeyError:
                    _d = tz[key]
                d[i:j,k] = _d[i:j]

        if sift is not None:
            for k, kv in enumerate(sift, 3+len(aux_keys)):
                key, value = kv
                d[i:j,k] = np.asarray(t[key][i:j] != value, dtype=int)

    f.close()
    return n


def h5_to_galaxy_catalog(h5_file, dataset='catalog', aux=False):
    """
    Returns a galaxy_catalog object based on an input 'dataset' (str) in 'h5_file' (str).
    It assumes the following columns along axis 1: (l_deg, b_deg, z, [aux])
    """

    cat = fx.utils.read_h5(h5_file, dataset)

    x = cat[:,3:] if aux else None
    return galaxy_catalog(z=cat[:,2], l_deg=cat[:,0], b_deg=cat[:,1], aux=x)


class galaxy_catalog_desilis_dr8:
    """
    Represents the 8th data release of the DESI Legacy Imaging Surveys.  Once constructed,
    it can be called to return a galaxy_catalog object based on real photometric data or
    random realizations.

    Refs:
    Zou Hu's catalog: http://batc.bao.ac.cn/~zouhu/doku.php?id=projects:desi_photoz:start
    Sweep catalogs:
    http://www.legacysurvey.org/dr8/files/#photometric-redshift-files-8-0-photo-z-sweep-brickmin-brickmax-pz-fits
    PRLS Sweep catalogs: https://arxiv.org/abs/2001.06018
    Sweep randoms: https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/randoms/
    LRG sample: https://arxiv.org/pdf/2010.11282.pdf
    ELG sample: https://arxiv.org/pdf/2010.11281.pdf
    BGS sample: https://arxiv.org/pdf/2010.11283.pdf
    self.sift_real: fig 14, https://hal.archives-ouvertes.fr/hal-02003991/document

    Constructor arguments:

        dirpath          str specifying a path to the directory of input/output files
        ref              str specifying the origin of data.
    """

    def __init__(self, dirpath=fx.data_path('archive/catalogs/desilis_dr8/'), ref='zhou20'):
        self.dirpath = dirpath

        self.in_randoms = glob.glob(self.dirpath + 'randoms-inside-dr8-0.31.0-*.fits')
        self.out_randoms='randoms_desilis_dr8.h5'
        self.aux_keys_randoms = ['GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'MASKBITS',
                                 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'EBV']

        self.dataset = 'catalog'
        self.sift_real = [['TYPE','PSF ']]      # Note the trailing space char!

        self.nexp = [2.0, 2.0, np.inf]

        self.__init_catalogs(self.in_randoms, self.out_randoms, self.aux_keys_randoms)

        self.ref = ref

        if self.ref == 'zou':
            self.in_real = glob.glob(self.dirpath + 'LS_DR8_total_csp.fits')
            self.photoz_real = None
            self.out_real = 'zou_desilis_dr8.h5'

            self.aux_keys_real = ['MAG_G', 'MAG_R', 'MAG_Z', 'MAG_W1', 'MAG_W2',
                                  'MAGERR_G', 'MAGERR_R', 'MAGERR_Z', 'MAGERR_W1',
                                  'MAGERR_W2', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
                                  'photo_zerr', 'spec_z']

            self.__init_catalogs(self.in_real, self.out_real, self.aux_keys_real, self.sift_real, self.photoz_real)
        elif self.ref == 'zhou20':
            self.in_real = [sorted(glob.glob(self.dirpath + 'north/sweep/8.0/*.fits')),
                            sorted(glob.glob(self.dirpath + 'south/sweep/8.0/*.fits'))]

            self.photoz_real = [sorted(glob.glob(self.dirpath + 'north/sweep/8.0-photo-z/*.fits')),
                                sorted(glob.glob(self.dirpath + 'south/sweep/8.0-photo-z/*.fits'))]

            self.out_real = ['zhou20_desilis_dr8_north.h5', 'zhou20_desilis_dr8_south.h5']

            self.aux_keys_real = ['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1',
                                  'FLUX_W2', 'GALDEPTH_G', 'GALDEPTH_R',
                                  'GALDEPTH_Z', 'MASKBITS', 'NOBS_G', 'NOBS_R',
                                  'NOBS_Z', 'z_phot_std', 'MW_TRANSMISSION_G',
                                  'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
                                  'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
                                  'FIBERFLUX_R', 'FIBERFLUX_Z',
                                  'FRACFLUX_G', 'FRACFLUX_R', 'FRACFLUX_Z',
                                  'FRACMASKED_G', 'FRACMASKED_R', 'FRACMASKED_Z',
                                  'FRACIN_G', 'FRACIN_R', 'FRACIN_Z']

            for (in_real, photoz_real, out_real) in zip(self.in_real, self.photoz_real, self.out_real):
                self.__init_catalogs(in_real, out_real, self.aux_keys_real, self.sift_real, photoz_real)
        else:
            raise RuntimeError('galaxy_catalog_desilis_dr8: invalid ref!')

    @staticmethod
    def maskbit(mode):
        """
        Maskbits
        1: Tycho-2 and GAIA bright stars
        5-7: ALLMASK_r/g/z bits
        8: WISE W1 bright stars
        9: WISE W2 bright stars
        11: fainter GAIA stars
        12: large galaxies
        13: globular clusters
        """

        if 'bgs' in mode:
            #maskbits = [1, 12, 13]
            maskbits = [1, 5, 6, 7, 8, 9, 11, 12, 13]
        else:
            maskbits = [1, 5, 6, 7, 8, 9, 11, 12, 13]

        ret = 2**np.array(maskbits)

        return np.bitwise_or.reduce(ret)

    def __call__(self, mode, maskbit=True, expcut=True, index=None, zstd_cut=0.08):
        assert index in (None, 0, 1)

        if any(s in mode for s in ('real', 'zou_training', 'zhou20_lrg', 'zhou20_elg', 'zhou20_bgs',
                                   'zhou20_lrg_elg_bgs', 'zhou20_lrg_elg', 'zhou20_all')):
            assert self.ref in mode.split('_'), f'galaxy_catalog_desilis_dr8: {mode} and {self.ref} are inconsistent!'
            _f = self.out_real
        elif mode in ('randoms_lrg', 'randoms_elg', 'randoms_bgs'):
            _f = self.out_randoms
        else:
            raise RuntimeError('galaxy_catalog_desilis_dr8: invalid mode!')

        if not isinstance(_f, list):
            h5_file = self.dirpath + _f
            g_cat = h5_to_galaxy_catalog(h5_file, self.dataset, aux=True)
        elif index is None:
            g_cat = h5_to_galaxy_catalog(self.dirpath+_f[0], self.dataset, aux=True)
            for i in _f[1:]:
                g_cat += h5_to_galaxy_catalog(self.dirpath+i, self.dataset, aux=True)
        else:
            g_cat = h5_to_galaxy_catalog(self.dirpath+_f[index], self.dataset, aux=True)

        print(f'galaxy_catalog_desilis_dr8: Ng = {g_cat.size} ({mode}, index={index})')

        if any(s in mode for s in ('lrg','elg','bgs')) and all(s not in mode for s in ('all','zou','randoms')):
            # extinction-corrected magnitudes
            mags = fx.nanomaggies_to_mag(g_cat.aux[:,:5], g_cat.aux[:,13:18])

            if 'lrg' in mode:  # 0.3 < redshift < 1.0 (preliminary)
                if index is None:
                    raise RuntimeError('zhou20_lrg sample requires a specific index: 0 -> North, 1 -> South.')

                m = mags[:,0] != -99.0
                for i in (1, 2, 3):
                    m *= mags[:,i] != -99.0

                print(f'mags != -99.0 : {np.sum(m)}')

                # non-stellar cut: (z-W1) > (0.8*(r-z) - 0.6)
                q = 0.6 if index else 0.65
                m *= (mags[:,2] - mags[:,3]) > (0.8*(mags[:,1] - mags[:,2]) - q)

                print(f'non-stellar cut: (z-W1) > (0.8*(r-z) - 0.6) : {np.sum(m)}')

                # (((g - W1) > 2.6) and ((g-r) > 1.4)) or ((r-W1) > 1.8)
                q = (2.6, 1.4, 1.8) if index else (2.67, 1.45, 1.85)
                _m = np.logical_and((mags[:,0] - mags[:,3]) > q[0], (mags[:,0] - mags[:,1]) > q[1])
                m *= np.logical_or(_m, (mags[:,1] - mags[:,3]) > q[2])

                print(f'(((g - W1) > 2.6) and ((g-r) > 1.4)) or ((r-W1) > 1.8) : {np.sum(m)}')

                # ((r-z) > (z-16.83)*0.45) and ((r-z) > (z-13.80)*0.19)
                q = (16.83, 13.80) if index else (16.69, 13.68)
                _m = (mags[:,1] - mags[:,2]) > ((mags[:,2] - q[0]) * 0.45)
                m *= np.logical_and(_m, (mags[:,1] - mags[:,2]) > ((mags[:,2] - q[1]) * 0.19))

                print(f'((r-z) > (z-16.83)*0.45) and ((r-z) > (z-13.80)*0.19) : {np.sum(m)}')

                # (r-z) > 0.7
                m *= (mags[:,1] - mags[:,2]) > 0.7

                print(f'(r-z) > 0.7 : {np.sum(m)}')

                if maskbit:
                    x = np.array(g_cat.aux[:,8], dtype=int)
                    m *= (x & self.maskbit(mode)) == 0

                print(f'maskbit : {np.sum(m)}')

                if expcut:
                    for i in (9, 10, 11):
                        m *= np.logical_and(g_cat.aux[:,i] >= self.nexp[1], g_cat.aux[:,i] <= self.nexp[2])

                print(f'expcut : {np.sum(m)}')

                # FIBERFLUX_Z < 21.5
                m *= fx.nanomaggies_to_mag(g_cat.aux[:,19], np.ones_like(g_cat.aux[:,19])) < 21.5

                print(f'FIBERFLUX_Z < 21.5 : {np.sum(m)}')

                if zstd_cut:
                    # z_phot_std <= zstd_cut
                    m *= g_cat.aux[:,12] <= zstd_cut
                    print(f'z_phot_std : {np.sum(m)}')

                # sift flags
                _m = np.sum(m)
                m *= g_cat.aux[:,29] == 1
                print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}): {np.sum(m)} / {_m} obj passed thru the sift.')

                m_lrg = m.copy()
            else:
                m_lrg = np.zeros_like(g_cat.size, dtype=bool)

            if 'elg' in mode:  # 0.6 < redshift < 1.6 (preliminary)
                if index is None:
                    raise RuntimeError('zhou20_elg sample requires a specific index: 0 -> North, 1 -> South.')

                m = mags[:,0] != -99.0
                for i in (1, 2, 3):
                    m *= mags[:,i] != -99.0

                print(f'mags != -99.0 : {np.sum(m)}')

                (gmax, zpt) = (23.5, -0.15) if index else (23.6, -0.35)

                # 20.0 < g < gmax
                m *= np.logical_and(mags[:,0] > 20.0, mags[:,0] < gmax)

                print(f'20.0 < g < gmax : {np.sum(m)}')

                # 0.3 < (r-z) < 1.6
                t = mags[:,1] - mags[:,2]
                m *= np.logical_and(t > 0.3, t < 1.6)

                print(f'0.3 < (r-z) < 1.6 : {np.sum(m)}')

                # (g-r) < 1.15 * (r-z) + zpt
                t1 = mags[:,0] - mags[:,1]
                t2 = mags[:,1] - mags[:,2]
                m *= t1 < (1.15 * t2 + zpt)

                print(f'(g-r) < 1.15 * (r-z) + zpt : {np.sum(m)}')

                # (g-r) < -1.20 * (r-z) + 1.6
                t1 = mags[:,0] - mags[:,1]
                t2 = mags[:,1] - mags[:,2]
                m *= t1 < (-1.20 * t2 + 1.6)

                print(f'(g-r) < -1.20 * (r-z) + 1.6 : {np.sum(m)}')

                if maskbit:
                    x = np.array(g_cat.aux[:,8], dtype=int)
                    m *= (x & self.maskbit(mode)) == 0

                print(f'maskbit : {np.sum(m)}')

                if expcut:
                    for i in (9, 10, 11):
                        m *= np.logical_and(g_cat.aux[:,i] >= self.nexp[0], g_cat.aux[:,i] <= self.nexp[2])

                print(f'expcut : {np.sum(m)}')

                # sift flags
                _m = np.sum(m)
                m *= g_cat.aux[:,29] == 1
                print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}): {np.sum(m)} / {_m} obj passed thru the sift.')

                m_elg = m.copy()
            else:
                m_elg = np.zeros_like(g_cat.size, dtype=bool)

            if 'bgs' in mode:  # 0.05 < redshift < 0.4 (preliminary)
                m = mags[:,0] != -99.0
                for i in (1, 2, 3):
                    m *= mags[:,i] != -99.0

                print(f'mags != -99.0 : {np.sum(m)}')

                # faint + bright: r < 20
                m *= mags[:,1] < 20.0

                print(f'faint + bright: r < 20 : {np.sum(m)}')

                # -1 < (g-r) < 4
                t = mags[:,0] - mags[:,1]
                m *= np.logical_and(t > -1.0, t < 4.0)

                print(f'-1 < (g-r) < 4 : {np.sum(m)}')

                # -1 < (r-z) < 4
                t = mags[:,1] - mags[:,2]
                m *= np.logical_and(t > -1.0, t < 4.0)

                print(f'-1 < (r-z) < 4 : {np.sum(m)}')

                # ((r <= 17.8) and (rfibmag < 22.9 + (r - 17.8))) or
                #       ((17.8 < r < 20) and (rfibmag < 22.9))
                rfibmag = fx.nanomaggies_to_mag(g_cat.aux[:,18], np.ones_like(g_cat.aux[:,18]))

                t1 = mags[:,1] <= 17.8
                _m1 = np.logical_and(t1, rfibmag < (22.9 + mags[:,1] - 17.8))

                t2 = np.logical_and(mags[:,1] > 17.8, mags[:,1] < 20.0)
                _m2 = np.logical_and(t2, rfibmag < 22.9)

                m *= np.logical_or(_m1, _m2)

                print(f'((r <= 17.8) and (rfibmag < 22.9 + (r - 17.8))) or\n'
                      f'((17.8 < r < 20) and (rfibmag < 22.9)) : {np.sum(m)}')

                # FRACFLUX_r/g/z < 5
                for i in (20, 21, 22):
                    m *= g_cat.aux[:,i] < 5

                print(f'FRACFLUX_r/g/z < 5 : {np.sum(m)}')

                # FRACMASKED_r/g/z < 0.4
                for i in (23, 24, 25):
                    m *= g_cat.aux[:,i] < 0.4

                print(f'FRACMASKED_r/g/z < 0.4 : {np.sum(m)}')

                # FRACIN_r/g/z > 0.3
                for i in (26, 27, 28):
                    m *= g_cat.aux[:,i] > 0.3

                print(f'FRACIN_r/g/z > 0.3 : {np.sum(m)}')

                if maskbit:
                    x = np.array(g_cat.aux[:,8], dtype=int)
                    m *= (x & self.maskbit(mode)) == 0

                print(f'maskbit : {np.sum(m)}')

                if expcut:
                    for i in (9, 10, 11):
                        m *= np.logical_and(g_cat.aux[:,i] >= self.nexp[1], g_cat.aux[:,i] <= self.nexp[2])

                print(f'expcut : {np.sum(m)}')

                if zstd_cut:
                    # z_phot_std <= zstd_cut
                    m *= g_cat.aux[:,12] <= zstd_cut
                    print(f'z_phot_std : {np.sum(m)}')

                # sift flags
                _m = np.sum(m)
                m *= g_cat.aux[:,29] == 1
                print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}): {np.sum(m)} / {_m} obj passed thru the sift.')

                m_bgs = m.copy()
            else:
                m_bgs = np.zeros_like(g_cat.size, dtype=bool)

            _m = np.logical_or(m_lrg, m_elg)
            print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}, lrg_and_elg): {np.sum(np.logical_and(m_lrg, m_elg))}')

            m = np.logical_or(_m, m_bgs)
            print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}, lrg_and_elg_and_bgs): {np.sum(np.logical_and(_m, m_bgs))}')

            ret = g_cat.make_subcatalog(m)

        elif mode == 'zou_training':
            if maskbit or expcut:
                raise RuntimeError(f"'maskbit' and 'expcut' args not available for {mode}.")

            m = g_cat.aux[:,5] < 0.27
            m *= g_cat.aux[:,6] < 0.14
            m *= g_cat.aux[:,7] < 0.14

            # 0.09 < (g-r) < 2.20
            t = g_cat.aux[:,0] - g_cat.aux[:,1]
            m *= np.logical_and(t > 0.09, t < 2.20)

            # 0.02 < (r-z) < 1.96
            t = g_cat.aux[:,1] - g_cat.aux[:,2]
            m *= np.logical_and(t > 0.02, t < 1.96)

            # -1.16 < (r-W1) < 3.52
            t = g_cat.aux[:,1] - g_cat.aux[:,3]
            m *= np.logical_and(t > -1.16, t < 3.52)

            # -1.94 < (r-W2) < 3.16
            t = g_cat.aux[:,1] - g_cat.aux[:,4]
            m *= np.logical_and(t > -1.94, t < 3.16)

            ret = g_cat.make_subcatalog(m)

        elif mode == 'zhou20_all':  # 0.0 < redshift < 1.0
            # extinction-corrected magnitudes
            mags = fx.nanomaggies_to_mag(g_cat.aux[:,:5], g_cat.aux[:,13:18])

            m = mags[:,0] != -99.0
            for i in (1, 2, 3):
                m *= mags[:,i] != -99.0

            # non-stellar cut: (z-W1) > (0.8*(r-z) - 0.6)
            m *= (mags[:,2] - mags[:,3]) > (0.8*(mags[:,1] - mags[:,2]) - 0.6)

            # faint limit cut: z < 20.41
            m *= mags[:,2] < 20.41

            # r < 23.0
            m *= mags[:,1] < 23.0

            if maskbit:
                x = np.array(g_cat.aux[:,8], dtype=int)
                m *= (x & self.maskbit(mode)) == 0

            if expcut:
                for i in (9, 10, 11):
                    m *= np.logical_and(g_cat.aux[:,i] >= self.nexp[1], g_cat.aux[:,i] <= self.nexp[2])

            if zstd_cut:
                # z_phot_std <= zstd_cut
                m *= g_cat.aux[:,12] <= zstd_cut

            # sift flags
            _m = np.sum(m)
            m *= g_cat.aux[:,29] == 1
            print(f'galaxy_catalog_desilis_dr8.__call__(mode={mode}): {np.sum(m)} / {_m} obj passed thru the sift.')

            ret = g_cat.make_subcatalog(m)

        elif 'randoms' in mode:
            if maskbit:
                x = np.array(g_cat.aux[:,3], dtype=int)
                m = (x & self.maskbit(mode)) == 0
            else:
                m = np.ones(g_cat.size, dtype=bool)

            if expcut:
                for i in (4, 5, 6):
                    j = 0 if ('elg' in mode) else 1
                    m *= np.logical_and(g_cat.aux[:,i] >= self.nexp[j], g_cat.aux[:,i] <= self.nexp[2])

            ret = g_cat.make_subcatalog(m)

        else:
            ret = g_cat

        return ret

    def __init_catalogs(self, fits_files, h5_file, aux_keys=None, sift=None, photoz_files=None):
        """Calls fits_to_h5 for a list of 'fits_files' in order to generate a single 'h5_file' (str)."""

        p = self.dirpath + h5_file

        try:
            _h5_file = fx.data_path(p, mode='r')
        except RuntimeError as err:
            print(err)
            _h5_file = fx.data_path(p, mode='w')

            s = [0]
            if len(fits_files) == 1:
                photoz_file = photoz_files[0] if (photoz_files is not None) else None
                s.append(fits_to_h5(fits_files[0], _h5_file, aux_keys=aux_keys, sift=sift, photoz_file=photoz_file))
            else:
                for i, fits_file in enumerate(fits_files):
                    print(f'__init_catalogs: {fits_file}')

                    photoz_file = photoz_files[i] if (photoz_files is not None) else None
                    s.append(fits_to_h5(fits_file, self.dirpath+f'temp_{i}.h5',
                                        aux_keys=aux_keys, sift=sift, photoz_file=photoz_file))

                s = np.cumsum(s)
                f = FileH5(_h5_file, mode='w')

                w = 3
                if aux_keys is not None:
                    w += len(aux_keys)
                if sift is not None:
                    w += len(sift)

                d = f.create_dataset(self.dataset, shape=(s[-1],w), dtype=np.float64)

                for i, fits_file in enumerate(fits_files):
                    print(f'__init_catalogs: concatenating {fits_file}')
                    _d = fx.utils.read_h5(self.dirpath+f'temp_{i}.h5', self.dataset)
                    d[s[i]:s[i+1],:] = _d[:]
                    os.remove(self.dirpath+f'temp_{i}.h5')

                f.close()

            print(f'__init_catalogs: total number of objects = {s[-1]/1.0e8} * 1.0e8')


class galaxy_catalog_sdss_dr8:
    """
    Represents the SDSS-DR8 photometric redshift catalog.  Once constructed, it can be called
    to return a galaxy_catalog object based on real photometric data or random realizations.

    Main catalog: https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/photoz-weight/pofz.html
    Depth variation: https://arxiv.org/pdf/1509.00870.pdf

    Constructor arguments:

        catpath          str specifying a path to the directory of input/output catalogs
        in_real          str specifying a fits file corresponding to real data in 'catpath'
        out_real         str specifying an h5 file corresponding to real data in 'catpath'
        randoms          str specifying an h5 file corresponding to randoms in 'catpath'
        maskpath         str specifying a path to the directory of angular masks
    """

    def __init__(self, catpath=fx.data_path('archive/catalogs/sdss_dr8/'),
                 in_real='pofz*.fits.gz',
                 out_real='sdss_dr8.h5',
                 randoms='randoms_sdss_dr8.h5',
                 maskpath=fx.data_path('archive/maps/sdss_dr8/boss_masks/')):

        self.catpath = catpath
        self.in_real = glob.glob(self.catpath + in_real)
        self.out_real = out_real
        self.randoms = randoms
        self.maskpath = maskpath

        self.dataset = 'catalog'
        self.aux_keys_real = ['objid', 'cmodelmag_r', 'pofz']   # TODO pofz.size == 35

        self.__init_real()
        self.__init_randoms()

    def __call__(self, mode):
        if mode == 'real':
            _f = self.out_real
            aux = True
        elif mode == 'randoms':
            _f = self.randoms
            aux = False
        else:
            raise RuntimeError('galaxy_catalog_sdss_dr8: invalid mode!')

        h5_file = self.catpath + _f
        ret = h5_to_galaxy_catalog(h5_file, self.dataset, aux)
        print(f'galaxy_catalog_sdss_dr8: Ng = {ret.size} ({mode})')

        return ret

    def __init_real(self):
        """Initializes the real catalog."""

        p = self.catpath + self.out_real

        try:
            _h5_file = fx.data_path(p, mode='r')
        except RuntimeError as err:
            print(err)
            _h5_file = fx.data_path(p, mode='w')

            s = [0]
            if len(self.in_real) == 1:
                s.append(fits_to_h5(self.in_real[0], _h5_file, aux_keys=self.aux_keys_real, frame='icrs'))
            else:
                for i, fits_file in enumerate(self.in_real):
                    print(f'__init_real: {fits_file}')
                    s.append(fits_to_h5(fits_file, self.catpath+f'temp_{i}.h5',
                                        aux_keys=self.aux_keys_real, frame='icrs'))

                s = np.cumsum(s)
                f = FileH5(_h5_file, mode='w')

                w = 3
                if self.aux_keys_real is not None:
                    w += len(self.aux_keys_real)

                d = f.create_dataset(self.dataset, shape=(s[-1],w), dtype=np.float64)

                for i, fits_file in enumerate(self.in_real):
                    print(f'__init_real: concatenating {fits_file}')
                    _d = fx.utils.read_h5(self.catpath+f'temp_{i}.h5', self.dataset)
                    d[s[i]:s[i+1],:] = _d[:]
                    os.remove(self.catpath+f'temp_{i}.h5')

                f.close()

            print(f'__init_real: total number of objects = {s[-1]/1.0e6} * 1.0e6')

    def __init_randoms(self):
        """Initializes the random catalog."""

        p = self.catpath + self.randoms

        try:
            _h5_file = fx.data_path(p, mode='r')
        except RuntimeError as err:
            print(err)
            _h5_file = fx.data_path(p, mode='w')
            # TODO


def galaxy_catalog_wise_scos(filename=fx.data_path('archive/catalogs/wise_scos/wiseScosPhotoz160708.csv')):
    """
    Returns the WISExSuperCOSMOS galaxy catalog.
    https://iopscience.iop.org/article/10.3847/0067-0049/225/1/5/pdf
    """

    cat = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if reader.line_num == 1:
                continue

            l_deg = float(r[9])
            b_deg = float(r[10])
            mag_r = float(r[15])
            z = float(r[17])

            cat.append([l_deg, b_deg, z, mag_r])

        print(f'galaxy_catalog_wise_scos: Ng = {reader.line_num}')

    cat = np.asarray(cat)
    ret = galaxy_catalog(l_deg=cat[:,0], b_deg=cat[:,1], z=cat[:,2], aux=cat[:,3])

    return ret


def galaxy_catalog_wise_scos_svm(filename=fx.data_path('archive/catalogs/wise_scos/wiseScosSvm.csv'),
                                 mode='g', pthresh=0.9):
    """
    Returns the WISExSuperCOSMOS galaxy catalog (SVM version).

    Main catalog: https://www.aanda.org/articles/aa/pdf/2016/12/aa29165-16.pdf
    Probability values: https://www.aanda.org/articles/aa/pdf/2016/08/aa28142-16.pdf
    """

    cat = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if reader.line_num == 1:
                continue

            prob_g = float(r[22])
            prob_s = float(r[23])
            prob_q = float(r[24])

            if (mode == 'g') and (prob_g <= pthresh):
                continue

            if (mode == 's') and (prob_s <= pthresh):
                continue

            if (mode == 'q') and (prob_q <= pthresh):
                continue

            z = float(r[19])
            if z < 0.0:
                continue

            ra_deg = float(r[7])
            dec_deg = float(r[8])

            mag_w1 = float(r[10])
            mag_w2 = float(r[12])
            mag_r = float(r[16])

            cat.append([ra_deg, dec_deg, z, mag_w1, mag_w2, mag_r, prob_g, prob_s, prob_q])

        print(f'galaxy_catalog_wise_scos_svm: Ng = {reader.line_num}')

    cat = np.asarray(cat)
    l_deg, b_deg = fx.utils.convert_ra_dec_to_l_b(cat[:,0], cat[:,1])

    ret = galaxy_catalog(l_deg=l_deg[:], b_deg=b_deg[:], z=cat[:,2], aux=cat[:,3:])

    return ret


######################          F R B   C A T A L O G S          ######################


class frb_catalog:
    """
    Represents an FRB catalog, i.e. list of objects with sky coordinates and DMs.

    See frb_catalog_may7() and frb_catalog_jun23() for helper functions which return specific CHIME catalogs.

    Constructor arguments:

       snr              1-d array containing redshifts
       dm_obs           1-d array containing observed (= galactic + IGM + host) DMs
       ra_deg           1-d array containing values of 'ra', the longitude coordinate in equatorial coordinates
       dec_deg          1-d array containing values of 'dec', the latitude coordinate in equatorial coordinates
       dm_gal           1-d array containing galactic DMs (from NE2001 or YMW2016 model)
       plt_args         dictionary containing optional keyword arguments for customizing plots
       ra_err_deg       1-d array containing values of 'ra_err', the longitude coordinate in equatorial coordinates
       dec_err_deg      1-d array containing values of 'dec_err', the latitude coordinate in equatorial coordinates
       eid              1-d array containing a set of identification numbers
       scattering       1-d array containing values of 'scattering_time_ms'
       pulse_width      1-d array containing the mean of 'pulse_width_ms'
       spectral_index   1-d array containing the mean of 'spectral_index'
       fluence          1-d array containing values of 'fluence'
       bandwidth_high   1-d array containing values of 'bandwidth_high'
       bandwidth_low    1-d array containing values of 'bandwidth_low'
       toa              1-d array containing values of 'toa'
       peak_freq        1-d array containing values of 'peak_frequency'
       aux              n-d array containing values of auxiliary data, e.g. additional dm_gal models
       mocks            (list of) n-d array(s) containing a set of mock catalogs (and binary masks)
       jackknife        int specifying the type of input mocks:
                        0 (randomized RA), 1 (jackknifed), -1 (flipped jackknifed)
       kernelize        If True, then sky positions are kernelized by their corresponding errors.

    In plot_*() methods, the 'doc' argument should either be None (to show a plot interactively),
    or a handout.Handout instance (to show a plot in a web-browsable output directory).

    Two or more unique catalogs can be concatenated as follows:
       fcat12 = fcat1 + fcat2
       fcat1234 = sum([fcat1, fcat2, fcat3, fcat4])
    """

    def __init__(self, snr, dm_obs, ra_deg, dec_deg, dm_gal, plt_args=None, ra_err_deg=None, dec_err_deg=None,
                 eid=None, scattering=None, pulse_width=None, spectral_index=None, fluence=None, bandwidth_high=None,
                 bandwidth_low=None, toa=None, peak_freq=None, aux=None, mocks=None, jackknife=0, kernelize=False):

        if plt_args is None:
            plt_args = {}
        assert snr.size > 0
        assert snr.ndim == 1
        assert dm_obs.shape == snr.shape
        assert ra_deg.shape == snr.shape
        assert dec_deg.shape == snr.shape
        assert dm_gal.shape == snr.shape
        assert jackknife in (-1, 0, 1)
        assert isinstance(kernelize, bool)

        if isinstance(mocks, list):
            assert mocks[1].dtype == bool

        self.snr = snr
        self.dm_obs = dm_obs
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.dm_gal = dm_gal
        self.size = self.snr.size
        self.plt_args = plt_args

        self.ra_err_deg = ra_err_deg
        self.dec_err_deg = dec_err_deg

        if self.ra_err_deg is None or self.dec_err_deg is None:
            self.loc_err_deg = None
        else:
            _ra_err = self.ra_err_deg * np.cos(self.dec_deg * np.pi / 180)
            self.loc_err_deg = (_ra_err**2 + self.dec_err_deg**2)**0.5
            self.loc_err_deg /= 2**0.5

        if kernelize:
            ra_kernel = np.random.normal(0.0, self.ra_err_deg)
            self.ra_deg += ra_kernel

            self.ra_deg = np.mod(self.ra_deg, 360.0)

            dec_kernel = np.random.normal(0.0, self.dec_err_deg)
            self.dec_deg += dec_kernel

            self.dec_deg = np.minimum(self.dec_deg, 90.0)
            self.dec_deg = np.maximum(self.dec_deg, -90.0)

        # Note: calls sanity_check_lon_lat_arrays().
        self.l_deg, self.b_deg = fx.utils.convert_ra_dec_to_l_b(self.ra_deg, self.dec_deg)

        if eid is not None:
            assert len(set(eid)) == eid.size, "frb_catalog has found 'eid' duplicates!"

        self.eid = eid
        self.scattering = scattering
        self.pulse_width = pulse_width
        self.spectral_index = spectral_index
        self.fluence = fluence
        self.bandwidth_high = bandwidth_high
        self.bandwidth_low = bandwidth_low
        self.toa = toa
        self.peak_freq = peak_freq
        self.aux = aux
        self.mocks = mocks
        self.jackknife = jackknife
        self.kernelize = kernelize

    def __add__(self, fcat):
        assert self.jackknife == fcat.jackknife
        assert self.kernelize == fcat.kernelize

        snr = np.append(self.snr, fcat.snr, axis=0)
        dm_obs = np.append(self.dm_obs, fcat.dm_obs, axis=0)
        ra_deg = np.append(self.ra_deg, fcat.ra_deg, axis=0)
        dec_deg = np.append(self.dec_deg, fcat.dec_deg, axis=0)
        dm_gal = np.append(self.dm_gal, fcat.dm_gal, axis=0)

        plt_args = {**fcat.plt_args, **self.plt_args}

        if self.ra_err_deg is not None:
            ra_err_deg = np.append(self.ra_err_deg, fcat.ra_err_deg, axis=0)
        else:
            ra_err_deg = None

        if self.dec_err_deg is not None:
            dec_err_deg = np.append(self.dec_err_deg, fcat.dec_err_deg, axis=0)
        else:
            dec_err_deg = None

        if self.eid is not None:
            eid = np.append(self.eid, fcat.eid, axis=0)
        else:
            eid = None

        if self.scattering is not None:
            scattering = np.append(self.scattering, fcat.scattering, axis=0)
        else:
            scattering = None

        if self.pulse_width is not None:
            pulse_width = np.append(self.pulse_width, fcat.pulse_width, axis=0)
        else:
            pulse_width = None

        if self.spectral_index is not None:
            spectral_index = np.append(self.spectral_index, fcat.spectral_index, axis=0)
        else:
            spectral_index = None

        if self.fluence is not None:
            fluence = np.append(self.fluence, fcat.fluence, axis=0)
        else:
            fluence = None

        if self.bandwidth_high is not None:
            bandwidth_high = np.append(self.bandwidth_high, fcat.bandwidth_high, axis=0)
        else:
            bandwidth_high = None

        if self.bandwidth_low is not None:
            bandwidth_low = np.append(self.bandwidth_low, fcat.bandwidth_low, axis=0)
        else:
            bandwidth_low = None

        if self.toa is not None:
            toa = np.append(self.toa, fcat.toa, axis=0)
        else:
            toa = None

        if self.peak_freq is not None:
            peak_freq = np.append(self.peak_freq, fcat.peak_freq, axis=0)
        else:
            peak_freq = None

        if (self.aux is not None) and (fcat.aux is not None) and (len(self.aux) == len(fcat.aux)):
            aux = np.append(self.aux, fcat.aux, axis=0)
        else:
            aux = None

        if isinstance(self.mocks, np.ndarray):
            mocks = np.append(self.mocks, fcat.mocks, axis=0)
        elif isinstance(self.mocks, list):
            mocks = []
            for i, j in zip(self.mocks, fcat.mocks):
                ij = np.append(i, j, axis=0)
                mocks.append(ij)
        else:
            mocks = None

        return frb_catalog(snr, dm_obs, ra_deg, dec_deg, dm_gal, plt_args, ra_err_deg, dec_err_deg,
                           eid, scattering, pulse_width, spectral_index, fluence, bandwidth_high,
                           bandwidth_low, toa, peak_freq, aux, mocks, self.jackknife, self.kernelize)

    def __radd__(self, fcat):
        if fcat == 0:
            return self
        else:
            return self.__add__(fcat)

    def make_subcatalog(self, mask):
        """
        The 'mask' argument is a boolean array of length self.size, indicating
        which FRBs are in the subcatalog (True), or not in the subcatalog (False).

        Returns an object of type frb_catalog.

        Example usage:

           # Full catalog
           f = frb_catalog_jun23()

           # Subcatalog consisting of objects whose extragalactic DM is >= 150
           f2 = f.make_subcatalog(f.dm_obs >= f.dm_gal + 150.)
        """

        assert mask.shape == (self.size,)
        assert mask.dtype == np.bool
        assert np.sum(mask) > 0

        ra_err_deg = self.ra_err_deg[mask] if self.ra_err_deg is not None else None
        dec_err_deg = self.dec_err_deg[mask] if self.dec_err_deg is not None else None
        eid = self.eid[mask] if self.eid is not None else None
        scattering = self.scattering[mask] if self.scattering is not None else None
        pulse_width = self.pulse_width[mask] if self.pulse_width is not None else None
        spectral_index = self.spectral_index[mask] if self.spectral_index is not None else None
        fluence = self.fluence[mask] if self.fluence is not None else None
        bandwidth_high = self.bandwidth_high[mask] if self.bandwidth_high is not None else None
        bandwidth_low = self.bandwidth_low[mask] if self.bandwidth_low is not None else None
        toa = self.toa[mask] if self.toa is not None else None
        peak_freq = self.peak_freq[mask] if self.peak_freq is not None else None
        aux = self.aux[mask] if self.aux is not None else None

        if self.mocks is None:
            mocks = None
        elif isinstance(self.mocks, np.ndarray):
            mocks = self.mocks[mask]
        elif isinstance(self.mocks, list):
            mocks = self.mocks
        else:
            raise RuntimeError('frb_catalog.make_subcatalog: invalid mock type!')

        return frb_catalog(self.snr[mask], self.dm_obs[mask], self.ra_deg[mask], self.dec_deg[mask],
                           self.dm_gal[mask], ra_err_deg=ra_err_deg, dec_err_deg=dec_err_deg, eid=eid,
                           scattering=scattering, pulse_width=pulse_width, spectral_index=spectral_index,
                           fluence=fluence, bandwidth_high=bandwidth_high, bandwidth_low=bandwidth_low,
                           toa=toa, peak_freq=peak_freq, aux=aux, mocks=mocks, jackknife=self.jackknife,
                           kernelize=self.kernelize)

    def plot_dm_histogram(self, doc=None):
        plt.hist(self.dm_obs-self.dm_gal, bins=20)
        plt.xlabel(r'${\rm DM}$')
        fx.showfig(doc, self.plt_args)

    def plot_dec_histogram(self, doc=None):
        plt.hist(self.dec_deg, bins=20)
        plt.xlabel(r'${\rm Dec~(deg)}$')
        fx.showfig(doc, self.plt_args)

    def plot_ra_histogram(self, doc=None):
        plt.hist(self.ra_deg, bins=20)
        plt.xlabel(r'${\rm RA~(deg)}$')
        fx.showfig(doc, self.plt_args)

    def plot_dec_err_histogram(self, doc=None):
        dec_err = self.dec_err_deg[np.isfinite(self.dec_err_deg)]

        plt.hist(dec_err, bins=20)
        plt.xlabel(r'$\sigma_{\rm Dec}~{\rm (deg)}$')
        fx.showfig(doc, self.plt_args)

    def plot_ra_err_histogram(self, doc=None, cos_dec=False):
        mask = np.isfinite(self.ra_err_deg)
        ra_err = self.ra_err_deg[mask]

        if cos_dec:
            ra_err *= np.cos(self.dec_deg[mask] * np.pi / 180)
            xlabel = r'$\sigma_{\rm RA~cos(Dec)}~{\rm (deg)}$'
        else:
            xlabel = r'$\sigma_{\rm RA}~{\rm (deg)}$'

        plt.hist(ra_err, bins=20)
        plt.xlabel(xlabel)
        fx.showfig(doc, self.plt_args)

    def plot_loc_err_deg(self, doc=None):
        mask = np.isfinite(self.loc_err_deg)
        plt.hist(self.loc_err_deg[mask], bins=20)
        plt.xlabel(r'${\rm Location~error~(deg)}$')
        fx.showfig(doc, self.plt_args)

    def plot_snr_histogram(self, doc=None):
        snr = self.snr[np.isfinite(self.snr)]
        plt.hist(snr, bins=20)
        plt.xlabel(r'${\rm SNR}$')
        fx.showfig(doc, self.plt_args)

    def plot_scattering_histogram(self, doc=None):
        scattering = self.scattering[np.isfinite(self.scattering)]
        plt.hist(scattering, bins=20)
        plt.xlabel(r'${\rm Scattering~(ms)}$')
        fx.showfig(doc, self.plt_args)

    def plot_pulse_width_histogram(self, doc=None):
        pulse_width = self.pulse_width[np.isfinite(self.pulse_width)]
        plt.hist(pulse_width, bins=20)
        plt.xlabel(r'${\rm Pulse~width~(ms)}$')
        fx.showfig(doc, self.plt_args)

    def plot_spectral_index_histogram(self, doc=None):
        spectral_index = self.spectral_index[np.isfinite(self.spectral_index)]
        plt.hist(spectral_index, bins=20)
        plt.xlabel(r'${\rm Spectral~index}$')
        fx.showfig(doc, self.plt_args)

    def plot_fluence_histogram(self, doc=None):
        fluence = self.fluence[np.isfinite(self.fluence)]
        plt.hist(fluence, bins=20)
        plt.xlabel(r'${\rm Fluence}$')
        fx.showfig(doc, self.plt_args)

    def plot_bandwidth_histogram(self, doc=None):
        mask = np.logical_and(np.isfinite(self.bandwidth_high), np.isfinite(self.bandwidth_low))
        bandwidth = self.bandwidth_high[mask] - self.bandwidth_low[mask]
        plt.hist(bandwidth, bins=20)
        plt.xlabel(r'${\rm Bandwidth}$')
        fx.showfig(doc, self.plt_args)

    def plot_histogram(self, arr, xlabel, bins=20, doc=None, label=None):
        plt.hist(arr, bins=bins, label=label)
        plt.xlabel(xlabel)
        if label is not None:
            plt.legend()
        fx.showfig(doc, self.plt_args)

    def plot_ra_dec(self, doc=None):
        ra = self.ra_deg[np.isfinite(self.ra_deg)]
        dec = self.dec_deg[np.isfinite(self.dec_deg)]
        assert ra.size == dec.size

        plt.scatter(ra, dec)
        plt.xlabel(r'${\rm RA~(deg)}$')
        plt.ylabel(r'${\rm Dec~(deg)}$')
        plt.xlim(0.0, 360.0)
        plt.ylim(-90.0, 90.0)
        fx.showfig(doc, self.plt_args)

    def plot_l_b(self, doc=None):
        plt.scatter(self.l_deg, self.b_deg)
        plt.xlabel(r'$l~{\rm (deg)}$')
        plt.ylabel(r'$b~{\rm (deg)}$')
        plt.xlim(0.0, 360.0)
        plt.ylim(-90.0, 90.0)
        fx.showfig(doc, self.plt_args)

    def plot_healpix_map(self, doc=None, nside=64):
        m = fx.utils.make_healpix_map_from_catalog(nside, self.l_deg, self.b_deg)
        fx.utils.show_healpix_map(m, doc)


def frb_catalog_mocks(n, nmc, path, jackknife=0):
    """
    Writes a set of mock FRB catalogs to disk.  If 'jackknife' is 0 (default), then it saves
    a 2-d array of shape (n,nmc), containing randomized RA values.  The 'jackknife' arg
    specifies the type of mocks: 0 (randomized RA), 1 (jackknifed), -1 (flipped jackknifed),
    where the last two cases result in binary arrays for masking an FRB catalog of length n.
    """

    assert isinstance(n, int)
    assert isinstance(nmc, int)
    assert isinstance(path, str)
    assert jackknife in (-1, 0, 1)

    try:
        ret = fx.read_arr(path)
        print(f'frb_catalog_mocks: a catalog already exists in {path}.')

        if (ret.shape[0] != n) or (ret.shape[-1] < nmc):
            raise RuntimeError(f'frb_catalog_mocks: invalid (n,nmc)={(n,nmc)} parameters; expected {ret.shape}.')

        if jackknife:
            if ret.dtype != bool:
                raise RuntimeError('frb_catalog_mocks: saved catalog is not jackknifed!')
        else:
            if not np.logical_and((ret >= 0.0), (ret <= 360.0)).all():
                raise RuntimeError('frb_catalog_mocks: saved catalog has out-of-bound RA values!')

        if jackknife == -1:
            ret = ~ret

        return ret
    except (IOError, AssertionError) as err:
        print(err)

        if jackknife == 0:
            ret = np.random.uniform(0.0, 360.0, size=(n,nmc))
        elif jackknife == 1:
            ret = np.random.randint(2, size=(n,nmc), dtype=bool)
        else:
            raise RuntimeError('frb_catalog_mocks: flipped jackknife is reserved for saved catalogs.')

        fx.write_arr(path, ret)
        print(f'frb_catalog_mocks wrote catalogs to {path}.')

        return ret


def frb_catalog_may7(filename=fx.data_path('archive/catalogs/chime_frb/CHIME_may7_all.txt')):
    """Returns the "May 7" CHIME FRB catalog made by Dongzi."""

    a = np.loadtxt(filename)
    print(f'read {filename}')

    # Column ordering is (snr, dm_obs, ra, dec, dm_gal).
    assert a.shape == (435, 5)
    return frb_catalog(snr=a[:,0], dm_obs=a[:,1], ra_deg=a[:,2], dec_deg=a[:,3], dm_gal=a[:,4])


def frb_catalog_jun23(filename=fx.data_path('archive/catalogs/chime_frb/chime_cat_Jun23_DMga.txt')):
    """Returns the "June 23" CHIME FRB catalog (full version) made by Dongzi."""

    a = np.loadtxt(filename)
    print(f'read {filename}')

    # Column ordering is (snr, dm_obs, ra, dec, dm_gal, width).
    # We currently don't use the width column.
    assert a.shape == (502, 6)
    return frb_catalog(snr=a[:,0], dm_obs=a[:,1], ra_deg=a[:,2], dec_deg=a[:,3], dm_gal=a[:,4])


def frb_catalog_jun23_non_repeaters(
        filename=fx.data_path('archive/catalogs/chime_frb/chime_cat_Jun23_noRepeater_DMga.txt')):
    """Returns the "June 23" CHIME FRB catalog, selecting only non-repeaters.  Made by Dongzi."""

    a = np.loadtxt(filename)
    print(f'read {filename}')

    # Column ordering is (snr, dm_obs, ra, dec, dm_gal, width).
    # We currently don't use the width column.
    assert a.shape == (459, 6)
    return frb_catalog(snr=a[:,0], dm_obs=a[:,1], ra_deg=a[:,2], dec_deg=a[:,3], dm_gal=a[:,4])


def frb_catalog_jun23_meridian(filename=fx.data_path('archive/catalogs/chime_frb/chime_cat_Jun23_beam1_DMga.txt')):
    """Returns the "June 23" CHIME FRB catalog, selecting only meridian events.  Made by Dongzi."""

    a = np.loadtxt(filename)
    print(f'read {filename}')

    # Column ordering is (snr, dm_obs, ra, dec, dm_gal, width).
    # We currently don't use the width column.
    assert a.shape == (129, 6)
    return frb_catalog(snr=a[:,0], dm_obs=a[:,1], ra_deg=a[:,2], dec_deg=a[:,3], dm_gal=a[:,4])


def frb_catalog_baseband():
    """Returns updated catalog of CHIME FRB baseband events.  Originally written by Chitrang Patel."""

    address = "https://frb.chimenet.ca/frb-master/v1/verification/get-verifications/NEW%20SOURCE"

    _, auth = fx.utils.ch_frb_master()
    events = requests.get(address, headers=auth)

    ret = [ ]
    for event in events.json():
        if event['data_status']['baseband_data']:
            ret.append(event['event_id'])

    return ret


def frb_catalog_new_candidates():
    """Returns updated catalog of CHIME FRB events."""

    address = "https://frb.chimenet.ca/frb-master/v1/verification/get-verifications/New%20CANDIDATE"

    _, auth = fx.utils.ch_frb_master()
    events = requests.get(address, headers=auth)

    _ret = [ ]
    for event in events.json():
        _ret.append([event['snr'],
                     event['dm'],
                     event['ra'],
                     event['dec'],
                     event['event_id']])

    _ret = np.asarray(_ret)

    gdm = fx.gal_dm()
    dm_gal = gdm(_ret[:,2], _ret[:,3])

    ret = frb_catalog(snr=_ret[:,0], dm_obs=_ret[:,1], ra_deg=_ret[:,2], dec_deg=_ret[:,3], dm_gal=dm_gal, eid=_ret[:,4])

    return ret


def frb_catalog_csv(filename=fx.data_path('archive/catalogs/chime_frb/baseband_catalog.csv'), plt_args=None):
    """
    Returns a CHIME FRB catalog based on events in a csv file with the following columns:
        [ eid, _, _, dm_obs, snr, ra_deg, ra_err_deg, dec, dec_err_deg, _ ]
    """

    if plt_args is None:
        plt_args = {}
    gdm = fx.gal_dm()

    a = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)

        for r in reader:
            if r[0].startswith('#'):
                continue

            eid = int(r[0])
            snr = float(r[4])
            dm_obs = float(r[3])
            ra_deg = float(r[5])
            dec_deg = float(r[7])
            ra_err_deg = float(r[6])
            dec_err_deg = float(r[8])

            dm_gal = gdm(ra_deg, dec_deg)
            a.append([eid, snr, dm_obs, ra_deg, dec_deg, dm_gal, ra_err_deg, dec_err_deg])

    a = np.asarray(a)
    fx.utils.assert_eid_unique(a[:,0])

    ret = frb_catalog(snr=a[:,1], dm_obs=a[:,2], ra_deg=a[:,3], dec_deg=a[:,4], dm_gal=a[:,5],
                      plt_args=plt_args, ra_err_deg=a[:,6], dec_err_deg=a[:,7], eid=a[:,0])

    return ret


def frb_catalog_dynamic(filename):
    """Returns a CHIME FRB catalog based on events processed by '../scripts/chime_events.py'."""

    a = np.loadtxt(filename)
    print(f'read {filename}')

    # Column ordering is
    #   dec(deg), dec_error(deg),
    #   ra(deg), ra_error(deg),
    #   snr, dm, dm_error, galactic_dm_realtime, galactic_dm,
    #   width(s), width_error(s),
    #   fluence(Jy-ms), fluence_error(Jy-ms),
    #   flux(Jy), flux_error(Jy).

    assert a.shape[-1] == 15
    mask = np.logical_and((a[:,0] != -99), (a[:,2] != -99))
    a = a[mask]

    return frb_catalog(snr=a[:,4], dm_obs=a[:,5], ra_deg=a[:,2], dec_deg=a[:,0], dm_gal=a[:,8])


def frb_catalog_cs(filename=fx.data_path('archive/catalogs/chime_frb/cs_011822.csv'),
                   mocks=fx.data_path('archive/catalogs/chime_frb/mocks_cs_011822_0.8.npy'),
                   plt_args=None, nmc=1000000, jackknife=0, t0_astro_fraction_thresh=0.8):
    """
    Returns a CHIME FRB Citizen Science catalog based on a CSV file with the following columns:
        ---
        0. event
        1. beam
        2. signal_to_noise
        3. dispersion
        4. subfolder
        5. absolute_loc
        6. plot_loc
        7. ml_prediction
        8. retired
        9. subject_id
        10. t0_rfi
        11. t0_blank
        12. t0_astro
        13. t0_cant-answer
        14. t1_overlapping
        15. t1_repeating
        16. t1_something-weird
        17. t0_total
        18. t1_total
        19. t0_astro_fraction
        20. t0_rfi_fraction
        21. t0_blank_fraction
        22. t0_cant-answer_fraction
        23. t1_overlapping_fraction
        24. t1_repeating_fraction
        25. t1_something-weird_fraction
        26. signal_to_noise_rounded
        27. deg_ew_from_meridian
        28. deg_nw_from_zenith
        ---
    """

    _path = filename.split('.')
    pickle_path = _path[0] + f'_{t0_astro_fraction_thresh}' + '.pkl'

    try:
        ret = fx.read_pickle(pickle_path)
        return ret
    except IOError as err:
        print(err)
        pass

    if plt_args is None:
        plt_args = {}

    gdm = fx.gal_dm()
    master, _ = fx.utils.ch_frb_master()

    a = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for event in reader:
            if reader.line_num == 1:
                continue
            else:
                print(f'frb_catalog_cs: parsing index {reader.line_num}')

            t0_astro_fraction = float(event[19])
            if t0_astro_fraction < t0_astro_fraction_thresh:
                continue

            eid = int(event[0])

            _event = master.events.get_event(eid)
            _event = _event['measured_parameters'][0]

            snr = float(event[2])
            dm_obs = float(event[3])
            ra_deg = float(_event['ra'])
            dec_deg = float(_event['dec'])
            ra_err_deg = np.nan
            dec_err_deg = np.nan

            dm_gal = gdm(ra_deg, dec_deg)

            scattering = np.nan
            pulse_width = np.nan
            spectral_index = np.nan

            fluence = np.nan
            bandwidth_high = np.nan
            bandwidth_low = np.nan

            toa = np.nan
            peak_freq = np.nan

            a.append([eid, snr, dm_obs, ra_deg, dec_deg, dm_gal, ra_err_deg, dec_err_deg,
                      scattering, pulse_width, spectral_index, fluence, bandwidth_high, bandwidth_low,
                      toa, peak_freq, t0_astro_fraction])

    a = np.asarray(a)
    fx.utils.assert_eid_unique(a[:,0])

    m = frb_catalog_mocks(n=a.shape[0], nmc=nmc, path=mocks, jackknife=jackknife) if (mocks is not None) else None
    if (mocks is not None) and jackknife:
        m = [a[:,3:5], m]

    ret = frb_catalog(snr=a[:,1], dm_obs=a[:,2], ra_deg=a[:,3], dec_deg=a[:,4], dm_gal=a[:,5],
                      plt_args=plt_args, ra_err_deg=a[:,6], dec_err_deg=a[:,7], eid=a[:,0],
                      scattering=a[:,8], pulse_width=a[:,9], spectral_index=a[:,10], fluence=a[:,11],
                      bandwidth_high=a[:,12], bandwidth_low=a[:,13], toa=a[:,14], peak_freq=a[:,15],
                      aux=a[:,16:], mocks=m, jackknife=jackknife)

    print(f'frb_catalog_cs: Nf = {len(a)}')

    fx.write_pickle(pickle_path, ret)

    return ret


def frb_catalog_rn3(filename=fx.data_path('archive/catalogs/chime_frb/rn3_sources_preliminary_all.npy'),
                    mocks=fx.data_path('archive/catalogs/chime_frb/mocks_rn3_121521.npy'),
                    plt_args=None, nmc=1000000, jackknife=0):
    """Returns a CHIME FRB RN3 catalog."""

    if plt_args is None:
        plt_args = {}

    d = fx.read_arr(filename, allow_pickle=True)

    gdm = fx.gal_dm()
    a = []

    for i, event in enumerate(d):
        eid = int(event[0])
        snr = np.nan
        dm_obs = float(event[3])
        ra_deg = float(event[1])
        dec_deg = float(event[2])
        ra_err_deg = np.nan
        dec_err_deg = np.nan
        pcc = float(event[6])

        baseband_loc = bool(event[5])

        dm_gal = gdm(ra_deg, dec_deg)

        scattering = np.nan
        pulse_width = np.nan
        spectral_index = np.nan

        fluence = np.nan
        bandwidth_high = np.nan
        bandwidth_low = np.nan

        toa = np.nan
        peak_freq = np.nan

        a.append([eid, snr, dm_obs, ra_deg, dec_deg, dm_gal, ra_err_deg, dec_err_deg,
                  scattering, pulse_width, spectral_index, fluence, bandwidth_high, bandwidth_low,
                  toa, peak_freq, baseband_loc, pcc])

    a = np.asarray(a)
    fx.utils.assert_eid_unique(a[:,0])

    m = frb_catalog_mocks(n=a.shape[0], nmc=nmc, path=mocks, jackknife=jackknife) if (mocks is not None) else None
    if (mocks is not None) and jackknife:
        m = [a[:,3:5], m]

    ret = frb_catalog(snr=a[:,1], dm_obs=a[:,2], ra_deg=a[:,3], dec_deg=a[:,4], dm_gal=a[:,5],
                      plt_args=plt_args, ra_err_deg=a[:,6], dec_err_deg=a[:,7], eid=a[:,0],
                      scattering=a[:,8], pulse_width=a[:,9], spectral_index=a[:,10], fluence=a[:,11],
                      bandwidth_high=a[:,12], bandwidth_low=a[:,13], toa=a[:,14], peak_freq=a[:,15],
                      aux=a[:,16:], mocks=m, jackknife=jackknife)

    return ret


def frb_catalog_published_repeaters(filename=fx.data_path('archive/catalogs/chime_frb/repeaters_010122.json'),
                                    mocks=fx.data_path('archive/catalogs/chime_frb/mocks_repeaters_010122.npy'),
                                    plt_args=None, nmc=1000000, jackknife=0):
    """Returns a CHIME FRB catalog based on the most precise data on published repeaters."""

    if plt_args is None:
        plt_args = {}
    with open(filename, 'r') as f:
        d = json.load(f)

    gdm = fx.gal_dm()
    a = []

    for eid, i in enumerate(d):
        print(eid, i)

        dm_obs = []
        for j in d[i]:
            try:
                dm_obs.append(d[i][j]['dm']['value'])
            except KeyError:
                pass
        dm_obs = np.mean(dm_obs)

        ra = d[i]['ra']['value'].split(':')
        ra_deg = float(ra[0]) + (float(ra[1]) / 60.)
        ra_deg *= 15

        dec = d[i]['dec']['value'].split(':')
        dec_deg = float(dec[0]) + (float(dec[1]) / 60.)

        dm_gal = gdm(ra_deg, dec_deg)

        snr = []
        for j in d[i]:
            try:
                snr.append(d[i][j]['snr']['value'])
            except KeyError:
                pass
        snr = np.mean(snr)

        ra_err_deg = np.nan
        dec_err_deg = np.nan

        scattering = []
        for j in d[i]:
            try:
                try:
                    x = float(d[i][j]['scattering_time']['value'])
                except ValueError:
                    x = float(d[i][j]['scattering_time']['value'][1:])
                scattering.append(x)
            except (KeyError, TypeError):
                pass

        pulse_width = []
        for j in d[i]:
            try:
                try:
                    x = float(d[i][j]['width']['value'])
                except ValueError:
                    x = float(d[i][j]['width']['value'][1:])
                pulse_width.append(x)
            except (KeyError, TypeError):
                pass

        fluence = []
        for j in d[i]:
            try:
                try:
                    x = float(d[i][j]['fluence']['value'])
                except ValueError:
                    x = float(d[i][j]['fluence']['value'][1:])
                fluence.append(x)
            except (KeyError, TypeError):
                pass

        scattering = np.mean(scattering)
        pulse_width = np.mean(pulse_width)
        spectral_index = np.nan

        fluence = np.mean(fluence)
        bandwidth_high = np.nan
        bandwidth_low = np.nan

        toa = np.nan
        peak_freq = np.nan

        a.append([eid, snr, dm_obs, ra_deg, dec_deg, dm_gal, ra_err_deg, dec_err_deg,
                  scattering, pulse_width, spectral_index, fluence, bandwidth_high, bandwidth_low,
                  toa, peak_freq])

    a = np.asarray(a)
    fx.utils.assert_eid_unique(a[:,0])

    m = frb_catalog_mocks(n=a.shape[0], nmc=nmc, path=mocks, jackknife=jackknife) if (mocks is not None) else None
    if (mocks is not None) and jackknife:
        m = [a[:,3:5], m]

    ret = frb_catalog(snr=a[:,1], dm_obs=a[:,2], ra_deg=a[:,3], dec_deg=a[:,4], dm_gal=a[:,5],
                      plt_args=plt_args, ra_err_deg=a[:,6], dec_err_deg=a[:,7], eid=a[:,0],
                      scattering=a[:,8], pulse_width=a[:,9], spectral_index=a[:,10], fluence=a[:,11],
                      bandwidth_high=a[:,12], bandwidth_low=a[:,13], toa=a[:,14], peak_freq=a[:,15],
                      mocks=m, jackknife=jackknife)

    return ret


def frb_catalog_json(filename, morphology_in=None, morphology_ex=None, single_burst=False,
                     flagged=fx.data_path('archive/catalogs/chime_frb/ignore_103120.npy'),
                     flagged_tns=fx.data_path('archive/catalogs/chime_frb/ignore_tns_112120.txt'),
                     mocks=fx.data_path('archive/catalogs/chime_frb/mocks_c112120_i103120_it112120.npy'),
                     id_nbeam=fx.data_path('archive/catalogs/chime_frb/id_nbeam.npy'),
                     nmc=1000000, plt_args=None, jackknife=0, kernelize=False):
    """Returns a CHIME FRB catalog based on events in a json file."""

    if plt_args is None:
        plt_args = {}

    def list_to_arr(x):
        _ret = []
        for e in x:
            try:
                _ret.append(float(e))
            except ValueError as _err:
                _ret.append(float(e[1:]))
        return np.array(_ret)

    with open(filename, 'r') as f:
        d = json.load(f)

    if flagged is not None:
        if isinstance(flagged, str):
            _flagged = np.load(flagged, allow_pickle=True)

            flagged = []
            for i in _flagged:
                if isinstance(i, (int, float, np.int64)):
                    flagged.append(int(i))
                else:
                    for j in i:
                        flagged.append(int(j))

        flagged = np.asarray(flagged)

    if flagged_tns is not None:
        flagged_tns = np.loadtxt(flagged_tns, dtype=str)

    gdm = fx.gal_dm()

    if id_nbeam is not None:
        x = np.load(id_nbeam)
        nbeams = dict(zip(x[0,:],x[1,:]))
    else:
        nbeams = None

    a = []
    nrepeaters = 0
    nflagged = 0
    eid_flagged = []
    tns_flagged = []
    invalid_coordinates = 0
    invalid_morphology_in = 0
    invalid_morphology_ex = 0
    invalid_dm = 0
    invalid_snr = 0
    invalid_scattering = 0
    invalid_pulse_width = 0
    invalid_spectral_index = 0
    invalid_fluence = 0
    invalid_bandwidth = 0
    invalid_toa = 0
    invalid_peak_freq = 0
    invalid_nbeam = 0

    r = len(d)
    for i in range(r):
        print(f'frb_catalog_json: {i}/{r}')

        try:
            eid = int(d[i]['event_id'])
        except KeyError:
            eid = int(d[i]['event_number'])

        if single_burst:
            try:
                repeater_of = d[i]['repeater_of']
                assert repeater_of == ''
            except AssertionError:
                nrepeaters += 1
                continue

        if flagged is not None:
            try:
                assert eid not in flagged
            except AssertionError:
                nflagged += 1
                eid_flagged.append(eid)
                continue

        if flagged_tns is not None:
            try:
                tns_name = str(d[i]['tns_name'])
                assert tns_name not in flagged_tns
            except AssertionError:
                nflagged += 1
                tns_flagged.append(tns_name)
                continue

        try:
            ra_deg = np.mean(d[i]['ra'])
            dec_deg = np.mean(d[i]['dec'])
            ra_err_deg = np.max(np.abs(d[i]['ra_error']))
            dec_err_deg = np.max(np.abs(d[i]['dec_error']))
        except KeyError:
            invalid_coordinates += 1
            continue

        if morphology_in is not None:
            try:
                assert morphology_in in d[i]['morphology']
            except (KeyError, AssertionError):
                invalid_morphology_in += 1
                continue

        if morphology_ex is not None:
            try:
                assert morphology_ex not in d[i]['morphology']
            except (KeyError, AssertionError):
                invalid_morphology_ex += 1
                continue

        for j in ('fitburst_dm', 'dm_structure', 'dm_snr', 'dm'):
            try:
                dm_obs = float(d[i][j])
                break
            except KeyError:
                dm_obs = None
                continue

        try:
            assert dm_obs is not None
            assert dm_obs >= 0.0
        except AssertionError:
            invalid_dm += 1
            continue

        for j in ('fitburst_snr', 'intensity_snr', 'bonsai_snr'):
            try:
                snr = float(d[i][j])
                break
            except KeyError:
                snr = None
                continue

        try:
            assert snr is not None
            assert snr >= 0.0
        except AssertionError:
            invalid_snr += 1
            continue

        _dm_gal_fx = gdm(ra_deg, dec_deg, mode='ymw16')
        _dm_gal_pygedm = gdm(ra_deg, dec_deg, mode=f'pygedm_ymw16')

        try:
            assert np.isclose(_dm_gal_fx, _dm_gal_pygedm, rtol=0.2, atol=1.0e-7),\
                f'frb_catalog_json: (eid, ra, dec, _dm_gal_fx, _dm_gal_pygedm)' \
                f'=({eid}, {ra_deg}, {dec_deg}, {_dm_gal_fx}, {_dm_gal_pygedm})'
        except AssertionError as err:
            print(err)

        try:
            dm_excess = np.mean(d[i][f'dm_excess_ymw16'])
            _dm_gal = dm_obs - dm_excess
            assert np.isclose(_dm_gal, _dm_gal_pygedm, rtol=0.2, atol=1.0e-7),\
                    f'frb_catalog_json: (eid, ra, dec, _dm_gal, _dm_gal_fx)' \
                    f'=({eid}, {ra_deg}, {dec_deg}, {_dm_gal}, {_dm_gal_fx})'
        except (KeyError, AssertionError) as err:
            print(err)

        # Catalog-based _dm_gal is based on the initial header-based localization and may not correspond
        # to the actual galactic contribution at a post-processed sky location.  So here we overwrite
        # the param by its value at the post-processed location.
        dm_gal = _dm_gal_pygedm
        dm_gal_ne01 = gdm(ra_deg, dec_deg, mode='pygedm_ne2001')
        dm_halo_yt20 = gdm(ra_deg, dec_deg, mode='pygedm_yt2020')

        try:
            scattering = d[i]['scattering_time_ms']

            if isinstance(scattering, list):
                scattering = np.mean(scattering)

            try:
                scattering = float(scattering)
            except ValueError:
                scattering = float(scattering[1:])
        except KeyError:
            invalid_scattering += 1
            continue

        try:
            pulse_width = list_to_arr(d[i]['pulse_width_ms']).mean()
        except KeyError:
            invalid_pulse_width += 1
            continue
        except ValueError:
            pulse_width = np.nan

        try:
            spectral_index = list_to_arr(d[i]['spectral_index']).mean()
        except KeyError:
            invalid_spectral_index += 1
            continue
        except ValueError:
            spectral_index = np.nan

        try:
            fluence = float(d[i]['fluence'])
        except KeyError:
            invalid_fluence += 1
            continue
        except ValueError:
            fluence = np.nan

        try:
            bandwidth_high = list_to_arr(d[i]['bandwidth_high']).max()
            bandwidth_low = list_to_arr(d[i]['bandwidth_low']).min()
        except KeyError:
            invalid_bandwidth += 1
            continue
        except ValueError:
            bandwidth_high = np.nan
            bandwidth_low = np.nan

        try:
            toa = d[i]['toa']
            if isinstance(toa, list):
                toa = list_to_arr(d[i]['toa']).min()
            else:
                toa = float(toa)
        except KeyError:
            invalid_toa += 1
            continue
        except ValueError:
            toa = np.nan

        try:
            peak_freq = d[i]['peak_frequency']
            if isinstance(peak_freq, list):
                peak_freq = list_to_arr(d[i]['peak_frequency']).min()
            else:
                peak_freq = float(peak_freq)
        except KeyError:
            invalid_peak_freq += 1
            continue
        except ValueError:
            peak_freq = np.nan

        if nbeams is not None:
            try:
                nbeam = nbeams[eid]
            except KeyError:
                invalid_nbeam += 1
                continue
        else:
            nbeam = 1

        a.append([eid, snr, dm_obs, ra_deg, dec_deg, dm_gal, ra_err_deg, dec_err_deg, scattering,
                  pulse_width, spectral_index, fluence, bandwidth_high, bandwidth_low, toa, peak_freq,
                  dm_gal_ne01, dm_halo_yt20, nbeam])

    print(f"frb_catalog_json: {len(a)}/{r} FRBs have passed the following validation process.")
    print(f"frb_catalog_json: {nrepeaters} FRBs have been repeating bursts,")
    print(f"frb_catalog_json: {nflagged} FRBs have been flagged,")
    print(f"frb_catalog_json: {invalid_coordinates} FRBs haven't been localized,")
    print(f"frb_catalog_json: {invalid_morphology_in} FRBs don't match to {morphology_in} morphology,")
    print(f"frb_catalog_json: {invalid_morphology_ex} FRBs match to {morphology_ex} morphology,")
    print(f"frb_catalog_json: {invalid_dm} FRBs don't have a valid DM,")
    print(f"frb_catalog_json: {invalid_snr} FRBs don't have a valid SNR,")
    print(f"frb_catalog_json: {invalid_scattering} FRBs don't have a valid scattering,")
    print(f"frb_catalog_json: {invalid_pulse_width} FRBs don't have a valid pulse width,")
    print(f"frb_catalog_json: {invalid_spectral_index} FRBs don't have a valid spectral index,")
    print(f"frb_catalog_json: {invalid_fluence} FRBs don't have a valid fluence,")
    print(f"frb_catalog_json: {invalid_bandwidth} FRBs don't have a valid bandwidth,")
    print(f"frb_catalog_json: {invalid_toa} FRBs don't have a valid TOA,")
    print(f"frb_catalog_json: {invalid_peak_freq} FRBs don't have a valid peak_freq,")
    print(f"frb_catalog_json: and finally, {invalid_nbeam} FRBs don't have a valid nbeam.")
    print(f"frb_catalog_json: eid_flagged={eid_flagged}.")
    print(f"frb_catalog_json: tns_flagged={tns_flagged}.")

    a = np.asarray(a)
    fx.utils.assert_eid_unique(a[:,0])

    m = frb_catalog_mocks(n=a.shape[0], nmc=nmc, path=mocks, jackknife=jackknife) if (mocks is not None) else None
    if (mocks is not None) and jackknife:
        m = [a[:,3:5], m]

    ret = frb_catalog(snr=a[:,1], dm_obs=a[:,2], ra_deg=a[:,3], dec_deg=a[:,4], dm_gal=a[:,5], plt_args=plt_args,
                      ra_err_deg=a[:,6], dec_err_deg=a[:,7], eid=a[:,0], scattering=a[:,8], pulse_width=a[:,9],
                      spectral_index=a[:,10], fluence=a[:,11], bandwidth_high=a[:,12], bandwidth_low=a[:,13],
                      toa=a[:,14], peak_freq=a[:,15], aux=a[:,16:], mocks=m, jackknife=jackknife, kernelize=kernelize)
    return ret


def frb_catalog_rn3_json():
    return frb_catalog_json(filename=fx.data_path('archive/catalogs/chime_frb/rn3_sources_083122.json'),
                            morphology_in=None, morphology_ex=None, single_burst=True,
                            flagged=fx.data_path('archive/catalogs/chime_frb/ignore_103120.npy'),
                            flagged_tns=fx.data_path('archive/catalogs/chime_frb/ignore_tns_112120.txt'),
                            mocks=fx.data_path('archive/catalogs/chime_frb/mocks_rn3_083122_i103120_it112120.npy'),
                            id_nbeam=None, nmc=1000000, plt_args=None, jackknife=0, kernelize=False)


def frb_host_catalog(filename=fx.data_path('archive/catalogs/frb_hosts/fh_040221.csv')):
    """Returns the FRB host catalog based on https://frbhosts.org."""

    gdm = fx.gal_dm()

    cat = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if reader.line_num == 1:
                continue

            name = str(r[0])
            ra_deg = float(r[1])
            dec_deg = float(r[2])
            dm = float(r[3])
            z = float(r[4])

            dm_ymw16 = gdm(ra_deg, dec_deg, mode=f'pygedm_ymw16')
            dm_ne01 = gdm(ra_deg, dec_deg, mode=f'pygedm_ne2001')

            cat.append([name, ra_deg, dec_deg, dm, dm_ymw16, dm_ne01, z])

    print(f'frb_host_catalog: Nf = {len(cat)}')

    return cat


def frb_catalog_rn12(filename=fx.data_path('archive/catalogs/chime_frb/rn12_sources_v2_062022.csv'),
                     mocks=fx.data_path('archive/catalogs/chime_frb/mocks_rn12_sources_v2_062022.npy'),
                     plt_args=None, nmc=1000000, jackknife=0, kernelize=False):
    """Returns the RN12 catalog, containing baseband localizations of repeating sources."""

    if plt_args is None:
        plt_args = {}

    gdm = fx.gal_dm()
    master, _ = fx.utils.ch_frb_master()

    cat = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for r in reader:
            if reader.line_num == 1:
                continue

            eid = int(r[8])
            _event = master.events.get_event(eid)
            _event = _event['measured_parameters'][0]

            name = re.split('(\d+)', str(r[0]))
            name = int(name[1])

            new_name = re.split('(\d+)', str(r[1]))
            new_name = int(new_name[1])

            old_name = str(r[2])
            ra_deg = float(r[4])
            ra_err_deg = float(r[5])
            dec_deg = float(r[6])
            dec_err_deg = float(r[7])

            dm_ymw16 = gdm(ra_deg, dec_deg, mode=f'pygedm_ymw16')
            dm_ne01 = gdm(ra_deg, dec_deg, mode=f'pygedm_ne2001')

            dm = float(_event['dm'])
            snr = np.nan
            scattering = np.nan
            pulse_width = np.nan
            spectral_index = np.nan
            fluence = np.nan
            bandwidth_high = np.nan
            bandwidth_low = np.nan
            toa = np.nan
            peak_freq = np.nan

            z = np.nan

            cat.append([name, new_name, ra_deg, dec_deg, ra_err_deg, dec_err_deg, snr, dm, dm_ymw16, dm_ne01, scattering,
                        pulse_width, spectral_index, fluence, bandwidth_high, bandwidth_low, toa, peak_freq, z])

    print(f'frb_catalog_rn12: Nf = {len(cat)}')

    a = np.asarray(cat)
    fx.utils.assert_eid_unique(a[:,0])

    m = frb_catalog_mocks(n=a.shape[0], nmc=nmc, path=mocks, jackknife=jackknife) if (mocks is not None) else None
    if (mocks is not None) and jackknife:
        m = [a[:,2:4], m]

    ret = frb_catalog(snr=a[:,6], dm_obs=a[:,7], ra_deg=a[:,2], dec_deg=a[:,3], dm_gal=a[:,8], plt_args=plt_args,
                      ra_err_deg=a[:,4], dec_err_deg=a[:,5], eid=a[:,0], scattering=a[:,10], pulse_width=a[:,11],
                      spectral_index=a[:,12], fluence=a[:,13], bandwidth_high=a[:,14], bandwidth_low=a[:,15],
                      toa=a[:,16], peak_freq=a[:,17], aux=a[:,18:], mocks=m, jackknife=jackknife, kernelize=kernelize)
    return ret
