import os
import warnings
import numpy as np
import numpy.random as rand
import scipy.optimize
import astropy.units as u
from h5py import File as FileH5
import matplotlib.pyplot as plt
import frbx_theory as ft
import frbx as fx
from frbx.configs import eps


class simcat:
    """
    This class can be used to generate a series of pseudo 3-d simulations of halos, galaxies, and FRBs.

    Implicit units:

        Mass:                   M_sun/h.
        Distance:               Mpc/h (comoving).
        Spatial wavenumber:     h/Mpc (comoving).
        Dispersion Measure:     pc/cm^3 (comoving).

    Members:

        self.ftc: (obj) instance of fx.cosmology, which is recommended to be pickled in advance.
        self.pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
        self.debug: (bool) enables an interactive mode for debugging.
        self.config: (obj) points to self.ftc.config, which is an instance of fx.configs.
        self.m_f: (float) min halo mass for hosting FRBs.  FIXME: See below.
        self.r_nfw_interp: (dict) interpolator for computing radial coordinates in the NFW halo profile.
        self.zmin: (float) min redshift for simulating sources.
        self.zmax: (float) max redshift for simulating sources.
        self.zz: (list) edges of redshift shells.
        self.hmb: (dict) halo-related stats containing mass bins, expected counts, and halo_mass_function objects.
        simcat.set_rng: (static method) sets the initial RNG seed.
        self.simulate: (method) generates and writes random catalogs of galaxies and FRBs.
        self._extract_random_intrahalo_source: (helper method) selects random intrahalo sources from a saved catalog.
        simcat.unique_random_halo_index: (static method) generates unique random halo indices
                                         for binned intrahalo sources.
        self._generate_halo: (helper method) generates randomly distributed halos in a redshift shell.
        self._generate_halo_catalog: (helper method) generates a catalog of randomly distributed halos in a
                                     redshift shell.
        self._generate_intrahalo_catalog: (helper method) generates a catalog of galaxies or FRBs in a redshift shell.
        self._mpc_to_angle: (helper method) converts transverse comoving distance (in Mpc/h) to
                            an angle in self.config.sim.unit.
        self._generate_nfw: (helper method) generates random 3-d positions in the NFW halo profile.
        self._r_nfw: (helper method) randomly samples radial positions in the NFW halo profile.
        simcat.nfw_prof: (static method) logarithmic part of the NFW characteristic overdensity.
        simcat._r_nfw_interp: (static method) returns self.r_nfw_interp.
        self._simcat__halo_mass_bins: (special method) returns self.hmb.
        self._simcat__test_*.
    """

    def __init__(self, ftc, pkl_dir, debug=False):
        """
        Constructor args:

            ftc: (obj) instance of fx.cosmology.
            pkl_dir: (str) path to an existing directory for pickling computationally expensive objects.
            debug: (bool) whether to enable an interactive debug mode.
        """

        assert isinstance(ftc, fx.cosmology)
        assert isinstance(pkl_dir, str) and pkl_dir.endswith('/')
        assert isinstance(debug, bool)

        self.ftc = ftc
        self.pkl_dir = pkl_dir
        self.debug = debug

        assert hasattr(self.ftc, 'config')
        self.config = self.ftc.config

        # FIXME Mf: This is fine for now given that we're currently using the same Mf for all frb_par's.
        self.m_f = self.ftc.frb_par[0][7]

        self.r_nfw_interp = simcat._r_nfw_interp()

        self.zmin = self.config.zmin
        self.zmax = self.config.fn_zmax
        self.zz = fx.slicer(self.zmin, self.zmax, self.config.sim.nz, log_spaced=True)

        _path = self.pkl_dir + 'hmb.pkl'
        try:
            hmb = fx.read_pickle(_path)
        except OSError as err:
            print(err)
            hmb = self.__halo_mass_bins()
            fx.write_pickle(_path, hmb)

        self.hmb = hmb

    @staticmethod
    def set_rng(seed=None):
        """Sets the initial seed for generating random numbers."""

        assert (seed is None) or (seed == int(seed))

        rand.seed(seed)
        print(f'seed = {seed}')

    def simulate(self, o_path, nsim, seed=None):
        """
        This method generates catalogs of galaxies and FRBs.

        At its core, there is an outer loop of Monte Carlo simulations containing an inner loop over independent
        redshift shells with vectorized halo mass bins.

        In each MC run, intrahalo sources from all redshifts are concatenated and saved in a unique
        (based on source type) hdf5 dataset.  Because of computational constraints we refrain from generating
        all halos (see below) and saving any halos to disk.  This format resembles real intrahalo catalogs
        covering a range of redshift values.  In addition, we allow intrahalo sources to jump over shell boundaries.

        In the inner loop over redshift shells (self.zmin, self.zmax),

            1. We collect numerically computed (i.e. expected) halo parameters, e.g., mass bins and halo mass functions,
               which are subsequently used to compute random (i.e. actual) parameters.

            2. For shells inside self.survey_galaxy.zmax:
                2.1. A catalog of halos is created for mass bins with m >= Mg, which can host galaxies.
                2.2. Using the existing halos with m >= Mg, a galaxy catalog is generated for the shell.

            3. Using all mass bins m >= Mf, we assign FRBs to halo indices.  If an index exists already (i.e. halos
               with m >= Mg), then we use the existing halo.  Otherwise, we make a new random halo and append it to
               the catalog.  Multiple FRB catalogs (saved in separate hdf5 datasets) can be generated for lists
               of frb_par values.  Throughout, we adopt a correlated hypothesis.

        Args:

            o_path: (str) path, ending with '.h5', for writing catalogs.
            nsim: (int) total number of simulations.
            seed: (int) the initial seed for generating random numbers.
        """

        assert isinstance(o_path, str) and o_path.endswith('.h5')
        assert isinstance(nsim, int) and (nsim % 2 == 0) and (nsim > 0)

        simcat.set_rng(seed)

        h5_mode = 'w'       # Becomes 'a' after writing the first catalog to disk.
        aux_axes = [-2, -1]

        psg = []
        for q in range(nsim):
            print(f'iter {q} of {nsim-1}')

            # Current number of galaxies and FRBs in saved catalogs.
            max_counts_g = 0
            max_counts_f = [0 for _ in range(len(self.ftc.frb_par))]

            # Iterating over redshift shells, catalogs are concatenated for the entire volume.
            for iz in range(self.config.sim.nz):
                print(f'iz = {iz}')

                z = np.mean((self.zz[0][iz], self.zz[1][iz]))

                m = self.hmb[iz]['m']               # Mass-bin centers.
                n_halo_e = self.hmb[iz]['n_e']      # Expected number of halos in mass bins.
                r = self.hmb[iz]['r']               # Comoving r_vir of mass bins.
                c = self.hmb[iz]['c']               # Concentration parameter of mass bins.
                m_mask = self.hmb[iz]['m_mask']     # Boolean mask for m >= Mg (galaxies reside in these halos).

                n_halo = rand.poisson(n_halo_e)             # Actual number of halos in mass bins.
                max_halo_index = np.cumsum(n_halo)          # Max halo index in mass bins.
                min_halo_index = max_halo_index - n_halo    # Min halo index in mass bins.

                m_m = m[m_mask]                 # m for mass bins with m >= Mg.
                r_m = r[m_mask]                 # r for mass bins with m >= Mg.
                c_m = c[m_mask]                 # c for mass bins with m >= Mg.
                n_halo_m = n_halo[m_mask]       # Expected number of halos in mass bins with m >= Mg.

                # Halo index offset due to partitioning catalogs by redshift and mass.
                if (z <= self.ftc.survey_galaxy.zmax) and np.any(n_halo_m != 0.0):
                    m_mask_offset = min_halo_index[m_mask][0]
                else:
                    m_mask_offset = 0

                if not q:
                    _psg_z = []
                    for qq, frb_par in enumerate(self.ftc.frb_par):
                        _psg_z.append(np.zeros_like(m_m, dtype=np.float64))

                        # psg values for halo mass bins in the current redshift shell.
                        for _i, _m in enumerate(m_m):
                            _psg_z[qq][_i] = self.ftc.psg[qq](z, _m)

                    psg.append(_psg_z)

                if (z <= self.ftc.survey_galaxy.zmax) and np.any(n_halo_m != 0):
                    # Generating halos with Mg <= m in the redshift range of the galaxy survey.
                    cat_halo = self._generate_halo_catalog(iz, n_halo_m, r_m, c_m, m_mask_offset)

                    # Generating galaxies which reside only in halos with m >= Mg.
                    ngalx_per_halo_mass_bin = rand.poisson(m_m / self.ftc.m_g(z) * n_halo_m)

                    gx, gx_counts = simcat.unique_random_halo_index(ngalx_per_halo_mass_bin, n_halo_m, m_mask_offset)

                    # FIXME sg: Conditional above is not satisfied where z > self.ftc.survey_galaxy.zmax.
                    if gx.size:
                        # Recording the current [redshift shell index, halo indices] for Nfrb1 sources below.
                        _gx = []
                        for (i,j) in zip(gx, gx_counts):
                            for jj in range(j):
                                _gx.extend([i])

                        _gx = np.asarray(_gx)
                        aux = [np.full_like(_gx, iz), _gx]

                        gx = np.asarray([cat_halo[halo_index] for halo_index in gx])     # dict -> array.

                        max_counts_g = self._generate_intrahalo_catalog(
                                     obj_type='galaxy', counts=gx_counts, center=gx[:,:3], r_vir=gx[:,3],
                                     conc=gx[:,4], z_halo=z, aux=aux, max_counts=max_counts_g,
                                     name_catalog=f'galaxy_{q}', o_path=o_path, h5_mode=h5_mode)

                        h5_mode = 'a'
                else:
                    cat_halo = {}
                    gx = np.array([])
                    ngalx_per_halo_mass_bin = None

                # Generating FRB catalogs (and halos if they don't exist already).
                for qq, frb_par in enumerate(self.ftc.frb_par):

                    # 'self.ftc.eta' values depend on 'self.ftc.survey_frb.f_sky' which is currently greater than
                    # the simulation box set by 'self.config.f_sky'.  In other words, 'self.__halo_mass_bins' uses
                    # 'self.config.f_sky' to get 'n_halo' values, which need to be re-normalized in order to get N_frb
                    # in the simulation box.

                    # Generating the general population of FRBs.
                    nfrb_per_halo_mass_bin = rand.poisson(
                                           max(0.0, self.ftc.eta[qq](z) * self.ftc.survey_frb.f_sky / self.config.f_sky)
                                           * (m/self.m_f) * n_halo)

                    fy, fy_counts = simcat.unique_random_halo_index(nfrb_per_halo_mass_bin, n_halo)

                    if fy.size:
                        # Use existing halos if available, otherwise make a new one.
                        _fy = []
                        for i in fy:
                            if i not in cat_halo:
                                _r = r[i <= max_halo_index][0]
                                _c = c[i <= max_halo_index][0]

                                new_halo = self._generate_halo(1, iz, _r, _c)[0]
                                new_halo[2] = self.ftc.z_to_chi(new_halo[2])

                                cat_halo.update({i: new_halo})

                            _fy.append(cat_halo[i])

                        fy = np.asarray(_fy)

                        max_counts_f[qq] = self._generate_intrahalo_catalog(
                                         obj_type='frb', counts=fy_counts, center=fy[:,:3], r_vir=fy[:,3],
                                         o_path=o_path, conc=fy[:,4], z_halo=z, max_counts=max_counts_f[qq],
                                         name_catalog=f'frb_{qq}_{q}', h5_mode=h5_mode, frb_par=frb_par)

                        h5_mode = 'a'

                    # FIXME sg:
                    #   - Need to "replace" cat_Nfrb1 sources with galaxies below.
                    #   - psg depends on Mg which is constrained by self.ftc.survey_galaxy.zmax.

                    # Generating FRBs which are bound to galaxies.
                    if gx.size and np.sum(psg[iz][qq]):
                        nfrb1_per_halo_mass_bin_g = rand.poisson(psg[iz][qq] * ngalx_per_halo_mass_bin)
                        nfrb1_per_halo_mass_bin_f = np.minimum(nfrb1_per_halo_mass_bin_g,
                                                               nfrb_per_halo_mass_bin[m_mask])

                        if np.sum(nfrb1_per_halo_mass_bin_f):
                            # Selecting host galaxies.
                            cat_nfrb1 = self._extract_random_intrahalo_source(
                                      name_catalog=f'galaxy_{q}', i_path=o_path, x=nfrb1_per_halo_mass_bin_f,
                                      n_halo=n_halo_m, iz=iz, halo_index_offset=m_mask_offset, axes=aux_axes)

                            max_counts_f[qq] = self._generate_intrahalo_catalog(
                                             obj_type='frb', max_counts=max_counts_f[qq], name_catalog=f'frb_{qq}_{q}',
                                             o_path=o_path, h5_mode=h5_mode, frb_par=frb_par, ext=cat_nfrb1[:,:3])

                            h5_mode = 'a'

        print('Simulation done!\n')

    def _extract_random_intrahalo_source(self, name_catalog, i_path, x, n_halo, iz, halo_index_offset=0, axes=None):
        """
        Using the total or per-halo-(*)-bin intrahalo source counts, this helper method extracts random intrahalo
        sources from a saved catalog.  It assumes a uniform probability distribution while selecting intrahalo sources
        from individual halos.

        (*) e.g. mass.

        Args:

            name_catalog: (str) name of the catalog/dataset as input.
            i_path: (str) input path, ending with '.h5', for reading catalogs.
            x: (1-d array or float) intrahalo source counts (per halo bin only if 1-d array).
            n_halo: (1-d array) halo counts per halo bin.
            iz: (int) redshift shell index.
            halo_index_offset: (int) smallest halo index for the zeroth halo bin.
            axes: (list) if not None, int values specifying the axes along which [redshift shell index, halo indices]
                  were saved in the catalog.

        Returns:

            array, catalog of randomly selected intrahalo sources.
        """

        assert isinstance(name_catalog, str)
        assert isinstance(i_path, str)
        assert i_path.endswith('.h5')

        if isinstance(x, np.ndarray) and (x.dtype == int):
            xv = True
        elif isinstance(x, (float, int)):
            xv = False
        else:
            raise RuntimeError('Invalid x arg!')

        assert isinstance(n_halo, np.ndarray) and (n_halo.dtype == int)

        if xv and ((x.shape != n_halo.shape) or (not np.all(x[n_halo == 0] == 0))):
            raise RuntimeError('Inconsistent x and n_halo args!')

        assert (iz == int(iz)) and (0 <= iz < self.config.sim.nz)
        assert (halo_index_offset == int(halo_index_offset)) and (halo_index_offset >= 0)

        axes = [-2, -1] if axes is None else axes
        assert isinstance(axes, list) and (len(axes) == 2)

        max_i = np.cumsum(n_halo) + halo_index_offset
        min_i = max_i - n_halo

        with FileH5(i_path, mode='r') as _file:
            _dataset = _file[name_catalog][:]

        assert _dataset.ndim == 2

        # Constraining by the redshift shell index.
        c1 = (_dataset[:,axes[0]] == iz)

        ret = []

        if xv:
            for i in range(n_halo.size):
                if n_halo[i] and x[i]:
                    # Constraining by the halo indices.
                    c2 = (_dataset[:,axes[1]] < max_i[i])
                    c3 = (min_i[i] <= _dataset[:,axes[1]])

                    # Applying the intersection of all conditions to the dataset.
                    c123 = np.logical_and(c1, np.logical_and(c2, c3))

                    assert c123.size == _dataset.shape[0]
                    _dataset_masked = _dataset[c123]

                    # Assuming a uniform probability distribution.
                    _xi = rand.randint(0, _dataset_masked.shape[0], size=x[i])

                    # Appending to a list of possibly repeated catalog entries.
                    for j in _xi:
                        ret.append(_dataset_masked[j])
        else:
            if x:
                _dataset_masked = _dataset[c1]
                _dataset_masked_len = _dataset_masked.shape[0]

                if _dataset_masked_len:
                    _xi = rand.randint(0, _dataset_masked_len, size=int(round(x)))

                    for i in _xi:
                        ret.append(_dataset_masked[i])

        if len(ret):
            ret = np.asarray(ret)

            sx = np.sum(x) if xv else int(round(x))

            if ret.size != (sx*_dataset.shape[1]):
                raise RuntimeError('simcat._extract_random_intrahalo_source: extracted sources leaked!')

            return ret
        else:
            return np.empty(0)

    @staticmethod
    def unique_random_halo_index(x, n_halo, halo_index_offset=0):
        """
        This static method converts per-halo-(*)-bin intrahalo source counts to unique random indices of halos
        which contain intrahalo sources.

        (*) e.g. mass.

        Args:

            x: (1-d array) intrahalo source counts per halo bin.
            n_halo: (1-d array) halo counts per halo bin.
            halo_index_offset: (int) smallest halo index for the zeroth halo bin.

        Returns:

            tuple, containing (1-d array of unique random halo indices, 1-d array of intrahalo source counts
            for the unique halos).
        """

        assert isinstance(x, np.ndarray) and (x.dtype == int)
        assert isinstance(n_halo, np.ndarray) and (n_halo.dtype == int)
        assert x.shape == n_halo.shape
        assert (x[n_halo == 0] == 0).all()
        assert (halo_index_offset == int(halo_index_offset)) and (halo_index_offset >= 0)

        max_i = np.cumsum(n_halo) + halo_index_offset
        min_i = max_i - n_halo

        ret = []
        for i in range(n_halo.size):
            if n_halo[i] and x[i]:
                _x = rand.randint(min_i[i], max_i[i], size=x[i])
                ret.append(_x)

        # Lists of possibly repeated indices -> arrays of unique indices, counts.
        if len(ret):
            ret = np.concatenate(ret).ravel()
            ret, ret_counts = np.unique(ret, return_counts=True)
            return ret, ret_counts
        else:
            return np.empty(0), np.empty(0)

    def _generate_halo(self, n, iz, r, c):
        """
        This helper method returns randomly distributed halos, all with the same radius and concentration parameter,
        in a redshift shell.

        Args:

            n: (int) number of halos.
            iz: (int) redshift shell index.
            r: (float) virial radius.
            c: (float) concentration parameter.

        Returns:

            array of shape (n,4) containing [3-d position (self.config.sim.unit,self.config.sim.unit,z), virial radius (Mpc/h), and
            concentration parameter].
        """

        assert (n == int(n)) and (n > 0)
        assert (iz == int(iz)) and (0 <= iz < self.config.sim.nz)
        assert isinstance(r, float) and (r > 0.0)
        assert isinstance(c, float) and (c > 0.0)

        ret = np.zeros((n,5), dtype=np.float64)
        ret[:,:2] = rand.uniform(0.0, self.config.xymax_cov.value, size=(n,2))

        zmin = self.zz[0][iz]
        zmax = self.zz[1][iz]

        assert 0.0 <= zmin < zmax

        ret[:,2] = rand.uniform(zmin, zmax, size=n)
        ret[:,3] = r
        ret[:,4] = c

        return ret

    def _generate_halo_catalog(self, iz, n_halo, r, c, halo_index_offset=0):
        """
        This helper method generates a catalog of randomly distributed halos in a redshift shell.

        Args:

            iz: (int) redshift shell index.
            n_halo: (1-d array) expected counts in mass bins.
            r: (1-d array) virial radii.
            c: (1-d array) concentration parameters.
            halo_index_offset: (int) smallest halo index to appear in the catalog.

        Returns:

            dictionary with the following {key: value} configuration:
            {halo index: [3-d position (self.config.sim.unit,self.config.sim.unit,Mpc/h) and virial radius (Mpc/h)]}
        """

        assert (iz == int(iz)) and (0 <= iz < self.config.sim.nz)
        assert isinstance(n_halo, np.ndarray) and (n_halo.dtype == int)
        assert isinstance(r, np.ndarray) and (r.dtype == np.float64)
        assert isinstance(c, np.ndarray) and (c.dtype == np.float64)
        assert n_halo.shape == r.shape == c.shape
        assert (halo_index_offset == int(halo_index_offset)) and (halo_index_offset >= 0)

        if np.all(n_halo == 0):
            raise RuntimeError('_generate_halo_catalog: halo bins are empty!'
                               + ' Inspect the expected halo counts in self.hmb.')

        cat_halo = []
        for i, n in enumerate(n_halo):
            if n:
                _cat = self._generate_halo(n, iz, r[i], c[i])   # Array.
                cat_halo.append(_cat)                           # List of arrays.

        cat_halo = np.concatenate(cat_halo)                     # Cat shape is preserved.
        cat_halo[:,2] = self.ftc.z_to_chi(cat_halo[:,2])        # Intrahalo catalogs require Mpc/h, not z, as input.

        # array -> dict
        cat_halo = dict(enumerate(cat_halo, halo_index_offset))

        return cat_halo

    def _generate_intrahalo_catalog(self,  name_catalog, o_path, obj_type, z_halo=None, counts=None, center=None,
                                    r_vir=None, conc=None, max_counts=0, aux=None, h5_mode='w', frb_par=None, ext=None):
        """
        This helper method generates random catalogs of galaxies or FRBs in a redshift shell.

        Args:

            name_catalog: (str) name of the catalog/dataset in the output (.h5 file).
            o_path: (str) path, ending with '.h5', for writing catalogs.
            obj_type: (str) type of sources to be cataloged.
            z_halo: (float) if not None, mean redshift of the shell.
            counts: (1-d array) if not None, integers corresponding to source (FRB or galaxy) counts in each
                    potential halo. (*)
            center: (2-d array) if not None, 3-d positions (2-d sky and line-of-sight comoving distance) organized
                    in three columns. (*)
            r_vir: (1-d array) if not None, floats specifying the comoving virial radii (Mpc/h) of potential halos. (*)
            conc: (1-d array) if not None, floats specifying the NFW concentration parameters.
            max_counts: (int) number of pre-existing sources in the same catalog/dataset.
            aux: optional (list) if not None, 1-d arrays supplying auxiliary columns to be appended to the catalog. (*)
            h5_mode: (bool) whether to 'w'rite or 'a'ppend a catalog.
            frb_par: (list) set of FRB parameters.
            ext: (array) if not None, supplies a catalog of sources that will be used instead of generating new
                 realizations for positions. It currently assumes the following format:
                 [[x_0, y_0, z_0], [x_1, y_1, z_1], ..., [x_N, y_N, z_N]], where (x,y) are angular positions and (z) is
                 the redshift.  Depending on the value of 'obj_type' arg, additional axes may be appended to the input
                 catalog.

            (*) must have the same length, which is equivalent to the number of potential halos being simulated!

        Returns:

            non-zero integer corresponding to the total number of sources in the catalog.  This can be used as
            the 'max_counts' argument for appending more catalogs to the same dataset in the same .h5 file.
        """

        nv = (z_halo is None) or (counts is None) or (center is None) or (r_vir is None) or (conc is None)
        if (ext is None) and nv:
            raise RuntimeError('simcat._generate_intrahalo_catalog: insufficient combination of args.')

        if (z_halo is not None) and (z_halo <= 0.0):
            raise RuntimeError('simcat._generate_intrahalo_catalog: invalid z_halo.')

        if ext is None:
            center = np.asarray(center)
            r_vir = np.asarray(r_vir)[:,np.newaxis]
            conc = np.asarray(conc)[:,np.newaxis]

            n_per_halo, dim = center.shape
            counts = np.asarray(counts, dtype=int)

            assert dim == 3
            assert r_vir.shape == (n_per_halo,1)
            assert conc.shape == (n_per_halo,1)
        else:
            assert isinstance(ext, np.ndarray) and (ext.ndim == 2)
            n_per_halo, dim = ext.shape

        assert isinstance(name_catalog, str)
        assert isinstance(o_path, str)
        assert o_path.endswith('.h5')

        if aux is not None:
            assert isinstance(aux, list)

            for _aux in aux:
                assert isinstance(_aux, np.ndarray)
                assert _aux.size == np.sum(counts)

            dim += len(aux)

        if obj_type == 'frb':
            dim += 1

        assert h5_mode in ('w', 'a')
        assert obj_type in ('frb', 'galaxy')
        assert (max_counts == int(max_counts)) and (max_counts >= 0)

        counts_total = np.sum(counts) if (ext is None) else n_per_halo

        q = max_counts + counts_total

        if not counts_total:
            return q
        else:
            pass

        if ext is None:
            v = self._generate_nfw(counts, conc)

            r_vir = np.repeat(r_vir, counts, axis=0)
            center = np.repeat(center, counts, axis=0)

            # (0,1] -> Mpc/h
            v *= r_vir

            # Mpc/h -> self.config.sim.unit
            v[:,0] = self._mpc_to_angle(r=v[:,0], z=z_halo) + center[:,0]
            v[:,1] = self._mpc_to_angle(r=v[:,1], z=z_halo) + center[:,1]

            # Mpc/h -> z
            v[:,2] = self.ftc.chi_to_z(v[:,2] + center[:,2])

            # constraining redshifts
            v[:,2][v[:,2] < self.zmin] = self.zmin
            v[:,2][v[:,2] > self.zmax] = self.zmax
        else:
            # copying in order to prevent an accidental overwrite
            v = ext.copy()

        if self.debug:
            print('#'*120, '\nsimcat._generate_intrahalo_catalog: Entering the debug mode\n', '#'*120)
            input('Press Enter to continue...')
            print(f'(max_counts = {max_counts}, counts_total = {counts_total})')
            input('Press Enter to continue...')

        o = FileH5(o_path, mode=h5_mode)
        o_dataset = o.require_dataset(
                  name_catalog, shape=(max_counts,dim), maxshape=(None,dim), dtype=np.float64, exact=True)

        if self.debug:
            print(f'Before h5 resize, {o_dataset.shape[0]}')
            input('Press Enter to continue...')

        o_dataset.resize((q,dim))

        if self.debug:
            print(f'After h5 resize, {o_dataset.shape[0]}')
            input('Press Enter to continue...')
            print(f"'A'*10, {o_dataset[0,:]}, {o_dataset[-1,:]}")

        o_dataset[max_counts:,:3] = v[:,:3]

        if self.debug:
            print(f"'B'*10, {o_dataset[0,:]}, {o_dataset[-1,:]}")

        if (obj_type == 'frb') and (frb_par is not None):
            dm_host = self.ftc.dm_h(counts_total, mu=frb_par[3], sigma=frb_par[4])

            for i, z in enumerate(v[:,2]):
                if self.debug:
                    print(f'(i={i},z={z}')
                    print(f"'C'*10, {o_dataset[0,:]}, {o_dataset[max_counts+i,:]}")

                # Constraining the min DM by self.zmin and the rtol in self.ftc.dm_igm_interp.
                _DM_e = max(self.ftc.dm_igm_interp(z), self.ftc.dm_igm_interp(self.zmin)*1.0001) + dm_host[i]
                o_dataset[max_counts+i,3] = _DM_e

                if self.debug:
                    print(f"'D'*10, {o_dataset[0,:]}, {o_dataset[max_counts+i,:]}, {_DM_e}, {max_counts+i}")
                    print(f"'E'*10, {o_dataset[0,:]}, {o_dataset[-1,:]}, {_DM_e}, {o_dataset.shape[0]}")

            if self.debug:
                input('Press Enter to continue...')

        if self.debug:
            print(f"'F'*10, {o_dataset[0,:]}, {o_dataset[-1,:]}")
            input('Press Enter to continue...')

        if aux is not None:
            a = len(aux)
            for i, _aux in enumerate(aux):
                if self.debug:
                    print(f"'AUX'*10, {a}, {_aux.shape}, {o_dataset.shape}")
                o_dataset[max_counts:, dim-a+i] = _aux[:]

        o_dataset.attrs['sky_unit'] = str(self.config.sim.unit)

        if obj_type == 'frb':
            o_dataset.attrs['name_key'] = ['obj_type, frb_par_i, simulation_i']
            o_dataset.attrs['col_key'] = ['x', 'y', 'redshift', 'DM']
        else:
            o_dataset.attrs['name_key'] = ['obj_type, simulation_i']
            o_dataset.attrs['col_key'] = ['x', 'y', 'redshift']

        o.close()

        _p = f'{name_catalog} catalog containing {counts_total} sources saved in {o_path}'

        if aux is not None:
            print(f'AUX -> {_p}')
        elif ext is not None:
            print(f'EXT -> {_p}')
        else:
            print(_p)

        return q

    def _mpc_to_angle(self, r, z):
        """
        This helper method converts transverse comoving distance to an angle in self.config.sim.unit.

        Args:

            r: (float or array) transverse comoving distance (implicitly in Mpc/h).
            z: (float) redshift for which angles are measured.

        Returns:

            (float or array) angle(s) in self.config.sim.unit (implicit).
        """

        assert isinstance(r, (float, np.ndarray))

        if z is None:
            raise RuntimeError('simcat._mpc_to_angle: z is None!')
        else:
            assert isinstance(z, float) and (z >= 1.0e-3)

        ret = (r / self.ftc.z_to_chi(z) * u.rad).to(self.config.sim.unit)

        return ret.value

    def _generate_nfw(self, counts, conc):
        """
        This helper method generates random 3-d positions in NFW halos.

        Args:

            counts: (1-d array) number of random intrahalo sources.
            conc: (1-d array) NFW concentration parameters.

        Returns:

            array, 3-d positions in NFW halos of unit radius.
        """

        assert isinstance(counts, np.ndarray) and (counts.dtype == int)
        assert isinstance(conc, np.ndarray) and (conc.dtype == float)
        assert counts.size == conc.size

        if np.any(counts == 0):
            warnings.warn('Zero counts are being passed to self._generate_nfw!')

        assert np.logical_and((conc > 0.0), (conc < self.r_nfw_interp['c_max'])).all()

        ret = []
        for (i,j) in zip(counts.ravel(), conc.ravel()):
            if i == 0:
                continue

            r_nfw = self._r_nfw(i, j)

            v = rand.normal(size=(i,3))
            norm = np.linalg.norm(v, axis=1)

            while np.sum(norm < self.r_nfw_interp['r_min']):
                mask = (norm < self.r_nfw_interp['r_min'])
                a = rand.normal(size=(np.sum(mask), 3))
                norm[mask] = np.linalg.norm(a, axis=1)

            assert r_nfw.shape == norm.shape
            v *= (r_nfw / norm)[:,np.newaxis]

            ret.append(v)

        ret = np.concatenate(ret)
        assert ret.shape == (np.sum(counts),3)

        return ret

    def _r_nfw(self, n, c):
        """
        This helper method randomly samples radial positions in the NFW halo profile.

        Args:

            n: (int) total number of random samples.
            c: (float) NFW concentration parameter.

        Returns:

            1-d array.
        """

        assert (n == int(n)) and (n >= 1), f'({type(n)}, {n})'
        assert isinstance(c, float)
        assert (0.0 < c < self.r_nfw_interp['c_max'])

        r = rand.uniform(self.r_nfw_interp['r_min'], simcat.nfw_prof(c), size=n)

        assert np.all(r > 0.0)

        _sp = self.r_nfw_interp['interp']

        # r/r_s -> r/r_vir.
        ret = _sp(np.log(r)) / c

        assert np.isfinite(ret).all()

        return ret

    @staticmethod
    def nfw_prof(c):
        """
        This static method computes the logarithmic part of the characteristic overdensity (Eq. 4 in astro-ph/9508025)
        for a virialized NFW halo profile.

        Args:

            c: (float) NFW concentration parameter.

        Returns:

            float.
        """

        assert isinstance(c, float) and (c >= 0.0)

        ret = np.log(1+c) - c/(1+c)

        return ret

    @staticmethod
    def _r_nfw_interp(r_min=1.0e-5, c_max=100.0, rstep=10000):
        """
        This static method constructs an interpolation table for computing radial coordinates in the NFW halo profile.

        Args:

            r_min: (float) minimum radius allowed.
            c_max: (float) maximum value of the NFW concentration parameter.
            rstep: (int) number of interpolation steps along the radial coordinate.

        Returns:

            dictionary, containing two floats ['r_min'] and ['c_max'], along with an interpolator
            object ['interp'] as a function of log(r).
        """

        assert isinstance(r_min, float) and (r_min > 0.0), r_min
        assert isinstance(c_max, float) and (c_max > 0.0), c_max
        assert (rstep == int(rstep)) and (rstep >= 100), rstep

        r_min = np.log(r_min)
        r_max = np.log(simcat.nfw_prof(c_max))

        r = np.linspace(r_min, r_max, rstep)

        ret = np.zeros(rstep)
        for i, _r in enumerate(np.exp(r)):
            try:
                ret[i] = scipy.optimize.brentq(lambda x: simcat.nfw_prof(x) - _r, 0.0, c_max+1.0e-7)
            except ValueError as err:
                print(f'simcat._r_nfw_interp: {err}')
                raise RuntimeError(f'scipy.optimize.brentq failed: _r={np.log(_r)}, r_max={r_max}')

        return {'r_min': np.exp(r_min),
                'c_max': c_max,
                'interp': fx.spline(r, ret)}

    def __halo_mass_bins(self, nm_halo=None):
        """
        Using the Sheth-Tormen halo mass function, this special method computes the expected number
        of halos for individual mass bins.

        Args:

            nm_halo: (int) if not None, total number of mass bins per redshift shell.

        Returns:

            list whose elements are dictionaries that correspond to redshift shells (increasing index as going to
            higher redshifts), each containing 1-d arrays of mass bins ['m'], expected halo counts ['n_e'], comoving
            virial radii ['r'], NFW concentration parameter ['c'], and a boolean mask specifying mass bins greater
            than or equal to Mg.
        """

        nm_halo = self.config.sim.nm_halo if nm_halo is None else nm_halo

        assert (nm_halo == int(nm_halo)) and (nm_halo >= 100)

        def _in(z_min, z_max, m_min, m_max, epsrel=1.0e-4):
            """This nested function returns the total number of halos per (redshift, mass) bin."""

            assert z_min >= self.zmin
            assert z_max <= self.zmax
            assert m_min >= self.ftc.hmf.m_min * 10.
            assert m_max <= self.ftc.hmf.m_max / 100.

            _ret = fx.dblquad(lambda x, M: self.ftc.dvoz(z) * self.ftc.hmf.interp_dn_dlog_m(x,M),
                              np.log(m_min), np.log(m_max), lambda M: z_min, lambda M: z_max, epsrel=epsrel)

            assert np.isfinite(_ret) and (_ret >= 0.0)
            _ret *= (4 * np.pi * self.config.f_sky)

            return _ret

        ret = []
        for iz in range(self.config.sim.nz):
            zmin = self.zz[0][iz]
            zmax = self.zz[1][iz]
            z = np.mean((zmin, zmax))

            assert nm_halo <= self.ftc.hmf.interp_nstep_m

            m0 = np.log(self.m_f)

            if z <= self.ftc.survey_galaxy.zmax:
                m1 = np.log(self.ftc.m_g(z))
            else:
                m1 = np.log(self.ftc.m_g(self.ftc.survey_galaxy.zmax))

            m2 = np.log(self.ftc.hmf.m_max/100.)

            assert m0 < m1 < m2

            nm_halo_01 = int(round(nm_halo * (m1-m0) / (m2-m0)))
            nm_halo_12 = nm_halo - nm_halo_01
            nm_halo_12 = max(nm_halo_12, nm_halo - nm_halo_01 + 1)

            # Making an array with unique values.
            m_bin = np.exp(np.append(
                  np.linspace(m0, m1, nm_halo_01),
                  np.linspace(m1, m2, nm_halo_12)[1:]))

            # Since end points could have gone off limits in the preceding steps.
            m_bin[0] = self.m_f
            m_bin[-1] = self.ftc.hmf.m_max/100.

            # Mid values, hence smaller by 1 element.
            m = np.zeros_like(m_bin)[:-1]
            n_e = np.zeros_like(m)

            for i in range(m.size):
                m[i] = np.mean((m_bin[i], m_bin[i+1]))
                n_e[i] = _in(zmin, zmax, m_bin[i], m_bin[i+1])

            r = self.ftc.hmf.r(z, m, vir=True)
            c = self.ftc.hmf.conc(z, m)

            if z <= self.ftc.survey_galaxy.zmax:
                m_mask = (m >= self.ftc.m_g(z))
            else:
                m_mask = np.zeros_like(m, dtype=bool)

            ret.append({'m': m,
                        'm_bin': m_bin,
                        'n_e': n_e,
                        'r': r,
                        'c': c,
                        'm_mask': m_mask})

        return ret

    def __test_all(self):
        self.__test_set_rng()
        self.__test_simulate()
        self.__test__extract_random_intrahalo_source()
        self.__test_unique_random_halo_index()
        self.__test__generate_halo()
        self.__test__generate_halo_catalog()
        self.__test__mpc_to_angle()
        self.__test__r_nfw_interp()

    def __test_set_rng(self):
        pass

    def __test_simulate(self, nsim=2, frb_par_index=0, seed=28):
        o_path = os.path.join(fx.utils.data_path('archive', envar='FRBXDATA', mode='w'), 'test-simcat.h5')

        # Edges of redshift bins to be histogrammed.
        bins = np.append(self.zz[0], self.zz[1][-1])

        # Max number of points in scatter plot.
        nxy_plt = 100000

        random_sim_index = rand.randint(nsim-1)
        self.simulate(o_path, nsim, seed)

        # Redshift distribution.
        nf = np.zeros(0)
        ng = np.zeros(0)

        xy_f = None
        xy_g = None
        for q in range(nsim):
            cat_f = fx.read_h5(o_path, f'frb_{frb_par_index}_{q}')
            cat_g = fx.read_h5(o_path, f'galaxy_{q}')

            if q == random_sim_index:
                xy_f = cat_f[cat_f[:,2].argsort()]     # Sort by redshift.
                xy_f = xy_f[:nxy_plt, :2]

                xy_g = cat_g[cat_g[:,2].argsort()]     # Sort by redshift.
                xy_g = xy_g[:nxy_plt, :2]

            nf = np.append(cat_f[:,2], nf, axis=0)
            ng = np.append(cat_g[:,2], ng, axis=0)

        # FRBs.
        plt.hist(nf, log=True, bins=bins, fc=(0,0,0,0), lw=1, ls='solid', histtype='step')

        dndz = self.ftc.dndz_frb[frb_par_index]
        zz = np.mean(self.zz, axis=0)

        y = []
        for i in range(self.config.sim.nz):
            (zmin, zmax) = (self.zz[0][i], self.zz[1][i])
            nb = fx.quad(lambda x: dndz(x), zmin, zmax)
            nb *= (4 * np.pi * self.ftc.survey_frb.f_sky)
            y.append(nb * nsim)

        plt.plot(zz, y)
        plt.xlabel(r'$z$')
        plt.xlim(self.ftc.interp_zmin,
                 self.ftc.survey_galaxy.zmax * self.ftc.config.md + self.config.zpad_interp_zmax)
        plt.savefig('test_simulate_f.pdf')
        plt.clf()

        # Galaxies.
        plt.hist(ng, log=True, bins=bins, fc=(0,0,0,0), lw=1, ls='solid', histtype='step')

        y = []
        for i in range(self.config.sim.nz):
            zmin = min(self.zz[0][i], self.ftc.survey_galaxy.zmax)
            zmax = min(self.zz[1][i], self.ftc.survey_galaxy.zmax)
            nb = fx.quad(lambda x: self.ftc.dndz_galaxy(x), zmin, zmax)
            nb *= (4 * np.pi * self.config.f_sky)
            y.append(nb * nsim)

        plt.plot(zz, y)
        plt.xlabel(r'$z$')
        plt.xlim(self.ftc.interp_zmin, self.ftc.survey_galaxy.zmax + self.config.zpad_interp_zmax)
        plt.savefig('test_simulate_g.pdf')
        plt.clf()

        # Sky positions.
        if xy_g is not None:
            plt.scatter(xy_g[:,0], xy_g[:,1], color='b', marker='o', s=4.0)
        else:
            raise RuntimeError('__test_simulate: xy_g is empty!')

        if xy_f is not None:
            plt.scatter(xy_f[:,0], xy_f[:,1], color='g', marker='o', s=1.0)
        else:
            raise RuntimeError('__test_simulate: xy_f is empty!')

        plt.xlim(0.0, self.config.xymax_cov.value)
        plt.ylim(0.0, self.config.xymax_cov.value)
        plt.xlabel(r'$\theta_x$')
        plt.ylabel(r'$\theta_y$')
        plt.savefig('test_simulate_xy_fg.pdf')
        plt.clf()

        os.remove(o_path)

    def __test__extract_random_intrahalo_source(self, niter=100):
        dir_path = fx.utils.data_path('archive', envar='FRBXDATA', mode='w')
        name_catalog1 = os.path.join(dir_path, './test-simcat_temp1.h5')
        name_catalog2 = os.path.join(dir_path, './test-simcat_temp2.h5')
        name_catalog3 = os.path.join(dir_path, './test-simcat_temp3.h5')

        o = FileH5(name_catalog1, mode='w')
        o_dataset = o.create_dataset('test', shape=(100,10), dtype=np.float64)
        o_dataset[:] = np.repeat(np.arange(100), 10, axis=0).reshape(100,10)
        o.close()

        o = FileH5(name_catalog3, mode='w')
        o_dataset = o.create_dataset('test', shape=(1001,10), dtype=np.float64)
        o_dataset[0:500,:] = 0.0
        o_dataset[500:1000,:] = 1.0
        o.close()

        n_halo = np.ones(100, dtype=int)

        max_counts12 = 0
        _e_sum12 = 0
        _e3 = 0
        for i in range(niter):
            redshift_shell_index = rand.randint(self.config.sim.nz)

            x = np.zeros(100, dtype=int)
            x[redshift_shell_index] = 1

            halo_index_offset = 0
            axes = [rand.randint(10), rand.randint(10)]

            # Arr x.
            cat_temp12 = self._extract_random_intrahalo_source(
                       'test', name_catalog1, x, n_halo, redshift_shell_index, halo_index_offset, axes)

            _e12 = np.repeat(np.asarray([redshift_shell_index]), 10, axis=0).reshape(1,10)

            assert np.all(cat_temp12 == _e12)

            max_counts12 = self._generate_intrahalo_catalog(
                         name_catalog='test', o_path=name_catalog2, obj_type='galaxy', max_counts=max_counts12,
                         h5_mode='w' if not i else 'a', ext=_e12)

            _e_sum12 += np.sum(_e12[:,:3])

            # Scalar x.
            cat_temp3 = self._extract_random_intrahalo_source(
                      'test', name_catalog3, 999.6, n_halo, redshift_shell_index % 2, halo_index_offset, axes)

            _e3 += (np.sum(cat_temp3) / 10.)

        with FileH5(name_catalog2, mode='r') as _f:
            ret = _f['test'][:]

        assert np.sum(ret) == _e_sum12

        _e3 /= niter
        assert (300 <= _e3 <= 700), f'{_e3}'

        os.remove(name_catalog1)
        os.remove(name_catalog2)
        os.remove(name_catalog3)

    def __test_unique_random_halo_index(self, niter=100, max_count=1000):
        for arr_size in range(1, niter):
            x = rand.randint(1, max_count, size=arr_size)
            n_halo = rand.randint(1, max_count, size=arr_size)
            index_offset = rand.randint(0, int(np.sum(n_halo)))

            n, nx = self.unique_random_halo_index(x, n_halo, halo_index_offset=index_offset)

            n_uniq, n_uniq_count = np.unique(n, return_counts=True)

            assert np.all(n == n_uniq)
            assert np.all(n_uniq <= (np.sum(n_halo) + index_offset))
            assert np.sum(x) == np.sum(nx)

    def __test__generate_halo(self, tol=eps, n=10000, r=1.0, c=5.0):
        iz = rand.randint(self.config.sim.nz)
        zmin = self.zz[0][iz]
        zmax = self.zz[1][iz]

        arr = self._generate_halo(n, iz, r, c)

        assert np.all(arr[:,:2] < (self.config.xymax_cov.value + tol))
        assert np.logical_and((zmin <= arr[:,2]), (arr[:,2] < (zmax + tol))).all()
        assert np.all(arr[:,3] == r)
        assert np.all(arr[:,4] == c)

    def __test__generate_halo_catalog(self):
        if self.config.sim.nz > 10:
            warnings.warn('test__generate_halo_catalog: generates reasonable results for wide redshift slices. '
                          'Consider lowering the self.config.sim.nz value to an int less than 10 for this test!')

        # Number of points for the scatter plot of halo positions.
        n = 1000

        # Number of bins for the histogram of halo radii (recommended value ~ 1e4).
        nbins = 10000

        iz = min(2, self.config.sim.nz)        # Avoiding an empty n_halo_m below.
        hmb = self.hmb[iz]

        n_halo_e = hmb['n_e']                  # Expected number of halos in mass bins.
        m_mask = hmb['m_mask']                 # Boolean mask.
        n_halo = rand.poisson(n_halo_e)        # Actual number of halos in mass bins.

        n_halo_m = n_halo[m_mask]              # Actual number of halos in mass bins with m > Mg.
        r_m = hmb['r'][m_mask]                 # r for mass bins with m > Mg.
        c_m = hmb['c'][m_mask]                 # Concentration parameter for mass bins with m > Mg.

        # Without loss of generality we set the index_offset to zero (default).
        cat_halo = self._generate_halo_catalog(iz, n_halo_m, r_m, c_m)

        assert isinstance(cat_halo, dict)

        # dict -> array.
        cat_halo = np.asarray([j for (i,j) in cat_halo.items()])

        # Random sample for scatter plot.
        cat_halo_s = cat_halo[rand.randint(cat_halo.shape[0], size=n)]

        plt.scatter(cat_halo_s[:,0], cat_halo_s[:,1], color='r', marker='o', s=3)
        plt.xlim(0.0, self.config.xymax_cov.value)
        plt.ylim(0.0, self.config.xymax_cov.value)
        plt.xlabel(r'$\theta_x$')
        plt.ylabel(r'$\theta_y$')
        plt.savefig('test__generate_halo_catalog_0.pdf')
        plt.clf()
        plt.clf()

        plt.hist(cat_halo[:, 3], log=True, bins=nbins, fc=(0,0,0,0), lw=1, ls='solid', histtype='step')

        x = r_m
        y = n_halo_e[m_mask]

        plt.plot(x, y)
        plt.xlim(0.0, x[n_halo_m != 0][-1])
        plt.ylim(1, max(y)*2)
        plt.savefig('test__generate_halo_catalog_1.pdf')
        plt.clf()

    def __test__mpc_to_angle(self, niter=1000, rtol=1.0e-4, atol=0.0):
        rvec = rand.uniform(1.0e2, size=niter)                                  # Mpc/h.
        zvec = rand.uniform(self.zmin, self.zmax, niter)

        for (r,z) in zip(rvec, zvec):
            a = self._mpc_to_angle(r=r, z=z)
            a = (a * self.config.sim.unit).to(u.rad).value

            e = self.ftc.base.arcsec_per_kpc_comoving(z).value
            e *= (r * 1.0e3 / self.ftc.base.h)                                  # arcsec/kpc -> arcsec.
            e *= (np.pi / 180 / 3600)                                           # arcsec -> rad.

            assert np.isclose(a, e, rtol, atol)

    def __test__r_nfw_interp(self, c=5.0, n=1e6, nbins=30):
        c = np.array([c])
        n = np.array([n], dtype=int)
        xyz = self._generate_nfw(n, c)

        # Compare histogram of r-values to analytic NFW profile.
        r = np.sum(xyz**2, axis=1)**0.5
        assert np.logical_and((0.0 < r), (r <= 1.0)).all()

        plt.hist(r, bins=nbins)

        # The following tests were originally written by Kendrick Smith.
        #
        # Overplot NFW density f(r) = (A r^2 rho(r)).  The normalizing prefactor A is
        # chosen so that int_0^1 f(r) = n/nbins.

        x = np.linspace(0.0, 1.0, 100)      # r / rvir.

        A = (n / float(nbins)) * c**2 / (np.log(1+c) - c/(1+c))
        y = A * x / (1 + c*x)**2
        plt.plot(x, y)
        plt.savefig('test__r_nfw_interp_0.pdf')
        plt.clf()

        # Verify spherical symmetry of the NFW simulation.
        for i in range(3):
            t = rand.standard_normal(size=3)
            t /= np.sum(t**2)**0.5
            d = np.dot(xyz, t)
            plt.hist(d, bins=nbins, fc=(0,0,0,0), lw=1, ls='solid', histtype='step')

        plt.savefig('test__r_nfw_interp_1.pdf')
        plt.clf()
