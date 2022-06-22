import numpy as np
import astropy.units as u
import frbx as fx


sky_sr = 4 * np.pi * u.steradian
float64 = np.finfo(np.float64)
tiny = float64.tiny
eps = float64.eps


class configs:
    """Contains essential parameters for configuring an fx.cosmology."""

    def __init__(self, name):

        self.name = name                    # Name of the current instance.

        self.tau = 0.06
        self.ns = 0.965
        self.As = 2.10e-9
        self.r = 0
        self.kmax = 5.0e2                   # Max k without extrapolation.
        self.kmax_extrap = 1.0e6            # Max k with extrapolation.

        self.m_min = 1.0e4                  # 100 * hmf.m_min.
        self.m_max = 1.0e17                 # hmf.m_max / 100.
        self.interp_nstep_m_g = 512         # Number of interpolation steps in M_g(z).
        self.interp_nstep_eta = 512         # Number of interpolation steps in eta(z).

        self.interp_zmin = 0.0      # Global zmin for interpolations.
        self.interp_zmax = 20.0     # Global zmax for interpolations.

        ###################### These attributes will be modified by self.init_parameters ############################
        self.survey_frb = 'chime_baseline'
        self.survey_galaxy = 'sdss_dr8'
        self.frb_par_index = [0, 1, 2, 3]
        self.sim_name = '1h'
        self.md = 2                 # Multiplicative factor (in z) for specifying the lower bound of highest DM bin.
        self.zmin = 1.0e-3          # Local zmin for computing power spectra.
        self.zmax = 1.05            # Local zmax for computing power spectra.
        self.fn_zmax = 5.0          # Max redshift for normalizing FRB dn/dz.
        self.dmin = 0.0             # Min DM for binning.
        self.dmax = 2.0e3           # Max DM for binning.
        ############################################################################################################

        self.init_parameters()

        self.survey_frb = fx.frb_configs(self.survey_frb, self.frb_par_index)     # FRB survey configs.
        self.survey_galaxy = fx.galaxy_configs(self.survey_galaxy)                # Galaxy survey configs.
        self.frb_par = self.survey_frb.frb_par                                    # List of FRB parameters.
        self.sim = fx.sim_configs(self.sim_name)                                  # Simulation configs.

        self.f_sky = (100 * u.deg**2 / sky_sr).to(u.dimensionless_unscaled).value   # Sky fraction for simulations.
        self.xymax_cov = np.sqrt(self.f_sky * sky_sr).to(self.sim.unit)             # Side of the simulation grid.

        self.t_i = 1.0e-3 * u.s                                 # FRB intrinsic width.
        self.noise_frb = self.survey_frb.beam                   # Localization FWHM for the FRB survey.
        self.mu = 4.15 * u.ms / (u.pc/u.cm**3) * u.GHz**2       # Coefficient in the dispersion relation.

        self.ll_min = 0.1
        self.ll_max = 4.0e4

        self.interp_nstep_kk = 512
        self.interp_nstep_ll = 512

        self.interp_nstep_zz_g = 512
        self.interp_nstep_zz_e = 512
        self.interp_nstep_zz_f = 512
        self.interp_nstep_zz_f_fine = 4 * 1024
        self.interp_nstep_zz_x = 512

        self.kpad_eps = 1.0e-7
        self.zpad_interp_zmax = 0.5
        self.zpad_eps = 1.0e-6

        self.zz_gi = np.linspace(self.survey_galaxy.zmin, self.survey_galaxy.zmax, self.survey_galaxy.nspl)
        self.zz_g = np.linspace(max(self.interp_zmin, self.survey_galaxy.zmin),
                                self.survey_galaxy.zmax, self.interp_nstep_zz_g)

        self.zz_e = np.linspace(self.interp_zmin, self.interp_zmax-self.zpad_eps, self.interp_nstep_zz_e)

        self._sanity_check()

    def init_parameters(self):
        """Modifies attributes based on the input self.name."""

        if self.name == 'high_z_sdss':
            pass

        elif self.name == 'high_z_dm_cut_sdss':
            self.survey_frb = 'chime_baseline_dm_cut'

        elif self.name == 'low_z_sdss':
            self.zmax = 0.12
            self.md = 17.5
            self.fn_zmax = 0.12

        elif self.name == 'high_z_2mpz':
            self.survey_galaxy = '2mpz'
            self.zmax = 0.4

        elif self.name == 'low_z_2mpz':
            self.survey_galaxy = '2mpz'
            self.zmax = 0.12
            self.md = 17.5
            self.fn_zmax = 0.12

        elif self.name == 'high_z_desi':
            self.survey_galaxy = 'desi_baseline'
            self.zmin = 0.6
            self.zmax = 1.7

        elif self.name == 'chimefrb_c1_2mpz_fp':
            self.survey_galaxy = '2mpz_fp'
            self.survey_frb = 'chimefrb_catalog1'
            self.zmax = 0.3
            self.md = 3.733333333333333
            self.dmax = 3020.0
            self.fn_zmax = 3.0
            self.frb_par_index = [4]

        elif self.name == 'chimefrb_c1_wise_scos_svm':
            self.survey_galaxy = 'wise_scos_svm'
            self.survey_frb = 'chimefrb_catalog1'
            self.zmax = 0.5
            self.md = 2.2399999999999998
            self.dmax = 3020.0
            self.fn_zmax = 3.0
            self.frb_par_index = [5]

        elif self.name == 'chimefrb_c1_desi_bgs':
            self.survey_galaxy = 'desi_bgs'
            self.survey_frb = 'chimefrb_catalog1'
            self.zmin = 0.05
            self.zmax = 0.4
            self.md = 2.8
            self.dmax = 3020.0
            self.fn_zmax = 3.0
            self.frb_par_index = [6]

        elif self.name == 'chimefrb_c1_desi_lrg':
            self.survey_galaxy = 'desi_lrg'
            self.survey_frb = 'chimefrb_catalog1'
            self.zmin = 0.3
            self.zmax = 1.0
            self.md = 2.3092783505154637
            self.dmax = 3020.0
            self.fn_zmax = 3.0
            self.frb_par_index = [6]

        elif self.name == 'chimefrb_c1_desi_elg':
            self.survey_galaxy = 'desi_elg'
            self.survey_frb = 'chimefrb_catalog1'
            self.zmin = 0.6
            self.zmax = 1.4
            self.md = 0.99999
            self.dmax = 3020.0
            self.fn_zmax = 3.0
            self.frb_par_index = [6]

        else:
            raise RuntimeError(f'configs.init_parameters: {self.name} is not a valid config name.')

    def t_d(self, d):
        """Returns the dispersion time delay (in ms) for a given DM (implicitly in pc/cm^3)."""

        ret = 2.0 * self.mu * (d * u.pc / u.cm**3) / self.survey_frb.freq**3 * self.survey_frb.dfreq
        return ret.to(u.ms)

    def t_x(self, d):
        """Evaluates (-3 * mu * dfreq * t_d / 2 / freq^3 / (t_i^2 + t_s^2 + t_d^2)) at a given DM (pc/cm^3)."""

        c = -3.0 * self.mu * self.survey_frb.dfreq / 2.0 / self.survey_frb.freq**3

        t_d = self.t_d(d)
        t_i2 = self.t_i**2
        t_s2 = self.survey_frb.t_s**2

        ret = c * t_d / (t_i2 + t_s2 + t_d**2)
        return ret.to(u.cm**3 / u.pc).value

    def _sanity_check(self):
        """Asserts all members of the class."""

        assert isinstance(self.name, str)
        assert isinstance(self.f_sky, float) or (self.f_sky is None)
        assert isinstance(self.m_min, float) and isinstance(self.m_max, float) and (self.m_min < self.m_max)
        assert isinstance(self.interp_nstep_m_g, int) and (self.interp_nstep_m_g >= 10)
        assert isinstance(self.interp_nstep_eta, int) and (self.interp_nstep_eta >= 10)
        assert isinstance(self.interp_zmin, float) and (self.interp_zmin >= 0.0)
        assert isinstance(self.interp_zmax, float) and (self.interp_zmin < self.interp_zmax <= 20.0)
        assert isinstance(self.zmin, float) and (self.zmin >= 0.0) and (self.zmin >= self.interp_zmin)
        assert isinstance(self.zmax, float) and (self.zmin < self.zmax < self.interp_zmax)
        assert isinstance(self.md, (int, float)) and (0.9 <= self.md)
        assert isinstance(self.fn_zmax, float)
        assert self.fn_zmax >= self.zmax
        assert self.fn_zmax <= (self.interp_zmax - self.zpad_interp_zmax - self.zpad_eps)
        assert isinstance(self.dmin, float) and (self.dmin >= self.survey_frb.dmin)
        assert isinstance(self.dmax, float) and (self.dmax <= self.survey_frb.dmax)
        assert isinstance(self.survey_frb, fx.frb_configs)
        assert isinstance(self.survey_galaxy, fx.galaxy_configs)
        assert self.zmin >= self.survey_galaxy.zmin
        assert self.zmax <= self.survey_galaxy.zmax
        assert self.survey_galaxy.zbin_fmt in ('edge', 'mid')
        assert isinstance(self.sim, fx.sim_configs)
        assert isinstance(self.ll_min, float) and (self.ll_min <= 1.0)
        assert isinstance(self.ll_max, float) and (self.ll_max >= 5.0e3)
        assert isinstance(self.frb_par, list) and (len(self.frb_par) >= 1)

        if self.f_sky is not None:
            assert 0.0 < self.f_sky <= 1.0

        for par in self.frb_par:
            assert isinstance(par, list)
            assert isinstance(par[0], int) and (par[0] >= 10), 'N_frb must be >= 10.'
            for i in range(1, 6):
                assert isinstance(par[i], float) and (par[i] >= 0.0), f'FRB par[{i}] = {par[i]}.'
            assert (par[5] <= 1.0), 'psg_frac entries of self.frb_par are required to be between 0 and 1.'
            assert isinstance(par[6], bool), f'FRB par[{6}] = {par[6]}.'
            assert isinstance(par[7], float) and (self.m_min < par[7] < self.m_max), 'Invalid Mf.'

        assert isinstance(self.noise_frb, list) and (len(self.noise_frb) >= 1)
        assert isinstance(self.interp_nstep_kk, int) and (self.interp_nstep_kk >= 64)
        assert isinstance(self.interp_nstep_ll, int) and (self.interp_nstep_ll >= 64)
        assert isinstance(self.interp_nstep_zz_g, int) and (self.interp_nstep_zz_g >= 64)
        assert isinstance(self.interp_nstep_zz_e, int) and (self.interp_nstep_zz_e >= 64)
        assert isinstance(self.kpad_eps, float) and (self.kpad_eps <= 1.0e-3)
        assert isinstance(self.zpad_interp_zmax, float) and (0.1 <= self.zpad_interp_zmax <= 1.0)
        assert isinstance(self.zpad_eps, float) and (self.zpad_eps <= 1.0e-3)

        for i in (self.zz_g, self.zz_e):
            assert i[0] >= self.interp_zmin
            assert i[-1] <= self.interp_zmax


###################################################################################################


class frb_configs:
    """FRB surveys and model parameters."""

    def __init__(self, name, frb_par_index=None):
        """
        Constructor arguments:

            name: (str) name of survey.
            frb_par_index: (int) if not None, it specifies a set of FRB model parameters.

        Members:

            self.survey_name: (str) name of survey.
            self.instrument: (str) name of instrument.
            self.obj_type: (str) type of objects.
            self.freq_min: (float) min frequency of observations.
            self.freq_max: (float) max frequency of observations.
            self.f_sky: (float) fraction of sky surveyed.
            self.dmin: (float) min extragalactic DM.
            self.dmax: (float) max extragalactic DM.
            self.beam: (list) beam FWHM values which specify a set of symmetric localization errors.
            frb_configs.surveys: (static method) catalog of survey parameters.
            self.frb_par_index: (int) if not None, index of a specific FRB model.
            self.frb_par: (list (of lists)) FRB model parameters in the following order:
                          [[N_frb, p, alpha, mu, sigma, psg_frac, z_log, Mf]]
        """

        survey = self.surveys(name)

        self.survey_name = survey['survey_name']
        self.instrument = survey['instrument']
        self.obj_type = survey['obj_type']
        self.freq_min = survey['freq_min']
        self.freq_max = survey['freq_max']
        self.f_sky = survey['f_sky']
        self.dmin = survey['dmin']
        self.dmax = survey['dmax']
        self.beam = survey['beam']
        self.t_s = survey['t_s']
        self.dfreq = survey['dfreq']
        self.freq = survey['freq']

        self.frb_par_index = frb_par_index
        self.frb_par = self.models(self.frb_par_index)

    @staticmethod
    def surveys(name):
        """Catalog of survey parameters.  The 'name' argument refers to the name of a specific survey."""

        assert isinstance(name, str)

        cat = {'chime_baseline': {'survey_name': 'chime_baseline',
                                  'instrument': 'CHIME/FRB',
                                  'obj_type': 'FRB',
                                  'freq_min': 400.0 * u.MHz,
                                  'freq_max': 800.0 * u.MHz,
                                  'f_sky': (8000.0 * u.deg**2 / sky_sr).to(u.dimensionless_unscaled).value,
                                  'dmin': 0.0,
                                  'dmax': 1.0e4,
                                  'beam': [0.0*u.arcmin, 1.0*u.arcmin, 15.0*u.arcmin, 30.0*u.arcmin],
                                  't_s': 1.0e-3 * u.s,
                                  'dfreq': 400.0/16834 * u.MHz,
                                  'freq': 600.0 * u.MHz},

               'chime_baseline_dm_cut': {'survey_name': 'chime_baseline_dm_cut',
                                         'instrument': 'CHIME/FRB',
                                         'obj_type': 'FRB',
                                         'freq_min': 400.0 * u.MHz,
                                         'freq_max': 800.0 * u.MHz,
                                         'f_sky': (8000.0 * u.deg**2 / sky_sr).to(u.dimensionless_unscaled).value,
                                         'dmin': 881.3394456383085,
                                         'dmax': 1.0e4,
                                         'beam': [0.0*u.arcmin, 1.0*u.arcmin, 15.0*u.arcmin, 30.0*u.arcmin],
                                         't_s': 1.0e-3 * u.s,
                                         'dfreq': 400.0/16834 * u.MHz,
                                         'freq': 600.0 * u.MHz},

               'chimefrb_catalog1': {'survey_name': 'chimefrb_catalog1',
                                     'instrument': 'CHIME/FRB',
                                     'obj_type': 'FRB',
                                     'freq_min': 400.0 * u.MHz,
                                     'freq_max': 800.0 * u.MHz,
                                     'f_sky': 0.5558211878647882,
                                     'dmin': 0.0,
                                     'dmax': 3020.0,
                                     'beam': [0.0*u.arcmin, 1.0*u.arcmin, 5.0*u.arcmin, 10.0*u.arcmin, 15.0*u.arcmin],
                                     't_s': 1.0e-3 * u.s,
                                     'dfreq': 400.0/16834 * u.MHz,
                                     'freq': 600.0 * u.MHz}}

        assert name in cat
        return cat[name]

    @staticmethod
    def models(index):
        """
        Returns a list of (lists of) model parameters in the following order:
        [[N_frb, p, alpha, mu, sigma, psg_frac, z_log, Mf]].

        The 'index' argument (list) can be used to specify model(s).
        """

        assert isinstance(index, list) or (index is None)

        ret = [[1000, 2.0, 3.5, 4.0, 1.0, 0.0, False, 1.0e9],           # High-z FRB's x SDSS
               [1000, 2.0, 3.5, 4.0, 1.0, 1.0, False, 1.0e9],           # High-z FRB's x SDSS (incl. single-galaxy term)
               [1000, 2.0, 120.0, 6.7755, 0.6316, 0.0, True, 1.0e9],    # Low-z FRB's x SDSS (requires log-spaced z)
               [10000, 2.0, 3.5, 4.0, 1.0, 0.0, False, 1.0e9],          # High-z FRB's x DESI (10x fid N_frb)
               [323, 2.0, 3.5, 4.0, 1.0, 0.0, False, 1.0e9],            # CHIMEFRB Catalog 1 FRB's x 2MPZ_FP
               [310, 2.0, 3.5, 4.0, 1.0, 0.0, False, 1.0e9],            # CHIMEFRB Catalog 1 FRB's x WISExSCOS_SVM
               [183, 2.0, 3.5, 4.0, 1.0, 0.0, False, 1.0e9]]            # CHIMEFRB Catalog 1 FRB's x DESI-BGS/LRG/ELG

        if index is None:
            return ret
        else:
            return [ret[i] for i in index]


###################################################################################################


class galaxy_configs:
    """Galaxy surveys and model parameters."""

    def __init__(self, name):
        """
        Constructor arguments:

            name: (str) name of survey.

        Members:

            self.survey_name: (str) name of survey.
            self.instrument: (str) name of instrument.
            self.obj_type: (str) type of objects.
            self.f_sky: (float) fraction of sky surveyed.
            self.n_total: (int) total number of objects.
            self.zmin: (float) min redshift.
            self.zmax: (float) max redshift.
            self.zerr: (float) redshift error.
            self.dndz_data: (str) full path to dndz data file.
            self.zbin_fmt: (str) format of z bins in self.dndz_data.
            self.nspl: (int) number of interpolation steps used to model M_g(z).
            self.ntrap: (int) number of sampling points used when integrating dn/dz
                        between bin endpoints using the trapezoid rule.
            self.solver: (str) if not None, name of a solver for computing ng^{3d}.
        """

        survey = self.surveys(name)

        self.survey_name = survey['survey_name']
        self.instrument = survey['instrument']
        self.obj_type = survey['obj_type']
        self.f_sky = survey['f_sky']
        self.n_total = survey['n_total']
        self.zmin = survey['zmin']
        self.zmax = survey['zmax']
        self.zerr = survey['zerr']
        self.dndz_data = survey['dndz_data']
        self.zbin_fmt = survey['zbin_fmt']
        self.nspl = survey['nspl']
        self.ntrap = survey['ntrap']
        self.solver = survey['solver']

    @staticmethod
    def surveys(name):
        """
        Catalog of survey parameters.  The 'name' argument refers to the name of a specific survey.

        'dndz_data' files must have the following structure:

            col 0: lower bound of redshift shells, (*)
            col 1: upper bound of redshift shells, (*)
            col 2: dn/dz values,
            col 3: dn/dz errors (stdv),
            and commented lines that have a '#' as the first character.
            (*) or a single column of mid points.
        """

        assert isinstance(name, str)

        cat = {'sdss_dr8': {'survey_name': 'sdss_dr8',
                            'instrument': 'SDSS',
                            'obj_type': 'GALAXY',
                            'f_sky': (10269.0 * u.deg**2 / sky_sr).to(u.dimensionless_unscaled).value,
                            'n_total': 58533603,
                            'zmin': 0.0,
                            'zmax': 1.1,
                            'zerr': 0.01,
                            'dndz_data': fx.data_path('dndz_sdss_dr8.txt', envar='FRBXTDATA'),
                            'zbin_fmt': 'edge',
                            'nspl': 10,
                            'ntrap': 10,
                            'solver': None},

               '2mpz': {'survey_name': '2mpz',
                        'instrument': '2MASS',
                        'obj_type': 'GALAXY',
                        'f_sky': 0.69,
                        'n_total': 934175,
                        'zmin': 0.0,
                        'zmax': 0.4010010063648224,
                        'zerr': 0.015,
                        'dndz_data': fx.data_path('dndz_2mpz.txt', envar='FRBXTDATA'),
                        'zbin_fmt': 'edge',
                        'nspl': 11,
                        'ntrap': 10,
                        'solver': None},

               'desi_baseline': {'survey_name': 'desi_baseline',
                                 'instrument': 'DESI',
                                 'obj_type': 'ELG',
                                 'f_sky': (14000.0 * u.deg**2 / sky_sr).to(u.dimensionless_unscaled).value,
                                 'n_total': 1.792e7,
                                 'zmin': 0.6,
                                 'zmax': 1.7,
                                 'zerr': 0.01,
                                 'dndz_data': fx.data_path('dndz_desi.txt', envar='FRBXTDATA'),
                                 'zbin_fmt': 'edge',
                                 'nspl': 7,
                                 'ntrap': 10,
                                 'solver': 'Nelder-Mead'},

               '2mpz_fp': {'survey_name': '2mpz_fp',
                           'instrument': '2MASS',
                           'obj_type': 'GALAXY',
                           'f_sky': 0.64678955078125,
                           'n_total': 670442,
                           'zmin': 0.0,
                           'zmax': 0.3,
                           'zerr': 0.015,
                           'dndz_data': fx.data_path('dndz_2mpz_fp.txt', envar='FRBXTDATA'),
                           'zbin_fmt': 'edge',
                           'nspl': 12,
                           'ntrap': 10,
                           'solver': 'Nelder-Mead'},

               'wise_scos_svm': {'survey_name': 'wise_scos_svm',
                                 'instrument': 'WISE',
                                 'obj_type': 'GALAXY',
                                 'f_sky': 0.6376864314079285,
                                 'n_total': 6931441,
                                 'zmin': 0.0,
                                 'zmax': 0.5,
                                 'zerr': 0.033,
                                 'dndz_data': fx.data_path('dndz_wise_scos_svm.txt', envar='FRBXTDATA'),
                                 'zbin_fmt': 'edge',
                                 'nspl': 18,
                                 'ntrap': 10,
                                 'solver': None},

               'desi_bgs': {'survey_name': 'desi_bgs',
                            'instrument': 'BASS+MzLS',
                            'obj_type': 'BGS',
                            'f_sky': 0.11824377377827962,
                            'n_total': 5869202,
                            'zmin': 0.0,
                            'zmax': 0.6,
                            'zerr': 0.08,
                            'dndz_data': fx.data_path('dndz_desi_bgs.txt', envar='FRBXTDATA'),
                            'zbin_fmt': 'edge',
                            'nspl': 7,
                            'ntrap': 10,
                            'solver': None},

               'desi_lrg': {'survey_name': 'desi_lrg',
                            'instrument': 'BASS+MzLS',
                            'obj_type': 'LRG',
                            'f_sky': 0.11824377377827962,
                            'n_total': 2389192,
                            'zmin': 0.2,
                            'zmax': 1.2,
                            'zerr': 0.08,
                            'dndz_data': fx.data_path('dndz_desi_lrg.txt', envar='FRBXTDATA'),
                            'zbin_fmt': 'edge',
                            'nspl': 10,
                            'ntrap': 10,
                            'solver': 'Nelder-Mead'},

               'desi_elg': {'survey_name': 'desi_elg',
                            'instrument': 'BASS+MzLS',
                            'obj_type': 'ELG',
                            'f_sky': 0.054674506187438965,
                            'n_total': 5314194,
                            'zmin': 0.6,
                            'zmax': 1.4,
                            'zerr': 0.2,
                            'dndz_data': fx.data_path('dndz_desi_elg.txt', envar='FRBXTDATA'),
                            'zbin_fmt': 'edge',
                            'nspl': 10,
                            'ntrap': 10,
                            'solver': 'Nelder-Mead'}}

        assert name in cat
        return cat[name]

    @staticmethod
    def models(index):
        raise NotImplementedError('galaxy_configs.models is empty!')


###################################################################################################


class sim_configs:
    """Simulation configs"""

    def __init__(self, name):
        """
        Constructor arguments:

            name: (str) specifies the name of a simulation run.

        Members:

            self.nz: (int) number of redshift shells.
            self.unit: (astropy unit) specifies the global unit of angle on the sky.
            self.nm_halo: (int) total number of mass bins per redshift shell.
        """

        run = self.runs(name)

        self.nz = run['nz']
        self.unit = run['unit']
        self.nm_halo = run['nm_halo']
        self.force_bin_d = run['force_bin_d']

    @staticmethod
    def runs(name):
        """Contains configs for various simulation runs.  The 'name' argument refers to the name of a specific run."""

        assert isinstance(name, str)

        r = {'1h': {'run_name': '1h',
                    'nz': 100,
                    'unit': u.deg,
                    'nm_halo': 500,
                    'force_bin_d': False}}

        assert name in r
        return r[name]
