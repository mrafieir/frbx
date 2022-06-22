import os
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
import pickle
import healpy
import astropy.units
import astropy.coordinates
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy import units as u
from astropy.time import Time
import pytz
from h5py import File as FileH5
import chime_frb_api
import frbx as fx

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True


####################################   Misc utility functions   ####################################


def ch_frb_master():
    """Returns an instance of the FRB master and its authorization token."""

    master = chime_frb_api.frb_master.FRBMaster()
    master.API.authorize()

    auth = {"authorization": master.API.access_token}

    return master, auth


def mjd_to_localhour(t, timezone='US/Pacific'):
    """Modified Julian Date -> local hour."""

    t = Time(t, format='mjd')
    tz = pytz.timezone(timezone)

    try:
        len(t)
    except TypeError:
        t = [t]

    ret = []
    for i in t:
        t_local = i.to_datetime(timezone=tz)

        h = t_local.hour
        m = t_local.minute
        s = t_local.second

        _t = h + ((m + (s/60.0)) / 60.0)
        ret.append(_t % 24)

    return np.array(ret)


def radec_to_altaz(ra, dec, mjd, lat='49.32d', lon='-119.62d', height=545.0*u.m, timezone='US/Pacific'):
    """(ra,dec) -> (alt,az) [in deg]"""

    loc = EarthLocation(lat=lat, lon=lon, height=height)

    t = Time(mjd, format='mjd')

    tz = pytz.timezone(timezone)
    t.to_datetime(timezone=tz)

    a = AltAz(location=loc, obstime=t)
    c = SkyCoord(ra*u.deg, dec*u.deg)

    ret = c.transform_to(a)
    return ret.alt.value, ret.az.value


def showfig(doc, kw=None):
    """
    The 'doc' argument should either be None (to show a plot interactively), or a
    handout.Handout instance (to show a plot in a web-browsable output directory).

    We use handout.Handout objects extensively, to organize pipeline plots/outputs.
    For more info about handout, see: https://pypi.org/project/handout/
    """

    if kw is None:
        kw = {}

    if doc is not None:
        doc.add_figure(plt.figure(num=1, **kw))
        doc.show()
    else:
        plt.show()

    plt.clf()


def write_pickle(filename, obj):
    assert filename.endswith('.pkl')
    os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f'wrote {filename}')   # if you get a syntax error on this line, you need python 3.6+!


def read_pickle(filename):
    assert filename.endswith('.pkl')
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f'read {filename}')
    return obj


def read_h5(file_path, data_name):
    """Reads a dataset from an h5 file."""

    assert file_path[-3:] == '.h5'
    with FileH5(file_path, 'r') as read:
        return read[data_name][:]


def read_arr(file_path, **kwargs):
    """Reads a numpy array from a file."""

    assert file_path[-4:] == '.npy'
    return np.load(file_path, **kwargs)


def write_arr(file_path, arr):
    """Writes a numpy array to disk."""

    assert file_path[-4:] == '.npy'
    np.save(file_path, arr)


def logspace(xmin, xmax, n=None, dlog=None):
    """
    Returns a 1-d array of values, uniformly log-spaced over the range (xmin, xmax).
    
    The spacing can be controlled by setting either the 'n' argument (number of points)
    or 'dlog' (largest allowed spacing in log(x)).

    This function is morally equivalent to np.logspace(), I just like the syntax better!
    """

    assert 0 < xmin < xmax
    assert (n is None) or (n >= 2)

    if (n is None) and (dlog is not None):
        n = int((np.log(xmax) - np.log(xmin)) / dlog) + 2
    elif (n is None) or (dlog is not None):
        raise RuntimeError("logspace: exactly one of 'n', 'dlog' should be None")

    ret = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    ret[0] = xmin   # get rid of roundoff error
    ret[-1] = xmax  # get rid of roundoff error

    return ret


def slicer(xmin, xmax, n=None, log_spaced=False, dlog=None):
    """
    This helper function slices a range of numbers.

    Args:

        xmin: (int or float) min value of the range.
        xmax: (int or float) max value of the range.
        n: (int) Specifies the number of slices if 'dlog' is None.
        log_spaced: (bool) if True, then intervals are semi-log-spaced in which
                    case the zeroth slice starts at 0 if and only if xmin is 0.
        dlog: (float) largest allowed spacing in log(x).

    Returns:

        list of 1-d arrays specifying min and max values which define slices.

    Raises:

        AssertionError: invalid input args.
        RuntimeError: invalid combination of input args.
    """

    assert isinstance(xmin, (int, float))
    assert isinstance(xmax, (int, float))
    assert xmin < xmax
    assert isinstance(log_spaced, bool)

    if (not log_spaced) and (n is None):
        raise RuntimeError("slicer: 'n' must be specified.")

    if n == 1:
        return list((np.asarray([xmin]), np.asarray([xmax])))
    else:
        if not log_spaced:
            x = np.linspace(xmin, xmax, (n+1))
        else:
            _n = n + 1 if (dlog is None) else None
            x = logspace(xmin, xmax, _n, dlog)

            if xmin == 0.0:
                x[0] = 0.0

        return list((x[0:-1], x[1:]))


def data_path(filename, envar='FRBXDATA', mode='w'):
    """Returns an absolute path to an FRBX data file."""

    assert isinstance(filename, str)
    assert isinstance(envar, str)
    assert envar in os.environ, '%s has not been defined!' % envar
    assert mode in ('r', 'w')

    r = os.environ[envar]

    if not os.path.exists(r):
        raise RuntimeError('data_path: invalid path for %s: %s' % (envar, r))

    f = os.path.join(r, filename)

    if (not os.path.exists(f)) and (mode == 'r'):
        raise RuntimeError('data_path: %s does not exist in %s' % (filename, f))

    return f


def quad(f, x_min, x_max, epsabs=0.0, epsrel=1.0e-4, **kwargs):
    """Provides a customized interface for calling scipy.integrate.quad."""

    ret, err = np.asarray(scipy.integrate.quad(f, x_min, x_max, epsabs=epsabs, epsrel=epsrel, **kwargs))

    if err > np.abs(ret):
        raise RuntimeError('fx.utils.quad: the absolute error is very large!')

    return ret


def spline(x_vec, y_vec, z_vec=None, s_grid=None, ext=2, deg=3):
    """
    Returns scipy.interpolate.InterpolatedUnivariateSpline (2-d) if 'z_vec' is None,
    or scipy.interpolate.RectBivariateSpline (2-d) if 'z_vec' is a 2-d grid,
    or scipy.interpolate.RegularGridInterpolator if 's_grid' is a 2-d grid.
    """

    if ((z_vec is None) or (z_vec.ndim == 2)) and (s_grid is not None):
        raise RuntimeError("fx.utils.spline: a 1-d 'z_vec' is required for interpolating a 2-d s_grid.")

    if z_vec is None:
        return scipy.interpolate.InterpolatedUnivariateSpline(x_vec, y_vec, ext=ext, k=deg)
    elif z_vec.ndim == 2:
        return scipy.interpolate.RectBivariateSpline(x_vec, y_vec, z_vec, kx=deg, ky=deg)
    else:
        fill = None if (ext == 0) else np.nan
        return scipy.interpolate.RegularGridInterpolator((x_vec, y_vec, z_vec), s_grid, fill_value=fill)


def assert_eid_unique(a):
    """Checks for any eid duplicates."""

    eid_sorted, counts = np.unique(a, return_counts=True)

    x = eid_sorted[counts > 1]
    if x.size != 0:
        raise RuntimeError(f'assert_eid_unique: duplicate events have been detected!\n'
                           f'Check the following event IDs: {x}')


def nanomaggies_to_mag(f, t):
    """Converts nanomaggies to extinction-corrected magnitude."""

    assert f.shape == t.shape
    mask = np.logical_and(f > 0.0, t > 0.0)

    ret = np.full_like(f, -99.0)

    ret[mask] = np.log10(f[mask]/t[mask]) - 9.0
    ret *= -2.5

    return ret


def gumbel_pdf(x, mu, beta):
    """Gumbel distribution."""

    p = (mu-x) / beta
    ret = np.exp(p - np.exp(p))
    ret /= beta

    return ret


def gev_t(x, mu, sigma, chi):
    """Returns the 't' parameter in the generalized extreme value distribution."""

    if chi > 0.0:
        assert (mu-sigma/chi) <= x < np.inf
    elif chi == 0.0:
        assert np.isfinite(x)
    else:
        assert np.inf < x <= (mu-sigma/chi)

    q = (x-mu) / sigma

    if chi:
        t = (1.0 + chi*q)**(-1.0/chi)
    else:
        t = np.exp(-q)

    return t


def gev_pdf(x, mu, sigma, chi):
    """Generalized extreme value PDF."""
    t = gev_t(x, mu, sigma, chi)
    return t**(chi+1) * np.exp(-t) / sigma


def gev_cdf(x, mu, sigma, chi):
    """Generalized extreme value CDF."""
    t = gev_t(x, mu, sigma, chi)
    return np.exp(-t)


################################   Coordinate conversion utilities   ###############################


def sanity_check_lon_lat_arrays(lon_deg, lat_deg):
    """
    The 'lon_deg' and 'lat_deg' arguments should be arrays of the same shape.

    Astronomer's conventions are assumed: angles are in degrees, the north pole
    is lat=90, south pole is lat=-90.

    Can be called with either:

        (lon_deg, lat_deg) = (ra, dec)   [ equatorial ]
        (lon_deg, lat_deg) = (l, b)      [ galactic ]
    """

    assert lon_deg.shape == lat_deg.shape
    assert np.all(lon_deg >= 0.0)
    assert np.all(lon_deg <= 360.0)
    assert np.all(lat_deg >= -90.0)
    assert np.all(lat_deg <= 90.0)


def convert_ra_dec_to_l_b(ra_deg, dec_deg, frame='icrs'):
    """
    The 'ra_deg' and 'dec_deg' arguments should be arrays of the same shape.
    This routine is inefficient if called with large array size, but very slow if called in a loop.

    TODO: how to unit-test this?

    CHIME FRB (ra, dec) values are in icrs.
    """

    sanity_check_lon_lat_arrays(ra_deg, dec_deg)

    c = astropy.coordinates.SkyCoord(ra=ra_deg, dec=dec_deg, frame=frame, unit='deg')
    c = c.galactic

    l_deg = c.l.deg
    b_deg = c.b.deg

    return l_deg, b_deg


def convert_l_b_to_ra_dec(l_deg, b_deg, frame='icrs'):
    """
    The 'ra_deg' and 'dec_deg' arguments should be arrays of the same shape.
    This routine is inefficient if called with large array size, but very slow if called in a loop.
    
    TODO: how to unit-test this?
    """

    sanity_check_lon_lat_arrays(l_deg, b_deg)

    c = astropy.coordinates.SkyCoord(l=l_deg, b=b_deg, frame='galactic', unit='deg')

    if frame == 'icrs':
        c = c.icrs
    elif frame == 'fk5':
        c = c.fk5
    else:
        raise RuntimeError("convert_l_b_to_ra_dec currently supports 'icrs' and 'fk5 frames.")

    ra_deg = c.ra.deg
    dec_deg = c.dec.deg

    return ra_deg, dec_deg


def mod_shift(x, dx, xmax):
    """Shifts and wraps 'x' by 'dx' within the range (0, 'xmax')."""

    return (x + dx) % xmax


#########################################   Healpix utils   #######################################


def make_catalog_mask_from_healpix_mask(nside, l_deg, b_deg, mask):
    """
    The (l_deg, b_deg) args should be 1-d arrays of the same length N,
    representing a catalog in galactic coordinates (l,b).

    The 'mask' argument specifies a healpix mask which will be down/up-graded
    to 'nside' resolution.

    Returns a 1-d array of length N, containing True (unmasked) and False (masked).
    """

    sanity_check_lon_lat_arrays(l_deg, b_deg)

    mask = healpy.pixelfunc.ud_grade(mask, nside)
    pix_arr = healpy.pixelfunc.ang2pix(nside, l_deg, b_deg, lonlat=True)

    if not isinstance(pix_arr, np.ndarray):
        pix_arr = np.array([pix_arr])

    ret = mask[pix_arr]

    return ret.astype(bool)


def make_healpix_map_from_catalog(nside, l_deg, b_deg, weight=1.0, interpolate=False):
    """
    The (l_deg, b_deg) args should be 1-d arrays of the same length N,
    representing a catalog in galactic coordinates (l,b).  If the 'interpolate'
    arg is True, then a CIC-like weighting scheme is assumed, enabling a bilinear
    interpolation.

    Returns healpix map containing (weight) * (number of objects in each pixel).

    Note: throughout the pipeline, healpix maps are always in galactic coordinates!
    """

    sanity_check_lon_lat_arrays(l_deg, b_deg)
    assert isinstance(interpolate, bool)

    npix = healpy.nside2npix(nside)
    ret = np.zeros(npix)

    if not interpolate:
        pix_arr = healpy.pixelfunc.ang2pix(nside, l_deg, b_deg, lonlat=True)

        if not isinstance(pix_arr, np.ndarray):
            pix_arr = np.array([pix_arr])

        for pix in pix_arr:
            ret[pix] += weight
    else:
        pix_arr, w = healpy.pixelfunc.get_interp_weights(nside, l_deg, b_deg, lonlat=True)

        p = np.nditer(pix_arr, flags=['multi_index'])
        while not p.finished:
            i = p.multi_index
            j = p[0]
            ret[j] += (weight * w[i])
            p.iternext()

    return ret


def make_healpix_l_b_maps(nside):
    """
    Computes galactic coordinates for each healpix pixel, and returns this
    data as a pair of Healpix maps (l,b).  Each map is an array of length 
    N_pix containing angles in degrees.

    Note: healpix maps are always in galactic coordinates!
    """

    npix = healpy.pixelfunc.nside2npix(nside)
    l_deg, b_deg = healpy.pixelfunc.pix2ang(nside, np.arange(npix), lonlat=True)
    return l_deg, b_deg


def make_healpix_ra_dec_maps(nside):
    """
    Computes equatorial coordinates for each healpix pixel, and returns this
    data as a pair of Healpix maps (ra,dec).  Each map is an array of length
    N_pix containing angles in degrees.

    Note: healpix maps are always in galactic coordinates!
    """

    l_deg, b_deg = make_healpix_l_b_maps(nside)
    ra_deg, dec_deg = convert_l_b_to_ra_dec(l_deg, b_deg)
    return ra_deg, dec_deg


def show_healpix_map(m, doc=None):
    """
    The 'doc' argument should either be None (to show a plot interactively), or a
    handout.Handout instance (to show a plot in a web-browsable output directory).
    """

    # I decided to plot healpix maps with rot=(180,0,0) and flip='geo'.
    # With this convention, the healpix map will visually resemble a scatterplot
    # of the catalog with (l,b) on the (x,y) axes.  (For example, such a scatterplot
    # is produced by frbx_catalog_base.plot_ra_dec(), see below.)

    healpy.visufunc.mollview(m, fig=1, rot=(180,0,0), flip='geo')
    showfig(doc)


def make_bthresh_mask(nside, bthresh, l=(0.0,360.0), bcut_min=-90.0):
    """
    Returns a simple galactic mask, as a healpix map.  Pixels with |b| < 'bthresh' (inside
    the optional 'l' domain) and b < 'bcut_min' (optional) are masked.  The 'bthresh' argument
    should be in degrees.  Intended for use in the 'galaxy_overdensity' constructor.
    """

    assert len(l) == 2

    l_deg, b_deg = make_healpix_l_b_maps(nside)

    l_mask = np.logical_and(l_deg >= l[0], l_deg <= l[1])
    b_mask = np.abs(b_deg) >= bthresh

    mask = np.logical_and(l_mask, b_mask)
    mask = np.logical_and(mask, (b_deg >= bcut_min))

    return np.array(mask, dtype=np.float)   # decided to convert bool -> float here


def apodize_mask():
    """Apodizes a mask."""

    raise RuntimeWarning('apodize_mask is yet to be implemented!  Returning the input mask..')


def convert_hit_to_mask(nside, m, apodize=False):
    """
    Converts a hit-like map to a mask.  If 'apodize' is True, then hard edges are smoothed
    by a tapering function, resulting in values between 0 and 1.  Otherwise, a boolean array
    containing 0.0 (masked) or 1.0 (unmasked) is returned.
    """

    assert np.all(m >= 0.0)

    # Upgrade/degrade.
    ret = healpy.pixelfunc.ud_grade(m, nside)

    # Convert to a boolean map.
    ret[ret != 0.0] = 1.0

    if apodize:
        ret = apodize_mask()

    return ret


def get_mask(nside, filename, bthresh=0):
    """Returns a standard mask."""

    m = healpy.read_map(filename, verbose=False)
    print(f'read {filename}')

    ret = healpy.pixelfunc.ud_grade(m, nside)
    ret[ret != 0.0] = 1.0

    if bthresh:
        ret *= make_bthresh_mask(nside, bthresh)

    return ret


def get_2mass_mask(nside, filename=data_path('archive/maps/2mpz/mask_2mpz_david.fits')):
    """
    Returns the 2MASS mask from https://arxiv.org/abs/1412.5151.
    The return value is a 1-d array (nside=4096) containing 0.0 (masked) or 1.0 (unmasked).
    """

    return get_mask(nside, filename)


def compute_desilis_dr8_hit(nside=1024, destriped=False, mode='lrg'):
    """
    Returns a hit-like map for the DESI Legacy Imaging Survey (DR8) catalog based on randoms. If 'destriped' is True,
    then the stripe between MzLS+BASS and DECaLS (32 <= dec <= 34) is masked out.
    """

    _cat = fx.galaxy_catalog_desilis_dr8()
    cat = _cat(f'randoms_{mode}', maskbit=False, expcut=False)

    pixarea = (4*np.pi) / healpy.pixelfunc.nside2npix(nside)

    ret = make_healpix_map_from_catalog(nside, cat.l_deg, cat.b_deg, weight=1.0/pixarea)
    ret /= np.mean(ret)

    # Smooth out by a Gaussian kernel.
    fwhm = healpy.pixelfunc.nside2resol(nside) * 2
    ret = healpy.sphtfunc.smoothing(ret, fwhm=fwhm, iter=3, verbose=False)
    ret /= np.mean(ret)
    ret[ret < 0.0] = 0.0

    # Apply maskbits and exposure cuts.
    cat = _cat(f'randoms_{mode}', maskbit=True, expcut=True)
    m = make_healpix_map_from_catalog(nside//2, cat.l_deg, cat.b_deg, weight=1.0/pixarea)
    m /= np.mean(m)
    m = convert_hit_to_mask(nside, m, apodize=False)
    ret *= m

    if destriped:
        # Mask out the stripe between MzLS+BASS and DECaLS.
        ra, dec = fx.utils.make_healpix_ra_dec_maps(nside)

        stripe = np.logical_and((dec >= 32.0), (dec <= 34.0))
        ret[stripe] = 0.0

    return ret


def get_desilis_dr8_mask(nside=2048, filename=None, bthresh=17, bcut_min=0.0):
    """Returns the DESI Legacy Imaging Survey (DR8) mask."""

    if filename is None:
        return make_bthresh_mask(nside, bthresh, bcut_min=bcut_min)
    else:
        return get_mask(nside, filename, bthresh)


def get_wise_scos_mask(nside=256, filename=data_path('archive/maps/wise_scos/WISExSCOSmask.fits'),
                       bthresh=17, trim=True):
    """Returns a customized WISExSuperCosmos mask."""

    ret = get_mask(nside, filename, bthresh)

    if trim:
        c = make_bthresh_mask(nside, bthresh=20.0, l=(0.0,30.0))
        c += make_bthresh_mask(nside, bthresh=18.0, l=(30.0,60.0))
        c += make_bthresh_mask(nside, bthresh=0.0, l=(60.0,300.0))
        c += make_bthresh_mask(nside, bthresh=18.0, l=(300.0,330.0))
        c += make_bthresh_mask(nside, bthresh=20.0, l=(330.0,360.0))
        ret *= c.astype(bool)

    return ret


####################################   l_binning helper class   ####################################


class l_binning:
    """
    Simple helper class representing l-bins over multipole range 2 <= l <= lmax.

    Easiest to explain by example: if the 'l_delim' constructor argument is [2, 3, 5, 9],
    we get three bins with (2 <= l < 3), (3 <= l < 5), and (5 <= l < 9).  (Thus lmax=8.)
    """

    def __init__(self, l_delim):
        self.l_delim = np.array(l_delim, dtype=np.int)

        assert self.l_delim.ndim == 1
        assert self.l_delim[0] == 2
        assert np.all(self.l_delim[:-1] < self.l_delim[1:])

        self.lmax = self.l_delim[-1] - 1
        self.nbins = len(self.l_delim) - 1
        self.l_vals = self.bin_average(np.arange(self.lmax+1))

    def bin_average(self, arr):
        """
        The argument should be a 1-d array arr[l] of length (lmax+1).
        Returns a 1-d array of length nbins, by averaging over each l-bin.
        """

        assert arr.shape == (self.lmax+1,)

        d = self.l_delim
        return np.array([ np.mean(arr[d[i]:d[i+1]]) for i in range(self.nbins) ])


def simple_l_binning(lmax, dlog=0.3):
    """
    Returns an instance of class l_binning, containing log-spaced bins
    over the range 2 <= l <= lmax.  The 'dlog' argument is the difference
    in log(l) between endpoints of each bin.
    """

    assert lmax >= 2
    assert dlog > 0.0

    l_delim = logspace(2, lmax+1, dlog=dlog)
    l_delim = np.round(l_delim).astype(np.int)
    l_delim = np.unique(l_delim)

    ret = l_binning(l_delim)
    assert ret.lmax == lmax
    return ret
