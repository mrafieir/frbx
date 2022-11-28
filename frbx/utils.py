import os
import inspect
import numpy as np
from math import erf
import scipy.integrate
import scipy.interpolate
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def dblquad(f, x_min, x_max, y_min, y_max, epsabs=0.0, epsrel=1.0e-4):
    """Provides a better interface for scipy.integrate.dblquad."""

    ret, err = np.asarray(scipy.integrate.dblquad(f, x_min, x_max, y_min, y_max, epsabs=epsabs, epsrel=epsrel))

    if err > np.abs(ret):
        raise RuntimeError('utils.dblquad: the absolute error is very large!')

    return ret


def mz(m):
    """Wraps a redshift-dependent mass limit in a log-spaced integral."""

    if isinstance(m, float):
        def lambda_m(_z):
            return m
    elif isinstance(m, scipy.interpolate.InterpolatedUnivariateSpline):
        def lambda_m(_z):
            return m(_z)
    elif callable(m):
        lambda_m = m
    else:
        raise RuntimeError('fx.mz: unsupported m!')

    return lambda z: np.log(lambda_m(z))


def read_h5(file_path, data_name):
    """Reads a dataset from an h5 file."""

    assert file_path[-3:] == '.h5'
    with FileH5(file_path, 'r') as read:
        return read[data_name][:]


def cmap(name, reverse=False, n=256):
    """Returns a home-made colormap."""

    cmaps = {'bright_day': [(255/256., 70/256., 0/256.),
                            (255/256., 217/256., 13/256.),
                            (8/256., 168/256., 197/256.),
                            (20/256., 72/256., 112/256.)]}

    ret = cmaps[name]

    if reverse:
        ret = ret[::-1]

    return colors.LinearSegmentedColormap.from_list(name, ret, n)


def plot_1d(x, y, xerr=None, yerr=None, layer=None, line_label=None, xlabel=r'$\ell$', ylabel=r'$C_{\ell}$', style='b',
            line_width=0.9, xlim=None, ylim=None, legend_loc='upper right', legend_size=8, legend_frame=False,
            legend_anchor=None, colorize_legend=False, name='temp', vector_graphic=False, fig=None, ax=None,
            xsc='log', ysc='log', minorticks=True, dpi=450, ticks_length=4.0, figsize=(3.4,2.55), axvline=None,
            axvline_linestyle=':', axv_color='lightgray', axv_lw=0.8, axh_lw=0.8, axhline=None, axhline_linestyle=':',
            axh_color='lightgray', ecolor='b', efill=None, ej=10, font_size=7.0, xlabel_pad=None, ylabel_pad=1.0,
            left_margin=0.17, linthreshy=None, linscaley=None, tick_width=0.75, fill_between=None, fill_between_alpha=0.1,
            bottom_margin=0.14, text=None, text_xpos=None, text_ypos=None, text_font_size=7.0, ysci=False, ax2=None,
            xlabel2=r'$\theta~{\rm [arcmin]}$', xlabel_pad2=None, **kwargs):
    """
    Generates a standard plot.  Its default arguments are customized for plotting angular power spectra.

    Args:

        x: (1-d array) x values.
        y: (1-d array) y values.
        xerr: (array) if not None, x error bars.
        yerr: (array) if not None, y error bars.
        layer: (int) if not None, it specifies a graphic layer for overplotting data in a single figure.
               (see `fig` and `ax` below)
        line_label: (str) line label which appears in the legend.
        xlabel: (str) label on the x axis.
        ylabel: (str) label on the y axis.
        style: (str) standard line style.
        line_width: (float) line width.
        xlim: (tuple) limits on (left, right) values.
        ylim: (tuple) limits on (bottom, top) values.
        legend_loc: (str) standard location for legends.
        legend_size: (int) font size for legends.
        legend_frame: (bool) whether to frame the legend.
        legend_anchor: (tuple) anchor from which the legend is expanded.
        colorize_legend: (bool) whether to colorize texts in the legend.
        name: (str) output file path.
        vector_graphic: (bool) whether to generate a vectorized or rasterized graphic output.
        fig: (obj) figure, returned by layer 0, to be updated with additional layers 1 and/or 2.
        ax: (obj) axes, returned by layer 0, to be updated in layer 2.
        xsc: (str) how to scale x axis, 'log' or 'linear'.
        ysc: (str) how to scale y axis, 'log', 'symlog', or 'linear'.
        minorticks: (bool) whether to show minor ticks on axes.
        ticks_length: (float) major ticks length on all axes.
        dpi: (int) dots-per-inch resolution of the output.
        figsize: (tuple) figure size.
        axvline: (list) x positions for vertical lines to be overplotted.
        axvline_linestyle: (str) a global style for the vertical lines.
        axv_color: (str) color of the vertical lines.
        axhline: (list) y positions for horizontal lines to be overplotted.
        axhline_linestyle: (str) a global style for the horizontal lines.
        axh_color: (str) color of the horizontal lines.
        ecolor: (str) color of error bars.
        efill: (1-d array) Specifies x edges for filled regions in place of error bars.
        ej: (int) number of error bars to jump over.
        font_size: (float) master font size.
        xlabel_pad: (float) if not None, label pad for x axis.
        ylabel_pad: (float) if not None, label pad for y axis.
        left_margin: (float) left margin.
        linthreshy: (float) if not None, specifies the linear range for a symlog-scale y axis.
        linscaley: (float) if not None, specifies the linear stretching factor for a symlog-scale y axis.
        tick_width: (float) Specifies tick widths.
        fill_between: (str) If a color is provided, then a histogram-like region is plotted underneath the main line.
        fill_between_alpha: (float) Specifies the alpha parameter for the 'fill_between' region.
        bottom_margin: (float) Bottom margin.
        text: (str) if not None, a text to be added to the plot.
        text_xpos: (float) horizontal position of the 'text'.
        text_ypos: (float) vertical position of the 'text'.
        text_font_size: (float) font size for the 'text'.
        ysci: (bool) whether to adopt a sci y axis.
        ax2: (tuple of functions) if not None, specifies the second axis on top.
        xlabel2: (str) label on the second x axis (on top).
        xlabel_pad2: (float) if not None, label pad for the second x axis (on top).
        **kwargs: optional matplotlib.pyplot parameters for customizing plots in plt.plot(**kwargs).

    Returns:

        fig, ax objects only if layer=0.
    """

    assert layer in (None, 0, 1, 2)
    assert isinstance(name, str)

    if (ysc == 'symlog') and ((linthreshy is None) or (linscaley is None)):
        raise AssertionError("A symlog-scale y axis requires a specific set of 'linthreshy' and 'linscaley'.")

    if layer in (None, 0, 2):
        if not vector_graphic:
            name += '.png'
        else:
            name += '.pdf'

    font_size_sys = float(matplotlib.rcParams['font.size'])
    matplotlib.rcParams['font.size'] = font_size

    if layer in (None, 0):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([left_margin, bottom_margin, 0.81, 0.81])     # left, bottom, width, height

        if text is not None:
            matplotlib.rcParams['font.size'] = text_font_size
            plt.text(text_xpos, text_ypos, text)
            matplotlib.rcParams['font.size'] = font_size

    if fill_between is not None:
        plt.fill_between(x, y, step='pre', alpha=fill_between_alpha, color=fill_between, linewidth=0.0)

    if axvline is not None:
        for i in axvline:
            plt.axvline(x=i, color=axv_color, linestyle=axvline_linestyle, lw=axv_lw)
    if axhline is not None:
        for i in axhline:
            plt.axhline(y=i, color=axh_color, linestyle=axhline_linestyle, lw=axh_lw)

    plt.plot(x, y, style, linewidth=line_width, label=line_label, **kwargs)

    if (xerr is not None) or (yerr is not None):
        if efill is None:
            _width = max(line_width-0.14, 0.14)
            plt.errorbar(x[::ej], y[::ej], xerr=xerr[::ej] if xerr is not None else xerr,
                         yerr=yerr[::ej] if yerr is not None else yerr, fmt='none',
                         ecolor=ecolor, elinewidth=_width, capsize=0.0, alpha=1.0,
                         markeredgewidth=_width)
        else:
            if yerr is not None:
                y_low = y-yerr if yerr.ndim == 1 else yerr[0]
                y_high = y+yerr if yerr.ndim == 1 else yerr[1]
            else:
                y_low = y_high = None

            plt.fill_between(efill, y_low, y_high, color=ecolor, linewidth=0.0, alpha=0.5, step='post')

    if layer == 0:
        return fig, ax

    if layer in (None, 2):
        plt.xlabel(xlabel, labelpad=xlabel_pad)
        plt.ylabel(ylabel, labelpad=ylabel_pad)

        try:
            plt.setp(ax.spines.values(), linewidth=0.9)
        except AttributeError:
            AssertionError('fig and ax args are missing')

        plt.xscale(xsc)

        if ysc == 'symlog':
            plt.yscale(ysc, linthreshy=linthreshy, linscaley=linscaley,
                       subsy=[2, 4, 6, 8])
        else:
            plt.yscale(ysc)

        if minorticks:
            plt.minorticks_on()

        axes = plt.gca()

        axes.get_xaxis().set_tick_params(which='both', length=ticks_length, width=tick_width, direction='in')
        axes.get_xaxis().set_tick_params(which='minor', length=ticks_length*0.5)
        axes.get_yaxis().set_tick_params(which='both', length=ticks_length, width=tick_width, direction='in')
        axes.get_yaxis().set_tick_params(which='minor', length=ticks_length*0.5)

        if xlim is not None:
            axes.set_xlim(xlim)
        if ylim is not None:
            axes.set_ylim(ylim)

        if ysc == 'symlog':
            plt.yticks([-10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(-plt.ylim()[0])), 1)] +
                       [0] +
                       [+10**i for i in range(int(np.log10(linthreshy))+1, int(np.log10(+plt.ylim()[1])), 1)])

        if colorize_legend:
            _legend = ax.legend()
            for (i,j) in zip(_legend.get_lines(), _legend.get_texts()):
                j.set_color(i.get_color())

        if line_label is not None:
            plt.legend(loc=legend_loc, prop={'size': legend_size}, frameon=legend_frame,
                       fancybox=False, bbox_to_anchor=legend_anchor)

        if ysci:
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if ax2 is not None:
            matplotlib.rcParams['font.size'] = font_size
            _ax2 = ax.secondary_xaxis('top', functions=ax2)
            _ax2.set_xlabel(xlabel2, labelpad=xlabel_pad2)
            _ax2.tick_params(which='both', length=ticks_length, width=tick_width, direction='in')
            _ax2.tick_params(which='minor', length=ticks_length*0.5)

        plt.savefig(name, dpi=dpi)
        plt.clf()
        plt.close('all')

        matplotlib.rcParams['font.size'] = font_size_sys

        print('plot_1d saved a figure in %s' % name)


def plot_2d(subplots, shape=(1,1), xticks=None, yticks=None, nxticks=5, nyticks=5, ticks_length=4.0,
            ticks_offset=0, name='temp', xlabel=r'$\ell$', ylabel=r'$\ell$', log_scale=False, invert_yaxis=False,
            vector_graphic=False, cmap='RdYlBu', alpha=1.0, extra_layer=None, hspace=0.12, wspace=0.12,
            legend_loc=None, dpi=450, figsize=(7,5.25), tick_width=0.75, share_cbar=True, cell_rot=0,
            share_xlabel=True, share_ylabel=True, font_size=10.0, label_size=10.0, dec=3, extend='both',
            norm=None, log_scale_cbar=True, cbar_format=True, cell_texts=None, cbar_size='5%', **kwargs):
    """
    Plots a list of 2-d arrays in a master figure.

    Args:

        subplots: an (array) or a (list) of dictionaries containing data along with corresponding configs,
                  superseding any default configs, to be subplotted.
        shape: (tuple) shape of the master figure. E.g., (3,2) means 3 subplot rows and 2 subplot columns.
        xticks: (list) tick values centered on pixels along the x axis. (*)
        yticks: (list) tick values centered on pixels along the y axis. (*)
        nxticks: (int) number of ticks along the x axis. (*)
        nyticks: (int) number of ticks along the y axis. (*)
        ticks_length: (float) major ticks length on all axes. (*)
        ticks_offset: (float) global offset for all ticks.
        name: (str) output file path.
        xlabel: (str) label on the x axis. (*)
        ylabel: (str) label on the y axis. (*)
        log_scale: (bool) whether to logarithmically scale (base 10) input values. (*)
        invert_yaxis: (bool) whether to invert y axis.
        vector_graphic: (bool) whether to generate a vectorized or rasterized graphic output.
        cmap: (str) name of a standard colormap. (*)
        alpha: (float) opacity of the map. (*)
        extra_layer: (list) optional dictionaries of 1d data to be plotted over the map. (*)
        legend_loc: (str) standard location for legends, applicable only if extra_layer is not None. (*)
        dpi: (int) dots-per-inch resolution of the output.
        figsize: (tuple) figure size.
        tick_width: (float) ticks width.
        share_cbar: (bool) whether to share a single colorbar.
        share_xlabel: (bool) whether to share xlabel among all subplots.
        share_ylabel: (bool) whether to share ylabel among all subplots.
        hspace: (float) spacing between rows of subplots.
        wspace: (float) spacing between columns of subplots.
        font_size: (float) master font size.
        label_size: (float) label font size.
        dec: (int) number of decimals in colorbar labels.
        extend: (str) whether to assume extended colorbars along 'min', 'max', 'both', or 'neither' directions.
        norm: (obj) if not None, instance of matplotlib.colors.Normalize.
        log_scale_cbar: (bool) if True, colorbar numbers, assuming to be log-spaced,
                        are displayed as if they are in the linear regime (i.e. 2.0 -> 100.0).
        cbar_format: (bool) whether to reformat colorbar texts using a nested Formatter object.
        cell_texts: (array) if not None, specifies an array of values for annotating 2-d cells.
        cell_rot: (float) angle (ccw, deg) of rotation for 'cell_texts'.
        cbar_size: (str) size of the colorbar in percent.
        **kwargs: optional matplotlib.pyplot parameters for customizing all subplots in plt.pcolor(**kwargs).
        (*) per subplot.
    """

    if isinstance(subplots, np.ndarray):
        subplots = [{'arr': subplots}]
    elif isinstance(subplots, list):
        for i in subplots:
            assert isinstance(i, dict)
    else:
        raise AssertionError('fx.utils.plot_2d: invalid subplots!')

    assert isinstance(shape, tuple)

    if extra_layer is not None:
        raise NotImplementedError('fx.utils.plot_2d does not accept any extra layers!')

    if not vector_graphic:
        name += '.png'
    else:
        name += '.pdf'

    font_size_sys = float(matplotlib.rcParams['font.size'])
    matplotlib.rcParams['font.size'] = font_size

    fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize, sharex='all', sharey='all')

    if not isinstance(ax, np.ndarray):
        ax = np.asarray([ax])

    frame = inspect.currentframe()
    frame_args, _, _, frame_values = inspect.getargvalues(frame)

    if share_xlabel:
        fig.text(0.5, 0.0005, xlabel, ha='center', fontsize=label_size)
    if share_ylabel:
        fig.text(0.0005, 0.5, ylabel, va='center', rotation='vertical', fontsize=label_size)

    for spi, sp in enumerate(subplots):
        fig_args = {}
        for arg_key in frame_args:
            try:
                _a = sp[arg_key]
            except KeyError:
                _a = frame_values[arg_key]

            fig_args.update({arg_key: _a})

        kw = {}
        for k, v in sp.items():
            if ((k not in frame_args) and
                    (k not in ('arr', 'cmap', 'alpha', 'text', 'text_xpos', 'text_ypos', 'text_font_size', 'cell_texts'))):
                kw.update({k: v})

        if fig_args['log_scale']:
            sp['arr'] = np.log10(sp['arr'])

        _subplot = ax.flat[spi].pcolor(
                 sp['arr'], cmap=fig_args['cmap'], snap=True, rasterized=True,
                 alpha=fig_args['alpha'], norm=fig_args['norm'], **kw)

        if cell_texts is not None:
            _subplot.update_scalarmappable()
            _ax = _subplot.axes

            for p, color, value in zip(_subplot.get_paths(), _subplot.get_facecolors(), cell_texts):
                if not np.isfinite(value):
                    continue
                x, y = p.vertices[:-2, :].mean(0)
                if np.all(color[:3] > 0.5):
                    color = (0.0, 0.0, 0.0)
                else:
                    color = (1.0, 1.0, 1.0)

                _ax.text(x, y, f'${value}$', ha='center', va='center', color=color, rotation=cell_rot)

        if fig_args['extra_layer'] is not None:
            for i, j in enumerate(fig_args['extra_layer']):
                for k in ('x', 'y', 'line_label', 'style'):
                    assert k in j, 'xlayer[%d] : %s is missing!' % (i, k)

                fx.plot_1d(x=j['x'], y=j['y'], line_label=j['line_label'], style=j['style'], layer=1,
                           fig=fig, ax=ax.flat[spi])

            if fig_args['legend_loc'] is not None:
                plt.legend(loc=fig_args['legend_loc'], frameon=False)

        (dy, dx) = sp['arr'].shape

        nxticks = len(fig_args['xticks']) if (fig_args['xticks'] is not None) else fig_args['nxticks']
        nyticks = len(fig_args['yticks']) if (fig_args['yticks'] is not None) else fig_args['nyticks']

        _dx_max = (dx-1) if ticks_offset else dx
        _dy_max = (dy-1) if ticks_offset else dy

        xticks_pos = np.linspace(0.0, _dx_max, nxticks)
        yticks_pos = np.linspace(0.0, _dy_max, nyticks)

        xticks_pos += ticks_offset
        yticks_pos += ticks_offset

        ax.flat[spi].set_xticks(xticks_pos, minor=False)
        ax.flat[spi].set_yticks(yticks_pos, minor=False)

        if fig_args['xticks'] is None:
            ax.flat[spi].set_xticklabels(['$%d$' % i for i in xticks_pos])
        else:
            ax.flat[spi].set_xticklabels(fig_args['xticks'])

        if fig_args['yticks'] is None:
            ax.flat[spi].set_yticklabels(['$%d$' % i for i in yticks_pos])
        else:
            ax.flat[spi].set_yticklabels(fig_args['yticks'])

        ax.flat[spi].get_xaxis().set_tick_params(which='major', length=fig_args['ticks_length'], width=tick_width,
                                                 direction='in', pad=3)

        ax.flat[spi].get_yaxis().set_tick_params(which='major', length=fig_args['ticks_length'], width=tick_width,
                                                 direction='in', pad=3)

        if not share_xlabel:
            ax.flat[spi].set_xlabel(fig_args['xlabel'], fontsize=fig_args['label_size'])
        if not share_ylabel:
            ax.flat[spi].set_ylabel(fig_args['ylabel'], fontsize=fig_args['label_size'])

        ax.flat[spi].set_aspect = 'auto'
        plt.setp(ax.flat[spi].spines.values(), linewidth=0.9)

        try:
            matplotlib.rcParams['font.size'] = sp['text_font_size']
            ax.flat[spi].text(sp['text_xpos'], sp['text_ypos'], sp['text'])
        except KeyError:
            pass

        matplotlib.rcParams['font.size'] = font_size

        def reformat_cbar_label(x, pos, _pow=fig_args['log_scale_cbar'], _dec=fig_args['dec']):
            if _pow:
                x = 10.0**x

            # Being explicit here.
            if int(_dec) == 2:
                return r'$%.2f$' % x
            elif int(_dec) == 1:
                return r'$%.1f$' % x
            else:
                return r'$%.3f$' % x

        if not share_cbar:
            _g = make_axes_locatable(ax.flat[spi])
            _g_cax = _g.append_axes('top', size=cbar_size, pad=0.0)
            cbar = fig.colorbar(
                 _subplot, cax=_g_cax, orientation='horizontal', extend=fig_args['extend'],
                 format=matplotlib.ticker.FuncFormatter(reformat_cbar_label) if cbar_format else None)
            cbar.outline.set_linewidth(0.9)
            cbar.ax.tick_params(axis='x', which='both', width=0.5, direction='in')
            _g_cax.xaxis.set_ticks_position('top')

    if invert_yaxis:
        ax[0].invert_yaxis()

    plt.tight_layout()

    if share_cbar:
        _g = make_axes_locatable(ax[0])
        _g_cax = _g.append_axes('right', size=cbar_size, pad=0.1)
        cbar = fig.colorbar(_subplot, cax=_g_cax, orientation='vertical', extend=fig_args['extend'],
                            format=matplotlib.ticker.FuncFormatter(reformat_cbar_label) if cbar_format else None)
        cbar.outline.set_linewidth(0.9)
        cbar.minorticks_on()
        cbar.ax.tick_params(which='both', axis='y', width=0.5, direction='in')

    plt.tight_layout()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    fig.savefig(name, dpi=dpi)

    fig.clf()
    plt.close('all')

    matplotlib.rcParams['font.size'] = font_size_sys

    print('plot_2d saved a figure in %s' % name)


def gaussian(mu, sigma, a=None):
    """Returns a 1-d gaussian function."""

    assert isinstance(mu, float)
    assert isinstance(sigma, float) and (0.0 <= sigma)

    a = (1.0/np.sqrt(2*np.pi)/sigma) if a is None else a

    return lambda x: a * np.exp(-((mu-x)/sigma)**2/2)


def lognorm(mu, sigma, cdf=False):
    """
    Returns the probability distribution function of a log-normal distribution

        PDF(x) = 1/(x*(2*pi*sigma^2)^0.5) * exp[ -(log(x)-mu)^2/(2*sigma^2) ],

    or its cumulative distribution function

        CDF(x' <= x) = 1/2 * [ 1 + erf( (log(x)-mu) / (2^0.5 * sigma) ) ].

    The pdf is prone to failure in some corner cases. Assert returned values before using them in production.

    Args:

        mu: (float) mean of the underlying normal distribution.
        sigma: (float) standard deviation of the underlying normal distribution.
        cdf: (bool) if True, returns the CDF instead.

    Returns:

        lambda of the pdf (or cdf).
    """

    s = np.sqrt(2.0) * sigma

    if not cdf:
        return lambda x: np.exp(-((np.log(x)-mu)/s)**2.0) / (x*s*np.sqrt(np.pi)) if 0.0 < x else 0.0
    else:
        return lambda x: 0.5 * (1.0 + erf((np.log(x)-mu)/s)) if 0.0 < x else 0.0


def upsample(arr, new_y, new_x):
    """Upsamples a 2-d array.  Copied from kmsmith137/rf_pipelines."""

    (old_y, old_x) = arr.shape
    assert new_y % old_y == 0
    assert new_x % old_x == 0

    (r_y, r_x) = (new_y // old_y, new_x // old_x)
    ret = np.zeros((old_y, r_y, old_x, r_x), dtype=arr.dtype)

    for i in range(r_y):
        for j in range(r_x):
            ret[:, i, :, j] = arr[:, :]

    return np.reshape(ret, (new_y, new_x))


def rpl_path(x, r='0', d='_'):
    """Replaces the str after the unique character 'd' in a str 'x' by another str 'r'."""

    assert isinstance(x, str) and isinstance(r, str) and isinstance(d, str)

    ret = x.split(d)[:-1]

    assert 1 <= len(ret), f'utils.rpl_path: missing the character {d}'

    ret.append(r)

    try:
        ret = '_'.join(ret)
    except TypeError as err:
        raise AssertionError('%s: %s' % (err, ret))

    return ret


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


####################################   l_binning helpers   ####################################


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
        return np.array([ np.mean(arr[d[i]:d[i+1] ]) for i in range(self.nbins) ])


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


def downsample_cl(ell, cl, nl):
    """
    Downsamples an array of angular power spectra along its last axis.

    Args:

        ell: (1-d array) angular wavenumbers.
        cl: (array) angular power spectra.
        nl: (int) number of modes which will be downsampled to a bin.

    Returns:

        tuple (l, cl), mean wavenumbers in bandpowers, along with the corresponding bandpower values.

    Raises:

        AssertionError: invalid input args.
    """

    assert isinstance(ell, np.ndarray)
    assert isinstance(cl, np.ndarray)
    assert (1 <= ell.size == cl.shape[-1])
    assert isinstance(nl, int) and (nl > 0)

    if ell[0] == 0.0:
        (ell, cl) = (ell[1:], cl[...,1:])

    if nl == 1:
        return ell, cl

    n_x = ell.size // nl
    assert n_x >= 1, 'nl is too large or ell is too short!'

    lx = np.zeros(n_x)
    s = list(cl.shape)
    s[-1] = n_x
    clx = np.zeros(s)

    trim = False
    for i in range(n_x):
        left = i * nl
        right = left + nl

        if ell.size < right:
            trim = True

        _s = np.sum(ell[left:right])

        lx[i] = _s / nl
        clx[...,i] = np.sum(ell[left:right] * cl[...,left:right], axis=-1) / _s

    if trim:
        lx = lx[:-1]
        clx = clx[...,:-1]

    return lx, clx


def bin_edges(ell):
    """Returns edges of bandpowers centered on input values."""

    assert isinstance(ell, np.ndarray) and np.isfinite(ell).all() and np.all(0.0 <= ell)
    assert (ell.ndim == 1) and (2 < ell.size)
    assert ell.dtype in (int, float, np.float64)

    dl = np.diff(ell) / 2.0

    l_low = ell - np.append(dl[0], dl)
    l_high = ell + np.append(dl, dl[-1])

    return l_low, l_high


def lspace(xmin, xmax, n, log, e=1.0e-7):
    """Returns a 1-d array of values, uniformly (log-)spaced over the range (xmin, xmax)."""

    if log:
        if xmin == 0.0:
            return np.append(0.0, fx.logspace(e,xmax,n))
        else:
            return fx.logspace(xmin, xmax, n)
    else:
        return np.linspace(xmin, xmax, n)
