#!/usr/bin/env python3
"""# Testing the nside parameter"""

import argparse
import handout
import numpy as np
import matplotlib.pyplot as plt
import frbx as fx

"""# Parameters """

parser = argparse.ArgumentParser(description='Testing the nside parameter.')

parser.add_argument('--frb_catalog', help="Path to an FRB catalog (.json).", type=str, default='chime_frb/catalog_081920.json')
parser.add_argument('--frb_flagged', help="Path to a .npy file specifying flagged FRBs.", type=str, default='chime_frb/ignore_081920.npy')
parser.add_argument('--frb_mocks', help="Path to a .npy file containing FRB mocks.", type=str, default='chime_frb/mocks_c081920_i081920.npy')
parser.add_argument('--galaxy_catalog', help="Name of a galaxy catalog.", type=str, default='wise_scos_svm')
parser.add_argument('--zmin', help='Min redshift on galaxy side.', type=float, default=0.35)
parser.add_argument('--zmax', help='Max redshift on galaxy side.', type=float, default=0.37)
parser.add_argument('--disable_single_burst', help='If set, then all repeating events are ignored.', action='store_true')

args = parser.parse_args()
print(args)

nside = [512, 1024, 2048, 4096, 8192]
lmax = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

assert np.all(np.array(lmax) == lmax[0]), 'We currently assume the same lmax throughout.'

outdir = 'archive/outputs/test-nside'
doc = handout.Handout(fx.data_path(outdir, mode='w'), title='test-nside')

cat_f_path = fx.data_path(f'archive/catalogs/{args.frb_catalog}')

if cat_f_path.endswith('.json'):
    _single_burst = not args.disable_single_burst
    _flagged = None if (args.frb_flagged == '') else fx.data_path(f'archive/catalogs/{args.frb_flagged}')
    cat_f = fx.frb_catalog_json(fx.data_path(f'archive/catalogs/{args.frb_catalog}'), single_burst=_single_burst,
                                morphology_in=None, morphology_ex=None, flagged=_flagged,
                                mocks=fx.data_path(f'archive/catalogs/{args.frb_mocks}'))
else:
    raise RuntimeError('test-nside: invalid frb catalog!')

if args.galaxy_catalog == '2mpz':
    cat_g = fx.galaxy_catalog_2mpz()
    g_randcat = None
    mask = [ fx.get_2mass_mask(n) for n in nside ]

elif 'desilis_dr8' in args.galaxy_catalog:
    _cat_g = fx.galaxy_catalog_desilis_dr8()
    if 'zhou20_lrg' in args.galaxy_catalog:
        cat_g = _cat_g('zhou20_lrg')
    elif 'zhou20_all' in args.galaxy_catalog:
        cat_g = _cat_g('zhou20_all')
    else:
        raise RuntimeError('test-nside: invalid desilis_dr8 mode!')

    g_randcat = None        #_cat_g('randoms')
    _destriped = '_destriped'
    _filename = fx.data_path(f'archive/maps/desilis_dr8/mask_desilis_dr8{_destriped}.fits')
    mask = [ fx.get_desilis_dr8_mask(n, filename=_filename) for n in nside ]

elif args.galaxy_catalog == 'wise_scos':
    cat_g = fx.galaxy_catalog_wise_scos()
    g_randcat = None
    mask = [ fx.get_wise_scos_mask(n) for n in nside ]

elif args.galaxy_catalog == 'wise_scos_svm':
    cat_g = fx.galaxy_catalog_wise_scos_svm(mode='g')
    g_randcat = None
    mask = [ fx.get_wise_scos_mask(n) for n in nside ]

else:
    raise RuntimeError('test-nside: invalid galaxy_catalog arg!')

cat_g_sub = cat_g.make_zbin_subcatalog(args.zmin, args.zmax)

"""### Overdensity fields, mask applied."""

deltaf = []
m_f = []
fsky_f = []
nf2d = []

deltag = []
m = []
fsky = []
ng2d = []

deltag_sub = []
m_sub = []
fsky_sub = []
ng2d_sub = []

for i, (n,l) in enumerate(zip(nside,lmax)):
    f = fx.frb_overdensity(cat_f, n, l, rmult=10000)
    deltaf.append(f)

    g = fx.galaxy_overdensity(cat_g, n, l, healpix_mask=mask[i])
    deltag.append(g)

    g_sub = fx.galaxy_overdensity(cat_g_sub, n, l, healpix_mask=mask[i])
    deltag_sub.append(g_sub)

    m_f.append(np.mean(f.deltaf_map))
    m.append(np.mean(g.deltag_map))
    m_sub.append(np.mean(g_sub.deltag_map))

    fsky_f.append(f.fsky)
    fsky.append(g.fsky)
    fsky_sub.append(g_sub.fsky)

    nf2d.append(f.nf_2d)
    ng2d.append(g.ng_2d)
    ng2d_sub.append(g_sub.ng_2d)

doc.add_text('mean deltaf:', m_f)
doc.add_text('mean deltag:', m)
doc.add_text('mean deltag (sub):', m_sub)

doc.add_text('fsky_f:', fsky_f)
doc.add_text('fsky:', fsky)
doc.add_text('fsky (sub):', fsky_sub)

doc.add_text('nf_2d:', nf2d)
doc.add_text('ng_2d:', ng2d)
doc.add_text('ng_2d (sub):', ng2d_sub)

doc.show()

"""### C_l^{ff}."""

b = fx.simple_l_binning(deltaf[0].lmax)
clff0 = b.bin_average(deltaf[0].clff)

for (f,n,l) in zip(deltaf,nside,lmax):
    clff= b.bin_average(f.clff)
    plt.plot(b.l_vals, clff/clff0, label=f'({n},{l})')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{ff}/C_\ell^{(ff,0)}$')
plt.legend(loc='upper right').draw_frame(False)
plt.savefig('clff-test-nside.pdf', dpi=300, bbox_inches='tight')
fx.showfig(doc)

"""### C_l^{gg}."""

b = fx.simple_l_binning(deltag[0].lmax)
clgg0 = b.bin_average(deltag[0].clgg)
clgg0_sub = b.bin_average(deltag_sub[0].clgg)

for (g,s,n,l) in zip(deltag,deltag_sub,nside,lmax):
    clgg = b.bin_average(g.clgg)
    clgg_sub = b.bin_average(s.clgg)

    plt.plot(b.l_vals, clgg/clgg0, label=f'({n},{l})')
    plt.plot(b.l_vals, clgg_sub/clgg0_sub, label=f's({n},{l})', linestyle='--')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{gg}/C_\ell^{(gg,0)}$')
plt.legend(loc='upper right').draw_frame(False)
plt.savefig('clgg-test-nside.pdf', dpi=300, bbox_inches='tight')
fx.showfig(doc)
