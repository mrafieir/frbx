#!/usr/bin/env python3
import time
import numpy as np
import numpy.random as rand
import frbx as fx


start = time.time()

#########################################################################################
#
# cl_models: (class) contains various models of angular power spectrum.

clm = fx.model.cl_models()
n_iter = 100

for i in range(n_iter):
    nl = max(rand.randint(10000), 10)

    ell = np.linspace(0.0, 1.0e4, nl)

    for j in range(n_iter):
        p = rand.ranf(2)
        p[1] *= 1.0e4

        exp = clm.exp(ell, p)
        assert np.isfinite(exp).all()

        _clm = fx.model.cl_models(p[1])
        _exp = _clm.exp(ell, p)
        assert np.all(exp == _exp)

        l2 = clm.l2(ell, p)
        assert np.isfinite(l2).all()

print('cl_models -> passed.')


#########################################################################################
#
# gal_dm: (class) computes the galactic DMs for arrays of equatorial coordinates.

pygedm = False
nsides = [4, 8, 16]
n_ra = 10
n_dec = 10

ret = np.zeros(2, 3, (len(nsides), n_ra, n_dec))

for i, nside in enumerate(nsides):
    gal_dm = fx.model_gal_dm(nside=nside)

    for j, ra in enumerate(range(0.0, 360.0, n_ra)):
        for k, dec in enumerate(range(-90.0, 90.0, n_dec)):
            ret[0,0,i,j,k] = gal_dm(ra, dec, 'ymw16')
            ret[0,1,i,j,k] = gal_dm(ra, dec, 'ymw16_hp')

            ret[1,0,i,j,k] = gal_dm(ra, dec, 'ne01')
            ret[1,1,i,j,k] = gal_dm(ra, dec, 'ne01_hp')

            if pygedm:
                ret[0,2,i,j,k] = gal_dm(ra, dec, 'pygedm_ymw16')
                ret[1,2,i,j,k] = gal_dm(ra, dec, 'pygedm_ne01')

#FIXME
print('gal_dm -> passed.')


print('test-model done!')
print(f'Elapsed time: {(time.time()-start)/60.:.2f} min.')
