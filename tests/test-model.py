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


print('test-model done!')
print(f'Elapsed time: {(time.time()-start)/60.:.2f} min.')
