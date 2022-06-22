#!/usr/bin/env python3
import time
import numpy as np
import numpy.random as rand
import astropy.units as u
import frbx as fx


start = time.time()
self = fx.cosmology_base()


#########################################################################################
#
# self.base: (obj) base cosmology.

for i in range(10):
    H0 = rand.uniform(1e-3, 2.0e2, 1)[0]
    Om0 = rand.uniform(size=1)[0]
    Ob0 = rand.uniform(size=1)[0]
    Ob0 = min(Ob0, Om0)
    Tcmb0 = rand.uniform(high=2.0e2, size=1)[0]
    Neff = rand.uniform(high=10.0, size=1)[0]
    m_nu = rand.uniform(high=0.9, size=1)[0]
    zmax = rand.uniform(low=3.0, high=20.0, size=1)[0]

    fx.cosmology_base(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0*u.K, Neff=Neff, m_nu=m_nu*u.eV, zmax=zmax)

print('self.base -> passed.')


#########################################################################################
#
# self.dm_igm_interp: (interp obj) interpolated version of self.dm_igm.

n_iter = 1000

z = rand.uniform(0.0, self.zmax, n_iter)
z = np.append(z, [0.0, self.zmax])

for j in z:
    a = self.dm_igm_interp(j)
    e = self.dm_igm(j)        # tested below

    assert np.isclose(a, e, rtol=1.0e-4, atol=1.0e-7), f'({a}, {e})'

print('self.dm_igm_interp -> passed.')


#########################################################################################
#
# self.dm_igm: (method) integrated DM.

assert self.dm_igm(0.0) == 0.0

for z in rand.uniform(1e4, size=10):
    ret = self.dm_igm(z)

    assert isinstance(ret, float) and np.isfinite(ret) and (ret >= 0.0)

print('self.dm_igm -> passed.')


#########################################################################################
#
# self.z_at_d: (method) returns the redshift at a given DM.

niter = 100

zvec = rand.uniform(0.0, self.zmax, size=niter)
d = self.dm_igm_interp(zvec)

for i, z in enumerate(zvec):
    e = z
    a = self.z_at_d(d[i])
    assert np.isclose(a, e, atol=0.0, rtol=1.0e-2), f'({a}, {e})'

print('self.z_at_d -> passed.')


#########################################################################################
#
# self._diff_dm_igm: (helper method) differential DM.

ddm = self.dm_igm_interp.derivative(1)

for z in rand.uniform(self.zmax, size=10):
    e = ddm(z)
    a = self._diff_dm_igm(z)

    assert np.isclose(a, e, rtol=1.0e-4, atol=0.0)

print('self._diff_dm_igm -> passed.')


print('test-cosmology done!')
print(f'Elapsed time: {(time.time()-start)/60.:.2f} min.')
