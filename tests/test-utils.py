#!/usr/bin/env python3
import os
import time
import string
import random
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import frbx as fx


start = time.time()

#########################################################################################
#
# data_path: Returns an absolute path to an FRBX data file.

envar = ('FRBXDATA', 'FAKE')

for _e in envar:
    filename = ''.join(random.choice(string.hexdigits) for _ in range(100))
    try:
        ret = fx.data_path(filename, envar=_e)
        e = os.environ[_e]
        assert ret == os.path.join(e, filename)
    except (AssertionError, RuntimeError):
        pass

print('data_path -> passed.')


#########################################################################################
#
# slicer: Returns a sliced range of numbers.

n_iter = 10

xmin = rand.uniform(high=1.0e2, size=n_iter)
xmax = xmin + rand.uniform(1.0e-4, 1.0e4, n_iter)

for (i, j) in zip(xmin, xmax):
    for n in rand.randint(1, 100, n_iter):
        log_spaced = bool(rand.randint(2))

        if log_spaced and bool(rand.randint(2)):
            _n = None
            dlog = rand.ranf()
        else:
            _n = n
            dlog = None

        r_min, r_max = fx.slicer(i, j, _n, log_spaced, dlog)

        assert r_min.size == r_max.size

        if _n is not None:
            assert n == r_min.size

        for k in range(r_min.size-1):
            assert np.isclose(r_min[k+1], r_max[k], atol=0.0, rtol=1.0e-9)

print('slicer -> passed.')


#########################################################################################
#
# quad

epsabs = 0.0
epsrel = 1.0e-4

atol = 0.0
rtol = 1.0e-10

n_part = 100

# test 0
_f = fx.quad(lambda _x: 1.0, 0.0, 10.0, epsabs=epsabs, epsrel=epsrel)

d = np.linspace(0.0, 10.0, n_part)
_fd = np.sum([fx.quad(lambda _x: 1.0, d[i], d[i+1], epsabs=epsabs, epsrel=epsrel) for i in range(d.size-1)])

_ref = 10.0

assert np.isclose(_f, _ref, atol=atol, rtol=rtol), f'({_f}, {_ref})'
assert np.isclose(_fd, _ref, atol=atol, rtol=rtol), f'({_fd}, {_ref})'

# test 1
_f = fx.quad(lambda _x: _x ** 2.0, 1.0, 10.0, epsabs=epsabs, epsrel=epsrel)

d = np.linspace(1.0, 10.0, n_part)
_fd = np.sum([fx.quad(lambda _x: _x**2.0, d[i], d[i+1], epsabs=epsabs, epsrel=epsrel) for i in range(d.size-1)])

_ref = (10.0**3.0 - 1.0) / 3.0

assert np.isclose(_f, _ref, atol=atol, rtol=rtol), f'({_f}, {_ref})'
assert np.isclose(_fd, _ref, atol=atol, rtol=rtol), f'({_fd}, {_ref})'

# test 2
_f = fx.quad(lambda _x: np.exp(_x), 1.0e-2, 1.0, epsabs=epsabs, epsrel=epsrel)

d = np.linspace(1.0e-2, 1.0, n_part)
_fd = np.sum([fx.quad(lambda _x: np.exp(_x), d[i], d[i+1], epsabs=epsabs, epsrel=epsrel) for i in range(d.size-1)])

_ref = np.exp(1.0) - np.exp(1.0e-2)

assert np.isclose(_f, _ref, atol=atol, rtol=rtol), f'({_f}, {_ref})'
assert np.isclose(_fd, _ref, atol=atol, rtol=rtol), f'({_fd}, {_ref})'

print('quad -> passed.')


#########################################################################################
#
# spline

def f(a):
    return np.exp(a) * a**3


n_iter = 100
rtol = 1.0e-3
nsteps = 1000
xmin = 1e1
xmax = 1e2

x = np.linspace(xmin, xmax, nsteps)
s = fx.spline(x, [f(i) for i in x])

for _a in rand.uniform(xmin, xmax, n_iter):
    _f = f(_a)
    _s = s(_a)

    assert np.isclose(_s, _f, atol=0.0, rtol=rtol), f'({_s}, {_f})'

print('spline -> passed.')


#########################################################################################
#
# gumbel_pdf

n = 10000
mu = 0.0
beta = 10 * rand.ranf()

r = rand.gumbel(mu, beta, size=n)
_, x, _ = plt.hist(r, bins=100, density=True)

plt.plot(x, fx.utils.gumbel_pdf(x, mu, beta))
plt.savefig('./gumbel_pdf.pdf')
plt.clf()

print('Visual test: inspect ./gumbel_pdf.pdf')
print('gumbel -> passed.')


print('test-utils done!')
print(f'Elapsed time: {(time.time()-start)/60.:.2f} min.')
