#!/usr/bin/env python3
import time
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import frbx as fx
import corner


start = time.time()

#########################################################################################
#
# chi2: (function) computes the sum of chi-square residuals.

ell = np.linspace(0.0, 1.0e10, 1000)
p = rand.ranf(2)
p[1] *= 1.0e4

clm = fx.model.cl_models()
cl_e = clm.exp(ell, p)

r = rand.ranf()
cl_a = cl_e + (r*cl_e)

cov_inv = np.eye(ell.size)

chi2_a = fx.fit.chi2(p, clm.exp, ell, cl_a, cov_inv, reduced=True)

assert np.isfinite(chi2_a)
assert 0.0 <= chi2_a <= 1.0

chi2_e = fx.fit.chi2(p, clm.exp, ell, cl_e, cov_inv, reduced=True)

assert chi2_e == 0.0

print('chi2 -> passed.')


#########################################################################################
#
# jcov: (function) converts a jacobian matrix of parameters to their covariance.

n_iter = 1000

for _ in range(n_iter):
    n = max(rand.randint(100), 2)

    jac = np.eye(n, dtype=np.float64)
    jac *= rand.ranf()

    _e = np.eye(n, dtype=np.float64) / np.diag(jac)**2.0

    for mode in ('naive', 'svd'):
        ret = fx.fit.jcov(jac, mode)

        assert np.isclose(ret, _e, atol=1.0e-16, rtol=1.0e-14).all()

    r = rand.ranf((n, n))

    # Positive semi-definite
    H = r.T.dot(r)

    e_cov = np.linalg.inv(H)

    for (mode, rtol) in zip(('naive', 'svd'), (1.0e-14, 1.0e-2)):
        a_cov = fx.fit.jcov(r, mode)

        assert np.isclose(a_cov, e_cov, atol=1.0e-16, rtol=rtol).all()

print('jcov -> passed.')


#########################################################################################
#
# fit_cl: (function) fits an array of angular power spectra to a model.

n_realizations = 10000
nl = 100

_p0 = np.asarray([1.0, 10.0])
ell = np.logspace(1, 8, nl)

clm = fx.model.cl_models(lchar=1.0e3)

p0 = _p0.copy()
p0[1] = clm.lchar
bounds = (p0-(0.4*p0), p0+(0.4*p0))

cl = clm.exp(ell, _p0)

cov = rand.ranf((nl,nl))
cov = fx.fit.jcov(cov, mode='svd')

cov_inv = np.linalg.inv(cov)

cl_r = np.zeros((n_realizations, nl))
for r in range(n_realizations):
    cl_r[r,:] = rand.multivariate_normal(cl, cov).reshape((nl,))

P, C = fx.fit.fit_cl(clm.exp, ell, cl_r, cov_inv, bounds, method='Nelder-Mead')

P[P < 0.0] = 1.0e-16
fig_corner = corner.corner(P, labels=[r'$\alpha$', r'$L$'], quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={'fontsize': 12})

fig_corner.savefig('test-fit_cl', dpi=300)
plt.clf()

print('Visual test: inspect ./test-fit_cl.png')
print(f'Expected params: {p0}')
print('fit_cl -> passed.')


#########################################################################################
#
# max_likelihood: (function) computes a set of parameters 't' along with their covariance
#                 which maximize the distribution f(x|t) within bounds 'p0'

n = 10000
mu = 4*rand.ranf()
beta = min(2.2999, 10*rand.ranf())

r = rand.gumbel(mu, beta, size=n)
_, x, _ = plt.hist(r, bins=100, density=True)

px = fx.max_likelihood(fx.utils.gumbel_pdf, r, [mu-1.0,beta+1.0])

plt.plot(x, fx.utils.gumbel_pdf(x, *px), 'k-')
plt.plot(x, fx.utils.gumbel_pdf(x, mu, beta), 'g:')
plt.savefig('./test-max_likelihood.pdf')
plt.clf()

print('Visual test: inspect ./test-max_likelihood.pdf')
print(f'Expected params: {(mu, beta)}')
print(f'Best-fit params: {px}')

print('max_likelihood -> passed.')


print('test-fit done!')
print(f'Elapsed time: {(time.time()-start)/60.:.2f} min.')
