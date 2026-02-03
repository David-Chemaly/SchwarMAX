import os
import numpy as np
import multiprocessing as mp

import dynesty
import dynesty.utils as dyut

def cpu_dynesty_fit(dict_data, logl_fn, prior_fn, ndim, num_Vbin, nlive=2000):
    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(logl_fn,
                                prior_fn,
                                ndim,
                                logl_args=(dict_data, num_Vbin),
                                nlive=nlive,
                                sample='rslice',
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)

    res   = dns.results
    inds  = np.arange(len(res.samples))
    inds  = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    dns_results = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    return dns_results

def gpu_dynesty_fit(dict_data, logl_fn, prior_fn, ndim, num_Vbin, nlive=2000):
    dns = dynesty.DynamicNestedSampler(logl_fn,
                            prior_fn,
                            ndim,
                            logl_args=(dict_data, num_Vbin),
                            nlive=nlive,
                            sample='rslice')
    dns.run_nested(n_effective=10000)

    res   = dns.results
    inds  = np.arange(len(res.samples))
    inds  = dyut.resample_equal(inds, weights=np.exp(res.logwt - res.logz[-1]))
    samps = res.samples[inds]
    logl  = res.logl[inds]

    dns_results = {
                    'dns': dns,
                    'samps': samps,
                    'logl': logl,
                    'logz': res.logz,
                    'logzerr': res.logzerr,
                }

    return dns_results