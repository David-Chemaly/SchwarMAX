import jax
import jax.numpy as jnp
import numpyro
from functools import partial

class numpyro_model(object):
    """
    Numpyro model class. This is a wrapper around a numpyro model that allows for easy sampling and plotting.

    Parameters
    ----------
    logL_fn : function
        Function that returns the log likelihood of the data given the parameters.
    parameters : dict
        Dictionary giving the prior distributions for each parameter (e.g. numpyro.distributions.Normal(0,1)).
    data : array or dict
        Data to be passed to the log likelihood function. 
    aux_parameters : dict, optional
        Dictionary of auxiliary parameters to be passed to the log likelihood function. The default is None.
    log_prior_fn : function, optional
        Function that returns the log prior of the parameters. The default is None.
    expand_args : bool, optional
        Whether to expand the aux_parameters as keyword arguments to the log likelihood function. The default is True.
    """
    def __init__(self, logL_fn, parameters, data, aux_parameters=None, log_prior_fn=None, expand_args=True):
        self.parameters = parameters
        self.data = data
        if aux_parameters is not None:
            if expand_args:
                # if fails here, you may need to do expand_args=False
                self.lnL = lambda x, y: logL_fn(x, y, **aux_parameters)
            else:
                self.lnL = lambda x, y: logL_fn(x, y, aux_parameters)
        else:
            self.lnL = logL_fn
        self.log_prior_fn = log_prior_fn
        self.aux_parameters = aux_parameters
        self.type = 'mcmc'
        if type(self.data) == dict:
            self.Ndata = self.data[list(self.data.keys())[0]].shape[0]
        else:
            self.Ndata = self.data.shape[0]

    def __call__(self):
        """
        Call the numpyro model.
        """
        params = {p:numpyro.sample(p, self.parameters[p]) for p in self.parameters}
        if self.log_prior_fn is not None:
            if type(self.log_prior_fn) is list:
                for i,p in enumerate(self.log_prior_fn):
                    numpyro.factor(f'prior{i}', p(params))
            else:
                numpyro.factor('prior', self.log_prior_fn(params))

        numpyro.factor('log_likelihood', self.lnL(params, self.data))
        
    def run_mcmc(self, num_warmup=100, num_samples=300, num_chains=1, init_strategy=numpyro.infer.init_to_sample(),
                 max_tree_depth=10, target_accept_prob=0.8, step_size=1., chain_method="vectorized", 
                 adapt_step_size=True, jit_model_args = False, extra_fields = ()):
        """
        Run the MCMC sampler.

        Parameters
        ----------
        num_warmup : int, optional
            Number of warmup samples. The default is 100.
        num_samples : int, optional
            Number of samples. The default is 300.
        num_chains : int, optional
            Number of chains. The default is 1.
        init_strategy : function, optional
            Initialization strategy. The default is numpyro.infer.init_to_sample().
        max_tree_depth : int, optional
            Maximum tree depth. The default is 10.
        chain_method : str, optional
            Chain method. The default is "vectorized".
        """
        self.mcmc = numpyro.infer.MCMC(numpyro.infer.NUTS(self.__call__, 
                                                          init_strategy=init_strategy,
                                                          max_tree_depth=max_tree_depth,
                                                          adapt_step_size=adapt_step_size,
                                                          target_accept_prob=target_accept_prob,
                                                          step_size = step_size,), 
                                       num_warmup=num_warmup, num_samples=num_samples,
                                       num_chains=num_chains, chain_method=chain_method,
                                       progress_bar = True,
                                        jit_model_args = jit_model_args)
        # self.mcmc.run(jax.random.PRNGKey(4567), extra_fields=extra_fields)
        self.mcmc.run(jax.random.PRNGKey(0))
        
    def samples(self):
        """
        Get the samples from the MCMC run.
        """
        return self.mcmc.get_samples()
    
    